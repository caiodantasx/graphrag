"""Backend Neo4j (produção). Requer Neo4j 5.13+ (índices vetoriais) e,
opcionalmente, o plugin Graph Data Science para PageRank/Louvain no servidor."""
from __future__ import annotations

import time

from ..ontology import Ontology
from ..text import normalize
from . import cypher as Q
from .base import GraphPath, GraphStore, StoreStats

LUCENE_SPECIAL = set('+-&|!(){}[]^"~*?:\\/')


def lucene_query(text: str) -> str:
    """Converte a pergunta em consulta Lucene: termos OR com busca difusa."""
    from ..text import content_tokens

    terms = []
    for t in content_tokens(text):
        t = "".join(ch for ch in t if ch not in LUCENE_SPECIAL)
        if not t:
            continue
        terms.append(f"{t}~" if len(t) >= 5 else t)
    return " OR ".join(terms) or "*"


class Neo4jGraphStore(GraphStore):
    backend = "neo4j"

    def __init__(self, ontology: Ontology, uri: str, user: str, password: str, database: str = "neo4j"):
        super().__init__(ontology)
        from neo4j import GraphDatabase

        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.driver.verify_connectivity()
        self.database = database

    def close(self):
        self.driver.close()

    # ------------------------------------------------------------ execução
    def run(self, cypher: str, params: dict | None = None, write: bool = False) -> list:
        params = params or {}
        t0 = time.perf_counter()
        with self.driver.session(database=self.database) as s:
            if write:
                records = s.execute_write(lambda tx: list(tx.run(cypher, params)))
            else:
                records = s.execute_read(lambda tx: list(tx.run(cypher, params)))
        self._log(cypher, params, t0, len(records))
        return records

    def run_auto(self, cypher: str, params: dict | None = None) -> list:
        """Transação implícita (necessária para comandos de schema e GDS)."""
        params = params or {}
        t0 = time.perf_counter()
        with self.driver.session(database=self.database) as s:
            records = list(s.run(cypher, params))
        self._log(cypher, params, t0, len(records))
        return records

    @staticmethod
    def _node(n) -> dict:
        return {"id": n.get("id"), "name": n.get("name"), "type": n.get("type"), "pagerank": n.get("pagerank")}

    def _path(self, p, source: str) -> GraphPath:
        nodes = [self._node(n) for n in p.nodes]
        rels = []
        for r in p.relationships:
            props = {k: v for k, v in dict(r).items() if k != "embedding"}
            rels.append({"type": r.type, "start": r.start_node.get("id"), "end": r.end_node.get("id"), "props": props})
        return GraphPath(nodes, rels, source)

    # ------------------------------------------------------------ schema / ingestão
    def setup_schema(self, dim):
        st = Q.schema_statements(self.ontology, dim)
        for s in st:
            self.run_auto(s)
        self.run_auto("CALL db.awaitIndexes(300)")
        return st

    def clear(self):
        self.run_auto(Q.CLEAR)

    def upsert_document(self, doc_id, title, source):
        self.run(Q.UPSERT_DOCUMENT, {"id": doc_id, "title": title, "source": source}, write=True)

    def upsert_chunk(self, chunk_id, doc_id, idx, text, embedding, prev_id):
        self.run(Q.UPSERT_CHUNK, {"id": chunk_id, "doc_id": doc_id, "idx": idx, "text": text,
                                  "embedding": list(map(float, embedding))}, write=True)
        if prev_id:
            self.run(Q.LINK_NEXT_CHUNK, {"prev": prev_id, "id": chunk_id}, write=True)

    def upsert_entity(self, eid, name, etype, aliases, embedding):
        label = Q.safe_label(etype, self.ontology)
        self.run(Q.upsert_entity(label), {"id": eid, "name": name, "type": etype, "norm_name": normalize(name),
                                          "aliases": list(aliases), "embedding": list(map(float, embedding))}, write=True)

    def add_mention(self, chunk_id, entity_id):
        self.run(Q.ADD_MENTION, {"chunk_id": chunk_id, "entity_id": entity_id}, write=True)

    def upsert_relation(self, src, rel, dst, chunk_id, year, confidence, evidence, extractor):
        rel = Q.safe_rel_types([rel], self.ontology)
        self.run(Q.upsert_relation(rel), {"src": src, "dst": dst, "chunk_id": chunk_id, "year": year,
                                          "confidence": confidence, "evidence": evidence, "extractor": extractor},
                 write=True)

    # ------------------------------------------------------------ leitura
    def all_entities(self):
        return [dict(r) for r in self.run(Q.ALL_ENTITIES)]

    def vector_chunks(self, embedding, k):
        return [dict(r) for r in self.run(Q.VECTOR_CHUNKS, {"k": k, "embedding": list(map(float, embedding))})]

    def vector_entities(self, embedding, k):
        return [(dict(r["entity"]), r["score"]) for r in self.run(Q.VECTOR_ENTITIES, {"k": k, "embedding": list(map(float, embedding))})]

    def fulltext_entities(self, query, k):
        return [(dict(r["entity"]), r["score"]) for r in self.run(Q.FULLTEXT_ENTITIES, {"q": lucene_query(query), "k": k})]

    def chunk_entities(self, chunk_ids):
        return [(r["chunk_id"], dict(r["entity"])) for r in self.run(Q.CHUNK_ENTITIES, {"chunk_ids": chunk_ids})]

    def chunks_by_ids(self, ids):
        return [dict(r) for r in self.run(Q.CHUNKS_BY_IDS, {"ids": ids})]

    def expand(self, seed_ids, hops, limit):
        rels = Q.safe_rel_types(self.entity_rel_types, self.ontology)
        return [self._path(r["p"], "expand") for r in self.run(Q.expand(rels, hops), {"seeds": seed_ids, "limit": limit})]

    def shortest_paths(self, a, b, max_hops):
        if a == b:
            return []
        rels = Q.safe_rel_types(self.entity_rel_types, self.ontology)
        return [self._path(r["p"], "shortest") for r in self.run(Q.shortest_paths(rels, max_hops), {"a": a, "b": b})]

    def step(self, from_ids, rel_types, direction, to_type):
        rels = Q.safe_rel_types(rel_types, self.ontology)
        if to_type is not None:
            Q.safe_label(to_type, self.ontology)
        out = []
        for r in self.run(Q.kag_step(rels, direction), {"ids": from_ids, "to_type": to_type}):
            out.append({"src": dict(r["src"]), "rel": r["rel"], "props": {k: v for k, v in dict(r["props"]).items()},
                        "dst": dict(r["dst"]), "next": dict(r["next"])})
        return out

    def entity_edges(self):
        rels = Q.safe_rel_types(self.entity_rel_types, self.ontology)
        return [(r["src"], r["dst"], r["rel"]) for r in self.run(Q.entity_edges(rels))]

    def write_node_props(self, rows):
        self.run(Q.WRITE_NODE_PROPS, {"rows": rows}, write=True)

    def replace_communities(self, communities):
        self.run(Q.DELETE_COMMUNITIES, write=True)
        for c in communities:
            self.run(Q.UPSERT_COMMUNITY, {"id": c["id"], "summary": c["summary"], "size": c["size"],
                                          "top_entities": c["top_entities"], "members": c["members"]}, write=True)

    def communities_of(self, entity_ids):
        return [dict(r["community"]) for r in self.run(Q.COMMUNITIES_OF, {"ids": entity_ids})]

    def stats(self):
        nodes = {r["label"]: r["count"] for r in self.run(Q.STATS_NODES)}
        rels = {r["type"]: r["count"] for r in self.run(Q.STATS_RELS)}
        return StoreStats(nodes, rels)

    def readonly_cypher(self, cypher, params=None):
        records = self.run(cypher, params or {}, write=False)  # transação de leitura
        return [r.data() for r in records]

    def explain(self, cypher: str, params: dict | None = None):
        """Valida sintaxe/planejamento sem executar (usado pelo guarda do Text2Cypher)."""
        with self.driver.session(database=self.database) as s:
            s.run("EXPLAIN " + cypher, params or {}).consume()

    def try_gds(self) -> bool:
        try:
            self.run_auto(Q.GDS_DROP)
            self.run_auto(Q.GDS_PROJECT)
            self.run_auto(Q.GDS_PAGERANK)
            self.run_auto(Q.GDS_LOUVAIN)
            self.run_auto(Q.GDS_DROP)
            return True
        except Exception:
            return False
