"""Backend em memória (NetworkX + NumPy) com a MESMA semântica do backend Neo4j.

Uso: testes automatizados, CI e demonstração offline sem Docker. Cada operação
registra no log a consulta Cypher equivalente (a mesma string usada no Neo4j),
para manter a rastreabilidade idêntica na interface.
"""
from __future__ import annotations

import time
from difflib import SequenceMatcher

import networkx as nx
import numpy as np

from ..ontology import Ontology
from ..text import content_tokens, normalize
from . import cypher as Q
from .base import GraphPath, GraphStore, StoreStats


class MemoryGraphStore(GraphStore):
    backend = "memory"

    def __init__(self, ontology: Ontology):
        super().__init__(ontology)
        self.clear()

    # ------------------------------------------------------------ util
    def _node(self, eid: str) -> dict:
        d = self.G.nodes[eid]
        return {"id": eid, "name": d["name"], "type": d["type"], "pagerank": d.get("pagerank")}

    def _rel(self, u: str, v: str, k: str) -> dict:
        props = {kk: vv for kk, vv in self.G.edges[u, v, k].items()}
        return {"type": k, "start": u, "end": v, "props": props}

    # ------------------------------------------------------------ schema / ingestão
    def setup_schema(self, dim: int) -> list[str]:
        st = Q.schema_statements(self.ontology, dim)
        t0 = time.perf_counter()
        for s in st:
            self._log(s, {}, t0)
        return st

    def clear(self):
        self.G = nx.MultiDiGraph()
        self.docs: dict[str, dict] = {}
        self.chunks: dict[str, dict] = {}
        self.chunk_emb: dict[str, np.ndarray] = {}
        self.entity_emb: dict[str, np.ndarray] = {}
        self.mentions: dict[str, set[str]] = {}
        self.communities: dict[str, dict] = {}
        self.entity_comm: dict[str, str] = {}

    def upsert_document(self, doc_id, title, source):
        self.docs[doc_id] = {"id": doc_id, "title": title, "source": source}

    def upsert_chunk(self, chunk_id, doc_id, idx, text, embedding, prev_id):
        self.chunks[chunk_id] = {"id": chunk_id, "doc_id": doc_id, "idx": idx, "text": text, "prev": prev_id}
        self.chunk_emb[chunk_id] = np.asarray(embedding, dtype=np.float32)

    def upsert_entity(self, eid, name, etype, aliases, embedding):
        if eid not in self.G:
            self.G.add_node(eid, name=name, type=etype, norm_name=normalize(name), aliases=[])
        node = self.G.nodes[eid]
        for a in aliases:
            if a not in node["aliases"]:
                node["aliases"].append(a)
        self.entity_emb[eid] = np.asarray(embedding, dtype=np.float32)

    def add_mention(self, chunk_id, entity_id):
        self.mentions.setdefault(chunk_id, set()).add(entity_id)

    def upsert_relation(self, src, rel, dst, chunk_id, year, confidence, evidence, extractor):
        if self.G.has_edge(src, dst, key=rel):
            e = self.G.edges[src, dst, rel]
            if chunk_id not in e["sources"]:
                e["sources"].append(chunk_id)
            e["mention_count"] += 1
            e["confidence"] = max(e["confidence"], confidence)
            if e.get("year") is None:
                e["year"] = year
        else:
            self.G.add_edge(src, dst, key=rel, sources=[chunk_id], confidence=confidence,
                            mention_count=1, year=year, evidence=evidence, extractor=extractor)

    # ------------------------------------------------------------ leitura
    def all_entities(self):
        return [
            {"id": n, "name": d["name"], "type": d["type"], "aliases": list(d["aliases"]),
             "pagerank": d.get("pagerank"), "community": d.get("community")}
            for n, d in self.G.nodes(data=True)
        ]

    @staticmethod
    def _topk(matrix_items: dict, emb, k: int):
        if not matrix_items:
            return []
        ids = list(matrix_items)
        M = np.stack([matrix_items[i] for i in ids])
        q = np.asarray(emb, dtype=np.float32)
        sims = M @ q / (np.linalg.norm(M, axis=1) * (np.linalg.norm(q) or 1.0) + 1e-9)
        # Neo4j normaliza o cosseno para [0, 1]: (1 + cos) / 2
        sims = (1 + sims) / 2
        order = np.argsort(-sims)[:k]
        return [(ids[i], float(sims[i])) for i in order]

    def vector_chunks(self, embedding, k):
        t0 = time.perf_counter()
        out = []
        for cid, s in self._topk(self.chunk_emb, embedding, k):
            c = self.chunks[cid]
            out.append({"id": cid, "text": c["text"], "doc_id": c["doc_id"],
                        "title": self.docs[c["doc_id"]]["title"], "score": s})
        self._log(Q.VECTOR_CHUNKS, {"k": k, "embedding": list(map(float, embedding))}, t0, len(out))
        return out

    def vector_entities(self, embedding, k):
        t0 = time.perf_counter()
        out = [(self._node(eid), s) for eid, s in self._topk(self.entity_emb, embedding, k)]
        self._log(Q.VECTOR_ENTITIES, {"k": k, "embedding": list(map(float, embedding))}, t0, len(out))
        return out

    def fulltext_entities(self, query, k):
        """Aproximação do Lucene: casamento de termos com tolerância difusa (~)."""
        t0 = time.perf_counter()
        q_terms = [t.rstrip("~") for t in content_tokens(query.replace("~", " "))]
        scored = []
        for n, d in self.G.nodes(data=True):
            fields = [d["name"], *d["aliases"]]
            terms = {t for f in fields for t in normalize(f).split()}
            s = 0.0
            for qt in q_terms:
                if qt in terms:
                    s += 1.0
                elif len(qt) >= 4 and any(SequenceMatcher(None, qt, t).ratio() >= 0.8 for t in terms):
                    s += 0.5
            if s > 0:
                s = s / (1 + 0.1 * len(terms))  # normalização de comprimento (estilo BM25)
                scored.append((self._node(n), s))
        scored.sort(key=lambda x: -x[1])
        out = scored[:k]
        self._log(Q.FULLTEXT_ENTITIES, {"q": query, "k": k}, t0, len(out))
        return out

    def chunk_entities(self, chunk_ids):
        t0 = time.perf_counter()
        out = [(cid, self._node(e)) for cid in chunk_ids for e in sorted(self.mentions.get(cid, ())) if e in self.G]
        self._log(Q.CHUNK_ENTITIES, {"chunk_ids": chunk_ids}, t0, len(out))
        return out

    def chunks_by_ids(self, ids):
        t0 = time.perf_counter()
        out = []
        for cid in ids:
            c = self.chunks.get(cid)
            if c:
                out.append({"id": cid, "text": c["text"], "title": self.docs[c["doc_id"]]["title"], "doc_id": c["doc_id"]})
        self._log(Q.CHUNKS_BY_IDS, {"ids": ids}, t0, len(out))
        return out

    def _incident(self, n: str):
        for _, v, k in self.G.out_edges(n, keys=True):
            yield v, (n, v, k)
        for u, _, k in self.G.in_edges(n, keys=True):
            yield u, (u, n, k)

    def expand(self, seed_ids, hops, limit):
        t0 = time.perf_counter()
        hops = max(1, min(int(hops), 4))
        paths: list[GraphPath] = []

        def dfs(node_seq, edge_seq):
            if edge_seq:
                paths.append(GraphPath([self._node(x) for x in node_seq],
                                       [self._rel(*e) for e in edge_seq], "expand"))
            if len(edge_seq) == hops:
                return
            for nb, e in self._incident(node_seq[-1]):
                if e in edge_seq:  # unicidade de relacionamento (semântica Cypher)
                    continue
                dfs(node_seq + [nb], edge_seq + [e])

        for s in seed_ids:
            if s in self.G:
                dfs([s], [])

        def pr(p):
            return sum((n.get("pagerank") or 0.0) for n in p.nodes)

        paths.sort(key=lambda p: (len(p.rels), -pr(p)))
        out = paths[:limit]
        self._log(Q.expand("|".join(self.entity_rel_types), hops), {"seeds": seed_ids, "limit": limit}, t0, len(out))
        return out

    def shortest_paths(self, a, b, max_hops):
        t0 = time.perf_counter()
        out: list[GraphPath] = []
        if a in self.G and b in self.G and a != b:
            U = nx.Graph(self.G)
            try:
                for seq in nx.all_shortest_paths(U, a, b):
                    if len(seq) - 1 > max_hops:
                        break
                    rels = []
                    for u, v in zip(seq, seq[1:]):
                        if self.G.has_edge(u, v):
                            k = sorted(self.G[u][v])[0]
                            rels.append(self._rel(u, v, k))
                        else:
                            k = sorted(self.G[v][u])[0]
                            rels.append(self._rel(v, u, k))
                    out.append(GraphPath([self._node(x) for x in seq], rels, "shortest"))
                    if len(out) >= 10:
                        break
            except nx.NetworkXNoPath:
                pass
        self._log(Q.shortest_paths("|".join(self.entity_rel_types), max_hops), {"a": a, "b": b}, t0, len(out))
        return out

    def step(self, from_ids, rel_types, direction, to_type):
        t0 = time.perf_counter()
        rows = []
        for x in from_ids:
            if x not in self.G:
                continue
            cand = []
            if direction in ("out", "any"):
                cand += [(x, v, k, v) for _, v, k in self.G.out_edges(x, keys=True)]
            if direction in ("in", "any"):
                cand += [(u, x, k, u) for u, _, k in self.G.in_edges(x, keys=True)]
            for u, v, k, nxt in cand:
                if k not in rel_types:
                    continue
                if to_type and self.G.nodes[nxt]["type"] != to_type:
                    continue
                rows.append({"src": self._node(u), "rel": k, "props": dict(self.G.edges[u, v, k]),
                             "dst": self._node(v), "next": self._node(nxt)})
        self._log(Q.kag_step(Q.safe_rel_types(rel_types, self.ontology), direction),
                  {"ids": from_ids, "to_type": to_type}, t0, len(rows))
        return rows[:200]

    def entity_edges(self):
        return [(u, v, k) for u, v, k in self.G.edges(keys=True)]

    def write_node_props(self, rows):
        t0 = time.perf_counter()
        for r in rows:
            if r["id"] in self.G:
                self.G.nodes[r["id"]].update(r["props"])
        self._log(Q.WRITE_NODE_PROPS, {"rows": f"<{len(rows)} linhas>"}, t0, len(rows))

    def replace_communities(self, communities):
        self.communities = {c["id"]: c for c in communities}
        self.entity_comm = {m: c["id"] for c in communities for m in c["members"]}

    def communities_of(self, entity_ids):
        t0 = time.perf_counter()
        ids = []
        for e in entity_ids:
            c = self.entity_comm.get(e)
            if c and c not in ids:
                ids.append(c)
        out = [{k: self.communities[c][k] for k in ("id", "summary", "size", "top_entities")} for c in ids]
        self._log(Q.COMMUNITIES_OF, {"ids": entity_ids}, t0, len(out))
        return out

    def stats(self):
        nodes: dict[str, int] = {"Entity": self.G.number_of_nodes(), "Document": len(self.docs),
                                 "Chunk": len(self.chunks), "Community": len(self.communities)}
        for _, d in self.G.nodes(data=True):
            nodes[d["type"]] = nodes.get(d["type"], 0) + 1
        rels: dict[str, int] = {}
        for _, _, k in self.G.edges(keys=True):
            rels[k] = rels.get(k, 0) + 1
        rels["HAS_CHUNK"] = len(self.chunks)
        rels["NEXT_CHUNK"] = sum(1 for c in self.chunks.values() if c["prev"])
        rels["MENTIONS"] = sum(len(v) for v in self.mentions.values())
        rels["IN_COMMUNITY"] = len(self.entity_comm)
        return StoreStats(dict(sorted(nodes.items(), key=lambda x: -x[1])), dict(sorted(rels.items(), key=lambda x: -x[1])))
