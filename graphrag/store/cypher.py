"""Todas as consultas Cypher do sistema, centralizadas.

Rótulos e tipos de relacionamento não podem ser parametrizados em Cypher;
quando precisam ser interpolados, são antes validados contra a ontologia
(``safe_label`` / ``safe_rel_types``), eliminando risco de injeção.
"""
from __future__ import annotations

import re

from ..ontology import Ontology

_IDENT = re.compile(r"^[A-Za-z][A-Za-z0-9_]{0,40}$")


def safe_label(label: str, ontology: Ontology) -> str:
    if label not in ontology.entity_types or not _IDENT.match(label):
        raise ValueError(f"Rótulo não permitido: {label}")
    return label


def safe_rel_types(types: list[str], ontology: Ontology) -> str:
    """Retorna 'A|B|C' com tipos validados para uso em padrões ``-[:A|B|C]-``."""
    out = []
    for t in types:
        if t not in ontology.relations or not _IDENT.match(t):
            raise ValueError(f"Tipo de relação não permitido: {t}")
        out.append(t)
    if not out:
        raise ValueError("Lista de tipos de relação vazia")
    return "|".join(out)


# ------------------------------------------------------------------ schema
def schema_statements(ontology: Ontology, dim: int) -> list[str]:
    dim = int(dim)
    st = [
        # Constraints de unicidade (também criam índice RANGE implícito)
        "CREATE CONSTRAINT entity_id IF NOT EXISTS FOR (e:Entity) REQUIRE e.id IS UNIQUE",
        "CREATE CONSTRAINT document_id IF NOT EXISTS FOR (d:Document) REQUIRE d.id IS UNIQUE",
        "CREATE CONSTRAINT chunk_id IF NOT EXISTS FOR (c:Chunk) REQUIRE c.id IS UNIQUE",
        "CREATE CONSTRAINT community_id IF NOT EXISTS FOR (m:Community) REQUIRE m.id IS UNIQUE",
    ]
    for label in ontology.entity_types:
        st.append(
            f"CREATE CONSTRAINT {label.lower()}_id IF NOT EXISTS FOR (n:{safe_label(label, ontology)}) REQUIRE n.id IS UNIQUE"
        )
    st += [
        # Índices RANGE (sucessores dos índices B-tree a partir do Neo4j 5)
        "CREATE INDEX entity_name IF NOT EXISTS FOR (e:Entity) ON (e.name)",
        "CREATE INDEX entity_norm_name IF NOT EXISTS FOR (e:Entity) ON (e.norm_name)",
        "CREATE INDEX entity_type IF NOT EXISTS FOR (e:Entity) ON (e.type)",
        "CREATE INDEX entity_pagerank IF NOT EXISTS FOR (e:Entity) ON (e.pagerank)",
        "CREATE INDEX entity_community IF NOT EXISTS FOR (e:Entity) ON (e.community)",
        "CREATE INDEX chunk_doc IF NOT EXISTS FOR (c:Chunk) ON (c.doc_id)",
        "CREATE INDEX received_year IF NOT EXISTS FOR ()-[r:RECEIVED]-() ON (r.year)",
        # Índices full-text (Lucene)
        "CREATE FULLTEXT INDEX entity_fulltext IF NOT EXISTS FOR (e:Entity) ON EACH [e.name, e.aliases_text]",
        "CREATE FULLTEXT INDEX chunk_fulltext IF NOT EXISTS FOR (c:Chunk) ON EACH [c.text] "
        "OPTIONS {indexConfig: {`fulltext.analyzer`: 'brazilian'}}",
        # Índices vetoriais (HNSW, similaridade de cosseno)
        f"CREATE VECTOR INDEX chunk_embedding IF NOT EXISTS FOR (c:Chunk) ON c.embedding "
        f"OPTIONS {{indexConfig: {{`vector.dimensions`: {dim}, `vector.similarity_function`: 'cosine'}}}}",
        f"CREATE VECTOR INDEX entity_embedding IF NOT EXISTS FOR (e:Entity) ON e.embedding "
        f"OPTIONS {{indexConfig: {{`vector.dimensions`: {dim}, `vector.similarity_function`: 'cosine'}}}}",
    ]
    return st


# ------------------------------------------------------------------ ingestão
UPSERT_DOCUMENT = """
MERGE (d:Document {id: $id})
SET d.title = $title, d.source = $source, d.ingested_at = datetime()
"""

UPSERT_CHUNK = """
MATCH (d:Document {id: $doc_id})
MERGE (c:Chunk {id: $id})
SET c.text = $text, c.doc_id = $doc_id, c.idx = $idx, c.embedding = $embedding
MERGE (d)-[:HAS_CHUNK]->(c)
"""

LINK_NEXT_CHUNK = """
MATCH (p:Chunk {id: $prev}), (c:Chunk {id: $id})
MERGE (p)-[:NEXT_CHUNK]->(c)
"""


def upsert_entity(label: str) -> str:
    return f"""
MERGE (e:Entity {{id: $id}})
ON CREATE SET e.name = $name, e.type = $type, e.norm_name = $norm_name,
              e.aliases = $aliases, e.created_at = datetime()
ON MATCH SET e.aliases = reduce(acc = coalesce(e.aliases, []), a IN $aliases |
                         CASE WHEN a IN acc THEN acc ELSE acc + a END)
SET e:{label}, e.embedding = $embedding
SET e.aliases_text = reduce(s = '', a IN e.aliases | s + ' ' + a)
"""


ADD_MENTION = """
MATCH (c:Chunk {id: $chunk_id}), (e:Entity {id: $entity_id})
MERGE (c)-[m:MENTIONS]->(e)
ON CREATE SET m.count = 1
ON MATCH SET m.count = m.count + 1
"""


def upsert_relation(rel_type: str) -> str:
    return f"""
MATCH (a:Entity {{id: $src}}), (b:Entity {{id: $dst}})
MERGE (a)-[r:{rel_type}]->(b)
ON CREATE SET r.sources = [$chunk_id], r.confidence = $confidence, r.mention_count = 1,
              r.year = $year, r.evidence = $evidence, r.extractor = $extractor
ON MATCH SET r.sources = CASE WHEN $chunk_id IN r.sources THEN r.sources ELSE r.sources + $chunk_id END,
             r.mention_count = r.mention_count + 1,
             r.confidence = CASE WHEN $confidence > r.confidence THEN $confidence ELSE r.confidence END,
             r.year = coalesce(r.year, $year)
"""


ALL_ENTITIES = """
MATCH (e:Entity)
RETURN e.id AS id, e.name AS name, e.type AS type, coalesce(e.aliases, []) AS aliases,
       e.pagerank AS pagerank, e.community AS community
"""

# ------------------------------------------------------------------ recuperação
VECTOR_CHUNKS = """
CALL db.index.vector.queryNodes('chunk_embedding', $k, $embedding) YIELD node, score
MATCH (d:Document)-[:HAS_CHUNK]->(node)
RETURN node.id AS id, node.text AS text, node.doc_id AS doc_id, d.title AS title, score
"""

VECTOR_ENTITIES = """
CALL db.index.vector.queryNodes('entity_embedding', $k, $embedding) YIELD node, score
RETURN node {.id, .name, .type, .pagerank} AS entity, score
"""

FULLTEXT_ENTITIES = """
CALL db.index.fulltext.queryNodes('entity_fulltext', $q, {limit: $k}) YIELD node, score
RETURN node {.id, .name, .type, .pagerank} AS entity, score
"""

CHUNK_ENTITIES = """
UNWIND $chunk_ids AS cid
MATCH (c:Chunk {id: cid})-[:MENTIONS]->(e:Entity)
RETURN cid AS chunk_id, e {.id, .name, .type, .pagerank} AS entity
"""

CHUNKS_BY_IDS = """
MATCH (d:Document)-[:HAS_CHUNK]->(c:Chunk)
WHERE c.id IN $ids
RETURN c.id AS id, c.text AS text, d.title AS title, d.id AS doc_id
"""


def expand(rel_types: str, hops: int) -> str:
    hops = max(1, min(int(hops), 4))
    return f"""
MATCH p = (s:Entity)-[:{rel_types}*1..{hops}]-(n:Entity)
WHERE s.id IN $seeds
WITH p, reduce(acc = 0.0, x IN nodes(p) | acc + coalesce(x.pagerank, 0.0)) AS pr
ORDER BY length(p) ASC, pr DESC
LIMIT $limit
RETURN p
"""


def shortest_paths(rel_types: str, max_hops: int) -> str:
    max_hops = max(1, min(int(max_hops), 6))
    return f"""
MATCH (a:Entity {{id: $a}}), (b:Entity {{id: $b}})
MATCH p = allShortestPaths((a)-[:{rel_types}*..{max_hops}]-(b))
RETURN p
LIMIT 10
"""


def kag_step(rel_types: str, direction: str) -> str:
    """Passo do executor KAG: (x)-[r]->(y), (x)<-[r]-(y) ou (x)-[r]-(y)."""
    pattern = {
        "out": f"(x)-[r:{rel_types}]->(y:Entity)",
        "in": f"(x)<-[r:{rel_types}]-(y:Entity)",
        "any": f"(x)-[r:{rel_types}]-(y:Entity)",
    }[direction]
    return f"""
MATCH (x:Entity) WHERE x.id IN $ids
MATCH {pattern}
WHERE $to_type IS NULL OR y.type = $to_type
WITH r, y, startNode(r) AS s, endNode(r) AS t
RETURN s {{.id, .name, .type}} AS src, type(r) AS rel, properties(r) AS props,
       t {{.id, .name, .type}} AS dst, y {{.id, .name, .type, .pagerank}} AS next
LIMIT 200
"""


# ------------------------------------------------------------------ algoritmos
def entity_edges(rel_types: str) -> str:
    return f"""
MATCH (a:Entity)-[r:{rel_types}]->(b:Entity)
RETURN a.id AS src, b.id AS dst, type(r) AS rel
"""


WRITE_NODE_PROPS = """
UNWIND $rows AS row
MATCH (e:Entity {id: row.id})
SET e += row.props
"""

GDS_DROP = "CALL gds.graph.drop('entity_graph', false) YIELD graphName RETURN graphName"
GDS_PROJECT = """
CALL gds.graph.project('entity_graph', 'Entity', {REL: {type: '*', orientation: 'UNDIRECTED'}})
YIELD nodeCount, relationshipCount
RETURN nodeCount, relationshipCount
"""
GDS_PAGERANK = """
CALL gds.pageRank.write('entity_graph', {writeProperty: 'pagerank', maxIterations: 20, dampingFactor: 0.85})
YIELD nodePropertiesWritten RETURN nodePropertiesWritten
"""
GDS_LOUVAIN = """
CALL gds.louvain.write('entity_graph', {writeProperty: 'community'})
YIELD communityCount, modularity RETURN communityCount, modularity
"""

DELETE_COMMUNITIES = "MATCH (m:Community) DETACH DELETE m"
UPSERT_COMMUNITY = """
MERGE (m:Community {id: $id})
SET m.summary = $summary, m.size = $size, m.top_entities = $top_entities, m.level = 0
WITH m
UNWIND $members AS mid
MATCH (e:Entity {id: mid})
MERGE (e)-[:IN_COMMUNITY]->(m)
"""
COMMUNITIES_OF = """
MATCH (e:Entity)-[:IN_COMMUNITY]->(m:Community)
WHERE e.id IN $ids
RETURN DISTINCT m {.id, .summary, .size, .top_entities} AS community
"""

STATS_NODES = "MATCH (n) UNWIND labels(n) AS label RETURN label, count(*) AS count ORDER BY count DESC"
STATS_RELS = "MATCH ()-[r]->() RETURN type(r) AS type, count(*) AS count ORDER BY count DESC"
CLEAR = "MATCH (n) DETACH DELETE n"
