"""Interface comum dos backends de grafo e estruturas de dados compartilhadas."""
from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from ..ontology import Ontology


@dataclass
class GraphPath:
    nodes: list[dict]  # {id, name, type, pagerank?}
    rels: list[dict]   # {type, start, end, props}
    source: str = ""   # expand | shortest | kag | text2cypher

    @property
    def key(self) -> tuple:
        return tuple((r["start"], r["type"], r["end"]) for r in self.rels)

    def node_ids(self) -> list[str]:
        return [n["id"] for n in self.nodes]


@dataclass
class QueryLogEntry:
    cypher: str
    params: dict
    ms: float
    backend: str
    rows: int = 0


@dataclass
class StoreStats:
    nodes: dict = field(default_factory=dict)
    rels: dict = field(default_factory=dict)


def _clean_params(params: dict) -> dict:
    out = {}
    for k, v in params.items():
        if isinstance(v, list) and len(v) > 12 and all(isinstance(x, float) for x in v[:12]):
            out[k] = f"<vetor de {len(v)} dimensões>"
        else:
            out[k] = v
    return out


class GraphStore(ABC):
    backend = "abstract"

    def __init__(self, ontology: Ontology):
        self.ontology = ontology
        self.query_log: list[QueryLogEntry] = []

    # ------------------------------------------------------------ log
    def _log(self, cypher: str, params: dict, t0: float, rows: int = 0):
        self.query_log.append(
            QueryLogEntry(" ".join(cypher.split()), _clean_params(params), (time.perf_counter() - t0) * 1000, self.backend, rows)
        )

    def reset_log(self):
        self.query_log = []

    @property
    def entity_rel_types(self) -> list[str]:
        return self.ontology.entity_relation_types()

    # ------------------------------------------------------------ schema / ingestão
    @abstractmethod
    def setup_schema(self, dim: int) -> list[str]: ...
    @abstractmethod
    def clear(self): ...
    @abstractmethod
    def upsert_document(self, doc_id: str, title: str, source: str): ...
    @abstractmethod
    def upsert_chunk(self, chunk_id: str, doc_id: str, idx: int, text: str, embedding: list[float], prev_id: str | None): ...
    @abstractmethod
    def upsert_entity(self, eid: str, name: str, etype: str, aliases: list[str], embedding: list[float]): ...
    @abstractmethod
    def add_mention(self, chunk_id: str, entity_id: str): ...
    @abstractmethod
    def upsert_relation(self, src: str, rel: str, dst: str, chunk_id: str, year, confidence: float, evidence: str, extractor: str): ...

    # ------------------------------------------------------------ leitura
    @abstractmethod
    def all_entities(self) -> list[dict]: ...
    @abstractmethod
    def vector_chunks(self, embedding, k: int) -> list[dict]: ...
    @abstractmethod
    def vector_entities(self, embedding, k: int) -> list[tuple[dict, float]]: ...
    @abstractmethod
    def fulltext_entities(self, query: str, k: int) -> list[tuple[dict, float]]: ...
    @abstractmethod
    def chunk_entities(self, chunk_ids: list[str]) -> list[tuple[str, dict]]: ...
    @abstractmethod
    def chunks_by_ids(self, ids: list[str]) -> list[dict]: ...
    @abstractmethod
    def expand(self, seed_ids: list[str], hops: int, limit: int) -> list[GraphPath]: ...
    @abstractmethod
    def shortest_paths(self, a: str, b: str, max_hops: int) -> list[GraphPath]: ...
    @abstractmethod
    def step(self, from_ids: list[str], rel_types: list[str], direction: str, to_type: str | None) -> list[dict]: ...
    @abstractmethod
    def entity_edges(self) -> list[tuple[str, str, str]]: ...
    @abstractmethod
    def write_node_props(self, rows: list[dict]): ...
    @abstractmethod
    def replace_communities(self, communities: list[dict]): ...
    @abstractmethod
    def communities_of(self, entity_ids: list[str]) -> list[dict]: ...
    @abstractmethod
    def stats(self) -> StoreStats: ...

    def readonly_cypher(self, cypher: str, params: dict | None = None) -> list[dict]:
        raise NotImplementedError("Execução de Cypher arbitrário requer o backend Neo4j.")

    def close(self):
        pass

    # ------------------------------------------------------------ algoritmos
    def try_gds(self) -> bool:
        """Backends com Neo4j GDS sobrescrevem; retorna True se rodou no servidor."""
        return False
