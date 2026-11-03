"""Fachada do sistema: monta todos os componentes a partir das configurações."""
from __future__ import annotations

import time
from dataclasses import dataclass

from .algorithms import run_graph_algorithms
from .config import Settings, get_settings
from .embeddings import get_embedder
from .extraction import GazetteerMatcher, build_extractor
from .generation import Answer, AnswerGenerator
from .ingest import IngestionPipeline, IngestReport, load_gazetteer
from .llm import LLMClient
from .ontology import Ontology
from .resolution import EntityResolver
from .retrieval import HybridRetriever, RetrievalResult
from .store import get_store


@dataclass
class QAResult:
    retrieval: RetrievalResult
    answer: Answer
    elapsed_ms: float


class GraphRAGEngine:
    def __init__(self, settings: Settings | None = None):
        self.s = settings or get_settings()
        self.ontology = Ontology(mode=self.s.schema_mode)
        self.llm = LLMClient(self.s)
        self.embedder = get_embedder(self.s)
        self.store = get_store(self.s, self.ontology)
        self.retriever = HybridRetriever(self.s, self.store, self.embedder, self.ontology, self.llm)
        self.generator = AnswerGenerator(self.llm, self.ontology, self.embedder, lambda: self.retriever.linker,
                                         self.s.abstain_threshold)

    # ------------------------------------------------------------ construção do grafo
    def setup_schema(self) -> list[str]:
        return self.store.setup_schema(self.embedder.dim)

    def build_pipeline(self) -> IngestionPipeline:
        gaz = load_gazetteer(self.s.gazetteer_path)
        matcher = GazetteerMatcher(gaz)
        resolver = EntityResolver(self.s.resolution_threshold)
        resolver.load(gaz)
        extractor = build_extractor(self.s, self.ontology, self.llm, matcher)
        pipe = IngestionPipeline(self.s, self.store, self.embedder, extractor, resolver, matcher)
        pipe.bootstrap_resolver()
        return pipe

    def ingest(self, directory=None, files=None) -> IngestReport:
        pipe = self.build_pipeline()
        rep = pipe.ingest_files(files) if files else pipe.ingest_corpus(directory)
        self.retriever.refresh()
        return rep

    def run_algorithms(self) -> dict:
        out = run_graph_algorithms(self.store, self.llm)
        self.retriever.refresh()
        return out

    def build(self, reset: bool = False) -> dict:
        """Fluxo completo: schema -> ingestão -> algoritmos."""
        if reset:
            self.store.clear()
        self.setup_schema()
        rep = self.ingest()
        alg = self.run_algorithms()
        return {"ingest": rep, "algorithms": alg}

    # ------------------------------------------------------------ perguntas
    def ask(self, question: str, mode: str = "kag", k: int | None = None, hops: int | None = None,
            use_text2cypher: bool = True) -> QAResult:
        t0 = time.perf_counter()
        r = self.retriever.retrieve(question, mode=mode, k=k, hops=hops, use_text2cypher=use_text2cypher)
        a = self.generator.answer(r)
        return QAResult(r, a, (time.perf_counter() - t0) * 1000)

    def close(self):
        self.store.close()
