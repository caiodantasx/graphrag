"""Pipeline de ingestão: documento -> chunks -> NER/RE -> resolução -> Neo4j."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from .config import Settings
from .embeddings import Embedder
from .extraction import GazetteerMatcher
from .resolution import EntityResolver
from .store.base import GraphStore
from .text import SourceDocument, chunk_document, load_corpus, parse_document


@dataclass
class IngestReport:
    documents: int = 0
    chunks: int = 0
    entities: int = 0
    relations: int = 0
    mentions: int = 0
    extractor: str = ""
    merged: list[dict] = field(default_factory=list)  # decisões não triviais de resolução
    per_document: list[dict] = field(default_factory=list)


def load_gazetteer(path: Path) -> list[dict]:
    if not Path(path).exists():
        return []
    return json.loads(Path(path).read_text(encoding="utf-8"))["entities"]


class IngestionPipeline:
    def __init__(self, settings: Settings, store: GraphStore, embedder: Embedder, extractor,
                 resolver: EntityResolver, matcher: GazetteerMatcher):
        self.s = settings
        self.store = store
        self.embedder = embedder
        self.extractor = extractor
        self.resolver = resolver
        self.matcher = matcher
        self._entity_cache: dict[str, tuple] = {}

    def bootstrap_resolver(self):
        """Carrega entidades já existentes no grafo (ingestão incremental)."""
        for e in self.store.all_entities():
            self.resolver.register(e["name"], e["type"], e.get("aliases") or [], e["id"])
            self.matcher.add(e["name"], e["type"], e.get("aliases") or [])

    def ingest_corpus(self, directory: Path | None = None) -> IngestReport:
        docs = load_corpus(Path(directory or self.s.corpus_dir))
        return self.ingest_documents(docs)

    def ingest_files(self, paths: list[Path]) -> IngestReport:
        return self.ingest_documents([parse_document(Path(p)) for p in paths])

    def ingest_documents(self, docs: list[SourceDocument]) -> IngestReport:
        rep = IngestReport(extractor=getattr(self.extractor, "name", "?"))
        touched_entities: set[str] = set()
        for doc in docs:
            d_rel = d_ent = 0
            self._register_subject(doc)
            self.store.upsert_document(doc.id, doc.title, doc.source)
            chunks = chunk_document(doc, self.s.chunk_max_chars)
            embs = self.embedder.embed([c.text for c in chunks])
            prev = None
            for ch, emb in zip(chunks, embs):
                self.store.upsert_chunk(ch.id, doc.id, ch.idx, ch.text, emb, prev)
                prev = ch.id
                rep.chunks += 1
                res = self.extractor.extract(ch.text, doc.subject)
                rep.extractor = res.extractor
                ids: dict[str, str] = {}
                # entidades
                for ent in res.entities:
                    r = self.resolver.resolve(ent.name, ent.type)
                    for a in ent.aliases:
                        self.resolver._add_alias(r.entity, a)
                    ids[(ent.name, ent.type)] = r.entity.id
                    self._upsert_entity(r.entity)
                    touched_entities.add(r.entity.id)
                    self.store.add_mention(ch.id, r.entity.id)
                    rep.mentions += 1
                    d_ent += 1
                # relações
                for t in res.triples:
                    src = ids.get((t.src, t.src_type)) or self._resolve_and_upsert(t.src, t.src_type, ch.id, touched_entities)
                    dst = ids.get((t.dst, t.dst_type)) or self._resolve_and_upsert(t.dst, t.dst_type, ch.id, touched_entities)
                    if src == dst:
                        continue
                    self.store.upsert_relation(src, t.rel, dst, ch.id, t.year, t.confidence, t.evidence[:300], res.extractor)
                    rep.relations += 1
                    d_rel += 1
            rep.documents += 1
            rep.per_document.append({"documento": doc.title, "chunks": len(chunks), "menções": d_ent, "relações": d_rel})
        rep.entities = len(touched_entities)
        rep.merged = list(self.resolver.log)
        return rep

    def _register_subject(self, doc: SourceDocument):
        """Entidade principal declarada no cabeçalho (``subject``/``subject_type``)
        entra no dicionário de NER — permite ao extrator por regras reconhecer
        entidades novas; o extrator LLM não depende disso."""
        subj = doc.subject
        if not subj or self.matcher.find(subj):
            return
        etype = doc.meta.get("subject_type", "Person")
        if etype not in self.store.ontology.entity_types:
            return
        aliases = [subj.split()[-1]] if etype == "Person" and len(subj.split()) > 1 else []
        self.matcher.add(subj, etype, aliases)
        self.resolver.register(subj, etype, aliases)

    def _resolve_and_upsert(self, name: str, etype: str, chunk_id: str, touched: set) -> str:
        r = self.resolver.resolve(name, etype)
        self._upsert_entity(r.entity)
        self.store.add_mention(chunk_id, r.entity.id)
        touched.add(r.entity.id)
        return r.entity.id

    def _upsert_entity(self, ent):
        aliases = sorted(ent.aliases)
        sig = (ent.name, tuple(aliases))
        if self._entity_cache.get(ent.id) == sig:
            return
        self._entity_cache[ent.id] = sig
        # embedding do nó = nome + tipo + aliases (permite busca vetorial por entidade)
        emb = self.embedder.embed_one(f"{ent.name} ({ent.type}) {' '.join(aliases)}")
        self.store.upsert_entity(ent.id, ent.name, ent.type, aliases, emb)
