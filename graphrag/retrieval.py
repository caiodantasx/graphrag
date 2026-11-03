"""Recuperação híbrida: vetor (chunks e entidades) + full-text + travessia
Cypher multi-hop + caminhos mínimos + comunidades + plano lógico KAG.

Modos:
  * ``vector``   — baseline RAG clássico (só chunks por similaridade de cosseno).
  * ``graphrag`` — sementes híbridas (vetor+full-text+linking, fundidas por RRF),
                   expansão de vizinhança k-hop, caminhos mínimos entre âncoras,
                   resumos de comunidade, ranqueamento com PageRank.
  * ``kag``      — ``graphrag`` + raciocínio por forma lógica (KAG) e, se houver
                   LLM + Neo4j, Text2Cypher validado. (padrão)
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .config import Settings
from .embeddings import Embedder
from .kag import HeuristicPlanner, KAGExecutor, KAGResult, LLMPlanner, LogicalForm, detect_cues
from .linking import EntityLinker, LinkedEntity
from .llm import LLMClient
from .ontology import Ontology
from .store.base import GraphPath, GraphStore, QueryLogEntry
from .text2cypher import Text2Cypher, Text2CypherResult

RRF_K = 60


@dataclass
class Fact:
    id: str
    src: dict
    rel: str
    dst: dict
    props: dict
    score: float
    origin: set[str] = field(default_factory=set)  # kag | shortest | expand

    @property
    def sources(self) -> list[str]:
        return list(self.props.get("sources") or [])

    def text(self, ontology: Ontology) -> str:
        return ontology.verbalize(self.src["name"], self.rel, self.dst["name"], self.props)

    def triple(self) -> str:
        y = f" {{year: {self.props['year']}}}" if self.props.get("year") else ""
        return f"({self.src['name']})-[:{self.rel}{y}]->({self.dst['name']})"


@dataclass
class RetrievalResult:
    question: str
    mode: str
    chunks: list[dict] = field(default_factory=list)
    anchors: list[LinkedEntity] = field(default_factory=list)
    unlinked: list[str] = field(default_factory=list)
    seeds: list[dict] = field(default_factory=list)       # {id,name,type,score,via}
    facts: list[Fact] = field(default_factory=list)
    paths: list[GraphPath] = field(default_factory=list)
    communities: list[dict] = field(default_factory=list)
    provenance: list[dict] = field(default_factory=list)  # chunks que sustentam os fatos
    plan: LogicalForm | None = None
    kag: KAGResult | None = None
    t2c: Text2CypherResult | None = None
    cue_relations: list[str] = field(default_factory=list)
    queries: list[QueryLogEntry] = field(default_factory=list)
    retrieval_confidence: float = 0.0

    def nodes(self) -> dict[str, dict]:
        out: dict[str, dict] = {}
        for f in self.facts:
            out.setdefault(f.src["id"], f.src)
            out.setdefault(f.dst["id"], f.dst)
        return out


class HybridRetriever:
    def __init__(self, settings: Settings, store: GraphStore, embedder: Embedder, ontology: Ontology, llm: LLMClient):
        self.s, self.store, self.embedder, self.o, self.llm = settings, store, embedder, ontology, llm
        self.heuristic = HeuristicPlanner(ontology)
        self.planner = LLMPlanner(llm, ontology, self.heuristic) if llm.enabled else self.heuristic
        self.executor = KAGExecutor(store, ontology)
        self.t2c = Text2Cypher(llm, ontology, store)
        self._linker: EntityLinker | None = None

    @property
    def linker(self) -> EntityLinker:
        if self._linker is None:
            self.refresh()
        return self._linker

    def refresh(self):
        """Recarrega o índice de nomes (após nova ingestão)."""
        self._linker = EntityLinker(self.store.all_entities())

    # ------------------------------------------------------------------
    def retrieve(self, question: str, mode: str = "kag", k: int | None = None, hops: int | None = None,
                 use_text2cypher: bool = True) -> RetrievalResult:
        k = k or self.s.top_k
        hops = hops or self.s.max_hops
        self.store.reset_log()
        res = RetrievalResult(question, mode)
        q_emb = self.embedder.embed_one(question)

        # 1) busca vetorial em chunks (comum a todos os modos)
        res.chunks = self.store.vector_chunks(q_emb, k)
        if mode == "vector":
            top = res.chunks[0]["score"] if res.chunks else 0.0
            res.retrieval_confidence = max(0.0, min(1.0, (top - 0.5) * 2))
            res.queries = list(self.store.query_log)
            return res

        # 2) entity linking + sementes híbridas (RRF)
        res.anchors = self.linker.link(question)
        res.unlinked = self.linker.unlinked_mentions(question, res.anchors)
        res.cue_relations = sorted({r for c in detect_cues(question) for r in c["rels"]})
        vec_e = self.store.vector_entities(q_emb, k)
        ft_e = self.store.fulltext_entities(question, k)
        ch_e = self.store.chunk_entities([c["id"] for c in res.chunks[:3]])
        res.seeds = self._fuse(res.anchors, vec_e, ft_e, ch_e)
        anchor_ids = [a.id for a in res.anchors]
        seed_ids = anchor_ids or [s["id"] for s in res.seeds[:3]]

        # 3) travessia multi-hop + caminhos mínimos entre âncoras
        expand_paths = self.store.expand(seed_ids, hops, limit=200)
        sp_paths: list[GraphPath] = []
        for i in range(len(anchor_ids)):
            for j in range(i + 1, len(anchor_ids)):
                sp_paths += self.store.shortest_paths(anchor_ids[i], anchor_ids[j], 4)

        # 4) KAG (forma lógica) e Text2Cypher
        if mode == "kag":
            res.plan = self.planner.plan(question, res.anchors)
            res.kag = self.executor.run(res.plan, self.linker) if res.anchors else KAGResult(ok=False)
            if res.plan.cue_relations:
                res.cue_relations = sorted(set(res.cue_relations) | set(res.plan.cue_relations))
            if use_text2cypher and self.t2c.available:
                res.t2c = self.t2c.run(question, res.anchors)

        # 5) comunidades das âncoras/sementes (contexto global)
        res.communities = self.store.communities_of(seed_ids)[:2]

        # 6) pontuação e seleção de fatos
        res.facts, res.paths = self._select(res, expand_paths, sp_paths, set(anchor_ids))

        # 7) proveniência: chunks que sustentam os fatos selecionados
        src_ids: list[str] = []
        for f in res.facts:
            for s in f.sources:
                if s not in src_ids:
                    src_ids.append(s)
        res.provenance = self.store.chunks_by_ids(src_ids[:12])

        res.retrieval_confidence = self._confidence(res)
        res.queries = list(self.store.query_log)
        return res

    # ------------------------------------------------------------------
    def _fuse(self, anchors, vec_e, ft_e, ch_e) -> list[dict]:
        scores: dict[str, float] = {}
        meta: dict[str, dict] = {}
        via: dict[str, set] = {}

        def add(rank_list, name):
            for rank, ent in enumerate(rank_list):
                eid = ent["id"]
                scores[eid] = scores.get(eid, 0.0) + 1.0 / (RRF_K + rank + 1)
                meta.setdefault(eid, ent)
                via.setdefault(eid, set()).add(name)

        add([{"id": a.id, "name": a.name, "type": a.type} for a in anchors], "linking")
        for a in anchors:  # âncoras exatas dominam
            scores[a.id] += 1.0
        add([e for e, _ in vec_e], "vector")
        add([e for e, _ in ft_e], "fulltext")
        seen, ch_list = set(), []
        for _, e in ch_e:
            if e["id"] not in seen:
                seen.add(e["id"])
                ch_list.append(e)
        add(ch_list, "chunk-mentions")
        ranked = sorted(scores.items(), key=lambda x: -x[1])
        return [{"id": eid, "name": meta[eid].get("name"), "type": meta[eid].get("type"),
                 "score": round(s, 4), "via": sorted(via[eid])} for eid, s in ranked[:10]]

    def _select(self, res: RetrievalResult, expand_paths, sp_paths, anchors: set[str]):
        facts: dict[tuple, Fact] = {}
        cues = set(res.cue_relations)

        def put(src, rel, dst, props, score, origin):
            key = (src["id"], rel, dst["id"])
            f = facts.get(key)
            if f is None:
                facts[key] = Fact("", src, rel, dst, dict(props or {}), score, {origin})
            else:
                f.score = max(f.score, score) + 0.1
                f.origin.add(origin)

        if res.kag and res.kag.facts:
            for f in res.kag.facts:
                put(f["src"], f["rel"], f["dst"], f["props"], 5.0, "kag")
        if res.t2c and res.t2c.ok:
            pass  # linhas tabulares exibidas à parte (não são triplas)

        def path_score(p: GraphPath) -> float:
            cue_hits = sum(1 for r in p.rels if r["type"] in cues)
            pr = sum((n.get("pagerank") or 0.0) for n in p.nodes)
            return (1.5 * cue_hits + 1.0) / len(p.rels) + 5 * pr

        for p in sp_paths:
            s = 2.5 + path_score(p)
            for r in p.rels:
                put(_find(p, r["start"]), r["type"], _find(p, r["end"]), r["props"], s, "shortest")
        for p in expand_paths:
            s = path_score(p)
            for r in p.rels:
                bonus = 0.5 if (r["start"] in anchors or r["end"] in anchors) else 0.0
                bonus += 1.0 if r["type"] in cues else 0.0
                put(_find(p, r["start"]), r["type"], _find(p, r["end"]), r["props"], s + bonus, "expand")

        ranked = sorted(facts.values(), key=lambda f: -f.score)[: self.s.max_facts]
        for i, f in enumerate(ranked, 1):
            f.id = f"F{i}"
        # caminhos exibidos: KAG/mínimos primeiro, depois os melhores da expansão
        kept_keys = {(f.src["id"], f.rel, f.dst["id"]) for f in ranked}
        shown: list[GraphPath] = []
        seen = set()
        candidates = (res.kag.paths if res.kag else []) + sp_paths + sorted(expand_paths, key=lambda p: -path_score(p))
        for p in candidates:
            if p.key in seen or not all((r["start"], r["type"], r["end"]) in kept_keys for r in p.rels):
                continue
            seen.add(p.key)
            shown.append(p)
            if len(shown) >= 8:
                break
        return ranked, shown

    def _confidence(self, res: RetrievalResult) -> float:
        c = 0.0
        if res.anchors:
            c += 0.45
        if res.kag is not None:
            c += 0.35 if res.kag.ok else 0.0
        else:
            c += 0.2 if any("shortest" in f.origin for f in res.facts) else 0.1
        if res.facts:
            c += 0.2
        if res.unlinked:
            c = min(c, 0.3)
        return round(min(c, 1.0), 3)


def _find(p: GraphPath, nid: str) -> dict:
    for n in p.nodes:
        if n["id"] == nid:
            return n
    return {"id": nid, "name": nid, "type": None}
