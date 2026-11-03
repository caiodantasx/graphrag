"""Geração ancorada (grounded generation) + verificação pós-geração.

Mitigação de alucinações em três estágios:
  1. ANTES: o contexto contém apenas fatos do grafo numerados (F#), caminhos
     (P#) e trechos (C#) com procedência; premissas com entidades inexistentes
     no grafo reduzem a confiança e disparam abstenção.
  2. DURANTE: prompt restritivo (somente contexto, citação obrigatória por
     frase, frase padrão de abstenção) e temperatura 0.
  3. DEPOIS: ``verify_answer`` checa citações válidas por sentença, se toda
     entidade citada na resposta existe no contexto recuperado e se há nomes
     próprios que não existem no grafo; calcula a confiança final.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .embeddings import Embedder, cosine
from .linking import EntityLinker
from .llm import LLMClient, LLMError
from .ontology import Ontology
from .retrieval import RetrievalResult
from .text import split_sentences

ABSTAIN_TEXT = "Não encontrei evidências suficientes no grafo de conhecimento para responder com segurança."
CITE_RE = re.compile(r"\[((?:[FCP]\d+)(?:\s*,\s*[FCP]\d+)*)\]")

ANSWER_SYSTEM = f"""Você é um assistente de perguntas e respostas ancorado em um grafo de conhecimento.
Regras obrigatórias:
1. Use SOMENTE as informações do CONTEXTO. Não use conhecimento externo, mesmo que saiba a resposta.
2. Toda frase da resposta deve terminar com a(s) citação(ões) da evidência usada: [F3], [C1], [P2].
3. Para perguntas multi-hop, explicite a cadeia de raciocínio seguindo os caminhos do grafo.
4. Se o contexto não for suficiente, responda exatamente: "{ABSTAIN_TEXT}"
5. Responda em português, de forma concisa (até 5 frases), terminando com uma linha "Resposta: ...".
"""


@dataclass
class Answer:
    text: str
    abstained: bool
    confidence: float
    generator: str
    verification: dict = field(default_factory=dict)
    answer_entity_ids: list[str] = field(default_factory=list)
    context: str = ""


# ---------------------------------------------------------------- contexto
def build_context(res: RetrievalResult, ontology: Ontology) -> str:
    parts = []
    if res.mode != "vector":
        if res.facts:
            titles = {c["id"]: c["title"] for c in res.provenance}
            lines = []
            for f in res.facts:
                src = ", ".join(sorted({titles.get(s, s) for s in f.sources})) or "grafo"
                lines.append(f"{f.id}: {f.triple()} — {f.text(ontology)}. (fonte: {src})")
            parts.append("FATOS DO GRAFO:\n" + "\n".join(lines))
        if res.paths:
            fid = {(f.src["id"], f.rel, f.dst["id"]): f.id for f in res.facts}
            lines = []
            for i, p in enumerate(res.paths, 1):
                lines.append(f"P{i}: {path_to_text(p)}  [{', '.join(fid.get((r['start'], r['type'], r['end']), '?') for r in p.rels)}]")
            parts.append("CAMINHOS NO GRAFO:\n" + "\n".join(lines))
        if res.kag is not None and res.plan is not None:
            ans = ", ".join(n["name"] for n in res.kag.answers) or "(vazio)"
            parts.append("RESULTADO DO PLANO LÓGICO (KAG):\n" + "\n".join(res.kag.trace) + f"\nValores encontrados: {ans}")
        if res.t2c is not None and res.t2c.ok:
            parts.append("RESULTADO CYPHER:\n" + "\n".join(str(r) for r in res.t2c.rows[:15]))
        if res.communities:
            parts.append("COMUNIDADES:\n" + "\n".join(c["summary"] for c in res.communities))
    chunks = res.chunks if res.mode == "vector" else res.chunks[:3]
    if chunks:
        parts.append("TRECHOS:\n" + "\n".join(f"C{i}: [{c['title']}] {c['text']}" for i, c in enumerate(chunks, 1)))
    return "\n\n".join(parts)


def path_to_text(p) -> str:
    out = p.nodes[0]["name"]
    for i, r in enumerate(p.rels):
        cur, nxt = p.nodes[i]["id"], p.nodes[i + 1]
        arrow = f"-[:{r['type']}]->" if r["start"] == cur else f"<-[:{r['type']}]-"
        out += f" {arrow} {nxt['name']}"
    return out


# ---------------------------------------------------------------- gerador
class AnswerGenerator:
    def __init__(self, llm: LLMClient, ontology: Ontology, embedder: Embedder, linker_provider, abstain_threshold: float):
        self.llm, self.o, self.embedder = llm, ontology, embedder
        self.linker_provider = linker_provider  # callable -> EntityLinker
        self.threshold = abstain_threshold

    def answer(self, res: RetrievalResult) -> Answer:
        ctx = build_context(res, self.o)
        generator = "llm" if self.llm.enabled else "extractive"
        text = None
        if self._must_abstain(res):
            text = self._abstain_reason(res)
        elif self.llm.enabled:
            try:
                text = self.llm.chat(ANSWER_SYSTEM, f"CONTEXTO:\n{ctx}\n\nPERGUNTA: {res.question}").strip()
            except LLMError as e:
                generator = f"extractive (LLM indisponível: {e.__class__.__name__})"
        if text is None:
            text = self._extractive(res)
        ans = Answer(text, ABSTAIN_TEXT in text, 0.0, generator, context=ctx)
        ans.verification = verify_answer(ans.text, res, self.linker_provider())
        ans.confidence = self._final_confidence(res, ans)
        if not ans.abstained and ans.confidence < self.threshold:
            ans.verification["warnings"].append(
                f"Confiança {ans.confidence:.2f} abaixo do limiar {self.threshold:.2f}: resposta substituída por abstenção."
            )
            ans.text = f"{ABSTAIN_TEXT}\n\n(Resposta candidata descartada pela verificação: {ans.text})"
            ans.abstained = True
        ans.answer_entity_ids = self._answer_entities(res, ans)
        return ans

    # ------------------------------------------------------------ abstenção
    def _must_abstain(self, res: RetrievalResult) -> bool:
        if res.mode == "vector":
            return not res.chunks
        if res.unlinked:
            return True
        return not res.anchors and not res.facts

    def _abstain_reason(self, res: RetrievalResult) -> str:
        if res.unlinked:
            return (f"{ABSTAIN_TEXT} A pergunta menciona {', '.join(repr(u) for u in res.unlinked)}, "
                    "que não existe(m) no grafo — a premissa não pode ser verificada.")
        return ABSTAIN_TEXT

    # ------------------------------------------------------------ extrativo (offline)
    def _extractive(self, res: RetrievalResult) -> str:
        if res.mode == "vector":
            return self._extractive_vector(res)
        fid = {(f.src["id"], f.rel, f.dst["id"]): f.id for f in res.facts}

        def cite(f):
            return fid.get((f["src"]["id"], f["rel"], f["dst"]["id"]), "")

        if res.kag is not None and res.kag.ok:
            lines = []
            if res.plan and res.plan.operation == "path":
                for p in res.kag.paths[:3]:
                    ids = [fid.get((r["start"], r["type"], r["end"])) for r in p.rels]
                    lines.append(f"Caminho no grafo: {path_to_text(p)} [{', '.join(i for i in ids if i)}].")
            else:
                for f in res.kag.facts[:8]:
                    c = cite(f)
                    lines.append(f"{self.o.verbalize(f['src']['name'], f['rel'], f['dst']['name'], f['props'])} [{c}]." if c else "")
            lines = [l for l in lines if l]
            if res.kag.answers:
                cites = sorted({cite(f) for f in res.kag.facts if cite(f) and
                                (f["dst"]["id"] in {a["id"] for a in res.kag.answers} or f["src"]["id"] in {a["id"] for a in res.kag.answers})},
                               key=lambda x: int(x[1:]))
                names = ", ".join(a["name"] for a in res.kag.answers)
                # em cadeias multi-hop, explicita as variáveis intermediárias
                if res.plan and res.plan.operation == "chain" and len(res.plan.steps) > 1:
                    inter = [n["name"] for st in res.plan.steps[:-1] for n in res.kag.bindings.get(st.output, [])
                             if n["id"] in {f["src"]["id"] for f in res.kag.facts} | {f["dst"]["id"] for f in res.kag.facts}]
                    if inter:
                        names += f" (via {', '.join(dict.fromkeys(inter))})"
                lines.append(f"Resposta: {names} [{', '.join(cites[:4])}]." if cites else f"Resposta: {names}.")
            return " ".join(lines) if lines else ABSTAIN_TEXT
        # GraphRAG sem plano, 2+ âncoras: caminho mínimo entre elas
        anchors = {a.id for a in res.anchors}
        sp = [p for p in res.paths if p.source == "shortest"]
        if len(anchors) >= 2 and sp:
            lines, inner, all_ids = [], [], []
            for p in sp[:3]:
                ids = [i for i in (fid.get((r["start"], r["type"], r["end"])) for r in p.rels) if i]
                all_ids += ids
                lines.append(f"Caminho no grafo: {path_to_text(p)} [{', '.join(ids)}].")
                inner += [n["name"] for n in p.nodes if n["id"] not in anchors]
            if inner:
                lines.append(f"Resposta: {', '.join(dict.fromkeys(inner))} [{', '.join(dict.fromkeys(all_ids))}].")
            return " ".join(lines)
        # GraphRAG sem plano: fatos ligados às âncoras com relação pedida na pergunta
        cues = set(res.cue_relations)
        rel_facts = [f for f in res.facts if f.rel in cues and (f.src["id"] in anchors or f.dst["id"] in anchors)]
        if not rel_facts:
            return ABSTAIN_TEXT
        rel_facts = rel_facts[:4]
        lines = [f"{f.text(self.o)} [{f.id}]." for f in rel_facts]
        others = []
        for f in rel_facts:
            for n in (f.src, f.dst):
                if n["id"] not in anchors and n["name"] not in others:
                    others.append(n["name"])
        if others:
            lines.append(f"Resposta: {', '.join(others)} [{', '.join(f.id for f in rel_facts)}].")
        return " ".join(lines)

    def _extractive_vector(self, res: RetrievalResult) -> str:
        q = self.embedder.embed_one(res.question)
        best = None
        for i, c in enumerate(res.chunks[:3], 1):
            for s in split_sentences(c["text"]):
                sc = cosine(q, self.embedder.embed_one(s))
                if best is None or sc > best[0]:
                    best = (sc, s, i)
        if not best:
            return ABSTAIN_TEXT
        return f"{best[1]} [C{best[2]}]"

    # ------------------------------------------------------------ confiança
    def _final_confidence(self, res: RetrievalResult, ans: Answer) -> float:
        v = ans.verification
        if ans.abstained:
            return round(res.retrieval_confidence * 0.5, 3)
        c = 0.45 * res.retrieval_confidence + 0.30 * v["citation_coverage"] + 0.25 * v["entity_grounding"]
        if v["invalid_citations"]:
            c *= 0.7
        if v["ungrounded_entities"] or v["unknown_names"]:
            c *= 0.5
        return round(min(c, 1.0), 3)

    def _answer_entities(self, res: RetrievalResult, ans: Answer) -> list[str]:
        if ans.abstained:
            return []
        if res.kag is not None and res.kag.ok and res.kag.answers:
            return [a["id"] for a in res.kag.answers]
        line = ans.text.split("Resposta:")[-1]
        return [l.id for l in self.linker_provider().link(line)]


# ---------------------------------------------------------------- verificação
def verify_answer(text: str, res: RetrievalResult, linker: EntityLinker) -> dict:
    valid = {f.id for f in res.facts}
    n_chunks = len(res.chunks) if res.mode == "vector" else min(3, len(res.chunks))
    valid |= {f"C{i}" for i in range(1, n_chunks + 1)}
    valid |= {f"P{i}" for i in range(1, len(res.paths) + 1)}
    report = {"sentences": 0, "cited_sentences": 0, "citation_coverage": 0.0, "invalid_citations": [],
              "mentioned_entities": [], "ungrounded_entities": [], "unknown_names": [], "entity_grounding": 1.0,
              "warnings": []}
    if ABSTAIN_TEXT in text:
        report["citation_coverage"] = 1.0
        return report
    sentences = [s for s in split_sentences(text.replace("\n", " ")) if len(s.strip()) > 2]
    report["sentences"] = len(sentences)
    for s in sentences:
        cites = [c.strip() for grp in CITE_RE.findall(s) for c in grp.split(",")]
        bad = [c for c in cites if c not in valid]
        report["invalid_citations"] += bad
        if any(c in valid for c in cites):
            report["cited_sentences"] += 1
    report["citation_coverage"] = round(report["cited_sentences"] / max(1, report["sentences"]), 3)

    # entidades citadas na resposta precisam estar no contexto recuperado
    ctx_ids = set(res.nodes()) | {a.id for a in res.anchors}
    ctx_text_entities = set()
    for c in (res.chunks if res.mode == "vector" else res.chunks[:3]) + res.provenance:
        ctx_text_entities |= {l.id for l in linker.link(c["text"])}
    ctx_ids |= ctx_text_entities
    clean = CITE_RE.sub("", text)
    clean = re.sub(r"\[:[A-Z_]+\]", "", clean)  # tipos de relação em caminhos
    mentioned = linker.link(clean)
    report["mentioned_entities"] = [m.name for m in mentioned]
    report["ungrounded_entities"] = [m.name for m in mentioned if m.id not in ctx_ids]
    # nomes próprios que não estão no grafo NEM nos trechos recuperados => provável invenção
    from .text import normalize

    ctx_norm = normalize(" ".join(c["text"] for c in res.chunks + res.provenance))
    for s in split_sentences(clean.replace("\n", " ")):
        s2 = re.sub(r"^(Resposta|Caminho no grafo|Conexão)\s*:\s*", "", s.strip())
        for u in linker.unlinked_mentions(s2, linker.link(s2)):
            if len(u) > 2 and normalize(u) not in ctx_norm:
                report["unknown_names"].append(u)
    if mentioned:
        report["entity_grounding"] = round(1 - len(report["ungrounded_entities"]) / len(mentioned), 3)
    if report["ungrounded_entities"]:
        report["warnings"].append(f"Entidades fora do contexto recuperado: {report['ungrounded_entities']}")
    if report["unknown_names"]:
        report["warnings"].append(f"Nomes próprios inexistentes no grafo: {report['unknown_names']}")
    if report["invalid_citations"]:
        report["warnings"].append(f"Citações inválidas: {report['invalid_citations']}")
    return report
