"""KAG — raciocínio guiado por forma lógica (Logical-Form-guided reasoning).

A pergunta é decomposta em um PLANO de operadores sobre o grafo, inspirado no
KAG (Liang et al., 2024). O LLM (ou o planejador heurístico, offline) só
escolhe operadores de um vocabulário fechado; quem gera o Cypher é o executor,
a partir de templates parametrizados e validados contra a ontologia. Isso
evita Cypher alucinado e torna cada passo do raciocínio auditável.

Operadores:
  * ``chain``: sequência de passos ``(var|âncora) -[REL]-> ?var`` (multi-hop);
    uma variável atribuída por dois passos é a INTERSEÇÃO dos resultados.
  * ``path``: caminhos mínimos entre âncoras (perguntas de "conexão" e
    perguntas com várias âncoras), filtrados pelo tipo da resposta e pelas
    relações citadas na pergunta.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field

from .linking import EntityLinker, LinkedEntity
from .llm import LLMClient, LLMError
from .ontology import Ontology
from .store.base import GraphPath, GraphStore
from .text import strip_accents


@dataclass
class Step:
    source: str            # "anchor:<id>" ou "var:<nome>"
    relations: list[str]
    direction: str         # out | in | any
    target_type: str | None
    output: str            # nome da variável


@dataclass
class LogicalForm:
    operation: str = "chain"  # chain | path
    anchors: list[str] = field(default_factory=list)  # ids
    steps: list[Step] = field(default_factory=list)
    answer_var: str | None = None
    answer_type: str | None = None
    cue_relations: list[str] = field(default_factory=list)
    planner: str = "heuristic"
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class KAGResult:
    ok: bool
    answers: list[dict] = field(default_factory=list)          # nós-resposta
    facts: list[dict] = field(default_factory=list)            # {src, rel, dst, props}
    paths: list[GraphPath] = field(default_factory=list)
    bindings: dict[str, list[dict]] = field(default_factory=dict)
    trace: list[str] = field(default_factory=list)


# ---------------------------------------------------------------- léxico
# (regex sobre a pergunta normalizada, relações, modo, prioridade)
#   modo: verb (direção pela posição da âncora), in, out, any
LEXICON: list[tuple[str, list[str], str, int]] = [
    (r"\borientador(a)?(es)? de\b", ["ADVISED"], "in", 3),
    (r"\b(orientand[oa]s?|alun[oa]s? de)\b", ["ADVISED"], "out", 3),
    (r"\borientad[oa]s? por\b", ["ADVISED"], "out", 3),
    (r"\borient(ou|aram)\b", ["ADVISED"], "verb", 3),
    (r"\b(nasceu|natal|nascid[oa]|nasceram)\b", ["BORN_IN"], "verb", 2),
    (r"\b(estudou|estudaram|doutorado|mestrado|formou|graduou)\b", ["STUDIED_AT"], "verb", 1),
    (r"\b(trabalh\w*|professor\w*|lecion\w*)\b", ["WORKED_AT"], "verb", 2),
    (r"\b(influenciad[oa]s?|inspirad[oa]s?) (por|pel[oa]s?)\b", ["INFLUENCED"], "out", 3),
    (r"\b(influenci\w*|inspir\w*)\b", ["INFLUENCED"], "verb", 2),
    (r"\b(cri(ou|aram|ador|adora|adores|ad[oa]s?)|desenvolv\w*|invent\w*|propos|propost\w*|projet\w*|formul\w*|cunhou|autor\w*)\b",
     ["CREATED", "DEVELOPED_AT"], "verb", 2),
    (r"\b(colabor\w*|parceria|junto com)\b", ["COLLABORATED_WITH"], "any", 2),
    (r"\b(encontrou|conheceu|encontraram)\b", ["MET"], "any", 2),
    (r"\b(receb\w*|ganh\w*|venc\w*|laureado\w*|premiad\w*)\b", ["RECEIVED"], "verb", 2),
    (r"\b(homenage\w*|nomead\w*|batizad\w*)\b", ["NAMED_AFTER"], "verb", 2),
    (r"\b(concedid\w*|concede|entregue)\b", ["AWARDED_BY"], "verb", 2),
    (r"\b(fund(ou|aram|ador\w*))\b", ["FOUNDED"], "verb", 2),
    (r"\b(fica|ficam|localiza\w*|sediad\w*|situad\w*)\b", ["LOCATED_IN"], "verb", 2),
    (r"\b(faz parte|pertence\w*)\b", ["PART_OF"], "verb", 2),
]
PATH_CUES = re.compile(r"\b(conexao|conexoes|relacao entre|ligacao|liga|conecta\w*|caminho)\b")

ANSWER_TYPES: list[tuple[str, str]] = [
    (r"\b(quem|que pessoa|qual pessoa|quais pessoas|pessoas?|cientistas?)\b", "Person"),
    (r"\b(cidade|pais|paises|estado|regiao)\b", "Location"),
    (r"\b(instituic\w*|universidade\w*|empresa\w*|laboratorio\w*|organizac\w*)\b", "Organization"),
    (r"\b(linguage\w*|sistema\w*|algoritmo\w*|tecnologi\w*|maquina\w*|compilador\w*)\b", "Technology"),
    (r"\b(premio\w*|distinc\w*)\b", "Award"),
    (r"\b(conceito\w*|teoria\w*)\b", "Concept"),
]


def _norm_q(q: str) -> str:
    return strip_accents(q).lower()


def detect_answer_type(question: str) -> str | None:
    q = _norm_q(question)
    best = None
    for pat, t in ANSWER_TYPES:
        m = re.search(pat, q)
        # o tipo pedido costuma aparecer cedo ("Em que cidade...", "Qual linguagem...")
        if m and (best is None or m.start() < best[0]):
            best = (m.start(), t)
    return best[1] if best else None


def detect_cues(question: str) -> list[dict]:
    q = _norm_q(question)
    cues = []
    for pat, rels, mode, prio in LEXICON:
        for m in re.finditer(pat, q):
            cues.append({"start": m.start(), "end": m.end(), "rels": rels, "mode": mode, "prio": prio, "text": m.group(0)})
    # 1) pistas sobrepostas: fica a de maior prioridade
    cues.sort(key=lambda c: (c["start"], -c["prio"]))
    kept: list[dict] = []
    for c in cues:
        clash = [k for k in kept if k["start"] <= c["start"] < k["end"] or c["start"] <= k["start"] < c["end"]]
        if clash:
            k = clash[0]
            if c["prio"] > k["prio"]:
                kept.remove(k)
                kept.append(c)
            continue
        kept.append(c)
    # 2) "orientou o doutorado de X" fala de orientação, não de onde X estudou
    advised = [c for c in kept if c["rels"] == ["ADVISED"]]
    kept = [c for c in kept if not (
        c["rels"] == ["STUDIED_AT"] and re.match(r"(doutorado|mestrado)", c["text"])
        and any(abs(a["start"] - c["start"]) <= 25 for a in advised))]
    return sorted(kept, key=lambda c: c["start"])


# ---------------------------------------------------------------- planejadores
class HeuristicPlanner:
    """Planejador determinístico (sem LLM)."""

    name = "heuristic"

    def __init__(self, ontology: Ontology):
        self.o = ontology

    def plan(self, question: str, anchors: list[LinkedEntity]) -> LogicalForm:
        q = _norm_q(question)
        lf = LogicalForm(planner=self.name, anchors=[a.id for a in anchors])
        lf.answer_type = detect_answer_type(question)
        cues = detect_cues(question)
        lf.cue_relations = sorted({r for c in cues for r in c["rels"]})
        if not anchors:
            lf.notes.append("Nenhuma entidade da pergunta foi encontrada no grafo.")
            return lf
        if len(anchors) >= 2 or PATH_CUES.search(q):
            lf.operation = "path"
            if len(anchors) < 2:
                lf.notes.append("Pergunta de conexão com apenas uma âncora; usando expansão.")
                lf.operation = "chain"
            else:
                return lf
        anchor = anchors[0]
        a_start = len(strip_accents(question[: anchor.start]))
        a_end = a_start + len(strip_accents(anchor.surface))

        def dist(c):
            return min(abs(c["start"] - a_end), abs(c["end"] - a_start))

        ordered = sorted(cues, key=dist)
        current_src, current_types = f"anchor:{anchor.id}", {anchor.type}
        last_rels = None
        for i, c in enumerate(ordered):
            if c["rels"] == last_rels:
                continue
            last_rels = c["rels"]
            var = f"x{i + 1}"
            first_hop = not lf.steps
            rels, direction, next_types = self._resolve(c, current_types, anchor_after=c["start"] < a_start,
                                                        first_hop=first_hop)
            if not rels:
                lf.notes.append(f"Pista '{c['text']}' incompatível com {sorted(current_types)}; ignorada.")
                continue
            lf.steps.append(Step(current_src, rels, direction, None, var))
            current_src = f"var:{var}"
            current_types = next_types
        if lf.steps:
            last = lf.steps[-1]
            if lf.answer_type and self._type_ok(last.relations, last.direction, lf.answer_type):
                last.target_type = lf.answer_type
            lf.answer_var = last.output
        else:
            lf.notes.append("Nenhuma relação identificada; resposta virá da vizinhança do grafo.")
        return lf

    def _resolve(self, cue: dict, current_types: set[str], anchor_after: bool, first_hop: bool):
        """Escolhe (relações, direção) compatíveis com os tipos atuais pela
        assinatura domínio/alcance da ontologia. Para relações em que domínio e
        alcance coincidem (ex.: ADVISED Pessoa->Pessoa) e verbo na voz ativa,
        decide pela posição da âncora (só no 1º salto; nos demais, a variável
        anterior é o sujeito do verbo -> "out")."""
        rels, dirs, next_types = [], set(), set()
        for r in cue["rels"]:
            rt = self.o.relations[r]
            possible = []
            if rt.symmetric:
                possible = ["any"] if current_types & (set(rt.domain) | set(rt.range)) else []
            else:
                if current_types & set(rt.domain):
                    possible.append("out")
                if current_types & set(rt.range):
                    possible.append("in")
            if not possible:
                continue
            if cue["mode"] in ("in", "out"):
                d = cue["mode"] if cue["mode"] in possible else ("any" if possible == ["any"] else None)
            elif cue["mode"] == "any" or possible == ["any"]:
                d = "any"
            elif len(possible) == 1:
                d = possible[0]
            elif first_hop:
                d = "in" if anchor_after else "out"
            else:
                d = "out"
            if d:
                rels.append(r)
                dirs.add(d)
                next_types |= set(rt.range if d == "out" else rt.domain if d == "in" else rt.range + rt.domain)
        if not rels:
            return [], None, set()
        direction = dirs.pop() if len(dirs) == 1 else "any"
        return rels, direction, next_types

    def _type_ok(self, rels, direction, t):
        for r in rels:
            rt = self.o.relations[r]
            pool = rt.range if direction == "out" else rt.domain if direction == "in" else rt.range + rt.domain
            if t in pool:
                return True
        return False


LF_SYSTEM = """Você converte perguntas em um PLANO LÓGICO executável sobre um grafo de conhecimento.
Use somente os tipos de relação do esquema. Não responda a pergunta; apenas planeje.

{schema}

Âncoras (entidades da pergunta já localizadas no grafo): {anchors}

Formato JSON:
{{"operation": "chain" | "path",
  "steps": [{{"source": "anchor:<id>" | "var:<nome>", "relations": ["REL"], "direction": "out"|"in"|"any",
             "target_type": "<Rótulo>"|null, "output": "<nome>"}}],
  "answer_var": "<nome>", "answer_type": "<Rótulo>"|null}}
Regras: "out" segue a seta (origem->destino); "in" vai contra a seta. Para restringir uma
variável por duas âncoras, crie dois passos com o MESMO "output" (interseção).
Use "path" apenas para perguntas sobre a conexão entre duas âncoras."""


class LLMPlanner:
    name = "llm"

    def __init__(self, llm: LLMClient, ontology: Ontology, fallback: HeuristicPlanner):
        self.llm, self.o, self.fallback = llm, ontology, fallback

    def plan(self, question: str, anchors: list[LinkedEntity]) -> LogicalForm:
        if not anchors:
            return self.fallback.plan(question, anchors)
        anchors_txt = "; ".join(f"{a.id} = {a.name} ({a.type})" for a in anchors)
        try:
            data = self.llm.chat_json(LF_SYSTEM.format(schema=self.o.describe(), anchors=anchors_txt),
                                      f"Pergunta: {question}")
            lf = self._validate(data, anchors)
            lf.cue_relations = sorted({r for s in lf.steps for r in s.relations})
            return lf
        except (LLMError, ValueError, KeyError, TypeError) as e:
            lf = self.fallback.plan(question, anchors)
            lf.notes.append(f"Planejador LLM falhou ({e.__class__.__name__}); usado heurístico.")
            return lf

    def _validate(self, d: dict, anchors) -> LogicalForm:
        ids = {a.id for a in anchors}
        lf = LogicalForm(operation=d.get("operation", "chain"), anchors=[a.id for a in anchors], planner=self.name)
        if lf.operation not in ("chain", "path"):
            raise ValueError("operação inválida")
        defined = set()
        for s in d.get("steps", []):
            src = s["source"]
            if src.startswith("anchor:") and src[7:] not in ids:
                raise ValueError(f"âncora desconhecida {src}")
            if src.startswith("var:") and src[4:] not in defined:
                raise ValueError(f"variável indefinida {src}")
            rels = [r for r in s["relations"] if r in self.o.relations]
            if not rels or s.get("direction") not in ("out", "in", "any"):
                raise ValueError("passo inválido")
            tt = s.get("target_type")
            tt = tt if tt in self.o.entity_types else None
            lf.steps.append(Step(src, rels, s["direction"], tt, str(s["output"])))
            defined.add(str(s["output"]))
        lf.answer_var = d.get("answer_var") if d.get("answer_var") in defined else (lf.steps[-1].output if lf.steps else None)
        at = d.get("answer_type")
        lf.answer_type = at if at in self.o.entity_types else None
        if lf.operation == "chain" and not lf.steps:
            raise ValueError("plano vazio")
        return lf


# ---------------------------------------------------------------- executor
class KAGExecutor:
    def __init__(self, store: GraphStore, ontology: Ontology):
        self.store, self.o = store, ontology

    def run(self, lf: LogicalForm, linker: EntityLinker) -> KAGResult:
        if lf.operation == "path":
            return self._run_path(lf, linker)
        return self._run_chain(lf, linker)

    def _run_chain(self, lf: LogicalForm, linker: EntityLinker) -> KAGResult:
        res = KAGResult(ok=False)
        vars_: dict[str, dict[str, dict]] = {}
        step_facts: list[list[dict]] = []
        for st in lf.steps:
            if st.source.startswith("anchor:"):
                aid = st.source[7:]
                e = linker.entities.get(aid, {"id": aid, "name": aid, "type": None})
                src_nodes = {aid: {"id": aid, "name": e["name"], "type": e["type"]}}
            else:
                src_nodes = vars_.get(st.source[4:], {})
            if not src_nodes:
                res.trace.append(f"{st.output}: origem vazia, passo ignorado")
                step_facts.append([])
                continue
            rows = self.store.step(list(src_nodes), st.relations, st.direction, st.target_type)
            found = {r["next"]["id"]: r["next"] for r in rows}
            facts = []
            for r in rows:
                frm = r["dst"] if r["src"]["id"] == r["next"]["id"] else r["src"]
                facts.append({"src": r["src"], "rel": r["rel"], "dst": r["dst"], "props": r["props"],
                              "var": st.output, "from": frm["id"], "next": r["next"]["id"]})
            step_facts.append(facts)
            if st.output in vars_:  # interseção
                inter = {k: v for k, v in vars_[st.output].items() if k in found}
                res.trace.append(f"{st.output} ∩= {len(found)} -> {len(inter)}")
                vars_[st.output] = inter
            else:
                vars_[st.output] = found
            res.trace.append(
                f"{st.source} -[{'|'.join(st.relations)}]{'->' if st.direction == 'out' else '<-' if st.direction == 'in' else '-'} "
                f"{st.output}{':' + st.target_type if st.target_type else ''} => {[n['name'] for n in vars_[st.output].values()]}"
            )
        # poda para trás: mantém só valores/fatos que de fato levam à resposta
        ans = vars_.get(lf.answer_var or "", {})
        needed: dict[str, set[str]] = {lf.answer_var: set(ans)} if lf.answer_var else {}
        kept_facts: list[dict] = []
        for st, facts in reversed(list(zip(lf.steps, step_facts))):
            want = needed.get(st.output, set(vars_.get(st.output, {})))
            want &= set(vars_.get(st.output, {}))
            good = [f for f in facts if f["next"] in want]
            kept_facts = good + kept_facts
            if st.source.startswith("var:"):
                v = st.source[4:]
                needed[v] = needed.get(v, set()) | {f["from"] for f in good}
        res.facts = _dedupe_facts(kept_facts)
        res.bindings = {k: [n for nid, n in v.items() if nid in needed.get(k, set(v))] for k, v in vars_.items()}
        res.answers = list(ans.values())
        res.ok = bool(res.answers)
        return res

    def _run_path(self, lf: LogicalForm, linker: EntityLinker) -> KAGResult:
        res = KAGResult(ok=False)
        anchors = lf.anchors
        cand_paths: list[GraphPath] = []
        for i in range(len(anchors)):
            for j in range(i + 1, len(anchors)):
                ps = self.store.shortest_paths(anchors[i], anchors[j], 5)
                cand_paths += ps
                res.trace.append(f"allShortestPaths({anchors[i]}, {anchors[j]}) => {len(ps)} caminho(s)")
        if not cand_paths:
            return res

        def score(p: GraphPath) -> int:
            return sum(1 for r in p.rels if r["type"] in lf.cue_relations)

        best = max(score(p) for p in cand_paths)
        paths = [p for p in cand_paths if score(p) == best]
        res.trace.append(f"filtro por relações da pergunta {lf.cue_relations}: {len(paths)} caminho(s) mantido(s)")
        anchor_set = set(anchors)
        answers: dict[str, dict] = {}
        for p in paths:
            inner = [n for n in p.nodes if n["id"] not in anchor_set]
            typed = [n for n in inner if not lf.answer_type or n["type"] == lf.answer_type]
            for n in typed or ([] if lf.answer_type else inner):
                answers[n["id"]] = n
        if lf.answer_type and not answers:
            res.trace.append(f"nenhum nó intermediário do tipo {lf.answer_type}")
        res.paths = paths
        res.facts = _dedupe_facts([
            {"src": _node(p, r["start"]), "rel": r["type"], "dst": _node(p, r["end"]), "props": r["props"], "var": "path"}
            for p in paths for r in p.rels
        ])
        res.answers = list(answers.values())
        res.ok = bool(paths)
        return res


def _node(p: GraphPath, nid: str) -> dict:
    for n in p.nodes:
        if n["id"] == nid:
            return n
    return {"id": nid, "name": nid, "type": None}


def _dedupe_facts(facts: list[dict]) -> list[dict]:
    seen, out = set(), []
    for f in facts:
        k = (f["src"]["id"], f["rel"], f["dst"]["id"])
        if k not in seen:
            seen.add(k)
            out.append(f)
    return out
