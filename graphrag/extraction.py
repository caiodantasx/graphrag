"""Pipeline Text-to-Graph: NER (Named Entity Recognition) e RE (Relation Extraction).

Dois extratores com a mesma interface:

* ``LLMExtractor`` — extração guiada por esquema: o LLM recebe a ontologia e
  devolve JSON {entities, relations}; a saída é validada (tipos, domínio/alcance,
  direção) antes de entrar no grafo. Em ``SCHEMA_MODE=dynamic`` tipos de
  relação novos são aceitos e registrados na ontologia.
* ``RuleExtractor`` — baseline determinístico (offline): NER por dicionário
  (casamento do maior trecho, ``GazetteerMatcher``) + RE por padrões lexicais
  em português com restrições de tipo da ontologia.

``build_extractor`` escolhe automaticamente (LLM com fallback para regras).
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from .llm import LLMClient, LLMError
from .ontology import Ontology
from .text import YEAR_RE, normalize, split_sentences, strip_accents


@dataclass
class ExtractedEntity:
    name: str
    type: str
    aliases: list[str] = field(default_factory=list)


@dataclass
class Triple:
    src: str
    src_type: str
    rel: str
    dst: str
    dst_type: str
    year: int | None = None
    evidence: str = ""
    confidence: float = 0.7


@dataclass
class ExtractionResult:
    entities: list[ExtractedEntity] = field(default_factory=list)
    triples: list[Triple] = field(default_factory=list)
    extractor: str = "rules"


# =====================================================================
# NER por dicionário
# =====================================================================
_TOKEN_RE = re.compile(r"[0-9A-Za-zÀ-ÖØ-öø-ÿ'’]+")


@dataclass
class Mention:
    name: str  # nome canônico
    type: str
    surface: str
    start: int
    end: int


class GazetteerMatcher:
    """Casamento do maior trecho (longest match) sobre tokens normalizados."""

    def __init__(self, entries: list[dict]):
        self.index: dict[tuple[str, ...], tuple[str, str]] = {}
        self.max_len = 1
        for e in entries:
            self.add(e["name"], e["type"], e.get("aliases", []))

    def add(self, name: str, etype: str, aliases: list[str] | None = None):
        for form in [name, *(aliases or [])]:
            key = tuple(normalize(form).split())
            if key and key not in self.index:
                self.index[key] = (name, etype)
                self.max_len = max(self.max_len, len(key))

    @classmethod
    def from_file(cls, path: Path) -> "GazetteerMatcher":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(data["entities"])

    def find(self, text: str) -> list[Mention]:
        toks = [(m.group(0), m.start(), m.end()) for m in _TOKEN_RE.finditer(text)]
        norm = [normalize(t[0]) for t in toks]
        # tokens como "D.C." viram 2 tokens; normalize pode devolver "" (ex.: "’")
        out: list[Mention] = []
        i = 0
        while i < len(toks):
            hit = None
            for L in range(min(self.max_len, len(toks) - i), 0, -1):
                key = tuple(w for n in norm[i : i + L] for w in n.split())
                if key in self.index:
                    hit = (L, self.index[key])
                    break
            if hit:
                L, (name, etype) = hit
                s, e = toks[i][1], toks[i + L - 1][2]
                out.append(Mention(name, etype, text[s:e], s, e))
                i += L
            else:
                i += 1
        return out


# =====================================================================
# RE por regras
# =====================================================================
P, O, L, T, C, A = "Person", "Organization", "Location", "Technology", "Concept", "Award"
TC = (T, C)


@dataclass
class Rule:
    rel: str
    pattern: re.Pattern
    first: tuple[str, ...]  # tipos aceitos para a 1ª menção (ordem no texto)
    second: tuple[str, ...]
    reverse: bool = False  # True => relação vai da 2ª para a 1ª menção


def _r(p: str) -> re.Pattern:
    return re.compile(p)


# Padrões aplicados ao texto entre as duas menções (minúsculo, sem acento).
CUE_RULES: list[Rule] = [
    Rule("BORN_IN", _r(r"\bnasceu\b"), (P,), (L,)),
    Rule("STUDIED_AT", _r(r"\b(estudou|estudaram|graduou|formou|doutorado|mestrado)\b"), (P,), (O,)),
    Rule("WORKED_AT", _r(r"\b(trabalhou|trabalhar|trabalhava|lecionou|professora?|pesquisadora?)\b"), (P,), (O,)),
    Rule("ADVISED", _r(r"\b(orientou|orientador de|orientadora de)\b"), (P,), (P,)),
    Rule("ADVISED", _r(r"\b(sob orientacao de|orientad[oa]s? por|orientand[oa]s? de)\b"), (P,), (P,), reverse=True),
    Rule("CREATED", _r(r"\b(criou|criaram|desenvolveu|desenvolveram|inventou|propos|projetou|formulou|formularam|descreveu|cunhou|autora? d[aeo])\b"), (P,), TC),
    Rule("CREATED", _r(r"\b(criad[oa]s?|desenvolvid[oa]s?|inventad[oa]s?|propost[oa]s?|projetad[oa]s?|formulad[oa]s?) por\b"), TC, (P,), reverse=True),
    Rule("DEVELOPED_AT", _r(r"\bonde (criou|criaram|desenvolveu|desenvolveram|formulou|propos|inventou|projetou)\b"), (O,), TC, reverse=True),
    Rule("COLLABORATED_WITH", _r(r"\b(colaborou com|colaboracao com|junto com|em parceria com)\b"), (P,), (P,)),
    Rule("MET", _r(r"\b(encontrou com|se encontrou com|conheceu)\b"), (P,), (P,)),
    Rule("INFLUENCED", _r(r"\b(influenciou|influenciaram|inspirou)\b"), (P, T, C), (P, T, C)),
    Rule("INFLUENCED", _r(r"\b(influenciad[oa]s? (por|pel[oa]s?)|inspirad[oa]s? (por|em|n[oa]))\b"), (P, T, C), (P, T, C), reverse=True),
    Rule("RECEIVED", _r(r"\b(recebeu|ganhou|venceu|receberam)\b"), (P,), (A,)),
    Rule("NAMED_AFTER", _r(r"\b(em homenagem a|em homenagem ao|em homenagem a)\b"), (A, T, C), (P,)),
    Rule("AWARDED_BY", _r(r"\bconcedid[oa] (anualmente )?(pel[oa]|por)\b"), (A,), (O,)),
    Rule("FOUNDED", _r(r"\b(fundou|cofundou|fundaram)\b"), (P,), (O,)),
    Rule("LOCATED_IN", _r(r"\b(fica|ficam|localizad[oa]s?|sediad[oa]s?)\b"), (O, L), (L,)),
    Rule("PART_OF", _r(r"\b(faz parte|fazem parte|pertence|pertencem)\b"), (O,), (O,)),
]

# Padrões que precisam casar o trecho INTEIRO entre as menções (lacunas curtas).
GAP_RULES: list[Rule] = [
    Rule("LOCATED_IN", _r(r"^\s*,?\s*(em|na|no|nos|nas)?\s*,?\s*$"), (O, L), (L,)),
    Rule("DEVELOPED_AT", _r(r"^\s*(foi |foram )?(desenvolvid[oa]s?|criad[oa]s?|inventad[oa]s?|construid[oa]s?)?\s*(no|na|nos|nas|em)\s*$"), TC, (O,)),
]

# Trecho que começa com particípio após entidade intermediária modifica a
# intermediária, não a 1ª menção (ex.: "C foi influenciada pela B, criada por X").
_PARTICIPLE_START = re.compile(
    r"^\s*,?\s*(que\s+)?(foi\s+)?(criad|desenvolvid|propost|inventad|projetad|formulad|nomead|influenciad|localizad|sediad|concedid)"
)
# Trecho só com preposição após entidade intermediária ("W3C) no MIT") é um
# adjunto da intermediária -> não liga a 1ª menção à 2ª.
_PREP_ONLY = re.compile(r"[\s,()]*(em|no|na|nos|nas)?[\s,()]*")
_PRONOUN_START = re.compile(r"^(Ele|Ela)\b")


class RuleExtractor:
    name = "rules"

    def __init__(self, matcher: GazetteerMatcher, ontology: Ontology):
        self.matcher = matcher
        self.ontology = ontology

    def extract(self, text: str, subject: str | None = None) -> ExtractionResult:
        res = ExtractionResult(extractor=self.name)
        seen_entities: dict[tuple[str, str], ExtractedEntity] = {}
        subject_mention = None
        if subject:
            sm = self.matcher.find(subject)
            subject_mention = sm[0] if sm else None

        for sent in split_sentences(text):
            mentions = self.matcher.find(sent)
            # correferência simples: pronome sujeito -> entidade principal do documento
            pm = _PRONOUN_START.match(sent)
            if pm and subject_mention:
                mentions.insert(0, Mention(subject_mention.name, subject_mention.type, pm.group(0), 0, pm.end()))
            for m in mentions:
                key = (m.name, m.type)
                ent = seen_entities.setdefault(key, ExtractedEntity(m.name, m.type))
                if normalize(m.surface) != normalize(m.name) and m.surface not in ent.aliases:
                    if not _PRONOUN_START.match(m.surface):
                        ent.aliases.append(m.surface)
            res.triples.extend(self._relations(sent, mentions))
        res.entities = list(seen_entities.values())
        # deduplicação de triplas
        uniq: dict[tuple, Triple] = {}
        for t in res.triples:
            uniq.setdefault((t.src, t.rel, t.dst), t)
        res.triples = list(uniq.values())
        return res

    def _relations(self, sent: str, mentions: list[Mention]) -> list[Triple]:
        out: list[Triple] = []
        low = strip_accents(sent).lower()
        sentence_years = YEAR_RE.findall(sent)
        for i in range(len(mentions)):
            for j in range(i + 1, len(mentions)):
                a, b = mentions[i], mentions[j]
                if a.name == b.name:
                    continue
                between = low[a.end : b.start]
                inter = [m for m in mentions[i + 1 : j]]
                if inter:
                    last_seg = low[inter[-1].end : b.start]
                    if _PARTICIPLE_START.match(last_seg) or _PREP_ONLY.fullmatch(last_seg):
                        continue
                rule = self._match(between, a.type, b.type, gap_only=False)
                if not rule:
                    continue
                src, dst = (b, a) if rule.reverse else (a, b)
                # ano: no trecho entre/após as menções; senão, único ano da sentença
                nxt = mentions[j + 1].start if j + 1 < len(mentions) else len(sent)
                ym = YEAR_RE.search(sent[a.end : nxt])
                year = int(ym.group(0)) if ym else (int(sentence_years[0]) if len(sentence_years) == 1 else None)
                ok, swap = self.ontology.check_triple(src.type, rule.rel, dst.type)
                if not ok:
                    continue
                if swap:
                    src, dst = dst, src
                out.append(Triple(src.name, src.type, rule.rel, dst.name, dst.type, year, sent, 0.7))
        return out

    @staticmethod
    def _match(between: str, t1: str, t2: str, gap_only: bool) -> Rule | None:
        if len(between) <= 40:
            for r in GAP_RULES:
                if t1 in r.first and t2 in r.second and r.pattern.fullmatch(between):
                    return r
        if gap_only:
            return None
        hits = []
        for r in CUE_RULES:
            for m in r.pattern.finditer(between):
                hits.append((m.end(), r))
        if not hits:
            return None
        last_end = max(h[0] for h in hits)
        # agrupa pistas que terminam na mesma região (ex.: "onde criou" e "criou")
        cluster = [r for end, r in hits if end >= last_end - 3]
        for r in cluster:
            if t1 in r.first and t2 in r.second:
                return r
        return None


# =====================================================================
# Extração por LLM guiada por esquema
# =====================================================================
EXTRACTION_SYSTEM = """Você é um sistema de extração de informação (NER + RE) para construir um grafo de conhecimento.
Extraia SOMENTE fatos explicitamente afirmados no texto. Não use conhecimento externo.
Use exclusivamente os tipos de entidade e de relação do esquema abaixo{dynamic}.
Use o nome mais completo da entidade como "name" e coloque variações (sobrenome, sigla) em "aliases".
Respeite a direção (origem -> destino) definida no esquema.

{schema}

Responda apenas com JSON no formato:
{{"entities": [{{"name": str, "type": str, "aliases": [str]}}],
  "relations": [{{"source": str, "type": str, "target": str, "year": int|null, "evidence": str}}]}}"""

EXTRACTION_EXAMPLE = """Exemplo:
Texto: "Grace Hopper trabalhou na Remington Rand, onde desenvolveu o compilador A-0 em 1952."
JSON: {"entities": [{"name": "Grace Hopper", "type": "Person", "aliases": []},
 {"name": "Remington Rand", "type": "Organization", "aliases": []},
 {"name": "compilador A-0", "type": "Technology", "aliases": ["A-0"]}],
 "relations": [{"source": "Grace Hopper", "type": "WORKED_AT", "target": "Remington Rand", "year": null, "evidence": "Hopper trabalhou na Remington Rand"},
 {"source": "Grace Hopper", "type": "CREATED", "target": "compilador A-0", "year": 1952, "evidence": "desenvolveu o compilador A-0 em 1952"},
 {"source": "compilador A-0", "type": "DEVELOPED_AT", "target": "Remington Rand", "year": 1952, "evidence": "onde desenvolveu o compilador A-0"}]}"""


class LLMExtractor:
    name = "llm"

    def __init__(self, llm: LLMClient, ontology: Ontology):
        self.llm = llm
        self.ontology = ontology

    def extract(self, text: str, subject: str | None = None) -> ExtractionResult:
        dynamic = (
            " (se um fato importante não couber, você pode propor um novo tipo de relação em MAIÚSCULAS_COM_UNDERSCORE)"
            if self.ontology.mode == "dynamic" else ""
        )
        system = EXTRACTION_SYSTEM.format(schema=self.ontology.describe(), dynamic=dynamic)
        user = f"{EXTRACTION_EXAMPLE}\n\nEntidade principal do documento: {subject or 'desconhecida'}\nTexto:\n\"\"\"{text}\"\"\""
        data = self.llm.chat_json(system, user)
        return self.validate(data)

    def validate(self, data: dict) -> ExtractionResult:
        res = ExtractionResult(extractor=self.name)
        types: dict[str, str] = {}
        for e in data.get("entities", []):
            name = str(e.get("name", "")).strip()
            etype = str(e.get("type", "")).strip()
            if not name or etype not in self.ontology.entity_types:
                continue
            aliases = [str(a).strip() for a in e.get("aliases") or [] if str(a).strip()]
            res.entities.append(ExtractedEntity(name, etype, aliases))
            types[normalize(name)] = etype
            for a in aliases:
                types.setdefault(normalize(a), etype)
        for r in data.get("relations", []):
            src, dst = str(r.get("source", "")).strip(), str(r.get("target", "")).strip()
            rel = self.ontology.sanitize_relation(str(r.get("type", "")))
            st, dt = types.get(normalize(src)), types.get(normalize(dst))
            if not (src and dst and st and dt) or normalize(src) == normalize(dst):
                continue
            if rel not in self.ontology.relations:
                if self.ontology.mode != "dynamic" or not self.ontology.register_relation(rel, st, dt):
                    continue
            ok, swap = self.ontology.check_triple(st, rel, dt)
            if not ok:
                continue
            if swap:
                src, dst, st, dt = dst, src, dt, st
            year = r.get("year")
            try:
                year = int(year) if year not in (None, "", "null") else None
            except (TypeError, ValueError):
                year = None
            res.triples.append(Triple(src, st, rel, dst, dt, year, str(r.get("evidence", ""))[:300], 0.85))
        return res


class HybridExtractor:
    """LLM quando disponível; regras como fallback e como NER complementar."""

    def __init__(self, llm_ex: LLMExtractor | None, rule_ex: RuleExtractor):
        self.llm_ex = llm_ex
        self.rule_ex = rule_ex
        self.name = "llm+rules" if llm_ex else "rules"

    def extract(self, text: str, subject: str | None = None) -> ExtractionResult:
        rules = self.rule_ex.extract(text, subject)
        if not self.llm_ex:
            return rules
        try:
            res = self.llm_ex.extract(text, subject)
        except (LLMError, ValueError, KeyError) as e:  # pragma: no cover - rede
            rules.extractor = f"rules (fallback: {e.__class__.__name__})"
            return rules
        # NER complementar: menções do dicionário que o LLM não listou
        known = {normalize(e.name) for e in res.entities}
        for e in rules.entities:
            if normalize(e.name) not in known:
                res.entities.append(e)
        res.extractor = self.name
        return res


def build_extractor(settings, ontology: Ontology, llm: LLMClient, matcher: GazetteerMatcher):
    rule = RuleExtractor(matcher, ontology)
    mode = settings.extractor.lower()
    if mode == "rules" or (mode == "auto" and not llm.enabled):
        return HybridExtractor(None, rule)
    return HybridExtractor(LLMExtractor(llm, ontology), rule)
