"""Resolução / canonicalização de entidades (Entity Resolution).

Estratégia em cascata (da mais precisa para a mais permissiva), sempre
restrita ao mesmo tipo de entidade:

1. Alias exato normalizado (minúsculas, sem acentos/pontuação) — inclui o
   dicionário canônico e aliases aprendidos durante a ingestão.
2. Sobrenome de pessoa ("Turing" -> "Alan Turing") quando o sobrenome é único.
3. Contenção de tokens para nomes longos ("Universidade Princeton" ⊂ "Universidade de Princeton").
4. Similaridade difusa (razão de Levenshtein via ``difflib``) acima do limiar.
5. (Opcional) similaridade de embeddings do nome acima de 0.95.

Se nada casar, cria-se uma entidade canônica nova com ID ``<tipo>:<slug>``,
garantindo unicidade via constraint ``Entity.id IS UNIQUE`` no Neo4j.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from difflib import SequenceMatcher

from .text import STOPWORDS, normalize, slugify


@dataclass
class CanonicalEntity:
    id: str
    name: str
    type: str
    aliases: set[str] = field(default_factory=set)


@dataclass
class Resolution:
    entity: CanonicalEntity
    method: str  # exact | surname | containment | fuzzy | embedding | new
    score: float = 1.0


def _fuzzy_compatible(a: str, b: str) -> bool:
    """Evita fundir nomes que diferem em tokens curtos/números
    ("linguagem B" x "linguagem C", "Windows 95" x "Windows 98")."""
    ta, tb = a.split(), b.split()
    diff = set(ta) ^ set(tb)
    return not any(len(t) <= 2 or t.isdigit() for t in diff)


def make_id(etype: str, name: str) -> str:
    return f"{etype.lower()}:{slugify(name)}"


class EntityResolver:
    def __init__(self, threshold: float = 0.90, embedder=None):
        self.threshold = threshold
        self.embedder = embedder
        self.by_id: dict[str, CanonicalEntity] = {}
        self.alias_index: dict[tuple[str, str], str] = {}  # (type, norm) -> id
        self._emb_cache: dict[str, list[float]] = {}
        self.log: list[dict] = []

    # ------------------------------------------------------------ registro
    def register(self, name: str, etype: str, aliases=(), eid: str | None = None) -> CanonicalEntity:
        eid = eid or make_id(etype, name)
        ent = self.by_id.get(eid)
        if ent is None:
            ent = CanonicalEntity(eid, name, etype)
            self.by_id[eid] = ent
        for form in [name, *aliases]:
            self._add_alias(ent, form)
        return ent

    def _add_alias(self, ent: CanonicalEntity, form: str):
        n = normalize(form)
        if not n:
            return
        if form != ent.name:
            ent.aliases.add(form)
        self.alias_index.setdefault((ent.type, n), ent.id)

    def load(self, entries: list[dict]):
        for e in entries:
            self.register(e["name"], e["type"], e.get("aliases", []), e.get("id"))

    # ------------------------------------------------------------ resolução
    def resolve(self, name: str, etype: str) -> Resolution:
        n = normalize(name)
        # 1) alias exato
        eid = self.alias_index.get((etype, n))
        if eid:
            return self._done(name, Resolution(self.by_id[eid], "exact"))
        same_type = [e for e in self.by_id.values() if e.type == etype]
        toks = n.split()
        # 2) sobrenome de pessoa
        if etype == "Person" and len(toks) == 1:
            cands = [e for e in same_type if normalize(e.name).split()[-1] == toks[0]]
            if len(cands) == 1:
                return self._done(name, Resolution(cands[0], "surname", 0.95))
        # 3) contenção de tokens significativos (nomes com >= 2 tokens)
        sig = {t for t in toks if t not in STOPWORDS}
        if len(sig) >= 2:
            cands = []
            for e in same_type:
                esig = {t for t in normalize(e.name).split() if t not in STOPWORDS}
                if sig == esig:
                    cands.append(e)
            if len(cands) == 1:
                return self._done(name, Resolution(cands[0], "containment", 0.93))
        # 4) similaridade difusa
        best, best_s = None, 0.0
        for e in same_type:
            for form in [e.name, *e.aliases]:
                nf = normalize(form)
                if not _fuzzy_compatible(n, nf):
                    continue
                s = SequenceMatcher(None, n, nf).ratio()
                if s > best_s:
                    best, best_s = e, s
        if best is not None and best_s >= self.threshold:
            return self._done(name, Resolution(best, "fuzzy", best_s))
        # 5) embeddings do nome (opcional)
        if self.embedder is not None and same_type:
            from .embeddings import cosine

            q = self.embedder.embed_one(name)
            for e in same_type:
                v = self._emb_cache.get(e.id)
                if v is None:
                    v = self._emb_cache[e.id] = self.embedder.embed_one(e.name)
                s = cosine(q, v)
                if s > best_s:
                    best, best_s = e, s
            if best is not None and best_s >= 0.95:
                return self._done(name, Resolution(best, "embedding", best_s))
        # nova entidade
        ent = self.register(name, etype)
        return self._done(name, Resolution(ent, "new", 1.0))

    def _done(self, surface: str, r: Resolution) -> Resolution:
        self._add_alias(r.entity, surface)
        if r.method not in ("exact", "new"):
            self.log.append({"surface": surface, "canonical": r.entity.name, "method": r.method, "score": round(r.score, 3)})
        return r
