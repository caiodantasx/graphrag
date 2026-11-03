"""Entity linking da pergunta: liga menções do texto a nós do grafo e detecta
entidades citadas que NÃO existem no grafo (premissas não verificáveis)."""
from __future__ import annotations

import re
from dataclasses import dataclass

from .extraction import GazetteerMatcher
from .text import normalize

QUESTION_WORDS = {
    "quem", "qual", "quais", "onde", "quando", "como", "por", "que", "em", "o", "a", "os", "as",
    "de", "do", "da", "no", "na", "e", "existe", "existem", "há", "liste", "cite", "explique",
}
_CAP_TOKEN = re.compile(r"[A-ZÁÉÍÓÚÂÊÔÃÕÇ][\wÀ-ÿ'’\-]*")


@dataclass
class LinkedEntity:
    id: str
    name: str
    type: str
    surface: str
    start: int
    end: int


class EntityLinker:
    def __init__(self, entities: list[dict]):
        self.by_key: dict[tuple[str, str], str] = {}
        self.matcher = GazetteerMatcher([])
        for e in entities:
            self.matcher.add(e["name"], e["type"], e.get("aliases") or [])
            self.by_key[(e["name"], e["type"])] = e["id"]
        self.entities = {e["id"]: e for e in entities}

    def link(self, question: str) -> list[LinkedEntity]:
        out, seen = [], set()
        for m in self.matcher.find(question):
            eid = self.by_key.get((m.name, m.type))
            if eid and eid not in seen:
                seen.add(eid)
                out.append(LinkedEntity(eid, m.name, m.type, m.surface, m.start, m.end))
        return out

    def unlinked_mentions(self, question: str, linked: list[LinkedEntity]) -> list[str]:
        """Sequências de palavras capitalizadas que não foram ligadas ao grafo."""
        covered = [(l.start, l.end) for l in linked]
        spans: list[list[tuple[int, int, str]]] = []
        cur: list[tuple[int, int, str]] = []
        for m in _CAP_TOKEN.finditer(question):
            s, e, tok = m.start(), m.end(), m.group(0)
            inside = any(a <= s < b for a, b in covered)
            first_word = question[:s].strip() == ""
            generic = normalize(tok) in QUESTION_WORDS
            if inside or (first_word and generic) or generic:
                if cur:
                    spans.append(cur)
                    cur = []
                continue
            if cur and question[cur[-1][1] : s].strip() not in ("", "de", "da", "do", "dos", "das"):
                spans.append(cur)
                cur = []
            cur.append((s, e, tok))
        if cur:
            spans.append(cur)
        out = []
        for sp in spans:
            text = question[sp[0][0] : sp[-1][1]]
            if sp[0][0] == 0 and len(sp) == 1:
                continue  # 1ª palavra da frase capitalizada por gramática
            out.append(text)
        return out
