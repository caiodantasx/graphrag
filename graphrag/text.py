"""Utilitários de texto: normalização, leitura de documentos, sentenças e chunking."""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

STOPWORDS = {
    "a", "o", "as", "os", "de", "da", "do", "das", "dos", "e", "em", "no", "na",
    "nos", "nas", "um", "uma", "the", "of", "and", "que", "qual", "quais", "quem",
    "onde", "como", "por", "para", "com", "se", "foi", "é", "ao", "aos", "à", "às",
}

YEAR_RE = re.compile(r"\b(1[6-9]\d\d|20\d\d)\b")


def strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))


def normalize(s: str) -> str:
    """Forma canônica para comparação: minúsculas, sem acento, sem pontuação."""
    s = strip_accents(s).lower()
    s = s.replace("'", "").replace("’", "")
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def slugify(s: str) -> str:
    return normalize(s).replace(" ", "_")


def tokens(s: str) -> list[str]:
    return normalize(s).split()


def content_tokens(s: str) -> list[str]:
    return [t for t in tokens(s) if t not in STOPWORDS]


# ---------------------------------------------------------------- sentenças
_ABBREV = ("D.C.", "Dr.", "Sr.", "Sra.", "EUA.", "St.", "Jr.")
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-ZÁÉÍÓÚÂÊÔÃÕÇ0-9\"“(])")


def split_sentences(text: str) -> list[str]:
    protected = text
    for i, ab in enumerate(_ABBREV):
        protected = protected.replace(ab, f"§{i}§")
    parts = _SENT_SPLIT.split(protected)
    out = []
    for p in parts:
        for i, ab in enumerate(_ABBREV):
            p = p.replace(f"§{i}§", ab)
        p = p.strip()
        if p:
            out.append(p)
    return out


# ---------------------------------------------------------------- documentos
@dataclass
class SourceDocument:
    id: str
    title: str
    text: str
    source: str
    meta: dict = field(default_factory=dict)

    @property
    def subject(self) -> str | None:
        return self.meta.get("subject")


def parse_document(path: Path) -> SourceDocument:
    raw = path.read_text(encoding="utf-8")
    meta: dict[str, str] = {}
    body = raw
    if raw.startswith("---"):
        _, header, body = raw.split("---", 2)
        for line in header.strip().splitlines():
            if ":" in line:
                k, v = line.split(":", 1)
                meta[k.strip()] = v.strip()
    title = meta.get("title") or path.stem.replace("_", " ").title()
    return SourceDocument(
        id=f"doc:{slugify(path.stem)}",
        title=title,
        text=body.strip(),
        source=meta.get("source", str(path.name)),
        meta=meta,
    )


def load_corpus(directory: Path) -> list[SourceDocument]:
    files = sorted(list(directory.glob("*.md")) + list(directory.glob("*.txt")))
    return [parse_document(p) for p in files]


# ---------------------------------------------------------------- chunking
@dataclass
class Chunk:
    id: str
    doc_id: str
    idx: int
    text: str


def chunk_document(doc: SourceDocument, max_chars: int = 700, overlap_sentences: int = 0) -> list[Chunk]:
    """Chunking orientado a parágrafos com limite de caracteres e corte em sentenças."""
    chunks: list[Chunk] = []
    buf: list[str] = []
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", doc.text) if p.strip()]

    def flush():
        if buf:
            idx = len(chunks)
            chunks.append(Chunk(f"{doc.id}#c{idx}", doc.id, idx, " ".join(buf)))

    for para in paragraphs:
        for sent in split_sentences(para.replace("\n", " ")):
            if buf and len(" ".join(buf)) + len(sent) + 1 > max_chars:
                flush()
                buf = buf[-overlap_sentences:] if overlap_sentences else []
            buf.append(sent)
        # fim de parágrafo: fecha chunk se já estiver razoavelmente cheio
        if len(" ".join(buf)) > max_chars * 0.5:
            flush()
            buf = []
    flush()
    return chunks
