"""Provedores de embeddings.

* ``HashEmbedder``: offline e determinístico (feature hashing de palavras +
  trigramas de caracteres). Não captura sinonímia como um modelo neural, mas
  permite rodar todo o pipeline sem GPU/API e torna os testes reprodutíveis.
* ``OpenAIEmbedder``: qualquer endpoint compatível com ``/v1/embeddings``
  (OpenAI, Ollama com ``nomic-embed-text``, LM Studio, etc.).
* ``SentenceTransformerEmbedder``: modelo local (ex.: ``paraphrase-multilingual-MiniLM-L12-v2``, 384 dims).

A dimensão precisa bater com ``EMBEDDING_DIM`` usada nos índices vetoriais do Neo4j.
"""
from __future__ import annotations

import hashlib
import math

import numpy as np

from .config import Settings
from .text import content_tokens


class Embedder:
    dim: int

    def embed(self, texts: list[str]) -> list[list[float]]:  # pragma: no cover - interface
        raise NotImplementedError

    def embed_one(self, text: str) -> list[float]:
        return self.embed([text])[0]


class HashEmbedder(Embedder):
    def __init__(self, dim: int = 384):
        self.dim = dim

    def _bucket(self, feat: str) -> tuple[int, float]:
        h = hashlib.blake2b(feat.encode("utf-8"), digest_size=8).digest()
        n = int.from_bytes(h, "little")
        return n % self.dim, (1.0 if (n >> 63) & 1 else -1.0)

    def _vec(self, text: str) -> list[float]:
        v = np.zeros(self.dim, dtype=np.float32)
        toks = content_tokens(text)
        for t in toks:
            i, s = self._bucket("w:" + t)
            v[i] += 2.0 * s
            padded = f"#{t}#"
            for j in range(len(padded) - 2):
                i, s = self._bucket("c:" + padded[j : j + 3])
                v[i] += 0.5 * s
        for a, b in zip(toks, toks[1:]):
            i, s = self._bucket(f"b:{a}_{b}")
            v[i] += 1.0 * s
        norm = float(np.linalg.norm(v))
        if norm > 0:
            v /= norm
        return v.tolist()

    def embed(self, texts):
        return [self._vec(t) for t in texts]


class OpenAIEmbedder(Embedder):
    def __init__(self, settings: Settings):
        import requests  # noqa: F401  (dependência declarada em requirements)

        self.s = settings
        self.dim = settings.embedding_dim
        self.base = (settings.embedding_base_url or settings.llm_base_url).rstrip("/")
        self.key = settings.embedding_api_key or settings.llm_api_key

    def embed(self, texts):
        import requests

        headers = {"Content-Type": "application/json"}
        if self.key:
            headers["Authorization"] = f"Bearer {self.key}"
        out: list[list[float]] = []
        for i in range(0, len(texts), 64):
            batch = texts[i : i + 64]
            r = requests.post(
                f"{self.base}/embeddings",
                json={"model": self.s.embedding_model, "input": batch},
                headers=headers,
                timeout=60,
            )
            r.raise_for_status()
            data = sorted(r.json()["data"], key=lambda d: d["index"])
            for d in data:
                vec = d["embedding"]
                if len(vec) != self.dim:
                    raise ValueError(
                        f"Embedding com {len(vec)} dimensões, mas EMBEDDING_DIM={self.dim}. "
                        "Ajuste EMBEDDING_DIM e recrie os índices vetoriais."
                    )
                out.append(vec)
        return out


class SentenceTransformerEmbedder(Embedder):
    def __init__(self, settings: Settings):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(settings.embedding_model)
        self.dim = self.model.get_sentence_embedding_dimension()

    def embed(self, texts):
        return self.model.encode(texts, normalize_embeddings=True).tolist()


def get_embedder(settings: Settings) -> Embedder:
    p = settings.embedding_provider.lower()
    if p == "openai":
        return OpenAIEmbedder(settings)
    if p in ("sentence-transformers", "st", "local"):
        return SentenceTransformerEmbedder(settings)
    return HashEmbedder(settings.embedding_dim)


def cosine(a, b) -> float:
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na == 0 or nb == 0 or math.isnan(na) or math.isnan(nb):
        return 0.0
    return float(a @ b / (na * nb))
