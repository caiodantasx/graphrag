"""Configuração centralizada, lida de variáveis de ambiente / arquivo .env."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

try:  # python-dotenv é opcional
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # pragma: no cover
    pass

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"


def _env(name: str, default: str | None = None) -> str | None:
    v = os.getenv(name)
    return v if v not in (None, "") else default


@dataclass
class Settings:
    # Backend de grafo: "neo4j" (padrão de produção) ou "memory" (testes/offline)
    graph_backend: str = field(default_factory=lambda: _env("GRAPH_BACKEND", "neo4j"))
    neo4j_uri: str = field(default_factory=lambda: _env("NEO4J_URI", "bolt://localhost:7687"))
    neo4j_user: str = field(default_factory=lambda: _env("NEO4J_USER", "neo4j"))
    neo4j_password: str = field(default_factory=lambda: _env("NEO4J_PASSWORD", "graphrag2026"))
    neo4j_database: str = field(default_factory=lambda: _env("NEO4J_DATABASE", "neo4j"))

    # LLM: "none" (modo offline determinístico) ou "openai" (qualquer API compatível
    # com OpenAI: OpenAI, Groq, OpenRouter, Ollama, LM Studio, Gemini-compat ...)
    llm_provider: str = field(default_factory=lambda: _env("LLM_PROVIDER", "none"))
    llm_base_url: str = field(default_factory=lambda: _env("LLM_BASE_URL", "https://api.openai.com/v1"))
    llm_api_key: str | None = field(default_factory=lambda: _env("LLM_API_KEY"))
    llm_model: str = field(default_factory=lambda: _env("LLM_MODEL", "gpt-4o-mini"))
    llm_temperature: float = field(default_factory=lambda: float(_env("LLM_TEMPERATURE", "0")))

    # Embeddings: "hash" (offline, determinístico), "openai" (API compatível) ou
    # "sentence-transformers" (local, requer o pacote instalado)
    embedding_provider: str = field(default_factory=lambda: _env("EMBEDDING_PROVIDER", "hash"))
    embedding_model: str = field(default_factory=lambda: _env("EMBEDDING_MODEL", "text-embedding-3-small"))
    embedding_dim: int = field(default_factory=lambda: int(_env("EMBEDDING_DIM", "384")))
    embedding_base_url: str | None = field(default_factory=lambda: _env("EMBEDDING_BASE_URL"))
    embedding_api_key: str | None = field(default_factory=lambda: _env("EMBEDDING_API_KEY"))

    # Pipeline
    schema_mode: str = field(default_factory=lambda: _env("SCHEMA_MODE", "strict"))
    extractor: str = field(default_factory=lambda: _env("EXTRACTOR", "auto"))  # auto|llm|rules
    chunk_max_chars: int = field(default_factory=lambda: int(_env("CHUNK_MAX_CHARS", "700")))
    resolution_threshold: float = field(default_factory=lambda: float(_env("RESOLUTION_THRESHOLD", "0.90")))
    corpus_dir: Path = field(default_factory=lambda: Path(_env("CORPUS_DIR", str(DATA_DIR / "corpus"))))
    gazetteer_path: Path = field(default_factory=lambda: Path(_env("GAZETTEER", str(DATA_DIR / "gazetteer.json"))))

    # Recuperação
    top_k: int = field(default_factory=lambda: int(_env("TOP_K", "5")))
    max_hops: int = field(default_factory=lambda: int(_env("MAX_HOPS", "2")))
    max_facts: int = field(default_factory=lambda: int(_env("MAX_FACTS", "30")))
    abstain_threshold: float = field(default_factory=lambda: float(_env("ABSTAIN_THRESHOLD", "0.35")))

    @property
    def llm_enabled(self) -> bool:
        return self.llm_provider.lower() not in ("none", "", "offline")


def get_settings(**overrides) -> Settings:
    s = Settings()
    for k, v in overrides.items():
        setattr(s, k, v)
    return s
