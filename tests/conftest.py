import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from graphrag.config import get_settings  # noqa: E402
from graphrag.engine import GraphRAGEngine  # noqa: E402


@pytest.fixture(scope="session")
def engine():
    """Sistema completo sobre o backend em memória, em modo offline determinístico."""
    eng = GraphRAGEngine(get_settings(graph_backend="memory", llm_provider="none",
                                      embedding_provider="hash", extractor="rules"))
    eng.build()
    return eng
