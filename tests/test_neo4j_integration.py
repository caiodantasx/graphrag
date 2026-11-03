"""Testes de integração com Neo4j real (executados só se NEO4J_TEST_URI estiver definido).

    docker compose up -d
    NEO4J_TEST_URI=bolt://localhost:7687 pytest tests/test_neo4j_integration.py -v
"""
import os

import pytest

URI = os.getenv("NEO4J_TEST_URI")
pytestmark = pytest.mark.skipif(not URI, reason="defina NEO4J_TEST_URI para rodar contra o Neo4j")


@pytest.fixture(scope="module")
def neo():
    from graphrag.config import get_settings
    from graphrag.engine import GraphRAGEngine

    eng = GraphRAGEngine(get_settings(graph_backend="neo4j", neo4j_uri=URI, llm_provider="none", extractor="rules"))
    eng.build(reset=True)
    yield eng
    eng.close()


def test_constraints_and_indexes_exist(neo):
    rows = neo.store.readonly_cypher("SHOW INDEXES YIELD name, type RETURN name, type")
    types = {r["type"] for r in rows}
    names = {r["name"] for r in rows}
    assert {"RANGE", "FULLTEXT", "VECTOR"} <= types
    assert {"entity_fulltext", "chunk_embedding", "entity_embedding"} <= names
    cons = neo.store.readonly_cypher("SHOW CONSTRAINTS YIELD name RETURN name")
    assert "entity_id" in {r["name"] for r in cons}


def test_uniqueness_constraint_enforced(neo):
    from neo4j.exceptions import ConstraintError

    with pytest.raises(ConstraintError):
        neo.store.run("CREATE (:Entity {id: 'person:alan_turing'})", write=True)


def test_same_answers_as_memory_backend(neo, engine):
    for q in ["Onde trabalhava o orientador de Alan Turing?", "Qual é a conexão entre Ada Lovelace e Claude Shannon?"]:
        a = neo.ask(q).answer
        b = engine.ask(q).answer
        assert set(a.answer_entity_ids) == set(b.answer_entity_ids)
