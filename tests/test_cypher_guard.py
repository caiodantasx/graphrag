"""Segurança: validação de Cypher gerado por LLM e interpolação de rótulos."""
import pytest

from graphrag.ontology import Ontology
from graphrag.store import cypher as Q
from graphrag.text2cypher import validate_cypher

O = Ontology()


def test_accepts_read_query_and_injects_limit():
    c = validate_cypher("MATCH (p:Person)-[:WORKED_AT]->(o:Organization) RETURN p.name, o.name", O)
    assert c.ok and c.cypher.strip().endswith("LIMIT 50")


@pytest.mark.parametrize("bad", [
    "MATCH (n) DETACH DELETE n RETURN 1",
    "MATCH (p:Person) SET p.name = 'x' RETURN p",
    "CREATE (n:Person {name:'x'}) RETURN n",
    "CALL dbms.security.listUsers() YIELD username RETURN username",
    "MATCH (p:Person) RETURN p; MATCH (n) DELETE n",
    "LOAD CSV FROM 'http://x' AS l RETURN l",
])
def test_rejects_write_or_admin(bad):
    assert not validate_cypher(bad, O).ok


def test_rejects_unknown_schema_elements():
    assert not validate_cypher("MATCH (p:Alien) RETURN p", O).ok
    assert not validate_cypher("MATCH (p:Person)-[:HATES]->(q) RETURN p", O).ok


def test_keywords_inside_strings_are_ignored():
    assert validate_cypher("MATCH (p:Person) WHERE p.name = 'Delete Set' RETURN p LIMIT 5", O).ok


def test_caps_large_limit():
    assert validate_cypher("MATCH (p:Person) RETURN p LIMIT 100000", O).cypher.endswith("LIMIT 50")


def test_label_interpolation_is_whitelisted():
    assert Q.safe_label("Person", O) == "Person"
    with pytest.raises(ValueError):
        Q.safe_label("Person) DETACH DELETE (n", O)
    with pytest.raises(ValueError):
        Q.safe_rel_types(["WORKED_AT", "X]->() DELETE"], O)


def test_schema_has_constraints_and_all_index_kinds():
    st = " ".join(Q.schema_statements(O, 384))
    assert "REQUIRE e.id IS UNIQUE" in st
    assert "CREATE INDEX entity_name" in st          # RANGE (sucessor do B-tree)
    assert "CREATE FULLTEXT INDEX entity_fulltext" in st
    assert "CREATE VECTOR INDEX chunk_embedding" in st and "`vector.dimensions`: 384" in st
