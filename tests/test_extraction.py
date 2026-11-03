"""Pipeline Text-to-Graph: NER por dicionário e RE por regras."""
from pathlib import Path

import pytest

from graphrag.extraction import GazetteerMatcher, LLMExtractor, RuleExtractor
from graphrag.ontology import Ontology
from graphrag.text import chunk_document, split_sentences, SourceDocument

GAZ = Path(__file__).resolve().parent.parent / "data" / "gazetteer.json"


@pytest.fixture(scope="module")
def ex():
    return RuleExtractor(GazetteerMatcher.from_file(GAZ), Ontology())


def triples(ex, text, subject=None):
    return {(t.src, t.rel, t.dst) for t in ex.extract(text, subject).triples}


def test_longest_match_prefers_full_names():
    m = GazetteerMatcher.from_file(GAZ)
    names = [x.name for x in m.find("O Prêmio Turing homenageia Turing e a Máquina de Turing.")]
    assert names == ["Prêmio Turing", "Alan Turing", "Máquina de Turing"]


def test_disambiguates_two_cambridges():
    m = GazetteerMatcher.from_file(GAZ)
    names = [x.name for x in m.find("O MIT fica em Cambridge, Massachusetts; Babbage estudou em Cambridge.")]
    assert "Cambridge, Massachusetts" in names and "Cambridge" in names


def test_passive_advisor_direction(ex):
    t = triples(ex, "Turing fez o doutorado na Universidade de Princeton sob orientação de Alonzo Church.")
    assert ("Alonzo Church", "ADVISED", "Alan Turing") in t
    assert ("Alan Turing", "STUDIED_AT", "Universidade de Princeton") in t


def test_active_advisor_coordination(ex):
    t = triples(ex, "Church orientou Alan Turing e Stephen Kleene em seus doutorados.")
    assert {("Alonzo Church", "ADVISED", "Alan Turing"), ("Alonzo Church", "ADVISED", "Stephen Kleene")} <= t


def test_participle_attaches_to_intermediate_entity(ex):
    t = triples(ex, "A linguagem C foi influenciada pela linguagem B, criada por Ken Thompson.")
    assert ("linguagem B", "INFLUENCED", "linguagem C") in t
    assert ("Ken Thompson", "CREATED", "linguagem B") in t
    assert ("Ken Thompson", "CREATED", "linguagem C") not in t  # erro clássico evitado


def test_prepositional_adjunct_not_linked(ex):
    t = triples(ex, "Em 1994, Berners-Lee fundou o World Wide Web Consortium (W3C) no MIT.")
    assert ("Tim Berners-Lee", "FOUNDED", "World Wide Web Consortium") in t
    assert ("Tim Berners-Lee", "FOUNDED", "MIT") not in t


def test_nearest_cue_and_type_constraints(ex):
    t = triples(ex, "Alan Turing nasceu em Londres em 1912 e estudou no King's College, em Cambridge.")
    assert ("Alan Turing", "BORN_IN", "Londres") in t
    assert ("Alan Turing", "STUDIED_AT", "King's College") in t
    assert ("King's College", "LOCATED_IN", "Cambridge") in t
    assert ("Alan Turing", "BORN_IN", "Cambridge") not in t


def test_year_is_extracted(ex):
    res = ex.extract("Em 1983, Ritchie recebeu o Prêmio Turing.")
    (t,) = [t for t in res.triples if t.rel == "RECEIVED"]
    assert t.year == 1983


def test_pronoun_coreference(ex):
    t = triples(ex, "Ele trabalhou nos Bell Labs.", subject="Claude Shannon")
    assert ("Claude Shannon", "WORKED_AT", "Bell Labs") in t


def test_llm_output_is_validated_against_schema():
    """Saída do LLM com tipo inexistente, direção invertida e entidade sem tipo."""
    ex = LLMExtractor(llm=None, ontology=Ontology(mode="strict"))
    data = {
        "entities": [{"name": "Alan Turing", "type": "Person"}, {"name": "Londres", "type": "Location"},
                     {"name": "Bletchley Park", "type": "Organization"}],
        "relations": [
            {"source": "Londres", "type": "BORN_IN", "target": "Alan Turing"},          # invertida -> corrige
            {"source": "Alan Turing", "type": "LIKED", "target": "Londres"},            # tipo fora do esquema
            {"source": "Alan Turing", "type": "WORKED_AT", "target": "Enigma"},         # entidade não declarada
            {"source": "Alan Turing", "type": "worked at", "target": "Bletchley Park", "year": "1939"},
        ],
    }
    res = ex.validate(data)
    got = {(t.src, t.rel, t.dst, t.year) for t in res.triples}
    assert got == {("Alan Turing", "BORN_IN", "Londres", None), ("Alan Turing", "WORKED_AT", "Bletchley Park", 1939)}


def test_dynamic_schema_registers_new_relation():
    onto = Ontology(mode="dynamic")
    ex = LLMExtractor(llm=None, ontology=onto)
    res = ex.validate({"entities": [{"name": "A", "type": "Person"}, {"name": "B", "type": "Person"}],
                       "relations": [{"source": "A", "type": "rivalizou com", "target": "B"}]})
    assert res.triples[0].rel == "RIVALIZOU_COM" and "RIVALIZOU_COM" in onto.relations


def test_sentence_split_keeps_abbreviations():
    assert len(split_sentences("Church nasceu em Washington, D.C., em 1903. Ele estudou.")) == 2


def test_chunking_respects_limit():
    doc = SourceDocument("doc:x", "X", " ".join(["Frase número %d termina aqui." % i for i in range(60)]), "x")
    chunks = chunk_document(doc, max_chars=200)
    assert len(chunks) > 3 and all(len(c.text) <= 230 for c in chunks)
