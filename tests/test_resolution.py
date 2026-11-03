"""Resolução de entidades (canonicalização)."""
from graphrag.resolution import EntityResolver


def make():
    r = EntityResolver(threshold=0.9)
    r.load([
        {"name": "Alan Turing", "type": "Person", "aliases": ["Turing"]},
        {"name": "Ada Lovelace", "type": "Person", "aliases": []},
        {"name": "Universidade de Princeton", "type": "Organization", "aliases": []},
        {"name": "Princeton", "type": "Location", "aliases": []},
        {"name": "linguagem B", "type": "Technology", "aliases": []},
        {"name": "Roterdã", "type": "Location", "aliases": []},
    ])
    return r


def test_exact_alias_and_accents():
    r = make()
    assert r.resolve("TURING", "Person").entity.name == "Alan Turing"
    assert r.resolve("Roterda", "Location").entity.name == "Roterdã"


def test_surname_rule():
    r = make()
    res = r.resolve("Lovelace", "Person")
    assert res.entity.name == "Ada Lovelace" and res.method == "surname"


def test_type_scoped_resolution():
    r = make()
    assert r.resolve("Princeton", "Location").entity.id == "location:princeton"
    assert r.resolve("Princeton University", "Organization").method == "new"  # sem alias, outro idioma


def test_containment_rule():
    r = make()
    res = r.resolve("Universidade Princeton", "Organization")
    assert res.entity.name == "Universidade de Princeton" and res.method == "containment"


def test_fuzzy_typo():
    r = make()
    res = r.resolve("Alan Turnig", "Person")
    assert res.entity.name == "Alan Turing" and res.method == "fuzzy"


def test_fuzzy_guard_short_tokens():
    r = make()
    res = r.resolve("linguagem C", "Technology")
    assert res.method == "new" and res.entity.name == "linguagem C"


def test_new_alias_is_learned():
    r = make()
    r.resolve("Alan Turnig", "Person")
    assert r.resolve("alan turnig", "Person").method == "exact"
