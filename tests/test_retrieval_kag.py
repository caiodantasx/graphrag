"""Recuperação híbrida, multi-hop e raciocínio KAG."""
import pytest


@pytest.mark.parametrize("q,expected", [
    ("Quem orientou o doutorado de Alan Turing?", ["Alonzo Church"]),
    ("Onde trabalhava o orientador de Alan Turing?", ["Universidade de Princeton"]),
    ("Em que cidade nasceu o criador da linguagem C?", ["Bronxville"]),
    ("Em que país fica a cidade onde nasceu John von Neumann?", ["Hungria"]),
    ("Quais pessoas que trabalharam nos Bell Labs receberam o Prêmio Turing?", ["Dennis Ritchie", "Ken Thompson"]),
    ("Qual é a conexão entre Ada Lovelace e Claude Shannon?", ["Alan Turing"]),
])
def test_kag_answers(engine, q, expected):
    r = engine.ask(q, mode="kag")
    assert not r.answer.abstained
    for e in expected:
        assert e in r.answer.text


def test_multihop_plan_has_two_steps(engine):
    r = engine.ask("Onde trabalhava o orientador de Alan Turing?", mode="kag")
    steps = r.retrieval.plan.steps
    assert [s.relations for s in steps] == [["ADVISED"], ["WORKED_AT"]]
    assert steps[0].direction == "in" and steps[1].direction == "out"


def test_backward_pruning_removes_dead_ends(engine):
    r = engine.ask("Em que cidade nasceu o criador da linguagem C?", mode="kag")
    names = {f["dst"]["name"] for f in r.retrieval.kag.facts} | {f["src"]["name"] for f in r.retrieval.kag.facts}
    assert "Bell Labs" not in names  # DEVELOPED_AT não leva à resposta


def test_hybrid_seeds_fuse_sources(engine):
    r = engine.ask("Quem criou o sistema TeX?", mode="graphrag")
    top = r.retrieval.seeds[0]
    assert top["name"] == "TeX" and "linking" in top["via"]
    assert any("vector" in s["via"] for s in r.retrieval.seeds)


def test_vector_mode_has_no_graph_context(engine):
    r = engine.ask("Onde nasceu Alan Turing?", mode="vector")
    assert r.retrieval.facts == [] and r.retrieval.chunks


def test_context_facts_are_numbered_and_sourced(engine):
    r = engine.ask("Onde trabalhava o orientador de Alan Turing?", mode="kag")
    ids = [f.id for f in r.retrieval.facts]
    assert ids[:3] == ["F1", "F2", "F3"]
    assert all(f.sources for f in r.retrieval.facts)
    assert r.retrieval.provenance  # chunks de origem recuperados


def test_every_query_is_logged(engine):
    r = engine.ask("Onde trabalhava o orientador de Alan Turing?", mode="kag")
    cyphers = " ".join(q.cypher for q in r.retrieval.queries)
    assert "db.index.vector.queryNodes" in cyphers
    assert "db.index.fulltext.queryNodes" in cyphers
    assert "*1..2" in cyphers  # expansão multi-hop
