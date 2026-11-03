"""Ingestão ponta a ponta no backend em memória."""


def test_graph_counts(engine):
    st = engine.store.stats()
    assert st.nodes["Document"] == 15
    assert st.nodes["Person"] >= 20
    assert st.rels["MENTIONS"] > 100
    assert st.rels["ADVISED"] >= 2


def test_no_duplicate_entities(engine):
    ents = engine.store.all_entities()
    names = [(e["name"], e["type"]) for e in ents]
    assert len(names) == len(set(names))
    turing = [e for e in ents if "Turing" in e["name"] and e["type"] == "Person"]
    assert len(turing) == 1  # "Turing", "Alan Turing" -> um único nó


def test_relation_has_provenance(engine):
    rows = engine.store.step(["person:alan_turing"], ["ADVISED"], "in", None)
    assert rows and rows[0]["src"]["name"] == "Alonzo Church"
    props = rows[0]["props"]
    assert props["sources"] and props["sources"][0].startswith("doc:")
    assert props["year"] == 1938 and 0 < props["confidence"] <= 1


def test_relation_merges_evidence_from_multiple_chunks(engine):
    rows = engine.store.step(["person:alonzo_church"], ["ADVISED"], "out", "Person")
    turing = [r for r in rows if r["next"]["name"] == "Alan Turing"][0]
    assert len(turing["props"]["sources"]) == 2  # citado no doc de Turing e no de Church


def test_graph_algorithms_written(engine):
    ents = engine.store.all_entities()
    assert all(e["pagerank"] is not None and e["community"] is not None for e in ents)
    top = max(ents, key=lambda e: e["pagerank"])
    assert top["type"] in ("Person", "Award")
    comms = engine.store.communities_of(["person:dennis_ritchie"])
    assert comms and "Comunidade" in comms[0]["summary"]


def test_incremental_ingestion(engine, tmp_path):
    p = tmp_path / "barbara_liskov.md"
    p.write_text("---\ntitle: Barbara Liskov\nsubject: Barbara Liskov\n---\n"
                 "Barbara Liskov trabalhou no MIT. Liskov recebeu o Prêmio Turing em 2008.", encoding="utf-8")
    before = engine.store.stats().nodes["Entity"]
    rep = engine.ingest(files=[p])
    assert rep.relations >= 2
    assert engine.store.stats().nodes["Entity"] == before + 1  # só a nova pessoa; MIT e Prêmio reaproveitados
    r = engine.ask("Quem trabalhou no MIT e recebeu o Prêmio Turing?", mode="kag")
    assert "Barbara Liskov" in r.answer.text
