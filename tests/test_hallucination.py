"""Mitigação de alucinações: abstenção, premissas falsas e verificação pós-geração."""
from graphrag.generation import ABSTAIN_TEXT, verify_answer


def test_abstains_when_relation_absent(engine):
    r = engine.ask("Quem orientou o doutorado de Grace Hopper?", mode="kag")
    assert r.answer.abstained and ABSTAIN_TEXT in r.answer.text


def test_false_premise_is_detected(engine):
    r = engine.ask("Em que ano Dennis Ritchie ganhou o Prêmio Nobel?", mode="kag")
    assert r.answer.abstained
    assert "Prêmio Nobel" in r.retrieval.unlinked


def test_unknown_entity_triggers_abstention(engine):
    r = engine.ask("Quem criou a linguagem Python?", mode="graphrag")
    assert r.answer.abstained


def test_verifier_flags_invented_entity_and_bad_citation(engine):
    res = engine.ask("Quem orientou o doutorado de Alan Turing?", mode="kag").retrieval
    linker = engine.retriever.linker
    good = verify_answer("Alonzo Church orientou Alan Turing [F1]. Resposta: Alonzo Church [F1].", res, linker)
    assert good["citation_coverage"] == 1.0 and not good["ungrounded_entities"] and not good["unknown_names"]
    bad = verify_answer("Alonzo Church e John von Neumann orientaram Turing [F99]. Kurt Gödel também ajudou.", res, linker)
    assert "F99" in bad["invalid_citations"]
    assert "John von Neumann" in bad["ungrounded_entities"]   # existe no grafo, mas não no contexto
    assert any("Kurt" in u for u in bad["unknown_names"])     # não existe no grafo
    assert bad["citation_coverage"] < 1.0


def test_low_confidence_answers_are_replaced(engine, monkeypatch):
    """Simula um LLM que inventa: a verificação deve rebaixar e abster."""
    gen = engine.generator
    monkeypatch.setattr(gen, "_extractive", lambda res: "Kurt Gödel orientou Alan Turing em Viena.")
    r = engine.ask("Quem orientou o doutorado de Alan Turing?", mode="graphrag")
    assert r.answer.abstained and r.answer.confidence < engine.s.abstain_threshold
