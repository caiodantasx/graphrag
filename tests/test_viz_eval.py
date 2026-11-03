"""Visualização do subgrafo e avaliação."""
import json
from pathlib import Path

from graphrag.evaluation import evaluate, to_latex
from graphrag.viz import render_html, subgraph_payload


def test_subgraph_payload_highlights_answer(engine):
    r = engine.ask("Onde trabalhava o orientador de Alan Turing?", mode="kag")
    p = subgraph_payload(r.retrieval, r.answer.answer_entity_ids)
    ids = {n["id"] for n in p["nodes"]}
    assert "organization:universidade_de_princeton" in ids
    hi = [n for n in p["nodes"] if n["borderWidth"] == 5]
    assert hi and any(e["width"] == 4 for e in p["edges"])
    html = render_html(p)
    assert "vis-network" in html and "</script>" in html and "Universidade de Princeton" in html


def test_payload_cannot_break_out_of_script():
    html = render_html({"nodes": [{"id": "x", "label": "</script><script>alert(1)</script>"}], "edges": []})
    assert "<\\/script><script>alert(1)" in html


def test_evaluation_kag_beats_vector(engine):
    qs = json.loads((Path(__file__).resolve().parent.parent / "data" / "eval_questions.json").read_text(encoding="utf-8"))
    res = evaluate(engine, qs[:12], modes=["vector", "kag"])
    s = res["summary"]
    assert s["kag"]["acuracia"] > s["vector"]["acuracia"]
    assert "\\begin{tabular}" in to_latex(s, "x")
