"""Avaliação comparativa: RAG vetorial (baseline) x GraphRAG x KAG.

Métricas (por modo):
  * acurácia          — resposta contém todas as entidades esperadas (ou se
                        absteve, nas perguntas sem resposta no corpus);
  * recall de contexto — fração das entidades esperadas presentes no contexto
                        entregue ao gerador (independe do LLM);
  * abstenção correta — % de perguntas sem resposta em que o sistema se absteve;
  * taxa de resposta não suportada (proxy de alucinação) — respostas emitidas
                        para perguntas sem resposta, ou com entidades/nomes que
                        não estão no contexto recuperado;
  * cobertura de citações e latência média.
"""
from __future__ import annotations

import json
import statistics
from pathlib import Path

from .engine import GraphRAGEngine
from .text import normalize

MODES = ["vector", "graphrag", "kag"]
MODE_LABEL = {"vector": "RAG vetorial", "graphrag": "GraphRAG híbrido", "kag": "KAG (forma lógica)"}


def _contains(text: str, name: str) -> bool:
    return f" {normalize(name)} " in f" {normalize(text)} "


def evaluate(engine: GraphRAGEngine, questions: list[dict], modes=MODES) -> dict:
    out = {"per_question": [], "summary": {}}
    for mode in modes:
        rows = []
        for q in questions:
            r = engine.ask(q["question"], mode=mode)
            exp = q.get("expected")
            ans, ctx = r.answer.text, r.answer.context
            if exp:
                answer_only = ans.split("(Resposta candidata descartada")[0]
                correct = (not r.answer.abstained) and all(_contains(answer_only, e) for e in exp)
                recall = sum(_contains(ctx, e) for e in exp) / len(exp)
            else:
                correct = r.answer.abstained
                recall = None
            v = r.answer.verification
            precision = None
            if exp and not r.answer.abstained:
                line = ans.split("Resposta:")[-1] if "Resposta:" in ans else ans
                got = {l.name for l in engine.retriever.linker.link(line)}
                if got:
                    precision = len(got & set(exp)) / len(got)
            unsupported = (not r.answer.abstained) and (exp is None or bool(v["ungrounded_entities"]) or bool(v["unknown_names"]))
            rows.append({
                "id": q["id"], "type": q["type"], "mode": mode, "question": q["question"],
                "correct": correct, "context_recall": recall, "precision": precision, "abstained": r.answer.abstained,
                "unsupported": unsupported, "citation_coverage": v["citation_coverage"],
                "confidence": r.answer.confidence, "latency_ms": round(r.elapsed_ms, 1),
                "answer": ans[:500],
            })
        answerable = [x for x in rows if x["context_recall"] is not None]
        unans = [x for x in rows if x["context_recall"] is None]
        emitted = [x for x in rows if not x["abstained"]]
        mh = [x for x in rows if x["type"] in ("multi-hop", "caminho")]
        out["summary"][mode] = {
            "acuracia": round(sum(x["correct"] for x in rows) / len(rows), 3),
            "acuracia_1hop": _rate([x for x in rows if x["type"] == "1-hop"]),
            "acuracia_multihop": _rate(mh),
            "precisao_resposta": _mean([x["precision"] for x in rows]),
            "recall_contexto": round(statistics.mean(x["context_recall"] for x in answerable), 3) if answerable else None,
            "abstencao_correta": _rate(unans),
            "resposta_nao_suportada": round(sum(x["unsupported"] for x in rows) / max(1, len(emitted)), 3),
            "cobertura_citacoes": round(statistics.mean(x["citation_coverage"] for x in emitted), 3) if emitted else None,
            "latencia_ms": round(statistics.mean(x["latency_ms"] for x in rows), 1),
        }
        out["per_question"] += rows
    return out


def _mean(vals):
    vals = [v for v in vals if v is not None]
    return round(statistics.mean(vals), 3) if vals else None


def _rate(rows):
    return round(sum(x["correct"] for x in rows) / len(rows), 3) if rows else None


def to_markdown(summary: dict) -> str:
    cols = [("acuracia", "Acurácia"), ("acuracia_1hop", "Acur. 1-hop"), ("acuracia_multihop", "Acur. multi-hop"),
            ("precisao_resposta", "Precisão resposta"), ("recall_contexto", "Recall contexto"), ("abstencao_correta", "Abstenção correta"),
            ("resposta_nao_suportada", "Resp. não suportada ↓"), ("cobertura_citacoes", "Cobertura citações"),
            ("latencia_ms", "Latência (ms)")]
    lines = ["| Modo | " + " | ".join(c[1] for c in cols) + " |", "|---" * (len(cols) + 1) + "|"]
    for mode, s in summary.items():
        lines.append(f"| {MODE_LABEL.get(mode, mode)} | " + " | ".join(_fmt(s[c[0]], c[0]) for c in cols) + " |")
    return "\n".join(lines)


def to_latex(summary: dict, caption: str, label: str = "tab:resultados") -> str:
    cols = [("acuracia", "Acur."), ("acuracia_multihop", "Multi-hop"), ("precisao_resposta", "Prec."), ("recall_contexto", "Recall ctx."),
            ("abstencao_correta", "Abst."), ("resposta_nao_suportada", "N. sup.$\\downarrow$"),
            ("cobertura_citacoes", "Citações")]
    body = "\n".join(
        f"{MODE_LABEL.get(m, m)} & " + " & ".join(_fmt(s[c[0]], c[0], latex=True) for c in cols) + " \\\\"
        for m, s in summary.items()
    )
    return (
        "\\begin{table}[ht]\n\\centering\n\\caption{" + caption + "}\n\\label{" + label + "}\n"
        "\\begin{tabular}{l" + "c" * len(cols) + "}\n\\hline\n"
        "Modo & " + " & ".join(c[1] for c in cols) + " \\\\\n\\hline\n" + body + "\n\\hline\n\\end{tabular}\n\\end{table}\n"
    )


def _fmt(v, key, latex=False):
    if v is None:
        return "--"
    if key == "latencia_ms":
        return f"{v:.0f}"
    pct = f"{100 * v:.0f}%"
    return pct.replace("%", "\\%") if latex else pct


def run_and_save(engine: GraphRAGEngine, questions_path: Path, out_dir: Path) -> dict:
    questions = json.loads(Path(questions_path).read_text(encoding="utf-8"))
    res = evaluate(engine, questions)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{Path(questions_path).stem}_{engine.s.graph_backend}_{'llm' if engine.llm.enabled else 'offline'}"
    (out_dir / f"avaliacao_{tag}.json").write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / f"avaliacao_{tag}.md").write_text(
        f"# Avaliação ({tag})\n\n{to_markdown(res['summary'])}\n\n"
        + "\n".join(f"- {x['id']} [{x['mode']}] {'✔' if x['correct'] else '✘'} — {x['answer'][:160]}" for x in res["per_question"]),
        encoding="utf-8",
    )
    n = len(questions)
    cap = (f"Resultados no conjunto de {n} perguntas (modo offline: extração por regras, embeddings por hashing "
           "e gerador extrativo)." if not engine.llm.enabled else f"Resultados no conjunto de {n} perguntas (LLM: {engine.s.llm_model}).")
    (out_dir / f"tabela_{tag}.tex").write_text(to_latex(res["summary"], cap, f"tab:{Path(questions_path).stem}"), encoding="utf-8")
    return res
