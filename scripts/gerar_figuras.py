"""Gera as figuras do artigo a partir do sistema real (backend em memória, offline).

    python scripts/gerar_figuras.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import networkx as nx  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from graphrag.config import get_settings  # noqa: E402
from graphrag.engine import GraphRAGEngine  # noqa: E402
from graphrag.viz import TYPE_COLORS  # noqa: E402

OUT = ROOT / "artigo" / "figuras"


def subgraph_figure(eng: GraphRAGEngine, question: str, path: Path, max_facts: int = 14):
    r = eng.ask(question, mode="kag")
    res = r.retrieval
    kag_keys = {(f["src"]["id"], f["rel"], f["dst"]["id"]) for f in res.kag.facts} if res.kag else set()
    facts = sorted(res.facts, key=lambda f: ((f.src["id"], f.rel, f.dst["id"]) not in kag_keys, -f.score))[:max_facts]
    G = nx.MultiDiGraph()
    for f in facts:
        G.add_node(f.src["id"], **f.src)
        G.add_node(f.dst["id"], **f.dst)
        G.add_edge(f.src["id"], f.dst["id"], rel=f.rel, hl=(f.src["id"], f.rel, f.dst["id"]) in kag_keys)
    pos = nx.spring_layout(nx.Graph(G), seed=7, k=1.4)
    fig, ax = plt.subplots(figsize=(8.5, 5.6))
    anchors = {a.id for a in res.anchors}
    answers = set(r.answer.answer_entity_ids)
    for n, d in G.nodes(data=True):
        c = TYPE_COLORS.get(d.get("type"), "#999")
        ec = "#D62728" if n in answers else ("#111" if n in anchors else c)
        lw = 3 if n in answers or n in anchors else 0.5
        shape = "D" if n in anchors else "o"
        nx.draw_networkx_nodes(G, pos, nodelist=[n], node_color=c, edgecolors=ec, linewidths=lw,
                               node_shape=shape, node_size=900, ax=ax)
    for u, v, d in G.edges(data=True):
        nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], width=3 if d["hl"] else 1,
                               edge_color="#D62728" if d["hl"] else "#9AA0A6", arrows=True, arrowsize=14,
                               node_size=900, ax=ax, connectionstyle="arc3,rad=0.06")
    nx.draw_networkx_labels(G, pos, labels={n: d["name"] for n, d in G.nodes(data=True)}, font_size=8, ax=ax)
    elabels = {(u, v): d["rel"] for u, v, d in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=elabels, font_size=6.5, ax=ax, label_pos=0.5)
    handles = [plt.Line2D([], [], marker="o", ls="", color=c, label=t) for t, c in TYPE_COLORS.items()
               if any(d.get("type") == t for _, d in G.nodes(data=True))]
    handles += [plt.Line2D([], [], color="#D62728", lw=3, label="caminho KAG")]
    ax.legend(handles=handles, fontsize=7, loc="lower left", frameon=False)
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return r


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    eng = GraphRAGEngine(get_settings(graph_backend="memory", llm_provider="none", extractor="rules"))
    eng.build()
    r = subgraph_figure(eng, "Onde trabalhava o orientador de Alan Turing?", OUT / "subgrafo_turing.png")
    print("subgrafo_turing.png ->", r.answer.text)
    st = eng.store.stats()
    print("nós:", st.nodes)
    print("relações:", st.rels)


if __name__ == "__main__":
    main()
