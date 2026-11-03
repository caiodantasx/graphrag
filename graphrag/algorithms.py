"""Algoritmos de grafo: PageRank, detecção de comunidades (Louvain) e
resumos de comunidade (estilo "community summaries" do GraphRAG da Microsoft).

Preferência: Neo4j Graph Data Science (no servidor). Fallback: NetworkX sobre
as arestas exportadas do grafo, com escrita das propriedades de volta.
"""
from __future__ import annotations

from collections import Counter

import networkx as nx

from .llm import LLMClient, LLMError
from .store.base import GraphStore

SUMMARY_SYSTEM = (
    "Você resume comunidades de um grafo de conhecimento. Escreva 2 frases em português, "
    "usando SOMENTE os fatos fornecidos, descrevendo o tema que une as entidades."
)


def run_graph_algorithms(store: GraphStore, llm: LLMClient | None = None, seed: int = 42) -> dict:
    used_gds = store.try_gds()
    ents = {e["id"]: e for e in store.all_entities()}
    edges = store.entity_edges()
    G = nx.DiGraph()
    G.add_nodes_from(ents)
    for u, v, _ in edges:
        G.add_edge(u, v)

    if not used_gds:
        pr = nx.pagerank(G.to_undirected(), alpha=0.85, max_iter=100) if G.number_of_nodes() else {}
        comms = nx.community.louvain_communities(G.to_undirected(), seed=seed) if G.number_of_nodes() else []
        comm_of = {n: i for i, c in enumerate(comms) for n in c}
        rows = [{"id": n, "props": {"pagerank": round(float(pr.get(n, 0.0)), 6), "community": int(comm_of.get(n, -1))}}
                for n in ents]
        store.write_node_props(rows)
        ents = {e["id"]: e for e in store.all_entities()}

    # ---- resumos de comunidade
    groups: dict[int, list[str]] = {}
    for eid, e in ents.items():
        groups.setdefault(e.get("community", -1), []).append(eid)
    names = {eid: e["name"] for eid, e in ents.items()}
    communities = []
    for cid, members in sorted(groups.items(), key=lambda x: -len(x[1])):
        members.sort(key=lambda m: -(ents[m].get("pagerank") or 0))
        top = [names[m] for m in members[:6]]
        internal = [(u, r, v) for u, v, r in edges if u in members and v in members]
        rel_counts = Counter(r for _, r, _ in internal)
        summary = None
        if llm is not None and llm.enabled and internal:
            facts = "\n".join(f"- {names[u]} {r} {names[v]}" for u, r, v in internal[:40])
            try:
                summary = llm.chat(SUMMARY_SYSTEM, f"Entidades principais: {', '.join(top)}\nFatos:\n{facts}").strip()
            except LLMError:
                summary = None
        if not summary:
            rels_txt = ", ".join(f"{r} ({n})" for r, n in rel_counts.most_common(3)) or "sem relações internas"
            summary = (f"Comunidade de {len(members)} entidades centrada em {', '.join(top[:4])}. "
                       f"Relações predominantes: {rels_txt}.")
        communities.append({"id": f"community:{cid}", "summary": summary, "size": len(members),
                            "top_entities": top, "members": members})
    store.replace_communities(communities)

    top_pr = sorted(ents.values(), key=lambda e: -(e.get("pagerank") or 0))[:10]
    return {
        "engine": "Neo4j GDS" if used_gds else "NetworkX (fallback)",
        "communities": len(communities),
        "top_pagerank": [(e["name"], round(e.get("pagerank") or 0, 4)) for e in top_pr],
        "community_list": [{k: c[k] for k in ("id", "size", "top_entities", "summary")} for c in communities],
    }
