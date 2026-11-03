"""Renderização interativa do subgrafo de contexto com vis-network (vis.js)."""
from __future__ import annotations

import json

TYPE_COLORS = {
    "Person": "#4C78A8",
    "Organization": "#F58518",
    "Location": "#54A24B",
    "Technology": "#B279A2",
    "Concept": "#E45756",
    "Award": "#EECA3B",
}


def subgraph_payload(res, answer_ids: list[str] | None = None) -> dict:
    answer_ids = set(answer_ids or [])
    anchors = {a.id for a in getattr(res, "anchors", [])}
    kag_keys = set()
    if getattr(res, "kag", None) is not None:
        kag_keys = {(f["src"]["id"], f["rel"], f["dst"]["id"]) for f in res.kag.facts}
    nodes, edges = {}, []
    for f in res.facts:
        for n in (f.src, f.dst):
            if n["id"] in nodes:
                continue
            pr = n.get("pagerank") or 0.0
            color = TYPE_COLORS.get(n.get("type"), "#888888")
            node = {
                "id": n["id"], "label": n["name"], "group": n.get("type") or "?",
                "title": f"{n['name']}\n{n.get('type')}\nPageRank: {pr:.4f}",
                "value": 10 + 400 * pr, "color": {"background": color, "border": color},
                "borderWidth": 1, "font": {"size": 14},
            }
            if n["id"] in anchors:
                node.update(borderWidth=4, shape="diamond")
                node["color"]["border"] = "#111111"
            if n["id"] in answer_ids:
                node.update(borderWidth=5)
                node["color"]["border"] = "#D62728"
                node["font"] = {"size": 16, "color": "#D62728"}
            nodes[n["id"]] = node
        key = (f.src["id"], f.rel, f.dst["id"])
        highlighted = key in kag_keys or "shortest" in f.origin
        yr = f" ({f.props['year']})" if f.props.get("year") else ""
        edges.append({
            "from": f.src["id"], "to": f.dst["id"], "label": f"{f.id}: {f.rel}", "arrows": "to",
            "title": f"{f.id} {f.rel}{yr}\nconfiança: {f.props.get('confidence', '?')}\nfontes: {', '.join(f.sources)}",
            "width": 4 if highlighted else 1.5, "color": {"color": "#D62728" if highlighted else "#9AA0A6"},
            "font": {"size": 10, "align": "middle"},
        })
    return {"nodes": list(nodes.values()), "edges": edges}


def render_html(payload: dict, height: int = 560) -> str:
    # JSON embutido em <script>: só é preciso neutralizar "</" (fechamento de tag)
    data = json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")
    legend = "".join(
        f'<span style="display:inline-flex;align-items:center;margin-right:12px">'
        f'<span style="width:12px;height:12px;border-radius:50%;background:{c};display:inline-block;margin-right:4px"></span>{t}</span>'
        for t, c in TYPE_COLORS.items()
    )
    return f"""<!doctype html><html><head><meta charset="utf-8">
<script src="https://unpkg.com/vis-network@9.1.9/standalone/umd/vis-network.min.js"></script>
<style>body{{margin:0;font-family:system-ui,sans-serif}}#g{{height:{height}px;border:1px solid #ddd;border-radius:8px}}
.lg{{font-size:12px;padding:6px 2px;color:#444}}</style></head><body>
<div class="lg">{legend} &nbsp;◆ âncora da pergunta &nbsp;<b style="color:#D62728">○</b> resposta &nbsp;<b style="color:#D62728">━</b> caminho do raciocínio</div>
<div id="g"></div>
<script type="application/json" id="data">{data}</script>
<script>
const d = JSON.parse(document.getElementById('data').textContent);
const net = new vis.Network(document.getElementById('g'),
  {{nodes: new vis.DataSet(d.nodes), edges: new vis.DataSet(d.edges)}},
  {{nodes: {{shape: 'dot', scaling: {{min: 10, max: 40}}}},
    edges: {{smooth: {{type: 'dynamic'}}}},
    physics: {{barnesHut: {{gravitationalConstant: -6000, springLength: 140}}, stabilization: {{iterations: 200}}}},
    interaction: {{hover: true, navigationButtons: true, tooltipDelay: 120}}}});
</script></body></html>"""
