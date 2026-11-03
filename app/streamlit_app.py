"""Interface Streamlit do sistema GraphRAG/KAG.

Execução:  streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from graphrag.config import DATA_DIR, get_settings  # noqa: E402
from graphrag.engine import GraphRAGEngine  # noqa: E402
from graphrag.evaluation import evaluate, to_markdown  # noqa: E402
from graphrag.generation import path_to_text  # noqa: E402
from graphrag.viz import TYPE_COLORS, render_html, subgraph_payload  # noqa: E402

st.set_page_config(page_title="GraphRAG & KAG — História da Computação", page_icon="🕸️", layout="wide")

EXAMPLES = [
    "Onde trabalhava o orientador de Alan Turing?",
    "Em que cidade nasceu o criador da linguagem C?",
    "Quais pessoas que trabalharam nos Bell Labs receberam o Prêmio Turing?",
    "Qual é a conexão entre Ada Lovelace e Claude Shannon?",
    "Qual linguagem criada no MIT foi influenciada por um conceito criado por Alonzo Church?",
    "Em que país fica a cidade onde nasceu John von Neumann?",
    "Em que ano Dennis Ritchie ganhou o Prêmio Nobel?",
]
MODE_HELP = {
    "kag": "KAG — GraphRAG + plano lógico executado no grafo (+ Text2Cypher se houver LLM e Neo4j)",
    "graphrag": "GraphRAG — busca híbrida (vetor + full-text + linking), multi-hop, caminhos mínimos",
    "vector": "Baseline — RAG vetorial puro (somente chunks)",
}


@st.cache_resource(show_spinner="Conectando ao grafo e preparando o sistema...")
def load_engine() -> GraphRAGEngine:
    eng = GraphRAGEngine(get_settings())
    if eng.s.graph_backend == "memory" or eng.store.stats().nodes.get("Entity", 0) == 0:
        eng.build()
    return eng


try:
    engine = load_engine()
except Exception as e:  # pragma: no cover - depende do ambiente
    st.error(f"Não foi possível iniciar o sistema: {e}")
    st.info("Verifique se o Neo4j está no ar (`docker compose up -d`) e o arquivo `.env`, "
            "ou rode em modo demonstração com `GRAPH_BACKEND=memory`.")
    st.stop()

# ------------------------------------------------------------------ barra lateral
with st.sidebar:
    st.title("🕸️ GraphRAG & KAG")
    st.caption("Projeto Final — Tendências em Ciência da Computação")
    page = st.radio("Seção", ["Perguntar", "Ingestão", "Explorar grafo", "Avaliação"], label_visibility="collapsed")
    st.divider()
    s = engine.s
    st.markdown(
        f"**Backend:** `{s.graph_backend}`  \n"
        f"**LLM:** `{s.llm_model if engine.llm.enabled else 'offline (extrativo)'}`  \n"
        f"**Embeddings:** `{s.embedding_provider}` ({engine.embedder.dim}d)  \n"
        f"**Esquema:** `{s.schema_mode}`"
    )


# ------------------------------------------------------------------ páginas
def page_ask():
    st.header("Perguntas multi-hop com rastreabilidade")
    c1, c2 = st.columns([3, 1])
    with c1:
        ex = st.selectbox("Exemplos", ["(escreva a sua)"] + EXAMPLES)
        q = st.text_input("Pergunta", value="" if ex == "(escreva a sua)" else ex,
                          placeholder="Ex.: Onde trabalhava o orientador de Alan Turing?")
    with c2:
        mode = st.radio("Modo", list(MODE_HELP), format_func=lambda m: m.upper(), help="\n\n".join(MODE_HELP.values()))
        hops = st.slider("Saltos (hops)", 1, 3, engine.s.max_hops)
        k = st.slider("Top-k", 2, 10, engine.s.top_k)
        t2c = st.checkbox("Text2Cypher (LLM)", value=True, disabled=not engine.retriever.t2c.available)
    if not st.button("Responder", type="primary", disabled=not q.strip()):
        return
    with st.spinner("Recuperando subgrafo e gerando resposta..."):
        r = engine.ask(q, mode=mode, k=k, hops=hops, use_text2cypher=t2c)
    res, ans = r.retrieval, r.answer

    box = st.warning if ans.abstained else st.success
    box(ans.text)
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Confiança", f"{ans.confidence:.0%}")
    m2.metric("Âncoras", len(res.anchors))
    m3.metric("Fatos no contexto", len(res.facts))
    m4.metric("Consultas Cypher", len(res.queries))
    m5.metric("Latência", f"{r.elapsed_ms:.0f} ms")
    st.progress(min(1.0, ans.confidence), text=f"Gerador: {ans.generator} · confiança da recuperação {res.retrieval_confidence:.0%}")
    for w in ans.verification.get("warnings", []):
        st.caption(f"⚠️ {w}")

    tabs = st.tabs(["🕸️ Subgrafo", "📜 Fatos e procedência", "🧠 Plano KAG", "🧾 Cypher executado", "✅ Verificação", "🔎 Recuperação"])
    with tabs[0]:
        if res.facts:
            components.html(render_html(subgraph_payload(res, ans.answer_entity_ids)), height=620, scrolling=False)
            if res.paths:
                st.markdown("**Caminhos usados como contexto**")
                for i, p in enumerate(res.paths, 1):
                    st.code(f"P{i}: {path_to_text(p)}", language=None)
        else:
            st.info("O modo vetorial não usa o grafo; veja os trechos na aba Recuperação.")
    with tabs[1]:
        if res.facts:
            st.dataframe(pd.DataFrame([{
                "id": f.id, "fato (tripla)": f.triple(), "origem": ", ".join(sorted(f.origin)),
                "score": round(f.score, 2), "confiança extração": f.props.get("confidence"),
                "fontes (chunks)": ", ".join(f.sources)} for f in res.facts]), hide_index=True, use_container_width=True)
        st.markdown("**Trechos que sustentam os fatos (proveniência)**")
        for c in res.provenance:
            with st.expander(f"{c['title']} — `{c['id']}`"):
                st.write(c["text"])
    with tabs[2]:
        if res.plan:
            st.markdown(f"Planejador: **{res.plan.planner}** · operação: **{res.plan.operation}** · "
                        f"tipo de resposta: **{res.plan.answer_type or '—'}**")
            st.json(res.plan.to_dict())
            if res.kag:
                st.markdown("**Execução passo a passo**")
                st.code("\n".join(res.kag.trace) or "(sem passos)", language=None)
        else:
            st.info("Plano lógico disponível no modo KAG.")
        if res.t2c is not None:
            st.markdown("**Text2Cypher**")
            for a in res.t2c.attempts:
                st.code(a["cypher"], language="cypher")
                if a["errors"]:
                    st.caption(f"Rejeitada/erro: {a['errors']}")
            if res.t2c.ok:
                st.dataframe(pd.DataFrame(res.t2c.rows), use_container_width=True)
    with tabs[3]:
        st.caption(f"Backend `{engine.store.backend}` — todas as consultas executadas para esta resposta, na ordem.")
        for i, qe in enumerate(res.queries, 1):
            with st.expander(f"#{i} · {qe.rows} linha(s) · {qe.ms:.1f} ms · {qe.cypher[:70]}..."):
                st.code(qe.cypher, language="cypher")
                st.json(qe.params)
    with tabs[4]:
        st.json(ans.verification)
        with st.expander("Contexto exato enviado ao gerador"):
            st.code(ans.context, language=None)
    with tabs[5]:
        if res.unlinked:
            st.error(f"Entidades da pergunta inexistentes no grafo: {res.unlinked}")
        if res.anchors:
            st.markdown("**Âncoras (entity linking):** " + ", ".join(f"`{a.name}` ({a.type})" for a in res.anchors))
        if res.seeds:
            st.markdown("**Sementes híbridas (fusão RRF)**")
            st.dataframe(pd.DataFrame([{**sd, "via": ", ".join(sd["via"])} for sd in res.seeds]), hide_index=True)
        st.markdown("**Chunks por similaridade vetorial**")
        st.dataframe(pd.DataFrame([{"score": round(c["score"], 3), "documento": c["title"], "texto": c["text"]}
                                   for c in res.chunks]), hide_index=True, use_container_width=True)
        for c in res.communities:
            st.caption(f"Comunidade `{c['id']}`: {c['summary']}")


def page_ingest():
    st.header("Ingestão de documentos (Text-to-Graph)")
    st.write("Envie arquivos `.md`/`.txt`. O pipeline faz chunking, NER/RE (LLM ou regras), "
             "resolução de entidades e grava no grafo. Opcionalmente use o cabeçalho "
             "`---\\ntitle: ...\\nsubject: ...\\n---`.")
    files = st.file_uploader("Documentos", type=["md", "txt"], accept_multiple_files=True)
    if files and st.button("Ingerir", type="primary"):
        tmp = Path(tempfile.mkdtemp())
        paths = []
        for f in files:
            p = tmp / f.name
            p.write_bytes(f.getvalue())
            paths.append(p)
        with st.spinner("Extraindo entidades e relações..."):
            rep = engine.ingest(files=paths)
            alg = engine.run_algorithms()
        st.success(f"{rep.documents} documento(s), {rep.chunks} chunks, {rep.entities} entidades, {rep.relations} relações "
                   f"(extrator: {rep.extractor}). PageRank/comunidades recalculados ({alg['engine']}).")
        st.dataframe(pd.DataFrame(rep.per_document), hide_index=True)
        if rep.merged:
            st.markdown("**Resoluções de entidade aplicadas**")
            st.dataframe(pd.DataFrame(rep.merged), hide_index=True)


def page_explore():
    st.header("Explorar o grafo")
    stats = engine.store.stats()
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Nós por rótulo")
        st.bar_chart(pd.Series(stats.nodes, name="nós"))
    with c2:
        st.subheader("Relações por tipo")
        st.bar_chart(pd.Series(stats.rels, name="relações"))
    ents = engine.store.all_entities()
    df = pd.DataFrame(ents)
    if not df.empty:
        st.subheader("Entidades mais centrais (PageRank)")
        df = df.sort_values("pagerank", ascending=False)
        st.dataframe(df[["name", "type", "pagerank", "community", "aliases"]].head(25), hide_index=True, use_container_width=True)
    st.subheader("Esquema LPG (constraints e índices)")
    from graphrag.store import cypher as Q

    st.code(";\n".join(Q.schema_statements(engine.ontology, engine.embedder.dim)) + ";", language="cypher")
    st.subheader("Ontologia")
    st.code(engine.ontology.describe(), language=None)
    st.caption("Cores na visualização: " + ", ".join(f"{t}" for t in TYPE_COLORS))


def page_eval():
    st.header("Avaliação comparativa")
    sets = {"Desenvolvimento (21 perguntas)": DATA_DIR / "eval_questions.json",
            "Teste — não usado no desenvolvimento (14 perguntas)": DATA_DIR / "eval_teste.json"}
    name = st.selectbox("Conjunto", list(sets))
    if st.button("Rodar avaliação", type="primary"):
        qs = json.loads(sets[name].read_text(encoding="utf-8"))
        with st.spinner(f"Executando {len(qs) * 3} consultas..."):
            res = evaluate(engine, qs)
        st.markdown(to_markdown(res["summary"]))
        st.dataframe(pd.DataFrame(res["per_question"])[["id", "type", "mode", "correct", "precision", "context_recall",
                                                       "abstained", "confidence", "answer"]],
                     hide_index=True, use_container_width=True)


{"Perguntar": page_ask, "Ingestão": page_ingest, "Explorar grafo": page_explore, "Avaliação": page_eval}[page]()
