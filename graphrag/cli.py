"""Linha de comando.

Exemplos:
  python -m graphrag build --reset          # schema + ingestão + algoritmos
  python -m graphrag ask "Onde trabalhava o orientador de Alan Turing?"
  python -m graphrag ask "..." --mode vector
  python -m graphrag evaluate               # gera results/ e tabela LaTeX
  python -m graphrag stats
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import DATA_DIR, ROOT, get_settings
from .engine import GraphRAGEngine


def main(argv=None):
    ap = argparse.ArgumentParser(prog="graphrag", description="Sistema GraphRAG/KAG sobre Neo4j")
    ap.add_argument("--backend", choices=["neo4j", "memory"], help="sobrescreve GRAPH_BACKEND")
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="cria schema, ingere o corpus e roda algoritmos")
    b.add_argument("--reset", action="store_true", help="apaga o banco antes")
    b.add_argument("--corpus", type=Path)
    sub.add_parser("schema", help="cria apenas constraints e índices")
    i = sub.add_parser("ingest", help="ingere arquivos/diretório (incremental)")
    i.add_argument("paths", nargs="+", type=Path)
    sub.add_parser("algorithms", help="PageRank + Louvain + resumos de comunidade")
    a = sub.add_parser("ask", help="faz uma pergunta")
    a.add_argument("question")
    a.add_argument("--mode", choices=["vector", "graphrag", "kag"], default="kag")
    a.add_argument("--hops", type=int)
    a.add_argument("--k", type=int)
    a.add_argument("--json", action="store_true")
    e = sub.add_parser("evaluate", help="avaliação comparativa dos três modos")
    e.add_argument("--questions", type=Path, default=DATA_DIR / "eval_questions.json")
    e.add_argument("--out", type=Path, default=ROOT / "results")
    sub.add_parser("stats", help="contagem de nós e relações")
    args = ap.parse_args(argv)

    overrides = {"graph_backend": args.backend} if args.backend else {}
    eng = GraphRAGEngine(get_settings(**overrides))
    # backend em memória não persiste: reconstrói o grafo a cada execução
    if eng.s.graph_backend == "memory" and args.cmd not in ("build", "schema"):
        eng.build()
    try:
        if args.cmd == "build":
            if args.corpus:
                eng.s.corpus_dir = args.corpus
            out = eng.build(reset=args.reset)
            rep, alg = out["ingest"], out["algorithms"]
            print(f"Extrator: {rep.extractor}")
            print(f"Documentos: {rep.documents} | chunks: {rep.chunks} | entidades: {rep.entities} | "
                  f"relações extraídas: {rep.relations} | menções: {rep.mentions}")
            if rep.merged:
                print("Resoluções de entidade não triviais:")
                for m in rep.merged[:20]:
                    print(f"  '{m['surface']}' -> '{m['canonical']}' ({m['method']}, {m['score']})")
            print(f"Algoritmos: {alg['engine']} | comunidades: {alg['communities']}")
            print("Top PageRank:", ", ".join(f"{n} ({v})" for n, v in alg["top_pagerank"][:5]))
        elif args.cmd == "schema":
            for s in eng.setup_schema():
                print(s)
        elif args.cmd == "ingest":
            files = [p for path in args.paths for p in ([path] if path.is_file() else sorted(path.glob("*.md")) + sorted(path.glob("*.txt")))]
            rep = eng.ingest(files=files)
            print(f"{rep.documents} documento(s), {rep.relations} relações. Rode 'algorithms' para atualizar PageRank/comunidades.")
        elif args.cmd == "algorithms":
            print(json.dumps(eng.run_algorithms(), ensure_ascii=False, indent=2))
        elif args.cmd == "ask":
            r = eng.ask(args.question, mode=args.mode, k=args.k, hops=args.hops)
            if args.json:
                print(json.dumps({
                    "answer": r.answer.text, "confidence": r.answer.confidence, "abstained": r.answer.abstained,
                    "facts": [{"id": f.id, "triple": f.triple(), "sources": f.sources} for f in r.retrieval.facts],
                    "plan": r.retrieval.plan.to_dict() if r.retrieval.plan else None,
                    "verification": r.answer.verification,
                    "cypher": [q.cypher for q in r.retrieval.queries],
                }, ensure_ascii=False, indent=2, default=str))
            else:
                print(r.answer.text)
                print(f"\nconfiança: {r.answer.confidence:.2f} | gerador: {r.answer.generator} | {r.elapsed_ms:.0f} ms")
                if r.retrieval.plan:
                    print("plano KAG:", " ; ".join(r.retrieval.kag.trace) if r.retrieval.kag else "-")
                for f in r.retrieval.facts[:10]:
                    print(f"  {f.id}: {f.triple()}  fontes={f.sources}")
        elif args.cmd == "evaluate":
            from .evaluation import run_and_save, to_markdown

            res = run_and_save(eng, args.questions, args.out)
            print(to_markdown(res["summary"]))
            print(f"\nDetalhes salvos em {args.out}/")
        elif args.cmd == "stats":
            st = eng.store.stats()
            print("Nós:", st.nodes)
            print("Relações:", st.rels)
    finally:
        eng.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
