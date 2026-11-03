# Arquitetura do sistema

```mermaid
flowchart LR
  subgraph P1["1 · Text-to-Graph"]
    A[Documentos .md/.txt] --> B[Chunking por parágrafo/sentença]
    B --> C{Extrator}
    C -->|LLM + esquema| D[NER/RE validados pela ontologia]
    C -->|offline| E[Dicionário + padrões lexicais]
    D & E --> F[Resolução de entidades<br/>alias · sobrenome · contenção · fuzzy · embedding]
  end
  subgraph P2["2 · Neo4j LPG"]
    F --> G[(Entity + rótulos de tipo<br/>Document · Chunk · Community)]
    G --- I1[Constraints UNIQUE]
    G --- I2[RANGE · FULLTEXT · VECTOR]
    G --> H[GDS: PageRank · Louvain<br/>+ resumos de comunidade]
  end
  subgraph P3["3 · Recuperação híbrida"]
    Q[Pergunta] --> L[Entity linking + premissas]
    Q --> V[Vector: chunks e entidades]
    Q --> FT[Full-text Lucene]
    L & V & FT --> R[Fusão RRF → sementes]
    R --> X[Expansão k-hop Cypher]
    R --> S[allShortestPaths entre âncoras]
    L --> K[Plano lógico KAG → executor Cypher]
    Q --> T2C[Text2Cypher validado (opcional)]
    X & S & K --> SEL[Seleção de fatos F# ranqueados<br/>pistas da pergunta + PageRank]
  end
  subgraph P4["4 · Geração e interface"]
    SEL --> GEN[Gerador ancorado: LLM ou extrativo]
    GEN --> VER[Verificação: citações, entidades, nomes inventados → confiança/abstenção]
    VER --> UI[Streamlit: resposta · subgrafo vis.js · fatos · plano · Cypher · verificação]
  end
```

## Módulos

| Arquivo | Responsabilidade |
|---|---|
| `graphrag/ontology.py` | Esquema LPG: rótulos, tipos de relação com domínio/alcance, verbalização, modo dinâmico |
| `graphrag/text.py` | Normalização, divisão em sentenças, leitura de documentos, chunking |
| `graphrag/extraction.py` | NER/RE: `LLMExtractor` (JSON validado), `RuleExtractor` (offline), `HybridExtractor` |
| `graphrag/resolution.py` | Canonicalização de entidades em cascata |
| `graphrag/ingest.py` | Pipeline de ingestão incremental |
| `graphrag/store/cypher.py` | **Todas** as consultas Cypher (schema, ingestão, recuperação, GDS) |
| `graphrag/store/neo4j_store.py` | Backend Neo4j (driver oficial, transações de leitura/escrita) |
| `graphrag/store/memory_store.py` | Backend em memória com a mesma semântica (testes/CI/demonstração) |
| `graphrag/algorithms.py` | PageRank, Louvain (GDS ou NetworkX) e resumos de comunidade |
| `graphrag/linking.py` | Entity linking da pergunta e detecção de entidades inexistentes |
| `graphrag/retrieval.py` | Recuperação híbrida, fusão RRF, seleção/numeração de fatos, proveniência |
| `graphrag/kag.py` | Formas lógicas: planejador LLM/heurístico + executor de operadores |
| `graphrag/text2cypher.py` | Cypher gerado por LLM com validação, EXPLAIN e autocorreção |
| `graphrag/generation.py` | Contexto, geração ancorada, verificação pós-geração, confiança |
| `graphrag/evaluation.py` | Avaliação comparativa vetor × GraphRAG × KAG |
| `graphrag/viz.py` | Subgrafo interativo (vis-network) |
| `app/streamlit_app.py` | Interface web |

## Decisões de projeto

1. **Rótulo base `Entity` + rótulo de tipo.** Um único índice vetorial e um único full-text cobrem todas as entidades, e a constraint `Entity.id` garante unicidade global; ao mesmo tempo, consultas por tipo (`:Person`) continuam seletivas.
2. **Cypher do executor KAG é gerado por template, não pelo LLM.** O LLM só escolhe operadores de um vocabulário fechado (relação, direção, tipo-alvo). Tipos e rótulos interpolados são verificados contra a ontologia (`safe_label`, `safe_rel_types`), eliminando injeção de Cypher e Cypher alucinado. O Text2Cypher livre existe como caminho complementar, protegido por validador estático + `EXPLAIN` + transação de leitura.
3. **Proveniência na aresta (`sources`).** Toda resposta pode ser rastreada até o trecho de texto que sustenta cada salto do raciocínio.
4. **Abstenção como resultado de primeira classe.** Perguntas com entidades inexistentes no grafo (premissas falsas) ou sem caminho de evidência resultam em “não sei”, e respostas com confiança abaixo do limiar são substituídas por abstenção.
5. **Dois backends com a mesma interface.** Os testes rodam sem Docker (backend em memória) e a interface registra, nos dois casos, a mesma string Cypher — o que mantém a transparência idêntica.
6. **Funciona sem LLM.** Todos os componentes que usam LLM têm alternativa determinística. Isso torna o sistema reprodutível na banca e permite medir o ganho que vem da estrutura do grafo, separado do ganho que vem do LLM.
