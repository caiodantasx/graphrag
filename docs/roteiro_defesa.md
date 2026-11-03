# Roteiro para a apresentação e defesa (Fase 4)

Tempo sugerido: 15 min de apresentação + 5 min de demonstração + perguntas.

## Estrutura dos slides

1. **Problema** — LLMs alucinam, falham em multi-hop e não mostram de onde tiraram a resposta. Exemplo: “Onde trabalhava o orientador de Alan Turing?” — nenhuma frase do corpus responde isso sozinha.
2. **Proposta** — GraphRAG + KAG sobre Neo4j: o texto vira grafo; a pergunta vira um plano executado no grafo; a resposta cita os fatos.
3. **Domínio e dados** — História da Computação, 15 documentos, 99 entidades, 134 relações extraídas.
4. **Modelagem LPG** — diagrama do esquema (docs/modelagem_lpg.md), constraints e os três tipos de índice e *por que* cada um.
5. **Pipeline Text-to-Graph** — chunking → NER/RE (LLM com esquema ou regras) → resolução de entidades (mostrar “Turing” → “Alan Turing”, as duas Cambridges).
6. **Recuperação híbrida** — vetor + full-text + linking fundidos por RRF; expansão de 2 saltos; caminhos mínimos; PageRank e comunidades Louvain.
7. **KAG** — plano lógico: `Alan Turing <-ADVISED- x1 -WORKED_AT-> x2`; o LLM escolhe operadores, o executor gera Cypher seguro.
8. **Mitigação de alucinações** — antes (contexto só com fatos), durante (prompt com citação obrigatória), depois (verificação + abstenção). Mostrar a pergunta do “Prêmio Nobel”.
9. **Resultados** — tabelas de desenvolvimento e de teste; destacar recall de contexto vs. acurácia do baseline vetorial e as falhas analisadas.
10. **Limitações e trabalhos futuros**.

## Demonstração (nesta ordem)

1. `docker compose up -d neo4j` e `python -m graphrag build --reset` (mostrar a saída com contagens e resoluções).
2. Neo4j Browser (`http://localhost:7474`): `SHOW INDEXES;` e a consulta multi-hop do documento de modelagem.
3. `streamlit run app/streamlit_app.py`:
   - “Onde trabalhava o orientador de Alan Turing?” → aba **Subgrafo** (caminho em vermelho), **Plano KAG**, **Cypher executado**.
   - Mesma pergunta no modo **VECTOR** → a resposta não fecha o raciocínio.
   - “Qual é a conexão entre Ada Lovelace e Claude Shannon?” → caminho mínimo.
   - “Em que ano Dennis Ritchie ganhou o Prêmio Nobel?” → abstenção por premissa falsa.
4. Página **Ingestão**: subir um arquivo novo (ex.: Barbara Liskov) e perguntar sobre ele.

## Perguntas prováveis da banca (e respostas curtas)

**Por que não só busca vetorial?** Porque a similaridade recupera trechos parecidos com a pergunta, mas não *compõe* fatos de documentos diferentes. No nosso teste, o baseline vetorial tinha a informação no contexto em ~90% das vezes, mas acertou bem menos, sobretudo em multi-hop.

**Onde estão os índices B-tree?** No Neo4j 5 o B-tree foi substituído pelo índice RANGE, que é o sucessor direto; usamos RANGE em `name`, `type`, `pagerank`, `community`, `doc_id` e em `RECEIVED.year`. Além disso, toda constraint de unicidade cria um índice implícito.

**Como garantem que não haja nós duplicados?** Constraint `Entity.id IS UNIQUE` + id determinístico `<tipo>:<slug>` + `MERGE` + resolução de entidades antes da escrita.

**Qual a diferença entre GraphRAG e KAG no projeto?** GraphRAG recupera um subgrafo relevante (vizinhança, caminhos) e entrega ao gerador. KAG vai além: transforma a pergunta numa forma lógica (sequência de operadores) e a *executa* no grafo, como no artigo do KAG (Liang et al., 2024).

**E se o LLM gerar um Cypher perigoso?** No KAG o LLM não escreve Cypher. No Text2Cypher opcional há validador (só leitura, rótulos/tipos do esquema, LIMIT), `EXPLAIN` antes e execução em transação de leitura.

**Como medem a confiança?** 45% recuperação (âncoras ligadas, plano executado, fatos), 30% cobertura de citações válidas por frase, 25% fração de entidades da resposta presentes no contexto; penalidades para citação inválida ou nome inventado; abaixo de 0,35 vira abstenção.

**O resultado de 100% no desenvolvimento não é viés?** Sim, por isso há o conjunto de teste separado (14 perguntas, escrito depois e não usado para ajustar o sistema), onde o KAG ficou em 79%. As falhas estão analisadas no artigo.

**Funciona com outro domínio?** A ontologia é um arquivo; com `SCHEMA_MODE=dynamic` e um LLM, o extrator aceita relações novas. O dicionário de entidades só é necessário no modo offline.

**Quais bibliotecas usaram?** Driver oficial `neo4j`, Neo4j GDS, NetworkX, NumPy, Streamlit, vis-network, Requests. Não usamos LangChain/LlamaIndex para manter cada etapa explícita e explicável (ver “Créditos” no README).
