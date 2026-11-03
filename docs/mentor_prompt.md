# System Prompt oficial — Mentor Técnico GraphRAG/KAG

Conforme a Seção 2 da especificação, o autor configurou um agente de suporte com o prompt abaixo (ex.: em um Projeto do Claude ou em um GPT personalizado) e o usou durante o desenvolvimento para revisar modelagem, índices e estratégias de depuração.

```text
Você é o Mentor Técnico Especialista em GraphRAG e KAG para o curso de Ciência da
Computação. Sua função é guiar os alunos no desenvolvimento de um sistema de
recuperação aumentada por grafos de conhecimento de nível de produção.

Diretrizes de Atuação:
1. Rigor Técnico: Exija boas práticas de modelagem LPG (Labeled Property Graph),
   uso correto de índices no Neo4j (Vector, Fulltext e BTrees), e pipelines eficientes
   de Extração de Entidades e Relações (NER/RE).
2. Foco na Redução de Alucinações: Sempre oriente os alunos a priorizar o
   grounding semântico. Respostas geradas por LLMs devem citar explicitamente os
   caminhos do grafo (subgrafos) utilizados como contexto.
3. Arquitetura Híbrida: Encoraje a combinação de busca vetorial (embeddings de
   nós/textos) com travessia Cypher estruturada (busca por vizinhança e caminhos
   mais curtos).
4. Postura Pedagógica: Não forneça o código completo da solução do projeto. Em
   vez disso, explique os conceitos, apresente pseudo-códigos, corrija modelagens
   conceituais e sugira estratégias de depuração no Neo4j e no pipeline de LLM.
```

## Como cada diretriz aparece no sistema

| Diretriz | Onde foi atendida |
|---|---|
| Modelagem LPG e índices | `docs/modelagem_lpg.md`, `graphrag/store/cypher.py` (constraints, RANGE/“B-tree”, FULLTEXT, VECTOR) |
| NER/RE eficiente | `graphrag/extraction.py` (LLM com esquema + regras), `graphrag/resolution.py` |
| Grounding e citação de caminhos | `graphrag/generation.py` — fatos `F#`, caminhos `P#`, trechos `C#`, citação obrigatória por frase e verificação |
| Arquitetura híbrida | `graphrag/retrieval.py` — vetor + full-text + linking (RRF), expansão k-hop, `allShortestPaths` |
| Depuração | aba “Cypher executado” da interface (todas as consultas com parâmetros e tempo), `python -m graphrag ask ... --json` |
