# Avaliação (eval_teste_memory_offline)

| Modo | Acurácia | Acur. 1-hop | Acur. multi-hop | Precisão resposta | Recall contexto | Abstenção correta | Resp. não suportada ↓ | Cobertura citações | Latência (ms) |
|---|---|---|---|---|---|---|---|---|---|
| RAG vetorial | 14% | 40% | 0% | 17% | 88% | 0% | 14% | 100% | 3 |
| GraphRAG híbrido | 57% | 100% | 14% | 46% | 100% | 100% | 0% | 100% | 12 |
| KAG (forma lógica) | 79% | 100% | 57% | 60% | 100% | 100% | 0% | 100% | 11 |

- T01 [vector] ✔ — Ada Lovelace nasceu em Londres em 1815. [C1]
- T02 [vector] ✘ — Dijkstra trabalhou no Centro de Matemática de Amsterdã, onde criou o algoritmo de Dijkstra em 1956. [C1]
- T03 [vector] ✔ — Em 1972, Dijkstra recebeu o Prêmio Turing. [C2]
- T04 [vector] ✘ — Em 1994, Berners-Lee fundou o World Wide Web Consortium (W3C) no MIT. [C1]
- T05 [vector] ✘ — Em 1971, McCarthy recebeu o Prêmio Turing. [C2]
- T06 [vector] ✘ — Shannon estudou na Universidade de Michigan e depois estudou no MIT. [C1]
- T07 [vector] ✘ — Knuth trabalhou na Universidade Stanford. [C1]
- T08 [vector] ✘ — A linguagem C foi influenciada pela linguagem B, criada por Ken Thompson. [C1]
- T09 [vector] ✘ — Edsger Dijkstra nasceu em Roterdã em 1930. [C1]
- T10 [vector] ✘ — McCarthy trabalhou no MIT, onde criou a linguagem Lisp em 1958. [C1]
- T11 [vector] ✘ — Em 1971, McCarthy recebeu o Prêmio Turing. [C2]
- T12 [vector] ✘ — Grace Hopper nasceu em Nova York em 1906. [C2]
- T13 [vector] ✘ — Church orientou Alan Turing e Stephen Kleene em seus doutorados. [C2]
- T14 [vector] ✘ — Ritchie criou a linguagem C nos Bell Labs em 1972. [C2]
- T01 [graphrag] ✔ — Ada Lovelace nasceu em Londres (em 1815) [F1]. Resposta: Londres [F1].
- T02 [graphrag] ✔ — Edsger Dijkstra criou algoritmo de Dijkstra (em 1956) [F1]. algoritmo de Dijkstra foi desenvolvido(a) em Centro de Matemática de Amsterdã (em 1956) [F2]. Respos
- T03 [graphrag] ✔ — Edsger Dijkstra recebeu Prêmio Turing (em 1972) [F1]. Resposta: Prêmio Turing [F1].
- T04 [graphrag] ✔ — Tim Berners-Lee fundou World Wide Web Consortium (em 1994) [F1]. Resposta: Tim Berners-Lee [F1].
- T05 [graphrag] ✔ — Prêmio Turing é concedido por Association for Computing Machinery (em 1966) [F1]. Resposta: Association for Computing Machinery [F1].
- T06 [graphrag] ✘ — Ken Thompson criou Unix (em 1969) [F1]. Unix foi desenvolvido(a) em Bell Labs (em 1969) [F2]. Dennis Ritchie criou Unix [F3]. Resposta: Ken Thompson, Bell Labs,
- T07 [graphrag] ✘ — Donald Knuth trabalhou em Universidade Stanford [F1]. Resposta: Universidade Stanford [F1].
- T08 [graphrag] ✘ — Ken Thompson criou linguagem B [F1]. linguagem B foi desenvolvido(a) em Bell Labs [F2]. linguagem B influenciou linguagem C [F3]. Resposta: Ken Thompson, Bell L
- T09 [graphrag] ✘ — Edsger Dijkstra nasceu em Roterdã (em 1930) [F1]. Resposta: Roterdã [F1].
- T10 [graphrag] ✘ — Caminho no grafo: MIT <-[:DEVELOPED_AT]- Lisp [F1].
- T11 [graphrag] ✘ — Ken Thompson recebeu Prêmio Turing (em 1983) [F1]. Edsger Dijkstra recebeu Prêmio Turing (em 1972) [F2]. John McCarthy recebeu Prêmio Turing (em 1971) [F3]. Den
- T12 [graphrag] ✔ — Caminho no grafo: Grace Hopper -[:WORKED_AT]-> Universidade Harvard <-[:STUDIED_AT]- Dennis Ritchie [F1, F2]. Resposta: Universidade Harvard [F1, F2].
- T13 [graphrag] ✔ — Não encontrei evidências suficientes no grafo de conhecimento para responder com segurança.
- T14 [graphrag] ✔ — Não encontrei evidências suficientes no grafo de conhecimento para responder com segurança. A pergunta menciona 'Python', que não existe(m) no grafo — a premiss
- T01 [kag] ✔ — Ada Lovelace nasceu em Londres (em 1815) [F1]. Resposta: Londres [F1].
- T02 [kag] ✔ — Edsger Dijkstra criou algoritmo de Dijkstra (em 1956) [F1]. Resposta: Edsger Dijkstra [F1].
- T03 [kag] ✔ — Edsger Dijkstra recebeu Prêmio Turing (em 1972) [F1]. Resposta: Prêmio Turing [F1].
- T04 [kag] ✔ — Tim Berners-Lee fundou World Wide Web Consortium (em 1994) [F1]. Resposta: Tim Berners-Lee [F1].
- T05 [kag] ✔ — Prêmio Turing é concedido por Association for Computing Machinery (em 1966) [F1]. Resposta: Association for Computing Machinery [F1].
- T06 [kag] ✔ — Dennis Ritchie criou Unix [F2]. Ken Thompson criou Unix (em 1969) [F1]. Dennis Ritchie estudou em Universidade Harvard [F3]. Ken Thompson estudou em Universidad
- T07 [kag] ✔ — Donald Knuth trabalhou em Universidade Stanford [F1]. Universidade Stanford fica em Califórnia [F2]. Resposta: Califórnia (via Universidade Stanford) [F2].
- T08 [kag] ✘ — linguagem B influenciou linguagem C [F1]. linguagem C foi desenvolvido(a) em Bell Labs (em 1972) [F3]. Resposta: Bell Labs (via linguagem C) [F3].
- T09 [kag] ✘ — Edsger Dijkstra nasceu em Roterdã (em 1930) [F1]. Resposta: Roterdã [F1].
- T10 [kag] ✘ — Caminho no grafo: MIT <-[:DEVELOPED_AT]- Lisp [F1].
- T11 [kag] ✔ — Dennis Ritchie recebeu Prêmio Turing (em 1983) [F2]. Donald Knuth recebeu Prêmio Turing (em 1974) [F5]. Edsger Dijkstra recebeu Prêmio Turing (em 1972) [F3]. Jo
- T12 [kag] ✔ — Caminho no grafo: Grace Hopper -[:WORKED_AT]-> Universidade Harvard <-[:STUDIED_AT]- Dennis Ritchie [F1, F2]. Resposta: Universidade Harvard [F1, F2].
- T13 [kag] ✔ — Não encontrei evidências suficientes no grafo de conhecimento para responder com segurança.
- T14 [kag] ✔ — Não encontrei evidências suficientes no grafo de conhecimento para responder com segurança. A pergunta menciona 'Python', que não existe(m) no grafo — a premiss