# Avaliação (eval_questions_memory_offline)

| Modo | Acurácia | Acur. 1-hop | Acur. multi-hop | Precisão resposta | Recall contexto | Abstenção correta | Resp. não suportada ↓ | Cobertura citações | Latência (ms) |
|---|---|---|---|---|---|---|---|---|---|
| RAG vetorial | 24% | 43% | 18% | 20% | 97% | 0% | 14% | 100% | 3 |
| GraphRAG híbrido | 71% | 100% | 46% | 56% | 100% | 100% | 0% | 100% | 13 |
| KAG (forma lógica) | 100% | 100% | 100% | 82% | 100% | 100% | 0% | 100% | 13 |

- Q01 [vector] ✔ — Alan Turing nasceu em Londres em 1912. [C1]
- Q02 [vector] ✘ — O Prêmio Turing foi nomeado em homenagem a Alan Turing. [C2]
- Q03 [vector] ✘ — Shannon trabalhou nos Bell Labs, onde formulou a teoria da informação em 1948. [C2]
- Q04 [vector] ✔ — O transistor foi desenvolvido nos Bell Labs em 1947. [C1]
- Q05 [vector] ✘ — Lovelace colaborou com Charles Babbage no projeto da Máquina Analítica. [C3]
- Q06 [vector] ✔ — A linguagem C foi influenciada pela linguagem B, criada por Ken Thompson. [C1]
- Q07 [vector] ✘ — A linguagem Lisp foi influenciada pelo cálculo lambda de Alonzo Church. [C3]
- Q08 [vector] ✘ — O Prêmio Turing foi nomeado em homenagem a Alan Turing. [C2]
- Q09 [vector] ✘ — A linguagem C foi influenciada pela linguagem B, criada por Ken Thompson. [C1]
- Q10 [vector] ✘ — O Prêmio Turing foi nomeado em homenagem a Alan Turing. [C1]
- Q11 [vector] ✘ — Em 1989, Berners-Lee propôs a World Wide Web no CERN. [C1]
- Q12 [vector] ✘ — John von Neumann nasceu em Budapeste em 1903. [C1]
- Q13 [vector] ✘ — Em 1978, Knuth criou o sistema TeX na Universidade Stanford. [C1]
- Q14 [vector] ✔ — Berners-Lee trabalhou no CERN, localizado em Genebra. [C1]
- Q15 [vector] ✘ — O Prêmio Turing foi nomeado em homenagem a Alan Turing. [C1]
- Q16 [vector] ✔ — A linguagem Lisp foi influenciada pelo cálculo lambda de Alonzo Church. [C2]
- Q17 [vector] ✘ — McCarthy trabalhou no MIT, onde criou a linguagem Lisp em 1958. [C1]
- Q18 [vector] ✘ — Ada Lovelace foi uma matemática e escritora inglesa. [C1]
- Q19 [vector] ✘ — Hopper estudou na Universidade Yale, onde obteve o doutorado em matemática em 1934. [C1]
- Q20 [vector] ✘ — Dennis Ritchie foi um cientista da computação norte-americano. [C1]
- Q21 [vector] ✘ — A linguagem de programação Ada, criada para o Departamento de Defesa dos Estados Unidos em 1980, foi nomeada em homenagem a Ada Lovelace. [C1]
- Q01 [graphrag] ✔ — Alan Turing nasceu em Londres (em 1912) [F1]. Resposta: Londres [F1].
- Q02 [graphrag] ✔ — Alonzo Church orientou Alan Turing (em 1938) [F1]. Resposta: Alonzo Church [F1].
- Q03 [graphrag] ✔ — teoria da informação foi desenvolvido(a) em Bell Labs (em 1948) [F1]. Claude Shannon criou teoria da informação (em 1948) [F2]. Resposta: Bell Labs, Claude Shan
- Q04 [graphrag] ✔ — Ken Thompson criou Unix (em 1969) [F1]. Unix foi desenvolvido(a) em Bell Labs (em 1969) [F2]. Dennis Ritchie criou Unix [F3]. Resposta: Ken Thompson, Bell Labs,
- Q05 [graphrag] ✔ — Charles Babbage colaborou com Ada Lovelace [F1]. Ada Lovelace colaborou com Charles Babbage [F2]. Resposta: Ada Lovelace [F1, F2].
- Q06 [graphrag] ✔ — linguagem B influenciou linguagem C [F1]. Resposta: linguagem B [F1].
- Q07 [graphrag] ✔ — Alonzo Church orientou Alan Turing (em 1938) [F1]. Alonzo Church orientou Stephen Kleene [F2]. Resposta: Alan Turing, Stephen Kleene [F1, F2].
- Q08 [graphrag] ✘ — Alonzo Church orientou Alan Turing (em 1938) [F1]. Alan Turing trabalhou em Bletchley Park [F2]. Alan Turing trabalhou em Universidade de Manchester [F3]. Respo
- Q09 [graphrag] ✘ — linguagem C foi desenvolvido(a) em Bell Labs (em 1972) [F1]. Dennis Ritchie criou linguagem C (em 1972) [F2]. Resposta: Bell Labs, Dennis Ritchie [F1, F2].
- Q10 [graphrag] ✘ — Prêmio Turing foi nomeado(a) em homenagem a Alan Turing [F1]. Resposta: Alan Turing [F1].
- Q11 [graphrag] ✘ — Tim Berners-Lee criou World Wide Web (em 1989) [F1]. World Wide Web foi desenvolvido(a) em CERN (em 1989) [F2]. Resposta: Tim Berners-Lee, CERN [F1, F2].
- Q12 [graphrag] ✘ — John von Neumann nasceu em Budapeste (em 1903) [F1]. Resposta: Budapeste [F1].
- Q13 [graphrag] ✔ — Donald Knuth criou TeX (em 1978) [F1]. TeX foi desenvolvido(a) em Universidade Stanford (em 1978) [F2]. Resposta: Donald Knuth, Universidade Stanford [F1, F2].
- Q14 [graphrag] ✘ — Tim Berners-Lee trabalhou em CERN [F1]. Resposta: CERN [F1].
- Q15 [graphrag] ✔ — Caminho no grafo: Bell Labs <-[:WORKED_AT]- Dennis Ritchie -[:RECEIVED]-> Prêmio Turing [F3, F4]. Caminho no grafo: Bell Labs <-[:WORKED_AT]- Ken Thompson -[:RE
- Q16 [graphrag] ✔ — Caminho no grafo: MIT <-[:DEVELOPED_AT]- Lisp <-[:INFLUENCED]- Alonzo Church [F1, F2]. Resposta: Lisp [F1, F2].
- Q17 [graphrag] ✔ — Caminho no grafo: Princeton <-[:LOCATED_IN]- Universidade de Princeton <-[:STUDIED_AT]- Alonzo Church -[:INFLUENCED]-> Lisp [F2, F6, F3]. Caminho no grafo: Prin
- Q18 [graphrag] ✔ — Caminho no grafo: Ada Lovelace -[:INFLUENCED]-> Alan Turing -[:MET]-> Claude Shannon [F1, F2]. Resposta: Alan Turing [F1, F2].
- Q19 [graphrag] ✔ — Não encontrei evidências suficientes no grafo de conhecimento para responder com segurança.
- Q20 [graphrag] ✔ — Não encontrei evidências suficientes no grafo de conhecimento para responder com segurança. A pergunta menciona 'Prêmio Nobel', que não existe(m) no grafo — a p
- Q21 [graphrag] ✔ — Não encontrei evidências suficientes no grafo de conhecimento para responder com segurança.
- Q01 [kag] ✔ — Alan Turing nasceu em Londres (em 1912) [F1]. Resposta: Londres [F1].
- Q02 [kag] ✔ — Alonzo Church orientou Alan Turing (em 1938) [F1]. Resposta: Alonzo Church [F1].
- Q03 [kag] ✔ — Claude Shannon criou teoria da informação (em 1948) [F1]. Resposta: Claude Shannon [F1].
- Q04 [kag] ✔ — Unix foi desenvolvido(a) em Bell Labs (em 1969) [F1]. Resposta: Bell Labs [F1].
- Q05 [kag] ✔ — Charles Babbage colaborou com Ada Lovelace [F1]. Ada Lovelace colaborou com Charles Babbage [F2]. Resposta: Ada Lovelace [F1, F2].
- Q06 [kag] ✔ — linguagem B influenciou linguagem C [F1]. Resposta: linguagem B [F1].
- Q07 [kag] ✔ — Alonzo Church orientou Alan Turing (em 1938) [F1]. Alonzo Church orientou Stephen Kleene [F2]. Resposta: Alan Turing, Stephen Kleene [F1, F2].
- Q08 [kag] ✔ — Alonzo Church orientou Alan Turing (em 1938) [F1]. Alonzo Church trabalhou em Universidade de Princeton (em 1929) [F2]. Alonzo Church trabalhou em Universidade 
- Q09 [kag] ✔ — Dennis Ritchie criou linguagem C (em 1972) [F1]. Dennis Ritchie nasceu em Bronxville (em 1941) [F2]. Resposta: Bronxville (via Dennis Ritchie) [F2].
- Q10 [kag] ✔ — Prêmio Turing foi nomeado(a) em homenagem a Alan Turing [F1]. Alan Turing estudou em King's College (em 1934) [F2]. Alan Turing estudou em Universidade de Princ
- Q11 [kag] ✔ — Tim Berners-Lee criou World Wide Web (em 1989) [F1]. Tim Berners-Lee nasceu em Londres (em 1955) [F2]. Resposta: Londres (via Tim Berners-Lee) [F2].
- Q12 [kag] ✔ — John von Neumann nasceu em Budapeste (em 1903) [F1]. Budapeste fica em Hungria [F2]. Resposta: Hungria (via Budapeste) [F2].
- Q13 [kag] ✔ — Donald Knuth criou TeX (em 1978) [F1]. Donald Knuth trabalhou em Universidade Stanford [F2]. Resposta: Universidade Stanford (via Donald Knuth) [F2].
- Q14 [kag] ✔ — Tim Berners-Lee trabalhou em CERN [F1]. CERN fica em Genebra [F2]. Resposta: Genebra (via CERN) [F2].
- Q15 [kag] ✔ — Caminho no grafo: Bell Labs <-[:WORKED_AT]- Dennis Ritchie -[:RECEIVED]-> Prêmio Turing [F3, F4]. Caminho no grafo: Bell Labs <-[:WORKED_AT]- Ken Thompson -[:RE
- Q16 [kag] ✔ — Caminho no grafo: MIT <-[:DEVELOPED_AT]- Lisp <-[:INFLUENCED]- Alonzo Church [F1, F2]. Resposta: Lisp [F1, F2].
- Q17 [kag] ✔ — Caminho no grafo: Princeton <-[:LOCATED_IN]- Universidade de Princeton <-[:STUDIED_AT]- John McCarthy -[:CREATED]-> Lisp [F2, F3, F1]. Resposta: John McCarthy [
- Q18 [kag] ✔ — Caminho no grafo: Ada Lovelace -[:INFLUENCED]-> Alan Turing -[:MET]-> Claude Shannon [F1, F2]. Resposta: Alan Turing [F1, F2].
- Q19 [kag] ✔ — Não encontrei evidências suficientes no grafo de conhecimento para responder com segurança.
- Q20 [kag] ✔ — Não encontrei evidências suficientes no grafo de conhecimento para responder com segurança. A pergunta menciona 'Prêmio Nobel', que não existe(m) no grafo — a p
- Q21 [kag] ✔ — Não encontrei evidências suficientes no grafo de conhecimento para responder com segurança.