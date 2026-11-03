"""Text2Cypher: geração de Cypher por LLM com guarda de segurança.

Camadas de proteção (defesa em profundidade):
1. Prompt com o esquema exato (rótulos, tipos e propriedades permitidos).
2. Validador estático: somente leitura (bloqueia CREATE/MERGE/DELETE/SET/
   REMOVE/DROP/LOAD CSV/CALL dbms|apoc|gds...), rótulos e tipos de relação
   precisam existir na ontologia, LIMIT obrigatório (injetado se ausente).
3. ``EXPLAIN`` no Neo4j antes de executar (erro de sintaxe => autocorreção
   com até 1 nova tentativa enviando o erro ao LLM).
4. Execução em transação de LEITURA (``execute_read``).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .llm import LLMClient, LLMError
from .ontology import CHUNK, COMMUNITY, DOCUMENT, ENTITY, STRUCTURAL_RELATIONS, Ontology

WRITE_CLAUSES = re.compile(
    r"\b(CREATE|MERGE|DELETE|DETACH|SET|REMOVE|DROP|LOAD\s+CSV|FOREACH|GRANT|REVOKE|DENY|ALTER|RENAME|TERMINATE|START|STOP)\b",
    re.I,
)
FORBIDDEN_CALLS = re.compile(r"\bCALL\s+(dbms|apoc|gds|db\.(create|drop|clear))", re.I)
NODE_LABEL_RE = re.compile(r"\(\s*\w*\s*:\s*`?([A-Za-z_][A-Za-z0-9_]*)`?")
REL_TYPE_RE = re.compile(r"\[\s*\w*\s*:\s*`?([A-Za-z_][A-Za-z0-9_|`:]*)`?")
LIMIT_RE = re.compile(r"\bLIMIT\s+(\d+)\s*;?\s*$", re.I)


@dataclass
class CypherCheck:
    ok: bool
    cypher: str
    errors: list[str] = field(default_factory=list)


def validate_cypher(cypher: str, ontology: Ontology, max_limit: int = 50) -> CypherCheck:
    c = cypher.strip().rstrip(";").strip()
    c = re.sub(r"^```(cypher)?|```$", "", c, flags=re.I).strip()
    errors = []
    if ";" in c:
        errors.append("Múltiplas instruções não são permitidas.")
    if WRITE_CLAUSES.search(_strip_strings(c)):
        errors.append("Consulta não é somente leitura.")
    if FORBIDDEN_CALLS.search(c):
        errors.append("Procedimento não permitido.")
    allowed_labels = set(ontology.entity_types) | {ENTITY, CHUNK, DOCUMENT, COMMUNITY}
    for lab in NODE_LABEL_RE.findall(c):
        if lab not in allowed_labels:
            errors.append(f"Rótulo desconhecido: {lab}")
    allowed_rels = set(ontology.relations) | STRUCTURAL_RELATIONS
    for grp in REL_TYPE_RE.findall(c):
        for t in re.split(r"[|:`]+", grp):
            if t and t not in allowed_rels:
                errors.append(f"Tipo de relação desconhecido: {t}")
    if not re.search(r"\bRETURN\b", c, re.I):
        errors.append("Consulta sem RETURN.")
    m = LIMIT_RE.search(c)
    if not m:
        c = f"{c}\nLIMIT {max_limit}"
    elif int(m.group(1)) > max_limit:
        c = LIMIT_RE.sub(f"LIMIT {max_limit}", c)
    return CypherCheck(not errors, c, errors)


def _strip_strings(c: str) -> str:
    return re.sub(r"'[^']*'|\"[^\"]*\"", "''", c)


T2C_SYSTEM = """Você traduz perguntas em português para UMA consulta Cypher (Neo4j 5) SOMENTE LEITURA.
Use apenas este esquema:
{schema}
Regras:
- Compare nomes com e.name (valores exatos fornecidos em "Entidades da pergunta").
- Retorne colunas com nomes claros e inclua os nomes das entidades usadas no caminho.
- Prefira padrões com caminhos explícitos para perguntas multi-hop. Use LIMIT.
- Responda apenas com o Cypher, sem explicações."""


@dataclass
class Text2CypherResult:
    cypher: str = ""
    rows: list[dict] = field(default_factory=list)
    attempts: list[dict] = field(default_factory=list)
    ok: bool = False
    error: str | None = None


class Text2Cypher:
    def __init__(self, llm: LLMClient, ontology: Ontology, store):
        self.llm, self.o, self.store = llm, ontology, store

    @property
    def available(self) -> bool:
        return self.llm.enabled and self.store.backend == "neo4j"

    def run(self, question: str, anchors: list) -> Text2CypherResult:
        res = Text2CypherResult()
        if not self.available:
            res.error = "Text2Cypher requer LLM configurado e backend Neo4j."
            return res
        ents = "; ".join(f'"{a.name}" ({a.type})' for a in anchors) or "nenhuma"
        user = f"Entidades da pergunta: {ents}\nPergunta: {question}"
        feedback = ""
        for attempt in range(2):
            try:
                raw = self.llm.chat(T2C_SYSTEM.format(schema=self.o.describe()), user + feedback)
            except LLMError as e:
                res.error = str(e)
                return res
            chk = validate_cypher(raw, self.o)
            step = {"raw": raw, "cypher": chk.cypher, "errors": list(chk.errors)}
            res.attempts.append(step)
            if not chk.ok:
                feedback = f"\nSua consulta anterior foi rejeitada: {chk.errors}. Corrija."
                continue
            try:
                self.store.explain(chk.cypher)
                res.rows = self.store.readonly_cypher(chk.cypher)
                res.cypher, res.ok = chk.cypher, True
                return res
            except Exception as e:  # erro de sintaxe/semântica do Neo4j
                step["errors"].append(str(e)[:300])
                feedback = f"\nO Neo4j retornou erro: {str(e)[:300]}. Corrija a consulta."
        res.error = "Consulta Cypher inválida após autocorreção."
        return res
