"""Ontologia (esquema LPG) do domínio "História da Computação".

Define rótulos de entidades, tipos de relacionamento com domínio/alcance e
verbalizações em português. É a fonte única de verdade usada por:
  * criação de constraints/índices no Neo4j (schema.py);
  * validação das triplas extraídas pelo LLM (modo de esquema estrito);
  * planejador lógico KAG (escolha de direção pela assinatura domínio/alcance);
  * validação de Cypher gerado pelo LLM (lista de rótulos/tipos permitidos).

O modo de esquema "dinâmico" permite registrar novos tipos de relação em
tempo de execução (ver ``Ontology.register_relation``).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

# Rótulos estruturais (não são entidades do domínio)
DOCUMENT = "Document"
CHUNK = "Chunk"
COMMUNITY = "Community"
ENTITY = "Entity"  # rótulo base compartilhado por todas as entidades

ENTITY_TYPES: dict[str, str] = {
    "Person": "Pessoa (cientista, engenheiro, inventor)",
    "Organization": "Universidade, laboratório, empresa ou instituição",
    "Location": "Cidade, estado ou país",
    "Technology": "Artefato técnico: linguagem, sistema, máquina, algoritmo",
    "Concept": "Teoria, modelo formal ou área do conhecimento",
    "Award": "Prêmio ou distinção",
}

STRUCTURAL_RELATIONS = {"HAS_CHUNK", "NEXT_CHUNK", "MENTIONS", "IN_COMMUNITY"}


@dataclass(frozen=True)
class RelationType:
    name: str
    domain: tuple[str, ...]
    range: tuple[str, ...]
    verb: str  # verbalização "{a} <verb> {b}"
    symmetric: bool = False
    description: str = ""


_ALL = tuple(ENTITY_TYPES)

DEFAULT_RELATIONS: list[RelationType] = [
    RelationType("BORN_IN", ("Person",), ("Location",), "nasceu em"),
    RelationType("STUDIED_AT", ("Person",), ("Organization",), "estudou em"),
    RelationType("WORKED_AT", ("Person",), ("Organization",), "trabalhou em"),
    RelationType("ADVISED", ("Person",), ("Person",), "orientou"),
    RelationType("CREATED", ("Person",), ("Technology", "Concept"), "criou"),
    RelationType("DEVELOPED_AT", ("Technology", "Concept"), ("Organization",), "foi desenvolvido(a) em"),
    RelationType("COLLABORATED_WITH", ("Person",), ("Person",), "colaborou com", symmetric=True),
    RelationType("MET", ("Person",), ("Person",), "encontrou-se com", symmetric=True),
    RelationType("INFLUENCED", _ALL, _ALL, "influenciou"),
    RelationType("RECEIVED", ("Person",), ("Award",), "recebeu"),
    RelationType("NAMED_AFTER", ("Award", "Technology", "Concept"), ("Person",), "foi nomeado(a) em homenagem a"),
    RelationType("AWARDED_BY", ("Award",), ("Organization",), "é concedido por"),
    RelationType("FOUNDED", ("Person",), ("Organization",), "fundou"),
    RelationType("LOCATED_IN", ("Organization", "Location"), ("Location",), "fica em"),
    RelationType("PART_OF", ("Organization",), ("Organization",), "faz parte de"),
]

_REL_NAME_RE = re.compile(r"^[A-Z][A-Z0-9_]{1,40}$")


@dataclass
class Ontology:
    entity_types: dict[str, str] = field(default_factory=lambda: dict(ENTITY_TYPES))
    relations: dict[str, RelationType] = field(
        default_factory=lambda: {r.name: r for r in DEFAULT_RELATIONS}
    )
    mode: str = "strict"  # "strict" | "dynamic"

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def sanitize_relation(name: str) -> str:
        name = re.sub(r"[^A-Za-z0-9]+", "_", name.strip()).strip("_").upper()
        return name

    def is_valid_relation_name(self, name: str) -> bool:
        return bool(_REL_NAME_RE.match(name)) and name not in STRUCTURAL_RELATIONS

    def register_relation(self, name: str, src_type: str, dst_type: str) -> RelationType | None:
        """Modo dinâmico: registra um tipo de relação desconhecido."""
        name = self.sanitize_relation(name)
        if not self.is_valid_relation_name(name):
            return None
        if name in self.relations:
            return self.relations[name]
        rel = RelationType(name, (src_type,), (dst_type,), name.lower().replace("_", " "))
        self.relations[name] = rel
        return rel

    def check_triple(self, src_type: str, rel: str, dst_type: str) -> tuple[bool, bool]:
        """Retorna (válida, precisa_inverter)."""
        r = self.relations.get(rel)
        if r is None:
            return False, False
        if src_type in r.domain and dst_type in r.range:
            return True, False
        if r.symmetric or (dst_type in r.domain and src_type in r.range):
            return True, not r.symmetric
        return False, False

    def directions_for(self, rel: str, current_type: str | None) -> list[str]:
        """Direções possíveis para sair de um nó do tipo ``current_type``."""
        r = self.relations.get(rel)
        if r is None:
            return []
        if r.symmetric:
            return ["any"]
        dirs = []
        if current_type is None or current_type in r.domain:
            dirs.append("out")
        if current_type is None or current_type in r.range:
            dirs.append("in")
        return dirs

    def verbalize(self, src: str, rel: str, dst: str, props: dict | None = None) -> str:
        r = self.relations.get(rel)
        verb = r.verb if r else rel.lower().replace("_", " ")
        s = f"{src} {verb} {dst}"
        year = (props or {}).get("year")
        if year:
            s += f" (em {year})"
        return s

    def entity_relation_types(self) -> list[str]:
        return sorted(self.relations)

    def describe(self) -> str:
        """Descrição textual do esquema para prompts (extração, Text2Cypher, KAG)."""
        lines = ["Rótulos de entidade (todos também possuem o rótulo :Entity):"]
        for k, v in self.entity_types.items():
            lines.append(f"  - {k}: {v}")
        lines.append("Tipos de relacionamento (origem -> destino):")
        for r in self.relations.values():
            sym = " [simétrica]" if r.symmetric else ""
            lines.append(f"  - (:{'|'.join(r.domain)})-[:{r.name}]->(:{'|'.join(r.range)}){sym}  // {r.verb}")
        lines.append(
            "Propriedades: Entity{id, name, type, aliases, pagerank, community}; "
            "relações de domínio têm {sources: [chunk_id], year, confidence, mention_count}; "
            "Chunk{id, text, doc_id, idx}; Document{id, title, source}. "
            "(:Document)-[:HAS_CHUNK]->(:Chunk)-[:MENTIONS]->(:Entity); "
            "(:Chunk)-[:NEXT_CHUNK]->(:Chunk); (:Entity)-[:IN_COMMUNITY]->(:Community{id, summary})."
        )
        return "\n".join(lines)
