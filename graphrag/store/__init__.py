from .base import GraphPath, GraphStore, StoreStats
from .memory_store import MemoryGraphStore


def get_store(settings, ontology) -> GraphStore:
    if settings.graph_backend.lower() == "memory":
        return MemoryGraphStore(ontology)
    from .neo4j_store import Neo4jGraphStore

    return Neo4jGraphStore(ontology, settings.neo4j_uri, settings.neo4j_user,
                           settings.neo4j_password, settings.neo4j_database)


__all__ = ["GraphPath", "GraphStore", "StoreStats", "MemoryGraphStore", "get_store"]
