# src/neo4jalchemy/orm/__init__.py
"""
Neo4jAlchemy ORM Module

This module provides the Object-Relational Mapping (ORM) layer for Neo4j,
including entity management, relationships, and database integration.
"""

from neo4jalchemy.orm.entities import (
    GraphEntity, 
    graph_entity, 
    get_entity_classes, 
    get_entity_by_label
)


from neo4jalchemy.orm.engine import (
    GraphEngine,
    create_graph_engine
)

__all__ = [
    # Entity system
    "GraphEntity",
    "graph_entity", 
    "get_entity_classes",
    "get_entity_by_label",
    
    # Engine
    "GraphEngine",
    "create_graph_engine",
]
