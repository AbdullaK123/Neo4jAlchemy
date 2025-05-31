# src/neo4jalchemy/__init__.py
r"""
Neo4jAlchemy - Modern Graph Database ORM for Python

Neo4jAlchemy brings SQLAlchemy-style elegance to Neo4j development with:
- Type-safe GraphEntity models with Pydantic validation
- Rich graph algorithms and analysis
- Automatic schema migrations  
- FastAPI integration
- ML framework exports (PyTorch Geometric, DGL)

Example:
    ```python
    from neo4jalchemy import Graph, GraphEntity, graph_entity
    
    @graph_entity(label="User")
    class User(GraphEntity):
        name: str = Field(min_length=1)
        email: str = Field(pattern=r'^[^@]+@[^@]+\.[^@]+$')
        age: int = Field(ge=0, le=150)
    
    # Create graph and entities
    graph = Graph(name="social_network")
    User.Config.graph = graph
    
    user = await User.create(
        name="Alice Johnson", 
        email="alice@example.com",
        age=30
    )
    ```
"""

# Core graph functionality
from neo4jalchemy.core.graph import Graph
from neo4jalchemy.core.graph_node import GraphNode
from neo4jalchemy.core.graph_edge import GraphEdge

# ORM system
from neo4jalchemy.orm.entities import GraphEntity, graph_entity
from neo4jalchemy.orm.fields import (
    StringField, IntegerField, FloatField, BooleanField,
    DateTimeField, ListField, DictField, ReferenceField
)

# Engine for Neo4j connections (when ready)
from neo4jalchemy.orm.engine import GraphEngine, create_graph_engine

# Version info
__version__ = "0.1.0"
__author__ = "AbdullaK123"

# Main exports
__all__ = [
    # Core classes
    "Graph",
    "GraphNode", 
    "GraphEdge",
    
    # ORM classes
    "GraphEntity",
    "graph_entity",
    
    # Field types
    "StringField",
    "IntegerField", 
    "FloatField",
    "BooleanField",
    "DateTimeField",
    "ListField",
    "DictField",
    "ReferenceField",
    
    # Engine
    "GraphEngine",
    "create_graph_engine",
    
    # Version
    "__version__",
]


def hello() -> str:
    """Legacy hello function - kept for compatibility."""
    return "Hello from neo4jalchemy!"


# =============================================================================
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

from neo4jalchemy.orm.fields import (
    Field,
    StringField,
    IntegerField, 
    FloatField,
    BooleanField,
    DateTimeField,
    ListField,
    DictField,
    ReferenceField,
    get_fields,
    validate_instance,
    get_dirty_fields,
    reset_instance_tracking
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
    
    # Field system
    "Field",
    "StringField",
    "IntegerField",
    "FloatField", 
    "BooleanField",
    "DateTimeField",
    "ListField",
    "DictField",
    "ReferenceField",
    "get_fields",
    "validate_instance",
    "get_dirty_fields",
    "reset_instance_tracking",
    
    # Engine
    "GraphEngine",
    "create_graph_engine",
]


# =============================================================================
# src/neo4jalchemy/core/__init__.py
"""
Neo4jAlchemy Core Module

This module provides the foundational graph data structures and algorithms
that power the entire Neo4jAlchemy ecosystem.
"""

from neo4jalchemy.core.graph import Graph
from neo4jalchemy.core.graph_node import GraphNode
from neo4jalchemy.core.graph_edge import GraphEdge

__all__ = [
    "Graph",
    "GraphNode", 
    "GraphEdge",
]