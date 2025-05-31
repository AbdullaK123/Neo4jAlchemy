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
    
    # Engine
    "GraphEngine",
    "create_graph_engine",
    
    # Version
    "__version__",
]


def hello() -> str:
    """Legacy hello function - kept for compatibility."""
    return "Hello from neo4jalchemy!"