# src/neo4jalchemy/__init__.py
r"""
Neo4jAlchemy - Modern Graph Database ORM for Python

Neo4jAlchemy brings SQLAlchemy-style elegance to Neo4j development with:
- Type-safe GraphEntity models with Pydantic validation
- Rich GraphRelationship system with automatic edge synchronization
- SQLAlchemy-style relationship navigation and querying
- Rich graph algorithms and analysis
- Automatic schema migrations  
- FastAPI integration
- ML framework exports (PyTorch Geometric, DGL)

Example:
    ```python
    from neo4jalchemy import Graph, GraphEntity, GraphRelationship, graph_entity, graph_relationship
    
    # Create graph
    graph = Graph(name="social_network")
    
    # Define entities with type safety
    @graph_entity(label="User", graph=graph)
    class User(GraphEntity):
        name: str = Field(min_length=1)
        email: str = Field(pattern=r'^[^@]+@[^@]+\.[^@]+$')
        age: int = Field(ge=0, le=150)
    
    @graph_entity(label="Project", graph=graph) 
    class Project(GraphEntity):
        title: str = Field(min_length=1)
        status: Literal["planning", "active", "completed"] = Field(default="planning")
    
    # Define relationships with rich properties
    @graph_relationship(from_entity=User, to_entity=Project)
    class WorksOn(GraphRelationship):
        role: Literal["developer", "lead", "architect"] = Field(...)
        since: datetime = Field(default_factory=datetime.now)
        hours_per_week: int = Field(ge=1, le=60)
        
        @field_validator('hours_per_week')
        @classmethod
        def validate_lead_hours(cls, v, info):
            if info.data.get('role') == 'lead' and v < 20:
                raise ValueError('Lead roles require at least 20 hours/week')
            return v
    
    # Create entities and relationships
    alice = await User.create(name="Alice Johnson", email="alice@example.com", age=30)
    project = await Project.create(title="AI Platform", status="active")
    
    # Automatic edge creation via metaclass magic
    works_on = await WorksOn.create(
        from_node=alice,
        to_node=project,
        role="lead",
        hours_per_week=40
    )
    
    # SQLAlchemy-style navigation
    alice_projects = await alice.works_on.filter(role="lead").target.all()
    project_team = await project.worked_on_by.source.all()
    ```
"""

# Core graph functionality
from neo4jalchemy.core.graph import Graph
from neo4jalchemy.core.graph_node import GraphNode
from neo4jalchemy.core.graph_edge import GraphEdge

# ORM system - Entities
from neo4jalchemy.orm.entities import GraphEntity, graph_entity

# ORM system - Relationships (NEW!)
from neo4jalchemy.orm.relationships import GraphRelationship, graph_relationship
from neo4jalchemy.orm.relationship_managers import RelationshipManager, relationship

# Engine for Neo4j connections
from neo4jalchemy.orm.engine import GraphEngine, create_graph_engine

# Version info
__version__ = "0.2.0"  # Updated for relationship system
__author__ = "AbdullaK123"

# Main exports
__all__ = [
    # Core classes
    "Graph",
    "GraphNode", 
    "GraphEdge",
    
    # ORM classes - Entities
    "GraphEntity",
    "graph_entity",
    
    # ORM classes - Relationships (NEW!)
    "GraphRelationship",
    "graph_relationship",
    "RelationshipManager", 
    "relationship",
    
    # Engine
    "GraphEngine",
    "create_graph_engine",
    
    # Version
    "__version__",
]


def hello() -> str:
    """Legacy hello function - kept for compatibility."""
    return "Hello from neo4jalchemy with GraphRelationship system!"