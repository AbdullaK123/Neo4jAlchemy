"""
Neo4jAlchemy Core Graph Implementation

This module contains the core Graph class that provides an in-memory graph
data structure with rich algorithms and analysis capabilities.
"""

from typing import (
    Dict, 
    List, 
    Set, 
    Optional, 
    Any, 
    Tuple, 
    Iterator,
    Union,
    Callable,
    DefaultDict,
)
from collections import defaultdict, deque
from datetime import datetime
import uuid
import json

from pydantic import BaseModel, Field, field_validator, model_validator, ConfigDict


class GraphNode(BaseModel):
    """
    Represents a node in the graph with properties and metadata.
    
    Uses Pydantic for automatic validation, serialization, and type coercion.
    """
    
    id: str = Field(..., min_length=1, description="Unique node identifier")
    label: str = Field(..., min_length=1, description="Node type/label")
    properties: Dict[str, Any] = Field(
        default_factory=dict, 
        description="Node properties and data"
    )
    created_at: datetime = Field(
        default_factory=datetime.now,
        description="When the node was created"
    )
    updated_at: datetime = Field(
        default_factory=datetime.now,
        description="When the node was last updated"
    )
    
    model_config = ConfigDict(
        # Validate on assignment 
        validate_assignment=True,
        # Use enum values for serialization
        use_enum_values=True,
        # JSON schema extras
        json_schema_extra={
            "example": {
                "id": "user_123",
                "label": "User",
                "properties": {
                    "name": "Alice Johnson",
                    "age": 30,
                    "active": True
                }
            }
        }
    )
    
    @field_validator('id', mode='before')
    @classmethod
    def coerce_id_to_string(cls, v):
        """Ensure ID is always a string."""
        if v is None:
            return str(uuid.uuid4())
        return str(v)
    
    @field_validator('label')
    @classmethod 
    def validate_label(cls, v):
        """Ensure label is not empty after stripping."""
        if not v or not v.strip():
            raise ValueError("Label cannot be empty")
        return v.strip()
    
    @field_validator('properties')
    @classmethod
    def validate_properties(cls, v):
        """Validate properties dictionary."""
        if v is None:
            return {}
        
        # Ensure all keys are strings
        if not all(isinstance(k, str) for k in v.keys()):
            raise ValueError("All property keys must be strings")
        
        # Could add more validation here (e.g., no None values, size limits)
        return v
    
    def update_properties(self, new_properties: Dict[str, Any]) -> None:
        """
        Update node properties and timestamp.
        
        Args:
            new_properties: Properties to update/add
        """
        self.properties.update(new_properties)
        self.updated_at = datetime.now()
    
    def get_property(self, key: str, default: Any = None) -> Any:
        """
        Get a specific property value.
        
        Args:
            key: Property key to retrieve
            default: Default value if key not found
            
        Returns:
            Property value or default
        """
        return self.properties.get(key, default)
    
    def set_property(self, key: str, value: Any) -> None:
        """
        Set a single property value.
        
        Args:
            key: Property key
            value: Property value
        """
        self.properties[key] = value
        self.updated_at = datetime.now()
    
    def remove_property(self, key: str) -> Any:
        """
        Remove a property and return its value.
        
        Args:
            key: Property key to remove
            
        Returns:
            Removed property value or None
        """
        if key in self.properties:
            value = self.properties.pop(key)
            self.updated_at = datetime.now()
            return value
        return None
    
    @property
    def property_count(self) -> int:
        """Get number of properties."""
        return len(self.properties)
    
    def has_property(self, key: str) -> bool:
        """Check if property exists."""
        return key in self.properties


class GraphEdge(BaseModel):
    """
    Represents an edge/relationship in the graph.
    
    Uses Pydantic for automatic validation, serialization, and type coercion.
    """
    
    from_id: str = Field(..., min_length=1, description="Source node ID")
    to_id: str = Field(..., min_length=1, description="Target node ID")
    relationship_type: str = Field(
        ..., 
        min_length=1, 
        description="Type of relationship"
    )
    properties: Dict[str, Any] = Field(
        default_factory=dict,
        description="Edge properties and metadata"
    )
    weight: float = Field(
        default=1.0, 
        ge=0.0, 
        description="Edge weight (must be non-negative)"
    )
    directed: bool = Field(
        default=True,
        description="Whether the edge is directed"
    )
    created_at: datetime = Field(
        default_factory=datetime.now,
        description="When the edge was created"
    )
    
    model_config = ConfigDict(
        validate_assignment=True,
        use_enum_values=True,
        json_schema_extra={
            "example": {
                "from_id": "user_123",
                "to_id": "user_456", 
                "relationship_type": "FOLLOWS",
                "properties": {
                    "since": "2024-01-15",
                    "strength": 0.8
                },
                "weight": 0.8,
                "directed": True
            }
        }
    )
    
    @field_validator('from_id', 'to_id', mode='before')
    @classmethod
    def coerce_ids_to_string(cls, v):
        """Ensure IDs are always strings."""
        return str(v)
    
    @field_validator('relationship_type')
    @classmethod
    def validate_relationship_type(cls, v):
        """Validate relationship type."""
        if not v or not v.strip():
            raise ValueError("Relationship type cannot be empty")
        # Convert to uppercase convention
        return v.strip().upper()
    
    @field_validator('properties')
    @classmethod
    def validate_properties(cls, v):
        """Validate properties dictionary."""
        if v is None:
            return {}
        
        # Ensure all keys are strings
        if not all(isinstance(k, str) for k in v.keys()):
            raise ValueError("All property keys must be strings")
        
        return v
    
    @model_validator(mode='after')
    def validate_different_nodes(self):
        """Ensure from_id and to_id are different (optional constraint)."""
        # Allow self-loops but could be configurable
        # if self.from_id == self.to_id:
        #     raise ValueError("Self-loops not allowed")
        
        return self
    
    @property
    def id(self) -> str:
        """Generate unique edge identifier."""
        direction = "->" if self.directed else "<->"
        return f"{self.from_id}{direction}[{self.relationship_type}]{direction}{self.to_id}"
    
    def reverse(self) -> 'GraphEdge':
        """
        Create reverse edge for undirected relationships.
        
        Returns:
            New GraphEdge with reversed direction
        """
        return GraphEdge(
            from_id=self.to_id,
            to_id=self.from_id,
            relationship_type=self.relationship_type,
            properties=self.properties.copy(),
            weight=self.weight,
            directed=self.directed,
            created_at=self.created_at,
        )
    
    def update_properties(self, new_properties: Dict[str, Any]) -> None:
        """
        Update edge properties.
        
        Args:
            new_properties: Properties to update/add
        """
        self.properties.update(new_properties)
    
    def get_property(self, key: str, default: Any = None) -> Any:
        """
        Get a specific property value.
        
        Args:
            key: Property key to retrieve
            default: Default value if key not found
            
        Returns:
            Property value or default
        """
        return self.properties.get(key, default)
    
    def set_property(self, key: str, value: Any) -> None:
        """
        Set a single property value.
        
        Args:
            key: Property key  
            value: Property value
        """
        self.properties[key] = value
    
    @property
    def property_count(self) -> int:
        """Get number of properties."""
        return len(self.properties)
    
    @property
    def is_self_loop(self) -> bool:
        """Check if edge is a self-loop."""
        return self.from_id == self.to_id


class Graph:
    """
    In-memory graph data structure with rich algorithms and analysis.
    
    This is the core of Neo4jAlchemy - a powerful graph that can work
    independently or be persisted to Neo4j later.
    """
    
    def __init__(self, name: Optional[str] = None):
        """Initialize empty graph."""
        self.name = name or f"graph_{uuid.uuid4().hex[:8]}"
        self.created_at = datetime.now()
        
        # Core storage
        self._nodes: Dict[str, GraphNode] = {}
        self._edges: Dict[str, GraphEdge] = {}
        
        # Adjacency lists for fast lookups
        self._outgoing: DefaultDict[str, Dict[str, List[GraphEdge]]] = defaultdict(lambda: defaultdict(list))
        self._incoming: DefaultDict[str, Dict[str, List[GraphEdge]]] = defaultdict(lambda: defaultdict(list))
        
        # Statistics cache
        self._stats_cache: Dict[str, Any] = {}
        self._cache_dirty = True
    
    # =============================================================================
    # BASIC GRAPH OPERATIONS
    # =============================================================================
    
    def add_node(
        self, 
        node_id: str, 
        label: str, 
        properties: Optional[Dict[str, Any]] = None
    ) -> GraphNode:
        """Add a node to the graph."""
        node_id = str(node_id)
        
        if node_id in self._nodes:
            # Update existing node
            existing = self._nodes[node_id]
            if properties:
                existing.update_properties(properties)
            return existing
        
        # Create new node
        node = GraphNode(
            id=node_id,
            label=label,
            properties=properties or {}
        )
        
        self._nodes[node_id] = node
        self._invalidate_cache()
        return node
    
    def add_edge(
        self,
        from_id: str,
        to_id: str,
        relationship_type: str,
        properties: Optional[Dict[str, Any]] = None,
        weight: float = 1.0,
        directed: bool = True
    ) -> GraphEdge:
        """Add an edge to the graph."""
        from_id, to_id = str(from_id), str(to_id)
        
        # Ensure nodes exist
        if from_id not in self._nodes:
            raise ValueError(f"Node {from_id} does not exist")
        if to_id not in self._nodes:
            raise ValueError(f"Node {to_id} does not exist")
        
        # Create edge
        edge = GraphEdge(
            from_id=from_id,
            to_id=to_id,
            relationship_type=relationship_type,
            properties=properties or {},
            weight=weight,
            directed=directed
        )
        
        # Store edge
        self._edges[edge.id] = edge
        
        # Update adjacency lists
        self._outgoing[from_id][relationship_type].append(edge)
        self._incoming[to_id][relationship_type].append(edge)
        
        # For undirected edges, add reverse direction
        if not directed:
            reverse_edge = edge.reverse()
            self._outgoing[to_id][relationship_type].append(reverse_edge)
            self._incoming[from_id][relationship_type].append(reverse_edge)
        
        self._invalidate_cache()
        return edge
    
    def remove_node(self, node_id: str) -> bool:
        """Remove a node and all its edges."""
        node_id = str(node_id)
        
        if node_id not in self._nodes:
            return False
        
        # Remove all edges connected to this node
        edges_to_remove = []
        for edge_id, edge in self._edges.items():
            if edge.from_id == node_id or edge.to_id == node_id:
                edges_to_remove.append(edge_id)
        
        for edge_id in edges_to_remove:
            self.remove_edge(edge_id)
        
        # Remove node
        del self._nodes[node_id]
        
        # Clean up adjacency lists
        if node_id in self._outgoing:
            del self._outgoing[node_id]
        if node_id in self._incoming:
            del self._incoming[node_id]
        
        self._invalidate_cache()
        return True
    
    def remove_edge(self, edge_id: str) -> bool:
        """Remove an edge by ID."""
        if edge_id not in self._edges:
            return False
        
        edge = self._edges[edge_id]
        
        # Remove from adjacency lists
        self._outgoing[edge.from_id][edge.relationship_type] = [
            e for e in self._outgoing[edge.from_id][edge.relationship_type] 
            if e.id != edge_id
        ]
        
        self._incoming[edge.to_id][edge.relationship_type] = [
            e for e in self._incoming[edge.to_id][edge.relationship_type] 
            if e.id != edge_id
        ]
        
        # Remove edge
        del self._edges[edge_id]
        
        self._invalidate_cache()
        return True
    
    # =============================================================================
    # GRAPH QUERIES
    # =============================================================================
    
    def get_node(self, node_id: str) -> Optional[GraphNode]:
        """Get a node by ID."""
        return self._nodes.get(str(node_id))
    
    def get_edge(self, edge_id: str) -> Optional[GraphEdge]:
        """Get an edge by ID."""
        return self._edges.get(edge_id)
    
    def has_node(self, node_id: str) -> bool:
        """Check if node exists."""
        return str(node_id) in self._nodes
    
    def has_edge(self, from_id: str, to_id: str, relationship_type: Optional[str] = None) -> bool:
        """Check if edge exists between two nodes."""
        from_id, to_id = str(from_id), str(to_id)
        
        if from_id not in self._outgoing:
            return False
        
        if relationship_type:
            return any(
                edge.to_id == to_id 
                for edge in self._outgoing[from_id].get(relationship_type, [])
            )
        else:
            return any(
                edge.to_id == to_id 
                for edges_list in self._outgoing[from_id].values()
                for edge in edges_list
            )
    
    def neighbors(
        self, 
        node_id: str, 
        relationship_type: Optional[str] = None,
        direction: str = "outgoing"
    ) -> List[str]:
        """Get neighboring node IDs."""
        node_id = str(node_id)
        
        if direction == "outgoing":
            adjacency = self._outgoing
        elif direction == "incoming":
            adjacency = self._incoming
        else:
            raise ValueError("Direction must be 'outgoing' or 'incoming'")
        
        if node_id not in adjacency:
            return []
        
        neighbors = []
        
        if relationship_type:
            edges = adjacency[node_id].get(relationship_type, [])
            neighbors = [
                edge.to_id if direction == "outgoing" else edge.from_id 
                for edge in edges
            ]
        else:
            for edges_list in adjacency[node_id].values():
                neighbors.extend([
                    edge.to_id if direction == "outgoing" else edge.from_id 
                    for edge in edges_list
                ])
        
        return list(set(neighbors))  # Remove duplicates
    
    def get_edges_between(self, from_id: str, to_id: str) -> List[GraphEdge]:
        """Get all edges between two nodes."""
        from_id, to_id = str(from_id), str(to_id)
        edges = []
        
        if from_id in self._outgoing:
            for edges_list in self._outgoing[from_id].values():
                edges.extend([edge for edge in edges_list if edge.to_id == to_id])
        
        return edges
    
    # =============================================================================
    # GRAPH ALGORITHMS
    # =============================================================================
    
    def bfs(self, start_id: str, max_depth: Optional[int] = None) -> Dict[str, int]:
        """Breadth-first search from start node. Returns node_id -> depth mapping."""
        start_id = str(start_id)
        
        if start_id not in self._nodes:
            return {}
        
        visited = {start_id: 0}
        queue = deque([(start_id, 0)])
        
        while queue:
            current_id, depth = queue.popleft()
            
            if max_depth is not None and depth >= max_depth:
                continue
            
            for neighbor_id in self.neighbors(current_id):
                if neighbor_id not in visited:
                    visited[neighbor_id] = depth + 1
                    queue.append((neighbor_id, depth + 1))
        
        return visited
    
    def dfs(self, start_id: str, max_depth: Optional[int] = None) -> List[str]:
        """Depth-first search from start node. Returns list of visited nodes."""
        start_id = str(start_id)
        
        if start_id not in self._nodes:
            return []
        
        visited = set()
        result = []
        
        def _dfs_recursive(node_id: str, depth: int):
            if node_id in visited:
                return
            if max_depth is not None and depth > max_depth:
                return
            
            visited.add(node_id)
            result.append(node_id)
            
            for neighbor_id in self.neighbors(node_id):
                _dfs_recursive(neighbor_id, depth + 1)
        
        _dfs_recursive(start_id, 0)
        return result
    
    def shortest_path(self, from_id: str, to_id: str) -> Optional[List[str]]:
        """Find shortest path between two nodes using BFS."""
        from_id, to_id = str(from_id), str(to_id)
        
        if from_id not in self._nodes or to_id not in self._nodes:
            return None
        
        if from_id == to_id:
            return [from_id]
        
        queue = deque([(from_id, [from_id])])
        visited = {from_id}
        
        while queue:
            current_id, path = queue.popleft()
            
            for neighbor_id in self.neighbors(current_id):
                if neighbor_id == to_id:
                    return path + [neighbor_id]
                
                if neighbor_id not in visited:
                    visited.add(neighbor_id)
                    queue.append((neighbor_id, path + [neighbor_id]))
        
        return None  # No path found
    
    def connected_components(self) -> List[List[str]]:
        """Find all connected components in the graph."""
        visited = set()
        components = []
        
        for node_id in self._nodes:
            if node_id not in visited:
                component = []
                stack = [node_id]
                
                while stack:
                    current = stack.pop()
                    if current not in visited:
                        visited.add(current)
                        component.append(current)
                        
                        # Add both outgoing and incoming neighbors
                        neighbors = set(self.neighbors(current, direction="outgoing"))
                        neighbors.update(self.neighbors(current, direction="incoming"))
                        
                        for neighbor in neighbors:
                            if neighbor not in visited:
                                stack.append(neighbor)
                
                if component:
                    components.append(component)
        
        return components
    
    def degree_centrality(self, node_id: str) -> float:
        """Calculate degree centrality for a node."""
        node_id = str(node_id)
        
        if node_id not in self._nodes:
            return 0.0
        
        if len(self._nodes) <= 1:
            return 0.0
        
        degree = len(set(
            self.neighbors(node_id, direction="outgoing") + 
            self.neighbors(node_id, direction="incoming")
        ))
        
        return degree / (len(self._nodes) - 1)
    
    # =============================================================================
    # GRAPH STATISTICS
    # =============================================================================
    
    def node_count(self) -> int:
        """Get total number of nodes."""
        return len(self._nodes)
    
    def edge_count(self) -> int:
        """Get total number of edges."""
        return len(self._edges)
    
    def density(self) -> float:
        """Calculate graph density."""
        n = self.node_count()
        if n <= 1:
            return 0.0
        
        max_edges = n * (n - 1)  # For directed graph
        return self.edge_count() / max_edges
    
    def average_degree(self) -> float:
        """Calculate average node degree."""
        if not self._nodes:
            return 0.0
        
        total_degree = sum(
            len(set(
                self.neighbors(node_id, direction="outgoing") + 
                self.neighbors(node_id, direction="incoming")
            ))
            for node_id in self._nodes
        )
        
        return total_degree / len(self._nodes)
    
    # =============================================================================
    # SERIALIZATION
    # =============================================================================
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert graph to dictionary representation."""
        return {
            "name": self.name,
            "created_at": self.created_at.isoformat(), 
            "nodes": [node.model_dump() for node in self._nodes.values()],
            "edges": [edge.model_dump() for edge in self._edges.values()],
            "statistics": {
                "node_count": self.node_count(),
                "edge_count": self.edge_count(),
                "density": self.density(),
                "average_degree": self.average_degree(),
            }
        }
    
    def to_json(self, indent: Optional[int] = None) -> str:
        """Convert graph to JSON string."""
        return json.dumps(self.to_dict(), indent=indent, default=str)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Graph':
        """Create graph from dictionary representation."""
        graph = cls(name=data.get("name"))
        
        # Add nodes
        for node_data in data.get("nodes", []):
            # Use Pydantic model parsing (updated for V2)
            node = GraphNode.model_validate(node_data)
            graph._nodes[node.id] = node
        
        # Add edges  
        for edge_data in data.get("edges", []):
            # Use Pydantic model parsing (updated for V2)
            edge = GraphEdge.model_validate(edge_data)
            graph._edges[edge.id] = edge
            
            # Update adjacency lists
            graph._outgoing[edge.from_id][edge.relationship_type].append(edge)
            graph._incoming[edge.to_id][edge.relationship_type].append(edge)
            
            # For undirected edges, add reverse direction
            if not edge.directed:
                reverse_edge = edge.reverse()
                graph._outgoing[edge.to_id][edge.relationship_type].append(reverse_edge)
                graph._incoming[edge.from_id][edge.relationship_type].append(reverse_edge)
        
        return graph
    
    @classmethod
    def from_json(cls, json_str: str) -> 'Graph':
        """Create graph from JSON string."""
        return cls.from_dict(json.loads(json_str))
    
    # =============================================================================
    # UTILITIES
    # =============================================================================
    
    def _invalidate_cache(self):
        """Invalidate statistics cache."""
        self._cache_dirty = True
        self._stats_cache.clear()
    
    def copy(self) -> 'Graph':
        """Create a deep copy of the graph."""
        return Graph.from_dict(self.to_dict())
    
    def clear(self):
        """Remove all nodes and edges."""
        self._nodes.clear()
        self._edges.clear()
        self._outgoing.clear()
        self._incoming.clear()
        self._invalidate_cache()
    
    def __len__(self) -> int:
        """Return number of nodes."""
        return len(self._nodes)
    
    def __contains__(self, node_id: str) -> bool:
        """Check if node exists in graph."""
        return str(node_id) in self._nodes
    
    def __iter__(self) -> Iterator[str]:
        """Iterate over node IDs."""
        return iter(self._nodes.keys())
    
    def __repr__(self) -> str:
        """String representation of graph."""
        return f"Graph(name='{self.name}', nodes={self.node_count()}, edges={self.edge_count()})"