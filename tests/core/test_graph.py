"""
Comprehensive tests for the core Graph class.

These tests verify all functionality of the Graph data structure including:
- Basic operations (add/remove nodes/edges)
- Graph queries and traversal
- Graph algorithms (BFS, DFS, shortest path, etc.)
- Statistics and analysis
- Serialization/deserialization

Updated for Pydantic-based GraphNode and GraphEdge models.
"""

import pytest
import json
from datetime import datetime
from typing import Dict, List, Set
from pydantic import ValidationError

# Import the classes we're testing
from neo4jalchemy.core.graph import Graph, GraphNode, GraphEdge


class TestGraphNode:
    """Test the Pydantic GraphNode model."""
    
    def test_node_creation(self):
        """Test basic node creation."""
        node = GraphNode(id="test_id", label="TestLabel")
        
        assert node.id == "test_id"
        assert node.label == "TestLabel"
        assert node.properties == {}
        assert isinstance(node.created_at, datetime)
        assert isinstance(node.updated_at, datetime)
    
    def test_node_with_properties(self):
        """Test node creation with properties."""
        properties = {"name": "Alice", "age": 30}
        node = GraphNode(id="user_1", label="User", properties=properties)
        
        assert node.properties == properties
        assert node.get_property("name") == "Alice"
        assert node.get_property("age") == 30
        assert node.get_property("missing", "default") == "default"
    
    def test_node_id_coercion(self):
        """Test ID coercion to string."""
        # Integer ID should be converted to string
        node = GraphNode(id=123, label="Test")
        assert node.id == "123"
        assert isinstance(node.id, str)
        
        # None ID should generate UUID
        node_with_none = GraphNode(id=None, label="Test")
        assert node_with_none.id is not None
        assert len(node_with_none.id) > 0
    
    def test_node_validation_errors(self):
        """Test validation errors."""
        # Empty label should raise error
        with pytest.raises(ValidationError):
            GraphNode(id="test", label="")
        
        # Whitespace-only label should raise error
        with pytest.raises(ValidationError):
            GraphNode(id="test", label="   ")
        
        # Non-string property keys should raise error
        with pytest.raises(ValidationError):
            GraphNode(id="test", label="Test", properties={123: "value"})
    
    def test_node_property_operations(self):
        """Test property manipulation methods."""
        node = GraphNode(id="test", label="Test")
        
        # Set property
        node.set_property("name", "Alice")
        assert node.get_property("name") == "Alice"
        assert node.property_count == 1
        
        # Has property
        assert node.has_property("name")
        assert not node.has_property("missing")
        
        # Remove property
        removed_value = node.remove_property("name")
        assert removed_value == "Alice"
        assert not node.has_property("name")
        assert node.property_count == 0
        
        # Remove non-existent property
        assert node.remove_property("missing") is None
    
    def test_node_property_updates(self):
        """Test updating node properties."""
        node = GraphNode(id="test", label="Test")
        original_updated = node.updated_at
        
        # Small delay to ensure timestamp changes
        import time
        time.sleep(0.001)
        
        new_props = {"name": "Updated", "value": 42}
        node.update_properties(new_props)
        
        assert node.properties == new_props
        assert node.updated_at > original_updated
    
    def test_node_serialization(self):
        """Test Pydantic serialization methods."""
        properties = {"name": "Test", "value": 123}
        node = GraphNode(id="test", label="TestLabel", properties=properties)
        
        # Test model_dump() method (Pydantic V2)
        node_dict = node.model_dump()
        assert node_dict["id"] == "test"
        assert node_dict["label"] == "TestLabel"
        assert node_dict["properties"] == properties
        assert "created_at" in node_dict
        assert "updated_at" in node_dict
        
        # Test model_dump_json() method (Pydantic V2)
        node_json = node.model_dump_json()
        parsed = json.loads(node_json)
        assert parsed["id"] == "test"
        assert parsed["label"] == "TestLabel"


class TestGraphEdge:
    """Test the Pydantic GraphEdge model."""
    
    def test_edge_creation(self):
        """Test basic edge creation."""
        edge = GraphEdge(
            from_id="node1",
            to_id="node2",
            relationship_type="connects"
        )
        
        assert edge.from_id == "node1"
        assert edge.to_id == "node2"
        assert edge.relationship_type == "CONNECTS"  # Should be uppercase
        assert edge.weight == 1.0
        assert edge.directed is True
        assert edge.properties == {}
        assert isinstance(edge.created_at, datetime)
    
    def test_edge_with_properties(self):
        """Test edge creation with properties."""
        properties = {"strength": 0.8, "type": "friendship"}
        edge = GraphEdge(
            from_id="user1",
            to_id="user2",
            relationship_type="friends",
            properties=properties,
            weight=0.8,
            directed=False
        )
        
        assert edge.properties == properties
        assert edge.weight == 0.8
        assert edge.directed is False
        assert edge.relationship_type == "FRIENDS"
    
    def test_edge_id_coercion(self):
        """Test ID coercion to string."""
        edge = GraphEdge(from_id=123, to_id=456, relationship_type="connects")
        assert edge.from_id == "123"
        assert edge.to_id == "456"
        assert isinstance(edge.from_id, str)
        assert isinstance(edge.to_id, str)
    
    def test_edge_validation_errors(self):
        """Test edge validation errors."""
        # Empty relationship type should raise error
        with pytest.raises(ValidationError):
            GraphEdge(from_id="a", to_id="b", relationship_type="")
        
        # Negative weight should raise error
        with pytest.raises(ValidationError):
            GraphEdge(from_id="a", to_id="b", relationship_type="test", weight=-1.0)
        
        # Non-string property keys should raise error
        with pytest.raises(ValidationError):
            GraphEdge(
                from_id="a", to_id="b", relationship_type="test",
                properties={123: "value"}
            )
    
    def test_edge_id_generation(self):
        """Test edge ID generation."""
        directed_edge = GraphEdge(from_id="a", to_id="b", relationship_type="loves", directed=True)
        undirected_edge = GraphEdge(from_id="a", to_id="b", relationship_type="friends", directed=False)
        
        assert "->[LOVES]->" in directed_edge.id
        assert "<->[FRIENDS]<->" in undirected_edge.id
        assert "a" in directed_edge.id and "b" in directed_edge.id
    
    def test_edge_properties(self):
        """Test edge property methods and computed properties."""
        edge = GraphEdge(from_id="a", to_id="b", relationship_type="test")
        
        # Property operations
        edge.set_property("strength", 0.5)
        assert edge.get_property("strength") == 0.5
        assert edge.property_count == 1
        
        # Self-loop detection
        self_loop = GraphEdge(from_id="a", to_id="a", relationship_type="self")
        normal_edge = GraphEdge(from_id="a", to_id="b", relationship_type="normal")
        
        assert self_loop.is_self_loop is True
        assert normal_edge.is_self_loop is False
    
    def test_edge_reverse(self):
        """Test creating reverse edge."""
        original = GraphEdge(
            from_id="a",
            to_id="b",
            relationship_type="follows",
            properties={"since": "2024"},
            weight=0.5
        )
        
        reversed_edge = original.reverse()
        
        assert reversed_edge.from_id == "b"
        assert reversed_edge.to_id == "a"
        assert reversed_edge.relationship_type == "FOLLOWS"
        assert reversed_edge.properties == {"since": "2024"}
        assert reversed_edge.weight == 0.5
    
    def test_edge_serialization(self):
        """Test Pydantic serialization methods."""
        properties = {"weight": 0.8}
        edge = GraphEdge(
            from_id="a",
            to_id="b",
            relationship_type="connects",
            properties=properties,
            weight=0.8
        )
        
        # Test model_dump() method (Pydantic V2)
        edge_dict = edge.model_dump()
        assert edge_dict["from_id"] == "a"
        assert edge_dict["to_id"] == "b"
        assert edge_dict["relationship_type"] == "CONNECTS"
        assert edge_dict["properties"] == properties
        assert edge_dict["weight"] == 0.8
        
        # Test model_dump_json() method (Pydantic V2)
        edge_json = edge.model_dump_json()
        parsed = json.loads(edge_json)
        assert parsed["from_id"] == "a"
        assert parsed["relationship_type"] == "CONNECTS"


class TestPydanticIntegration:
    """Test Pydantic-specific features and validation."""
    
    def test_node_schema_generation(self):
        """Test that Pydantic generates proper JSON schema."""
        schema = GraphNode.model_json_schema()
        
        assert "properties" in schema
        assert "id" in schema["properties"]
        assert "label" in schema["properties"]
        assert schema["properties"]["id"]["type"] == "string"
    
    def test_edge_schema_generation(self):
        """Test edge schema generation."""
        schema = GraphEdge.model_json_schema()
        
        assert "properties" in schema
        assert "from_id" in schema["properties"]
        assert "to_id" in schema["properties"]
        assert "relationship_type" in schema["properties"]
        assert "weight" in schema["properties"]
    
    def test_model_parsing(self):
        """Test parsing models from dictionaries."""
        node_data = {
            "id": 123,  # Will be coerced to string
            "label": "User",
            "properties": {"name": "Alice"}
        }
        
        node = GraphNode.model_validate(node_data)
        assert node.id == "123"
        assert node.label == "User"
        assert node.properties["name"] == "Alice"
        
        edge_data = {
            "from_id": 456,  # Will be coerced
            "to_id": 789,    # Will be coerced
            "relationship_type": "follows",  # Will be uppercased
            "weight": "0.5"  # Will be coerced to float
        }
        
        edge = GraphEdge.model_validate(edge_data)
        assert edge.from_id == "456"
        assert edge.to_id == "789"
        assert edge.relationship_type == "FOLLOWS"
        assert edge.weight == 0.5
    
    def test_validation_assignment(self):
        """Test validation on assignment."""
        node = GraphNode(id="test", label="Test")
        
        # Valid assignment
        node.properties = {"valid": "data"}
        assert node.properties == {"valid": "data"}
        
        # Invalid assignment should raise error
        with pytest.raises(ValidationError):
            node.properties = {123: "invalid_key"}


class TestGraphBasicOperations:
    """Test basic graph operations with Pydantic models."""
    
    def test_graph_creation(self):
        """Test creating an empty graph."""
        graph = Graph()
        
        assert graph.node_count() == 0
        assert graph.edge_count() == 0
        assert len(graph) == 0
        assert isinstance(graph.name, str)
        assert isinstance(graph.created_at, datetime)
    
    def test_graph_with_name(self):
        """Test creating graph with name."""
        graph = Graph(name="test_graph")
        assert graph.name == "test_graph"
    
    def test_add_node(self):
        """Test adding nodes to graph."""
        graph = Graph()
        
        # Add first node
        node1 = graph.add_node("user1", "User", {"name": "Alice"})
        
        assert isinstance(node1, GraphNode)
        assert node1.id == "user1"
        assert node1.label == "User"
        assert node1.properties == {"name": "Alice"}
        assert graph.node_count() == 1
        assert len(graph) == 1
        assert "user1" in graph
        
        # Add second node with ID coercion
        node2 = graph.add_node(123, "User", {"name": "Bob"})
        
        assert node2.id == "123"  # Coerced to string
        assert graph.node_count() == 2
        assert graph.has_node("user1")
        assert graph.has_node("123")
        assert not graph.has_node("user3")
    
    def test_add_node_validation(self):
        """Test node validation during addition."""
        graph = Graph()
        
        # Invalid label should be caught
        with pytest.raises(ValidationError):
            graph.add_node("test", "", {"name": "Alice"})
        
        # Invalid properties should be caught
        with pytest.raises(ValidationError):
            graph.add_node("test", "User", {123: "invalid"})
    
    def test_add_duplicate_node(self):
        """Test adding duplicate node updates existing."""
        graph = Graph()
        
        # Add original node
        node1 = graph.add_node("user1", "User", {"name": "Alice", "age": 25})
        original_created = node1.created_at
        
        # Add same ID with new properties
        node2 = graph.add_node("user1", "User", {"age": 26, "city": "NYC"})
        
        assert graph.node_count() == 1  # Still only one node
        assert node1 is node2  # Same object returned
        assert node1.properties == {"name": "Alice", "age": 26, "city": "NYC"}
        assert node1.created_at == original_created  # Created time unchanged
    
    def test_add_edge(self):
        """Test adding edges to graph."""
        graph = Graph()
        
        # Add nodes first
        graph.add_node("user1", "User", {"name": "Alice"})
        graph.add_node("user2", "User", {"name": "Bob"})
        
        # Add edge with validation
        edge = graph.add_edge(
            "user1", "user2", "friends",  # lowercase will be converted
            properties={"since": "2024"},
            weight=0.8
        )
        
        assert isinstance(edge, GraphEdge)
        assert edge.from_id == "user1"
        assert edge.to_id == "user2"
        assert edge.relationship_type == "FRIENDS"  # Uppercased
        assert edge.properties == {"since": "2024"}
        assert edge.weight == 0.8
        assert graph.edge_count() == 1
    
    def test_add_edge_validation(self):
        """Test edge validation during addition."""
        graph = Graph()
        graph.add_node("user1", "User")
        graph.add_node("user2", "User")
        
        # Invalid relationship type
        with pytest.raises(ValidationError):
            graph.add_edge("user1", "user2", "")
        
        # Invalid weight
        with pytest.raises(ValidationError):
            graph.add_edge("user1", "user2", "test", weight=-1.0)
        
        # Invalid properties
        with pytest.raises(ValidationError):
            graph.add_edge("user1", "user2", "test", properties={123: "invalid"})
    
    def test_add_edge_missing_nodes(self):
        """Test adding edge with missing nodes raises error."""
        graph = Graph()
        
        with pytest.raises(ValueError, match="Node user1 does not exist"):
            graph.add_edge("user1", "user2", "FRIENDS")
        
        # Add one node
        graph.add_node("user1", "User")
        
        with pytest.raises(ValueError, match="Node user2 does not exist"):
            graph.add_edge("user1", "user2", "FRIENDS")
    
    def test_remove_node(self):
        """Test removing nodes from graph."""
        graph = Graph()
        
        # Add nodes and edges
        graph.add_node("user1", "User")
        graph.add_node("user2", "User")
        graph.add_node("user3", "User")
        graph.add_edge("user1", "user2", "FRIENDS")
        graph.add_edge("user2", "user3", "FRIENDS")
        
        assert graph.node_count() == 3
        assert graph.edge_count() == 2
        
        # Remove middle node
        result = graph.remove_node("user2")
        
        assert result is True
        assert graph.node_count() == 2
        assert graph.edge_count() == 0  # All edges to user2 removed
        assert not graph.has_node("user2")
        assert graph.has_node("user1")
        assert graph.has_node("user3")
    
    def test_remove_nonexistent_node(self):
        """Test removing non-existent node returns False."""
        graph = Graph()
        result = graph.remove_node("nonexistent")
        assert result is False


class TestGraphQueries:
    """Test graph query and lookup operations."""
    
    def test_get_node(self):
        """Test retrieving nodes by ID."""
        graph = Graph()
        graph.add_node("user1", "User", {"name": "Alice"})
        
        node = graph.get_node("user1")
        assert node is not None
        assert node.id == "user1"
        assert node.properties["name"] == "Alice"
        assert isinstance(node, GraphNode)  # Should be Pydantic model
        
        # Test non-existent node
        missing = graph.get_node("missing")
        assert missing is None
    
    def test_has_edge(self):
        """Test checking if edges exist."""
        graph = Graph()
        graph.add_node("a", "Node")
        graph.add_node("b", "Node")
        graph.add_node("c", "Node")
        
        # Add edges
        graph.add_edge("a", "b", "CONNECTS")
        graph.add_edge("a", "c", "OTHER")
        
        # Test edge existence
        assert graph.has_edge("a", "b")
        assert graph.has_edge("a", "c")
        assert not graph.has_edge("b", "c")
        assert not graph.has_edge("b", "a")  # Directed
        
        # Test with relationship type (should match uppercased version)
        assert graph.has_edge("a", "b", "CONNECTS")
        assert graph.has_edge("a", "c", "OTHER")
        assert not graph.has_edge("a", "b", "OTHER")
        assert not graph.has_edge("a", "c", "CONNECTS")


# Continue with remaining test classes (abbreviated for space)
class TestGraphAlgorithms:
    """Test graph algorithm implementations."""
    
    def setup_sample_graph(self) -> Graph:
        """Create a sample graph for testing algorithms."""
        graph = Graph()
        
        # Add nodes
        for i in range(1, 7):  # nodes 1-6
            graph.add_node(str(i), "Node", {"value": i})
        
        # Create connected components:
        # Component 1: 1-2-3 (linear)
        graph.add_edge("1", "2", "CONNECTS")
        graph.add_edge("2", "3", "CONNECTS")
        
        # Component 2: 4-5-6 (triangle)
        graph.add_edge("4", "5", "CONNECTS")
        graph.add_edge("5", "6", "CONNECTS")
        graph.add_edge("6", "4", "CONNECTS")
        
        return graph
    
    def test_bfs(self):
        """Test breadth-first search."""
        graph = self.setup_sample_graph()
        
        # BFS from node 1
        result = graph.bfs("1")
        
        assert result["1"] == 0  # Start node
        assert result["2"] == 1  # Direct neighbor
        assert result["3"] == 2  # Two steps away
        assert "4" not in result  # Different component
        
    def test_shortest_path(self):
        """Test shortest path finding."""
        graph = self.setup_sample_graph()
        
        # Path within component
        path = graph.shortest_path("1", "3")
        assert path == ["1", "2", "3"]
        
        # Path to same node
        same_path = graph.shortest_path("1", "1")
        assert same_path == ["1"]
        
        # Path between different components (should be None)
        no_path = graph.shortest_path("1", "4")
        assert no_path is None


class TestGraphSerialization:
    """Test graph serialization with Pydantic models."""
    
    def test_to_dict_with_pydantic(self):
        """Test converting graph to dictionary with Pydantic models."""
        graph = Graph(name="test_graph")
        graph.add_node("user1", "User", {"name": "Alice", "age": 30})
        graph.add_edge("user1", "user1", "SELF_REFERENCE", {"type": "test"})
        
        graph_dict = graph.to_dict()
        
        assert graph_dict["name"] == "test_graph"
        assert len(graph_dict["nodes"]) == 1
        assert len(graph_dict["edges"]) == 1
        
        # Check that Pydantic .dict() was used
        node_data = graph_dict["nodes"][0]
        assert node_data["id"] == "user1"
        assert node_data["label"] == "User"
        assert node_data["properties"]["name"] == "Alice"
        
        # Check edge data
        edge_data = graph_dict["edges"][0]
        assert edge_data["relationship_type"] == "SELF_REFERENCE"
    
    def test_from_dict_with_pydantic(self):
        """Test creating graph from dictionary using Pydantic parsing."""
        graph_data = {
            "name": "restored_graph",
            "nodes": [
                {
                    "id": 999,  # Will be coerced to string
                    "label": "NodeA",
                    "properties": {"value": 1}
                }
            ],
            "edges": [
                {
                    "from_id": 999,  # Will be coerced
                    "to_id": 999,    # Will be coerced
                    "relationship_type": "self_loop",  # Will be uppercased  
                    "weight": "0.5"  # Will be coerced to float
                }
            ]
        }
        
        graph = Graph.from_dict(graph_data)
        
        assert graph.name == "restored_graph"
        assert graph.node_count() == 1
        assert graph.edge_count() == 1
        
        # Check that coercion worked
        node = graph.get_node("999")  # ID coerced to string
        assert node.id == "999"
        assert isinstance(node, GraphNode)
        
        edges = graph.get_edges_between("999", "999")
        assert len(edges) == 1
        assert edges[0].relationship_type == "SELF_LOOP"
        assert edges[0].weight == 0.5
        assert isinstance(edges[0], GraphEdge)

        properties = {"weight": 0.8}
        edge = GraphEdge(
            from_id="a",
            to_id="b", 
            relationship_type="CONNECTS",
            properties=properties,
            weight=0.8
        )
        
        edge_dict = edge.to_dict()
        
        assert edge_dict["from_id"] == "a"
        assert edge_dict["to_id"] == "b"
        assert edge_dict["relationship_type"] == "CONNECTS"
        assert edge_dict["properties"] == properties
        assert edge_dict["weight"] == 0.8


class TestGraphBasicOperations:
    """Test basic graph operations."""
    
    def test_graph_creation(self):
        """Test creating an empty graph."""
        graph = Graph()
        
        assert graph.node_count() == 0
        assert graph.edge_count() == 0
        assert len(graph) == 0
        assert isinstance(graph.name, str)
        assert isinstance(graph.created_at, datetime)
    
    def test_graph_with_name(self):
        """Test creating graph with name."""
        graph = Graph(name="test_graph")
        assert graph.name == "test_graph"
    
    def test_add_node(self):
        """Test adding nodes to graph."""
        graph = Graph()
        
        # Add first node
        node1 = graph.add_node("user1", "User", {"name": "Alice"})
        
        assert isinstance(node1, GraphNode)
        assert node1.id == "user1"
        assert node1.label == "User"
        assert node1.properties == {"name": "Alice"}
        assert graph.node_count() == 1
        assert len(graph) == 1
        assert "user1" in graph
        
        # Add second node
        node2 = graph.add_node("user2", "User", {"name": "Bob"})
        
        assert graph.node_count() == 2
        assert graph.has_node("user1")
        assert graph.has_node("user2")
        assert not graph.has_node("user3")
    
    def test_add_duplicate_node(self):
        """Test adding duplicate node updates existing."""
        graph = Graph()
        
        # Add original node
        node1 = graph.add_node("user1", "User", {"name": "Alice", "age": 25})
        original_created = node1.created_at
        
        # Add same ID with new properties
        node2 = graph.add_node("user1", "User", {"age": 26, "city": "NYC"})
        
        assert graph.node_count() == 1  # Still only one node
        assert node1 is node2  # Same object returned
        assert node1.properties == {"name": "Alice", "age": 26, "city": "NYC"}
        assert node1.created_at == original_created  # Created time unchanged
    
    def test_add_edge(self):
        """Test adding edges to graph."""
        graph = Graph()
        
        # Add nodes first
        graph.add_node("user1", "User", {"name": "Alice"})
        graph.add_node("user2", "User", {"name": "Bob"})
        
        # Add edge
        edge = graph.add_edge(
            "user1", "user2", "FRIENDS", 
            properties={"since": "2024"},
            weight=0.8
        )
        
        assert isinstance(edge, GraphEdge)
        assert edge.from_id == "user1"
        assert edge.to_id == "user2"
        assert edge.relationship_type == "FRIENDS"
        assert edge.properties == {"since": "2024"}
        assert edge.weight == 0.8
        assert graph.edge_count() == 1
    
    def test_add_edge_missing_nodes(self):
        """Test adding edge with missing nodes raises error."""
        graph = Graph()
        
        with pytest.raises(ValueError, match="Node user1 does not exist"):
            graph.add_edge("user1", "user2", "FRIENDS")
        
        # Add one node
        graph.add_node("user1", "User")
        
        with pytest.raises(ValueError, match="Node user2 does not exist"):
            graph.add_edge("user1", "user2", "FRIENDS")
    
    def test_remove_node(self):
        """Test removing nodes from graph."""
        graph = Graph()
        
        # Add nodes and edges
        graph.add_node("user1", "User")
        graph.add_node("user2", "User")
        graph.add_node("user3", "User")
        graph.add_edge("user1", "user2", "FRIENDS")
        graph.add_edge("user2", "user3", "FRIENDS")
        
        assert graph.node_count() == 3
        assert graph.edge_count() == 2
        
        # Remove middle node
        result = graph.remove_node("user2")
        
        assert result is True
        assert graph.node_count() == 2
        assert graph.edge_count() == 0  # All edges to user2 removed
        assert not graph.has_node("user2")
        assert graph.has_node("user1")
        assert graph.has_node("user3")
    
    def test_remove_nonexistent_node(self):
        """Test removing non-existent node returns False."""
        graph = Graph()
        result = graph.remove_node("nonexistent")
        assert result is False


class TestGraphQueries:
    """Test graph query and lookup operations."""
    
    def test_get_node(self):
        """Test retrieving nodes by ID."""
        graph = Graph()
        graph.add_node("user1", "User", {"name": "Alice"})
        
        node = graph.get_node("user1")
        assert node is not None
        assert node.id == "user1"
        assert node.properties["name"] == "Alice"
        
        # Test non-existent node
        missing = graph.get_node("missing")
        assert missing is None
    
    def test_has_edge(self):
        """Test checking if edges exist."""
        graph = Graph()
        graph.add_node("a", "Node")
        graph.add_node("b", "Node")
        graph.add_node("c", "Node")
        
        # Add edges
        graph.add_edge("a", "b", "CONNECTS")
        graph.add_edge("a", "c", "OTHER")
        
        # Test edge existence
        assert graph.has_edge("a", "b")
        assert graph.has_edge("a", "c")
        assert not graph.has_edge("b", "c")
        assert not graph.has_edge("b", "a")  # Directed
        
        # Test with relationship type
        assert graph.has_edge("a", "b", "CONNECTS")
        assert graph.has_edge("a", "c", "OTHER")
        assert not graph.has_edge("a", "b", "OTHER")
        assert not graph.has_edge("a", "c", "CONNECTS")
    
    def test_neighbors_outgoing(self):
        """Test getting outgoing neighbors."""
        graph = Graph()
        graph.add_node("center", "Node")
        graph.add_node("neighbor1", "Node")
        graph.add_node("neighbor2", "Node")
        graph.add_node("neighbor3", "Node")
        
        # Add outgoing edges
        graph.add_edge("center", "neighbor1", "TYPE_A")
        graph.add_edge("center", "neighbor2", "TYPE_A")
        graph.add_edge("center", "neighbor3", "TYPE_B")
        
        # Test all outgoing neighbors
        all_neighbors = graph.neighbors("center", direction="outgoing")
        assert set(all_neighbors) == {"neighbor1", "neighbor2", "neighbor3"}
        
        # Test neighbors by relationship type
        type_a_neighbors = graph.neighbors("center", "TYPE_A", "outgoing")
        assert set(type_a_neighbors) == {"neighbor1", "neighbor2"}
        
        type_b_neighbors = graph.neighbors("center", "TYPE_B", "outgoing")
        assert type_b_neighbors == ["neighbor3"]
    
    def test_neighbors_incoming(self):
        """Test getting incoming neighbors."""
        graph = Graph()
        graph.add_node("center", "Node")
        graph.add_node("source1", "Node")
        graph.add_node("source2", "Node")
        
        # Add incoming edges
        graph.add_edge("source1", "center", "POINTS_TO")
        graph.add_edge("source2", "center", "POINTS_TO")
        
        # Test incoming neighbors
        incoming = graph.neighbors("center", direction="incoming")
        assert set(incoming) == {"source1", "source2"}
        
        # Test with relationship type
        incoming_typed = graph.neighbors("center", "POINTS_TO", "incoming")
        assert set(incoming_typed) == {"source1", "source2"}
    
    def test_get_edges_between(self):
        """Test getting edges between two specific nodes."""
        graph = Graph()
        graph.add_node("a", "Node")
        graph.add_node("b", "Node")
        
        # Add multiple edges between same nodes
        edge1 = graph.add_edge("a", "b", "TYPE1", weight=1.0)
        edge2 = graph.add_edge("a", "b", "TYPE2", weight=2.0)
        
        edges = graph.get_edges_between("a", "b")
        assert len(edges) == 2
        
        edge_types = {edge.relationship_type for edge in edges}
        assert edge_types == {"TYPE1", "TYPE2"}
        
        # Test non-existent connection
        empty_edges = graph.get_edges_between("a", "nonexistent")
        assert empty_edges == []


class TestGraphAlgorithms:
    """Test graph algorithm implementations."""
    
    def setup_sample_graph(self) -> Graph:
        """Create a sample graph for testing algorithms."""
        graph = Graph()
        
        # Add nodes
        for i in range(1, 7):  # nodes 1-6
            graph.add_node(str(i), "Node", {"value": i})
        
        # Create connected components:
        # Component 1: 1-2-3 (linear)
        graph.add_edge("1", "2", "CONNECTS")
        graph.add_edge("2", "3", "CONNECTS")
        
        # Component 2: 4-5-6 (triangle)
        graph.add_edge("4", "5", "CONNECTS")
        graph.add_edge("5", "6", "CONNECTS")
        graph.add_edge("6", "4", "CONNECTS")
        
        return graph
    
    def test_bfs(self):
        """Test breadth-first search."""
        graph = self.setup_sample_graph()
        
        # BFS from node 1
        result = graph.bfs("1")
        
        assert result["1"] == 0  # Start node
        assert result["2"] == 1  # Direct neighbor
        assert result["3"] == 2  # Two steps away
        assert "4" not in result  # Different component
        assert "5" not in result
        assert "6" not in result
        
        # BFS with max depth
        limited_result = graph.bfs("1", max_depth=1)
        assert set(limited_result.keys()) == {"1", "2"}
        assert limited_result["1"] == 0
        assert limited_result["2"] == 1
    
    def test_bfs_nonexistent_node(self):
        """Test BFS with non-existent start node."""
        graph = self.setup_sample_graph()
        result = graph.bfs("nonexistent")
        assert result == {}
    
    def test_dfs(self):
        """Test depth-first search."""
        graph = self.setup_sample_graph()
        
        # DFS from node 1
        result = graph.dfs("1")
        
        # Should visit all nodes in component 1
        assert "1" in result
        assert "2" in result
        assert "3" in result
        assert "4" not in result  # Different component
        
        # First node should be start node
        assert result[0] == "1"
    
    def test_dfs_with_max_depth(self):
        """Test DFS with depth limit."""
        graph = self.setup_sample_graph()
        
        result = graph.dfs("1", max_depth=1)
        
        # Should only go 1 level deep
        assert "1" in result
        assert "2" in result
        assert "3" not in result  # Too deep
    
    def test_shortest_path(self):
        """Test shortest path finding."""
        graph = self.setup_sample_graph()
        
        # Path within component
        path = graph.shortest_path("1", "3")
        assert path == ["1", "2", "3"]
        
        # Path to same node
        same_path = graph.shortest_path("1", "1")
        assert same_path == ["1"]
        
        # Path between different components (should be None)
        no_path = graph.shortest_path("1", "4")
        assert no_path is None
        
        # Path with non-existent nodes
        invalid_path = graph.shortest_path("1", "nonexistent")
        assert invalid_path is None
    
    def test_connected_components(self):
        """Test finding connected components."""
        graph = self.setup_sample_graph()
        
        components = graph.connected_components()
        
        assert len(components) == 2
        
        # Sort components by size for consistent testing
        components.sort(key=len)
        
        # First component should have 3 nodes
        component1 = set(components[0])
        assert len(component1) == 3
        assert component1 == {"1", "2", "3"}
        
        # Second component should have 3 nodes
        component2 = set(components[1])
        assert len(component2) == 3
        assert component2 == {"4", "5", "6"}
    
    def test_connected_components_single_node(self):
        """Test connected components with isolated nodes."""
        graph = Graph()
        graph.add_node("isolated", "Node")
        
        components = graph.connected_components()
        assert len(components) == 1
        assert components[0] == ["isolated"]
    
    def test_degree_centrality(self):
        """Test degree centrality calculation."""
        graph = Graph()
        
        # Create a star graph: center connected to 3 others
        graph.add_node("center", "Node")
        graph.add_node("leaf1", "Node")
        graph.add_node("leaf2", "Node")
        graph.add_node("leaf3", "Node")
        
        graph.add_edge("center", "leaf1", "CONNECTS")
        graph.add_edge("center", "leaf2", "CONNECTS")
        graph.add_edge("center", "leaf3", "CONNECTS")
        
        # Center node should have highest centrality
        center_centrality = graph.degree_centrality("center")
        leaf_centrality = graph.degree_centrality("leaf1")
        
        assert center_centrality == 1.0  # Connected to all other nodes
        assert leaf_centrality == 1.0 / 3  # Connected to 1 out of 3 other nodes
        
        # Test non-existent node
        assert graph.degree_centrality("nonexistent") == 0.0
    
    def test_degree_centrality_single_node(self):
        """Test degree centrality with single node."""
        graph = Graph()
        graph.add_node("single", "Node")
        
        # Single node has centrality 0
        assert graph.degree_centrality("single") == 0.0


class TestGraphStatistics:
    """Test graph statistics and analysis."""
    
    def test_basic_statistics(self):
        """Test basic graph statistics."""
        graph = Graph()
        
        # Empty graph
        assert graph.node_count() == 0
        assert graph.edge_count() == 0
        assert graph.density() == 0.0
        assert graph.average_degree() == 0.0
        
        # Add nodes and edges
        graph.add_node("a", "Node")
        graph.add_node("b", "Node")
        graph.add_node("c", "Node")
        graph.add_edge("a", "b", "CONNECTS")
        graph.add_edge("b", "c", "CONNECTS")
        
        assert graph.node_count() == 3
        assert graph.edge_count() == 2
        assert graph.density() == 2.0 / (3 * 2)  # 2 edges / (3 * 2 possible)
    
    def test_average_degree(self):
        """Test average degree calculation."""
        graph = Graph()
        
        # Create simple chain: a -> b -> c
        graph.add_node("a", "Node")
        graph.add_node("b", "Node") 
        graph.add_node("c", "Node")
        graph.add_edge("a", "b", "CONNECTS")
        graph.add_edge("b", "c", "CONNECTS")
        
        # Node degrees: a=1, b=2, c=1, average = 4/3
        expected_avg = (1 + 2 + 1) / 3
        assert abs(graph.average_degree() - expected_avg) < 0.001


class TestGraphSerialization:
    """Test graph serialization and deserialization."""
    
    def test_to_dict(self):
        """Test converting graph to dictionary."""
        graph = Graph(name="test_graph")
        graph.add_node("user1", "User", {"name": "Alice", "age": 30})
        graph.add_node("user2", "User", {"name": "Bob", "age": 25})
        graph.add_edge("user1", "user2", "FRIENDS", {"since": "2024"}, weight=0.8)
        
        graph_dict = graph.to_dict()
        
        assert graph_dict["name"] == "test_graph"
        assert "created_at" in graph_dict
        assert len(graph_dict["nodes"]) == 2
        assert len(graph_dict["edges"]) == 1
        
        # Check node data
        node_ids = {node["id"] for node in graph_dict["nodes"]}
        assert node_ids == {"user1", "user2"}
        
        # Check edge data
        edge = graph_dict["edges"][0]
        assert edge["from_id"] == "user1"
        assert edge["to_id"] == "user2"
        assert edge["relationship_type"] == "FRIENDS"
        assert edge["properties"] == {"since": "2024"}
        assert edge["weight"] == 0.8
        
        # Check statistics
        stats = graph_dict["statistics"]
        assert stats["node_count"] == 2
        assert stats["edge_count"] == 1
    
    def test_to_json(self):
        """Test converting graph to JSON."""
        graph = Graph(name="json_test")
        graph.add_node("1", "Node", {"value": 42})
        
        json_str = graph.to_json()
        
        # Should be valid JSON
        parsed = json.loads(json_str)
        assert parsed["name"] == "json_test"
        assert len(parsed["nodes"]) == 1
        assert parsed["nodes"][0]["properties"]["value"] == 42
        
        # Test with indentation
        pretty_json = graph.to_json(indent=2)
        assert "\n" in pretty_json
        assert "  " in pretty_json
    
    def test_from_dict(self):
        """Test creating graph from dictionary."""
        graph_data = {
            "name": "restored_graph",
            "nodes": [
                {
                    "id": "a",
                    "label": "NodeA", 
                    "properties": {"value": 1},
                    "created_at": "2024-01-01T00:00:00",
                    "updated_at": "2024-01-01T00:00:00"
                },
                {
                    "id": "b",
                    "label": "NodeB",
                    "properties": {"value": 2},
                    "created_at": "2024-01-01T00:00:00", 
                    "updated_at": "2024-01-01T00:00:00"
                }
            ],
            "edges": [
                {
                    "from_id": "a",
                    "to_id": "b",
                    "relationship_type": "CONNECTS",
                    "properties": {"strength": 0.5},
                    "weight": 0.8,
                    "directed": True,
                    "created_at": "2024-01-01T00:00:00"
                }
            ]
        }
        
        graph = Graph.from_dict(graph_data)
        
        assert graph.name == "restored_graph"
        assert graph.node_count() == 2
        assert graph.edge_count() == 1
        
        # Check nodes
        node_a = graph.get_node("a")
        assert node_a.label == "NodeA"
        assert node_a.properties["value"] == 1
        
        # Check edges
        assert graph.has_edge("a", "b", "CONNECTS")
        edges = graph.get_edges_between("a", "b")
        assert len(edges) == 1
        assert edges[0].properties["strength"] == 0.5
        assert edges[0].weight == 0.8
    
    def test_from_json(self):
        """Test creating graph from JSON string."""
        json_data = '''
        {
            "name": "json_graph",
            "nodes": [
                {"id": "1", "label": "First", "properties": {"name": "Node1"}}
            ],
            "edges": []
        }
        '''
        
        graph = Graph.from_json(json_data)
        
        assert graph.name == "json_graph"
        assert graph.node_count() == 1
        assert graph.get_node("1").properties["name"] == "Node1"
    
    def test_round_trip_serialization(self):
        """Test that serialization and deserialization preserves graph."""
        original = Graph(name="round_trip_test")
        original.add_node("user1", "User", {"name": "Alice", "score": 95.5})
        original.add_node("user2", "User", {"name": "Bob", "score": 87.2})
        original.add_edge("user1", "user2", "FRIENDS", {"since": 2020}, weight=0.9)
        
        # Serialize and deserialize
        graph_dict = original.to_dict()
        restored = Graph.from_dict(graph_dict)
        
        # Compare key properties
        assert restored.name == original.name
        assert restored.node_count() == original.node_count()
        assert restored.edge_count() == original.edge_count()
        
        # Compare specific data
        original_user1 = original.get_node("user1")
        restored_user1 = restored.get_node("user1")
        assert restored_user1.properties == original_user1.properties
        
        # Compare edges
        original_edges = original.get_edges_between("user1", "user2")
        restored_edges = restored.get_edges_between("user1", "user2")
        assert len(restored_edges) == len(original_edges)
        assert restored_edges[0].properties == original_edges[0].properties


class TestGraphUtilities:
    """Test graph utility methods."""
    
    def test_copy(self):
        """Test graph copying."""
        original = Graph(name="original")
        original.add_node("1", "Node", {"value": 42})
        original.add_node("2", "Node", {"value": 24})
        original.add_edge("1", "2", "CONNECTS")
        
        copy = original.copy()
        
        # Should be separate objects
        assert copy is not original
        assert copy.name == original.name
        assert copy.node_count() == original.node_count()
        assert copy.edge_count() == original.edge_count()
        
        # Modifying copy shouldn't affect original
        copy.add_node("3", "Node")
        assert copy.node_count() == 3
        assert original.node_count() == 2
    
    def test_clear(self):
        """Test clearing graph."""
        graph = Graph()
        graph.add_node("1", "Node")
        graph.add_node("2", "Node")
        graph.add_edge("1", "2", "CONNECTS")
        
        assert graph.node_count() == 2
        assert graph.edge_count() == 1
        
        graph.clear()
        
        assert graph.node_count() == 0
        assert graph.edge_count() == 0
        assert len(graph) == 0
    
    def test_magic_methods(self):
        """Test magic methods (__len__, __contains__, __iter__, __repr__)."""
        graph = Graph(name="magic_test")
        graph.add_node("a", "Node")
        graph.add_node("b", "Node")
        graph.add_node("c", "Node")
        
        # Test __len__
        assert len(graph) == 3
        
        # Test __contains__
        assert "a" in graph
        assert "b" in graph
        assert "nonexistent" not in graph
        
        # Test __iter__
        node_ids = list(graph)
        assert set(node_ids) == {"a", "b", "c"}
        
        # Test __repr__
        repr_str = repr(graph)
        assert "magic_test" in repr_str
        assert "nodes=3" in repr_str
        assert "edges=0" in repr_str


class TestUndirectedEdges:
    """Test undirected edge functionality."""
    
    def test_undirected_edge_creation(self):
        """Test creating undirected edges."""
        graph = Graph()
        graph.add_node("a", "Node")
        graph.add_node("b", "Node")
        
        # Add undirected edge
        edge = graph.add_edge("a", "b", "FRIENDS", directed=False)
        
        assert edge.directed is False
        assert graph.edge_count() == 1  # Only one edge object
        
        # Should be able to traverse in both directions
        a_neighbors = graph.neighbors("a", direction="outgoing")
        b_neighbors = graph.neighbors("b", direction="outgoing")
        
        assert "b" in a_neighbors
        assert "a" in b_neighbors
    
    def test_undirected_path_finding(self):
        """Test path finding with undirected edges."""
        graph = Graph()
        graph.add_node("1", "Node")
        graph.add_node("2", "Node")
        graph.add_node("3", "Node")
        
        # Create undirected chain
        graph.add_edge("1", "2", "CONNECTS", directed=False)
        graph.add_edge("2", "3", "CONNECTS", directed=False)
        
        # Should find path in both directions
        path_forward = graph.shortest_path("1", "3")
        path_backward = graph.shortest_path("3", "1")
        
        assert path_forward == ["1", "2", "3"]
        assert path_backward == ["3", "2", "1"]


if __name__ == "__main__":
    # Run tests if script is executed directly
    pytest.main([__file__, "-v"])