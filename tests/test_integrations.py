# tests/test_integration.py
"""
Integration tests for Neo4jAlchemy GraphEntity system.

Tests the complete flow from entity definition to graph operations,
ensuring all components work together seamlessly.
"""

import pytest
import pytest_asyncio
from datetime import datetime
from typing import List, Optional

from pydantic import Field, ValidationError

# Test the main import flow
from neo4jalchemy import Graph, GraphEntity, graph_entity


@pytest.mark.asyncio
class TestGraphEntityIntegration:
    """Test complete GraphEntity workflow integration."""
    
    async def test_complete_workflow(self):
        """Test the complete entity lifecycle with modern Pydantic V2."""
        
        # 1. Create graph
        graph = Graph(name="integration_test")
        
        # 2. Define entity with pure Pydantic V2 fields
        @graph_entity(label="TestUser", graph=graph)
        class TestUser(GraphEntity):
            # Core fields with validation
            name: str = Field(min_length=1, max_length=100)
            email: str = Field(pattern=r'^[^@]+@[^@]+\.[^@]+$')
            age: int = Field(ge=0, le=150)
            
            # Additional fields
            username: str = Field(min_length=3, max_length=20)
            score: int = Field(default=0, ge=0, le=100)
            tags: List[str] = Field(default_factory=list)
            is_active: bool = Field(default=True)
        
        # 3. Create entity instance
        user = TestUser(
            name="Integration Test User",
            email="integration@test.com", 
            age=25,
            username="integration_user",
            score=85,
            tags=["test", "integration"]
        )
        
        # 4. Verify entity properties
        assert user.name == "Integration Test User"
        assert user.username == "integration_user"
        assert user.score == 85
        assert user.tags == ["test", "integration"]
        assert user.is_active is True
        assert isinstance(user.id, str)
        assert len(user.id) > 0
        
        # 5. Verify graph node was created
        assert user._graph_node is not None
        assert graph.has_node(user.id)
        
        node = graph.get_node(user.id)
        assert node.label == "TestUser"
        assert node.properties["name"] == "Integration Test User"
        assert node.properties["username"] == "integration_user"
        assert node.properties["score"] == 85
        assert node.properties["tags"] == ["test", "integration"]
        
        # 6. Test save operation
        await user.save()
        assert user._is_persisted
        assert not user.is_dirty()
        
        # 7. Test entity modification
        user.score = 95
        user.name = "Updated Integration User"
        user.tags.append("updated")
        assert user.is_dirty()
        
        await user.save()
        assert not user.is_dirty()
        
        # 8. Verify graph was updated
        updated_node = graph.get_node(user.id)
        assert updated_node.properties["name"] == "Updated Integration User"
        assert updated_node.properties["score"] == 95
        assert updated_node.properties["tags"] == ["test", "integration", "updated"]
        
        # 9. Test class methods
        retrieved = await TestUser.get(user.id)
        assert retrieved is not None
        assert retrieved.name == "Updated Integration User"
        assert retrieved.score == 95
        
        # 10. Test entity creation via class method
        new_user = await TestUser.create(
            name="Class Method User",
            email="classmethod@test.com",
            age=30,
            username="class_method_user",
            score=70,
            tags=["created", "via", "class"]
        )
        
        assert new_user._is_persisted
        assert graph.has_node(new_user.id)
        
        # 11. Test get_or_create
        existing, created = await TestUser.get_or_create(
            id=user.id,
            defaults={"name": "Should not use this"}
        )
        assert not created
        assert existing.id == user.id
        assert existing.name == "Updated Integration User"
        
        another_new, created = await TestUser.get_or_create(
            id="brand_new_user",
            defaults={
                "name": "Brand New User",
                "email": "brandnew@test.com",
                "age": 22,
                "username": "brand_new",
                "score": 60,
                "tags": ["new"]
            }
        )
        assert created
        assert another_new.name == "Brand New User"
        
        # 12. Test deletion
        delete_result = await new_user.delete()
        assert delete_result is True
        assert not graph.has_node(new_user.id)
        assert not new_user._is_persisted
        
        # 13. Verify final graph state
        assert graph.node_count() == 2  # user and another_new
        
        # 14. Test graph algorithms work with entities
        components = graph.connected_components()
        assert len(components) >= 1  # All nodes should be in components
        
        # 15. Test graph export includes entity data
        graph_dict = graph.to_dict()
        assert len(graph_dict["nodes"]) == graph.node_count()
        
        # Find our user in the export
        user_nodes = [n for n in graph_dict["nodes"] if n["id"] == user.id]
        assert len(user_nodes) == 1
        assert user_nodes[0]["label"] == "TestUser"
        assert user_nodes[0]["properties"]["name"] == "Updated Integration User"
        
        print("✅ Integration test passed - GraphEntity system working perfectly!")
    
    async def test_field_validation_integration(self):
        """Test that Pydantic V2 field validation works correctly."""
        
        graph = Graph(name="validation_test")
        
        @graph_entity(graph=graph)
        class ValidatedEntity(GraphEntity):
            # Pydantic validation
            email: str = Field(pattern=r'^[^@]+@[^@]+\.[^@]+$')
            age: int = Field(ge=0, le=100)
            
            # Additional validation
            username: str = Field(min_length=3, max_length=20)
            score: int = Field(ge=0, le=100)
            bio: Optional[str] = Field(None, max_length=500)
        
        # Test successful creation
        entity = ValidatedEntity(
            email="valid@test.com",
            age=25,
            username="validuser",
            score=75,
            bio="A valid user bio"
        )
        
        await entity.save()
        assert entity._is_persisted
        
        # Test Pydantic validation failure
        with pytest.raises(ValidationError):
            ValidatedEntity(
                email="invalid-email",  # Bad email format
                age=25,
                username="test",
                score=50
            )
        
        # Test field validation on assignment
        with pytest.raises(ValidationError):
            entity.score = 150  # Over max value
        
        with pytest.raises(ValidationError):
            entity.username = "xy"  # Too short
        
        # Test valid assignments
        entity.score = 80
        entity.bio = "Updated bio"
        assert entity.score == 80
        assert entity.bio == "Updated bio"
    
    async def test_multiple_entity_types(self):
        """Test multiple entity types in same graph."""
        
        graph = Graph(name="multi_entity_test")
        
        @graph_entity(label="Person", graph=graph)
        class Person(GraphEntity):
            name: str = Field(min_length=1, max_length=100)
            age: int = Field(ge=0, le=150)
            occupation: Optional[str] = Field(None, max_length=100)
            
        @graph_entity(label="Company", graph=graph) 
        class Company(GraphEntity):
            name: str = Field(min_length=1, max_length=200)
            industry: str = Field(min_length=1, max_length=100)
            founded_year: int = Field(ge=1800, le=2025)
            employees: int = Field(default=1, ge=1)
            
        # Create entities of different types
        person = await Person.create(
            name="John Doe", 
            age=30, 
            occupation="Software Engineer"
        )
        company = await Company.create(
            name="TechCorp", 
            industry="Technology",
            founded_year=2020,
            employees=50
        )
        
        # Verify both are in graph
        assert graph.node_count() == 2
        assert graph.has_node(person.id)
        assert graph.has_node(company.id)
        
        # Verify they have different labels
        person_node = graph.get_node(person.id)
        company_node = graph.get_node(company.id)
        
        assert person_node.label == "Person"
        assert company_node.label == "Company"
        
        # Test type-specific retrieval
        retrieved_person = await Person.get(person.id)
        retrieved_company = await Company.get(company.id)
        
        assert retrieved_person is not None
        assert retrieved_company is not None
        assert retrieved_person.name == "John Doe"
        assert retrieved_company.industry == "Technology"
        
        # Test business logic
        assert retrieved_person.age == 30
        assert retrieved_company.employees == 50
    
    async def test_entity_lifecycle_hooks(self):
        """Test entity lifecycle hooks integration."""
        
        graph = Graph(name="lifecycle_test")
        
        @graph_entity(label="TrackedUser", graph=graph)
        class TrackedUser(GraphEntity):
            name: str
            status: str = Field(default="draft")
            processed_at: Optional[datetime] = None
            
            async def _pre_save(self):
                """Business logic in pre-save hook."""
                if self.status == "draft":
                    self.status = "active"
                    self.processed_at = datetime.now()
            
            async def _post_save(self):
                """Log after saving."""
                pass  # In real app, might log to external system
        
        # Create user
        user = TrackedUser(name="Hook Test User")
        assert user.status == "draft"
        assert user.processed_at is None
        
        # Save triggers hooks
        await user.save()
        
        # Verify hook executed
        assert user.status == "active"
        assert user.processed_at is not None
        assert isinstance(user.processed_at, datetime)
        
        # Verify in graph
        node = graph.get_node(user.id)
        assert node.properties["status"] == "active"
        assert node.properties["processed_at"] is not None
    
    async def test_entity_registry_integration(self):
        """Test entity registry and discovery."""
        
        from neo4jalchemy.orm.entities import get_entity_classes, get_entity_by_label
        
        @graph_entity(label="RegistryTestEntity")
        class RegistryTestEntity(GraphEntity):
            name: str
            test_field: str = "test_value"
        
        # Should be in registry
        entity_classes = get_entity_classes()
        class_names = {cls.__name__ for cls in entity_classes}
        assert "RegistryTestEntity" in class_names
        
        # Should be findable by label
        found_class = get_entity_by_label("RegistryTestEntity")
        assert found_class is RegistryTestEntity
        
        # Test entity works
        entity = RegistryTestEntity(name="Registry Test")
        assert entity.name == "Registry Test"
        assert entity.test_field == "test_value"
    
    async def test_graph_serialization_with_entities(self):
        """Test graph serialization with entity data."""
        
        graph = Graph(name="serialization_test")
        
        @graph_entity(label="SerialUser", graph=graph)
        class SerialUser(GraphEntity):
            name: str
            metadata: dict = Field(default_factory=dict)
            tags: List[str] = Field(default_factory=list)
        
        # Create entities with complex data
        user1 = await SerialUser.create(
            name="User One",
            metadata={"role": "admin", "last_login": "2024-01-01"},
            tags=["admin", "active"]
        )
        
        user2 = await SerialUser.create(
            name="User Two", 
            metadata={"role": "user", "preferences": {"theme": "dark"}},
            tags=["user"]
        )
        
        # Test serialization
        graph_json = graph.to_json(indent=2)
        assert "User One" in graph_json
        assert "User Two" in graph_json
        assert "admin" in graph_json
        assert "SerialUser" in graph_json
        
        # Test deserialization
        graph_dict = graph.to_dict()
        restored_graph = Graph.from_dict(graph_dict)
        
        assert restored_graph.node_count() == 2
        assert restored_graph.name == "serialization_test"
        
        # Verify node data preserved
        nodes = list(restored_graph._nodes.values())
        node_names = {node.properties["name"] for node in nodes}
        assert node_names == {"User One", "User Two"}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])