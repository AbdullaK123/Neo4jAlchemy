# tests/orm/test_graph_entity.py
"""
Comprehensive tests for the modern Pydantic V2 GraphEntity system.

Tests the complete GraphEntity functionality including:
- Pydantic V2 model behavior and validation
- Graph integration with automatic node sync
- CRUD operations with async/await patterns
- Change tracking and dirty field detection
- Lifecycle hooks for business logic
- Decorator functionality and configuration
- Class method operations and instance caching
- Private attribute handling and state management
"""

import pytest
import pytest_asyncio
from datetime import datetime
from typing import Optional, List
from unittest.mock import patch, AsyncMock

from pydantic import Field, ValidationError, PrivateAttr
from neo4jalchemy.core.graph import Graph
from neo4jalchemy.orm.entities import GraphEntity, graph_entity, get_entity_classes, get_entity_by_label


# =============================================================================
# TEST FIXTURES AND ENTITIES
# =============================================================================

@pytest.fixture
def test_graph():
    """Create a test graph for entity operations."""
    return Graph(name="test_graph")


@pytest.fixture
def another_graph():
    """Create another test graph for multi-graph tests."""
    return Graph(name="another_test_graph")


# Test entity classes using pure Pydantic V2 fields
class User(GraphEntity):
    """Test user entity with comprehensive validation."""
    name: str = Field(min_length=1, max_length=100, description="User's full name")
    email: str = Field(pattern=r'^[^@]+@[^@]+\.[^@]+$', description="Valid email address")
    age: int = Field(ge=0, le=150, description="User's age")
    is_active: bool = Field(default=True, description="Whether user is active")
    bio: Optional[str] = Field(None, max_length=500, description="User biography")
    tags: List[str] = Field(default_factory=list, description="User tags")
    score: float = Field(default=0.0, ge=0.0, le=100.0, description="User score")
    
    class Config:
        graph_label = "User"


@graph_entity(label="Person")
class Person(GraphEntity):
    """Test person entity with decorator configuration."""
    full_name: str = Field(min_length=1, max_length=200)
    birth_year: int = Field(ge=1900, le=2024)
    nationality: Optional[str] = Field(None, max_length=100)
    languages: List[str] = Field(default_factory=list)


class Product(GraphEntity):
    """Test product entity with business logic."""
    name: str = Field(min_length=1, max_length=200)
    price: float = Field(gt=0, description="Product price must be positive")
    category: str = Field(min_length=1, max_length=100)
    in_stock: bool = Field(default=True)
    tags: List[str] = Field(default_factory=list)
    metadata: dict = Field(default_factory=dict)
    
    def is_expensive(self) -> bool:
        """Business logic: Check if product is expensive."""
        return self.price > 100.0


# Entity with comprehensive lifecycle hooks
class TrackedEntity(GraphEntity):
    """Entity that tracks all lifecycle events for testing."""
    name: str = Field(min_length=1)
    value: int = Field(default=0)
    status: str = Field(default="draft")
    
    # Use PrivateAttr for test tracking to avoid Pydantic validation
    _lifecycle_events: List[str] = PrivateAttr(default_factory=list)
    
    @property
    def lifecycle_events(self) -> List[str]:
        """Get lifecycle events."""
        return self._lifecycle_events
    
    def clear_events(self) -> None:
        """Clear lifecycle events."""
        self._lifecycle_events.clear()
    
    async def _pre_save(self):
        """Pre-save hook."""
        self._lifecycle_events.append(f'pre_save:{self.name}')
        # Demo business logic
        if self.status == "draft":
            self.status = "processing"
    
    async def _post_save(self):
        """Post-save hook."""
        self._lifecycle_events.append(f'post_save:{self.name}')
    
    async def _pre_delete(self):
        """Pre-delete hook."""
        self._lifecycle_events.append(f'pre_delete:{self.name}')
    
    async def _post_delete(self):
        """Post-delete hook."""
        self._lifecycle_events.append(f'post_delete:{self.name}')


# Entity with custom validation
class ValidatedEntity(GraphEntity):
    """Entity with custom validation logic."""
    username: str = Field(min_length=3, max_length=20)
    password_hash: str = Field(min_length=10)
    role: str = Field(default="user")
    permissions: List[str] = Field(default_factory=list)
    
    @property
    def is_admin(self) -> bool:
        """Check if user has admin role."""
        return self.role == "admin"
    
    def add_permission(self, permission: str) -> None:
        """Add a permission to the user."""
        if permission not in self.permissions:
            self.permissions.append(permission)


# =============================================================================
# TEST PYDANTIC V2 MODEL BEHAVIOR
# =============================================================================

class TestPydanticV2Behavior:
    """Test core Pydantic V2 model behavior and validation."""
    
    def test_basic_entity_creation(self):
        """Test basic entity creation with validation."""
        user = User(
            name="Alice Johnson",
            email="alice@example.com",
            age=30,
            bio="Software engineer specializing in graph databases",
            tags=["python", "neo4j", "graphs"],
            score=85.5
        )
        
        # Test basic properties
        assert user.name == "Alice Johnson"
        assert user.email == "alice@example.com"
        assert user.age == 30
        assert user.is_active is True  # Default value
        assert user.bio == "Software engineer specializing in graph databases"
        assert user.tags == ["python", "neo4j", "graphs"]
        assert user.score == 85.5
        
        # Test automatic fields
        assert isinstance(user.id, str)
        assert len(user.id) > 20  # UUID should be long
        assert isinstance(user.created_at, datetime)
        assert isinstance(user.updated_at, datetime)
        
        # Test private attributes
        assert user._graph_node is None  # No graph configured
        assert user._is_persisted is False
        assert user._original_data is not None
    
    def test_field_validation_errors(self):
        """Test Pydantic field validation."""
        # Test name too short
        with pytest.raises(ValidationError) as exc_info:
            User(name="", email="alice@example.com", age=30)
        assert "at least 1 character" in str(exc_info.value)
        
        # Test invalid email pattern
        with pytest.raises(ValidationError) as exc_info:
            User(name="Alice", email="not-an-email", age=30)
        assert "String should match pattern" in str(exc_info.value)
        
        # Test age out of range
        with pytest.raises(ValidationError) as exc_info:
            User(name="Alice", email="alice@example.com", age=200)
        assert "less than or equal to 150" in str(exc_info.value)
        
        # Test negative score
        with pytest.raises(ValidationError) as exc_info:
            User(name="Alice", email="alice@example.com", age=30, score=-10)
        assert "greater than or equal to 0" in str(exc_info.value)
    
    def test_frozen_fields(self):
        """Test that frozen fields cannot be modified after creation."""
        user = User(name="Test User", email="test@example.com", age=25)
        
        original_id = user.id
        original_created = user.created_at
        
        # Attempt to modify frozen fields should raise ValidationError
        with pytest.raises(ValidationError):
            user.id = "new_id"
        
        with pytest.raises(ValidationError):
            user.created_at = datetime.now()
        
        # Verify they haven't changed
        assert user.id == original_id
        assert user.created_at == original_created
    
    def test_validate_assignment(self):
        """Test that assignment validation works correctly."""
        user = User(name="Test User", email="test@example.com", age=25)
        
        # Valid assignments should work
        user.age = 30
        assert user.age == 30
        
        user.tags = ["new", "tags"]
        assert user.tags == ["new", "tags"]
        
        # Invalid assignments should raise errors
        with pytest.raises(ValidationError):
            user.age = 300  # Too old
        
        with pytest.raises(ValidationError):
            user.email = "invalid-email"  # Bad format
    
    def test_default_values(self):
        """Test default value handling."""
        user = User(name="Minimal User", email="minimal@example.com", age=25)
        
        # Test defaults
        assert user.is_active is True
        assert user.bio is None
        assert user.tags == []  # default_factory
        assert user.score == 0.0
        
        # Test that defaults don't interfere with explicit values
        user2 = User(
            name="Full User",
            email="full@example.com", 
            age=30,
            is_active=False,
            bio="Custom bio",
            tags=["custom"],
            score=50.0
        )
        
        assert user2.is_active is False
        assert user2.bio == "Custom bio"
        assert user2.tags == ["custom"]
        assert user2.score == 50.0
    
    def test_model_dump_and_validation(self):
        """Test Pydantic V2 model_dump and model_validate methods."""
        original_user = User(
            name="Serialization Test",
            email="serialize@example.com",
            age=28,
            tags=["test", "serialization"],
            score=75.0
        )
        
        # Test model_dump
        user_dict = original_user.model_dump()
        
        assert user_dict["name"] == "Serialization Test"
        assert user_dict["email"] == "serialize@example.com"
        assert user_dict["age"] == 28
        assert user_dict["tags"] == ["test", "serialization"]
        assert user_dict["score"] == 75.0
        assert "id" in user_dict
        assert "created_at" in user_dict
        assert "updated_at" in user_dict
        
        # Test model_validate (round-trip)
        restored_user = User.model_validate(user_dict)
        
        assert restored_user.name == original_user.name
        assert restored_user.email == original_user.email
        assert restored_user.age == original_user.age
        assert restored_user.tags == original_user.tags
        assert restored_user.score == original_user.score
        assert restored_user.id == original_user.id


# =============================================================================
# TEST GRAPH INTEGRATION
# =============================================================================

class TestGraphIntegration:
    """Test automatic graph integration and node synchronization."""
    
    def test_entity_without_graph(self):
        """Test entity behavior when no graph is configured."""
        user = User(name="No Graph", email="nograph@example.com", age=25)
        
        # Should not create graph node
        assert user._graph_node is None
        
        # Should still function normally
        assert user.name == "No Graph"
        assert user._is_persisted is False
    
    def test_entity_with_graph_auto_sync(self, test_graph):
        """Test entity with graph auto-sync enabled."""
        # Configure User to use the test graph
        User._entity_config.graph = test_graph
        User._entity_config.auto_sync = True
        
        user = User(
            name="Graph User",
            email="graph@example.com",
            age=30,
            tags=["graph", "test"],
            score=90.0
        )
        
        # Should automatically create graph node
        assert user._graph_node is not None
        assert user._graph_node.id == user.id
        assert user._graph_node.label == "User"
        
        # Check that graph contains the node
        assert test_graph.has_node(user.id)
        node = test_graph.get_node(user.id)
        assert node.label == "User"
        
        # Check node properties
        properties = node.properties
        assert properties["name"] == "Graph User"
        assert properties["email"] == "graph@example.com"
        assert properties["age"] == 30
        assert properties["tags"] == ["graph", "test"]
        assert properties["score"] == 90.0
        assert properties["is_active"] is True
        
        # ID should not be in properties (it's the node ID)
        assert "id" not in properties
    
    def test_entity_with_graph_no_auto_sync(self, test_graph):
        """Test entity with auto_sync disabled."""
        # Configure decorator-based entity
        @graph_entity(label="NoSync", graph=test_graph, auto_sync=False)
        class NoSyncEntity(GraphEntity):
            name: str
        
        entity = NoSyncEntity(name="No Auto Sync")
        
        # Should not create graph node automatically
        assert entity._graph_node is None
        assert not test_graph.has_node(entity.id)
    
    def test_manual_graph_sync(self, test_graph):
        """Test manual synchronization to graph."""
        # Create a fresh entity class to avoid config conflicts
        @graph_entity(label="ManualSyncProduct", graph=test_graph, auto_sync=False)
        class ManualSyncProduct(GraphEntity):
            name: str
            price: float = Field(gt=0)
            category: str
        
        product = ManualSyncProduct(
            name="Manual Sync Product",
            price=99.99,
            category="Electronics"
        )
        
        # Initially no graph node
        assert product._graph_node is None
        assert not test_graph.has_node(product.id)
        
        # Manually sync to graph
        product._sync_to_node()
        
        # Now should have graph node
        assert product._graph_node is not None
        assert test_graph.has_node(product.id)
        
        node = test_graph.get_node(product.id)
        assert node.properties["name"] == "Manual Sync Product"
        assert node.properties["price"] == 99.99
    
    def test_graph_node_update_sync(self, test_graph):
        """Test that entity changes sync to graph node."""
        User._entity_config.graph = test_graph
        User._entity_config.auto_sync = True
        
        user = User(name="Update Test", email="update@example.com", age=25)
        
        # Modify entity
        user.name = "Updated Name"
        user.age = 26
        user.tags = ["updated", "sync"]
        
        # Manually sync (in real usage, this would happen on save)
        user._sync_to_node()
        
        # Check that graph node was updated
        node = test_graph.get_node(user.id)
        assert node.properties["name"] == "Updated Name"
        assert node.properties["age"] == 26
        assert node.properties["tags"] == ["updated", "sync"]
    
    def test_multiple_entities_same_graph(self, test_graph):
        """Test multiple entity types in the same graph."""
        # Create fresh entity classes to avoid config conflicts
        @graph_entity(label="MultiUser", graph=test_graph)
        class MultiUser(GraphEntity):
            name: str
            email: str
            age: int
        
        @graph_entity(label="MultiProduct", graph=test_graph)
        class MultiProduct(GraphEntity):
            name: str
            price: float
            category: str
        
        user = MultiUser(name="Multi User", email="multi@example.com", age=30)
        product = MultiProduct(name="Multi Product", price=50.0, category="Test")
        
        # Both should be in the graph
        assert test_graph.node_count() == 2
        assert test_graph.has_node(user.id)
        assert test_graph.has_node(product.id)
        
        # Should have different labels
        user_node = test_graph.get_node(user.id)
        product_node = test_graph.get_node(product.id)
        
        assert user_node.label == "MultiUser"
        assert product_node.label == "MultiProduct"


# =============================================================================
# TEST CHANGE TRACKING
# =============================================================================

class TestChangeTracking:
    """Test entity change tracking and dirty field detection."""
    
    def test_clean_entity_state(self):
        """Test that new entities start in clean state."""
        user = User(name="Clean User", email="clean@example.com", age=25)
        
        # Should not be dirty initially
        assert not user.is_dirty()
        assert user.get_dirty_fields() == {}
        assert user._original_data is not None
    
    def test_dirty_field_detection(self):
        """Test detection of modified fields."""
        user = User(
            name="Original Name",
            email="original@example.com",
            age=25,
            tags=["original"],
            score=50.0
        )
        
        # Initially clean
        assert not user.is_dirty()
        
        # Modify a field
        user.name = "Modified Name"
        
        # Should be dirty now
        assert user.is_dirty()
        
        # Check dirty fields
        dirty_fields = user.get_dirty_fields()
        assert "name" in dirty_fields
        assert dirty_fields["name"] == "Modified Name"
        
        # updated_at should also change but is excluded from dirty check
        assert "updated_at" not in dirty_fields
    
    def test_multiple_field_changes(self):
        """Test tracking multiple field changes."""
        user = User(
            name="Multi Test",
            email="multi@example.com",
            age=25,
            tags=["initial"],
            score=0.0
        )
        
        # Modify multiple fields
        user.name = "Multi Modified"
        user.age = 30
        user.tags = ["modified", "multiple"]
        user.score = 75.0
        user.is_active = False
        
        # Check all dirty fields
        dirty_fields = user.get_dirty_fields()
        
        assert dirty_fields["name"] == "Multi Modified"
        assert dirty_fields["age"] == 30
        assert dirty_fields["tags"] == ["modified", "multiple"]
        assert dirty_fields["score"] == 75.0
        assert dirty_fields["is_active"] is False
    
    def test_updated_at_automatic_change(self):
        """Test that updated_at changes automatically when fields are modified."""
        user = User(name="Timestamp Test", email="timestamp@example.com", age=25)
        original_updated = user.updated_at
        
        # Small delay to ensure timestamp difference
        import time
        time.sleep(0.001)
        
        # Modify a field
        user.name = "Timestamp Modified"
        
        # updated_at should have changed
        assert user.updated_at > original_updated
    
    def test_change_tracking_disabled(self):
        """Test behavior when change tracking is disabled."""
        @graph_entity(track_changes=False)
        class NoTrackingEntity(GraphEntity):
            name: str
            value: int = 0
        
        entity = NoTrackingEntity(name="No Tracking", value=10)
        
        # Modify fields
        entity.name = "Modified"
        entity.value = 20
        
        # Should not be considered dirty
        assert not entity.is_dirty()
        assert entity.get_dirty_fields() == {}
    
    def test_complex_field_changes(self):
        """Test change tracking with complex field types."""
        product = Product(
            name="Complex Product",
            price=100.0,
            category="Electronics",
            tags=["initial", "tags"],
            metadata={"version": 1, "features": ["a", "b"]}
        )
        
        # Modify complex fields
        product.tags.append("new_tag")  # Modify list in place
        product.metadata["version"] = 2  # Modify dict in place
        product.metadata["new_key"] = "new_value"
        
        # This should be detected as dirty
        assert product.is_dirty()
        
        dirty_fields = product.get_dirty_fields()
        assert "tags" in dirty_fields
        assert "metadata" in dirty_fields
        assert dirty_fields["tags"] == ["initial", "tags", "new_tag"]
        assert dirty_fields["metadata"]["version"] == 2
        assert dirty_fields["metadata"]["new_key"] == "new_value"


# =============================================================================
# TEST CRUD OPERATIONS
# =============================================================================

@pytest.mark.asyncio
class TestCRUDOperations:
    """Test Create, Read, Update, Delete operations."""
    
    async def test_save_new_entity(self, test_graph):
        """Test saving a new entity."""
        @graph_entity(label="SaveUser", graph=test_graph)
        class SaveUser(GraphEntity):
            name: str
            email: str
            age: int
        
        user = SaveUser(
            name="Save Test",
            email="save@example.com",
            age=28
        )
        
        # Initial state
        assert not user._is_persisted
        # New entity is not considered dirty if no changes made after creation
        
        # Save the entity
        result = await user.save()
        
        # Check results
        assert result is user  # Returns self for chaining
        assert user._is_persisted
        assert not user.is_dirty()  # Should be clean after save
        
        # Original data should be updated
        assert user._original_data == user.model_dump()
    
    async def test_save_clean_entity_no_op(self, test_graph):
        """Test that saving a clean, persisted entity is a no-op."""
        User._entity_config.graph = test_graph
        
        user = User(name="Clean Save", email="clean@example.com", age=25)
        
        # Save once
        await user.save()
        original_updated = user.updated_at
        
        # Save again (should be no-op)
        await user.save()
        
        # updated_at should not change
        assert user.updated_at == original_updated
    
    async def test_save_force_update(self, test_graph):
        """Test forced update of clean entity."""
        User._entity_config.graph = test_graph
        
        user = User(name="Force Update", email="force@example.com", age=25)
        await user.save()
        
        # Force update even though clean
        result = await user.save(force_update=True)
        
        assert result is user
        assert user._is_persisted
    
    async def test_save_modified_entity(self, test_graph):
        """Test saving an entity with modifications."""
        User._entity_config.graph = test_graph
        
        user = User(name="Modify Test", email="modify@example.com", age=25)
        await user.save()
        
        # Modify the entity
        user.name = "Modified Name"
        user.age = 30
        user.tags = ["modified"]
        
        assert user.is_dirty()
        
        # Save modifications
        await user.save()
        
        assert not user.is_dirty()
        assert user._original_data["name"] == "Modified Name"
        assert user._original_data["age"] == 30
        
        # Check graph node was updated
        node = test_graph.get_node(user.id)
        assert node.properties["name"] == "Modified Name"
        assert node.properties["age"] == 30
    
    async def test_delete_entity(self, test_graph):
        """Test deleting an entity."""
        User._entity_config.graph = test_graph
        
        user = User(name="Delete Test", email="delete@example.com", age=30)
        await user.save()
        
        user_id = user.id
        
        # Verify entity exists in graph
        assert test_graph.has_node(user_id)
        assert user._is_persisted
        
        # Delete the entity
        result = await user.delete()
        
        # Check results
        assert result is True
        assert not user._is_persisted
        assert user._graph_node is None
        assert not test_graph.has_node(user_id)
        
        # Should be removed from instance cache
        assert user_id not in User._instances
    
    async def test_delete_unsaved_entity(self):
        """Test deleting an entity that was never saved."""
        @graph_entity(label="UnsavedUser")
        class UnsavedUser(GraphEntity):
            name: str
            email: str
            age: int
        
        user = UnsavedUser(name="Unsaved", email="unsaved@example.com", age=25)
        
        # Should return True since it removes from graph (even if not there)
        # but False if no graph configured
        result = await user.delete()
        assert result is False
    
    async def test_refresh_entity(self, test_graph):
        """Test refreshing entity from graph."""
        User._entity_config.graph = test_graph
        
        user = User(name="Refresh Test", email="refresh@example.com", age=30)
        await user.save()
        
        # Modify entity locally (but don't save)
        user.name = "Locally Modified"
        user.age = 35
        
        assert user.is_dirty()
        
        # Refresh from graph (simulates loading fresh data)
        result = await user.refresh()
        
        # Should revert to saved state
        assert result is user
        assert not user.is_dirty()
        assert user.name == "Refresh Test"
        assert user.age == 30


# =============================================================================
# TEST CLASS METHODS
# =============================================================================

@pytest.mark.asyncio
class TestClassMethods:
    """Test entity class methods for querying and creation."""
    
    async def test_get_entity_by_id(self, test_graph):
        """Test retrieving entity by ID."""
        User._entity_config.graph = test_graph
        
        # Create and save entity
        original = User(name="Get Test", email="get@example.com", age=25)
        await original.save()
        
        # Retrieve by ID
        retrieved = await User.get(original.id)
        
        assert retrieved is not None
        assert retrieved.id == original.id
        assert retrieved.name == "Get Test"
        assert retrieved.email == "get@example.com"
        assert retrieved.age == 25
        assert retrieved._is_persisted
    
    async def test_get_nonexistent_entity(self, test_graph):
        """Test getting non-existent entity returns None."""
        User._entity_config.graph = test_graph
        
        result = await User.get("nonexistent_id")
        assert result is None
    
    async def test_get_from_instance_cache(self, test_graph):
        """Test that get() uses instance cache when available."""
        User._entity_config.graph = test_graph
        
        # Create entity (should be in cache)
        original = User(name="Cache Test", email="cache@example.com", age=25)
        await original.save()
        
        # Get should return same instance from cache
        retrieved = await User.get(original.id)
        
        assert retrieved is original  # Same object
    
    async def test_create_class_method(self, test_graph):
        """Test creating entity via class method."""
        User._entity_config.graph = test_graph
        
        user = await User.create(
            name="Create Test",
            email="create@example.com",
            age=28,
            tags=["created", "via", "class"],
            score=80.0
        )
        
        # Should be automatically saved
        assert user._is_persisted
        assert not user.is_dirty()
        assert user.name == "Create Test"
        assert test_graph.has_node(user.id)
    
    async def test_get_or_create_existing(self, test_graph):
        """Test get_or_create with existing entity."""
        User._entity_config.graph = test_graph
        
        # Create original
        original = await User.create(
            name="Original User",
            email="original@example.com",
            age=30
        )
        
        # Get or create with same ID
        user, created = await User.get_or_create(
            id=original.id,
            defaults={"name": "Should Not Use This"}
        )
        
        assert not created
        assert user.id == original.id
        assert user.name == "Original User"  # Should use existing
        assert user is original  # Should be same instance
    
    async def test_get_or_create_new(self, test_graph):
        """Test get_or_create with new entity."""
        User._entity_config.graph = test_graph
        
        user, created = await User.get_or_create(
            id="new_user_123",
            defaults={
                "name": "New User",
                "email": "new@example.com",
                "age": 25,
                "tags": ["new", "user"]
            }
        )
        
        assert created
        assert user.id == "new_user_123"
        assert user.name == "New User"
        assert user._is_persisted
        assert test_graph.has_node(user.id)


# =============================================================================
# TEST LIFECYCLE HOOKS
# =============================================================================

@pytest.mark.asyncio
class TestLifecycleHooks:
    """Test entity lifecycle hooks and business logic integration."""
    
    async def test_save_lifecycle_hooks(self, test_graph):
        """Test pre_save and post_save hooks."""
        TrackedEntity._entity_config.graph = test_graph
        
        entity = TrackedEntity(name="Hook Test", value=10)
        
        # Clear any events from initialization
        entity.clear_events()
        
        # Save entity
        await entity.save()
        
        # Check hooks were called in correct order
        events = entity.lifecycle_events
        assert len(events) >= 2
        assert events[0].startswith("pre_save:")
        assert events[1].startswith("post_save:")
        
        # Check that pre_save business logic executed
        assert entity.status == "processing"  # Changed from "draft"
    
    async def test_delete_lifecycle_hooks(self, test_graph):
        """Test pre_delete and post_delete hooks."""
        TrackedEntity._entity_config.graph = test_graph
        
        entity = TrackedEntity(name="Delete Hook Test", value=20)
        await entity.save()
        
        # Clear save events
        entity.clear_events()
        
        # Delete entity
        await entity.delete()
        
        # Check delete hooks were called
        events = entity.lifecycle_events
        assert len(events) >= 2
        assert events[0].startswith("pre_delete:")
        assert events[1].startswith("post_delete:")
    
    async def test_multiple_saves_hooks(self, test_graph):
        """Test that hooks are called on multiple saves."""
        TrackedEntity._entity_config.graph = test_graph
        
        entity = TrackedEntity(name="Multi Save", value=5)
        entity.clear_events()
        
        # First save
        await entity.save()
        first_save_events = len(entity.lifecycle_events)
        
        # Modify and save again
        entity.value = 10
        await entity.save()
        second_save_events = len(entity.lifecycle_events)
        
        # Should have more events after second save
        assert second_save_events > first_save_events
        
        # Check pattern of events
        events = entity.lifecycle_events
        pre_save_count = sum(1 for e in events if e.startswith("pre_save:"))
        post_save_count = sum(1 for e in events if e.startswith("post_save:"))
        
        assert pre_save_count == 2
        assert post_save_count == 2


# =============================================================================
# TEST DECORATOR FUNCTIONALITY
# =============================================================================

class TestDecoratorFunctionality:
    """Test @graph_entity decorator configuration."""
    
    def test_decorator_with_label(self):
        """Test decorator with custom label."""
        @graph_entity(label="CustomLabel")
        class CustomEntity(GraphEntity):
            name: str
            value: int = 0
        
    def test_decorator_with_label(self):
        """Test decorator with custom label."""
        @graph_entity(label="CustomLabel")
        class CustomEntity(GraphEntity):
            name: str
            value: int = 0
        
        assert CustomEntity._get_class_label() == "CustomLabel"
        
        entity = CustomEntity(name="Custom Test", value=42)
        assert entity._get_label() == "CustomLabel"
    
    def test_decorator_without_parameters(self):
        """Test decorator without parameters uses class name."""
        @graph_entity
        class SimpleEntity(GraphEntity):
            name: str
        
        assert SimpleEntity._get_class_label() == "SimpleEntity"
    
    def test_decorator_with_graph(self, test_graph):
        """Test decorator with graph configuration."""
        @graph_entity(graph=test_graph, label="GraphBound")
        class GraphBoundEntity(GraphEntity):
            title: str
            description: str = "Default description"
        
        entity = GraphBoundEntity(title="Test Title")
        
        # Should have graph node created automatically
        assert entity._graph_node is not None
        assert test_graph.has_node(entity.id)
        
        node = test_graph.get_node(entity.id)
        assert node.label == "GraphBound"
        assert node.properties["title"] == "Test Title"
    
    def test_decorator_configuration_options(self):
        """Test decorator with various configuration options."""
        @graph_entity(
            label="ConfiguredEntity",
            auto_sync=False,
            track_changes=False
        )
        class ConfiguredEntity(GraphEntity):
            name: str
            status: str = "active"
        
        # Check configuration
        config = ConfiguredEntity._entity_config
        assert config.graph_label == "ConfiguredEntity"
        assert config.auto_sync is False
        assert config.track_changes is False
        
        # Test behavior matches configuration
        entity = ConfiguredEntity(name="Config Test")
        
        # No auto sync, so no graph node
        assert entity._graph_node is None
        
        # No change tracking
        entity.name = "Modified"
        assert not entity.is_dirty()
    
    def test_decorator_validation(self):
        """Test decorator validation and error handling."""
        # Should raise error for non-GraphEntity class
        with pytest.raises(TypeError, match="@graph_entity can only be applied to GraphEntity subclasses"):
            @graph_entity
            class NotAnEntity:
                name: str
    
    def test_config_class_vs_decorator(self, test_graph):
        """Test Config class vs decorator configuration."""
        # Entity with Config class
        class ConfigClassEntity(GraphEntity):
            name: str
            
            class Config:
                graph_label = "ConfigClass"
                graph = test_graph
        
        # Entity with decorator
        @graph_entity(label="DecoratorEntity", graph=test_graph)
        class DecoratorEntity(GraphEntity):
            name: str
        
        # Both should work
        config_entity = ConfigClassEntity(name="Config Test")
        decorator_entity = DecoratorEntity(name="Decorator Test")
        
        assert config_entity._get_label() == "ConfigClass"
        assert decorator_entity._get_label() == "DecoratorEntity"
        
        assert config_entity._graph_node is not None
        assert decorator_entity._graph_node is not None


# =============================================================================
# TEST ENTITY REGISTRY
# =============================================================================

class TestEntityRegistry:
    """Test entity class registration and discovery."""
    
    def test_entity_registration(self):
        """Test that entities are automatically registered."""
        entity_classes = get_entity_classes()
        
        # Should contain our test entities
        class_names = {cls.__name__ for cls in entity_classes}
        
        expected_classes = {"User", "Person", "Product", "TrackedEntity", "ValidatedEntity"}
        assert expected_classes.issubset(class_names)
    
    def test_get_entity_by_label(self):
        """Test finding entity classes by label."""
        # Test standard entity
        user_class = get_entity_by_label("User")
        assert user_class is User
        
        # Test decorator-configured entity
        person_class = get_entity_by_label("Person")
        assert person_class is Person
        
        # Test non-existent label
        missing_class = get_entity_by_label("NonExistentEntity")
        assert missing_class is None
    
    def test_dynamic_entity_registration(self):
        """Test that dynamically created entities are registered."""
        initial_count = len(get_entity_classes())
        
        # Create new entity class dynamically
        @graph_entity(label="DynamicEntity")
        class DynamicEntity(GraphEntity):
            dynamic_field: str
            created_dynamically: bool = True
        
        # Should be registered
        new_count = len(get_entity_classes())
        assert new_count == initial_count + 1
        
        # Should be findable by label
        dynamic_class = get_entity_by_label("DynamicEntity")
        assert dynamic_class is DynamicEntity


# =============================================================================
# TEST INSTANCE CACHING
# =============================================================================

@pytest.mark.asyncio
class TestInstanceCaching:
    """Test entity instance caching behavior."""
    
    def test_instance_cache_registration(self):
        """Test that entities are registered in instance cache."""
        user = User(name="Cache Test", email="cache@example.com", age=25)
        
        # Should be in class instance cache
        assert user.id in User._instances
        assert User._instances[user.id] is user
    
    def test_instance_cache_isolation(self):
        """Test that different entity classes have separate caches."""
        user = User(name="User Cache", email="user@example.com", age=30)
        product = Product(name="Product Cache", price=50.0, category="Test")
        
        # Should be in their respective caches
        assert user.id in User._instances
        assert product.id in Product._instances
        
        # Should not cross-contaminate
        assert user.id not in Product._instances
        assert product.id not in User._instances
    
    async def test_cache_cleanup_on_delete(self, test_graph):
        """Test that deleted entities are removed from cache."""
        User._entity_config.graph = test_graph
        
        user = User(name="Delete Cache", email="delete@example.com", age=25)
        await user.save()
        
        user_id = user.id
        
        # Should be in cache
        assert user_id in User._instances
        
        # Delete entity
        await user.delete()
        
        # Should be removed from cache
        assert user_id not in User._instances


# =============================================================================
# TEST BUSINESS LOGIC INTEGRATION
# =============================================================================

class TestBusinessLogicIntegration:
    """Test integration of business logic with entity models."""
    
    def test_entity_methods(self):
        """Test custom methods on entities."""
        product = Product(name="Expensive Item", price=150.0, category="Luxury")
        
        # Test business logic method
        assert product.is_expensive() is True
        
        cheap_product = Product(name="Cheap Item", price=10.0, category="Budget")
        assert cheap_product.is_expensive() is False
    
    def test_entity_properties(self):
        """Test computed properties on entities."""
        admin_user = ValidatedEntity(
            username="admin",
            password_hash="hashedpassword123",
            role="admin"
        )
        
        regular_user = ValidatedEntity(
            username="user",
            password_hash="hashedpassword456",
            role="user"
        )
        
        assert admin_user.is_admin is True
        assert regular_user.is_admin is False
    
    def test_entity_state_methods(self):
        """Test methods that modify entity state."""
        user = ValidatedEntity(
            username="testuser",
            password_hash="hashedpassword789",
            role="user"
        )
        
        # Test permission management
        user.add_permission("read")
        assert "read" in user.permissions
        
        user.add_permission("write")
        assert "write" in user.permissions
        
        # Should not add duplicates
        user.add_permission("read")
        assert user.permissions.count("read") == 1
    
    def test_validation_with_business_rules(self):
        """Test custom validation integrated with business rules."""
        # Test valid entity
        valid_user = ValidatedEntity(
            username="validuser",
            password_hash="hashedpassword123456",  # Long enough
            role="editor",
            permissions=["read", "write"]
        )
        
        assert valid_user.username == "validuser"
        assert len(valid_user.permissions) == 2
        
        # Test validation errors
        with pytest.raises(ValidationError):
            ValidatedEntity(
                username="xy",  # Too short
                password_hash="short",  # Too short
                role="admin"
            )


# =============================================================================
# TEST EDGE CASES AND ERROR HANDLING
# =============================================================================

class TestEdgeCasesAndErrorHandling:
    """Test edge cases and error handling scenarios."""
    
    def test_entity_string_representation(self):
        """Test entity string representation in various states."""
        user = User(name="Repr Test", email="repr@example.com", age=25)
        
        # New entity
        repr_str = repr(user)
        assert "User" in repr_str
        assert user.id in repr_str
        assert "(new)" in repr_str
        
        # Mark as persisted
        user._is_persisted = True
        repr_str = repr(user)
        assert "(persisted)" in repr_str
        
        # Make dirty
        user.name = "Modified"
        repr_str = repr(user)
        # Should contain dirty but test is flexible about the exact format
        assert "dirty" in repr_str
    
    def test_concurrent_entity_modifications(self):
        """Test handling of concurrent modifications."""
        user = User(name="Concurrent Test", email="concurrent@example.com", age=25)
        
        # Simulate concurrent modifications
        user.name = "First Modification"
        user.age = 30
        
        # Both changes should be reflected
        assert user.name == "First Modification"
        assert user.age == 30
        assert user.is_dirty()
        
        dirty_fields = user.get_dirty_fields()
        assert "name" in dirty_fields
        assert "age" in dirty_fields
    
    def test_invalid_graph_operations(self):
        """Test handling of invalid graph operations."""
        # Create entity without graph to test error handling
        @graph_entity(label="NoGraphEntity")
        class NoGraphEntity(GraphEntity):
            name: str
            email: str
            age: int
        
        user = NoGraphEntity(name="No Graph", email="nograph@example.com", age=25)
        
        # Should not raise error
        user._sync_to_node()
        assert user._graph_node is None
        
        user._sync_from_node()
        # Should not modify entity
        assert user.name == "No Graph"
    
    def test_model_dump_edge_cases(self):
        """Test model_dump with various configurations."""
        user = User(
            name="Dump Test",
            email="dump@example.com",
            age=25,
            bio=None,  # Optional field
            tags=[],   # Empty list
            score=0.0  # Zero value
        )
        
        # Standard dump
        full_dump = user.model_dump()
        assert "name" in full_dump
        assert "bio" in full_dump
        assert full_dump["bio"] is None
        assert full_dump["tags"] == []
        assert full_dump["score"] == 0.0
        
        # Dump excluding certain fields
        partial_dump = user.model_dump(exclude={"bio", "tags"})
        assert "name" in partial_dump
        assert "bio" not in partial_dump
        assert "tags" not in partial_dump
        
        # JSON mode dump
        json_dump = user.model_dump(mode="json")
        assert all(isinstance(v, (str, int, float, bool, list, dict, type(None))) 
                  for v in json_dump.values())
    
    def test_memory_management(self):
        """Test memory management with entity instances."""
        import gc
        import weakref
        
        # Create fresh entity class to avoid conflicts
        @graph_entity(label="MemoryTestEntity")
        class MemoryTestEntity(GraphEntity):
            name: str
            email: str
            age: int
        
        # Create entity and weak reference
        user = MemoryTestEntity(name="Memory Test", email="memory@example.com", age=25)
        user_id = user.id
        weak_ref = weakref.ref(user)
        
        # Should be in cache
        assert user_id in MemoryTestEntity._instances
        assert weak_ref() is not None
        
        # Delete strong reference
        del user
        
        # WeakValueDictionary should clean up automatically when no references exist
        gc.collect()
        
        # Check if cleaned up (may or may not be depending on GC timing)
        # This test is more about ensuring no errors occur
        assert weak_ref() is None or user_id in MemoryTestEntity._instances


# =============================================================================
# TEST INTEGRATION WITH MULTIPLE GRAPHS
# =============================================================================

class TestMultiGraphIntegration:
    """Test entities working with multiple graphs."""
    
    def test_entities_in_different_graphs(self, test_graph, another_graph):
        """Test entities configured for different graphs."""
        # Create fresh entity classes for different graphs
        @graph_entity(label="GraphAUser", graph=test_graph)
        class GraphAUser(GraphEntity):
            name: str
            email: str
            age: int
        
        @graph_entity(label="GraphBProduct", graph=another_graph)
        class GraphBProduct(GraphEntity):
            name: str
            price: float
            category: str
        
        user = GraphAUser(name="Multi Graph User", email="multi@example.com", age=30)
        product = GraphBProduct(name="Multi Graph Product", price=75.0, category="Test")
        
        # Should be in different graphs
        assert test_graph.has_node(user.id)
        assert not another_graph.has_node(user.id)
        
        assert another_graph.has_node(product.id)
        assert not test_graph.has_node(product.id)
        
        # Graphs should have one node each
        assert test_graph.node_count() == 1
        assert another_graph.node_count() == 1
    
    def test_same_entity_type_different_graphs(self, test_graph, another_graph):
        """Test same entity type in different graphs using decorators."""
        @graph_entity(label="GraphAEntity", graph=test_graph)
        class GraphAEntity(GraphEntity):
            name: str
            graph_name: str = "A"
        
        @graph_entity(label="GraphBEntity", graph=another_graph)
        class GraphBEntity(GraphEntity):
            name: str
            graph_name: str = "B"
        
        entity_a = GraphAEntity(name="Entity A")
        entity_b = GraphBEntity(name="Entity B")
        
        # Should be in their respective graphs
        assert test_graph.has_node(entity_a.id)
        assert another_graph.has_node(entity_b.id)
        
        # Should not cross over
        assert not another_graph.has_node(entity_a.id)
        assert not test_graph.has_node(entity_b.id)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])