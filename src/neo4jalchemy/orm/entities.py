# src/neo4jalchemy/orm/entities.py
"""
Neo4jAlchemy GraphEntity - Modern Pydantic V2 Native Implementation

This module provides a completely modern GraphEntity system built specifically
for Pydantic V2, leveraging all its advanced features and best practices.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Type, ClassVar, Set, List, Union, TypeVar
from datetime import datetime
import uuid
import weakref

from pydantic import BaseModel, Field, ConfigDict, model_validator, PrivateAttr
from pydantic.fields import FieldInfo

from neo4jalchemy.core.graph import Graph
from neo4jalchemy.core.graph_node import GraphNode


# Type variable for GraphEntity subclasses
EntityType = TypeVar('EntityType', bound='GraphEntity')


class GraphEntityConfig:
    """Configuration for GraphEntity classes."""
    
    def __init__(
        self,
        graph_label: Optional[str] = None,
        graph: Optional[Graph] = None,
        auto_sync: bool = True,
        track_changes: bool = True
    ):
        self.graph_label = graph_label
        self.graph = graph
        self.auto_sync = auto_sync
        self.track_changes = track_changes


class GraphEntityMeta(type(BaseModel)):
    """
    Modern Pydantic V2 metaclass for GraphEntity.
    
    Handles entity registration, configuration, and graph integration
    using Pydantic V2's advanced metaclass features.
    """
    
    def __new__(
        mcs, 
        name: str, 
        bases: tuple[type, ...], 
        namespace: dict[str, Any],
        **kwargs: Any
    ) -> GraphEntityMeta:
        
        # Extract entity configuration before Pydantic processes the class
        entity_config = namespace.pop('_entity_config', None)
        if entity_config is None:
            # Use default configuration (no more Config class support)
            entity_config = GraphEntityConfig(graph_label=name)
        
        # Create the Pydantic model
        cls = super().__new__(mcs, name, bases, namespace, **kwargs)
        
        # Skip processing for the base GraphEntity class
        if name == 'GraphEntity':
            return cls
        
        # Attach configuration
        cls._entity_config = entity_config
        
        # Initialize class-level registries
        cls._instances: weakref.WeakValueDictionary[str, GraphEntity] = weakref.WeakValueDictionary()
        
        # Register the entity class
        if not hasattr(GraphEntity, '_entity_registry'):
            GraphEntity._entity_registry = weakref.WeakSet()
        GraphEntity._entity_registry.add(cls)
        
        return cls


class GraphEntity(BaseModel, metaclass=GraphEntityMeta):
    """
    Modern Pydantic V2 GraphEntity base class.
    
    Features:
    - Full Pydantic V2 compatibility and validation
    - Automatic graph node synchronization
    - Smart change tracking with dirty field detection
    - Async CRUD operations
    - Lifecycle hooks for business logic
    - Type-safe field access
    - Automatic entity registration and caching
    
    Example:
        ```python
        class User(GraphEntity):
            name: str = Field(min_length=1, max_length=100)
            email: str = Field(pattern=r'^[^@]+@[^@]+\\.[^@]+$')
            age: int = Field(ge=0, le=150)
            is_active: bool = True
            
            class Config:
                graph_label = "User"
                graph = my_graph
        
        user = await User.create(
            name="Alice Johnson",
            email="alice@example.com", 
            age=30
        )
        ```
    """
    
    # Core identity and metadata
    id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique entity identifier",
        frozen=True  # ID cannot be changed after creation
    )
    
    created_at: datetime = Field(
        default_factory=datetime.now,
        description="When the entity was created",
        frozen=True  # Creation time cannot be changed
    )
    
    updated_at: datetime = Field(
        default_factory=datetime.now,
        description="When the entity was last updated"
    )
    
    # Pydantic V2 configuration
    model_config = ConfigDict(
        # Validation behavior
        validate_assignment=True,
        validate_default=True,
        use_enum_values=True,
        
        # Extra field behavior
        extra='forbid',
        
        # Performance optimizations
        arbitrary_types_allowed=True,
        
        # Serialization
        ser_json_nan='null',
        
        # JSON schema
        json_schema_mode='validation',
    )
    
    # Class-level attributes
    _entity_registry: ClassVar[weakref.WeakSet] = weakref.WeakSet()
    _instances: ClassVar[weakref.WeakValueDictionary[str, GraphEntity]]
    _entity_config: ClassVar[GraphEntityConfig]
    
    # Private attributes (Pydantic V2 way to handle internal state)
    _graph_node: Optional[GraphNode] = PrivateAttr(default=None)
    _is_persisted: bool = PrivateAttr(default=False)
    _original_data: Optional[Dict[str, Any]] = PrivateAttr(default=None)
    
    def __init__(self, **data: Any) -> None:
        """Initialize GraphEntity with automatic graph integration."""
        super().__init__(**data)
        
        # Register this instance
        self._instances[self.id] = self
        
        # Store original data for change tracking
        self._original_data = self.model_dump()
        
        # Create graph node if auto_sync is enabled
        if self._entity_config.auto_sync:
            self._create_graph_node()
    
    @model_validator(mode='after')
    def _update_timestamp(self) -> GraphEntity:
        """Update timestamp when model changes."""
        if self._entity_config.track_changes and self._original_data:
            current_data = self.model_dump(exclude={'updated_at'})
            original_data = self._original_data.copy()
            original_data.pop('updated_at', None)
            
            if current_data != original_data:
                # Use object.__setattr__ to bypass frozen field restriction
                object.__setattr__(self, 'updated_at', datetime.now())
        
        return self
    
    def _create_graph_node(self) -> None:
        """Create and sync with graph node."""
        graph = self._get_graph()
        if graph is None:
            return
        
        properties = self._get_node_properties()
        
        self._graph_node = graph.add_node(
            node_id=self.id,
            label=self._get_label(),
            properties=properties
        )
    
    def _get_graph(self) -> Optional[Graph]:
        """Get the graph instance for this entity."""
        return self._entity_config.graph
    
    def _get_label(self) -> str:
        """Get the graph label for this entity."""
        return self._entity_config.graph_label
    
    def _get_node_properties(self) -> Dict[str, Any]:
        """Get properties to store in graph node."""
        # Use Pydantic's model_dump with proper exclusions
        return self.model_dump(
            exclude={'id'},
            mode='json'  # Ensure JSON-serializable output
        )
    
    def _sync_to_node(self) -> None:
        """Synchronize entity state to graph node."""
        graph = self._get_graph()
        if graph is None:
            return
            
        if self._graph_node is None:
            self._create_graph_node()
            return
        
        properties = self._get_node_properties()
        self._graph_node.update_properties(properties)
    
    def _sync_from_node(self) -> None:
        """Synchronize entity state from graph node."""
        if self._graph_node is None:
            return
        
        # Update entity from node properties
        node_data = {'id': self.id}
        node_data.update(self._graph_node.properties)
        
        # Use Pydantic's model validation for type safety
        validated_data = self.__class__.model_validate(node_data)
        
        # Update self with validated data
        for field_name, field_value in validated_data.model_dump().items():
            if field_name not in {'id', 'created_at'}:  # Don't update frozen fields
                setattr(self, field_name, field_value)
    
    # =============================================================================
    # CRUD OPERATIONS
    # =============================================================================
    
    async def save(self, *, force_update: bool = False) -> GraphEntity:
        """
        Save entity to the graph/database.
        
        Args:
            force_update: Force update even if no changes detected
            
        Returns:
            Self for method chaining
        """
        if not self.is_dirty() and not force_update and self._is_persisted:
            return self
        
        # Pre-save hook
        await self._pre_save()
        
        # Sync to graph node
        self._sync_to_node()
        
        # Mark as persisted and update tracking data
        self._is_persisted = True
        self._original_data = self.model_dump()
        
        # Post-save hook
        await self._post_save()
        
        return self
    
    async def delete(self) -> bool:
        """
        Delete entity from graph/database.
        
        Returns:
            True if deleted, False if not found
        """
        # Pre-delete hook
        await self._pre_delete()
        
        graph = self._get_graph()
        if graph and self._graph_node:
            success = graph.remove_node(self.id)
            if success:
                self._graph_node = None
                self._is_persisted = False
                
                # Remove from instance registry
                if self.id in self._instances:
                    del self._instances[self.id]
                
                # Post-delete hook
                await self._post_delete()
                
                return True
        
        return False
    
    async def refresh(self) -> GraphEntity:
        """
        Refresh entity from database/graph.
        
        Returns:
            Self for method chaining
        """
        # Sync from graph node
        self._sync_from_node()
        
        # Update tracking data
        self._original_data = self.model_dump()
        
        return self
    
    # =============================================================================
    # LIFECYCLE HOOKS
    # =============================================================================
    
    async def _pre_save(self) -> None:
        """Hook called before saving entity. Override in subclasses."""
        pass
    
    async def _post_save(self) -> None:
        """Hook called after saving entity. Override in subclasses."""
        pass
    
    async def _pre_delete(self) -> None:
        """Hook called before deleting entity. Override in subclasses."""
        pass
    
    async def _post_delete(self) -> None:
        """Hook called after deleting entity. Override in subclasses."""
        pass
    
    # =============================================================================
    # CLASS METHODS
    # =============================================================================
    
    @classmethod
    async def get(cls: Type[EntityType], entity_id: str) -> Optional[EntityType]:
        """
        Get entity by ID.
        
        Args:
            entity_id: Entity identifier
            
        Returns:
            Entity instance or None if not found
        """
        # Check instance cache first
        if entity_id in cls._instances:
            return cls._instances[entity_id]
        
        # Query from graph
        graph = cls._get_class_graph()
        if graph:
            node = graph.get_node(entity_id)
            if node and node.label == cls._get_class_label():
                return cls._from_node(node)
        
        return None
    
    @classmethod
    async def create(cls: Type[EntityType], **kwargs: Any) -> EntityType:
        """
        Create and save new entity.
        
        Args:
            **kwargs: Entity field values
            
        Returns:
            Created entity instance
        """
        entity = cls(**kwargs)
        await entity.save()
        return entity
    
    @classmethod
    async def get_or_create(
        cls: Type[EntityType], 
        defaults: Optional[Dict[str, Any]] = None, 
        **kwargs: Any
    ) -> tuple[EntityType, bool]:
        """
        Get existing entity or create new one.
        
        Args:
            defaults: Default values for creation
            **kwargs: Lookup criteria
            
        Returns:
            (entity, created) tuple
        """
        if 'id' in kwargs:
            entity = await cls.get(kwargs['id'])
            if entity:
                return entity, False
        
        # Create new entity
        create_data = {**(defaults or {}), **kwargs}
        entity = await cls.create(**create_data)
        return entity, True
    
    @classmethod
    def _get_class_graph(cls) -> Optional[Graph]:
        """Get graph for this entity class."""
        return cls._entity_config.graph
    
    @classmethod
    def _get_class_label(cls) -> str:
        """Get label for this entity class."""
        return cls._entity_config.graph_label
    
    @classmethod
    def _from_node(cls: Type[EntityType], node: GraphNode) -> EntityType:
        """Create entity instance from graph node."""
        # Prepare data for entity creation
        data = {'id': node.id}
        data.update(node.properties)
        
        # Use Pydantic's validation
        entity = cls.model_validate(data)
        entity._graph_node = node
        entity._is_persisted = True
        entity._original_data = entity.model_dump()
        
        return entity
    
    # =============================================================================
    # UTILITY METHODS
    # =============================================================================
    
    def is_dirty(self) -> bool:
        """Check if entity has unsaved changes."""
        if not self._entity_config.track_changes or not self._original_data:
            return False
        
        current_data = self.model_dump(exclude={'updated_at'})
        original_data = self._original_data.copy()
        original_data.pop('updated_at', None)
        
        return current_data != original_data
    
    def get_dirty_fields(self) -> Dict[str, Any]:
        """Get fields that have changed."""
        if not self.is_dirty():
            return {}
        
        current_data = self.model_dump()
        dirty_fields = {}
        
        for key, current_value in current_data.items():
            if key == 'updated_at':
                continue
            original_value = self._original_data.get(key)
            if current_value != original_value:
                dirty_fields[key] = current_value
        
        return dirty_fields
    
    def to_graph_dict(self) -> Dict[str, Any]:
        """Convert entity to graph-compatible dictionary."""
        return {
            'id': self.id,
            'label': self._get_label(),
            'properties': self._get_node_properties()
        }
    
    def __repr__(self) -> str:
        """String representation of entity."""
        status_parts = []
        if self._is_persisted:
            status_parts.append("persisted")
        else:
            status_parts.append("new")
        
        if self.is_dirty():
            status_parts.append("dirty")
        
        status = f" ({', '.join(status_parts)})" if status_parts else ""
        
        return f"{self.__class__.__name__}(id='{self.id}'{status})"


# =============================================================================
# DECORATOR FUNCTION
# =============================================================================

def graph_entity(
    cls: Optional[Type] = None,
    *,
    label: Optional[str] = None,
    graph: Optional[Graph] = None,
    auto_sync: bool = True,
    track_changes: bool = True
) -> Union[Type[GraphEntity], callable]:
    """
    Modern Pydantic V2 decorator for graph entities.
    
    Args:
        cls: The class being decorated
        label: Override the default label (class name)
        graph: Graph instance to use for this entity
        auto_sync: Whether to automatically sync with graph nodes
        track_changes: Whether to track field changes
    
    Returns:
        Decorated class or decorator function
    
    Example:
        ```python
        @graph_entity(label="Person", graph=my_graph)
        class User(GraphEntity):
            name: str = Field(min_length=1)
            email: str = Field(pattern=r'^[^@]+@[^@]+\\.[^@]+$')
        ```
    """
    def decorator(target_cls: Type) -> Type:
        if not issubclass(target_cls, GraphEntity):
            raise TypeError("@graph_entity can only be applied to GraphEntity subclasses")
        
        # Create entity configuration
        entity_config = GraphEntityConfig(
            graph_label=label or target_cls.__name__,
            graph=graph,
            auto_sync=auto_sync,
            track_changes=track_changes
        )
        
        # Set configuration on class
        target_cls._entity_config = entity_config
        
        return target_cls
    
    if cls is None:
        return decorator
    else:
        return decorator(cls)


# =============================================================================
# REGISTRY FUNCTIONS
# =============================================================================

def get_entity_classes() -> Set[Type[GraphEntity]]:
    """Get all registered GraphEntity classes."""
    if hasattr(GraphEntity, '_entity_registry'):
        return set(GraphEntity._entity_registry)
    return set()


def get_entity_by_label(label: str) -> Optional[Type[GraphEntity]]:
    """Get entity class by its graph label."""
    for entity_class in get_entity_classes():
        if entity_class._get_class_label() == label:
            return entity_class
    return None