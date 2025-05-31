# src/neo4jalchemy/orm/relationships.py
"""
Neo4jAlchemy GraphRelationship - Pure Pydantic V2 Relationship System

This module provides type-safe graph relationship modeling with automatic
edge synchronization, following SQLAlchemy patterns for familiar developer experience.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Type, ClassVar, Set, List, Union, TypeVar, TYPE_CHECKING
from datetime import datetime
import uuid
import weakref

from pydantic import BaseModel, Field, ConfigDict, model_validator, PrivateAttr
from pydantic.fields import FieldInfo

from neo4jalchemy.core.graph import Graph
from neo4jalchemy.core.graph_edge import GraphEdge

if TYPE_CHECKING:
    from neo4jalchemy.orm.entities import GraphEntity

# Type variable for GraphRelationship subclasses
RelationshipType = TypeVar('RelationshipType', bound='GraphRelationship')


class GraphRelationshipConfig:
    """Configuration for GraphRelationship classes."""
    
    def __init__(
        self,
        relationship_type: Optional[str] = None,
        from_entity: Optional[Type['GraphEntity']] = None,
        to_entity: Optional[Type['GraphEntity']] = None,
        graph: Optional[Graph] = None,
        directed: bool = True,
        auto_sync: bool = True,
        track_changes: bool = True
    ):
        self.relationship_type = relationship_type
        self.from_entity = from_entity
        self.to_entity = to_entity
        self.graph = graph
        self.directed = directed
        self.auto_sync = auto_sync
        self.track_changes = track_changes


class GraphRelationshipMeta(type(BaseModel)):
    """
    Metaclass for GraphRelationship with automatic edge synchronization.
    
    Handles relationship registration, configuration, and graph integration
    using Pydantic V2's advanced metaclass features.
    """
    
    def __new__(
        mcs, 
        name: str, 
        bases: tuple[type, ...], 
        namespace: dict[str, Any],
        **kwargs: Any
    ) -> GraphRelationshipMeta:
        
        # Extract relationship configuration
        relationship_config = namespace.pop('_relationship_config', None)
        if relationship_config is None:
            # Use default configuration
            relationship_config = GraphRelationshipConfig(relationship_type=name)
        
        # Create the Pydantic model
        cls = super().__new__(mcs, name, bases, namespace, **kwargs)
        
        # Skip processing for the base GraphRelationship class
        if name == 'GraphRelationship':
            return cls
        
        # Attach configuration
        cls._relationship_config = relationship_config
        
        # Initialize class-level registries
        cls._instances: weakref.WeakValueDictionary[str, GraphRelationship] = weakref.WeakValueDictionary()
        
        # Register the relationship class
        if not hasattr(GraphRelationship, '_relationship_registry'):
            GraphRelationship._relationship_registry = weakref.WeakSet()
        GraphRelationship._relationship_registry.add(cls)
        
        return cls


class GraphRelationship(BaseModel, metaclass=GraphRelationshipMeta):
    """
    Modern Pydantic V2 GraphRelationship base class.
    
    Features:
    - Full Pydantic V2 compatibility and validation
    - Automatic graph edge synchronization
    - Smart change tracking with dirty field detection
    - Async CRUD operations
    - Lifecycle hooks for business logic
    - Type-safe entity references
    - Bidirectional navigation support
    """
    
    # Core identity and references
    id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique relationship identifier",
        frozen=True
    )
    
    from_id: str = Field(
        ...,
        description="Source entity ID",
        frozen=True  # Cannot change after creation
    )
    
    to_id: str = Field(
        ...,
        description="Target entity ID", 
        frozen=True  # Cannot change after creation
    )
    
    # Metadata
    created_at: datetime = Field(
        default_factory=datetime.now,
        description="When the relationship was created",
        frozen=True
    )
    
    updated_at: datetime = Field(
        default_factory=datetime.now,
        description="When the relationship was last updated"
    )
    
    # Optional weight for algorithms
    weight: float = Field(
        default=1.0,
        ge=0.0,
        description="Relationship weight for algorithms"
    )
    
    # Pydantic V2 configuration
    model_config = ConfigDict(
        validate_assignment=True,
        validate_default=True,
        use_enum_values=True,
        extra='forbid',
        arbitrary_types_allowed=True,
        ser_json_nan='null',
        json_schema_mode='validation',
    )
    
    # Class-level attributes
    _relationship_registry: ClassVar[weakref.WeakSet] = weakref.WeakSet()
    _instances: ClassVar[weakref.WeakValueDictionary[str, GraphRelationship]]
    _relationship_config: ClassVar[GraphRelationshipConfig]
    
    # Private attributes for internal state
    _graph_edge: Optional[GraphEdge] = PrivateAttr(default=None)
    _is_persisted: bool = PrivateAttr(default=False)
    _original_data: Optional[Dict[str, Any]] = PrivateAttr(default=None)
    _from_entity: Optional['GraphEntity'] = PrivateAttr(default=None)
    _to_entity: Optional['GraphEntity'] = PrivateAttr(default=None)
    
    def __init__(self, **data: Any) -> None:
        """Initialize GraphRelationship with automatic graph integration."""
        super().__init__(**data)
        
        # Register this instance
        self._instances[self.id] = self
        
        # Store original data for change tracking
        self._original_data = self.model_dump()
        
        # Create graph edge if auto_sync is enabled
        if hasattr(self, '_relationship_config') and self._relationship_config.auto_sync:
            self._create_graph_edge()
    
    @model_validator(mode='after')
    def _update_timestamp(self) -> GraphRelationship:
        """Update timestamp when model changes."""
        if (hasattr(self, '_relationship_config') and 
            self._relationship_config.track_changes and 
            self._original_data):
            current_data = self.model_dump(exclude={'updated_at'})
            original_data = self._original_data.copy()
            original_data.pop('updated_at', None)
            
            if current_data != original_data:
                object.__setattr__(self, 'updated_at', datetime.now())
        
        return self
    
    def _create_graph_edge(self) -> None:
        """Create and sync with graph edge."""
        graph = self._get_graph()
        if graph is None:
            return
        
        properties = self._get_edge_properties()
        
        try:
            self._graph_edge = graph.add_edge(
                from_id=self.from_id,
                to_id=self.to_id,
                relationship_type=self._get_relationship_type(),
                properties=properties,
                weight=self.weight,
                directed=getattr(self._relationship_config, 'directed', True)
            )
        except Exception:
            # Silently handle graph creation failures (nodes might not exist yet)
            pass
    
    def _get_graph(self) -> Optional[Graph]:
        """Get the graph instance for this relationship."""
        if not hasattr(self, '_relationship_config'):
            return None
            
        # Try relationship config first
        if self._relationship_config.graph:
            return self._relationship_config.graph
        
        # Try to get graph from entity configuration
        if self._relationship_config.from_entity:
            from_entity_config = getattr(self._relationship_config.from_entity, '_entity_config', None)
            if from_entity_config and from_entity_config.graph:
                return from_entity_config.graph
        
        return None
    
    def _get_relationship_type(self) -> str:
        """Get the relationship type for this relationship."""
        if hasattr(self, '_relationship_config'):
            return self._relationship_config.relationship_type
        return self.__class__.__name__.upper()
    
    def _get_edge_properties(self) -> Dict[str, Any]:
        """Get properties to store in graph edge."""
        return self.model_dump(
            exclude={'id', 'from_id', 'to_id', 'weight'},
            mode='json'
        )
    
    def _sync_to_edge(self) -> None:
        """Synchronize relationship state to graph edge."""
        graph = self._get_graph()
        if graph is None:
            return
            
        if self._graph_edge is None:
            self._create_graph_edge()
            return
        
        properties = self._get_edge_properties()
        self._graph_edge.update_properties(properties)
        
        # Update weight if changed
        if self._graph_edge.weight != self.weight:
            self._graph_edge.weight = self.weight
    
    def _sync_from_edge(self) -> None:
        """Synchronize relationship state from graph edge."""
        if self._graph_edge is None:
            return
        
        # Update relationship from edge properties
        edge_data = {
            'id': self.id,
            'from_id': self.from_id,
            'to_id': self.to_id,
            'weight': self._graph_edge.weight
        }
        edge_data.update(self._graph_edge.properties)
        
        # Use Pydantic's model validation for type safety
        validated_data = self.__class__.model_validate(edge_data)
        
        # Update self with validated data
        for field_name, field_value in validated_data.model_dump().items():
            if field_name not in {'id', 'from_id', 'to_id', 'created_at'}:
                setattr(self, field_name, field_value)
    
    # =============================================================================
    # ENTITY REFERENCES
    # =============================================================================
    
    async def get_from_entity(self) -> Optional['GraphEntity']:
        """Get the source entity of this relationship."""
        if self._from_entity is not None:
            return self._from_entity
        
        if hasattr(self, '_relationship_config') and self._relationship_config.from_entity:
            entity = await self._relationship_config.from_entity.get(self.from_id)
            self._from_entity = entity
            return entity
        
        return None
    
    async def get_to_entity(self) -> Optional['GraphEntity']:
        """Get the target entity of this relationship."""
        if self._to_entity is not None:
            return self._to_entity
        
        if hasattr(self, '_relationship_config') and self._relationship_config.to_entity:
            entity = await self._relationship_config.to_entity.get(self.to_id)
            self._to_entity = entity
            return entity
        
        return None
    
    @property
    def from_entity(self) -> Optional['GraphEntity']:
        """Synchronous access to cached from entity."""
        return self._from_entity
    
    @property
    def to_entity(self) -> Optional['GraphEntity']:
        """Synchronous access to cached to entity."""
        return self._to_entity
    
    # =============================================================================
    # CRUD OPERATIONS
    # =============================================================================
    
    async def save(self, *, force_update: bool = False) -> GraphRelationship:
        """
        Save relationship to the graph/database.
        
        Args:
            force_update: Force update even if no changes detected
            
        Returns:
            Self for method chaining
        """
        if not self.is_dirty() and not force_update and self._is_persisted:
            return self
        
        # Pre-save hook
        await self._pre_save()
        
        # Sync to graph edge
        self._sync_to_edge()
        
        # Mark as persisted and update tracking data
        self._is_persisted = True
        self._original_data = self.model_dump()
        
        # Post-save hook
        await self._post_save()
        
        return self
    
    async def delete(self) -> bool:
        """
        Delete relationship from graph/database.
        
        Returns:
            True if deleted, False if not found
        """
        # Pre-delete hook
        await self._pre_delete()
        
        graph = self._get_graph()
        if graph and self._graph_edge:
            success = graph.remove_edge(self._graph_edge.id)
            if success:
                self._graph_edge = None
                self._is_persisted = False
                
                # Remove from instance registry
                if self.id in self._instances:
                    del self._instances[self.id]
                
                # Post-delete hook
                await self._post_delete()
                
                return True
        
        return False
    
    async def refresh(self) -> GraphRelationship:
        """
        Refresh relationship from database/graph.
        
        Returns:
            Self for method chaining
        """
        # Sync from graph edge
        self._sync_from_edge()
        
        # Update tracking data
        self._original_data = self.model_dump()
        
        return self
    
    # =============================================================================
    # LIFECYCLE HOOKS
    # =============================================================================
    
    async def _pre_save(self) -> None:
        """Hook called before saving relationship. Override in subclasses."""
        pass
    
    async def _post_save(self) -> None:
        """Hook called after saving relationship. Override in subclasses."""
        pass
    
    async def _pre_delete(self) -> None:
        """Hook called before deleting relationship. Override in subclasses."""
        pass
    
    async def _post_delete(self) -> None:
        """Hook called after deleting relationship. Override in subclasses."""
        pass
    
    # =============================================================================
    # CLASS METHODS
    # =============================================================================
    
    @classmethod
    async def get(cls: Type[RelationshipType], relationship_id: str) -> Optional[RelationshipType]:
        """
        Get relationship by ID.
        
        Args:
            relationship_id: Relationship identifier
            
        Returns:
            Relationship instance or None if not found
        """
        # Check instance cache first
        if relationship_id in cls._instances:
            return cls._instances[relationship_id]
        
        # Query from graph (implementation depends on graph structure)
        graph = cls._get_class_graph()
        if graph:
            # Find edge by relationship type and ID
            edge = graph.get_edge(relationship_id)
            if edge and edge.relationship_type == cls._get_class_relationship_type():
                return cls._from_edge(edge)
        
        return None
    
    @classmethod
    async def create(
        cls: Type[RelationshipType], 
        from_node: 'GraphEntity',
        to_node: 'GraphEntity',
        **kwargs: Any
    ) -> RelationshipType:
        """
        Create and save new relationship.
        
        Args:
            from_node: Source entity
            to_node: Target entity
            **kwargs: Relationship field values
            
        Returns:
            Created relationship instance
        """
        # Validate entity types
        if (hasattr(cls, '_relationship_config') and 
            cls._relationship_config.from_entity and 
            not isinstance(from_node, cls._relationship_config.from_entity)):
            raise TypeError(f"from_node must be instance of {cls._relationship_config.from_entity.__name__}")
        
        if (hasattr(cls, '_relationship_config') and 
            cls._relationship_config.to_entity and 
            not isinstance(to_node, cls._relationship_config.to_entity)):
            raise TypeError(f"to_node must be instance of {cls._relationship_config.to_entity.__name__}")
        
        # Create relationship
        relationship = cls(
            from_id=from_node.id,
            to_id=to_node.id,
            **kwargs
        )
        
        # Cache entity references
        relationship._from_entity = from_node
        relationship._to_entity = to_node
        
        await relationship.save()
        return relationship
    
    @classmethod
    async def get_or_create(
        cls: Type[RelationshipType],
        from_node: 'GraphEntity',
        to_node: 'GraphEntity',
        defaults: Optional[Dict[str, Any]] = None,
        **kwargs: Any
    ) -> tuple[RelationshipType, bool]:
        """
        Get existing relationship or create new one.
        
        Args:
            from_node: Source entity
            to_node: Target entity
            defaults: Default values for creation
            **kwargs: Lookup/creation criteria
            
        Returns:
            (relationship, created) tuple
        """
        # Try to find existing relationship by checking all instances first
        relationship_type = cls._get_class_relationship_type()
        
        # Check instance cache for existing relationships between these nodes
        for instance in cls._instances.values():
            if (instance.from_id == from_node.id and 
                instance.to_id == to_node.id):
                # Found existing relationship
                return instance, False
        
        # Try to find in graph if not in cache
        graph = cls._get_class_graph()
        if graph:
            edges = graph.get_edges_between(from_node.id, to_node.id)
            for edge in edges:
                if edge.relationship_type == relationship_type:
                    # Found existing, create instance from edge
                    existing = cls._from_edge(edge)
                    return existing, False
        
        # Create new relationship
        create_data = defaults or {}
        create_data.update(kwargs)  # Add any additional criteria as properties
        relationship = await cls.create(from_node, to_node, **create_data)
        return relationship, True
    
    @classmethod
    def _get_class_graph(cls) -> Optional[Graph]:
        """Get graph for this relationship class."""
        if hasattr(cls, '_relationship_config'):
            return cls._relationship_config.graph
        return None
    
    @classmethod
    def _get_class_relationship_type(cls) -> str:
        """Get relationship type for this relationship class."""
        if hasattr(cls, '_relationship_config'):
            return cls._relationship_config.relationship_type
        return cls.__name__.upper()
    
    @classmethod
    def _from_edge(cls: Type[RelationshipType], edge: GraphEdge) -> RelationshipType:
        """Create relationship instance from graph edge."""
        # Prepare data for relationship creation
        data = {
            'from_id': edge.from_id,
            'to_id': edge.to_id,
            'weight': edge.weight
        }
        
        # Get all fields from the model
        for field_name, field_info in cls.model_fields.items():
            if field_name not in ['id', 'from_id', 'to_id', 'weight', 'created_at', 'updated_at']:
                # Get value from edge properties or use field default
                if field_name in edge.properties:
                    data[field_name] = edge.properties[field_name]
                elif field_info.default is not None:
                    data[field_name] = field_info.default
                elif field_info.default_factory is not None:
                    data[field_name] = field_info.default_factory()
        
        # Add any additional properties from edge that aren't in model fields
        for key, value in edge.properties.items():
            if key not in data:
                data[key] = value
        
        # Use Pydantic's validation
        relationship = cls.model_validate(data)
        relationship._graph_edge = edge
        relationship._is_persisted = True
        relationship._original_data = relationship.model_dump()
        
        return relationship
    
    # =============================================================================
    # UTILITY METHODS
    # =============================================================================
    
    def is_dirty(self) -> bool:
        """Check if relationship has unsaved changes."""
        if (not hasattr(self, '_relationship_config') or 
            not self._relationship_config.track_changes or 
            not self._original_data):
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
        """Convert relationship to graph-compatible dictionary."""
        return {
            'id': self.id,
            'from_id': self.from_id,
            'to_id': self.to_id,
            'relationship_type': self._get_relationship_type(),
            'properties': self._get_edge_properties(),
            'weight': self.weight,
            'directed': getattr(self._relationship_config, 'directed', True) if hasattr(self, '_relationship_config') else True
        }
    
    def __repr__(self) -> str:
        """String representation of relationship."""
        status_parts = []
        if self._is_persisted:
            status_parts.append("persisted")
        else:
            status_parts.append("new")
        
        if self.is_dirty():
            status_parts.append("dirty")
        
        status = f" ({', '.join(status_parts)})" if status_parts else ""
        
        return f"{self.__class__.__name__}(from={self.from_id}, to={self.to_id}{status})"


# =============================================================================
# DECORATOR FUNCTION
# =============================================================================

def graph_relationship(
    cls: Optional[Type] = None,
    *,
    relationship_type: Optional[str] = None,
    from_entity: Optional[Type['GraphEntity']] = None,
    to_entity: Optional[Type['GraphEntity']] = None,
    graph: Optional[Graph] = None,
    directed: bool = True,
    auto_sync: bool = True,
    track_changes: bool = True
) -> Union[Type[GraphRelationship], callable]:
    """
    Modern Pydantic V2 decorator for graph relationships.
    
    Args:
        cls: The class being decorated
        relationship_type: Override the default relationship type (class name)
        from_entity: Source entity type constraint
        to_entity: Target entity type constraint
        graph: Graph instance to use for this relationship
        directed: Whether the relationship is directed (default: True)
        auto_sync: Whether to automatically sync with graph edges
        track_changes: Whether to track field changes
    
    Returns:
        Decorated class or decorator function
    """
    def decorator(target_cls: Type) -> Type:
        if not issubclass(target_cls, GraphRelationship):
            raise TypeError("@graph_relationship can only be applied to GraphRelationship subclasses")
        
        # Create relationship configuration
        relationship_config = GraphRelationshipConfig(
            relationship_type=relationship_type or target_cls.__name__.upper(),
            from_entity=from_entity,
            to_entity=to_entity,
            graph=graph,
            directed=directed,
            auto_sync=auto_sync,
            track_changes=track_changes
        )
        
        # Set configuration on class
        target_cls._relationship_config = relationship_config
        
        return target_cls
    
    if cls is None:
        return decorator
    else:
        return decorator(cls)


# =============================================================================
# REGISTRY FUNCTIONS
# =============================================================================

def get_relationship_classes() -> Set[Type[GraphRelationship]]:
    """Get all registered GraphRelationship classes."""
    if hasattr(GraphRelationship, '_relationship_registry'):
        return set(GraphRelationship._relationship_registry)
    return set()


def get_relationship_by_type(relationship_type: str) -> Optional[Type[GraphRelationship]]:
    """Get relationship class by its type."""
    for relationship_class in get_relationship_classes():
        if relationship_class._get_class_relationship_type() == relationship_type:
            return relationship_class
    return None