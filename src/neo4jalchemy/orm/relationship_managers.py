# src/neo4jalchemy/orm/relationship_managers.py
"""
Neo4jAlchemy Relationship Managers - SQLAlchemy-Style Navigation

This module provides relationship managers for type-safe navigation between
entities, following SQLAlchemy's relationship patterns.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Type, TypeVar, Generic, Union, AsyncIterator, TYPE_CHECKING
from collections import defaultdict
import weakref

if TYPE_CHECKING:
    from neo4jalchemy.orm.entities import GraphEntity
    from neo4jalchemy.orm.relationships import GraphRelationship


EntityType = TypeVar('EntityType', bound='GraphEntity')
RelationshipType = TypeVar('RelationshipType', bound='GraphRelationship')


class RelationshipQuery(Generic[RelationshipType]):
    """
    Query builder for relationships, following SQLAlchemy patterns.
    
    Provides filtering, ordering, and lazy loading for relationships.
    """
    
    def __init__(
        self,
        relationship_class: Type[RelationshipType],
        source_entity: 'GraphEntity',
        direction: str = "outgoing"
    ):
        self.relationship_class = relationship_class
        self.source_entity = source_entity
        self.direction = direction
        self._filters: List[Dict[str, Any]] = []
        self._limit: Optional[int] = None
        self._offset: Optional[int] = None
        self._order_by: List[str] = []
    
    def filter(self, **kwargs) -> RelationshipQuery[RelationshipType]:
        """
        Add filters to the relationship query.
        
        Args:
            **kwargs: Field filters
        
        Returns:
            Self for method chaining
        """
        self._filters.append(kwargs)
        return self
    
    def limit(self, count: int) -> RelationshipQuery[RelationshipType]:
        """Limit the number of results."""
        self._limit = count
        return self
    
    def offset(self, count: int) -> RelationshipQuery[RelationshipType]:
        """Skip a number of results."""
        self._offset = count
        return self
    
    def order_by(self, *fields: str) -> RelationshipQuery[RelationshipType]:
        """
        Order results by fields.
        
        Args:
            *fields: Field names, prefix with '-' for descending
        """
        self._order_by.extend(fields)
        return self
    
    async def all(self) -> List[RelationshipType]:
        """
        Execute query and return all matching relationships.
        
        Returns:
            List of relationship instances
        """
        # Get all relationships from graph
        relationships = await self._get_base_relationships()
        
        # Apply filters
        filtered = self._apply_filters(relationships)
        
        # Apply ordering
        ordered = self._apply_ordering(filtered)
        
        # Apply pagination
        paginated = self._apply_pagination(ordered)
        
        return paginated
    
    async def first(self) -> Optional[RelationshipType]:
        """Get the first matching relationship."""
        results = await self.limit(1).all()
        return results[0] if results else None
    
    async def count(self) -> int:
        """Count matching relationships."""
        relationships = await self._get_base_relationships()
        filtered = self._apply_filters(relationships)
        return len(filtered)
    
    async def exists(self) -> bool:
        """Check if any matching relationships exist."""
        return await self.count() > 0
    
    async def _get_base_relationships(self) -> List[RelationshipType]:
        """Get base relationships from graph."""
        graph = self.source_entity._get_graph()
        if not graph:
            return []
        
        relationships = []
        relationship_type = self.relationship_class._get_class_relationship_type()
        
        if self.direction == "outgoing":
            # Find outgoing edges of this type
            for edge in graph._edges.values():
                if (edge.from_id == self.source_entity.id and 
                    edge.relationship_type == relationship_type):
                    relationship = self.relationship_class._from_edge(edge)
                    relationships.append(relationship)
        
        elif self.direction == "incoming":
            # Find incoming edges of this type
            for edge in graph._edges.values():
                if (edge.to_id == self.source_entity.id and 
                    edge.relationship_type == relationship_type):
                    relationship = self.relationship_class._from_edge(edge)
                    relationships.append(relationship)
        
        return relationships
    
    
    def _apply_filters(self, relationships: List[RelationshipType]) -> List[RelationshipType]:
        """Apply filters to relationship list."""
        if not self._filters:
            return relationships
        
        filtered = []
        for relationship in relationships:
            match = True
            
            for filter_dict in self._filters:
                for field, value in filter_dict.items():
                    # Handle field lookups (like Django ORM)
                    if '__' in field:
                        field_name, lookup = field.split('__', 1)
                        relationship_value = getattr(relationship, field_name, None)
                        
                        if lookup == 'gte' and (relationship_value is None or relationship_value < value):
                            match = False
                            break
                        elif lookup == 'lte' and (relationship_value is None or relationship_value > value):
                            match = False
                            break
                        elif lookup == 'gt' and (relationship_value is None or relationship_value <= value):
                            match = False
                            break
                        elif lookup == 'lt' and (relationship_value is None or relationship_value >= value):
                            match = False
                            break
                        elif lookup == 'contains' and (relationship_value is None or value not in relationship_value):
                            match = False
                            break
                        elif lookup == 'in' and (relationship_value is None or relationship_value not in value):
                            match = False
                            break
                    else:
                        # Exact match
                        relationship_value = getattr(relationship, field, None)
                        if relationship_value != value:
                            match = False
                            break
                
                if not match:
                    break
            
            if match:
                filtered.append(relationship)
        
        return filtered

    def _apply_ordering(self, relationships: List[RelationshipType]) -> List[RelationshipType]:
        """Apply ordering to relationship list."""
        if not self._order_by:
            return relationships
        
        # Create a copy to avoid modifying the original
        sorted_relationships = list(relationships)
        
        # Apply sorts in reverse order so the first field specified has highest priority
        for field in reversed(self._order_by):
            reverse = field.startswith('-')
            field_name = field[1:] if reverse else field
            
            def sort_key(rel):
                value = getattr(rel, field_name, None)
                # Handle None values by putting them at the end
                if value is None:
                    return float('inf') if not reverse else float('-inf')
                return value
            
            sorted_relationships.sort(key=sort_key, reverse=reverse)
        
        return sorted_relationships
    
    def _apply_pagination(self, relationships: List[RelationshipType]) -> List[RelationshipType]:
        """Apply pagination to relationship list."""
        start = self._offset or 0
        end = start + self._limit if self._limit else None
        return relationships[start:end]
    
    @property
    def target(self) -> TargetEntityQuery:
        """
        Navigate to target entities through this relationship.
        
        Returns:
            Query for target entities
        """
        return TargetEntityQuery(self, self.relationship_class._relationship_config.to_entity)
    
    @property
    def source(self) -> SourceEntityQuery:
        """
        Navigate to source entities through this relationship (for incoming relationships).
        
        Returns:
            Query for source entities
        """
        return SourceEntityQuery(self, self.relationship_class._relationship_config.from_entity)


class TargetEntityQuery(Generic[EntityType]):
    """Query builder for target entities through relationships."""
    
    def __init__(self, relationship_query: RelationshipQuery, entity_class: Type[EntityType]):
        self.relationship_query = relationship_query
        self.entity_class = entity_class
        self._entity_filters: List[Dict[str, Any]] = []
    
    def filter(self, **kwargs) -> TargetEntityQuery[EntityType]:
        """Filter target entities."""
        self._entity_filters.append(kwargs)
        return self
    
    async def all(self) -> List[EntityType]:
        """Get all target entities."""
        # Get relationships first
        relationships = await self.relationship_query.all()
        
        # Get target entities
        entities = []
        for relationship in relationships:
            target_entity = await relationship.get_to_entity()
            if target_entity:
                entities.append(target_entity)
        
        # Apply entity filters
        if self._entity_filters:
            filtered_entities = []
            for entity in entities:
                match = True
                for filter_dict in self._entity_filters:
                    for field, value in filter_dict.items():
                        # Handle lookup operations
                        if '__' in field:
                            field_name, lookup = field.split('__', 1)
                            entity_value = getattr(entity, field_name, None)
                            
                            if lookup == 'in' and entity_value not in value:
                                match = False
                                break
                            elif lookup == 'contains' and value not in entity_value:
                                match = False
                                break
                        else:
                            # Exact match
                            if getattr(entity, field, None) != value:
                                match = False
                                break
                    if not match:
                        break
                if match:
                    filtered_entities.append(entity)
            entities = filtered_entities
        
        return entities
    
    async def first(self) -> Optional[EntityType]:
        """Get first target entity."""
        entities = await self.all()
        return entities[0] if entities else None
    
    async def count(self) -> int:
        """Count target entities."""
        entities = await self.all()
        return len(entities)


class SourceEntityQuery(Generic[EntityType]):
    """Query builder for source entities through relationships."""
    
    def __init__(self, relationship_query: RelationshipQuery, entity_class: Type[EntityType]):
        self.relationship_query = relationship_query
        self.entity_class = entity_class
        self._entity_filters: List[Dict[str, Any]] = []
    
    def filter(self, **kwargs) -> SourceEntityQuery[EntityType]:
        """Filter source entities."""
        self._entity_filters.append(kwargs)
        return self
    
    async def all(self) -> List[EntityType]:
        """Get all source entities."""
        # Get relationships first
        relationships = await self.relationship_query.all()
        
        # Get source entities
        entities = []
        for relationship in relationships:
            source_entity = await relationship.get_from_entity()
            if source_entity:
                entities.append(source_entity)
        
        # Apply entity filters
        if self._entity_filters:
            filtered_entities = []
            for entity in entities:
                match = True
                for filter_dict in self._entity_filters:
                    for field, value in filter_dict.items():
                        if getattr(entity, field, None) != value:
                            match = False
                            break
                    if not match:
                        break
                if match:
                    filtered_entities.append(entity)
            entities = filtered_entities
        
        return entities


class RelationshipManager:
    """
    Manages relationships for a specific entity instance.
    
    Provides access to outgoing and incoming relationships with type safety.
    """
    
    def __init__(self, entity: 'GraphEntity'):
        self.entity = entity
        self._outgoing_cache: Dict[str, RelationshipQuery] = {}
        self._incoming_cache: Dict[str, RelationshipQuery] = {}
    
    def outgoing(self, relationship_class: Type[RelationshipType]) -> RelationshipQuery[RelationshipType]:
        """
        Get query for outgoing relationships of specified type.
        
        Args:
            relationship_class: Type of relationship to query
        
        Returns:
            Relationship query builder
        """
        cache_key = relationship_class.__name__
        if cache_key not in self._outgoing_cache:
            self._outgoing_cache[cache_key] = RelationshipQuery(
                relationship_class, self.entity, "outgoing"
            )
        return self._outgoing_cache[cache_key]
    
    def incoming(self, relationship_class: Type[RelationshipType]) -> RelationshipQuery[RelationshipType]:
        """
        Get query for incoming relationships of specified type.
        
        Args:
            relationship_class: Type of relationship to query
        
        Returns:
            Relationship query builder
        """
        cache_key = relationship_class.__name__
        if cache_key not in self._incoming_cache:
            self._incoming_cache[cache_key] = RelationshipQuery(
                relationship_class, self.entity, "incoming"
            )
        return self._incoming_cache[cache_key]
    
    async def all_outgoing(self) -> List['GraphRelationship']:
        """Get all outgoing relationships regardless of type."""
        graph = self.entity._get_graph()
        if not graph:
            return []
        
        from neo4jalchemy.orm.relationships import get_relationship_classes
        
        all_relationships = []
        for relationship_class in get_relationship_classes():
            relationships = await self.outgoing(relationship_class).all()
            all_relationships.extend(relationships)
        
        return all_relationships
    
    async def all_incoming(self) -> List['GraphRelationship']:
        """Get all incoming relationships regardless of type."""
        graph = self.entity._get_graph()
        if not graph:
            return []
        
        from neo4jalchemy.orm.relationships import get_relationship_classes
        
        all_relationships = []
        for relationship_class in get_relationship_classes():
            relationships = await self.incoming(relationship_class).all()
            all_relationships.extend(relationships)
        
        return all_relationships


class RelationshipDescriptor:
    """
    Descriptor for type-safe relationship access on entities.
    
    Enables SQLAlchemy-style syntax like: user.works_on.filter(role="lead").all()
    """
    
    def __init__(
        self, 
        relationship_class: Type['GraphRelationship'],
        direction: str = "outgoing",
        related_name: Optional[str] = None
    ):
        self.relationship_class = relationship_class
        self.direction = direction
        self.related_name = related_name
        self._name: Optional[str] = None
    
    def __set_name__(self, owner, name):
        """Called when descriptor is assigned to a class attribute."""
        self._name = name
    
    def __get__(self, instance, owner) -> RelationshipQuery:
        """Return relationship query when accessed."""
        if instance is None:
            return self
        
        return RelationshipQuery(
            self.relationship_class,
            instance,
            self.direction
        )


def relationship(
    relationship_class: Type['GraphRelationship'],
    direction: str = "outgoing",
    related_name: Optional[str] = None
) -> RelationshipDescriptor:
    """
    Create a relationship descriptor for entity classes.
    
    Args:
        relationship_class: The relationship class to use
        direction: "outgoing" or "incoming"
        related_name: Name for reverse relationship
    
    Returns:
        Relationship descriptor
    """
    return RelationshipDescriptor(relationship_class, direction, related_name)


# =============================================================================
# BIDIRECTIONAL RELATIONSHIP SETUP
# =============================================================================

def setup_bidirectional_relationships():
    """
    Automatically setup reverse relationships for bidirectional navigation.
    
    This function analyzes all registered relationship classes and creates
    reverse relationship descriptors on the target entity classes.
    """
    from neo4jalchemy.orm.relationships import get_relationship_classes
    
    for relationship_class in get_relationship_classes():
        config = relationship_class._relationship_config
        
        if config.from_entity and config.to_entity:
            # Create forward relationship on from_entity
            forward_name = _generate_relationship_name(relationship_class, "forward")
            if not hasattr(config.from_entity, forward_name):
                setattr(
                    config.from_entity,
                    forward_name,
                    RelationshipDescriptor(relationship_class, "outgoing")
                )
            
            # Create reverse relationship on to_entity
            reverse_name = _generate_relationship_name(relationship_class, "reverse")
            if not hasattr(config.to_entity, reverse_name):
                setattr(
                    config.to_entity,
                    reverse_name,
                    RelationshipDescriptor(relationship_class, "incoming")
                )


def _generate_relationship_name(relationship_class: Type['GraphRelationship'], direction: str) -> str:
    """Generate relationship attribute name from class name."""
    class_name = relationship_class.__name__
    
    if direction == "forward":
        # WorksOn -> works_on
        return _camel_to_snake(class_name)
    else:
        # WorksOn -> worked_on_by
        base_name = _camel_to_snake(class_name)
        if base_name.endswith('s'):
            return f"{base_name[:-1]}ed_by"
        else:
            return f"{base_name}ed_by"


def _camel_to_snake(name: str) -> str:
    """Convert CamelCase to snake_case."""
    import re
    # Insert underscore before uppercase letters that follow lowercase letters
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    # Insert underscore before uppercase letters that follow lowercase letters or digits
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()