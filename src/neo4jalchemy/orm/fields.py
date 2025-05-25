"""
Neo4jAlchemy Type System - Minimal implementation for GraphEntity/GraphRelationship support

This provides the core field types and descriptors needed for the ORM layer.
"""
from typing import Any, Optional, Type, Union, List, Dict, Callable, Generic, TypeVar
from datetime import datetime, date
from abc import ABC, abstractmethod
import weakref
from pydantic import BaseModel


T = TypeVar('T')


class Field(ABC, Generic[T]):
    """
    Base field descriptor for Neo4jAlchemy ORM.
    
    Handles type validation, default values, and change tracking.
    """
    
    def __init__(
        self,
        *,
        default: Optional[Union[T, Callable]] = None,
        required: bool = False,
        index: bool = False,
        unique: bool = False,
        db_field: Optional[str] = None,
        description: Optional[str] = None
    ):
        self.default = default
        self.required = required
        self.index = index
        self.unique = unique
        self.db_field = db_field
        self.description = description
        self.name: Optional[str] = None  # Set by metaclass
        
        # Track values per instance using weak references
        self._values: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()
        self._original_values: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()
    
    def __set_name__(self, owner: Type, name: str):
        """Called when field is assigned to a class attribute."""
        self.name = name
        if self.db_field is None:
            self.db_field = name
    
    def __get__(self, instance: Any, owner: Type) -> Union[T, 'Field']:
        """Get field value from instance."""
        if instance is None:
            return self
        
        if instance in self._values:
            return self._values[instance]
        
        # Return default value
        if self.default is not None:
            if callable(self.default):
                value = self.default()
            else:
                value = self.default
            self._values[instance] = value
            self._original_values[instance] = value
            return value
        
        if self.required:
            raise AttributeError(f"Required field '{self.name}' has no value")
        
        return None
    
    def __set__(self, instance: Any, value: T):
        """Set field value on instance."""
        if value is not None:
            value = self.validate(value)
            value = self.to_python(value)
        
        # Track original value for change detection
        if instance not in self._original_values:
            self._original_values[instance] = value
        
        self._values[instance] = value
    
    @abstractmethod
    def validate(self, value: Any) -> Any:
        """Validate and potentially transform the value."""
        pass
    
    @abstractmethod
    def to_python(self, value: Any) -> T:
        """Convert value to Python type."""
        pass
    
    @abstractmethod
    def to_neo4j(self, value: T) -> Any:
        """Convert value to Neo4j-compatible type."""
        pass
    
    def is_dirty(self, instance: Any) -> bool:
        """Check if field value has changed."""
        if instance not in self._values:
            return False
        
        current = self._values.get(instance)
        original = self._original_values.get(instance)
        return current != original
    
    def reset_tracking(self, instance: Any):
        """Reset change tracking for this field."""
        if instance in self._values:
            self._original_values[instance] = self._values[instance]


class StringField(Field[str]):
    """String field type."""
    
    def __init__(
        self, 
        *,
        max_length: Optional[int] = None,
        min_length: Optional[int] = None,
        choices: Optional[List[str]] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.max_length = max_length
        self.min_length = min_length
        self.choices = choices
    
    def validate(self, value: Any) -> str:
        """Validate string value."""
        if not isinstance(value, str):
            value = str(value)
        
        if self.min_length and len(value) < self.min_length:
            raise ValueError(f"{self.name} must be at least {self.min_length} characters")
        
        if self.max_length and len(value) > self.max_length:
            raise ValueError(f"{self.name} must be at most {self.max_length} characters")
        
        if self.choices and value not in self.choices:
            raise ValueError(f"{self.name} must be one of {self.choices}")
        
        return value
    
    def to_python(self, value: Any) -> str:
        return str(value) if value is not None else None
    
    def to_neo4j(self, value: str) -> str:
        return value


class IntegerField(Field[int]):
    """Integer field type."""
    
    def __init__(
        self,
        *,
        min_value: Optional[int] = None,
        max_value: Optional[int] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.min_value = min_value
        self.max_value = max_value
    
    def validate(self, value: Any) -> int:
        """Validate integer value."""
        try:
            value = int(value)
        except (TypeError, ValueError):
            raise ValueError(f"{self.name} must be an integer")
        
        if self.min_value is not None and value < self.min_value:
            raise ValueError(f"{self.name} must be at least {self.min_value}")
        
        if self.max_value is not None and value > self.max_value:
            raise ValueError(f"{self.name} must be at most {self.max_value}")
        
        return value
    
    def to_python(self, value: Any) -> int:
        return int(value) if value is not None else None
    
    def to_neo4j(self, value: int) -> int:
        return value


class FloatField(Field[float]):
    """Float field type."""
    
    def __init__(
        self,
        *,
        min_value: Optional[float] = None,
        max_value: Optional[float] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.min_value = min_value
        self.max_value = max_value
    
    def validate(self, value: Any) -> float:
        """Validate float value."""
        try:
            value = float(value)
        except (TypeError, ValueError):
            raise ValueError(f"{self.name} must be a float")
        
        if self.min_value is not None and value < self.min_value:
            raise ValueError(f"{self.name} must be at least {self.min_value}")
        
        if self.max_value is not None and value > self.max_value:
            raise ValueError(f"{self.name} must be at most {self.max_value}")
        
        return value
    
    def to_python(self, value: Any) -> float:
        return float(value) if value is not None else None
    
    def to_neo4j(self, value: float) -> float:
        return value


class BooleanField(Field[bool]):
    """Boolean field type."""
    
    def validate(self, value: Any) -> bool:
        """Validate boolean value."""
        if isinstance(value, bool):
            return value
        
        if isinstance(value, str):
            if value.lower() in ('true', '1', 'yes', 'on'):
                return True
            elif value.lower() in ('false', '0', 'no', 'off'):
                return False
        
        return bool(value)
    
    def to_python(self, value: Any) -> bool:
        return self.validate(value) if value is not None else None
    
    def to_neo4j(self, value: bool) -> bool:
        return value


class DateTimeField(Field[datetime]):
    """DateTime field type."""
    
    def __init__(
        self,
        *,
        auto_now: bool = False,
        auto_now_add: bool = False,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.auto_now = auto_now
        self.auto_now_add = auto_now_add
        
        # Set default if auto_now_add
        if self.auto_now_add and self.default is None:
            self.default = datetime.now
    
    def validate(self, value: Any) -> datetime:
        """Validate datetime value."""
        if isinstance(value, datetime):
            return value
        
        if isinstance(value, date):
            return datetime.combine(value, datetime.min.time())
        
        if isinstance(value, str):
            try:
                return datetime.fromisoformat(value)
            except ValueError:
                raise ValueError(f"{self.name} must be a valid datetime")
        
        raise ValueError(f"{self.name} must be a datetime")
    
    def to_python(self, value: Any) -> datetime:
        if value is None:
            return None
        return self.validate(value)
    
    def to_neo4j(self, value: datetime) -> str:
        """Convert to ISO format for Neo4j."""
        if value is None:
            return None
        return value.isoformat()
    
    def __set__(self, instance: Any, value: datetime):
        """Override to handle auto_now."""
        if self.auto_now:
            value = datetime.now()
        super().__set__(instance, value)


class ListField(Field[List]):
    """List field type."""
    
    def __init__(
        self,
        item_field: Optional[Field] = None,
        *,
        min_length: Optional[int] = None,
        max_length: Optional[int] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.item_field = item_field or StringField()
        self.min_length = min_length
        self.max_length = max_length
    
    def validate(self, value: Any) -> List:
        """Validate list value."""
        if not isinstance(value, (list, tuple)):
            raise ValueError(f"{self.name} must be a list")
        
        value = list(value)
        
        if self.min_length and len(value) < self.min_length:
            raise ValueError(f"{self.name} must have at least {self.min_length} items")
        
        if self.max_length and len(value) > self.max_length:
            raise ValueError(f"{self.name} must have at most {self.max_length} items")
        
        # Validate each item
        if self.item_field:
            validated_items = []
            for item in value:
                validated_items.append(self.item_field.validate(item))
            value = validated_items
        
        return value
    
    def to_python(self, value: Any) -> List:
        if value is None:
            return None
        
        if isinstance(value, list):
            return value
        
        # Handle Neo4j arrays
        return list(value)
    
    def to_neo4j(self, value: List) -> List:
        """Convert list items for Neo4j."""
        if value is None:
            return None
        
        if self.item_field:
            return [self.item_field.to_neo4j(item) for item in value]
        return value


class DictField(Field[Dict]):
    """Dictionary/Map field type for nested properties."""
    
    def validate(self, value: Any) -> Dict:
        """Validate dict value."""
        if not isinstance(value, dict):
            raise ValueError(f"{self.name} must be a dictionary")
        
        # Ensure all keys are strings (Neo4j requirement)
        for key in value.keys():
            if not isinstance(key, str):
                raise ValueError(f"All keys in {self.name} must be strings")
        
        return value
    
    def to_python(self, value: Any) -> Dict:
        if value is None:
            return None
        
        if isinstance(value, dict):
            return value
        
        # Handle Neo4j maps
        return dict(value)
    
    def to_neo4j(self, value: Dict) -> Dict:
        """Ensure dict is Neo4j compatible."""
        if value is None:
            return None
        
        # Convert any non-primitive values
        result = {}
        for k, v in value.items():
            if isinstance(v, (str, int, float, bool, type(None))):
                result[k] = v
            elif isinstance(v, datetime):
                result[k] = v.isoformat()
            elif isinstance(v, (list, tuple)):
                result[k] = list(v)
            else:
                result[k] = str(v)
        
        return result


class ReferenceField(Field[str]):
    """Reference to another entity by ID."""
    
    def __init__(
        self,
        to: Optional[Union[str, Type]] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.to = to  # Entity class or label string
    
    def validate(self, value: Any) -> str:
        """Validate reference value."""
        if hasattr(value, 'id'):
            # Entity instance passed
            return str(value.id)
        
        # ID string passed
        return str(value)
    
    def to_python(self, value: Any) -> str:
        return str(value) if value is not None else None
    
    def to_neo4j(self, value: str) -> str:
        return value


# Utility functions for the type system

def get_fields(cls: Type) -> Dict[str, Field]:
    """Get all Field descriptors from a class."""
    fields = {}
    for name in dir(cls):
        attr = getattr(cls, name)
        if isinstance(attr, Field):
            fields[name] = attr
    return fields


def validate_instance(instance: Any) -> Dict[str, Any]:
    """Validate all fields on an instance and return Neo4j-ready data."""
    fields = get_fields(type(instance))
    data = {}
    
    for name, field in fields.items():
        value = getattr(instance, name, None)
        
        if value is None and field.required:
            raise ValueError(f"Required field '{name}' is missing")
        
        if value is not None:
            data[field.db_field] = field.to_neo4j(value)
    
    return data


def get_dirty_fields(instance: Any) -> Dict[str, Any]:
    """Get fields that have changed since last save."""
    fields = get_fields(type(instance))
    dirty = {}
    
    for name, field in fields.items():
        if field.is_dirty(instance):
            value = getattr(instance, name)
            dirty[field.db_field] = field.to_neo4j(value)
    
    return dirty


def reset_instance_tracking(instance: Any):
    """Reset change tracking for all fields."""
    fields = get_fields(type(instance))
    for field in fields.values():
        field.reset_tracking(instance)