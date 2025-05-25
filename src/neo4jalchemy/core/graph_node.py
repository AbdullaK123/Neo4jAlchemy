from pydantic import BaseModel, Field, field_validator, ConfigDict
from typing import Dict, Any
from datetime import datetime
import uuid


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
