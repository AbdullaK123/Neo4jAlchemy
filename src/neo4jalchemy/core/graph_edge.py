from pydantic import BaseModel, Field, field_validator, ConfigDict, model_validator
from typing import Dict, Any
from datetime import datetime


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