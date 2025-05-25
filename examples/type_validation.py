"""
Example demonstrating the Neo4jAlchemy type system.

This shows how the field descriptors work before we build GraphEntity.
"""
from datetime import datetime
from neo4jalchemy.orm.fields import (
    StringField, IntegerField, FloatField, BooleanField,
    DateTimeField, ListField, DictField, ReferenceField,
    get_fields, validate_instance, get_dirty_fields, reset_instance_tracking
)


# Example class using the type system (what GraphEntity will build on)
class Person:
    """Example entity using our field system."""
    
    # Define fields
    name = StringField(required=True, min_length=1, max_length=100)
    age = IntegerField(min_value=0, max_value=150)
    email = StringField(unique=True, index=True)
    is_active = BooleanField(default=True)
    joined_at = DateTimeField(auto_now_add=True)
    tags = ListField(StringField())
    metadata = DictField()
    
    def __init__(self, **kwargs):
        # Set field values from kwargs
        for key, value in kwargs.items():
            setattr(self, key, value)


def demonstrate_type_system():
    """Show how the type system works."""
    
    print("=== Neo4jAlchemy Type System Demo ===\n")
    
    # 1. Create instance with validation
    print("1. Creating a Person with validation:")
    person = Person(
        name="Alice Johnson",
        age=30,
        email="alice@example.com",
        tags=["developer", "python", "neo4j"]
    )
    
    print(f"Name: {person.name}")
    print(f"Age: {person.age}")
    print(f"Email: {person.email}")
    print(f"Active: {person.is_active}")  # Uses default
    print(f"Joined: {person.joined_at}")  # Auto-set
    print(f"Tags: {person.tags}")
    
    # 2. Validation errors
    print("\n2. Testing validation:")
    try:
        person.age = 200  # Too old!
    except ValueError as e:
        print(f"✓ Age validation: {e}")
    
    try:
        person.name = ""  # Too short!
    except ValueError as e:
        print(f"✓ Name validation: {e}")
    
    # 3. Type coercion
    print("\n3. Type coercion:")
    person.age = "25"  # String to int
    print(f"Age after coercion: {person.age} (type: {type(person.age)})")
    
    person.is_active = "false"  # String to bool
    print(f"Active after coercion: {person.is_active} (type: {type(person.is_active)})")
    
    # 4. Change tracking
    print("\n4. Change tracking:")
    reset_instance_tracking(person)  # Reset tracking
    
    print("Initial state saved...")
    
    person.name = "Alice Smith"  # Change name
    person.age = 31  # Change age
    
    dirty_fields = get_dirty_fields(person)
    print(f"Changed fields: {list(dirty_fields.keys())}")
    print(f"New values: {dirty_fields}")
    
    # 5. Neo4j serialization
    print("\n5. Neo4j serialization:")
    neo4j_data = validate_instance(person)
    print("Data ready for Neo4j:")
    for key, value in neo4j_data.items():
        print(f"  {key}: {value} (type: {type(value).__name__})")
    
    # 6. Field introspection
    print("\n6. Field introspection:")
    fields = get_fields(Person)
    for name, field in fields.items():
        print(f"  {name}: {field.__class__.__name__}")
        if field.index:
            print(f"    - indexed")
        if field.unique:
            print(f"    - unique")
        if field.required:
            print(f"    - required")
    
    # 7. Complex types
    print("\n7. Complex field types:")
    person.metadata = {
        "preferences": {
            "theme": "dark",
            "notifications": True
        },
        "last_login": datetime.now()
    }
    
    neo4j_metadata = DictField().to_neo4j(person.metadata)
    print(f"Metadata for Neo4j: {neo4j_metadata}")
    print(f"DateTime converted to: {neo4j_metadata['last_login']}")
    
    # 8. List field with validation
    print("\n8. List field validation:")
    try:
        person.tags = ["valid", 123, "tag"]  # Mixed types
    except ValueError as e:
        print(f"✓ List validation: {e}")
    
    # Correct usage
    person.tags = ["python", "neo4j", "graph"]
    print(f"Valid tags: {person.tags}")


if __name__ == "__main__":
    demonstrate_type_system()
    
    print("\n=== Summary ===")
    print("This type system provides:")
    print("✓ Type validation and coercion")
    print("✓ Change tracking for updates")
    print("✓ Neo4j serialization")
    print("✓ Field metadata (index, unique, required)")
    print("✓ Complex types (lists, dicts, datetime)")
    print("\nReady to build GraphEntity and GraphRelationship on top!")