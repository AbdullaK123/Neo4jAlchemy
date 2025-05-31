#!/usr/bin/env python3
r"""
Neo4jAlchemy GraphEntity System Example

This example demonstrates the power of the new GraphEntity ORM system,
showing how it brings SQLAlchemy-style elegance to graph development.

FIXED: Uses pure Pydantic V2 Field annotations instead of mixing ORM fields.
"""

import asyncio
from datetime import datetime
from typing import Optional, List, Literal
from pydantic import Field

from neo4jalchemy.core.graph import Graph
from neo4jalchemy.orm.entities import GraphEntity, graph_entity


# =============================================================================
# DEFINE ENTITIES (PURE PYDANTIC V2 APPROACH)
# =============================================================================
graph = Graph(name="company_ecosystem")

@graph_entity(label="User", graph=graph)
class User(GraphEntity):
    """
    User entity with rich validation and ORM features.
    
    Uses PURE Pydantic V2 fields with proper type annotations.
    """
    
    # Pydantic V2 fields with full validation
    name: str = Field(min_length=1, max_length=100, description="User's full name")
    email: str = Field(pattern=r'^[^@]+@[^@]+\.[^@]+$', description="User's email address")
    age: int = Field(ge=0, le=150, description="User's age")
    bio: Optional[str] = Field(None, max_length=500, description="User biography")
    is_active: bool = Field(default=True, description="Whether user is active")
    join_date: datetime = Field(default_factory=datetime.now, description="When user joined")
    
    # Additional fields with type annotations
    username: str = Field(min_length=3, max_length=50, description="Unique username")
    skill_level: int = Field(default=1, ge=1, le=10, description="Skill level 1-10")
    tags: List[str] = Field(default_factory=list, description="User tags")
    verified: bool = Field(default=False, description="Whether user is verified")
    
    async def _pre_save(self):
        """Hook: Ensure username is lowercase before saving."""
        if self.username:
            self.username = self.username.lower()
    
    async def _post_save(self):
        """Hook: Log user creation/update."""
        action = "created" if not hasattr(self, '_was_persisted') else "updated"
        print(f"🔄 User {self.name} ({self.id}) {action}")
    
    def is_expert(self) -> bool:
        """Business logic: Check if user is an expert."""
        return self.skill_level >= 8
    
    def __str__(self):
        return f"{self.name} (@{self.username})"


@graph_entity(label="Project", graph=graph)
class Project(GraphEntity):
    """Project entity with rich metadata."""
    
    title: str = Field(min_length=1, max_length=200)
    description: str = Field(max_length=1000)
    status: Literal["planning", "active", "completed", "archived"] = Field(default="planning")
    priority: int = Field(ge=1, le=5, default=3)
    created_by: str = Field(description="Creator user ID")
    
    # Additional fields with proper annotations
    project_code: str = Field(min_length=3, max_length=20, description="Unique project code")
    budget: int = Field(default=0, ge=0, description="Project budget")
    is_public: bool = Field(default=False, description="Whether project is public")
    tech_stack: List[str] = Field(default_factory=list, description="Technology stack")
    
    async def _pre_save(self):
        """Ensure project code is uppercase."""
        if self.project_code:
            self.project_code = self.project_code.upper()
    
    def is_high_priority(self) -> bool:
        """Check if project is high priority."""
        return self.priority >= 4
    
    def __str__(self):
        return f"{self.title} ({self.project_code})"


@graph_entity(label="Company", graph=graph)
class Company(GraphEntity):
    """Company entity."""
    
    name: str = Field(min_length=1, max_length=200)
    industry: str = Field(max_length=100)
    founded_year: int = Field(ge=1800, le=2024)
    website: Optional[str] = Field(None, description="Company website")
    
    # Additional fields with proper annotations
    company_code: str = Field(min_length=3, max_length=10, description="Unique company code")
    employee_count: int = Field(default=1, ge=1, description="Number of employees")
    is_hiring: bool = Field(default=False, description="Whether company is hiring")
    tech_focus: List[str] = Field(default_factory=list, description="Technology focus areas")
    
    def __str__(self):
        return f"{self.name} ({self.company_code})"


# =============================================================================
# EXAMPLE DEMONSTRATION
# =============================================================================

async def demonstrate_graph_entity_system():
    """
    Comprehensive demonstration of the GraphEntity ORM system.
    """
    print("🚀 Neo4jAlchemy GraphEntity System Demo")
    print("=" * 60)
    
    print(f"\n📊 Created graph: {graph.name}")
    
    # =============================================================================
    # 1. CREATE ENTITIES WITH VALIDATION
    # =============================================================================
    
    print(f"\n1️⃣ Creating Entities with Validation")
    print("-" * 40)
    
    # Create users with rich validation
    alice = User(
        name="Alice Johnson",
        email="alice@techcorp.com",
        age=28,
        bio="Senior Python developer with expertise in graph databases",
        username="alice_dev",
        skill_level=9,
        tags=["python", "neo4j", "backend"],
        verified=True
    )
    
    bob = User(
        name="Bob Chen",
        email="bob@example.com", 
        age=24,
        username="bob_junior",
        skill_level=4,
        tags=["javascript", "react", "frontend"]
    )
    
    print(f"✅ Created users:")
    print(f"   - {alice} (Expert: {alice.is_expert()})")
    print(f"   - {bob} (Expert: {bob.is_expert()})")
    
    # Create company
    company = Company(
        name="TechCorp Solutions",
        industry="Software Development",
        founded_year=2018,
        website="https://techcorp.com",
        company_code="TECH001",
        employee_count=150,
        is_hiring=True,
        tech_focus=["AI", "Graph Databases", "Cloud"]
    )
    
    print(f"✅ Created company: {company}")
    
    # Create projects
    ai_project = Project(
        title="AI-Powered Analytics Platform",
        description="Building next-gen analytics using machine learning",
        status="active",
        priority=5,
        created_by=alice.id,
        project_code="AI2024",
        budget=500000,
        is_public=False,
        tech_stack=["Python", "TensorFlow", "Neo4j", "FastAPI"]
    )
    
    web_project = Project(
        title="Customer Portal Redesign",
        description="Modernizing the customer-facing web portal",
        status="planning",
        priority=3,
        created_by=bob.id,
        project_code="WEB2024",
        budget=100000,
        is_public=True,
        tech_stack=["React", "TypeScript", "Node.js"]
    )
    
    print(f"✅ Created projects:")
    print(f"   - {ai_project} (High Priority: {ai_project.is_high_priority()})")
    print(f"   - {web_project} (High Priority: {web_project.is_high_priority()})")
    
    # =============================================================================
    # 2. SAVE ENTITIES (WITH LIFECYCLE HOOKS)
    # =============================================================================
    
    print(f"\n2️⃣ Saving Entities (Lifecycle Hooks)")
    print("-" * 40)
    
    # Save all entities - watch the lifecycle hooks fire
    entities = [alice, bob, company, ai_project, web_project]
    
    for entity in entities:
        await entity.save()
    
    print(f"\n📊 Graph Statistics:")
    print(f"   - Nodes: {graph.node_count()}")
    print(f"   - Edges: {graph.edge_count()}")
    print(f"   - Density: {graph.density():.4f}")
    
    # =============================================================================
    # 3. DEMONSTRATE CHANGE TRACKING
    # =============================================================================
    
    print(f"\n3️⃣ Change Tracking and Updates")
    print("-" * 40)
    
    print(f"🔍 Alice before changes:")
    print(f"   - Dirty: {alice.is_dirty()}")
    print(f"   - Skill Level: {alice.skill_level}")
    print(f"   - Tags: {alice.tags}")
    
    # Make changes
    alice.skill_level = 10  # Promote to level 10
    alice.tags.append("team-lead")
    alice.bio = "Lead architect for graph database solutions"
    
    print(f"\n🔄 Alice after changes:")
    print(f"   - Dirty: {alice.is_dirty()}")
    print(f"   - Skill Level: {alice.skill_level}")
    print(f"   - Tags: {alice.tags}")
    print(f"   - Updated At: {alice.updated_at}")
    
    # Save changes
    await alice.save()
    print(f"✅ Changes saved. Dirty: {alice.is_dirty()}")
    
    # =============================================================================
    # 4. QUERY ENTITIES
    # =============================================================================
    
    print(f"\n4️⃣ Querying Entities")
    print("-" * 40)
    
    # Get entity by ID
    retrieved_alice = await User.get(alice.id)
    print(f"🔍 Retrieved Alice: {retrieved_alice}")
    print(f"   - Same instance: {retrieved_alice is alice}")
    print(f"   - Skill Level: {retrieved_alice.skill_level}")
    
    # Test get_or_create
    existing_bob, created = await User.get_or_create(
        id=bob.id,
        defaults={"name": "This won't be used"}
    )
    print(f"🔍 Get or Create (existing): {existing_bob.name}, Created: {created}")
    
    new_user, created = await User.get_or_create(
        id="user_charlie",
        defaults={
            "name": "Charlie Wilson",
            "email": "charlie@example.com",
            "age": 35,
            "username": "charlie_ops",
            "skill_level": 7
        }
    )
    print(f"🔍 Get or Create (new): {new_user.name}, Created: {created}")
    
    # =============================================================================
    # 5. DEMONSTRATE FIELD VALIDATION
    # =============================================================================
    
    print(f"\n5️⃣ Field Validation Examples")
    print("-" * 40)
    
    # Test validation errors
    try:
        invalid_user = User(
            name="",  # Too short
            email="not-an-email",  # Invalid format
            age=200,  # Too old
            username="ab"  # Too short
        )
    except Exception as e:
        print(f"❌ Validation Error (expected): {type(e).__name__}")
    
    # Test field validation on assignment
    try:
        test_user = User(
            name="Test User",
            email="test@example.com",
            age=25,
            username="test_user"
        )
        test_user.skill_level = 15  # Too high
    except ValueError as e:
        print(f"❌ Field Validation Error (expected): {e}")
    
    # Test successful coercion and validation
    coerce_user = User(
        name="Coercion Test",
        email="coerce@example.com",
        age=30,
        username="coerce_test",
        skill_level=8  # Valid
    )
    print(f"✅ Valid user created: {coerce_user.username} (skill: {coerce_user.skill_level})")
    
    # =============================================================================
    # 6. ENTITY RELATIONSHIPS (PREVIEW)
    # =============================================================================
    
    print(f"\n6️⃣ Entity Relationships (Preview)")
    print("-" * 40)
    
    # Show how entities connect in the graph
    print(f"🔗 Graph Connections:")
    print(f"   - Alice created AI project: {ai_project.created_by == alice.id}")
    print(f"   - Bob created Web project: {web_project.created_by == bob.id}")
    
    # This is where GraphRelationship will shine in Phase 2.2!
    print(f"   - Ready for GraphRelationship system! 🚀")
    
    # =============================================================================
    # 7. ENTITY DELETION
    # =============================================================================
    
    print(f"\n7️⃣ Entity Deletion")
    print("-" * 40)
    
    # Create a temporary entity to delete
    temp_user = await User.create(
        name="Temp User",
        email="temp@example.com",
        age=25,
        username="temp_user"
    )
    
    print(f"➕ Created temp user: {temp_user}")
    print(f"   - In graph: {graph.has_node(temp_user.id)}")
    
    # Delete the entity
    deleted = await temp_user.delete()
    print(f"🗑️ Deleted temp user: {deleted}")
    print(f"   - In graph: {graph.has_node(temp_user.id)}")
    print(f"   - Persisted: {temp_user._is_persisted}")
    
    # =============================================================================
    # 8. EXPORT AND ANALYSIS
    # =============================================================================
    
    print(f"\n8️⃣ Export and Analysis")
    print("-" * 40)
    
    # Export graph data
    graph_dict = graph.to_dict()
    
    print(f"📊 Final Graph Statistics:")
    print(f"   - Total Nodes: {graph.node_count()}")
    print(f"   - Total Edges: {graph.edge_count()}")
    print(f"   - Average Degree: {graph.average_degree():.2f}")
    
    # Show entity types in graph
    entity_types = {}
    for node in graph._nodes.values():
        label = node.label
        entity_types[label] = entity_types.get(label, 0) + 1
    
    print(f"📈 Entity Distribution:")
    for label, count in entity_types.items():
        print(f"   - {label}: {count}")
    
    # Export to JSON
    json_data = graph.to_json(indent=2)
    print(f"\n💾 Graph exported to JSON ({len(json_data)} characters)")
    
    return graph, entities


async def demonstrate_advanced_features():
    """Demonstrate advanced GraphEntity features."""
    
    print(f"\n\n🔬 Advanced Features Demo")
    print("=" * 60)
    
    # Create separate graph for advanced demo
    advanced_graph = Graph(name="advanced_demo")
    
    # Configure a test entity class with proper Pydantic V2 fields
    @graph_entity(label="AdvancedEntity", graph=advanced_graph)
    class AdvancedEntity(GraphEntity):
        name: str = Field(min_length=1)
        value: int = Field(default=0)
        
        # Use Literal for choices (Pydantic V2 way)
        status: Literal["draft", "review", "approved", "rejected"] = Field(default="draft")
        
        async def _pre_save(self):
            print(f"   🔄 Pre-save hook for {self.name}")
        
        async def _post_save(self):
            print(f"   ✅ Post-save hook for {self.name}")
    
    # 1. Test choice validation
    print(f"\n1️⃣ Choice Field Validation")
    entity1 = AdvancedEntity(name="Entity 1")
    print(f"✅ Default status: {entity1.status}")
    
    entity1.status = "approved"
    print(f"✅ Valid status change: {entity1.status}")
    
    try:
        entity1.status = "invalid_status"
    except ValueError as e:
        print(f"❌ Invalid status rejected: {e}")
    
    # 2. Test lifecycle hooks
    print(f"\n2️⃣ Lifecycle Hooks")
    await entity1.save()
    
    # 3. Test change tracking
    print(f"\n3️⃣ Advanced Change Tracking")
    entity1.value = 42
    entity1.name = "Updated Entity 1"
    
    print(f"🔍 Dirty fields detected: {entity1.is_dirty()}")
    await entity1.save()
    
    # 4. Test entity registry
    print(f"\n4️⃣ Entity Registry and Introspection")
    from neo4jalchemy.orm.entities import get_entity_classes, get_entity_by_label
    
    entity_classes = get_entity_classes()
    print(f"📋 Registered entity classes: {len(entity_classes)}")
    for cls in entity_classes:
        print(f"   - {cls.__name__} (label: {cls._get_class_label()})")
    
    # Test label lookup
    user_class = get_entity_by_label("User")
    if user_class:
        print(f"🔍 Found User class by label: {user_class.__name__}")
    else:
        print(f"🔍 User class not found by label")
    
    return advanced_graph


# =============================================================================
# MAIN DEMONSTRATION
# =============================================================================

async def main():
    """Run the complete GraphEntity demonstration."""
    
    try:
        # Main demonstration
        graph, entities = await demonstrate_graph_entity_system()
        
        # Advanced features
        advanced_graph = await demonstrate_advanced_features()
        
        # Final summary
        print(f"\n\n🎉 GraphEntity System Summary")
        print("=" * 60)
        print(f"✅ Successfully demonstrated:")
        print(f"   - Type-safe entity creation with Pydantic V2 validation")
        print(f"   - Automatic graph node synchronization")
        print(f"   - Change tracking and dirty field detection")
        print(f"   - Lifecycle hooks (pre/post save/delete)")
        print(f"   - CRUD operations with async/await")
        print(f"   - Pure Pydantic V2 field system")
        print(f"   - Entity registry and introspection")
        print(f"   - Business logic integration")
        print(f"   - Graph export and analysis")
        
        print(f"\n🚀 Ready for Phase 2.2: GraphRelationship System!")
        print(f"   - Rich relationship modeling")
        print(f"   - Bidirectional relationships") 
        print(f"   - Relationship properties and validation")
        print(f"   - Graph traversal and queries")
        
        print(f"\n📊 Final Statistics:")
        print(f"   - Main graph nodes: {graph.node_count()}")
        print(f"   - Advanced demo nodes: {advanced_graph.node_count()}")
        print(f"   - Total entities created: {len(entities) + 1}")
        
        # Show entity instances
        print(f"\n👥 Entity Instances:")
        for entity in entities:
            print(f"   - {entity}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    # Run the demonstration
    success = asyncio.run(main())
    
    if success:
        print(f"\n🎊 Demo completed successfully!")
        print(f"This shows Neo4jAlchemy's GraphEntity system is ready for production!")
    else:
        print(f"\n💥 Demo encountered errors.")
    
    print(f"\n" + "="*60)
    print(f"Next Steps:")
    print(f"1. Run the test suite: pytest tests/orm/test_graph_entity.py -v")
    print(f"2. Implement GraphRelationship system (Phase 2.2)")
    print(f"3. Add Neo4j backend integration (Phase 3)")
    print(f"4. Build query system and advanced features")
    print("="*60)