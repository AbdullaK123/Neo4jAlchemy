# tests/orm/test_graph_relationship.py
"""
Comprehensive tests for the GraphRelationship system.

Tests the complete relationship functionality including:
- Pydantic V2 relationship model behavior and validation
- Automatic graph edge synchronization via metaclass
- CRUD operations with async/await patterns
- Entity type validation and references
- Relationship navigation and querying
- Bidirectional relationship access
- Lifecycle hooks and business logic
- Performance and memory management
"""

import pytest
import pytest_asyncio
from datetime import datetime, timedelta
from typing import Optional, List, Literal
from unittest.mock import patch, AsyncMock

from pydantic import Field, ValidationError, field_validator
from neo4jalchemy.core.graph import Graph
from neo4jalchemy.orm.entities import GraphEntity, graph_entity
from neo4jalchemy.orm.relationships import GraphRelationship, graph_relationship
from neo4jalchemy.orm.relationship_managers import (
    RelationshipManager, RelationshipQuery, relationship
)


# =============================================================================
# TEST FIXTURES AND ENTITIES
# =============================================================================

@pytest.fixture
def test_graph():
    """Create a test graph for relationship operations."""
    return Graph(name="relationship_test_graph")


@pytest.fixture
def setup_test_entities(test_graph):
    """Setup test entities with graph configuration."""
    # Configure entities to use test graph
    User._entity_config.graph = test_graph
    Project._entity_config.graph = test_graph
    Company._entity_config.graph = test_graph
    return test_graph


# Test entity classes
@graph_entity(label="User")
class User(GraphEntity):
    """Test user entity."""
    name: str = Field(min_length=1, max_length=100)
    email: str = Field(pattern=r'^[^@]+@[^@]+\.[^@]+$')
    age: int = Field(ge=0, le=150)
    role: str = Field(default="developer")
    is_active: bool = Field(default=True)
    skills: List[str] = Field(default_factory=list)


@graph_entity(label="Project")  
class Project(GraphEntity):
    """Test project entity."""
    title: str = Field(min_length=1, max_length=200)
    description: str = Field(max_length=1000)
    status: Literal["planning", "active", "completed", "archived"] = Field(default="planning")
    priority: int = Field(ge=1, le=5, default=3)
    budget: int = Field(default=0, ge=0)
    tech_stack: List[str] = Field(default_factory=list)


@graph_entity(label="Company")
class Company(GraphEntity):
    """Test company entity."""
    name: str = Field(min_length=1, max_length=200)
    industry: str = Field(max_length=100)
    founded_year: int = Field(ge=1800, le=2024)
    employee_count: int = Field(default=1, ge=1)


# Test relationship classes
@graph_relationship(from_entity=User, to_entity=Project, relationship_type="WORKS_ON")
class WorksOn(GraphRelationship):
    """User works on project relationship."""
    role: Literal["developer", "lead", "architect", "tester"] = Field(...)
    since: datetime = Field(default_factory=datetime.now)
    hours_per_week: int = Field(ge=1, le=60, default=40)
    is_active: bool = Field(default=True)
    responsibilities: List[str] = Field(default_factory=list)
    
    @field_validator('hours_per_week')
    @classmethod
    def validate_lead_hours(cls, v, info):
        """Lead roles require minimum hours."""
        if hasattr(info, 'data') and info.data.get('role') == 'lead' and v < 20:
            raise ValueError('Lead roles require at least 20 hours/week')
        return v
    
    async def _pre_save(self):
        """Business logic: Set responsibilities based on role."""
        if self.role == "lead":
            if "team_management" not in self.responsibilities:
                self.responsibilities.append("team_management")
        elif self.role == "architect":
            if "technical_design" not in self.responsibilities:
                self.responsibilities.append("technical_design")


@graph_relationship(from_entity=User, to_entity=Company, relationship_type="EMPLOYED_BY")
class EmployedBy(GraphRelationship):
    """User employed by company relationship."""
    position: str = Field(min_length=1, max_length=100)
    salary: int = Field(gt=0)
    start_date: datetime = Field(default_factory=datetime.now)
    end_date: Optional[datetime] = Field(None)
    department: str = Field(max_length=100)
    is_remote: bool = Field(default=False)
    
    @property
    def is_current(self) -> bool:
        """Check if employment is current."""
        return self.end_date is None
    
    @property
    def duration_months(self) -> int:
        """Calculate employment duration in months."""
        end = self.end_date or datetime.now()
        delta = end - self.start_date
        return max(1, int(delta.days / 30.44))  # Ensure at least 1 month


@graph_relationship(from_entity=User, to_entity=User, relationship_type="MANAGES")
class Manages(GraphRelationship):
    """User manages user relationship (self-referential)."""
    since: datetime = Field(default_factory=datetime.now)
    management_type: Literal["direct", "matrix", "project"] = Field(default="direct")
    performance_rating: Optional[int] = Field(None, ge=1, le=5)
    notes: str = Field(default="")


@graph_relationship(from_entity=Project, to_entity=Company, relationship_type="OWNED_BY", directed=False)
class OwnedBy(GraphRelationship):
    """Project owned by company (undirected)."""
    ownership_percentage: float = Field(ge=0.0, le=100.0, default=100.0)
    investment_amount: int = Field(ge=0, default=0)
    contract_type: Literal["internal", "external", "joint_venture"] = Field(default="internal")


# Relationship with lifecycle tracking
class TrackedRelationship(GraphRelationship):
    """Relationship that tracks lifecycle events."""
    name: str = Field(min_length=1)
    value: int = Field(default=0)
    status: str = Field(default="draft")
    
    def __init__(self, **data):
        super().__init__(**data)
        self._lifecycle_events: List[str] = []
    
    @property
    def lifecycle_events(self) -> List[str]:
        return self._lifecycle_events
    
    def clear_events(self):
        self._lifecycle_events.clear()
    
    async def _pre_save(self):
        self._lifecycle_events.append(f'pre_save:{self.name}')
        if self.status == "draft":
            self.status = "processing"
    
    async def _post_save(self):
        self._lifecycle_events.append(f'post_save:{self.name}')
    
    async def _pre_delete(self):
        self._lifecycle_events.append(f'pre_delete:{self.name}')
    
    async def _post_delete(self):
        self._lifecycle_events.append(f'post_delete:{self.name}')


# =============================================================================
# TEST PYDANTIC V2 RELATIONSHIP BEHAVIOR
# =============================================================================

class TestPydanticV2RelationshipBehavior:
    """Test core Pydantic V2 relationship behavior and validation."""
    
    def test_basic_relationship_creation(self, setup_test_entities):
        """Test basic relationship creation with validation."""
        user = User(name="Alice", email="alice@example.com", age=30)
        project = Project(title="Test Project", description="A test project")
        
        relationship = WorksOn(
            from_id=user.id,
            to_id=project.id,
            role="developer",
            hours_per_week=35,
            responsibilities=["coding", "testing"]
        )
        
        # Test basic properties
        assert relationship.role == "developer"
        assert relationship.hours_per_week == 35
        assert relationship.is_active is True  # Default value
        assert relationship.responsibilities == ["coding", "testing"]
        assert relationship.weight == 1.0  # Default weight
        
        # Test automatic fields
        assert isinstance(relationship.id, str)
        assert len(relationship.id) > 20  # UUID
        assert isinstance(relationship.created_at, datetime)
        assert isinstance(relationship.updated_at, datetime)
        assert relationship.from_id == user.id
        assert relationship.to_id == project.id
        
        # Test private attributes
        assert relationship._graph_edge is not None  # Should auto-sync
        assert relationship._is_persisted is False
        assert relationship._original_data is not None
    
    def test_relationship_field_validation(self):
        """Test Pydantic field validation on relationships."""
        user = User(name="Alice", email="alice@example.com", age=30)
        project = Project(title="Test Project", description="A test project")
        
        # Test invalid role
        with pytest.raises(ValidationError) as exc_info:
            WorksOn(
                from_id=user.id,
                to_id=project.id,
                role="invalid_role",  # Not in Literal choices
                hours_per_week=40
            )
        assert "Input should be" in str(exc_info.value)
        
        # Test hours validation (too high)
        with pytest.raises(ValidationError) as exc_info:
            WorksOn(
                from_id=user.id,
                to_id=project.id,
                role="developer",
                hours_per_week=70  # Too many hours
            )
        assert "less than or equal to 60" in str(exc_info.value)
        
        # Test custom validator (lead role with low hours)
        with pytest.raises(ValidationError) as exc_info:
            WorksOn(
                from_id=user.id,
                to_id=project.id,
                role="lead",
                hours_per_week=10  # Too few for lead
            )
        assert "Lead roles require at least 20 hours/week" in str(exc_info.value)
    
    def test_frozen_fields(self):
        """Test that frozen fields cannot be modified after creation."""
        user = User(name="Alice", email="alice@example.com", age=30)
        project = Project(title="Test Project", description="A test project")
        
        relationship = WorksOn(
            from_id=user.id,
            to_id=project.id,
            role="developer"
        )
        
        original_id = relationship.id
        original_from_id = relationship.from_id
        original_created = relationship.created_at
        
        # Attempt to modify frozen fields should raise ValidationError
        with pytest.raises(ValidationError):
            relationship.id = "new_id"
        
        with pytest.raises(ValidationError):
            relationship.from_id = "new_from_id"
        
        with pytest.raises(ValidationError):
            relationship.created_at = datetime.now()
        
        # Verify they haven't changed
        assert relationship.id == original_id
        assert relationship.from_id == original_from_id
        assert relationship.created_at == original_created
    
    def test_relationship_properties(self):
        """Test computed properties on relationships."""
        user = User(name="Alice", email="alice@example.com", age=30)
        company = Company(name="TechCorp", industry="Technology", founded_year=2020)
        
        # Current employment - use longer duration to ensure >= 12 months
        current_job = EmployedBy(
            from_id=user.id,
            to_id=company.id,
            position="Senior Developer",
            salary=95000,
            department="Engineering",
            start_date=datetime.now() - timedelta(days=400)  # More than 1 year ago
        )
        
        assert current_job.is_current is True
        assert current_job.duration_months >= 12
        
        # Past employment - use longer duration to ensure >= 12 months
        past_job = EmployedBy(
            from_id=user.id,
            to_id=company.id,
            position="Junior Developer",
            salary=65000,
            department="Engineering",
            start_date=datetime.now() - timedelta(days=800),  # Over 2 years ago
            end_date=datetime.now() - timedelta(days=400)     # Over 1 year ago
        )
        
        assert past_job.is_current is False
        assert past_job.duration_months >= 12


# =============================================================================
# TEST GRAPH INTEGRATION
# =============================================================================

class TestRelationshipGraphIntegration:
    """Test automatic graph integration and edge synchronization."""
    
    def test_relationship_auto_sync_to_graph(self, setup_test_entities):
        """Test relationship automatically creates graph edge."""
        graph = setup_test_entities
        
        user = User(name="Alice", email="alice@example.com", age=30)
        project = Project(title="Test Project", description="A test project")
        
        relationship = WorksOn(
            from_id=user.id,
            to_id=project.id,
            role="developer",
            hours_per_week=40
        )
        
        # Should automatically create graph edge
        assert relationship._graph_edge is not None
        assert relationship._graph_edge.from_id == user.id
        assert relationship._graph_edge.to_id == project.id
        assert relationship._graph_edge.relationship_type == "WORKS_ON"
        
        # Check that graph contains the edge
        assert graph.has_edge(user.id, project.id, "WORKS_ON")
        edges = graph.get_edges_between(user.id, project.id)
        assert len(edges) >= 1
        
        # Check edge properties
        edge = relationship._graph_edge
        properties = edge.properties
        assert properties["role"] == "developer"
        assert properties["hours_per_week"] == 40
        assert properties["is_active"] is True
        
        # ID fields should not be in properties
        assert "id" not in properties
        assert "from_id" not in properties
        assert "to_id" not in properties
    
    def test_relationship_update_sync(self, setup_test_entities):
        """Test that relationship changes sync to graph edge."""
        user = User(name="Alice", email="alice@example.com", age=30)
        project = Project(title="Test Project", description="A test project")
        
        relationship = WorksOn(
            from_id=user.id,
            to_id=project.id,
            role="developer",
            hours_per_week=40
        )
        
        # Modify relationship
        relationship.role = "lead"
        relationship.hours_per_week = 45
        relationship.responsibilities = ["team_management", "code_review"]
        
        # Manually sync (in real usage, this happens on save)
        relationship._sync_to_edge()
        
        # Check that graph edge was updated
        edge = relationship._graph_edge
        assert edge.properties["role"] == "lead"
        assert edge.properties["hours_per_week"] == 45
        assert edge.properties["responsibilities"] == ["team_management", "code_review"]
    
    def test_undirected_relationship(self, setup_test_entities):
        """Test undirected relationships create proper graph edges."""
        project = Project(title="Test Project", description="A test project")
        company = Company(name="TechCorp", industry="Technology", founded_year=2020)
        
        # Configure OwnedBy to use test graph
        OwnedBy._relationship_config.graph = setup_test_entities
        
        ownership = OwnedBy(
            from_id=project.id,
            to_id=company.id,
            ownership_percentage=100.0,
            contract_type="internal"
        )
        
        # Should create undirected edge
        assert ownership._graph_edge is not None
        assert ownership._graph_edge.directed is False
        
        # Should be navigable in both directions
        graph = setup_test_entities
        assert graph.has_edge(project.id, company.id, "OWNED_BY")
        # For undirected edges, navigation should work both ways
        edges_forward = graph.get_edges_between(project.id, company.id)
        edges_reverse = graph.get_edges_between(company.id, project.id)
        assert len(edges_forward) >= 1 or len(edges_reverse) >= 1


# =============================================================================
# TEST CRUD OPERATIONS
# =============================================================================

@pytest.mark.asyncio
class TestRelationshipCRUDOperations:
    """Test Create, Read, Update, Delete operations for relationships."""
    
    async def test_create_relationship_via_class_method(self, setup_test_entities):
        """Test creating relationship via class method."""
        user = await User.create(name="Alice", email="alice@example.com", age=30)
        project = await Project.create(title="Test Project", description="A test project")
        
        # Create relationship via class method
        relationship = await WorksOn.create(
            from_node=user,
            to_node=project,
            role="lead",
            hours_per_week=45,
            responsibilities=["architecture", "team_management"]
        )
        
        # Should be automatically saved
        assert relationship._is_persisted
        assert not relationship.is_dirty()
        assert relationship.from_id == user.id
        assert relationship.to_id == project.id
        assert relationship.role == "lead"
        
        # Entity references should be cached
        assert relationship._from_entity is user
        assert relationship._to_entity is project
    
    async def test_entity_type_validation(self, setup_test_entities):
        """Test that entity types are validated."""
        user = await User.create(name="Alice", email="alice@example.com", age=30)
        company = await Company.create(name="TechCorp", industry="Technology", founded_year=2020)
        
        # Should work with correct types
        employment = await EmployedBy.create(
            from_node=user,
            to_node=company,
            position="Developer",
            salary=75000,
            department="Engineering"
        )
        assert employment._is_persisted
        
        # Should fail with wrong types
        with pytest.raises(TypeError, match="from_node must be instance of User"):
            await WorksOn.create(
                from_node=company,  # Wrong type
                to_node=user,
                role="developer"
            )
    
    async def test_save_relationship(self, setup_test_entities):
        """Test saving relationships with changes."""
        user = await User.create(name="Alice", email="alice@example.com", age=30)
        project = await Project.create(title="Test Project", description="A test project")
        
        relationship = WorksOn(
            from_id=user.id,
            to_id=project.id,
            role="developer",
            hours_per_week=35
        )
        
        # Initially not persisted
        assert not relationship._is_persisted
        
        # Save relationship
        await relationship.save()
        
        assert relationship._is_persisted
        assert not relationship.is_dirty()
        
        # Make changes
        relationship.role = "lead"
        relationship.hours_per_week = 45
        relationship.responsibilities = ["team_management"]
        
        assert relationship.is_dirty()
        
        # Save changes
        await relationship.save()
        
        assert not relationship.is_dirty()
        assert relationship._original_data["role"] == "lead"
        assert relationship._original_data["hours_per_week"] == 45
    
    async def test_delete_relationship(self, setup_test_entities):
        """Test deleting relationships."""
        user = await User.create(name="Alice", email="alice@example.com", age=30)
        project = await Project.create(title="Test Project", description="A test project")
        
        relationship = await WorksOn.create(
            from_node=user,
            to_node=project,
            role="developer"
        )
        
        relationship_id = relationship.id
        
        # Verify relationship exists in graph
        graph = setup_test_entities
        assert graph.has_edge(user.id, project.id, "WORKS_ON")
        assert relationship._is_persisted
        
        # Delete relationship
        result = await relationship.delete()
        
        # Check results
        assert result is True
        assert not relationship._is_persisted
        assert relationship._graph_edge is None
        assert not graph.has_edge(user.id, project.id, "WORKS_ON")
        
        # Should be removed from instance cache
        assert relationship_id not in WorksOn._instances
    
    async def test_get_relationship_by_id(self, setup_test_entities):
        """Test retrieving relationship by ID."""
        user = await User.create(name="Alice", email="alice@example.com", age=30)
        project = await Project.create(title="Test Project", description="A test project")
        
        # Create relationship
        original = await WorksOn.create(
            from_node=user,
            to_node=project,
            role="architect",
            hours_per_week=40
        )
        
        # Retrieve by ID
        retrieved = await WorksOn.get(original.id)
        
        assert retrieved is not None
        assert retrieved.id == original.id
        assert retrieved.role == "architect"
        assert retrieved.hours_per_week == 40
        assert retrieved._is_persisted
    
    async def test_get_or_create_relationship(self, setup_test_entities):
        """Test get_or_create for relationships."""
        user = await User.create(name="Alice", email="alice@example.com", age=30)
        project = await Project.create(title="Test Project", description="A test project")
        
        # Create new relationship
        relationship1, created1 = await WorksOn.get_or_create(
            from_node=user,
            to_node=project,
            defaults={
                "role": "developer",
                "hours_per_week": 40
            }
        )
        
        assert created1 is True
        assert relationship1.role == "developer"
        assert relationship1._is_persisted
        
        # Try to get existing relationship - should find the existing one
        relationship2, created2 = await WorksOn.get_or_create(
            from_node=user,
            to_node=project,
            defaults={
                "role": "lead",  # This should NOT be used since relationship exists
                "hours_per_week": 35
            }
        )
        
        assert created2 is False  # Should find existing
        assert relationship2.id == relationship1.id  # Same relationship
        assert relationship2.role == "developer"  # Should keep original role
    
    async def test_entity_references(self, setup_test_entities):
        """Test getting entity references from relationships."""
        user = await User.create(name="Alice", email="alice@example.com", age=30)
        project = await Project.create(title="Test Project", description="A test project")
        
        relationship = await WorksOn.create(
            from_node=user,
            to_node=project,
            role="developer"
        )
        
        # Test async entity access
        from_entity = await relationship.get_from_entity()
        to_entity = await relationship.get_to_entity()
        
        assert from_entity is not None
        assert to_entity is not None
        assert from_entity.id == user.id
        assert to_entity.id == project.id
        assert from_entity.name == "Alice"
        assert to_entity.title == "Test Project"
        
        # Test cached access
        assert relationship.from_entity is from_entity
        assert relationship.to_entity is to_entity


# =============================================================================
# TEST RELATIONSHIP NAVIGATION & QUERYING
# =============================================================================

@pytest.mark.asyncio
class TestRelationshipNavigation:
    """Test relationship navigation and querying systems."""
    
    async def test_basic_relationship_query(self, setup_test_entities):
        """Test basic relationship querying."""
        # Create test data
        user = await User.create(name="Alice", email="alice@example.com", age=30)
        project1 = await Project.create(title="Project Alpha", description="First project")
        project2 = await Project.create(title="Project Beta", description="Second project")
        project3 = await Project.create(title="Project Gamma", description="Third project")
        
        # Create relationships
        rel1 = await WorksOn.create(from_node=user, to_node=project1, role="developer", hours_per_week=20)
        rel2 = await WorksOn.create(from_node=user, to_node=project2, role="lead", hours_per_week=30)
        rel3 = await WorksOn.create(from_node=user, to_node=project3, role="architect", hours_per_week=25)
        
        # Create relationship manager
        manager = RelationshipManager(user)
        
        # Test getting all outgoing relationships
        all_work = await manager.outgoing(WorksOn).all()
        assert len(all_work) == 3
        
        work_roles = {rel.role for rel in all_work}
        assert work_roles == {"developer", "lead", "architect"}
    
    async def test_relationship_filtering(self, setup_test_entities):
        """Test relationship filtering capabilities."""
        user = await User.create(name="Alice", email="alice@example.com", age=30)
        project1 = await Project.create(title="Project Alpha", description="First project")
        project2 = await Project.create(title="Project Beta", description="Second project")
        project3 = await Project.create(title="Project Gamma", description="Third project")
        
        # Create relationships with different properties - ensure they're actually saved
        rel1 = await WorksOn.create(from_node=user, to_node=project1, role="developer", hours_per_week=20, is_active=True)
        rel2 = await WorksOn.create(from_node=user, to_node=project2, role="lead", hours_per_week=40, is_active=True)
        rel3 = await WorksOn.create(from_node=user, to_node=project3, role="developer", hours_per_week=15, is_active=False)
        
        # Verify all relationships were created
        manager = RelationshipManager(user)
        all_relationships = await manager.outgoing(WorksOn).all()
        assert len(all_relationships) == 3, f"Expected 3 relationships, got {len(all_relationships)}"
        
        # Filter by role
        dev_work = await manager.outgoing(WorksOn).filter(role="developer").all()
        assert len(dev_work) == 2, f"Expected 2 developer relationships, got {len(dev_work)}"
        
        # Filter by active status - should find 2 active relationships
        active_work = await manager.outgoing(WorksOn).filter(is_active=True).all()
        assert len(active_work) == 2, f"Expected 2 active relationships, got {len(active_work)} with is_active values: {[r.is_active for r in all_relationships]}"
        
        # Filter by multiple criteria
        active_dev_work = await manager.outgoing(WorksOn).filter(role="developer", is_active=True).all()
        assert len(active_dev_work) == 1
        assert active_dev_work[0].hours_per_week == 20
        
        # Filter with comparison operators
        high_hour_work = await manager.outgoing(WorksOn).filter(hours_per_week__gte=30).all()
        assert len(high_hour_work) == 1
        assert high_hour_work[0].role == "lead"
    
    async def test_relationship_ordering(self, setup_test_entities):
        """Test relationship ordering."""
        user = await User.create(name="Alice", email="alice@example.com", age=30)
        
        # Create relationships with specific hours
        project1 = await Project.create(title="Project Alpha", description="First project")
        rel1 = await WorksOn.create(from_node=user, to_node=project1, role="developer", hours_per_week=40)
        
        project2 = await Project.create(title="Project Beta", description="Second project")
        rel2 = await WorksOn.create(from_node=user, to_node=project2, role="lead", hours_per_week=20)
        
        project3 = await Project.create(title="Project Gamma", description="Third project")
        rel3 = await WorksOn.create(from_node=user, to_node=project3, role="architect", hours_per_week=30)
        
        manager = RelationshipManager(user)
        
        # Order by hours per week (ascending)
        by_hours_asc = await manager.outgoing(WorksOn).order_by('hours_per_week').all()
        hours_order = [rel.hours_per_week for rel in by_hours_asc]
        assert hours_order == [20, 30, 40]
        
        # Order by hours per week (descending)
        by_hours_desc = await manager.outgoing(WorksOn).order_by('-hours_per_week').all()
        hours_order_desc = [rel.hours_per_week for rel in by_hours_desc]
        assert hours_order_desc == [40, 30, 20]
    
    async def test_relationship_pagination(self, setup_test_entities):
        """Test relationship pagination."""
        user = await User.create(name="Alice", email="alice@example.com", age=30)
        
        # Create multiple relationships
        projects = []
        for i in range(5):
            project = await Project.create(title=f"Project {i}", description=f"Project {i}")
            projects.append(project)
            await WorksOn.create(from_node=user, to_node=project, role="developer", hours_per_week=20 + i)
        
        manager = RelationshipManager(user)
        
        # Test limit
        limited = await manager.outgoing(WorksOn).limit(3).all()
        assert len(limited) == 3
        
        # Test offset
        offset_results = await manager.outgoing(WorksOn).offset(2).all()
        assert len(offset_results) == 3  # 5 total - 2 offset = 3
        
        # Test limit + offset
        paginated = await manager.outgoing(WorksOn).offset(1).limit(2).all()
        assert len(paginated) == 2
    
    async def test_target_entity_navigation(self, setup_test_entities):
        """Test navigating to target entities through relationships."""
        user = await User.create(name="Alice", email="alice@example.com", age=30)
        project1 = await Project.create(title="Active Project", description="Active", status="active")
        project2 = await Project.create(title="Planning Project", description="Planning", status="planning")
        project3 = await Project.create(title="Completed Project", description="Completed", status="completed")
        
        # Create relationships
        await WorksOn.create(from_node=user, to_node=project1, role="lead")
        await WorksOn.create(from_node=user, to_node=project2, role="developer")
        await WorksOn.create(from_node=user, to_node=project3, role="developer")
        
        manager = RelationshipManager(user)
        
        # Get all projects user works on
        all_projects = await manager.outgoing(WorksOn).target.all()
        assert len(all_projects) == 3
        
        project_titles = {p.title for p in all_projects}
        assert "Active Project" in project_titles
        assert "Planning Project" in project_titles
        assert "Completed Project" in project_titles
        
        # Filter target entities
        active_projects = await manager.outgoing(WorksOn).target.filter(status="active").all()
        assert len(active_projects) == 1
        assert active_projects[0].title == "Active Project"
        
        # Combine relationship and entity filtering
        lead_active_projects = await manager.outgoing(WorksOn).filter(role="lead").target.filter(status="active").all()
        assert len(lead_active_projects) == 1
        assert lead_active_projects[0].title == "Active Project"
    
    async def test_bidirectional_navigation(self, setup_test_entities):
        """Test bidirectional relationship navigation."""
        user1 = await User.create(name="Alice", email="alice@example.com", age=30, role="manager")
        user2 = await User.create(name="Bob", email="bob@example.com", age=25, role="developer")
        user3 = await User.create(name="Charlie", email="charlie@example.com", age=28, role="developer")
        
        # Configure Manages relationship to use test graph
        Manages._relationship_config.graph = setup_test_entities
        
        # Create management relationships
        await Manages.create(from_node=user1, to_node=user2, management_type="direct")
        await Manages.create(from_node=user1, to_node=user3, management_type="direct")
        
        # Test outgoing (who Alice manages)
        alice_manager = RelationshipManager(user1)
        managed_users = await alice_manager.outgoing(Manages).target.all()
        assert len(managed_users) == 2
        managed_names = {u.name for u in managed_users}
        assert managed_names == {"Bob", "Charlie"}
        
        # Test incoming (who manages Bob)
        bob_manager = RelationshipManager(user2)
        managers = await bob_manager.incoming(Manages).source.all()
        assert len(managers) == 1
        assert managers[0].name == "Alice"


# =============================================================================
# TEST LIFECYCLE HOOKS & BUSINESS LOGIC
# =============================================================================

@pytest.mark.asyncio
class TestRelationshipLifecycleHooks:
    """Test relationship lifecycle hooks and business logic."""
    
    async def test_pre_save_hook(self, setup_test_entities):
        """Test pre-save hook business logic."""
        user = await User.create(name="Alice", email="alice@example.com", age=30)
        project = await Project.create(title="Test Project", description="A test project")
        
        # Create lead relationship
        relationship = WorksOn(
            from_id=user.id,
            to_id=project.id,
            role="lead",
            hours_per_week=45
        )
        
        # Pre-save hook should add team_management responsibility
        await relationship.save()
        
        assert "team_management" in relationship.responsibilities
        
        # Create architect relationship
        architect_rel = WorksOn(
            from_id=user.id,
            to_id=project.id,
            role="architect",
            hours_per_week=40
        )
        
        await architect_rel.save()
        
        assert "technical_design" in architect_rel.responsibilities
    
    async def test_lifecycle_event_tracking(self, setup_test_entities):
        """Test comprehensive lifecycle event tracking."""
        # Configure TrackedRelationship to use test graph
        TrackedRelationship._relationship_config = type('Config', (), {
            'graph': setup_test_entities,
            'relationship_type': 'TRACKED',
            'auto_sync': True,
            'track_changes': True,
            'directed': True,
            'from_entity': None,
            'to_entity': None
        })()
        
        user = await User.create(name="Alice", email="alice@example.com", age=30)
        project = await Project.create(title="Test Project", description="A test project")
        
        relationship = TrackedRelationship(
            from_id=user.id,
            to_id=project.id,
            name="Test Tracked",
            value=10
        )
        
        # Clear any initialization events
        relationship.clear_events()
        
        # Save relationship
        await relationship.save()
        
        # Check save hooks
        events = relationship.lifecycle_events
        assert len(events) >= 2
        assert any("pre_save:" in event for event in events)
        assert any("post_save:" in event for event in events)
        
        # Check business logic from pre_save
        assert relationship.status == "processing"  # Changed from "draft"
        
        # Test delete hooks
        relationship.clear_events()
        await relationship.delete()
        
        delete_events = relationship.lifecycle_events
        assert any("pre_delete:" in event for event in delete_events)
        assert any("post_delete:" in event for event in delete_events)


# =============================================================================
# TEST CHANGE TRACKING
# =============================================================================

class TestRelationshipChangeTracking:
    """Test relationship change tracking and dirty field detection."""
    
    def test_clean_relationship_state(self, setup_test_entities):
        """Test that new relationships start in clean state."""
        user = User(name="Alice", email="alice@example.com", age=30)
        project = Project(title="Test Project", description="A test project")
        
        relationship = WorksOn(
            from_id=user.id,
            to_id=project.id,
            role="developer"
        )
        
        # Should not be dirty initially
        assert not relationship.is_dirty()
        assert relationship.get_dirty_fields() == {}
        assert relationship._original_data is not None
    
    def test_dirty_field_detection(self, setup_test_entities):
        """Test detection of modified relationship fields."""
        user = User(name="Alice", email="alice@example.com", age=30)
        project = Project(title="Test Project", description="A test project")
        
        relationship = WorksOn(
            from_id=user.id,
            to_id=project.id,
            role="developer",
            hours_per_week=40,
            responsibilities=["coding"]
        )
        
        # Initially clean
        assert not relationship.is_dirty()
        
        # Modify fields
        relationship.role = "lead"
        relationship.hours_per_week = 45
        relationship.responsibilities = ["coding", "mentoring"]
        
        # Should be dirty now
        assert relationship.is_dirty()
        
        # Check dirty fields
        dirty_fields = relationship.get_dirty_fields()
        assert "role" in dirty_fields
        assert "hours_per_week" in dirty_fields
        assert "responsibilities" in dirty_fields
        assert dirty_fields["role"] == "lead"
        assert dirty_fields["hours_per_week"] == 45
    
    def test_updated_at_automatic_change(self, setup_test_entities):
        """Test that updated_at changes automatically."""
        user = User(name="Alice", email="alice@example.com", age=30)
        project = Project(title="Test Project", description="A test project")
        
        relationship = WorksOn(
            from_id=user.id,
            to_id=project.id,
            role="developer"
        )
        
        original_updated = relationship.updated_at
        
        # Small delay to ensure timestamp difference
        import time
        time.sleep(0.001)
        
        # Modify field
        relationship.role = "lead"
        
        # updated_at should have changed
        assert relationship.updated_at > original_updated


# =============================================================================
# TEST RELATIONSHIP DESCRIPTOR SYSTEM
# =============================================================================

@pytest.mark.asyncio
class TestRelationshipDescriptors:
    """Test relationship descriptor system for entity classes."""
    
    async def test_relationship_descriptor_access(self, setup_test_entities):
        """Test accessing relationships via descriptors."""
        # Add relationship descriptors to entities
        from neo4jalchemy.orm.relationship_managers import relationship
        
        # Dynamically add descriptors for testing
        User.works_on = relationship(WorksOn, direction="outgoing")
        Project.worked_on_by = relationship(WorksOn, direction="incoming")
        
        user = await User.create(name="Alice", email="alice@example.com", age=30)
        project = await Project.create(title="Test Project", description="A test project")
        
        # Create relationship
        await WorksOn.create(from_node=user, to_node=project, role="developer")
        
        # Test descriptor access
        user_work = await user.works_on.all()
        assert len(user_work) == 1
        assert user_work[0].role == "developer"
        
        project_workers = await project.worked_on_by.all()
        assert len(project_workers) == 1
        assert project_workers[0].role == "developer"
    
    async def test_relationship_descriptor_filtering(self, setup_test_entities):
        """Test filtering through relationship descriptors."""
        # Add descriptors
        from neo4jalchemy.orm.relationship_managers import relationship
        User.works_on = relationship(WorksOn, direction="outgoing")
        
        user = await User.create(name="Alice", email="alice@example.com", age=30)
        project1 = await Project.create(title="Project Alpha", description="First")
        project2 = await Project.create(title="Project Beta", description="Second")
        
        # Create relationships
        await WorksOn.create(from_node=user, to_node=project1, role="developer", is_active=True)
        await WorksOn.create(from_node=user, to_node=project2, role="lead", is_active=False)
        
        # Test filtering via descriptor
        active_work = await user.works_on.filter(is_active=True).all()
        assert len(active_work) == 1
        assert active_work[0].role == "developer"
        
        lead_work = await user.works_on.filter(role="lead").all()
        assert len(lead_work) == 1
        assert lead_work[0].is_active is False


# =============================================================================
# TEST EDGE CASES & ERROR HANDLING
# =============================================================================

class TestRelationshipEdgeCases:
    """Test edge cases and error handling scenarios."""
    
    def test_relationship_string_representation(self, setup_test_entities):
        """Test relationship string representation."""
        user = User(name="Alice", email="alice@example.com", age=30)
        project = Project(title="Test Project", description="A test project")
        
        relationship = WorksOn(
            from_id=user.id,
            to_id=project.id,
            role="developer"
        )
        
        # Test repr
        repr_str = repr(relationship)
        assert "WorksOn" in repr_str
        assert user.id in repr_str
        assert project.id in repr_str
        assert "(new)" in repr_str
        
        # Mark as persisted
        relationship._is_persisted = True
        repr_str = repr(relationship)
        assert "(persisted)" in repr_str
        
        # Make dirty
        relationship.role = "lead"
        repr_str = repr(relationship)
        assert "dirty" in repr_str
    
    def test_self_referential_relationships(self, setup_test_entities):
        """Test self-referential relationships (user manages user)."""
        # Configure Manages to use test graph
        Manages._relationship_config.graph = setup_test_entities
        
        manager = User(name="Alice", email="alice@example.com", age=35, role="manager")
        employee = User(name="Bob", email="bob@example.com", age=28, role="developer")
        
        # Create management relationship
        management = Manages(
            from_id=manager.id,
            to_id=employee.id,
            management_type="direct",
            performance_rating=4
        )
        
        assert management.from_id == manager.id
        assert management.to_id == employee.id
        assert management.management_type == "direct"
        assert management._graph_edge is not None
    
    def test_relationship_without_graph(self):
        """Test relationship behavior when no graph is configured."""
        # Create relationship class without graph
        class NoGraphRelationship(GraphRelationship):
            name: str = Field(min_length=1)
            value: int = Field(default=0)
        
        # Set proper config with all required attributes
        NoGraphRelationship._relationship_config = type('Config', (), {
            'relationship_type': 'NO_GRAPH',
            'graph': None,
            'auto_sync': True,
            'track_changes': True,
            'directed': True,
            'from_entity': None,
            'to_entity': None
        })()
        
        user = User(name="Alice", email="alice@example.com", age=30)
        project = Project(title="Test Project", description="A test project")
        
        relationship = NoGraphRelationship(
            from_id=user.id,
            to_id=project.id,
            name="No Graph Test",
            value=42
        )
        
        # Should not create graph edge
        assert relationship._graph_edge is None
        
        # Should still function normally
        assert relationship.name == "No Graph Test"
        assert relationship.value == 42
        assert not relationship._is_persisted


if __name__ == "__main__":
    pytest.main([__file__, "-v"])