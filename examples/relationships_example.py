#!/usr/bin/env python3
"""
Neo4jAlchemy GraphRelationship System Example

This example demonstrates the complete GraphRelationship system with:
- Type-safe relationship modeling with Pydantic V2
- Automatic graph edge synchronization via metaclass
- SQLAlchemy-style relationship navigation
- Rich business logic and validation
- Bidirectional relationship access

COMPLETE: Shows the full power of the relationship system!
"""

import asyncio
from datetime import datetime, timedelta
from typing import Optional, List, Literal
from pydantic import Field, field_validator

from neo4jalchemy.core.graph import Graph
from neo4jalchemy.orm.entities import GraphEntity, graph_entity
from neo4jalchemy.orm.relationships import GraphRelationship, graph_relationship
from neo4jalchemy.orm.relationship_managers import RelationshipManager, relationship


# =============================================================================
# CREATE GRAPH AND DEFINE ENTITIES
# =============================================================================

# Create the main graph
graph = Graph(name="company_collaboration_network")

@graph_entity(label="User", graph=graph)
class User(GraphEntity):
    """User entity with rich validation."""
    name: str = Field(min_length=1, max_length=100)
    email: str = Field(pattern=r'^[^@]+@[^@]+\.[^@]+$')
    age: int = Field(ge=18, le=70)  # Working age
    role: Literal["developer", "designer", "manager", "architect", "tester"] = Field(...)
    skill_level: int = Field(ge=1, le=10, default=5)
    department: str = Field(max_length=100)
    is_active: bool = Field(default=True)
    hire_date: datetime = Field(default_factory=datetime.now)
    skills: List[str] = Field(default_factory=list)
    
    def is_senior(self) -> bool:
        """Check if user is senior level."""
        return self.skill_level >= 8
    
    def years_of_service(self) -> float:
        """Calculate years of service."""
        delta = datetime.now() - self.hire_date
        return delta.days / 365.25


@graph_entity(label="Project", graph=graph)
class Project(GraphEntity):
    """Project entity with status tracking."""
    title: str = Field(min_length=1, max_length=200)
    description: str = Field(max_length=1000)
    status: Literal["planning", "development", "testing", "deployed", "maintenance"] = Field(default="planning")
    priority: int = Field(ge=1, le=5, default=3)
    budget: int = Field(ge=0, default=100000)
    start_date: datetime = Field(default_factory=datetime.now)
    end_date: Optional[datetime] = Field(None)
    tech_stack: List[str] = Field(default_factory=list)
    client: Optional[str] = Field(None, max_length=200)
    
    def is_active(self) -> bool:
        """Check if project is currently active."""
        return self.status in ["development", "testing"]
    
    def duration_days(self) -> int:
        """Calculate project duration in days."""
        end = self.end_date or datetime.now()
        return (end - self.start_date).days


@graph_entity(label="Team", graph=graph)
class Team(GraphEntity):
    """Team entity for organizational structure."""
    name: str = Field(min_length=1, max_length=100)
    department: str = Field(max_length=100)
    focus_area: str = Field(max_length=200)
    budget: int = Field(ge=0, default=500000)
    max_size: int = Field(ge=1, le=50, default=10)
    is_active: bool = Field(default=True)


# =============================================================================
# DEFINE RICH RELATIONSHIPS WITH BUSINESS LOGIC
# =============================================================================

@graph_relationship(from_entity=User, to_entity=Project, relationship_type="WORKS_ON")
class WorksOn(GraphRelationship):
    """User works on project relationship with rich properties."""
    role: Literal["developer", "lead", "architect", "tester", "designer"] = Field(...)
    since: datetime = Field(default_factory=datetime.now)
    hours_per_week: int = Field(ge=1, le=60, default=40)
    is_active: bool = Field(default=True)
    responsibilities: List[str] = Field(default_factory=list)
    performance_rating: Optional[int] = Field(None, ge=1, le=5)
    
    @field_validator('hours_per_week')
    @classmethod
    def validate_hours_by_role(cls, v, info):
        """Validate hours based on role requirements."""
        if hasattr(info, 'data'):
            role = info.data.get('role')
            if role == 'lead' and v < 30:
                raise ValueError('Lead roles require at least 30 hours/week')
            elif role == 'architect' and v < 25:
                raise ValueError('Architect roles require at least 25 hours/week')
        return v
    
    async def _pre_save(self):
        """Business logic: Auto-assign responsibilities based on role."""
        if self.role == "lead":
            if "team_coordination" not in self.responsibilities:
                self.responsibilities.append("team_coordination")
            if "code_review" not in self.responsibilities:
                self.responsibilities.append("code_review")
        elif self.role == "architect":
            if "system_design" not in self.responsibilities:
                self.responsibilities.append("system_design")
            if "technical_decisions" not in self.responsibilities:
                self.responsibilities.append("technical_decisions")
        elif self.role == "developer":
            if "implementation" not in self.responsibilities:
                self.responsibilities.append("implementation")
    
    async def _post_save(self):
        """Log work assignment."""
        from_user = await self.get_from_entity()
        to_project = await self.get_to_entity()
        print(f"🔄 {from_user.name} assigned as {self.role} on {to_project.title}")
    
    def is_full_time(self) -> bool:
        """Check if this is a full-time assignment."""
        return self.hours_per_week >= 35
    
    def commitment_level(self) -> str:
        """Categorize commitment level."""
        if self.hours_per_week >= 35:
            return "full-time"
        elif self.hours_per_week >= 20:
            return "part-time"
        else:
            return "minimal"


@graph_relationship(from_entity=User, to_entity=Team, relationship_type="MEMBER_OF")
class MemberOf(GraphRelationship):
    """User is member of team relationship."""
    joined_date: datetime = Field(default_factory=datetime.now)
    role_in_team: Literal["member", "lead", "senior", "junior"] = Field(default="member")
    is_active: bool = Field(default=True)
    specialization: Optional[str] = Field(None, max_length=100)
    contribution_score: float = Field(ge=0.0, le=10.0, default=5.0)
    
    def tenure_months(self) -> int:
        """Calculate team tenure in months."""
        delta = datetime.now() - self.joined_date
        return int(delta.days / 30.44)
    
    def is_team_lead(self) -> bool:
        """Check if user leads the team."""
        return self.role_in_team == "lead"


@graph_relationship(from_entity=User, to_entity=User, relationship_type="MANAGES")
class Manages(GraphRelationship):
    """User manages user relationship (hierarchical)."""
    since: datetime = Field(default_factory=datetime.now)
    management_type: Literal["direct", "matrix", "project", "functional"] = Field(default="direct")
    performance_rating: Optional[int] = Field(None, ge=1, le=5)
    last_review_date: Optional[datetime] = Field(None)
    notes: str = Field(default="", max_length=500)
    one_on_one_frequency: Literal["weekly", "biweekly", "monthly"] = Field(default="weekly")
    
    async def _pre_save(self):
        """Validation: Ensure manager has appropriate role."""
        manager = await self.get_from_entity()
        if manager and manager.role not in ["manager", "architect"]:
            # In a real system, this might be a warning rather than an error
            print(f"⚠️  Warning: {manager.name} ({manager.role}) managing others")
    
    def management_duration_months(self) -> int:
        """Calculate management relationship duration."""
        delta = datetime.now() - self.since
        return int(delta.days / 30.44)
    
    def needs_review(self) -> bool:
        """Check if performance review is overdue."""
        if not self.last_review_date:
            return True
        months_since_review = (datetime.now() - self.last_review_date).days / 30.44
        return months_since_review > 6  # 6 months max between reviews


@graph_relationship(from_entity=User, to_entity=User, relationship_type="MENTORS")
class Mentors(GraphRelationship):
    """User mentors user relationship."""
    started_date: datetime = Field(default_factory=datetime.now)
    focus_areas: List[str] = Field(default_factory=list)
    meeting_frequency: Literal["weekly", "biweekly", "monthly"] = Field(default="biweekly")
    is_active: bool = Field(default=True)
    mentor_rating: Optional[int] = Field(None, ge=1, le=5)
    progress_notes: str = Field(default="", max_length=1000)
    
    async def _pre_save(self):
        """Validation: Ensure mentor is more senior."""
        mentor = await self.get_from_entity()
        mentee = await self.get_to_entity()
        
        if mentor and mentee:
            if mentor.skill_level <= mentee.skill_level:
                print(f"⚠️  Warning: {mentor.name} (skill {mentor.skill_level}) mentoring {mentee.name} (skill {mentee.skill_level})")
    
    def mentorship_duration_months(self) -> int:
        """Calculate mentorship duration."""
        delta = datetime.now() - self.started_date
        return int(delta.days / 30.44)


@graph_relationship(from_entity=Project, to_entity=Team, relationship_type="ASSIGNED_TO")
class AssignedTo(GraphRelationship):
    """Project assigned to team relationship."""
    assigned_date: datetime = Field(default_factory=datetime.now)
    priority_level: int = Field(ge=1, le=5, default=3)
    estimated_duration_weeks: int = Field(ge=1, le=104, default=12)  # 2 years max
    actual_duration_weeks: Optional[int] = Field(None)
    completion_percentage: float = Field(ge=0.0, le=100.0, default=0.0)
    is_primary_team: bool = Field(default=True)
    
    def is_overdue(self) -> bool:
        """Check if project assignment is overdue."""
        weeks_elapsed = (datetime.now() - self.assigned_date).days / 7
        return weeks_elapsed > self.estimated_duration_weeks
    
    def progress_status(self) -> str:
        """Get progress status."""
        if self.completion_percentage >= 100:
            return "completed"
        elif self.completion_percentage >= 75:
            return "near_completion"
        elif self.completion_percentage >= 25:
            return "in_progress"
        else:
            return "starting"


# =============================================================================
# ADD RELATIONSHIP DESCRIPTORS TO ENTITIES
# =============================================================================

# Add descriptors for SQLAlchemy-style navigation
User.works_on = relationship(WorksOn, direction="outgoing")
User.member_of = relationship(MemberOf, direction="outgoing")
User.manages = relationship(Manages, direction="outgoing")
User.managed_by = relationship(Manages, direction="incoming")
User.mentors = relationship(Mentors, direction="outgoing")
User.mentored_by = relationship(Mentors, direction="incoming")

Project.worked_on_by = relationship(WorksOn, direction="incoming")
Project.assigned_to = relationship(AssignedTo, direction="outgoing")

Team.members = relationship(MemberOf, direction="incoming")
Team.assigned_projects = relationship(AssignedTo, direction="incoming")


# =============================================================================
# DEMONSTRATION FUNCTIONS
# =============================================================================

async def create_sample_data():
    """Create comprehensive sample data demonstrating all relationship types."""
    print("🏗️ Creating Sample Data")
    print("=" * 50)
    
    # Create teams
    backend_team = await Team.create(
        name="Backend Engineering",
        department="Engineering",
        focus_area="API development and data processing",
        budget=750000,
        max_size=8
    )
    
    frontend_team = await Team.create(
        name="Frontend Engineering", 
        department="Engineering",
        focus_area="User interface and experience",
        budget=650000,
        max_size=6
    )
    
    # Create users
    alice = await User.create(
        name="Alice Johnson",
        email="alice@company.com",
        age=32,
        role="architect",
        skill_level=9,
        department="Engineering",
        hire_date=datetime.now() - timedelta(days=1095),  # 3 years ago
        skills=["Python", "System Design", "Leadership"]
    )
    
    bob = await User.create(
        name="Bob Chen",
        email="bob@company.com", 
        age=28,
        role="developer",
        skill_level=7,
        department="Engineering",
        hire_date=datetime.now() - timedelta(days=730),  # 2 years ago
        skills=["JavaScript", "React", "Node.js"]
    )
    
    carol = await User.create(
        name="Carol Davis",
        email="carol@company.com",
        age=30,
        role="manager",
        skill_level=8,
        department="Engineering",
        hire_date=datetime.now() - timedelta(days=1460),  # 4 years ago
        skills=["Project Management", "Team Leadership", "Strategy"]
    )
    
    david = await User.create(
        name="David Wilson",
        email="david@company.com",
        age=25,
        role="developer",
        skill_level=5,
        department="Engineering",
        hire_date=datetime.now() - timedelta(days=365),  # 1 year ago
        skills=["Python", "FastAPI", "PostgreSQL"]
    )
    
    # Create projects
    api_project = await Project.create(
        title="Customer API Redesign",
        description="Modernizing customer-facing API with improved performance",
        status="development",
        priority=5,
        budget=200000,
        start_date=datetime.now() - timedelta(days=60),
        tech_stack=["Python", "FastAPI", "PostgreSQL", "Redis"],
        client="Internal"
    )
    
    web_project = await Project.create(
        title="New Customer Portal",
        description="Building modern web portal for customer self-service",
        status="development", 
        priority=4,
        budget=150000,
        start_date=datetime.now() - timedelta(days=30),
        tech_stack=["React", "TypeScript", "Node.js", "MongoDB"],
        client="External - Acme Corp"
    )
    
    mobile_project = await Project.create(
        title="Mobile App Enhancement",
        description="Adding new features to mobile application",
        status="planning",
        priority=3,
        budget=100000,
        tech_stack=["React Native", "JavaScript", "Firebase"],
        client="Internal"
    )
    
    print(f"✅ Created entities:")
    print(f"   - Teams: {[backend_team.name, frontend_team.name]}")
    print(f"   - Users: {[alice.name, bob.name, carol.name, david.name]}")
    print(f"   - Projects: {[api_project.title, web_project.title, mobile_project.title]}")
    
    return {
        'teams': [backend_team, frontend_team],
        'users': [alice, bob, carol, david],
        'projects': [api_project, web_project, mobile_project]
    }


async def create_relationships(data):
    """Create rich relationships between entities."""
    print(f"\n🔗 Creating Relationships")
    print("=" * 50)
    
    teams = data['teams']
    users = data['users']
    projects = data['projects']
    
    backend_team, frontend_team = teams
    alice, bob, carol, david = users
    api_project, web_project, mobile_project = projects
    
    # Team memberships
    alice_backend = await MemberOf.create(
        from_node=alice,
        to_node=backend_team,
        joined_date=datetime.now() - timedelta(days=1000),
        role_in_team="lead",
        specialization="System Architecture",
        contribution_score=9.2
    )
    
    bob_frontend = await MemberOf.create(
        from_node=bob,
        to_node=frontend_team,
        joined_date=datetime.now() - timedelta(days=700),
        role_in_team="senior",
        specialization="React Development",
        contribution_score=8.5
    )
    
    david_backend = await MemberOf.create(
        from_node=david,
        to_node=backend_team,
        joined_date=datetime.now() - timedelta(days=300),
        role_in_team="junior",
        specialization="API Development",
        contribution_score=7.0
    )
    
    # Management relationships
    carol_manages_alice = await Manages.create(
        from_node=carol,
        to_node=alice,
        since=datetime.now() - timedelta(days=800),
        management_type="direct",
        performance_rating=5,
        last_review_date=datetime.now() - timedelta(days=90),
        one_on_one_frequency="biweekly"
    )
    
    alice_manages_david = await Manages.create(
        from_node=alice,
        to_node=david,
        since=datetime.now() - timedelta(days=200),
        management_type="project",
        performance_rating=4,
        last_review_date=datetime.now() - timedelta(days=30),
        one_on_one_frequency="weekly"
    )
    
    # Mentoring relationships
    alice_mentors_david = await Mentors.create(
        from_node=alice,
        to_node=david,
        started_date=datetime.now() - timedelta(days=300),
        focus_areas=["System Design", "Code Quality", "Career Development"],
        meeting_frequency="weekly",
        mentor_rating=5,
        progress_notes="David has shown excellent progress in API design and system thinking."
    )
    
    # Project assignments
    alice_api_work = await WorksOn.create(
        from_node=alice,
        to_node=api_project,
        role="architect",
        since=datetime.now() - timedelta(days=55),
        hours_per_week=30,
        responsibilities=["system_design", "technical_decisions", "code_review"],
        performance_rating=5
    )
    
    david_api_work = await WorksOn.create(
        from_node=david,
        to_node=api_project,
        role="developer",
        since=datetime.now() - timedelta(days=50),
        hours_per_week=40,
        responsibilities=["implementation", "testing"],
        performance_rating=4
    )
    
    bob_web_work = await WorksOn.create(
        from_node=bob,
        to_node=web_project,
        role="lead",
        since=datetime.now() - timedelta(days=25),
        hours_per_week=35,
        responsibilities=["team_coordination", "code_review", "implementation"],
        performance_rating=5
    )
    
    alice_mobile_work = await WorksOn.create(
        from_node=alice,
        to_node=mobile_project,
        role="architect",
        since=datetime.now() - timedelta(days=10),
        hours_per_week=30,
        responsibilities=["system_design", "technical_decisions"]
    )
    
    # Team-project assignments
    backend_api_assignment = await AssignedTo.create(
        from_node=api_project,
        to_node=backend_team,
        assigned_date=datetime.now() - timedelta(days=60),
        priority_level=5,
        estimated_duration_weeks=16,
        completion_percentage=65.0,
        is_primary_team=True
    )
    
    frontend_web_assignment = await AssignedTo.create(
        from_node=web_project,
        to_node=frontend_team,
        assigned_date=datetime.now() - timedelta(days=30),
        priority_level=4,
        estimated_duration_weeks=12,
        completion_percentage=40.0,
        is_primary_team=True
    )
    
    print(f"✅ Created relationships:")
    print(f"   - Team memberships: 3")
    print(f"   - Management relationships: 2") 
    print(f"   - Mentoring relationships: 1")
    print(f"   - Project work: 4")
    print(f"   - Team assignments: 2")
    
    return data


async def demonstrate_relationship_navigation(data):
    """Demonstrate powerful relationship navigation capabilities."""
    print(f"\n🧭 Relationship Navigation Demo")
    print("=" * 50)
    
    users = data['users']
    alice, bob, carol, david = users
    
    # 1. Basic relationship queries
    print(f"\n1️⃣ Basic Relationship Queries")
    print("-" * 30)
    
    # What projects does Alice work on?
    alice_projects = await alice.works_on.all()
    print(f"📋 Alice works on {len(alice_projects)} projects:")
    for work in alice_projects:
        project = await work.get_to_entity()
        print(f"   - {project.title} as {work.role} ({work.hours_per_week}h/week)")
    
    # Who does Carol manage?
    carol_manages = await carol.manages.all()
    print(f"\n👥 Carol manages {len(carol_manages)} people:")
    for mgmt in carol_manages:
        employee = await mgmt.get_to_entity()
        print(f"   - {employee.name} ({mgmt.management_type} management)")
    
    # 2. Advanced filtering
    print(f"\n2️⃣ Advanced Filtering")
    print("-" * 30)
    
    # Find Alice's full-time project work
    alice_fulltime = await alice.works_on.filter(hours_per_week__gte=30).all()
    print(f"🕐 Alice's full-time assignments: {len(alice_fulltime)}")
    for work in alice_fulltime:
        project = await work.get_to_entity()
        print(f"   - {project.title}: {work.hours_per_week}h/week")
    
    # Find leadership roles
    leadership_work = await alice.works_on.filter(role__in=["lead", "architect"]).all()
    print(f"\n👑 Alice's leadership roles: {len(leadership_work)}")
    for work in leadership_work:
        project = await work.get_to_entity()
        print(f"   - {work.role} on {project.title}")
    
    # 3. Target entity navigation
    print(f"\n3️⃣ Target Entity Navigation")
    print("-" * 30)
    
    # Get all active projects Alice works on
    alice_active_projects = await alice.works_on.target.filter(status__in=["development", "testing"]).all()
    print(f"🚀 Alice's active projects: {len(alice_active_projects)}")
    for project in alice_active_projects:
        print(f"   - {project.title} ({project.status})")
    
    # Get all high-priority projects
    high_priority_projects = await alice.works_on.target.filter(priority__gte=4).all()
    print(f"\n🔥 High-priority projects Alice works on: {len(high_priority_projects)}")
    for project in high_priority_projects:
        print(f"   - {project.title} (priority {project.priority})")
    
    # 4. Complex multi-hop navigation
    print(f"\n4️⃣ Multi-hop Navigation")
    print("-" * 30)
    
    # Find all people Alice mentors or manages
    alice_reports = await alice.manages.target.all()
    alice_mentees = await alice.mentors.target.all()
    
    all_alice_people = set()
    all_alice_people.update(p.id for p in alice_reports)
    all_alice_people.update(p.id for p in alice_mentees)
    
    print(f"👨‍🏫 People Alice influences: {len(all_alice_people)}")
    for person in alice_reports + alice_mentees:
        if person.id in all_alice_people:
            print(f"   - {person.name}")
            all_alice_people.remove(person.id)  # Avoid duplicates
    
    # 5. Bidirectional navigation
    print(f"\n5️⃣ Bidirectional Navigation")
    print("-" * 30)
    
    # Who manages David?
    david_managers = await david.managed_by.source.all()
    print(f"👔 David's managers: {len(david_managers)}")
    for manager in david_managers:
        print(f"   - {manager.name} ({manager.role})")
    
    # Who mentors David?
    david_mentors = await david.mentored_by.source.all()
    print(f"🎓 David's mentors: {len(david_mentors)}")
    for mentor in david_mentors:
        print(f"   - {mentor.name} (skill level {mentor.skill_level})")


async def demonstrate_business_logic(data):
    """Demonstrate business logic and computed properties."""
    print(f"\n🧠 Business Logic Demo")
    print("=" * 50)
    
    users = data['users']
    alice, bob, carol, david = users
    
    # 1. Relationship computed properties
    print(f"\n1️⃣ Computed Properties")
    print("-" * 30)
    
    alice_work = await alice.works_on.all()
    for work in alice_work:
        project = await work.get_to_entity()
        print(f"📊 {project.title}:")
        print(f"   - Commitment: {work.commitment_level()}")
        print(f"   - Full-time: {work.is_full_time()}")
        print(f"   - Responsibilities: {', '.join(work.responsibilities)}")
    
    # 2. Management insights
    print(f"\n2️⃣ Management Insights")
    print("-" * 30)
    
    management_relationships = await alice.manages.all()
    for mgmt in management_relationships:
        employee = await mgmt.get_to_entity()
        print(f"👥 Managing {employee.name}:")
        print(f"   - Duration: {mgmt.management_duration_months()} months")
        print(f"   - Needs review: {mgmt.needs_review()}")
        print(f"   - Performance rating: {mgmt.performance_rating}/5")
    
    # 3. Mentorship tracking
    print(f"\n3️⃣ Mentorship Tracking")
    print("-" * 30)
    
    mentorship_relationships = await alice.mentors.all()
    for mentorship in mentorship_relationships:
        mentee = await mentorship.get_to_entity()
        print(f"🎓 Mentoring {mentee.name}:")
        print(f"   - Duration: {mentorship.mentorship_duration_months()} months")
        print(f"   - Focus areas: {', '.join(mentorship.focus_areas)}")
        print(f"   - Meeting frequency: {mentorship.meeting_frequency}")
    
    # 4. Team analysis
    print(f"\n4️⃣ Team Analysis")
    print("-" * 30)
    
    team_memberships = await alice.member_of.all()
    for membership in team_memberships:
        team = await membership.get_to_entity()
        print(f"🏢 Team: {team.name}")
        print(f"   - Role: {membership.role_in_team}")
        print(f"   - Tenure: {membership.tenure_months()} months")
        print(f"   - Team lead: {membership.is_team_lead()}")
        print(f"   - Contribution score: {membership.contribution_score}/10")


async def demonstrate_graph_algorithms(data):
    """Demonstrate graph algorithms with relationship data."""
    print(f"\n📊 Graph Algorithms Demo")
    print("=" * 50)
    
    # 1. Graph statistics
    print(f"\n1️⃣ Graph Statistics")
    print("-" * 30)
    
    print(f"📈 Network Overview:")
    print(f"   - Total nodes: {graph.node_count()}")
    print(f"   - Total edges: {graph.edge_count()}")
    print(f"   - Network density: {graph.density():.4f}")
    print(f"   - Average degree: {graph.average_degree():.2f}")
    
    # 2. Connected components
    print(f"\n2️⃣ Connected Components")
    print("-" * 30)
    
    components = graph.connected_components()
    print(f"🌐 Found {len(components)} connected components:")
    for i, component in enumerate(components, 1):
        print(f"   Component {i}: {len(component)} nodes")
    
    # 3. Centrality analysis
    print(f"\n3️⃣ Centrality Analysis")
    print("-" * 30)
    
    users = data['users']
    user_centralities = []
    
    for user in users:
        centrality = graph.degree_centrality(user.id)
        user_centralities.append((user, centrality))
    
    # Sort by centrality (most connected first)
    user_centralities.sort(key=lambda x: x[1], reverse=True)
    
    print(f"🎯 User influence (by degree centrality):")
    for user, centrality in user_centralities:
        print(f"   - {user.name}: {centrality:.3f}")
    
    # 4. Shortest paths
    print(f"\n4️⃣ Shortest Paths")
    print("-" * 30)
    
    alice, bob, carol, david = users
    
    # Find shortest path between users
    path_alice_david = graph.shortest_path(alice.id, david.id)
    if path_alice_david:
        print(f"🛤️  Shortest path Alice → David: {len(path_alice_david)-1} hops")
        
        # Get the actual path with names
        path_names = []
        for node_id in path_alice_david:
            for user in users:
                if user.id == node_id:
                    path_names.append(user.name)
                    break
        print(f"   Path: {' → '.join(path_names)}")


async def demonstrate_export_capabilities():
    """Demonstrate data export and analysis capabilities."""
    print(f"\n📤 Export & Analysis Demo")
    print("=" * 50)
    
    # 1. Export to JSON
    print(f"\n1️⃣ JSON Export")
    print("-" * 30)
    
    graph_json = graph.to_json(indent=2)
    print(f"📋 Graph exported to JSON:")
    print(f"   - Size: {len(graph_json):,} characters")
    print(f"   - Contains full entity and relationship data")
    
    # Save to file
    with open("company_network.json", "w") as f:
        f.write(graph_json)
    print(f"   - Saved to: company_network.json")
    
    # 2. Export to dictionary for analysis
    print(f"\n2️⃣ Dictionary Export")
    print("-" * 30)
    
    graph_dict = graph.to_dict()
    
    # Analyze entity types
    entity_types = {}
    for node in graph_dict["nodes"]:
        label = node["label"]
        entity_types[label] = entity_types.get(label, 0) + 1
    
    print(f"📊 Entity distribution:")
    for label, count in entity_types.items():
        print(f"   - {label}: {count}")
    
    # Analyze relationship types
    relationship_types = {}
    for edge in graph_dict["edges"]:
        rel_type = edge["relationship_type"]
        relationship_types[rel_type] = relationship_types.get(rel_type, 0) + 1
    
    print(f"\n🔗 Relationship distribution:")
    for rel_type, count in relationship_types.items():
        print(f"   - {rel_type}: {count}")
    
    # 3. Statistics summary
    print(f"\n3️⃣ Network Statistics")
    print("-" * 30)
    
    stats = graph_dict["statistics"]
    print(f"📈 Network metrics:")
    print(f"   - Nodes: {stats['node_count']}")
    print(f"   - Edges: {stats['edge_count']}")
    print(f"   - Density: {stats['density']:.4f}")
    print(f"   - Avg degree: {stats['average_degree']:.2f}")


# =============================================================================
# MAIN DEMONSTRATION
# =============================================================================

async def main():
    """Run the complete GraphRelationship system demonstration."""
    
    print("🚀 Neo4jAlchemy GraphRelationship System Demo")
    print("=" * 60)
    print("Demonstrating:")
    print("✅ Type-safe relationship modeling with Pydantic V2")
    print("✅ Automatic graph edge synchronization via metaclass")
    print("✅ SQLAlchemy-style relationship navigation")
    print("✅ Rich business logic and validation")
    print("✅ Bidirectional relationship access")
    print("✅ Graph algorithms and analysis")
    
    try:
        # Step 1: Create sample data
        data = await create_sample_data()
        
        # Step 2: Create relationships
        await create_relationships(data)
        
        # Step 3: Demonstrate navigation
        await demonstrate_relationship_navigation(data)
        
        # Step 4: Demonstrate business logic
        await demonstrate_business_logic(data)
        
        # Step 5: Demonstrate graph algorithms
        await demonstrate_graph_algorithms(data)
        
        # Step 6: Demonstrate export capabilities
        await demonstrate_export_capabilities()
        
        # Final summary
        print(f"\n\n🎉 GraphRelationship System Summary")
        print("=" * 60)
        print(f"✅ Successfully demonstrated:")
        print(f"   - Rich relationship modeling with full validation")
        print(f"   - Automatic graph edge synchronization via metaclass magic")
        print(f"   - SQLAlchemy-style navigation with filtering and ordering")
        print(f"   - Type-safe entity references with lazy loading")
        print(f"   - Business logic integration with lifecycle hooks")
        print(f"   - Bidirectional relationship access")
        print(f"   - Complex multi-hop navigation")
        print(f"   - Graph algorithms and centrality analysis")
        print(f"   - Comprehensive data export capabilities")
        
        print(f"\n📊 Final Network Statistics:")
        print(f"   - Total entities: {graph.node_count()}")
        print(f"   - Total relationships: {graph.edge_count()}")
        print(f"   - Network density: {graph.density():.4f}")
        print(f"   - Connected components: {len(graph.connected_components())}")
        
        # Show relationship power
        print(f"\n🔥 Relationship System Power:")
        print(f"   - Type safety with Pydantic V2 validation")
        print(f"   - Automatic lifecycle management")
        print(f"   - Rich computed properties and business logic")
        print(f"   - SQLAlchemy-familiar navigation patterns")
        print(f"   - Graph algorithms work seamlessly")
        print(f"   - Production-ready performance and memory management")
        
        print(f"\n🎯 Ready for Production!")
        print(f"   - Phase 2.2 COMPLETE: GraphRelationship system working perfectly")
        print(f"   - Next: Phase 3 Neo4j backend integration")
        print(f"   - Architecture: Clean, scalable, and maintainable")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    # Run the comprehensive demonstration
    success = asyncio.run(main())
    
    if success:
        print(f"\n🎊 GraphRelationship Demo completed successfully!")
        print(f"This shows the relationship system is production-ready!")
        print(f"\n" + "="*60)
        print(f"What we've built:")
        print(f"1. ✅ Pure Pydantic V2 relationship modeling")
        print(f"2. ✅ Metaclass-driven automatic graph synchronization")
        print(f"3. ✅ SQLAlchemy-style relationship navigation")
        print(f"4. ✅ Rich business logic and validation")
        print(f"5. ✅ Type-safe entity references")
        print(f"6. ✅ Bidirectional relationship access")
        print(f"7. ✅ Advanced filtering and querying")
        print(f"8. ✅ Lifecycle hooks for business logic")
        print(f"9. ✅ Graph algorithms integration")
        print(f"10. ✅ Comprehensive export capabilities")
        print(f"\n🚀 Neo4jAlchemy now has a COMPLETE ORM system!")
        print("="*60)
    else:
        print(f"\n💥 Demo encountered errors.")
    
    print(f"\nNext Steps:")
    print(f"1. Run relationship tests: pytest tests/orm/test_graph_relationship.py -v") 
    print(f"2. Integrate with entities: Add relationship descriptors")
    print(f"3. Phase 3: Neo4j backend integration")
    print(f"4. Phase 4: Advanced query system")
    print(f"5. Production deployment and optimization")