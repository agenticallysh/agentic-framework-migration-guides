#!/usr/bin/env python3
"""
Interactive LangChain to LangGraph Migration Planner
Helps developers create detailed migration plans with timelines and priorities.
"""

import os
import json
import argparse
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import sys

@dataclass
class MigrationTask:
    """Individual migration task."""
    id: str
    title: str
    description: str
    category: str
    complexity: int  # 1-5 scale
    estimated_hours: float
    dependencies: List[str]
    priority: str  # high/medium/low
    file_paths: List[str]
    checklist: List[str]
    resources: List[str]

@dataclass
class MigrationPlan:
    """Complete migration plan."""
    project_name: str
    total_estimated_hours: float
    timeline_weeks: int
    phases: Dict[str, List[MigrationTask]]
    risk_factors: List[str]
    mitigation_strategies: List[str]
    success_criteria: List[str]
    rollback_plan: List[str]
    created_at: str

class MigrationPlannerTool:
    """Interactive tool for creating migration plans."""
    
    def __init__(self):
        self.task_templates = self._load_task_templates()
        self.common_patterns = self._load_common_patterns()
    
    def _load_task_templates(self) -> Dict[str, Dict[str, Any]]:
        """Load common migration task templates."""
        return {
            "setup_environment": {
                "title": "Set up LangGraph development environment",
                "description": "Install LangGraph and set up development environment",
                "category": "setup",
                "complexity": 1,
                "estimated_hours": 1.0,
                "checklist": [
                    "Install langgraph package",
                    "Set up virtual environment",
                    "Update requirements.txt",
                    "Verify compatibility with existing dependencies"
                ],
                "resources": [
                    "https://langchain-ai.github.io/langgraph/",
                    "LangGraph installation guide"
                ]
            },
            "analyze_agent_patterns": {
                "title": "Analyze existing agent patterns",
                "description": "Document current agent architecture and identify migration strategy",
                "category": "analysis",
                "complexity": 2,
                "estimated_hours": 3.0,
                "checklist": [
                    "Map all AgentExecutor instances",
                    "Document agent workflow patterns",
                    "Identify state management requirements",
                    "Catalog custom tools and memory usage"
                ],
                "resources": [
                    "Agent architecture documentation",
                    "Migration pattern guides"
                ]
            },
            "convert_tools": {
                "title": "Convert tools to LangGraph format",
                "description": "Update tool definitions to use @tool decorator",
                "category": "implementation",
                "complexity": 2,
                "estimated_hours": 2.0,
                "checklist": [
                    "Update tool function signatures",
                    "Add @tool decorators",
                    "Update tool descriptions",
                    "Test tool functionality"
                ],
                "resources": [
                    "LangGraph tools documentation",
                    "Tool migration examples"
                ]
            },
            "design_state_schema": {
                "title": "Design state schema",
                "description": "Create TypedDict schema for workflow state",
                "category": "implementation",
                "complexity": 3,
                "estimated_hours": 4.0,
                "checklist": [
                    "Identify all state variables",
                    "Create TypedDict definitions",
                    "Add proper type annotations",
                    "Design state update functions"
                ],
                "resources": [
                    "State management guide",
                    "TypedDict documentation"
                ]
            },
            "create_graph_structure": {
                "title": "Create graph workflow structure",
                "description": "Build StateGraph with nodes and edges",
                "category": "implementation",
                "complexity": 4,
                "estimated_hours": 6.0,
                "checklist": [
                    "Define workflow nodes",
                    "Create conditional edge logic",
                    "Implement node functions",
                    "Add error handling"
                ],
                "resources": [
                    "Graph building guide",
                    "Conditional edges examples"
                ]
            },
            "implement_checkpointing": {
                "title": "Implement state checkpointing",
                "description": "Add persistent state management with checkpointing",
                "category": "implementation",
                "complexity": 3,
                "estimated_hours": 3.0,
                "checklist": [
                    "Set up SQLite checkpointer",
                    "Configure checkpoint behavior",
                    "Test state persistence",
                    "Implement recovery logic"
                ],
                "resources": [
                    "Checkpointing documentation",
                    "State persistence examples"
                ]
            },
            "migrate_memory": {
                "title": "Migrate memory management",
                "description": "Convert LangChain memory to LangGraph state",
                "category": "implementation",
                "complexity": 3,
                "estimated_hours": 4.0,
                "checklist": [
                    "Analyze current memory patterns",
                    "Design state-based memory",
                    "Implement memory functions",
                    "Test memory persistence"
                ],
                "resources": [
                    "Memory migration guide",
                    "State management examples"
                ]
            },
            "testing_validation": {
                "title": "Create comprehensive tests",
                "description": "Build test suite for migrated components",
                "category": "testing",
                "complexity": 3,
                "estimated_hours": 5.0,
                "checklist": [
                    "Create unit tests for nodes",
                    "Test state transitions",
                    "Validate workflow behavior",
                    "Performance testing"
                ],
                "resources": [
                    "Testing best practices",
                    "LangGraph testing examples"
                ]
            },
            "performance_optimization": {
                "title": "Optimize performance",
                "description": "Tune graph performance and efficiency",
                "category": "optimization",
                "complexity": 2,
                "estimated_hours": 3.0,
                "checklist": [
                    "Profile workflow execution",
                    "Optimize state updates",
                    "Minimize unnecessary nodes",
                    "Optimize checkpoint frequency"
                ],
                "resources": [
                    "Performance optimization guide",
                    "Profiling tools"
                ]
            },
            "documentation": {
                "title": "Update documentation",
                "description": "Document new LangGraph architecture",
                "category": "documentation",
                "complexity": 2,
                "estimated_hours": 4.0,
                "checklist": [
                    "Document graph structure",
                    "Update API documentation",
                    "Create usage examples",
                    "Update deployment guides"
                ],
                "resources": [
                    "Documentation templates",
                    "Architecture diagrams"
                ]
            }
        }
    
    def _load_common_patterns(self) -> Dict[str, List[str]]:
        """Load common migration patterns and their task sequences."""
        return {
            "simple_agent": [
                "setup_environment",
                "analyze_agent_patterns", 
                "convert_tools",
                "design_state_schema",
                "create_graph_structure",
                "testing_validation",
                "documentation"
            ],
            "complex_agent": [
                "setup_environment",
                "analyze_agent_patterns",
                "convert_tools", 
                "design_state_schema",
                "migrate_memory",
                "create_graph_structure",
                "implement_checkpointing",
                "testing_validation",
                "performance_optimization",
                "documentation"
            ],
            "multi_agent_system": [
                "setup_environment",
                "analyze_agent_patterns",
                "convert_tools",
                "design_state_schema", 
                "migrate_memory",
                "create_graph_structure",
                "implement_checkpointing",
                "testing_validation",
                "performance_optimization",
                "documentation"
            ]
        }
    
    def create_interactive_plan(self) -> MigrationPlan:
        """Create migration plan through interactive prompts."""
        print("🚀 LangChain to LangGraph Migration Planner")
        print("=" * 50)
        
        # Get project information
        project_name = input("\n📝 Project name: ").strip()
        if not project_name:
            project_name = "LangChain Migration Project"
        
        # Assess project complexity
        complexity = self._assess_project_complexity()
        
        # Select migration pattern
        pattern = self._select_migration_pattern(complexity)
        
        # Customize tasks
        tasks = self._customize_tasks(pattern)
        
        # Organize into phases
        phases = self._organize_phases(tasks)
        
        # Calculate timeline
        total_hours = sum(task.estimated_hours for task in tasks)
        timeline_weeks = self._calculate_timeline(total_hours)
        
        # Add risk factors and strategies
        risk_factors = self._identify_risks(complexity, tasks)
        mitigation_strategies = self._create_mitigation_strategies(risk_factors)
        
        # Define success criteria
        success_criteria = self._define_success_criteria()
        
        # Create rollback plan
        rollback_plan = self._create_rollback_plan()
        
        return MigrationPlan(
            project_name=project_name,
            total_estimated_hours=total_hours,
            timeline_weeks=timeline_weeks,
            phases=phases,
            risk_factors=risk_factors,
            mitigation_strategies=mitigation_strategies,
            success_criteria=success_criteria,
            rollback_plan=rollback_plan,
            created_at=datetime.now().isoformat()
        )
    
    def _assess_project_complexity(self) -> str:
        """Assess project complexity through questions."""
        print("\n📊 Project Complexity Assessment")
        print("-" * 30)
        
        score = 0
        
        # Number of agents
        try:
            num_agents = int(input("Number of agents in your system (1-10): "))
            score += min(num_agents, 5)
        except ValueError:
            score += 2
        
        # Custom components
        has_custom = input("Do you have custom chains or agents? (y/n): ").lower().startswith('y')
        if has_custom:
            score += 3
        
        # Memory complexity
        memory_complex = input("Complex memory management? (y/n): ").lower().startswith('y')
        if memory_complex:
            score += 2
        
        # Callbacks
        has_callbacks = input("Custom callbacks or monitoring? (y/n): ").lower().startswith('y')
        if has_callbacks:
            score += 2
        
        # Async usage
        has_async = input("Async/await patterns used? (y/n): ").lower().startswith('y')
        if has_async:
            score += 1
        
        if score <= 3:
            complexity = "low"
        elif score <= 7:
            complexity = "medium"
        else:
            complexity = "high"
        
        print(f"\n🎯 Assessed complexity: {complexity.upper()} (score: {score})")
        return complexity
    
    def _select_migration_pattern(self, complexity: str) -> str:
        """Select migration pattern based on complexity."""
        print(f"\n🎨 Migration Pattern Selection")
        print("-" * 30)
        
        if complexity == "low":
            recommended = "simple_agent"
        elif complexity == "medium":
            recommended = "complex_agent"
        else:
            recommended = "multi_agent_system"
        
        print("Available patterns:")
        for i, (pattern, _) in enumerate(self.common_patterns.items(), 1):
            marker = "👉" if pattern == recommended else "  "
            print(f"{marker} {i}. {pattern.replace('_', ' ').title()}")
        
        print(f"\nRecommended: {recommended.replace('_', ' ').title()}")
        
        choice = input("\nSelect pattern (1-3) or press Enter for recommended: ").strip()
        
        if choice in ['1', '2', '3']:
            patterns = list(self.common_patterns.keys())
            return patterns[int(choice) - 1]
        
        return recommended
    
    def _customize_tasks(self, pattern: str) -> List[MigrationTask]:
        """Customize tasks for the selected pattern."""
        task_ids = self.common_patterns[pattern]
        tasks = []
        
        print(f"\n🛠️  Customizing Tasks for {pattern.replace('_', ' ').title()}")
        print("-" * 50)
        
        for i, task_id in enumerate(task_ids, 1):
            template = self.task_templates[task_id]
            
            print(f"\n{i}. {template['title']}")
            print(f"   Description: {template['description']}")
            print(f"   Estimated: {template['estimated_hours']} hours")
            
            # Allow customization
            custom_hours = input(f"   Custom hours (or Enter for {template['estimated_hours']}): ").strip()
            if custom_hours:
                try:
                    estimated_hours = float(custom_hours)
                except ValueError:
                    estimated_hours = template['estimated_hours']
            else:
                estimated_hours = template['estimated_hours']
            
            # Create task
            task = MigrationTask(
                id=task_id,
                title=template['title'],
                description=template['description'],
                category=template['category'],
                complexity=template['complexity'],
                estimated_hours=estimated_hours,
                dependencies=self._get_dependencies(task_id, task_ids),
                priority=self._determine_priority(template['category']),
                file_paths=[],  # To be filled in later
                checklist=template['checklist'],
                resources=template['resources']
            )
            
            tasks.append(task)
        
        return tasks
    
    def _get_dependencies(self, task_id: str, all_task_ids: List[str]) -> List[str]:
        """Get task dependencies."""
        dependencies_map = {
            "convert_tools": ["setup_environment"],
            "design_state_schema": ["analyze_agent_patterns"],
            "create_graph_structure": ["design_state_schema", "convert_tools"],
            "implement_checkpointing": ["create_graph_structure"],
            "migrate_memory": ["design_state_schema"],
            "testing_validation": ["create_graph_structure"],
            "performance_optimization": ["testing_validation"],
            "documentation": ["testing_validation"]
        }
        
        deps = dependencies_map.get(task_id, [])
        return [dep for dep in deps if dep in all_task_ids]
    
    def _determine_priority(self, category: str) -> str:
        """Determine task priority based on category."""
        priority_map = {
            "setup": "high",
            "analysis": "high", 
            "implementation": "medium",
            "testing": "high",
            "optimization": "low",
            "documentation": "medium"
        }
        return priority_map.get(category, "medium")
    
    def _organize_phases(self, tasks: List[MigrationTask]) -> Dict[str, List[MigrationTask]]:
        """Organize tasks into logical phases."""
        phases = {
            "Phase 1: Setup & Analysis": [],
            "Phase 2: Core Implementation": [],
            "Phase 3: Testing & Optimization": [],
            "Phase 4: Documentation & Deployment": []
        }
        
        for task in tasks:
            if task.category in ["setup", "analysis"]:
                phases["Phase 1: Setup & Analysis"].append(task)
            elif task.category == "implementation":
                phases["Phase 2: Core Implementation"].append(task)
            elif task.category in ["testing", "optimization"]:
                phases["Phase 3: Testing & Optimization"].append(task)
            else:
                phases["Phase 4: Documentation & Deployment"].append(task)
        
        return phases
    
    def _calculate_timeline(self, total_hours: float) -> int:
        """Calculate timeline in weeks."""
        hours_per_week = float(input(f"\nHours per week for migration (default 10): ") or "10")
        weeks = max(1, int(total_hours / hours_per_week))
        return weeks
    
    def _identify_risks(self, complexity: str, tasks: List[MigrationTask]) -> List[str]:
        """Identify project risks."""
        risks = []
        
        if complexity == "high":
            risks.extend([
                "Complex custom components may require significant refactoring",
                "Extended timeline may impact other project priorities",
                "Team learning curve for LangGraph concepts"
            ])
        
        if any(task.id == "migrate_memory" for task in tasks):
            risks.append("Memory migration may require data structure changes")
        
        if any(task.id == "implement_checkpointing" for task in tasks):
            risks.append("State persistence changes may affect system behavior")
        
        risks.extend([
            "Breaking changes in dependencies",
            "Performance regressions during migration",
            "User experience disruption during deployment"
        ])
        
        return risks
    
    def _create_mitigation_strategies(self, risks: List[str]) -> List[str]:
        """Create risk mitigation strategies."""
        return [
            "Create comprehensive backup before starting migration",
            "Implement feature flags for gradual rollout",
            "Set up monitoring and rollback procedures",
            "Conduct thorough testing at each phase",
            "Maintain parallel systems during transition",
            "Document all changes and decision rationale",
            "Plan team training sessions for new concepts",
            "Establish clear rollback criteria and procedures"
        ]
    
    def _define_success_criteria(self) -> List[str]:
        """Define migration success criteria."""
        return [
            "All existing functionality preserved",
            "Performance equal or better than original",
            "State persistence working correctly",
            "All tests passing",
            "Documentation updated and complete",
            "Team comfortable with new architecture",
            "Monitoring and logging operational",
            "Zero data loss during migration"
        ]
    
    def _create_rollback_plan(self) -> List[str]:
        """Create rollback plan."""
        return [
            "Maintain original LangChain codebase as backup",
            "Use feature flags to quickly switch implementations",
            "Monitor key performance metrics after deployment",
            "Have rollback scripts ready for database changes",
            "Communicate rollback procedures to team",
            "Test rollback process in staging environment",
            "Define clear rollback trigger conditions",
            "Plan communication strategy for stakeholders"
        ]
    
    def export_plan(self, plan: MigrationPlan, format: str = "json", output_file: str = None) -> str:
        """Export migration plan to file."""
        if format == "json":
            content = json.dumps(asdict(plan), indent=2)
            extension = ".json"
        elif format == "markdown":
            content = self._plan_to_markdown(plan)
            extension = ".md"
        else:
            raise ValueError("Unsupported format. Use 'json' or 'markdown'.")
        
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"migration_plan_{timestamp}{extension}"
        
        with open(output_file, 'w') as f:
            f.write(content)
        
        return output_file
    
    def _plan_to_markdown(self, plan: MigrationPlan) -> str:
        """Convert plan to markdown format."""
        lines = []
        
        # Header
        lines.extend([
            f"# {plan.project_name} - Migration Plan",
            "",
            f"**Created:** {plan.created_at}",
            f"**Estimated Time:** {plan.total_estimated_hours:.1f} hours",
            f"**Timeline:** {plan.timeline_weeks} weeks",
            ""
        ])
        
        # Overview
        lines.extend([
            "## 📊 Overview",
            "",
            f"This document outlines the migration plan for converting {plan.project_name} from LangChain to LangGraph.",
            f"The migration is estimated to take {plan.total_estimated_hours:.1f} hours over {plan.timeline_weeks} weeks.",
            ""
        ])
        
        # Phases
        lines.extend([
            "## 🗓️ Migration Phases",
            ""
        ])
        
        for phase_name, phase_tasks in plan.phases.items():
            lines.append(f"### {phase_name}")
            lines.append("")
            
            for task in phase_tasks:
                lines.extend([
                    f"#### {task.title}",
                    f"**Estimated Time:** {task.estimated_hours} hours",
                    f"**Priority:** {task.priority.title()}",
                    f"**Complexity:** {task.complexity}/5",
                    "",
                    f"{task.description}",
                    "",
                    "**Checklist:**"
                ])
                
                for item in task.checklist:
                    lines.append(f"- [ ] {item}")
                
                lines.append("")
                
                if task.dependencies:
                    lines.append(f"**Dependencies:** {', '.join(task.dependencies)}")
                    lines.append("")
                
                if task.resources:
                    lines.append("**Resources:**")
                    for resource in task.resources:
                        lines.append(f"- {resource}")
                    lines.append("")
        
        # Risk factors
        lines.extend([
            "## ⚠️ Risk Factors",
            ""
        ])
        for risk in plan.risk_factors:
            lines.append(f"- {risk}")
        lines.append("")
        
        # Mitigation strategies
        lines.extend([
            "## 🛡️ Mitigation Strategies",
            ""
        ])
        for strategy in plan.mitigation_strategies:
            lines.append(f"- {strategy}")
        lines.append("")
        
        # Success criteria
        lines.extend([
            "## ✅ Success Criteria",
            ""
        ])
        for criteria in plan.success_criteria:
            lines.append(f"- {criteria}")
        lines.append("")
        
        # Rollback plan
        lines.extend([
            "## 🔄 Rollback Plan",
            ""
        ])
        for step in plan.rollback_plan:
            lines.append(f"- {step}")
        lines.append("")
        
        return "\n".join(lines)

def main():
    """Main CLI interface."""
    parser = argparse.ArgumentParser(description="Create LangChain to LangGraph migration plan")
    parser.add_argument("--output", "-o", help="Output file path")
    parser.add_argument("--format", "-f", choices=["json", "markdown"], default="markdown",
                       help="Output format (default: markdown)")
    parser.add_argument("--interactive", "-i", action="store_true", default=True,
                       help="Interactive mode (default)")
    
    args = parser.parse_args()
    
    # Create planner
    planner = MigrationPlannerTool()
    
    # Create plan
    if args.interactive:
        plan = planner.create_interactive_plan()
    else:
        print("Non-interactive mode not yet implemented")
        sys.exit(1)
    
    # Export plan
    output_file = planner.export_plan(plan, args.format, args.output)
    
    print(f"\n✅ Migration plan created successfully!")
    print(f"📄 Plan saved to: {output_file}")
    print(f"⏱️  Total estimated time: {plan.total_estimated_hours:.1f} hours")
    print(f"📅 Estimated timeline: {plan.timeline_weeks} weeks")
    
    # Show summary
    total_tasks = sum(len(tasks) for tasks in plan.phases.values())
    print(f"📝 Total tasks: {total_tasks}")
    print(f"🚨 Risk factors identified: {len(plan.risk_factors)}")
    
    print("\n🎯 Next steps:")
    print("1. Review the generated plan with your team")
    print("2. Adjust timelines and priorities as needed")
    print("3. Set up project tracking (GitHub issues, Jira, etc.)")
    print("4. Begin with Phase 1 tasks")
    print("5. Schedule regular progress reviews")

if __name__ == "__main__":
    main()