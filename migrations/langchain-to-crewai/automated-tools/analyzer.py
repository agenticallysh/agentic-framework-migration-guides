#!/usr/bin/env python3
"""
LangChain Project Analyzer
==========================

Analyzes LangChain projects to assess migration complexity and provide recommendations.
Helps developers understand what needs to be migrated and estimate effort required.
"""

import os
import ast
import argparse
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from collections import defaultdict

@dataclass
class AnalysisResult:
    """Results of project analysis"""
    agents_count: int = 0
    tools_count: int = 0
    chains_count: int = 0
    memory_systems: int = 0
    custom_chains: int = 0
    langchain_imports: List[str] = None
    complexity_score: float = 0.0
    estimated_hours: float = 0.0
    compatibility_percentage: float = 0.0
    recommendations: List[str] = None
    migration_blockers: List[str] = None
    
    def __post_init__(self):
        if self.langchain_imports is None:
            self.langchain_imports = []
        if self.recommendations is None:
            self.recommendations = []
        if self.migration_blockers is None:
            self.migration_blockers = []

class LangChainAnalyzer:
    """Analyzes LangChain projects for migration assessment"""
    
    def __init__(self):
        self.result = AnalysisResult()
        self.file_contents = {}
        
    def analyze_project(self, project_path: str) -> AnalysisResult:
        """Analyze a LangChain project directory"""
        
        print(f"🔍 Analyzing project: {project_path}")
        
        # Find Python files
        python_files = self._find_python_files(project_path)
        print(f"📁 Found {len(python_files)} Python files")
        
        # Analyze each file
        for file_path in python_files:
            self._analyze_file(file_path)
        
        # Calculate complexity and estimates
        self._calculate_complexity()
        self._generate_recommendations()
        
        return self.result
    
    def _find_python_files(self, project_path: str) -> List[Path]:
        """Find all Python files in the project"""
        
        project_dir = Path(project_path)
        python_files = []
        
        for file_path in project_dir.rglob("*.py"):
            # Skip virtual environments and common irrelevant directories
            if any(part in str(file_path) for part in [
                "venv", "env", ".venv", "__pycache__", 
                "node_modules", ".git", "dist", "build"
            ]):
                continue
            python_files.append(file_path)
        
        return python_files
    
    def _analyze_file(self, file_path: Path) -> None:
        """Analyze a single Python file for LangChain usage"""
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                self.file_contents[str(file_path)] = content
            
            # Parse AST
            tree = ast.parse(content)
            
            # Analyze imports
            self._analyze_imports(tree)
            
            # Analyze usage patterns
            self._analyze_usage_patterns(tree, content)
            
        except Exception as e:
            print(f"⚠️  Warning: Could not analyze {file_path}: {e}")
    
    def _analyze_imports(self, tree: ast.AST) -> None:
        """Analyze LangChain imports"""
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if 'langchain' in alias.name:
                        self.result.langchain_imports.append(alias.name)
            
            elif isinstance(node, ast.ImportFrom):
                if node.module and 'langchain' in node.module:
                    for alias in node.names:
                        import_name = f"{node.module}.{alias.name}"
                        self.result.langchain_imports.append(import_name)
    
    def _analyze_usage_patterns(self, tree: ast.AST, content: str) -> None:
        """Analyze LangChain usage patterns in the code"""
        
        # Count agents
        agent_patterns = [
            'create_openai_tools_agent', 'create_openai_functions_agent',
            'create_react_agent', 'AgentExecutor', 'Agent('
        ]
        for pattern in agent_patterns:
            self.result.agents_count += content.count(pattern)
        
        # Count tools
        tool_patterns = [
            'Tool(', '@tool', 'BaseTool', 'StructuredTool'
        ]
        for pattern in tool_patterns:
            self.result.tools_count += content.count(pattern)
        
        # Count chains
        chain_patterns = [
            'LLMChain', 'SequentialChain', 'SimpleSequentialChain',
            'ConversationChain', 'RetrievalQA', 'ConversationalRetrievalQA'
        ]
        for pattern in chain_patterns:
            self.result.chains_count += content.count(pattern)
        
        # Count memory systems
        memory_patterns = [
            'ConversationBufferMemory', 'ConversationSummaryMemory',
            'ConversationBufferWindowMemory', 'ConversationSummaryBufferMemory'
        ]
        for pattern in memory_patterns:
            self.result.memory_systems += content.count(pattern)
        
        # Detect custom chains (complex migration)
        if 'class' in content and ('Chain' in content or 'BaseChain' in content):
            self.result.custom_chains += 1
    
    def _calculate_complexity(self) -> None:
        """Calculate migration complexity score and time estimate"""
        
        # Base complexity factors
        complexity_factors = {
            'agents': self.result.agents_count * 1.0,
            'tools': self.result.tools_count * 0.5,
            'chains': self.result.chains_count * 1.5,
            'memory': self.result.memory_systems * 1.0,
            'custom_chains': self.result.custom_chains * 3.0,  # Much more complex
        }
        
        # Calculate total complexity score (0-100)
        raw_score = sum(complexity_factors.values())
        self.result.complexity_score = min(raw_score * 10, 100.0)
        
        # Calculate estimated hours
        base_hours = 2.0  # Minimum setup time
        agent_hours = self.result.agents_count * 0.5
        tool_hours = self.result.tools_count * 0.25
        chain_hours = self.result.chains_count * 0.75
        memory_hours = self.result.memory_systems * 0.5
        custom_hours = self.result.custom_chains * 2.0
        
        self.result.estimated_hours = (
            base_hours + agent_hours + tool_hours + 
            chain_hours + memory_hours + custom_hours
        )
        
        # Calculate compatibility percentage
        total_components = (
            self.result.agents_count + self.result.tools_count + 
            self.result.chains_count + self.result.memory_systems
        )
        
        if total_components == 0:
            self.result.compatibility_percentage = 100.0
        else:
            # Custom chains reduce compatibility significantly
            compatibility_reduction = self.result.custom_chains * 15
            base_compatibility = 95.0  # CrewAI is highly compatible
            self.result.compatibility_percentage = max(
                base_compatibility - compatibility_reduction, 60.0
            )
    
    def _generate_recommendations(self) -> None:
        """Generate migration recommendations based on analysis"""
        
        # General recommendations
        if self.result.agents_count > 0:
            self.result.recommendations.append(
                f"✅ {self.result.agents_count} agents found - Good fit for CrewAI's role-based architecture"
            )
        
        if self.result.tools_count > 0:
            self.result.recommendations.append(
                f"🛠️ {self.result.tools_count} tools found - Most will migrate directly to CrewAI"
            )
        
        if self.result.chains_count > 0:
            self.result.recommendations.append(
                f"🔗 {self.result.chains_count} chains found - Convert to CrewAI Task sequences"
            )
        
        if self.result.memory_systems > 0:
            self.result.recommendations.append(
                f"🧠 {self.result.memory_systems} memory systems - CrewAI has built-in memory management"
            )
        
        # Complexity-based recommendations
        if self.result.complexity_score < 30:
            self.result.recommendations.append("💚 Low complexity - Straightforward migration expected")
        elif self.result.complexity_score < 60:
            self.result.recommendations.append("💛 Medium complexity - Plan for testing and validation")
        else:
            self.result.recommendations.append("🧡 High complexity - Consider incremental migration approach")
        
        # Migration blockers
        if self.result.custom_chains > 0:
            self.result.migration_blockers.append(
                f"⚠️ {self.result.custom_chains} custom chains require manual migration"
            )
        
        # Check for problematic imports
        problematic_imports = [
            'langchain.experimental', 'langchain.llms.llamacpp',
            'langchain.vectorstores.redis', 'langchain.schema.runnable'
        ]
        
        for imp in self.result.langchain_imports:
            for problematic in problematic_imports:
                if problematic in imp:
                    self.result.migration_blockers.append(
                        f"⚠️ Potentially problematic import: {imp}"
                    )
                    break
    
    def generate_report(self) -> str:
        """Generate a human-readable analysis report"""
        
        complexity_emoji = {
            "LOW": "💚",
            "MEDIUM": "💛", 
            "HIGH": "🧡"
        }
        
        if self.result.complexity_score < 30:
            complexity_level = "LOW"
        elif self.result.complexity_score < 60:
            complexity_level = "MEDIUM"
        else:
            complexity_level = "HIGH"
        
        report = f"""
📊 LangChain Project Analysis Report
====================================

🎯 Migration Assessment:
  • Agents found: {self.result.agents_count}
  • Tools found: {self.result.tools_count}
  • Chains found: {self.result.chains_count}
  • Memory systems: {self.result.memory_systems}
  • Custom chains: {self.result.custom_chains}

📈 Migration Metrics:
  • Complexity score: {self.result.complexity_score:.1f}/100
  • Complexity level: {complexity_emoji[complexity_level]} {complexity_level}
  • Estimated time: {self.result.estimated_hours:.1f} hours
  • CrewAI compatibility: {self.result.compatibility_percentage:.1f}%

💡 Recommendations:
"""
        
        for rec in self.result.recommendations:
            report += f"  • {rec}\n"
        
        if self.result.migration_blockers:
            report += "\n⚠️ Migration Blockers:\n"
            for blocker in self.result.migration_blockers:
                report += f"  • {blocker}\n"
        
        report += f"""
🚀 Next Steps:
  1. Review the migration guide: README.md
  2. Start with basic agents and tools
  3. Migrate chains to CrewAI Tasks
  4. Test thoroughly with existing test cases
  5. Optimize using CrewAI-specific features

📚 Resources:
  • Migration guide: https://www.agentically.sh/ai-agentic-frameworks/migrate/langchain-to-crewai/
  • CrewAI documentation: https://docs.crewai.com/
  • Support community: https://discord.gg/agentically
"""
        
        return report

def main():
    """Main CLI function"""
    
    parser = argparse.ArgumentParser(
        description="Analyze LangChain projects for CrewAI migration"
    )
    parser.add_argument(
        "--project-path", "-p",
        required=True,
        help="Path to the LangChain project directory"
    )
    parser.add_argument(
        "--output", "-o",
        help="Output file for JSON results (optional)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )
    
    args = parser.parse_args()
    
    # Validate project path
    if not os.path.exists(args.project_path):
        print(f"❌ Error: Project path '{args.project_path}' does not exist")
        return
    
    # Run analysis
    analyzer = LangChainAnalyzer()
    result = analyzer.analyze_project(args.project_path)
    
    # Generate and display report
    report = analyzer.generate_report()
    print(report)
    
    # Save JSON output if requested
    if args.output:
        json_data = {
            "agents_count": result.agents_count,
            "tools_count": result.tools_count,
            "chains_count": result.chains_count,
            "memory_systems": result.memory_systems,
            "custom_chains": result.custom_chains,
            "complexity_score": result.complexity_score,
            "estimated_hours": result.estimated_hours,
            "compatibility_percentage": result.compatibility_percentage,
            "langchain_imports": result.langchain_imports,
            "recommendations": result.recommendations,
            "migration_blockers": result.migration_blockers
        }
        
        with open(args.output, 'w') as f:
            json.dump(json_data, f, indent=2)
        
        print(f"\n💾 Results saved to: {args.output}")

if __name__ == "__main__":
    main()

"""
Usage Examples:
===============

# Analyze a project
python analyzer.py --project-path ./my-langchain-project

# Analyze with JSON output
python analyzer.py --project-path ./my-project --output analysis.json

# Verbose analysis
python analyzer.py --project-path ./my-project --verbose

This tool helps developers understand:
1. What components need migration
2. Expected time investment
3. Potential migration challenges
4. Compatibility with CrewAI
5. Step-by-step recommendations
"""