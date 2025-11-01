#!/usr/bin/env python3
"""
LangChain to LangGraph Migration Analyzer
Automated tool to analyze LangChain codebases and assess migration complexity.
"""

import os
import ast
import json
import re
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass, asdict
from pathlib import Path
import argparse

@dataclass
class MigrationAssessment:
    """Assessment result for LangChain to LangGraph migration."""
    file_path: str
    complexity_score: int  # 1-10 scale
    migration_effort: str  # low/medium/high
    langchain_components: List[str]
    migration_blockers: List[str]
    migration_recommendations: List[str]
    code_patterns: Dict[str, int]
    estimated_hours: float

@dataclass
class ProjectAssessment:
    """Overall project migration assessment."""
    total_files: int
    python_files: int
    langchain_files: int
    overall_complexity: int
    total_estimated_hours: float
    migration_priority: str
    blockers_summary: List[str]
    recommendations_summary: List[str]
    file_assessments: List[MigrationAssessment]

class LangChainCodeAnalyzer:
    """Analyzes LangChain code for migration assessment."""
    
    def __init__(self):
        # LangChain imports to look for
        self.langchain_imports = {
            'langchain.agents': {'complexity': 3, 'category': 'agents'},
            'langchain.chains': {'complexity': 2, 'category': 'chains'},
            'langchain.memory': {'complexity': 2, 'category': 'memory'},
            'langchain.tools': {'complexity': 1, 'category': 'tools'},
            'langchain.schema': {'complexity': 1, 'category': 'schema'},
            'langchain.callbacks': {'complexity': 2, 'category': 'callbacks'},
            'langchain.chat_models': {'complexity': 1, 'category': 'models'},
            'langchain.llms': {'complexity': 1, 'category': 'models'},
            'langchain.prompts': {'complexity': 1, 'category': 'prompts'},
            'langchain.output_parsers': {'complexity': 2, 'category': 'parsers'},
            'langchain.document_loaders': {'complexity': 1, 'category': 'loaders'},
            'langchain.vectorstores': {'complexity': 1, 'category': 'vectorstores'},
            'langchain.embeddings': {'complexity': 1, 'category': 'embeddings'},
            'langchain.text_splitter': {'complexity': 1, 'category': 'text_processing'},
            'langchain.retrievers': {'complexity': 1, 'category': 'retrievers'},
        }
        
        # LangChain classes that indicate complexity
        self.complex_classes = {
            'AgentExecutor': 4,
            'ConversationalRetrievalChain': 3,
            'RetrievalQAWithSourcesChain': 3,
            'LLMChain': 2,
            'SequentialChain': 2,
            'ConversationBufferMemory': 2,
            'ConversationBufferWindowMemory': 2,
            'VectorStoreRetrieverMemory': 3,
        }
        
        # Migration blockers
        self.migration_blockers = [
            'custom_chain_classes',
            'complex_memory_patterns',
            'custom_callback_handlers',
            'heavy_langchain_ecosystem_usage',
            'custom_output_parsers',
            'complex_agent_architectures'
        ]
    
    def analyze_file(self, file_path: str) -> Optional[MigrationAssessment]:
        """Analyze a single Python file for LangChain usage."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Parse AST
            tree = ast.parse(content)
            
            # Check if file uses LangChain
            if not self._contains_langchain(content):
                return None
            
            # Analyze the code
            analysis = self._analyze_ast(tree, content)
            
            # Calculate complexity and effort
            complexity_score = self._calculate_complexity(analysis)
            migration_effort = self._determine_effort(complexity_score)
            estimated_hours = self._estimate_hours(complexity_score, analysis)
            
            # Get migration recommendations
            recommendations = self._get_recommendations(analysis)
            blockers = self._identify_blockers(analysis)
            
            return MigrationAssessment(
                file_path=file_path,
                complexity_score=complexity_score,
                migration_effort=migration_effort,
                langchain_components=analysis['components'],
                migration_blockers=blockers,
                migration_recommendations=recommendations,
                code_patterns=analysis['patterns'],
                estimated_hours=estimated_hours
            )
            
        except Exception as e:
            print(f"Error analyzing {file_path}: {str(e)}")
            return None
    
    def _contains_langchain(self, content: str) -> bool:
        """Check if file contains LangChain imports."""
        return any(import_name in content for import_name in self.langchain_imports.keys()) or \
               'langchain' in content
    
    def _analyze_ast(self, tree: ast.AST, content: str) -> Dict[str, Any]:
        """Analyze AST for LangChain patterns."""
        analysis = {
            'imports': [],
            'components': [],
            'patterns': {},
            'classes': [],
            'functions': [],
            'complexity_indicators': []
        }
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if 'langchain' in alias.name:
                        analysis['imports'].append(alias.name)
                        analysis['components'].append(alias.name)
            
            elif isinstance(node, ast.ImportFrom):
                if node.module and 'langchain' in node.module:
                    for alias in node.names:
                        component = f"{node.module}.{alias.name}"
                        analysis['imports'].append(component)
                        analysis['components'].append(alias.name)
            
            elif isinstance(node, ast.ClassDef):
                analysis['classes'].append(node.name)
                # Check for custom chain classes
                for base in node.bases:
                    if isinstance(base, ast.Name) and 'Chain' in base.id:
                        analysis['complexity_indicators'].append('custom_chain_class')
            
            elif isinstance(node, ast.FunctionDef):
                analysis['functions'].append(node.name)
        
        # Analyze code patterns
        analysis['patterns'] = self._analyze_patterns(content)
        
        return analysis
    
    def _analyze_patterns(self, content: str) -> Dict[str, int]:
        """Analyze code patterns that affect migration complexity."""
        patterns = {
            'agent_executors': len(re.findall(r'AgentExecutor', content)),
            'custom_tools': len(re.findall(r'Tool\(', content)),
            'memory_usage': len(re.findall(r'Memory', content)),
            'callbacks': len(re.findall(r'callback', content, re.IGNORECASE)),
            'chains': len(re.findall(r'Chain', content)),
            'async_usage': len(re.findall(r'async def|await', content)),
            'error_handling': len(re.findall(r'try:|except:', content)),
            'custom_classes': len(re.findall(r'class.*Chain|class.*Agent', content)),
        }
        
        return patterns
    
    def _calculate_complexity(self, analysis: Dict[str, Any]) -> int:
        """Calculate migration complexity score (1-10)."""
        score = 1
        
        # Base complexity from imports
        for component in analysis['components']:
            for import_pattern, info in self.langchain_imports.items():
                if import_pattern in component:
                    score += info['complexity']
        
        # Additional complexity from patterns
        patterns = analysis['patterns']
        score += patterns.get('agent_executors', 0) * 2
        score += patterns.get('custom_tools', 0) * 1
        score += patterns.get('memory_usage', 0) * 1
        score += patterns.get('callbacks', 0) * 1
        score += patterns.get('chains', 0) * 1
        score += patterns.get('custom_classes', 0) * 3
        
        # Complexity indicators
        score += len(analysis['complexity_indicators']) * 2
        
        return min(score, 10)  # Cap at 10
    
    def _determine_effort(self, complexity_score: int) -> str:
        """Determine migration effort level."""
        if complexity_score <= 3:
            return "low"
        elif complexity_score <= 6:
            return "medium"
        else:
            return "high"
    
    def _estimate_hours(self, complexity_score: int, analysis: Dict[str, Any]) -> float:
        """Estimate migration time in hours."""
        base_hours = complexity_score * 0.5
        
        # Additional time for specific patterns
        patterns = analysis['patterns']
        base_hours += patterns.get('agent_executors', 0) * 2
        base_hours += patterns.get('custom_classes', 0) * 4
        base_hours += patterns.get('callbacks', 0) * 1
        base_hours += patterns.get('memory_usage', 0) * 1
        
        # Minimum 0.5 hours, maximum 20 hours per file
        return max(0.5, min(base_hours, 20.0))
    
    def _get_recommendations(self, analysis: Dict[str, Any]) -> List[str]:
        """Get migration recommendations based on analysis."""
        recommendations = []
        patterns = analysis['patterns']
        
        if patterns.get('agent_executors', 0) > 0:
            recommendations.append("Convert AgentExecutor to StateGraph with conditional edges")
        
        if patterns.get('memory_usage', 0) > 0:
            recommendations.append("Replace memory objects with typed state management")
        
        if patterns.get('custom_tools', 0) > 0:
            recommendations.append("Update tool definitions to use @tool decorator")
        
        if patterns.get('chains', 0) > 0:
            recommendations.append("Redesign chains as graph nodes with explicit state flow")
        
        if patterns.get('callbacks', 0) > 0:
            recommendations.append("Implement callback functionality using graph events")
        
        if patterns.get('async_usage', 0) > 0:
            recommendations.append("Ensure async compatibility with LangGraph async methods")
        
        # Default recommendations
        recommendations.extend([
            "Add proper type hints for state management",
            "Implement checkpointing for state persistence",
            "Design graph structure with clear node responsibilities",
            "Add comprehensive error handling for each node"
        ])
        
        return recommendations
    
    def _identify_blockers(self, analysis: Dict[str, Any]) -> List[str]:
        """Identify potential migration blockers."""
        blockers = []
        patterns = analysis['patterns']
        
        if patterns.get('custom_classes', 0) > 2:
            blockers.append("Heavy use of custom Chain/Agent classes requires significant refactoring")
        
        if patterns.get('callbacks', 0) > 5:
            blockers.append("Extensive callback usage may need custom implementation")
        
        if 'custom_chain_class' in analysis['complexity_indicators']:
            blockers.append("Custom chain inheritance patterns need complete redesign")
        
        # Check for complex memory patterns
        memory_components = [c for c in analysis['components'] if 'memory' in c.lower()]
        if len(memory_components) > 2:
            blockers.append("Complex memory management patterns require state redesign")
        
        return blockers

class ProjectMigrationAnalyzer:
    """Analyzes entire projects for LangChain to LangGraph migration."""
    
    def __init__(self):
        self.file_analyzer = LangChainCodeAnalyzer()
    
    def analyze_project(self, project_path: str, exclude_dirs: List[str] = None) -> ProjectAssessment:
        """Analyze entire project for migration assessment."""
        if exclude_dirs is None:
            exclude_dirs = ['__pycache__', '.git', 'node_modules', 'venv', '.env']
        
        print(f"🔍 Analyzing project: {project_path}")
        
        # Find all Python files
        python_files = self._find_python_files(project_path, exclude_dirs)
        total_files = len(list(Path(project_path).rglob('*'))) if os.path.exists(project_path) else 0
        
        print(f"📁 Found {len(python_files)} Python files in project")
        
        # Analyze each file
        file_assessments = []
        for file_path in python_files:
            assessment = self.file_analyzer.analyze_file(file_path)
            if assessment:  # Only include files that use LangChain
                file_assessments.append(assessment)
                print(f"✅ Analyzed: {os.path.relpath(file_path, project_path)}")
        
        print(f"🎯 Found {len(file_assessments)} files using LangChain")
        
        # Calculate overall assessment
        overall_assessment = self._calculate_overall_assessment(
            total_files, len(python_files), file_assessments
        )
        
        return ProjectAssessment(
            total_files=total_files,
            python_files=len(python_files),
            langchain_files=len(file_assessments),
            overall_complexity=overall_assessment['complexity'],
            total_estimated_hours=overall_assessment['hours'],
            migration_priority=overall_assessment['priority'],
            blockers_summary=overall_assessment['blockers'],
            recommendations_summary=overall_assessment['recommendations'],
            file_assessments=file_assessments
        )
    
    def _find_python_files(self, project_path: str, exclude_dirs: List[str]) -> List[str]:
        """Find all Python files in project."""
        python_files = []
        
        for root, dirs, files in os.walk(project_path):
            # Remove excluded directories
            dirs[:] = [d for d in dirs if d not in exclude_dirs]
            
            for file in files:
                if file.endswith('.py'):
                    python_files.append(os.path.join(root, file))
        
        return python_files
    
    def _calculate_overall_assessment(self, total_files: int, python_files: int, 
                                    assessments: List[MigrationAssessment]) -> Dict[str, Any]:
        """Calculate overall project assessment."""
        if not assessments:
            return {
                'complexity': 1,
                'hours': 0,
                'priority': 'low',
                'blockers': [],
                'recommendations': ['No LangChain usage detected']
            }
        
        # Calculate averages and totals
        avg_complexity = sum(a.complexity_score for a in assessments) / len(assessments)
        total_hours = sum(a.estimated_hours for a in assessments)
        
        # Collect unique blockers and recommendations
        all_blockers = set()
        all_recommendations = set()
        
        for assessment in assessments:
            all_blockers.update(assessment.migration_blockers)
            all_recommendations.update(assessment.migration_recommendations)
        
        # Determine priority
        if avg_complexity >= 7 or total_hours > 40:
            priority = 'high'
        elif avg_complexity >= 4 or total_hours > 15:
            priority = 'medium'
        else:
            priority = 'low'
        
        return {
            'complexity': int(avg_complexity),
            'hours': total_hours,
            'priority': priority,
            'blockers': list(all_blockers),
            'recommendations': list(all_recommendations)
        }
    
    def generate_report(self, assessment: ProjectAssessment, output_file: str = None) -> str:
        """Generate a detailed migration report."""
        report = []
        
        # Header
        report.append("# LangChain to LangGraph Migration Assessment Report")
        report.append("=" * 60)
        report.append("")
        
        # Project Overview
        report.append("## 📊 Project Overview")
        report.append(f"- **Total Files**: {assessment.total_files}")
        report.append(f"- **Python Files**: {assessment.python_files}")
        report.append(f"- **LangChain Files**: {assessment.langchain_files}")
        report.append(f"- **Overall Complexity**: {assessment.overall_complexity}/10")
        report.append(f"- **Estimated Hours**: {assessment.total_estimated_hours:.1f}")
        report.append(f"- **Migration Priority**: {assessment.migration_priority.upper()}")
        report.append("")
        
        # Migration Blockers
        if assessment.blockers_summary:
            report.append("## 🚫 Migration Blockers")
            for blocker in assessment.blockers_summary:
                report.append(f"- {blocker}")
            report.append("")
        
        # Recommendations
        report.append("## 💡 Migration Recommendations")
        for rec in assessment.recommendations_summary[:10]:  # Top 10
            report.append(f"- {rec}")
        report.append("")
        
        # File-by-file breakdown
        report.append("## 📋 File-by-file Analysis")
        report.append("")
        
        # Sort files by complexity (highest first)
        sorted_files = sorted(assessment.file_assessments, 
                            key=lambda x: x.complexity_score, reverse=True)
        
        for i, file_assessment in enumerate(sorted_files[:20], 1):  # Top 20 files
            rel_path = os.path.relpath(file_assessment.file_path)
            effort = file_assessment.migration_effort.upper()
            hours = file_assessment.estimated_hours
            complexity = file_assessment.complexity_score
            
            report.append(f"### {i}. {rel_path}")
            report.append(f"- **Complexity**: {complexity}/10")
            report.append(f"- **Effort**: {effort}")
            report.append(f"- **Estimated Hours**: {hours:.1f}")
            report.append(f"- **Components**: {', '.join(file_assessment.langchain_components[:5])}")
            
            if file_assessment.migration_blockers:
                report.append(f"- **Blockers**: {'; '.join(file_assessment.migration_blockers)}")
            
            report.append("")
        
        # Summary recommendations
        report.append("## 🎯 Next Steps")
        report.append("1. **Start with low-complexity files** to build migration experience")
        report.append("2. **Address blockers** in high-complexity files first")
        report.append("3. **Create migration plan** with timeline based on estimated hours")
        report.append("4. **Set up testing strategy** to validate migrated components")
        report.append("5. **Consider gradual migration** rather than big-bang approach")
        report.append("")
        
        report_text = "\n".join(report)
        
        # Save to file if specified
        if output_file:
            with open(output_file, 'w') as f:
                f.write(report_text)
            print(f"📄 Report saved to: {output_file}")
        
        return report_text

def main():
    """Main CLI interface."""
    parser = argparse.ArgumentParser(description="Analyze LangChain project for LangGraph migration")
    parser.add_argument("project_path", help="Path to project directory")
    parser.add_argument("--output", "-o", help="Output file for report")
    parser.add_argument("--json", help="Output JSON assessment to file")
    parser.add_argument("--exclude", nargs="+", default=["__pycache__", ".git", "venv"], 
                       help="Directories to exclude from analysis")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.project_path):
        print(f"❌ Error: Path {args.project_path} does not exist")
        return
    
    # Run analysis
    analyzer = ProjectMigrationAnalyzer()
    assessment = analyzer.analyze_project(args.project_path, args.exclude)
    
    # Generate and display report
    report = analyzer.generate_report(assessment, args.output)
    
    if not args.output:
        print("\n" + report)
    
    # Save JSON if requested
    if args.json:
        with open(args.json, 'w') as f:
            json.dump(asdict(assessment), f, indent=2)
        print(f"📄 JSON assessment saved to: {args.json}")
    
    # Print summary
    print("\n" + "="*60)
    print("🎯 MIGRATION ASSESSMENT SUMMARY")
    print("="*60)
    print(f"📁 LangChain files found: {assessment.langchain_files}")
    print(f"⚡ Overall complexity: {assessment.overall_complexity}/10")
    print(f"⏱️  Estimated time: {assessment.total_estimated_hours:.1f} hours")
    print(f"🚨 Priority level: {assessment.migration_priority.upper()}")
    
    if assessment.langchain_files == 0:
        print("✅ No LangChain usage detected - no migration needed!")
    elif assessment.migration_priority == "low":
        print("✅ Low complexity migration - good candidate for LangGraph!")
    elif assessment.migration_priority == "medium":
        print("⚠️  Medium complexity - plan carefully and migrate incrementally")
    else:
        print("🚨 High complexity - consider if migration benefits justify effort")

if __name__ == "__main__":
    main()