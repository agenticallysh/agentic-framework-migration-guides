#!/usr/bin/env python3
"""
Flowise Flow Analyzer
Analyzes Flowise exported flows and provides migration guidance for LangChain conversion.
"""

import json
import re
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum

class NodeType(Enum):
    """Flowise node types and their complexity"""
    LLM = "llm"
    MEMORY = "memory"
    DOCUMENT_LOADER = "documentLoader"
    TEXT_SPLITTER = "textSplitter"
    VECTOR_STORE = "vectorStore"
    RETRIEVER = "retriever"
    CHAIN = "chain"
    TOOL = "tool"
    CUSTOM_FUNCTION = "customFunction"
    API_CALL = "apiCall"
    CONDITIONAL = "conditional"
    SEQUENTIAL = "sequential"
    UNKNOWN = "unknown"

@dataclass
class NodeAnalysis:
    """Analysis of a single Flowise node"""
    node_id: str
    node_type: NodeType
    node_name: str
    complexity_score: int  # 1-5 scale
    langchain_equivalent: str
    migration_notes: str
    dependencies: List[str]

@dataclass
class FlowAnalysis:
    """Complete analysis of a Flowise flow"""
    flow_name: str
    total_nodes: int
    complexity_score: int  # 1-10 scale
    estimated_hours: float
    migration_strategy: str
    required_chains: List[str]
    memory_type: str
    llm_model: str
    nodes: List[NodeAnalysis]
    dependencies: List[str]
    warnings: List[str]
    recommendations: List[str]

class FlowiseAnalyzer:
    """
    Analyzes Flowise flows and provides LangChain migration guidance
    """
    
    def __init__(self):
        self.node_mappings = self._initialize_node_mappings()
        self.complexity_weights = self._initialize_complexity_weights()
    
    def _initialize_node_mappings(self) -> Dict[str, Dict[str, str]]:
        """Map Flowise nodes to LangChain equivalents"""
        return {
            "chatOpenAI": {
                "langchain_class": "ChatOpenAI",
                "import": "from langchain_openai import ChatOpenAI",
                "complexity": "1"
            },
            "chatAnthropic": {
                "langchain_class": "ChatAnthropic", 
                "import": "from langchain_anthropic import ChatAnthropic",
                "complexity": "1"
            },
            "conversationBufferMemory": {
                "langchain_class": "ConversationBufferMemory",
                "import": "from langchain.memory import ConversationBufferMemory",
                "complexity": "2"
            },
            "conversationSummaryMemory": {
                "langchain_class": "ConversationSummaryMemory",
                "import": "from langchain.memory import ConversationSummaryMemory", 
                "complexity": "3"
            },
            "pdfLoader": {
                "langchain_class": "PyPDFLoader",
                "import": "from langchain.document_loaders import PyPDFLoader",
                "complexity": "2"
            },
            "textLoader": {
                "langchain_class": "TextLoader",
                "import": "from langchain.document_loaders import TextLoader",
                "complexity": "1"
            },
            "recursiveCharacterTextSplitter": {
                "langchain_class": "RecursiveCharacterTextSplitter",
                "import": "from langchain.text_splitter import RecursiveCharacterTextSplitter",
                "complexity": "2"
            },
            "chroma": {
                "langchain_class": "Chroma",
                "import": "from langchain.vectorstores import Chroma",
                "complexity": "3"
            },
            "pinecone": {
                "langchain_class": "Pinecone", 
                "import": "from langchain.vectorstores import Pinecone",
                "complexity": "4"
            },
            "conversationalRetrievalQAChain": {
                "langchain_class": "ConversationalRetrievalChain",
                "import": "from langchain.chains import ConversationalRetrievalChain",
                "complexity": "4"
            },
            "llmChain": {
                "langchain_class": "LLMChain",
                "import": "from langchain.chains import LLMChain", 
                "complexity": "2"
            },
            "sequentialChain": {
                "langchain_class": "SequentialChain",
                "import": "from langchain.chains import SequentialChain",
                "complexity": "4"
            },
            "customFunction": {
                "langchain_class": "Tool (custom implementation)",
                "import": "from langchain.tools import tool",
                "complexity": "5"
            }
        }
    
    def _initialize_complexity_weights(self) -> Dict[NodeType, int]:
        """Define complexity weights for different node types"""
        return {
            NodeType.LLM: 1,
            NodeType.MEMORY: 2, 
            NodeType.DOCUMENT_LOADER: 2,
            NodeType.TEXT_SPLITTER: 2,
            NodeType.VECTOR_STORE: 3,
            NodeType.RETRIEVER: 3,
            NodeType.CHAIN: 4,
            NodeType.TOOL: 3,
            NodeType.CUSTOM_FUNCTION: 5,
            NodeType.API_CALL: 4,
            NodeType.CONDITIONAL: 5,
            NodeType.SEQUENTIAL: 4,
            NodeType.UNKNOWN: 3
        }
    
    def load_flow(self, json_path: str) -> Dict[str, Any]:
        """Load Flowise flow from JSON export"""
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                flow_data = json.load(f)
            return flow_data
        except Exception as e:
            raise ValueError(f"Could not load flow: {str(e)}")
    
    def analyze_flow(self, flow_data: Dict[str, Any]) -> FlowAnalysis:
        """Analyze complete Flowise flow"""
        
        # Extract basic info
        flow_name = flow_data.get('name', 'Unnamed Flow')
        nodes = flow_data.get('nodes', [])
        edges = flow_data.get('edges', [])
        
        # Analyze individual nodes
        node_analyses = []
        total_complexity = 0
        required_imports = set()
        warnings = []
        recommendations = []
        
        for node in nodes:
            analysis = self._analyze_node(node)
            node_analyses.append(analysis)
            total_complexity += analysis.complexity_score
            
            # Collect dependencies
            mapping = self.node_mappings.get(node.get('data', {}).get('name', ''))
            if mapping:
                required_imports.add(mapping['import'])
        
        # Calculate overall complexity
        complexity_score = min(10, max(1, total_complexity // len(nodes) if nodes else 1))
        
        # Estimate migration time
        estimated_hours = self._estimate_migration_time(
            len(nodes), complexity_score, node_analyses
        )
        
        # Determine migration strategy
        migration_strategy = self._determine_strategy(complexity_score, node_analyses)
        
        # Extract LLM and memory info
        llm_model = self._extract_llm_model(nodes)
        memory_type = self._extract_memory_type(nodes)
        required_chains = self._extract_required_chains(nodes)
        
        # Generate warnings and recommendations
        warnings = self._generate_warnings(node_analyses)
        recommendations = self._generate_recommendations(node_analyses, complexity_score)
        
        return FlowAnalysis(
            flow_name=flow_name,
            total_nodes=len(nodes),
            complexity_score=complexity_score,
            estimated_hours=estimated_hours,
            migration_strategy=migration_strategy,
            required_chains=required_chains,
            memory_type=memory_type,
            llm_model=llm_model,
            nodes=node_analyses,
            dependencies=list(required_imports),
            warnings=warnings,
            recommendations=recommendations
        )
    
    def _analyze_node(self, node: Dict[str, Any]) -> NodeAnalysis:
        """Analyze a single Flowise node"""
        
        node_data = node.get('data', {})
        node_name = node_data.get('name', 'unknown')
        node_id = node.get('id', 'unknown')
        
        # Determine node type
        node_type = self._classify_node_type(node_name)
        
        # Get mapping info
        mapping = self.node_mappings.get(node_name, {})
        langchain_equivalent = mapping.get('langchain_class', 'Custom implementation required')
        complexity_score = int(mapping.get('complexity', '3'))
        
        # Generate migration notes
        migration_notes = self._generate_migration_notes(node_name, node_data)
        
        # Extract dependencies
        dependencies = self._extract_node_dependencies(node_data)
        
        return NodeAnalysis(
            node_id=node_id,
            node_type=node_type,
            node_name=node_name,
            complexity_score=complexity_score,
            langchain_equivalent=langchain_equivalent,
            migration_notes=migration_notes,
            dependencies=dependencies
        )
    
    def _classify_node_type(self, node_name: str) -> NodeType:
        """Classify node type based on name"""
        
        name_lower = node_name.lower()
        
        if 'llm' in name_lower or 'chat' in name_lower:
            return NodeType.LLM
        elif 'memory' in name_lower:
            return NodeType.MEMORY
        elif 'loader' in name_lower:
            return NodeType.DOCUMENT_LOADER
        elif 'splitter' in name_lower:
            return NodeType.TEXT_SPLITTER
        elif 'vector' in name_lower or 'chroma' in name_lower or 'pinecone' in name_lower:
            return NodeType.VECTOR_STORE
        elif 'retriever' in name_lower:
            return NodeType.RETRIEVER
        elif 'chain' in name_lower:
            return NodeType.CHAIN
        elif 'tool' in name_lower:
            return NodeType.TOOL
        elif 'function' in name_lower:
            return NodeType.CUSTOM_FUNCTION
        elif 'api' in name_lower:
            return NodeType.API_CALL
        else:
            return NodeType.UNKNOWN
    
    def _generate_migration_notes(self, node_name: str, node_data: Dict) -> str:
        """Generate specific migration notes for a node"""
        
        if node_name == "chatOpenAI":
            model = node_data.get('model', 'gpt-3.5-turbo')
            temp = node_data.get('temperature', 0.7)
            return f"Direct mapping: ChatOpenAI(model='{model}', temperature={temp})"
        
        elif node_name == "conversationBufferMemory":
            return "Direct mapping: ConversationBufferMemory()"
        
        elif node_name == "pdfLoader":
            return "Direct mapping: PyPDFLoader(file_path). Ensure file path handling."
        
        elif node_name == "customFunction":
            return "Requires custom tool implementation. Review function logic carefully."
        
        elif node_name == "sequentialChain":
            return "Map to SequentialChain. Review input/output variable mapping."
        
        else:
            return "Review Flowise configuration and map to appropriate LangChain component"
    
    def _extract_node_dependencies(self, node_data: Dict) -> List[str]:
        """Extract dependencies for a node"""
        dependencies = []
        
        # Common dependencies based on node configuration
        if 'openai' in str(node_data).lower():
            dependencies.append('openai')
        if 'anthropic' in str(node_data).lower():
            dependencies.append('anthropic')
        if 'chroma' in str(node_data).lower():
            dependencies.append('chromadb')
        if 'pinecone' in str(node_data).lower():
            dependencies.append('pinecone-client')
        
        return dependencies
    
    def _estimate_migration_time(self, num_nodes: int, complexity: int, nodes: List[NodeAnalysis]) -> float:
        """Estimate migration time in hours"""
        
        base_time = 0.5  # Base time per node
        
        # Factor in complexity
        complexity_multiplier = {
            1: 0.5, 2: 0.75, 3: 1.0, 4: 1.5, 5: 2.0,
            6: 2.5, 7: 3.0, 8: 3.5, 9: 4.0, 10: 5.0
        }
        
        node_time = num_nodes * base_time * complexity_multiplier.get(complexity, 1.0)
        
        # Add time for custom functions
        custom_functions = sum(1 for node in nodes if node.node_type == NodeType.CUSTOM_FUNCTION)
        custom_time = custom_functions * 2.0  # 2 hours per custom function
        
        # Add testing and integration time
        integration_time = max(1.0, num_nodes * 0.2)
        
        total_time = node_time + custom_time + integration_time
        
        return round(total_time, 1)
    
    def _determine_strategy(self, complexity: int, nodes: List[NodeAnalysis]) -> str:
        """Determine the best migration strategy"""
        
        if complexity <= 3:
            return "Direct Migration - Straightforward node-to-node mapping"
        elif complexity <= 6:
            return "Incremental Migration - Migrate in phases, test each component"
        elif complexity <= 8:
            return "Refactor Migration - Significant restructuring recommended"
        else:
            return "Custom Migration - Complex flow requires custom LangChain implementation"
    
    def _extract_llm_model(self, nodes: List[Dict]) -> str:
        """Extract LLM model from nodes"""
        for node in nodes:
            node_data = node.get('data', {})
            if 'chat' in node_data.get('name', '').lower():
                return node_data.get('model', 'gpt-3.5-turbo')
        return 'gpt-3.5-turbo'
    
    def _extract_memory_type(self, nodes: List[Dict]) -> str:
        """Extract memory type from nodes"""
        for node in nodes:
            node_data = node.get('data', {})
            if 'memory' in node_data.get('name', '').lower():
                return node_data.get('name', 'ConversationBufferMemory')
        return 'ConversationBufferMemory'
    
    def _extract_required_chains(self, nodes: List[Dict]) -> List[str]:
        """Extract required LangChain chains"""
        chains = []
        for node in nodes:
            node_data = node.get('data', {})
            node_name = node_data.get('name', '')
            
            if 'chain' in node_name.lower():
                mapping = self.node_mappings.get(node_name, {})
                if mapping:
                    chains.append(mapping['langchain_class'])
        
        return chains
    
    def _generate_warnings(self, nodes: List[NodeAnalysis]) -> List[str]:
        """Generate warnings for potential migration issues"""
        warnings = []
        
        custom_functions = [n for n in nodes if n.node_type == NodeType.CUSTOM_FUNCTION]
        if custom_functions:
            warnings.append(f"{len(custom_functions)} custom functions require manual implementation")
        
        unknown_nodes = [n for n in nodes if n.node_type == NodeType.UNKNOWN]
        if unknown_nodes:
            warnings.append(f"{len(unknown_nodes)} unknown node types need investigation")
        
        high_complexity = [n for n in nodes if n.complexity_score >= 4]
        if len(high_complexity) > len(nodes) * 0.3:
            warnings.append("High proportion of complex nodes - consider phased migration")
        
        return warnings
    
    def _generate_recommendations(self, nodes: List[NodeAnalysis], complexity: int) -> List[str]:
        """Generate migration recommendations"""
        recommendations = []
        
        if complexity >= 7:
            recommendations.append("Consider breaking into smaller, independent chains")
        
        if any(n.node_type == NodeType.CUSTOM_FUNCTION for n in nodes):
            recommendations.append("Test custom functions thoroughly in LangChain environment")
        
        if any(n.node_type == NodeType.VECTOR_STORE for n in nodes):
            recommendations.append("Plan vector store migration separately - consider data backup")
        
        recommendations.append("Implement comprehensive testing suite for migrated code")
        recommendations.append("Consider gradual rollout with A/B testing")
        
        return recommendations

def main():
    """Demo the analyzer"""
    print("🔍 Flowise Flow Analyzer")
    print("=" * 30)
    
    # Example usage
    analyzer = FlowiseAnalyzer()
    
    # Sample Flowise flow data
    sample_flow = {
        "name": "Sample Chat Flow",
        "nodes": [
            {
                "id": "1",
                "data": {
                    "name": "chatOpenAI", 
                    "model": "gpt-4",
                    "temperature": 0.7
                }
            },
            {
                "id": "2", 
                "data": {
                    "name": "conversationBufferMemory"
                }
            },
            {
                "id": "3",
                "data": {
                    "name": "llmChain"
                }
            }
        ],
        "edges": []
    }
    
    analysis = analyzer.analyze_flow(sample_flow)
    
    print(f"Flow: {analysis.flow_name}")
    print(f"Nodes: {analysis.total_nodes}")
    print(f"Complexity: {analysis.complexity_score}/10")
    print(f"Estimated time: {analysis.estimated_hours} hours")
    print(f"Strategy: {analysis.migration_strategy}")
    print(f"LLM Model: {analysis.llm_model}")
    print(f"Memory Type: {analysis.memory_type}")
    
    if analysis.warnings:
        print(f"\n⚠️ Warnings:")
        for warning in analysis.warnings:
            print(f"  - {warning}")
    
    if analysis.recommendations:
        print(f"\n💡 Recommendations:")
        for rec in analysis.recommendations:
            print(f"  - {rec}")

if __name__ == "__main__":
    main()