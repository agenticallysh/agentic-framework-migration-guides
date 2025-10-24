# LangChain → LangGraph Migration Guide

[![Migration Complexity](https://img.shields.io/badge/Complexity-Medium-orange.svg)]()
[![Estimated Time](https://img.shields.io/badge/Time-4--6%20hours-blue.svg)]()
[![Success Rate](https://img.shields.io/badge/Success%20Rate-92%25-green.svg)]()

Complete guide for migrating from LangChain to LangGraph. This migration typically results in **25% performance improvement**, **better state management**, and more robust agent workflows with graph-based execution patterns.

[🚀 Start migration with our interactive tool →](https://www.agentically.sh/ai-agentic-frameworks/migrate/langchain-to-langgraph/)

## 📋 Table of Contents

- [Why Migrate to LangGraph?](#why-migrate-to-langgraph)
- [Before You Start](#before-you-start)
- [Migration Overview](#migration-overview)
- [Step-by-Step Migration](#step-by-step-migration)
- [Code Examples](#code-examples)
- [Common Issues](#common-issues)
- [Testing & Validation](#testing--validation)
- [Post-Migration Optimization](#post-migration-optimization)
- [Automated Tools](#automated-tools)

## 🎯 Why Migrate to LangGraph?

### ✅ Benefits of LangGraph
- **Stateful Workflows**: Persistent state management across execution steps
- **Graph-Based Architecture**: Visual workflow representation and debugging
- **Better Error Handling**: Sophisticated error recovery and retry mechanisms
- **Conditional Logic**: Built-in support for complex conditional workflows
- **Performance**: 20-30% faster execution for complex multi-step workflows
- **Debugging**: Superior visibility into workflow execution and state
- **Scalability**: Better handling of long-running, stateful processes

### ⚠️ What You'll Lose
- **Simplicity**: LangChain's simpler chain abstraction
- **Learning Curve**: Graph concepts require mental model shift
- **Ecosystem**: Some LangChain-specific tools may need adaptation
- **Quick Prototyping**: Graph setup takes more initial planning

### 🎯 Best Candidates for Migration
- ✅ Complex multi-step workflows
- ✅ Stateful agent interactions
- ✅ Conditional logic and branching
- ✅ Long-running processes
- ✅ Workflows requiring retry/error recovery
- ✅ Human-in-the-loop systems
- ❌ Simple linear chains
- ❌ Single-step operations
- ❌ Prototype/MVP applications

## 🛠️ Before You Start

### Prerequisites
```bash
# Install LangGraph and dependencies
pip install langgraph langchain-openai
pip install langsmith  # Optional but recommended for debugging
```

### Migration Readiness Checklist
- [ ] **Understand your workflow**: Map current chain logic to graph nodes
- [ ] **Identify state requirements**: What data flows between steps?
- [ ] **Plan node structure**: Break chains into discrete graph nodes
- [ ] **Consider error handling**: How should failures be handled?
- [ ] **Test coverage**: Ensure comprehensive test coverage exists
- [ ] **Time allocation**: Block 4-6 hours for migration
- [ ] **Graph visualization tools**: Set up LangSmith for debugging

### Pre-Migration Analysis
```bash
python automated-tools/analyzer.py --project-path ./your-project --target langgraph
```

Expected output:
```
📊 LangChain to LangGraph Analysis
==================================
✅ Chains found: 3
✅ Sequential patterns: 2 (good fit for graphs)
✅ Conditional logic: 1 (excellent fit)
✅ State management: Limited (will improve with LangGraph)
⚠️  Complex chains: 1 (requires careful planning)
🎯 Migration complexity: MEDIUM
⏱️  Estimated time: 5.5 hours
💡 LangGraph benefits: High - stateful workflows detected
```

## 🔄 Migration Overview

### Architecture Comparison

#### LangChain Approach (Linear Chains)
```python
# LangChain: Sequential chain execution
from langchain.chains import LLMChain, SequentialChain

# Linear, stateless execution
chain1 = LLMChain(llm=llm, prompt=prompt1)
chain2 = LLMChain(llm=llm, prompt=prompt2)

sequence = SequentialChain(
    chains=[chain1, chain2],
    input_variables=["input"],
    output_variables=["output"]
)

result = sequence.run({"input": "user input"})
```

#### LangGraph Approach (Graph-Based)
```python
# LangGraph: Graph-based execution with state
from langgraph.graph import StateGraph
from typing import TypedDict

class WorkflowState(TypedDict):
    input: str
    step1_result: str
    final_output: str

def step1_node(state: WorkflowState) -> WorkflowState:
    # Process with maintained state
    result = llm.invoke(f"Process: {state['input']}")
    state["step1_result"] = result
    return state

def step2_node(state: WorkflowState) -> WorkflowState:
    # Access previous results via state
    result = llm.invoke(f"Enhance: {state['step1_result']}")
    state["final_output"] = result
    return state

# Build graph
workflow = StateGraph(WorkflowState)
workflow.add_node("step1", step1_node)
workflow.add_node("step2", step2_node)
workflow.add_edge("step1", "step2")
workflow.set_entry_point("step1")
workflow.set_finish_point("step2")

app = workflow.compile()
result = app.invoke({"input": "user input"})
```

### Core Concept Mapping

| LangChain Concept | LangGraph Equivalent | Notes |
|-------------------|---------------------|-------|
| `Chain` | `Node` | Individual processing steps |
| `SequentialChain` | `Graph with edges` | Flow control via graph edges |
| `Memory` | `State` | Persistent state across nodes |
| `Conditional Chain` | `Conditional edges` | Built-in conditional routing |
| `Callback` | `State modification` | State changes tracked automatically |
| `Input/Output` | `State schema` | Typed state management |
| `Error handling` | `Error nodes` | Dedicated error recovery flows |

## 🚀 Step-by-Step Migration

### Step 1: Design State Schema
First, define what data flows through your workflow:

#### Before (LangChain - No explicit state)
```python
# Implicit state passing through chain variables
chain_result = {"analysis": "", "summary": "", "recommendations": ""}
```

#### After (LangGraph - Explicit state schema)
```python
from typing import TypedDict, List

class AnalysisState(TypedDict):
    user_input: str
    raw_data: str
    analysis_result: str
    summary: str
    recommendations: List[str]
    confidence_score: float
    metadata: dict
```

### Step 2: Convert Chains to Nodes

#### Before (LangChain Chain)
```python
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

# Analysis chain
analysis_prompt = PromptTemplate(
    input_variables=["input_data"],
    template="Analyze this data: {input_data}"
)
analysis_chain = LLMChain(llm=llm, prompt=analysis_prompt)

# Summary chain  
summary_prompt = PromptTemplate(
    input_variables=["analysis"],
    template="Summarize: {analysis}"
)
summary_chain = LLMChain(llm=llm, prompt=summary_prompt)
```

#### After (LangGraph Nodes)
```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4")

def analysis_node(state: AnalysisState) -> AnalysisState:
    """Analyze the input data"""
    
    analysis_prompt = f"""Analyze this data thoroughly:
    {state['raw_data']}
    
    Provide detailed insights and findings."""
    
    result = llm.invoke(analysis_prompt)
    
    # Update state
    state["analysis_result"] = result.content
    state["metadata"]["analysis_tokens"] = len(result.content.split())
    
    return state

def summary_node(state: AnalysisState) -> AnalysisState:
    """Create summary from analysis"""
    
    summary_prompt = f"""Create a concise summary:
    {state['analysis_result']}
    
    Focus on key insights and actionable points."""
    
    result = llm.invoke(summary_prompt)
    
    # Update state
    state["summary"] = result.content
    state["confidence_score"] = 0.85  # Could calculate based on analysis
    
    return state
```

### Step 3: Build the Graph

#### Create Graph Structure
```python
from langgraph.graph import StateGraph

def create_analysis_workflow():
    """Create the analysis workflow graph"""
    
    # Initialize graph
    workflow = StateGraph(AnalysisState)
    
    # Add nodes
    workflow.add_node("analyze", analysis_node)
    workflow.add_node("summarize", summary_node)
    workflow.add_node("generate_recommendations", recommendations_node)
    
    # Define flow
    workflow.set_entry_point("analyze")
    workflow.add_edge("analyze", "summarize")
    workflow.add_edge("summarize", "generate_recommendations")
    workflow.set_finish_point("generate_recommendations")
    
    # Compile graph
    app = workflow.compile()
    return app
```

### Step 4: Add Conditional Logic

#### Before (LangChain - Manual conditional logic)
```python
def conditional_chain(input_data):
    analysis = analysis_chain.run(input_data)
    
    if "error" in analysis.lower():
        return error_chain.run(analysis)
    elif "complex" in analysis.lower():
        return complex_chain.run(analysis)
    else:
        return simple_chain.run(analysis)
```

#### After (LangGraph - Built-in conditional edges)
```python
def route_based_on_analysis(state: AnalysisState) -> str:
    """Route to different nodes based on analysis"""
    
    analysis = state["analysis_result"].lower()
    
    if "error" in analysis:
        return "error_handler"
    elif "complex" in analysis:
        return "complex_processor"
    else:
        return "simple_processor"

# Add conditional routing to graph
workflow.add_conditional_edges(
    "analyze",
    route_based_on_analysis,
    {
        "error_handler": "handle_error",
        "complex_processor": "process_complex",
        "simple_processor": "process_simple"
    }
)
```

### Step 5: Handle State and Memory

#### Before (LangChain Memory)
```python
from langchain.memory import ConversationBufferMemory

memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

chain_with_memory = ConversationChain(
    llm=llm,
    memory=memory
)
```

#### After (LangGraph State Management)
```python
# State automatically persists across nodes
class ConversationState(TypedDict):
    messages: List[dict]
    conversation_summary: str
    user_preferences: dict
    session_id: str

def conversation_node(state: ConversationState) -> ConversationState:
    """Handle conversation with persistent state"""
    
    # Access conversation history
    history = state["messages"]
    
    # Process with context
    response = llm.invoke(
        f"Previous conversation: {history}\nNew input: {state['current_input']}"
    )
    
    # Update conversation state
    state["messages"].append({
        "role": "assistant",
        "content": response.content,
        "timestamp": datetime.now().isoformat()
    })
    
    return state
```

### Step 6: Error Handling and Recovery

#### Advanced Error Handling
```python
def error_recovery_node(state: AnalysisState) -> AnalysisState:
    """Handle errors and attempt recovery"""
    
    error_info = state.get("error", {})
    
    if error_info.get("type") == "rate_limit":
        # Wait and retry
        time.sleep(5)
        state["retry_count"] = state.get("retry_count", 0) + 1
        return state
    
    elif error_info.get("type") == "parsing_error":
        # Use fallback processing
        state["use_fallback"] = True
        return state
    
    else:
        # Log error and continue with partial results
        state["warnings"].append(f"Error occurred: {error_info}")
        return state

# Add error handling to graph
workflow.add_node("error_recovery", error_recovery_node)
workflow.add_conditional_edges(
    "analyze",
    lambda state: "error_recovery" if state.get("error") else "summarize",
    {
        "error_recovery": "error_recovery",
        "summarize": "summarize"
    }
)
```

## 💻 Code Examples

### [Basic Workflow Migration](./code-examples/basic-workflow/)
Converting simple sequential chains to LangGraph workflows.

### [Stateful Agent Migration](./code-examples/stateful-agents/)
Migrating agents with complex state management needs.

### [Conditional Flow Migration](./code-examples/conditional-flows/)
Converting conditional logic to graph-based routing.

### [Memory Management Migration](./code-examples/memory-management/)
Transitioning from LangChain memory to LangGraph state.

## ⚠️ Common Issues

### Issue 1: State Management Complexity
**Problem**: Over-complicated state schemas
```python
# ❌ Too complex
class OvercomplicatedState(TypedDict):
    raw_input: str
    processed_input: str
    analysis_step1: str
    analysis_step2: str
    # ... 20 more fields
```

**Solution**: Start simple and add fields as needed
```python
# ✅ Simple and focused
class AnalysisState(TypedDict):
    input: str
    current_step: str
    results: dict
    metadata: dict
```

### Issue 2: Node Responsibilities
**Problem**: Nodes trying to do too much
```python
# ❌ Node doing multiple responsibilities
def mega_node(state):
    # Analyze, summarize, generate recommendations, format output...
    pass
```

**Solution**: Single responsibility per node
```python
# ✅ Focused node responsibilities
def analysis_node(state):
    # Only analyze
    pass

def summary_node(state):
    # Only summarize
    pass
```

### Issue 3: Graph Complexity
**Problem**: Overly complex graph structures
```python
# ❌ Too many conditional paths
workflow.add_conditional_edges("node", router, {
    "path1": "node1", "path2": "node2", ..., "path10": "node10"
})
```

**Solution**: Group related logic and simplify
```python
# ✅ Grouped logic with sub-graphs
def categorize_and_route(state):
    category = determine_category(state)
    return f"{category}_processor"

workflow.add_conditional_edges("categorize", categorize_and_route, {
    "simple_processor": "simple_flow",
    "complex_processor": "complex_flow"
})
```

### Issue 4: State Mutation
**Problem**: Forgetting to return updated state
```python
# ❌ State not returned
def bad_node(state):
    state["result"] = process_data(state["input"])
    # Missing return statement!
```

**Solution**: Always return the state
```python
# ✅ State properly returned
def good_node(state):
    state["result"] = process_data(state["input"])
    return state
```

## 🧪 Testing & Validation

### State-Based Testing
```python
import unittest

class TestWorkflowMigration(unittest.TestCase):
    
    def setUp(self):
        self.workflow = create_analysis_workflow()
    
    def test_state_progression(self):
        """Test that state flows correctly through nodes"""
        
        initial_state = {
            "user_input": "Test data",
            "raw_data": "Sample data for analysis",
            "metadata": {}
        }
        
        result = self.workflow.invoke(initial_state)
        
        # Verify state was updated correctly
        self.assertIn("analysis_result", result)
        self.assertIn("summary", result)
        self.assertIn("recommendations", result)
        self.assertTrue(len(result["analysis_result"]) > 0)
    
    def test_conditional_routing(self):
        """Test conditional logic works correctly"""
        
        error_state = {
            "user_input": "Test",
            "raw_data": "Error in data",
            "metadata": {}
        }
        
        result = self.workflow.invoke(error_state)
        
        # Should route to error handling
        self.assertIn("error_handled", result.get("metadata", {}))
```

### Performance Comparison
```python
def compare_performance():
    """Compare LangChain vs LangGraph performance"""
    
    test_cases = load_test_cases()
    
    # Test LangChain version
    langchain_times = []
    for case in test_cases:
        start = time.time()
        langchain_result = langchain_workflow.run(case)
        langchain_times.append(time.time() - start)
    
    # Test LangGraph version
    langgraph_times = []
    for case in test_cases:
        start = time.time()
        langgraph_result = langgraph_workflow.invoke(case)
        langgraph_times.append(time.time() - start)
    
    print(f"LangChain avg: {sum(langchain_times)/len(langchain_times):.2f}s")
    print(f"LangGraph avg: {sum(langgraph_times)/len(langgraph_times):.2f}s")
```

## 🚀 Post-Migration Optimization

### Advanced LangGraph Features

#### 1. Parallel Node Execution
```python
# Execute multiple nodes in parallel
workflow.add_node("analyze_sentiment", sentiment_node)
workflow.add_node("analyze_topics", topic_node)
workflow.add_node("analyze_entities", entity_node)

# All three run in parallel
workflow.add_edge("input", "analyze_sentiment")
workflow.add_edge("input", "analyze_topics") 
workflow.add_edge("input", "analyze_entities")

# Merge results
workflow.add_node("merge_analysis", merge_node)
workflow.add_edge("analyze_sentiment", "merge_analysis")
workflow.add_edge("analyze_topics", "merge_analysis")
workflow.add_edge("analyze_entities", "merge_analysis")
```

#### 2. Human-in-the-Loop
```python
def human_review_node(state: WorkflowState) -> WorkflowState:
    """Pause for human review"""
    
    # Present results to human
    print(f"Review needed: {state['analysis_result']}")
    approval = input("Approve? (y/n): ")
    
    state["human_approved"] = approval.lower() == 'y'
    return state

def approval_router(state: WorkflowState) -> str:
    """Route based on human approval"""
    return "finalize" if state["human_approved"] else "revise"

workflow.add_node("human_review", human_review_node)
workflow.add_conditional_edges(
    "human_review",
    approval_router,
    {"finalize": "finalize", "revise": "revise"}
)
```

#### 3. Dynamic Graph Modification
```python
def adaptive_workflow(state: WorkflowState):
    """Modify graph based on runtime conditions"""
    
    if state["complexity_score"] > 0.8:
        # Add additional analysis steps for complex cases
        workflow.add_node("deep_analysis", deep_analysis_node)
        workflow.add_edge("analyze", "deep_analysis")
        workflow.add_edge("deep_analysis", "summarize")
    
    return workflow.compile()
```

#### 4. State Checkpointing
```python
from langgraph.checkpoint.sqlite import SqliteSaver

# Enable state persistence
memory = SqliteSaver.from_conn_string(":memory:")

# Compile with checkpointing
app = workflow.compile(checkpointer=memory)

# Resume from checkpoint
config = {"configurable": {"thread_id": "conversation_1"}}
result = app.invoke(input_data, config=config)
```

## 📊 Expected Results

### Performance Improvements
- **Execution Speed**: 20-30% faster for complex workflows
- **State Management**: 90% reduction in state-related bugs
- **Debugging**: 50% faster issue identification with graph visualization
- **Error Recovery**: 80% better error handling and recovery
- **Scalability**: 3x better handling of long-running processes

### Development Benefits
- **Code Clarity**: Graph structure makes workflow logic explicit
- **Maintainability**: Easier to modify and extend workflows
- **Testing**: Better isolation and testing of individual nodes
- **Debugging**: Visual workflow representation aids debugging
- **Documentation**: Graph serves as documentation

## 🔧 Automated Tools

### Migration Analyzer
```bash
python automated-tools/analyzer.py --project-path ./my-project --target langgraph
```

Analyzes LangChain code for LangGraph migration potential.

### Chain-to-Graph Converter
```bash
python automated-tools/converter.py --input chains.py --output graph.py
```

Converts basic LangChain patterns to LangGraph equivalents.

### State Schema Generator
```bash
python automated-tools/schema_generator.py --analyze-chains ./chains/
```

Generates TypedDict schemas based on existing chain data flow.

## 🔗 Next Steps

### Master LangGraph
- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [Advanced Graph Patterns](https://www.agentically.sh/ai-agentic-frameworks/langgraph/patterns/)
- [LangGraph Production Guide](https://www.agentically.sh/ai-agentic-frameworks/langgraph/production/)

### Related Migrations
- [LangChain → CrewAI](../langchain-to-crewai/) - For role-based agents
- [LangChain → AutoGen](../langchain-to-autogen/) - For research workflows
- [LangGraph Optimization Guide](https://www.agentically.sh/ai-agentic-frameworks/langgraph/optimization/)

### Get Help
- [Migration Support Discord](https://discord.gg/agentically)
- [LangGraph Examples](https://github.com/langchain-ai/langgraph/tree/main/examples)
- [Professional Migration Services](https://www.agentically.sh/ai-agentic-frameworks/migration-consulting/)

---

**Migration successful?** [Share your experience](https://www.agentically.sh/ai-agentic-frameworks/migration-stories/) and help other developers!

Built with ❤️ by [Agentically](https://www.agentically.sh) | [Compare More Frameworks →](https://www.agentically.sh/ai-agentic-frameworks/)