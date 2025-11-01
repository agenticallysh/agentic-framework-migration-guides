#!/usr/bin/env python3
"""
LangGraph Basic Workflow - After Migration
Demonstrates the same workflow using LangGraph's stateful graph-based approach.
"""

import os
from typing import List, Dict, Any, TypedDict, Annotated
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
import sqlite3

# Initialize LLM
llm = ChatOpenAI(
    model="gpt-4",
    temperature=0.1,
    openai_api_key=os.getenv("OPENAI_API_KEY")
)

# Define state schema
class WorkflowState(TypedDict):
    messages: Annotated[List[Any], add_messages]
    current_task: str
    task_results: List[Dict[str, Any]]
    task_index: int
    total_tasks: int

# Define tools using the @tool decorator for better integration
@tool
def search_web(query: str) -> str:
    """Search the web for information."""
    # Simulated web search - replace with actual implementation
    return f"Search results for '{query}': Found relevant information about {query}"

@tool
def get_weather(location: str) -> str:
    """Get current weather for a location."""
    # Simulated weather API - replace with actual implementation
    return f"Weather in {location}: 72°F, sunny with light clouds"

@tool
def save_to_file(content: str, filename: str) -> str:
    """Save content to a file."""
    try:
        with open(f"outputs/{filename}", "w") as f:
            f.write(content)
        return f"Successfully saved content to outputs/{filename}"
    except Exception as e:
        return f"Error saving file: {str(e)}"

# Create tools list
tools = [search_web, get_weather, save_to_file]

# Bind tools to LLM
llm_with_tools = llm.bind_tools(tools)

def should_continue(state: WorkflowState) -> str:
    """Determine if workflow should continue or end."""
    messages = state["messages"]
    last_message = messages[-1]
    
    # If the last message has tool calls, continue to tools
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        return "tools"
    
    # Check if we have more tasks to process
    if state["task_index"] < state["total_tasks"] - 1:
        return "continue_workflow"
    
    # Otherwise end
    return END

def call_model(state: WorkflowState) -> Dict[str, Any]:
    """Call the model with the current state."""
    messages = state["messages"]
    
    # Add system message if this is the start
    if len(messages) == 1:
        system_msg = SystemMessage(content="""You are a helpful research assistant that can search for information, 
        check weather, and save findings to files. Use the available tools to help users with their requests.
        
        Be thorough in your research and always save important findings to files for future reference.""")
        messages = [system_msg] + messages
    
    response = llm_with_tools.invoke(messages)
    
    return {"messages": [response]}

def process_next_task(state: WorkflowState) -> Dict[str, Any]:
    """Process the next task in the workflow."""
    task_index = state["task_index"] + 1
    
    if task_index < state["total_tasks"]:
        # Get next task (this would be configured externally)
        next_task = state.get("pending_tasks", [])[task_index] if "pending_tasks" in state else ""
        
        return {
            "messages": [HumanMessage(content=next_task)],
            "current_task": next_task,
            "task_index": task_index
        }
    
    return {"task_index": task_index}

def record_task_result(state: WorkflowState) -> Dict[str, Any]:
    """Record the result of the current task."""
    messages = state["messages"]
    last_message = messages[-1]
    
    task_result = {
        "task": state["current_task"],
        "result": last_message.content if hasattr(last_message, 'content') else str(last_message),
        "success": True
    }
    
    current_results = state.get("task_results", [])
    current_results.append(task_result)
    
    return {"task_results": current_results}

class LangGraphWorkflow:
    """LangGraph-based workflow with persistent state management."""
    
    def __init__(self, checkpoint_path: str = "workflow_checkpoints.db"):
        self.checkpoint_path = checkpoint_path
        self.memory = SqliteSaver.from_conn_string(f"sqlite:///{checkpoint_path}")
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow graph."""
        # Create the state graph
        workflow = StateGraph(WorkflowState)
        
        # Add nodes
        workflow.add_node("model", call_model)
        workflow.add_node("tools", ToolNode(tools))
        workflow.add_node("record_result", record_task_result)
        workflow.add_node("next_task", process_next_task)
        
        # Set entry point
        workflow.set_entry_point("model")
        
        # Add conditional edges
        workflow.add_conditional_edges(
            "model",
            should_continue,
            {
                "tools": "tools",
                "continue_workflow": "record_result",
                END: END
            }
        )
        
        # Add edges
        workflow.add_edge("tools", "model")
        workflow.add_edge("record_result", "next_task")
        workflow.add_edge("next_task", "model")
        
        # Compile with checkpoints
        return workflow.compile(checkpointer=self.memory)
    
    def execute_task(self, task: str, thread_id: str = "main") -> Dict[str, Any]:
        """Execute a single task with state persistence."""
        print(f"\n🔍 Executing task: {task}")
        
        try:
            # Create initial state
            initial_state = {
                "messages": [HumanMessage(content=task)],
                "current_task": task,
                "task_results": [],
                "task_index": 0,
                "total_tasks": 1
            }
            
            # Execute the graph
            config = {"configurable": {"thread_id": thread_id}}
            result = self.graph.invoke(initial_state, config)
            
            return {
                "success": True,
                "output": result["messages"][-1].content if result["messages"] else "No output",
                "state": result,
                "checkpointed": True
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "output": f"Error executing task: {str(e)}"
            }
    
    def execute_workflow(self, tasks: List[str], thread_id: str = "workflow") -> List[Dict[str, Any]]:
        """Execute a sequence of tasks with persistent state."""
        results = []
        
        print("🚀 Starting LangGraph workflow execution...")
        
        # Create initial state with all tasks
        initial_state = {
            "messages": [HumanMessage(content=tasks[0])],
            "current_task": tasks[0],
            "task_results": [],
            "task_index": 0,
            "total_tasks": len(tasks),
            "pending_tasks": tasks
        }
        
        config = {"configurable": {"thread_id": thread_id}}
        
        try:
            # Execute the entire workflow
            final_state = self.graph.invoke(initial_state, config)
            
            # Extract results from final state
            task_results = final_state.get("task_results", [])
            
            for i, task in enumerate(tasks):
                if i < len(task_results):
                    results.append({
                        "step": i + 1,
                        "task": task,
                        "result": task_results[i]
                    })
                else:
                    results.append({
                        "step": i + 1,
                        "task": task,
                        "result": {"success": False, "error": "Task not completed"}
                    })
            
        except Exception as e:
            print(f"❌ Workflow error: {str(e)}")
            results.append({
                "step": 1,
                "task": tasks[0] if tasks else "Unknown",
                "result": {"success": False, "error": str(e)}
            })
        
        return results
    
    def get_workflow_state(self, thread_id: str = "main") -> Dict[str, Any]:
        """Get the current state of a workflow thread."""
        try:
            config = {"configurable": {"thread_id": thread_id}}
            state = self.graph.get_state(config)
            return {
                "exists": True,
                "state": state.values if state else None,
                "next_steps": state.next if state else None
            }
        except Exception:
            return {"exists": False, "state": None}
    
    def resume_workflow(self, thread_id: str = "main") -> Dict[str, Any]:
        """Resume a workflow from its last checkpoint."""
        state_info = self.get_workflow_state(thread_id)
        
        if not state_info["exists"]:
            return {"success": False, "error": "No workflow state found for thread"}
        
        try:
            config = {"configurable": {"thread_id": thread_id}}
            # Continue from last checkpoint
            result = self.graph.invoke(None, config)
            
            return {
                "success": True,
                "resumed": True,
                "final_state": result
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to resume workflow: {str(e)}"
            }

def main():
    """Demonstrate LangGraph workflow with state persistence."""
    # Create outputs directory
    os.makedirs("outputs", exist_ok=True)
    
    # Initialize workflow
    workflow = LangGraphWorkflow()
    
    # Define research workflow
    research_tasks = [
        "Search for the latest trends in artificial intelligence for 2024",
        "Get the weather in San Francisco", 
        "Save a summary of AI trends and SF weather to a file called 'research_summary.txt'"
    ]
    
    # Execute workflow
    results = workflow.execute_workflow(research_tasks)
    
    # Print summary
    print("\n" + "="*60)
    print("📊 LANGGRAPH WORKFLOW SUMMARY")
    print("="*60)
    
    successful_steps = sum(1 for r in results if r["result"]["success"])
    print(f"✅ Successful steps: {successful_steps}/{len(results)}")
    print(f"💾 State persisted to: {workflow.checkpoint_path}")
    
    # Show state information
    state_info = workflow.get_workflow_state("workflow")
    print(f"🔄 Workflow state available: {state_info['exists']}")
    
    for result in results:
        status = "✅" if result["result"]["success"] else "❌"
        print(f"{status} Step {result['step']}: {result['task'][:50]}...")
        if not result["result"]["success"]:
            print(f"   Error: {result['result'].get('error', 'Unknown error')}")
    
    # Demonstrate state persistence
    print("\n🔄 Workflow can be resumed from any checkpoint!")
    print("💡 State is automatically saved after each step.")

if __name__ == "__main__":
    main()