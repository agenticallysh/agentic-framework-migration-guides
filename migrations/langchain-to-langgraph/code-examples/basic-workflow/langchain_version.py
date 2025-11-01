#!/usr/bin/env python3
"""
LangChain Basic Agent Workflow - Before Migration
Demonstrates a typical LangChain agent with sequential tool usage and memory.
"""

import os
from typing import List, Dict, Any
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.agents.output_parsers.openai_tools import OpenAIToolsAgentOutputParser
from langchain.agents.format_scratchpad.openai_tools import format_to_openai_tool_messages
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain.tools import Tool
from langchain.memory import ConversationBufferWindowMemory
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
import requests
import json

# Initialize LLM
llm = ChatOpenAI(
    model="gpt-4",
    temperature=0.1,
    openai_api_key=os.getenv("OPENAI_API_KEY")
)

# Define custom tools
def search_web(query: str) -> str:
    """Search the web for information."""
    # Simulated web search - replace with actual implementation
    return f"Search results for '{query}': Found relevant information about {query}"

def get_weather(location: str) -> str:
    """Get current weather for a location."""
    # Simulated weather API - replace with actual implementation
    return f"Weather in {location}: 72°F, sunny with light clouds"

def save_to_file(content: str, filename: str) -> str:
    """Save content to a file."""
    try:
        with open(f"outputs/{filename}", "w") as f:
            f.write(content)
        return f"Successfully saved content to outputs/{filename}"
    except Exception as e:
        return f"Error saving file: {str(e)}"

# Create tools list
tools = [
    Tool(
        name="search_web",
        description="Search the web for current information",
        func=search_web
    ),
    Tool(
        name="get_weather", 
        description="Get current weather information for a location",
        func=get_weather
    ),
    Tool(
        name="save_to_file",
        description="Save content to a file in the outputs directory",
        func=save_to_file
    )
]

# Create memory
memory = ConversationBufferWindowMemory(
    memory_key="chat_history",
    return_messages=True,
    k=10  # Keep last 10 exchanges
)

# Create prompt template
prompt = ChatPromptTemplate.from_messages([
    SystemMessage(content="""You are a helpful research assistant that can search for information, 
    check weather, and save findings to files. Use the available tools to help users with their requests.
    
    Be thorough in your research and always save important findings to files for future reference."""),
    MessagesPlaceholder(variable_name="chat_history"),
    HumanMessage(content="{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad")
])

# Create agent
agent = create_openai_tools_agent(llm, tools, prompt)

# Create agent executor
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    memory=memory,
    verbose=True,
    max_iterations=5,
    handle_parsing_errors=True
)

class LangChainWorkflow:
    """LangChain-based workflow for research and analysis tasks."""
    
    def __init__(self):
        self.executor = agent_executor
        self.conversation_history = []
    
    def execute_task(self, task: str) -> Dict[str, Any]:
        """Execute a research task using LangChain agent."""
        print(f"\n🔍 Executing task: {task}")
        
        try:
            # Execute the task
            result = self.executor.invoke({
                "input": task,
                "chat_history": self.conversation_history
            })
            
            # Store in conversation history
            self.conversation_history.extend([
                HumanMessage(content=task),
                AIMessage(content=result["output"])
            ])
            
            return {
                "success": True,
                "output": result["output"],
                "intermediate_steps": result.get("intermediate_steps", []),
                "total_tokens": getattr(result, "total_tokens", 0)
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "output": f"Error executing task: {str(e)}"
            }
    
    def execute_workflow(self, tasks: List[str]) -> List[Dict[str, Any]]:
        """Execute a sequence of related tasks."""
        results = []
        
        print("🚀 Starting LangChain workflow execution...")
        
        for i, task in enumerate(tasks, 1):
            print(f"\n📋 Step {i}/{len(tasks)}")
            result = self.execute_task(task)
            results.append({
                "step": i,
                "task": task,
                "result": result
            })
            
            # Break if error occurs
            if not result["success"]:
                print(f"❌ Workflow stopped due to error in step {i}")
                break
        
        return results

def main():
    """Demonstrate LangChain workflow."""
    # Create outputs directory
    os.makedirs("outputs", exist_ok=True)
    
    # Initialize workflow
    workflow = LangChainWorkflow()
    
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
    print("📊 WORKFLOW SUMMARY")
    print("="*60)
    
    successful_steps = sum(1 for r in results if r["result"]["success"])
    print(f"✅ Successful steps: {successful_steps}/{len(results)}")
    print(f"📝 Total conversation exchanges: {len(workflow.conversation_history)}")
    
    for result in results:
        status = "✅" if result["result"]["success"] else "❌"
        print(f"{status} Step {result['step']}: {result['task'][:50]}...")
        if not result["result"]["success"]:
            print(f"   Error: {result['result']['error']}")

if __name__ == "__main__":
    main()