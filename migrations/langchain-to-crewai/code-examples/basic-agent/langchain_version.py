#!/usr/bin/env python3
"""
LangChain Basic Agent - BEFORE Migration
========================================

This example shows a typical LangChain agent setup that we'll migrate to CrewAI.
Demonstrates the complexity and boilerplate required for basic agent functionality.
"""

import os
from typing import List, Any
from langchain.agents import create_openai_tools_agent, AgentExecutor
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools import Tool
import requests

# Setup
def setup_environment():
    """Ensure required environment variables are set"""
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("Please set OPENAI_API_KEY environment variable")

# Tools Definition
def search_web(query: str) -> str:
    """Simple web search simulation"""
    try:
        # In real implementation, use actual search API
        return f"Search results for '{query}': Latest AI trends show significant growth in agent frameworks, particularly in multi-agent coordination and autonomous task execution."
    except Exception as e:
        return f"Search failed: {str(e)}"

def analyze_data(data: str) -> str:
    """Analyze provided data"""
    try:
        word_count = len(data.split())
        return f"Analysis complete. Data contains {word_count} words. Key insights: The data shows trends in AI development with focus on automation and intelligent systems."
    except Exception as e:
        return f"Analysis failed: {str(e)}"

# Create tools
search_tool = Tool(
    name="WebSearch",
    description="Search the web for current information about any topic",
    func=search_web
)

analysis_tool = Tool(
    name="DataAnalysis", 
    description="Analyze and provide insights on given data or text",
    func=analyze_data
)

tools = [search_tool, analysis_tool]

def create_langchain_agent():
    """Create a LangChain research agent with tools"""
    
    # Initialize LLM
    llm = ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=0.1,
        max_tokens=1000
    )
    
    # Create detailed prompt template
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a Senior Research Analyst with expertise in technology trends.
        
        Your capabilities:
        - Conduct thorough research using web search
        - Analyze complex data and extract insights
        - Provide well-structured, actionable reports
        
        Instructions:
        - Always search for the most current information
        - Analyze findings thoroughly before responding
        - Present insights in a clear, professional manner
        - Include specific examples and data points when available
        """),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad")
    ])
    
    # Create the agent
    agent = create_openai_tools_agent(
        llm=llm,
        tools=tools,
        prompt=prompt
    )
    
    # Create executor with error handling
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        max_iterations=5,
        max_execution_time=60,
        handle_parsing_errors=True,
        return_intermediate_steps=True
    )
    
    return agent_executor

def run_research_task(agent_executor: AgentExecutor, topic: str) -> dict:
    """Execute a research task using the LangChain agent"""
    
    try:
        print(f"🔍 Starting research on: {topic}")
        
        # Execute the agent
        result = agent_executor.invoke({
            "input": f"""Please research the topic: {topic}
            
            Your task:
            1. Search for current information about this topic
            2. Analyze the key findings and trends
            3. Provide a comprehensive summary with insights
            
            Focus on recent developments and practical implications.
            """
        })
        
        return {
            "success": True,
            "output": result.get("output", ""),
            "intermediate_steps": result.get("intermediate_steps", []),
            "total_tokens": "Unknown"  # LangChain doesn't track this easily
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "output": ""
        }

def main():
    """Main execution function"""
    print("🤖 LangChain Research Agent")
    print("=" * 50)
    
    try:
        # Setup
        setup_environment()
        
        # Create agent
        print("🏗️  Creating LangChain agent...")
        agent_executor = create_langchain_agent()
        print("✅ Agent created successfully")
        
        # Example research tasks
        research_topics = [
            "AI agent frameworks in 2025",
            "Multi-agent systems best practices",
            "LangChain vs CrewAI comparison"
        ]
        
        for topic in research_topics:
            print(f"\n📋 Research Task: {topic}")
            print("-" * 40)
            
            result = run_research_task(agent_executor, topic)
            
            if result["success"]:
                print("✅ Research completed successfully!")
                print(f"📄 Output: {result['output'][:200]}...")
                print(f"🔧 Steps taken: {len(result['intermediate_steps'])}")
            else:
                print(f"❌ Research failed: {result['error']}")
            
            print()
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()

"""
Issues with this LangChain approach:
=====================================

1. Complexity: Lots of boilerplate code for basic functionality
2. Manual orchestration: Need to handle agent coordination manually
3. Error handling: Complex error management required
4. Memory: Manual memory management if needed
5. Prompt engineering: Verbose prompt templates
6. Tool integration: Complex tool setup and management
7. Monitoring: Limited built-in monitoring and metrics
8. Scaling: Difficult to coordinate multiple agents

This is what we'll simplify with CrewAI migration.
"""