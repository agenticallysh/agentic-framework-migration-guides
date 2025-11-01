#!/usr/bin/env python3
"""
LangGraph Conditional Workflow - After Migration
Demonstrates sophisticated conditional logic using LangGraph's native graph capabilities.
"""

import os
from typing import List, Dict, Any, TypedDict, Annotated, Literal
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
import json

# Initialize LLM
llm = ChatOpenAI(
    model="gpt-4",
    temperature=0.1,
    openai_api_key=os.getenv("OPENAI_API_KEY")
)

# Define comprehensive state schema
class ModerationState(TypedDict):
    messages: Annotated[List[Any], add_messages]
    content: str
    sentiment_analysis: Dict[str, Any]
    safety_check: Dict[str, Any]
    workflow_decision: str
    action_result: Dict[str, Any]
    escalation_reason: str
    processing_stage: str
    content_summary: str

# Define tools with proper typing
@tool
def analyze_text_sentiment(text: str) -> str:
    """Analyze the sentiment of given text."""
    positive_words = ['good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic']
    negative_words = ['bad', 'terrible', 'awful', 'horrible', 'disappointing', 'poor']
    
    text_lower = text.lower()
    positive_count = sum(1 for word in positive_words if word in text_lower)
    negative_count = sum(1 for word in negative_words if word in text_lower)
    
    if positive_count > negative_count:
        return json.dumps({"sentiment": "positive", "confidence": 0.85, "score": positive_count})
    elif negative_count > positive_count:
        return json.dumps({"sentiment": "negative", "confidence": 0.82, "score": negative_count})
    else:
        return json.dumps({"sentiment": "neutral", "confidence": 0.60, "score": 0})

@tool
def check_content_safety(content: str) -> str:
    """Check if content is safe for publication."""
    unsafe_patterns = ['violence', 'hate', 'explicit', 'harmful', 'illegal']
    
    content_lower = content.lower()
    safety_issues = [pattern for pattern in unsafe_patterns if pattern in content_lower]
    
    return json.dumps({
        "safe": len(safety_issues) == 0,
        "issues": safety_issues,
        "confidence": 0.95 if len(safety_issues) == 0 else 0.85
    })

@tool
def generate_summary(text: str) -> str:
    """Generate a summary of the provided text."""
    sentences = text.split('.')
    if len(sentences) <= 2:
        return f"Summary: {text[:100]}..."
    else:
        return f"Summary: {'. '.join(sentences[:2])}... (original length: {len(text)} chars)"

@tool
def escalate_to_human(reason: str, content: str) -> str:
    """Escalate content to human review."""
    escalation_id = f"ESC-{hash(content) % 10000:04d}"
    return json.dumps({
        "escalated": True,
        "escalation_id": escalation_id,
        "reason": reason,
        "status": "pending_human_review"
    })

@tool
def publish_content(content: str, summary: str) -> str:
    """Publish approved content."""
    content_id = f"PUB-{hash(content) % 10000:04d}"
    return json.dumps({
        "published": True,
        "content_id": content_id,
        "url": f"https://example.com/content/{content_id}",
        "summary": summary
    })

# Organize tools
analysis_tools = [analyze_text_sentiment, check_content_safety]
action_tools = [generate_summary, escalate_to_human, publish_content]
all_tools = analysis_tools + action_tools

# Create tool nodes
analysis_tool_node = ToolNode(analysis_tools)
action_tool_node = ToolNode(action_tools)

# Node functions
def start_analysis(state: ModerationState) -> Dict[str, Any]:
    """Start the content analysis process."""
    content = state["content"]
    
    # Create analysis message
    analysis_prompt = f"""Analyze this content for moderation:

Content: "{content}"

Please use the available tools to:
1. analyze_text_sentiment - to check sentiment
2. check_content_safety - to verify safety

Use both tools and provide the results."""
    
    message = HumanMessage(content=analysis_prompt)
    
    return {
        "messages": [message],
        "processing_stage": "analysis"
    }

def process_analysis_results(state: ModerationState) -> Dict[str, Any]:
    """Process the results from analysis tools."""
    messages = state["messages"]
    
    # Extract tool results from messages
    sentiment_result = {}
    safety_result = {}
    
    for message in reversed(messages):
        if hasattr(message, 'tool_calls') and message.tool_calls:
            for tool_call in message.tool_calls:
                if tool_call["name"] == "analyze_text_sentiment":
                    # Find corresponding result
                    for result_msg in messages:
                        if hasattr(result_msg, 'tool_call_id') and result_msg.tool_call_id == tool_call["id"]:
                            try:
                                sentiment_result = json.loads(result_msg.content)
                            except json.JSONDecodeError:
                                sentiment_result = {"error": "Failed to parse sentiment"}
                            break
                elif tool_call["name"] == "check_content_safety":
                    for result_msg in messages:
                        if hasattr(result_msg, 'tool_call_id') and result_msg.tool_call_id == tool_call["id"]:
                            try:
                                safety_result = json.loads(result_msg.content)
                            except json.JSONDecodeError:
                                safety_result = {"error": "Failed to parse safety"}
                            break
    
    return {
        "sentiment_analysis": sentiment_result,
        "safety_check": safety_result,
        "processing_stage": "decision"
    }

def make_moderation_decision(state: ModerationState) -> Dict[str, Any]:
    """Make moderation decision based on analysis results."""
    sentiment = state.get("sentiment_analysis", {})
    safety = state.get("safety_check", {})
    content = state["content"]
    
    # Decision logic
    is_safe = safety.get("safe", False)
    sentiment_type = sentiment.get("sentiment", "unknown")
    sentiment_score = sentiment.get("score", 0)
    
    # Determine workflow decision
    if not is_safe:
        decision = "escalate_safety"
        escalation_reason = f"Content safety issues: {safety.get('issues', [])}"
    elif sentiment_type == "negative" and sentiment_score > 2:
        decision = "escalate_sentiment"
        escalation_reason = f"Highly negative sentiment (score: {sentiment_score})"
    else:
        decision = "approve_content"
        escalation_reason = ""
    
    return {
        "workflow_decision": decision,
        "escalation_reason": escalation_reason,
        "processing_stage": "action"
    }

def execute_escalation(state: ModerationState) -> Dict[str, Any]:
    """Execute escalation to human review."""
    content = state["content"]
    reason = state["escalation_reason"]
    
    escalation_prompt = f"""Please escalate this content to human review:

Content: "{content}"
Reason: {reason}

Use the escalate_to_human tool with the reason and content."""
    
    message = HumanMessage(content=escalation_prompt)
    
    return {
        "messages": [message],
        "processing_stage": "escalation"
    }

def execute_publication(state: ModerationState) -> Dict[str, Any]:
    """Execute content publication workflow."""
    content = state["content"]
    
    publication_prompt = f"""This content has been approved for publication:

Content: "{content}"

Please:
1. Use generate_summary to create a summary
2. Use publish_content to publish it with the summary

Execute both steps."""
    
    message = HumanMessage(content=publication_prompt)
    
    return {
        "messages": [message],
        "processing_stage": "publication"
    }

def finalize_workflow(state: ModerationState) -> Dict[str, Any]:
    """Finalize the workflow and extract results."""
    messages = state["messages"]
    
    # Extract final action results
    action_result = {}
    content_summary = ""
    
    for message in reversed(messages):
        if hasattr(message, 'tool_calls') and message.tool_calls:
            for tool_call in message.tool_calls:
                # Find corresponding result
                for result_msg in messages:
                    if hasattr(result_msg, 'tool_call_id') and result_msg.tool_call_id == tool_call["id"]:
                        try:
                            result_data = json.loads(result_msg.content)
                            if tool_call["name"] in ["escalate_to_human", "publish_content"]:
                                action_result = result_data
                            elif tool_call["name"] == "generate_summary":
                                content_summary = result_data if isinstance(result_data, str) else result_msg.content
                        except json.JSONDecodeError:
                            if tool_call["name"] == "generate_summary":
                                content_summary = result_msg.content
                        break
    
    return {
        "action_result": action_result,
        "content_summary": content_summary,
        "processing_stage": "completed"
    }

# Conditional routing functions
def should_continue_analysis(state: ModerationState) -> Literal["tools", "process_results"]:
    """Determine if we need to continue with analysis tools."""
    messages = state["messages"]
    last_message = messages[-1]
    
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        return "tools"
    return "process_results"

def route_moderation_decision(state: ModerationState) -> Literal["escalate", "publish", "end"]:
    """Route based on moderation decision."""
    decision = state.get("workflow_decision", "")
    
    if decision.startswith("escalate"):
        return "escalate"
    elif decision == "approve_content":
        return "publish"
    else:
        return "end"

def should_continue_action(state: ModerationState) -> Literal["action_tools", "finalize"]:
    """Determine if we need to continue with action tools."""
    messages = state["messages"]
    last_message = messages[-1]
    
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        return "action_tools"
    return "finalize"

class LangGraphConditionalWorkflow:
    """LangGraph-based conditional content moderation workflow."""
    
    def __init__(self, checkpoint_path: str = "moderation_checkpoints.db"):
        self.checkpoint_path = checkpoint_path
        self.memory = SqliteSaver.from_conn_string(f"sqlite:///{checkpoint_path}")
        self.graph = self._build_graph()
        self.workflow_stats = {
            "processed": 0,
            "escalated": 0,
            "published": 0
        }
    
    def _build_graph(self) -> StateGraph:
        """Build the moderation workflow graph."""
        workflow = StateGraph(ModerationState)
        
        # Add nodes
        workflow.add_node("start_analysis", start_analysis)
        workflow.add_node("analysis_tools", analysis_tool_node)
        workflow.add_node("process_results", process_analysis_results)
        workflow.add_node("make_decision", make_moderation_decision)
        workflow.add_node("escalate", execute_escalation)
        workflow.add_node("publish", execute_publication)
        workflow.add_node("action_tools", action_tool_node)
        workflow.add_node("finalize", finalize_workflow)
        
        # Set entry point
        workflow.set_entry_point("start_analysis")
        
        # Add conditional edges
        workflow.add_conditional_edges(
            "start_analysis",
            should_continue_analysis,
            {
                "tools": "analysis_tools",
                "process_results": "process_results"
            }
        )
        
        workflow.add_conditional_edges(
            "analysis_tools",
            should_continue_analysis,
            {
                "tools": "analysis_tools",
                "process_results": "process_results"
            }
        )
        
        workflow.add_conditional_edges(
            "make_decision",
            route_moderation_decision,
            {
                "escalate": "escalate",
                "publish": "publish",
                "end": END
            }
        )
        
        workflow.add_conditional_edges(
            "escalate",
            should_continue_action,
            {
                "action_tools": "action_tools",
                "finalize": "finalize"
            }
        )
        
        workflow.add_conditional_edges(
            "publish",
            should_continue_action,
            {
                "action_tools": "action_tools",
                "finalize": "finalize"
            }
        )
        
        workflow.add_conditional_edges(
            "action_tools",
            should_continue_action,
            {
                "action_tools": "action_tools",
                "finalize": "finalize"
            }
        )
        
        # Add edges
        workflow.add_edge("process_results", "make_decision")
        workflow.add_edge("finalize", END)
        
        # Compile with checkpoints
        return workflow.compile(checkpointer=self.memory)
    
    def process_content(self, content: str, thread_id: str = None) -> Dict[str, Any]:
        """Process content through the moderation workflow."""
        if thread_id is None:
            thread_id = f"content_{hash(content) % 10000:04d}"
        
        print(f"\n🔍 Processing content: {content[:50]}...")
        
        try:
            # Create initial state
            initial_state = {
                "content": content,
                "messages": [],
                "processing_stage": "start",
                "sentiment_analysis": {},
                "safety_check": {},
                "workflow_decision": "",
                "action_result": {},
                "escalation_reason": "",
                "content_summary": ""
            }
            
            # Execute workflow
            config = {"configurable": {"thread_id": thread_id}}
            final_state = self.graph.invoke(initial_state, config)
            
            # Update stats
            self.workflow_stats["processed"] += 1
            decision = final_state.get("workflow_decision", "")
            if decision.startswith("escalate"):
                self.workflow_stats["escalated"] += 1
            elif decision == "approve_content":
                self.workflow_stats["published"] += 1
            
            return {
                "success": True,
                "content": content,
                "thread_id": thread_id,
                "sentiment_analysis": final_state.get("sentiment_analysis", {}),
                "safety_check": final_state.get("safety_check", {}),
                "workflow_decision": final_state.get("workflow_decision", ""),
                "action_result": final_state.get("action_result", {}),
                "escalation_reason": final_state.get("escalation_reason", ""),
                "content_summary": final_state.get("content_summary", ""),
                "processing_stage": final_state.get("processing_stage", ""),
                "checkpointed": True
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "content": content,
                "thread_id": thread_id
            }
    
    def get_workflow_stats(self) -> Dict[str, Any]:
        """Get workflow statistics."""
        total = self.workflow_stats["processed"]
        return {
            "total_processed": total,
            "escalations": self.workflow_stats["escalated"],
            "publications": self.workflow_stats["published"],
            "escalation_rate": self.workflow_stats["escalated"] / total if total > 0 else 0,
            "publication_rate": self.workflow_stats["published"] / total if total > 0 else 0
        }
    
    def get_workflow_state(self, thread_id: str) -> Dict[str, Any]:
        """Get current state of a workflow thread."""
        try:
            config = {"configurable": {"thread_id": thread_id}}
            state = self.graph.get_state(config)
            return {
                "exists": True,
                "current_stage": state.values.get("processing_stage") if state else None,
                "next_steps": state.next if state else None,
                "values": state.values if state else None
            }
        except Exception:
            return {"exists": False}

def main():
    """Demonstrate LangGraph conditional workflow."""
    # Create workflow
    workflow = LangGraphConditionalWorkflow()
    
    # Test content samples
    test_contents = [
        "This is a wonderful product! I highly recommend it to everyone. Great quality and amazing customer service.",
        "This product is terrible and I hate it. Complete waste of money and time. Awful experience.",
        "The product arrived on time and works as expected. Standard quality for the price point.",
        "This content contains explicit violence and harmful content that should not be published.",
        "Looking forward to trying this new feature. Hope it works well for my use case."
    ]
    
    print("🚀 Starting LangGraph Conditional Workflow Demo")
    print("="*60)
    
    results = []
    for i, content in enumerate(test_contents, 1):
        print(f"\n📋 Processing Content {i}/{len(test_contents)}")
        result = workflow.process_content(content)
        results.append(result)
        
        if result["success"]:
            decision = result["workflow_decision"]
            stage = result["processing_stage"]
            print(f"✅ Decision: {decision}")
            print(f"🔄 Final Stage: {stage}")
            if result["escalation_reason"]:
                print(f"🚨 Escalation Reason: {result['escalation_reason']}")
        else:
            print(f"❌ Error: {result['error']}")
    
    # Print workflow statistics
    print("\n" + "="*60)
    print("📊 LANGGRAPH WORKFLOW STATISTICS")
    print("="*60)
    
    stats = workflow.get_workflow_stats()
    print(f"📝 Total Processed: {stats['total_processed']}")
    print(f"🚨 Escalations: {stats['escalations']} ({stats['escalation_rate']:.1%})")
    print(f"✅ Publications: {stats['publications']} ({stats['publication_rate']:.1%})")
    print(f"💾 State persisted to: {workflow.checkpoint_path}")
    
    # Show detailed results
    print(f"\n📋 DETAILED RESULTS")
    print("-" * 40)
    for i, result in enumerate(results, 1):
        if result["success"]:
            sentiment = result.get("sentiment_analysis", {}).get("sentiment", "unknown")
            safety = "safe" if result.get("safety_check", {}).get("safe", False) else "unsafe"
            decision = result["workflow_decision"]
            print(f"{i}. Sentiment: {sentiment}, Safety: {safety} → {decision}")

if __name__ == "__main__":
    main()