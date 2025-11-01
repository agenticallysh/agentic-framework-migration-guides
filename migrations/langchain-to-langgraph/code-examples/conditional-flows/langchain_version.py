#!/usr/bin/env python3
"""
LangChain Conditional Workflow - Before Migration
Demonstrates complex conditional logic using LangChain with manual flow control.
"""

import os
from typing import List, Dict, Any, Optional
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain.tools import Tool
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
import json
import re

# Initialize LLM
llm = ChatOpenAI(
    model="gpt-4",
    temperature=0.1,
    openai_api_key=os.getenv("OPENAI_API_KEY")
)

# Define tools with different complexity levels
def analyze_text_sentiment(text: str) -> str:
    """Analyze the sentiment of given text."""
    # Simulated sentiment analysis
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

def check_content_safety(content: str) -> str:
    """Check if content is safe for publication."""
    # Simulated content safety check
    unsafe_patterns = ['violence', 'hate', 'explicit', 'harmful', 'illegal']
    
    content_lower = content.lower()
    safety_issues = [pattern for pattern in unsafe_patterns if pattern in content_lower]
    
    return json.dumps({
        "safe": len(safety_issues) == 0,
        "issues": safety_issues,
        "confidence": 0.95 if len(safety_issues) == 0 else 0.85
    })

def generate_summary(text: str) -> str:
    """Generate a summary of the provided text."""
    # Simulated text summarization
    sentences = text.split('.')
    if len(sentences) <= 2:
        return f"Summary: {text[:100]}..."
    else:
        return f"Summary: {'. '.join(sentences[:2])}... (original length: {len(text)} chars)"

def escalate_to_human(reason: str, content: str) -> str:
    """Escalate content to human review."""
    escalation_id = f"ESC-{hash(content) % 10000:04d}"
    return json.dumps({
        "escalated": True,
        "escalation_id": escalation_id,
        "reason": reason,
        "status": "pending_human_review"
    })

def publish_content(content: str, summary: str) -> str:
    """Publish approved content."""
    content_id = f"PUB-{hash(content) % 10000:04d}"
    return json.dumps({
        "published": True,
        "content_id": content_id,
        "url": f"https://example.com/content/{content_id}",
        "summary": summary
    })

# Create tools
tools = [
    Tool(name="analyze_sentiment", description="Analyze sentiment of text", func=analyze_text_sentiment),
    Tool(name="check_safety", description="Check content safety", func=check_content_safety),
    Tool(name="generate_summary", description="Generate text summary", func=generate_summary),
    Tool(name="escalate_to_human", description="Escalate to human review", func=escalate_to_human),
    Tool(name="publish_content", description="Publish approved content", func=publish_content)
]

# Create memory
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

# Create prompt template for content moderation workflow
prompt = ChatPromptTemplate.from_messages([
    SystemMessage(content="""You are a content moderation assistant that processes user-submitted content.

Your workflow should be:
1. Analyze sentiment of the content
2. Check content safety 
3. Based on results, decide the next action:
   - If content is safe and sentiment is positive/neutral: generate summary and publish
   - If content is unsafe: escalate to human review
   - If sentiment is very negative: escalate to human review
   - Otherwise: generate summary and publish

Always use the tools in the correct order and make decisions based on their outputs.
Be very careful about safety and escalate when in doubt."""),
    MessagesPlaceholder(variable_name="chat_history"),
    HumanMessage(content="{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad")
])

# Create agent
agent = create_openai_tools_agent(llm, tools, prompt)
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    memory=memory,
    verbose=True,
    max_iterations=10
)

class LangChainConditionalWorkflow:
    """LangChain-based conditional content moderation workflow."""
    
    def __init__(self):
        self.executor = agent_executor
        self.workflow_state = {
            "processed_content": [],
            "escalations": [],
            "publications": []
        }
    
    def process_content(self, content: str) -> Dict[str, Any]:
        """Process content through the moderation workflow."""
        print(f"\n🔍 Processing content: {content[:50]}...")
        
        workflow_prompt = f"""
        Please process this content through the moderation workflow:
        
        Content: "{content}"
        
        Follow these steps:
        1. Use analyze_sentiment to check the sentiment
        2. Use check_safety to verify safety
        3. Based on the results, decide whether to:
           - Use escalate_to_human if unsafe or very negative
           - Use generate_summary and publish_content if safe
        
        Make your decision based on the tool outputs and explain your reasoning.
        """
        
        try:
            result = self.executor.invoke({
                "input": workflow_prompt
            })
            
            # Parse the result to extract workflow decisions
            output = result["output"]
            workflow_result = self._parse_workflow_result(content, output, result.get("intermediate_steps", []))
            
            # Update internal state
            self._update_workflow_state(workflow_result)
            
            return workflow_result
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "content": content,
                "action": "error"
            }
    
    def _parse_workflow_result(self, content: str, output: str, steps: List) -> Dict[str, Any]:
        """Parse the workflow result from agent output."""
        # Extract tool results from intermediate steps
        sentiment_result = None
        safety_result = None
        action_taken = "unknown"
        action_result = None
        
        for step in steps:
            if hasattr(step, 'tool') and hasattr(step, 'tool_input') and hasattr(step, 'observation'):
                tool_name = step.tool
                observation = step.observation
                
                try:
                    parsed_obs = json.loads(observation)
                except (json.JSONDecodeError, TypeError):
                    parsed_obs = {"raw_output": observation}
                
                if tool_name == "analyze_sentiment":
                    sentiment_result = parsed_obs
                elif tool_name == "check_safety":
                    safety_result = parsed_obs
                elif tool_name in ["escalate_to_human", "publish_content"]:
                    action_taken = tool_name
                    action_result = parsed_obs
        
        # Determine workflow decision
        decision_logic = self._determine_decision_logic(sentiment_result, safety_result)
        
        return {
            "success": True,
            "content": content,
            "sentiment_analysis": sentiment_result,
            "safety_check": safety_result,
            "action": action_taken,
            "action_result": action_result,
            "decision_logic": decision_logic,
            "agent_output": output
        }
    
    def _determine_decision_logic(self, sentiment: Optional[Dict], safety: Optional[Dict]) -> str:
        """Determine the decision logic based on analysis results."""
        if not sentiment or not safety:
            return "insufficient_analysis"
        
        is_safe = safety.get("safe", False)
        sentiment_type = sentiment.get("sentiment", "unknown")
        
        if not is_safe:
            return "escalated_due_to_safety_concerns"
        elif sentiment_type == "negative" and sentiment.get("score", 0) > 2:
            return "escalated_due_to_negative_sentiment"
        else:
            return "approved_for_publication"
    
    def _update_workflow_state(self, result: Dict[str, Any]):
        """Update internal workflow state."""
        self.workflow_state["processed_content"].append(result)
        
        if result["action"] == "escalate_to_human":
            self.workflow_state["escalations"].append(result)
        elif result["action"] == "publish_content":
            self.workflow_state["publications"].append(result)
    
    def get_workflow_stats(self) -> Dict[str, Any]:
        """Get workflow statistics."""
        total_processed = len(self.workflow_state["processed_content"])
        escalations = len(self.workflow_state["escalations"])
        publications = len(self.workflow_state["publications"])
        
        return {
            "total_processed": total_processed,
            "escalations": escalations,
            "publications": publications,
            "escalation_rate": escalations / total_processed if total_processed > 0 else 0,
            "publication_rate": publications / total_processed if total_processed > 0 else 0
        }

def main():
    """Demonstrate LangChain conditional workflow."""
    workflow = LangChainConditionalWorkflow()
    
    # Test content samples
    test_contents = [
        "This is a wonderful product! I highly recommend it to everyone. Great quality and amazing customer service.",
        "This product is terrible and I hate it. Complete waste of money and time. Awful experience.",
        "The product arrived on time and works as expected. Standard quality for the price point.",
        "This content contains explicit violence and harmful content that should not be published.",
        "Looking forward to trying this new feature. Hope it works well for my use case."
    ]
    
    print("🚀 Starting LangChain Conditional Workflow Demo")
    print("="*60)
    
    results = []
    for i, content in enumerate(test_contents, 1):
        print(f"\n📋 Processing Content {i}/{len(test_contents)}")
        result = workflow.process_content(content)
        results.append(result)
        
        # Print result summary
        if result["success"]:
            action = result["action"]
            decision = result["decision_logic"]
            print(f"✅ Action: {action}")
            print(f"🧠 Decision: {decision}")
        else:
            print(f"❌ Error: {result['error']}")
    
    # Print workflow statistics
    print("\n" + "="*60)
    print("📊 WORKFLOW STATISTICS")
    print("="*60)
    
    stats = workflow.get_workflow_stats()
    print(f"📝 Total Processed: {stats['total_processed']}")
    print(f"🚨 Escalations: {stats['escalations']} ({stats['escalation_rate']:.1%})")
    print(f"✅ Publications: {stats['publications']} ({stats['publication_rate']:.1%})")
    
    # Show detailed results
    print(f"\n📋 DETAILED RESULTS")
    print("-" * 40)
    for i, result in enumerate(results, 1):
        if result["success"]:
            sentiment = result.get("sentiment_analysis", {}).get("sentiment", "unknown")
            safety = "safe" if result.get("safety_check", {}).get("safe", False) else "unsafe"
            action = result["action"]
            print(f"{i}. Sentiment: {sentiment}, Safety: {safety} → {action}")

if __name__ == "__main__":
    main()