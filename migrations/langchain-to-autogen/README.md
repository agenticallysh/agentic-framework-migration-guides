# LangChain → AutoGen Migration Guide

[![Migration Complexity](https://img.shields.io/badge/Complexity-Medium--High-red.svg)]()
[![Estimated Time](https://img.shields.io/badge/Time-6--12%20hours-blue.svg)]()
[![Success Rate](https://img.shields.io/badge/Success%20Rate-89%25-green.svg)]()

Complete guide for migrating from LangChain to AutoGen. This migration typically results in **advanced conversation capabilities**, **better multi-agent coordination**, and more sophisticated agent interaction patterns.

[🚀 Start migration with our interactive tool →](https://www.agentically.sh/ai-agentic-frameworks/migrate/langchain-to-autogen/)

## 📋 Table of Contents

- [Why Migrate to AutoGen?](#why-migrate-to-autogen)
- [Before You Start](#before-you-start)
- [Migration Overview](#migration-overview)
- [Step-by-Step Migration](#step-by-step-migration)
- [Code Examples](#code-examples)
- [Common Issues](#common-issues)
- [Testing & Validation](#testing--validation)
- [Post-Migration Optimization](#post-migration-optimization)
- [Automated Tools](#automated-tools)

## 🎯 Why Migrate to AutoGen?

### ✅ Benefits of AutoGen
- **Advanced Multi-Agent Conversations**: Sophisticated agent-to-agent communication patterns
- **Research-Grade Capabilities**: Built for complex problem-solving and reasoning
- **Flexible Agent Architectures**: Easy to create specialized agent roles and hierarchies
- **Human-in-the-Loop**: Native support for human intervention and guidance
- **Code Execution**: Built-in code generation and execution capabilities
- **Conversation Management**: Advanced conversation flow control and termination
- **Group Chat Support**: Multi-agent group conversations with dynamic participant management

### ⚠️ What You'll Lose
- **Simpler Chain Abstractions**: LangChain's sequential chains are more straightforward
- **Ecosystem Integration**: LangChain has broader tool and integration ecosystem
- **Learning Curve**: AutoGen requires understanding of agent conversation patterns
- **Resource Usage**: Multi-agent conversations can be more resource-intensive

### 🎯 Best Candidates for Migration
- ✅ Multi-agent systems requiring sophisticated interactions
- ✅ Research and development workflows
- ✅ Complex problem-solving scenarios
- ✅ Code generation and execution workflows
- ✅ Human-AI collaborative systems
- ❌ Simple single-agent chains
- ❌ Heavy dependency on LangChain-specific tools
- ❌ Systems requiring minimal agent interaction

## 🛠️ Before You Start

### Prerequisites
- Python 3.8+ installed
- OpenAI API key or compatible LLM access
- Understanding of agent-based architectures
- Familiarity with conversation patterns
- Basic understanding of code execution security

### Project Assessment
Run our automated assessment tool:
```bash
python ../automated-tools/migration_analyzer.py /path/to/your/project
```

### Environment Setup
```bash
# Install AutoGen
pip install pyautogen

# Install additional dependencies
pip install docker  # For code execution
pip install jupyter  # For notebook examples

# Verify installation
python -c "import autogen; print('AutoGen installed successfully')"
```

## 📊 Migration Overview

### Architecture Comparison

| Aspect | LangChain | AutoGen |
|--------|-----------|---------|
| **Primary Pattern** | Chains and agents | Agent conversations |
| **Agent Communication** | Through tools/callbacks | Direct conversation |
| **Execution Model** | Sequential/parallel chains | Conversation rounds |
| **State Management** | Memory objects | Conversation history |
| **Code Execution** | External tools | Built-in execution |
| **Human Interaction** | Manual intervention | Native HITL support |

### Key Conceptual Changes

1. **Chains → Conversations**: Replace chain logic with agent conversations
2. **Tools → Functions**: Convert tools to function calling capabilities
3. **Memory → History**: Use conversation history for context
4. **Executors → Chat Managers**: Replace executors with group chat managers
5. **Callbacks → Event Handlers**: Implement custom event handling

## 🔄 Step-by-Step Migration

### Step 1: Design Agent Architecture

**Before (LangChain)**:
```python
# Single agent with tools
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    memory=memory
)
```

**After (AutoGen)**:
```python
# Multiple specialized agents
user_proxy = autogen.UserProxyAgent(
    name="user_proxy",
    human_input_mode="TERMINATE",
    max_consecutive_auto_reply=10
)

assistant = autogen.AssistantAgent(
    name="assistant", 
    llm_config={"config_list": config_list}
)
```

### Step 2: Convert Tools to Functions

**Before (LangChain)**:
```python
def search_tool(query: str) -> str:
    return web_search(query)

tools = [
    Tool(name="search", description="Search the web", func=search_tool)
]
```

**After (AutoGen)**:
```python
def search_function(query: str) -> str:
    """Search the web for information."""
    return web_search(query)

# Register function with assistant
assistant.register_function(
    function_map={"search_web": search_function}
)
```

### Step 3: Replace Chains with Conversations

**Before (LangChain)**:
```python
# Sequential chain execution
chain = LLMChain(llm=llm, prompt=prompt)
result = chain.run(input_text)
```

**After (AutoGen)**:
```python
# Conversation-based execution
user_proxy.initiate_chat(
    assistant,
    message=input_text
)
```

### Step 4: Implement Group Chat (if needed)

**Before (LangChain)**:
```python
# Multiple agents via sequential execution
agent1_result = agent1.run(input)
agent2_result = agent2.run(agent1_result)
```

**After (AutoGen)**:
```python
# Group chat with multiple agents
groupchat = autogen.GroupChat(
    agents=[user_proxy, researcher, writer, reviewer],
    messages=[],
    max_round=10
)

manager = autogen.GroupChatManager(
    groupchat=groupchat,
    llm_config={"config_list": config_list}
)

user_proxy.initiate_chat(manager, message="Start the research process")
```

### Step 5: Handle Code Execution

**Before (LangChain)**:
```python
# External code execution tool
def execute_code(code: str) -> str:
    # Custom execution logic
    return subprocess.run(code, capture_output=True)
```

**After (AutoGen)**:
```python
# Built-in code execution
code_executor = autogen.UserProxyAgent(
    name="code_executor",
    human_input_mode="NEVER",
    max_consecutive_auto_reply=1,
    code_execution_config={
        "work_dir": "coding",
        "use_docker": True
    }
)
```

## 💻 Code Examples

### Basic Single Agent Migration

**LangChain Version**:
```python
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.tools import Tool
from langchain.chat_models import ChatOpenAI

llm = ChatOpenAI(model="gpt-4")

def calculator(expression: str) -> str:
    try:
        return str(eval(expression))
    except:
        return "Invalid expression"

tools = [Tool(name="calculator", description="Calculate math expressions", func=calculator)]
agent = create_openai_tools_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools)

result = agent_executor.invoke({"input": "What is 15 * 23 + 47?"})
```

**AutoGen Version**:
```python
import autogen

config_list = [{"model": "gpt-4", "api_key": "your-api-key"}]

def calculator(expression: str) -> str:
    """Calculate mathematical expressions."""
    try:
        return str(eval(expression))
    except:
        return "Invalid expression"

assistant = autogen.AssistantAgent(
    name="math_assistant",
    llm_config={"config_list": config_list},
    system_message="You are a helpful math assistant. Use the calculator function for computations."
)

user_proxy = autogen.UserProxyAgent(
    name="user_proxy",
    human_input_mode="TERMINATE",
    max_consecutive_auto_reply=10,
    function_map={"calculator": calculator}
)

user_proxy.initiate_chat(assistant, message="What is 15 * 23 + 47?")
```

### Multi-Agent Conversation Migration

**LangChain Version (Sequential)**:
```python
# Research → Write → Review chain
research_agent = create_agent("researcher", research_tools)
writer_agent = create_agent("writer", writing_tools) 
reviewer_agent = create_agent("reviewer", review_tools)

research_result = research_agent.run("Research AI trends")
draft = writer_agent.run(f"Write article based on: {research_result}")
final_article = reviewer_agent.run(f"Review and improve: {draft}")
```

**AutoGen Version (Conversational)**:
```python
researcher = autogen.AssistantAgent(
    name="researcher",
    llm_config={"config_list": config_list},
    system_message="You are a research specialist. Gather comprehensive information on topics."
)

writer = autogen.AssistantAgent(
    name="writer", 
    llm_config={"config_list": config_list},
    system_message="You are a skilled writer. Create engaging articles from research."
)

reviewer = autogen.AssistantAgent(
    name="reviewer",
    llm_config={"config_list": config_list}, 
    system_message="You are an editor. Review and improve written content."
)

user_proxy = autogen.UserProxyAgent(
    name="user_proxy",
    human_input_mode="TERMINATE",
    max_consecutive_auto_reply=1
)

# Group chat for collaborative work
groupchat = autogen.GroupChat(
    agents=[user_proxy, researcher, writer, reviewer],
    messages=[],
    max_round=15,
    speaker_selection_method="round_robin"
)

manager = autogen.GroupChatManager(
    groupchat=groupchat,
    llm_config={"config_list": config_list}
)

user_proxy.initiate_chat(
    manager, 
    message="Create a comprehensive article about AI trends in 2024"
)
```

## ⚠️ Common Issues

### 1. Conversation Loops
**Problem**: Agents get stuck in infinite conversation loops
**Solution**: 
```python
# Set proper termination conditions
user_proxy = autogen.UserProxyAgent(
    name="user_proxy",
    is_termination_msg=lambda x: x.get("content", "").rstrip().endswith("TERMINATE"),
    max_consecutive_auto_reply=5
)
```

### 2. Function Registration Issues
**Problem**: Functions not properly available to agents
**Solution**:
```python
# Ensure function_map is properly configured
user_proxy = autogen.UserProxyAgent(
    name="user_proxy",
    function_map={
        "function_name": function_implementation,
        # Add all functions here
    }
)
```

### 3. Code Execution Security
**Problem**: Unsafe code execution in production
**Solution**:
```python
# Use Docker for safe code execution
code_execution_config = {
    "work_dir": "coding",
    "use_docker": True,
    "timeout": 60,
    "last_n_messages": 2
}
```

### 4. Group Chat Management
**Problem**: Conversations become chaotic with multiple agents
**Solution**:
```python
# Use structured speaker selection
groupchat = autogen.GroupChat(
    agents=[user_proxy, agent1, agent2],
    messages=[],
    max_round=10,
    speaker_selection_method="manual",  # or "round_robin"
    allow_repeat_speaker=False
)
```

## 🧪 Testing & Validation

### Conversation Testing
```python
def test_conversation_flow():
    """Test agent conversation produces expected results."""
    user_proxy.initiate_chat(
        assistant,
        message="Test message",
        silent=True  # Suppress output during testing
    )
    
    # Validate conversation history
    assert len(user_proxy.chat_messages) > 0
    assert "expected_content" in user_proxy.last_message()["content"]
```

### Function Testing
```python
def test_function_execution():
    """Test registered functions work correctly."""
    result = assistant.execute_function("calculator", "2+2")
    assert result == "4"
```

### Group Chat Testing
```python
def test_group_collaboration():
    """Test multi-agent collaboration."""
    initial_message = "Solve this complex problem together"
    
    user_proxy.initiate_chat(manager, message=initial_message)
    
    # Validate each agent contributed
    messages = groupchat.messages
    agent_names = [msg.get("name") for msg in messages]
    
    assert "researcher" in agent_names
    assert "writer" in agent_names
    assert "reviewer" in agent_names
```

## 🎛️ Post-Migration Optimization

### Performance Optimization
```python
# Optimize model usage
config_list = [
    {
        "model": "gpt-4",
        "api_key": "your-key",
        "max_tokens": 500,  # Limit token usage
        "temperature": 0.3   # More deterministic
    }
]

# Use caching for repeated conversations
autogen.cache.Cache.set_cache_path(".cache")
```

### Conversation Flow Optimization
```python
# Implement custom speaker selection
def custom_speaker_selection(last_speaker, groupchat):
    """Select next speaker based on conversation context."""
    if "research" in groupchat.messages[-1]["content"].lower():
        return "researcher"
    elif "write" in groupchat.messages[-1]["content"].lower():
        return "writer"
    else:
        return "reviewer"

groupchat.speaker_selection_method = custom_speaker_selection
```

### Memory Management
```python
# Implement conversation summarization
def summarize_conversation(messages):
    """Summarize long conversations to save context."""
    if len(messages) > 20:
        # Use LLM to summarize early messages
        summary = llm.summarize([msg["content"] for msg in messages[:-10]])
        return [{"role": "system", "content": f"Previous conversation summary: {summary}"}] + messages[-10:]
    return messages
```

## 🛠️ Automated Tools

### Migration Assessment
```bash
# Analyze your LangChain project
python automated-tools/langchain_to_autogen_analyzer.py /path/to/project

# Generate migration recommendations
python automated-tools/autogen_migration_planner.py --project /path/to/project
```

### Conversion Helpers
```bash
# Convert LangChain agents to AutoGen format
python automated-tools/agent_converter.py --input langchain_agent.py --output autogen_agent.py

# Validate AutoGen conversation patterns
python automated-tools/conversation_validator.py --config autogen_config.json
```

## 📚 Additional Resources

### AutoGen Documentation
- [AutoGen GitHub Repository](https://github.com/microsoft/autogen)
- [AutoGen Documentation](https://microsoft.github.io/autogen/)
- [Multi-Agent Conversation Patterns](https://microsoft.github.io/autogen/docs/notebooks/)

### Migration Examples
- [Basic Agent Migration](./code-examples/basic-migration/)
- [Multi-Agent Workflows](./code-examples/multi-agent/)
- [Code Execution Examples](./code-examples/code-execution/)
- [Human-in-the-Loop Examples](./code-examples/human-in-loop/)

### Community Resources
- [AutoGen Discord Community](https://discord.gg/autogen)
- [Migration Best Practices](../../docs/autogen-best-practices.md)
- [Performance Optimization Guide](../../docs/autogen-performance.md)

## 🔗 Related Migrations

- [LangChain → CrewAI](../langchain-to-crewai/) - For role-based multi-agent systems
- [LangChain → LangGraph](../langchain-to-langgraph/) - For stateful workflows
- [OpenAI Assistants → AutoGen](../openai-assistants-to-autogen/) - From assistants to conversations

---

**Ready to start your migration?** Use our [interactive migration tool](https://www.agentically.sh/ai-agentic-frameworks/migrate/langchain-to-autogen/) or follow the step-by-step guide above.

For questions and support, join our [community Discord](https://discord.gg/agentically) or check the [GitHub discussions](https://github.com/agenticallysh/agentic-framework-migration-guides/discussions).