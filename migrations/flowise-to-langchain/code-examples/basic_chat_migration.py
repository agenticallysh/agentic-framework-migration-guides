#!/usr/bin/env python3
"""
Flowise → LangChain Migration: Basic Chat Example
This example shows how to migrate a simple chat flow from Flowise to LangChain.

Flowise Flow: LLM Node → Memory Node → Output
LangChain Equivalent: ConversationChain with ConversationBufferMemory
"""

import os
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory, ConversationSummaryMemory
from langchain_openai import ChatOpenAI

# Configuration (same API keys as Flowise)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

class BasicChatMigration:
    """
    Migrated basic chat functionality from Flowise visual flow
    """
    
    def __init__(self, model="gpt-4", temperature=0.7):
        """
        Initialize the chat system
        
        Args:
            model: OpenAI model name (same as configured in Flowise)
            temperature: LLM temperature (same as Flowise LLM node setting)
        """
        self.llm = ChatOpenAI(
            api_key=OPENAI_API_KEY,
            model=model,
            temperature=temperature
        )
        
        # Basic memory (replaces Flowise Memory node)
        self.memory = ConversationBufferMemory()
        
        # Create conversation chain (replaces Flowise visual flow)
        self.conversation = ConversationChain(
            llm=self.llm,
            memory=self.memory,
            verbose=True  # Enable for debugging
        )
    
    def chat(self, message):
        """
        Process a chat message (replaces Flowise output)
        
        Args:
            message: User input message
            
        Returns:
            AI response
        """
        try:
            response = self.conversation.predict(input=message)
            return response
        except Exception as e:
            return f"Error: {str(e)}"
    
    def get_conversation_history(self):
        """
        Get conversation history (similar to Flowise memory viewer)
        """
        return self.memory.buffer
    
    def clear_memory(self):
        """
        Clear conversation memory (reset session)
        """
        self.memory.clear()

class AdvancedChatMigration:
    """
    Advanced chat with summary memory for longer conversations
    """
    
    def __init__(self, model="gpt-4", max_buffer_size=10):
        self.llm = ChatOpenAI(
            api_key=OPENAI_API_KEY,
            model=model,
            temperature=0.7
        )
        
        # Advanced memory management
        self.buffer_memory = ConversationBufferMemory()
        self.summary_memory = ConversationSummaryMemory(llm=self.llm)
        self.max_buffer_size = max_buffer_size
        self.message_count = 0
        
        self.conversation = ConversationChain(
            llm=self.llm,
            memory=self.buffer_memory,
            verbose=True
        )
    
    def chat(self, message):
        """
        Advanced chat with automatic memory management
        """
        # Process message
        response = self.conversation.predict(input=message)
        self.message_count += 1
        
        # Manage memory size
        if self.message_count > self.max_buffer_size:
            self._compress_memory()
        
        return response
    
    def _compress_memory(self):
        """
        Compress old memories into summary
        """
        # Get current conversation
        current_buffer = self.buffer_memory.buffer
        
        # Save to summary memory
        self.summary_memory.save_context(
            {"input": "Previous conversation"},
            {"output": current_buffer}
        )
        
        # Clear buffer but keep recent messages
        self.buffer_memory.clear()
        self.message_count = 0

def demonstrate_migration():
    """
    Demonstrate the migration from Flowise to LangChain
    """
    print("🔄 Flowise → LangChain Migration Demo")
    print("=" * 50)
    
    # Create migrated chat system
    chat = BasicChatMigration()
    
    # Test conversations (same as you would test in Flowise)
    test_messages = [
        "Hello, I'm testing the migration",
        "Can you remember what I just said?",
        "What's the capital of France?",
        "Do you remember my first message?"
    ]
    
    print("\n📝 Testing conversation flow:")
    for i, message in enumerate(test_messages, 1):
        print(f"\n{i}. User: {message}")
        response = chat.chat(message)
        print(f"   AI: {response}")
    
    print(f"\n💭 Conversation History:")
    print(chat.get_conversation_history())
    
    print(f"\n🧪 Testing advanced memory management:")
    advanced_chat = AdvancedChatMigration(max_buffer_size=2)
    
    for i in range(5):
        message = f"This is message number {i+1}"
        response = advanced_chat.chat(message)
        print(f"Message {i+1}: {response[:50]}...")

if __name__ == "__main__":
    # Ensure API key is set
    if not OPENAI_API_KEY:
        print("❌ Please set OPENAI_API_KEY environment variable")
        print("export OPENAI_API_KEY='your-api-key-here'")
    else:
        demonstrate_migration()