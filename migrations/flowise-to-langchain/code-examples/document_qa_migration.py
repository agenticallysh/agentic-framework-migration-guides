#!/usr/bin/env python3
"""
Flowise → LangChain Migration: Document Q&A Example
This example shows how to migrate a document Q&A flow from Flowise to LangChain.

Flowise Flow: Document Loader → Text Splitter → Vector Store → Retriever → QA Chain
LangChain Equivalent: Complete RAG pipeline with enhanced capabilities
"""

import os
from typing import List, Dict, Any
from langchain.document_loaders import PyPDFLoader, TextLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain.schema import Document

# Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

class DocumentQAMigration:
    """
    Migrated Document Q&A system from Flowise visual flow
    Provides enhanced capabilities beyond what's possible in Flowise
    """
    
    def __init__(self, 
                 model="gpt-4", 
                 chunk_size=1000, 
                 chunk_overlap=200,
                 k_documents=3):
        """
        Initialize the Document Q&A system
        
        Args:
            model: OpenAI model (same as Flowise LLM node)
            chunk_size: Text splitter chunk size (same as Flowise Text Splitter)
            chunk_overlap: Overlap between chunks
            k_documents: Number of documents to retrieve
        """
        self.llm = ChatOpenAI(
            api_key=OPENAI_API_KEY,
            model=model,
            temperature=0.1  # Lower for factual responses
        )
        
        # Text splitter configuration (replaces Flowise Text Splitter node)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""]  # Enhanced splitting
        )
        
        # Embeddings (same as Flowise vector store)
        self.embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
        
        # Configuration
        self.k_documents = k_documents
        self.vectorstore = None
        self.qa_chain = None
    
    def load_documents(self, file_path: str) -> List[Document]:
        """
        Load documents (replaces Flowise Document Loader node)
        Enhanced with multiple file type support
        """
        file_extension = file_path.lower().split('.')[-1]
        
        try:
            if file_extension == 'pdf':
                loader = PyPDFLoader(file_path)
            elif file_extension in ['txt', 'md']:
                loader = TextLoader(file_path)
            else:
                raise ValueError(f"Unsupported file type: {file_extension}")
            
            documents = loader.load()
            print(f"✅ Loaded {len(documents)} documents from {file_path}")
            return documents
            
        except Exception as e:
            print(f"❌ Error loading documents: {str(e)}")
            return []
    
    def load_directory(self, directory_path: str) -> List[Document]:
        """
        Load multiple documents from directory
        Enhanced capability not available in Flowise
        """
        try:
            loader = DirectoryLoader(
                directory_path,
                glob="**/*.{pdf,txt,md}",
                show_progress=True
            )
            documents = loader.load()
            print(f"✅ Loaded {len(documents)} documents from {directory_path}")
            return documents
            
        except Exception as e:
            print(f"❌ Error loading directory: {str(e)}")
            return []
    
    def process_documents(self, documents: List[Document]) -> None:
        """
        Process documents and create vector store
        (replaces Flowise Vector Store node)
        """
        if not documents:
            print("❌ No documents to process")
            return
        
        # Split documents (replaces Flowise Text Splitter)
        splits = self.text_splitter.split_documents(documents)
        print(f"📄 Split into {len(splits)} chunks")
        
        # Create vector store (replaces Flowise Vector Store node)
        self.vectorstore = Chroma.from_documents(
            documents=splits,
            embedding=self.embeddings,
            collection_name="migrated_docs"
        )
        print(f"🗂️ Created vector store with {len(splits)} embeddings")
        
        # Create retriever (replaces Flowise Retriever node)
        retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": self.k_documents}
        )
        
        # Create QA chain (replaces Flowise QA Chain node)
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",  # Can be customized
            retriever=retriever,
            return_source_documents=True,
            verbose=True
        )
        
        print("✅ Document Q&A system ready")
    
    def ask_question(self, question: str) -> Dict[str, Any]:
        """
        Ask a question about the documents
        Enhanced with source document tracking
        """
        if not self.qa_chain:
            return {
                "error": "No documents processed. Call process_documents() first."
            }
        
        try:
            result = self.qa_chain({"query": question})
            
            # Enhanced response with metadata
            response = {
                "question": question,
                "answer": result["result"],
                "source_documents": [
                    {
                        "content": doc.page_content[:200] + "...",
                        "metadata": doc.metadata
                    }
                    for doc in result["source_documents"]
                ],
                "confidence": self._calculate_confidence(result)
            }
            
            return response
            
        except Exception as e:
            return {"error": f"Error processing question: {str(e)}"}
    
    def _calculate_confidence(self, result: Dict) -> str:
        """
        Calculate confidence based on source document relevance
        Enhanced feature not available in Flowise
        """
        # Simple confidence calculation based on number of sources
        num_sources = len(result.get("source_documents", []))
        if num_sources >= 3:
            return "High"
        elif num_sources >= 2:
            return "Medium"
        else:
            return "Low"
    
    def get_similar_documents(self, query: str, k: int = 5) -> List[Dict]:
        """
        Get similar documents without Q&A
        Additional capability beyond Flowise
        """
        if not self.vectorstore:
            return []
        
        similar_docs = self.vectorstore.similarity_search(query, k=k)
        
        return [
            {
                "content": doc.page_content,
                "metadata": doc.metadata,
                "relevance_score": "High"  # Could be enhanced with actual scores
            }
            for doc in similar_docs
        ]

class AdvancedDocumentQA(DocumentQAMigration):
    """
    Advanced Document Q&A with additional features
    Beyond what's possible in Flowise visual interface
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conversation_history = []
    
    def conversational_qa(self, question: str) -> Dict[str, Any]:
        """
        Conversational Q&A that remembers context
        """
        # Add conversation context
        context = ""
        if self.conversation_history:
            recent_context = self.conversation_history[-3:]  # Last 3 exchanges
            context = "Previous conversation:\n"
            for qa in recent_context:
                context += f"Q: {qa['question']}\nA: {qa['answer']}\n\n"
        
        # Enhanced question with context
        enhanced_question = f"{context}Current question: {question}"
        
        # Get response
        result = self.ask_question(enhanced_question)
        
        # Store in history
        if "answer" in result:
            self.conversation_history.append({
                "question": question,
                "answer": result["answer"]
            })
        
        return result
    
    def summarize_documents(self) -> str:
        """
        Summarize all processed documents
        Advanced feature beyond Flowise capabilities
        """
        if not self.qa_chain:
            return "No documents processed."
        
        summary_question = """
        Please provide a comprehensive summary of all the documents, 
        highlighting the main topics, key points, and important information.
        """
        
        result = self.ask_question(summary_question)
        return result.get("answer", "Unable to generate summary")

def demonstrate_migration():
    """
    Demonstrate the Document Q&A migration
    """
    print("🔄 Flowise → LangChain Document Q&A Migration Demo")
    print("=" * 60)
    
    # Create migrated Q&A system
    qa_system = DocumentQAMigration()
    
    # Demonstrate with a sample text file
    sample_text = """
    Artificial Intelligence (AI) is a branch of computer science that aims to create 
    intelligent machines that can think and learn like humans. Machine Learning is a 
    subset of AI that enables computers to learn and improve from experience without 
    being explicitly programmed. Deep Learning is a subset of Machine Learning that 
    uses neural networks with multiple layers to analyze data.
    
    Natural Language Processing (NLP) is a field of AI that focuses on the interaction 
    between computers and human language. It enables machines to understand, interpret, 
    and generate human language in a valuable way.
    """
    
    # Create sample document
    sample_doc = Document(
        page_content=sample_text,
        metadata={"source": "ai_overview.txt", "type": "educational"}
    )
    
    print("\n📄 Processing sample document...")
    qa_system.process_documents([sample_doc])
    
    # Test questions
    test_questions = [
        "What is Artificial Intelligence?",
        "How does Machine Learning relate to AI?",
        "What is the difference between ML and Deep Learning?",
        "Explain Natural Language Processing"
    ]
    
    print("\n❓ Testing Q&A functionality:")
    for i, question in enumerate(test_questions, 1):
        print(f"\n{i}. Question: {question}")
        result = qa_system.ask_question(question)
        
        if "answer" in result:
            print(f"   Answer: {result['answer']}")
            print(f"   Confidence: {result['confidence']}")
            print(f"   Sources: {len(result['source_documents'])} documents")
        else:
            print(f"   Error: {result.get('error')}")
    
    print("\n🔍 Testing similarity search:")
    similar_docs = qa_system.get_similar_documents("neural networks", k=2)
    for i, doc in enumerate(similar_docs, 1):
        print(f"{i}. {doc['content'][:100]}...")
    
    print("\n🧠 Testing advanced conversational Q&A:")
    advanced_qa = AdvancedDocumentQA()
    advanced_qa.process_documents([sample_doc])
    
    # Conversational questions
    conv_questions = [
        "What is AI?",
        "How does it relate to machine learning?",
        "Can you explain more about the neural networks part?"
    ]
    
    for question in conv_questions:
        print(f"\nQ: {question}")
        result = advanced_qa.conversational_qa(question)
        if "answer" in result:
            print(f"A: {result['answer']}")

if __name__ == "__main__":
    if not OPENAI_API_KEY:
        print("❌ Please set OPENAI_API_KEY environment variable")
        print("export OPENAI_API_KEY='your-api-key-here'")
    else:
        demonstrate_migration()