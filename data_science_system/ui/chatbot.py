"""
Post-Analysis Chatbot with RAG
Answer questions about previous analyses using ChromaDB for memory
"""

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
import os
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger("Chatbot")


class RAGChatbot:
    """Retrieval-Augmented Generation chatbot for data analysis Q&A"""
    
    def __init__(self, persist_directory: str = None, 
                 model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize RAG chatbot
        
        Args:
            persist_directory: Directory for ChromaDB persistence
            model_name: Sentence transformer model for embeddings
        """
        self.persist_directory = persist_directory or "output/chromadb"
        os.makedirs(self.persist_directory, exist_ok=True)
        
        # Initialize embedding model
        self.embedder = SentenceTransformer(model_name)
        logger.info(f"Loaded embedding model: {model_name}")
        
        # Initialize ChromaDB
        self.client = chromadb.PersistentClient(
            path=self.persist_directory,
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Get or create collections
        self.collection = self.client.get_or_create_collection(
            name="analysis_memory",
            metadata={"hnsw:space": "cosine"}
        )
        
        self.insight_collection = self.client.get_or_create_collection(
            name="insights_memory",
            metadata={"hnsw:space": "cosine"}
        )
        
        logger.info("RAG chatbot initialized")
    
    def add_analysis_document(self, analysis_id: str, content: str,
                            metadata: Dict[str, Any] = None):
        """
        Add analysis document to memory
        
        Args:
            analysis_id: Unique identifier for analysis
            content: Document text content
            metadata: Additional metadata
        """
        # Generate embeddings
        embedding = self.embedder.encode(content).tolist()
        
        # Add to collection
        self.collection.add(
            documents=[content],
            embeddings=[embedding],
            metadatas=[metadata or {}],
            ids=[analysis_id]
        )
        
        logger.info(f"Added document: {analysis_id}")
    
    def add_insights(self, analysis_id: str, insights: List[Dict[str, Any]]):
        """
        Add insights to memory
        
        Args:
            analysis_id: Unique identifier for analysis
            insights: List of insight dictionaries
        """
        for i, insight in enumerate(insights):
            # Create searchable text from insight
            content = f"""
            Insight: {insight.get('title', 'Untitled')}
            Type: {insight.get('type', 'unknown')}
            What: {insight.get('what', '')}
            Why: {insight.get('why', '')}
            How: {insight.get('how', '')}
            Recommendation: {insight.get('recommendation', '')}
            """
            
            # Generate embeddings
            embedding = self.embedder.encode(content).tolist()
            
            # Add to insight collection
            self.insight_collection.add(
                documents=[content],
                embeddings=[embedding],
                metadatas={
                    "analysis_id": analysis_id,
                    "insight_id": insight.get('id', f"{analysis_id}_{i}"),
                    "type": insight.get('type', 'unknown'),
                    "title": insight.get('title', '')
                },
                ids=[insight.get('id', f"{analysis_id}_{i}")]
            )
        
        logger.info(f"Added {len(insights)} insights to memory")
    
    def retrieve_relevant_context(self, query: str, top_k: int = 5,
                                collection_name: str = "analysis_memory") -> List[Dict[str, Any]]:
        """
        Retrieve relevant context for a query
        
        Args:
            query: User question
            top_k: Number of top documents to retrieve
            collection_name: Collection to search ('analysis_memory' or 'insights_memory')
        
        Returns:
            List of retrieved documents with metadata
        """
        # Generate query embedding
        query_embedding = self.embedder.encode(query).tolist()
        
        # Select collection
        collection = self.collection if collection_name == "analysis_memory" else self.insight_collection
        
        # Query collection
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )
        
        # Format results
        retrieved_docs = []
        if results['documents'] and results['documents'][0]:
            for i, doc in enumerate(results['documents'][0]):
                retrieved_docs.append({
                    "content": doc,
                    "metadata": results['metadatas'][0][i] if results['metadatas'] else {},
                    "distance": results['distances'][0][i] if results['distances'] else 0
                })
        
        logger.info(f"Retrieved {len(retrieved_docs)} documents for query: {query}")
        return retrieved_docs
    
    def retrieve_insights(self, query: str, top_k: int = 5,
                        insight_type: str = None) -> List[Dict[str, Any]]:
        """
        Retrieve relevant insights for a query
        
        Args:
            query: User question
            top_k: Number of top insights to retrieve
            insight_type: Filter by insight type (optional)
        
        Returns:
            List of retrieved insights
        """
        # Generate query embedding
        query_embedding = self.embedder.encode(query).tolist()
        
        # Build query filter
        where_clause = None
        if insight_type:
            where_clause = {"type": insight_type}
        
        # Query insight collection
        results = self.insight_collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where_clause
        )
        
        # Format results
        retrieved_insights = []
        if results['documents'] and results['documents'][0]:
            for i, doc in enumerate(results['documents'][0]):
                retrieved_insights.append({
                    "content": doc,
                    "metadata": results['metadatas'][0][i] if results['metadatas'] else {},
                    "distance": results['distances'][0][i] if results['distances'] else 0
                })
        
        logger.info(f"Retrieved {len(retrieved_insights)} insights for query: {query}")
        return retrieved_insights
    
    def load_analysis_from_directory(self, analysis_dir: str):
        """
        Load all analysis results from a directory into memory
        
        Args:
            analysis_dir: Path to analysis output directory
        """
        analysis_id = os.path.basename(analysis_dir)
        
        # Load markdown report
        report_path = os.path.join(analysis_dir, f"{analysis_id}_report.md")
        if os.path.exists(report_path):
            with open(report_path, 'r') as f:
                report_content = f.read()
            
            self.add_analysis_document(
                analysis_id=analysis_id,
                content=report_content,
                metadata={
                    "type": "report",
                    "analysis_id": analysis_id,
                    "file_path": report_path
                }
            )
        
        # Load insights
        insights_path = os.path.join(analysis_dir, "insights", "insights.json")
        if os.path.exists(insights_path):
            import json
            with open(insights_path, 'r') as f:
                insights = json.load(f)
            
            self.add_insights(analysis_id=analysis_id, insights=insights)
        
        # Load statistical tests
        tests_path = os.path.join(analysis_dir, "insights", "statistical_tests.json")
        if os.path.exists(tests_path):
            import json
            with open(tests_path, 'r') as f:
                tests = json.load(f)
            
            # Format as document
            tests_content = f"Statistical Tests for {analysis_id}:\n"
            for test in tests:
                tests_content += f"- {test.get('test', 'unknown')}: {test.get('interpretation', '')}\n"
            
            self.add_analysis_document(
                analysis_id=f"{analysis_id}_tests",
                content=tests_content,
                metadata={
                    "type": "statistical_tests",
                    "analysis_id": analysis_id,
                    "file_path": tests_path
                }
            )
        
        # Load modeling results
        models_path = os.path.join(analysis_dir, "insights", "models.json")
        if os.path.exists(models_path):
            import json
            with open(models_path, 'r') as f:
                models = json.load(f)
            
            # Format as document
            models_content = f"Modeling Results for {analysis_id}:\n"
            if 'models' in models:
                for model_name, model_info in models['models'].items():
                    if isinstance(model_info, dict) and 'interpretation' in model_info.get('metrics', {}):
                        models_content += f"- {model_name}: {model_info['metrics']['interpretation']}\n"
            
            self.add_analysis_document(
                analysis_id=f"{analysis_id}_models",
                content=models_content,
                metadata={
                    "type": "models",
                    "analysis_id": analysis_id,
                    "file_path": models_path
                }
            )
        
        logger.info(f"Loaded analysis from: {analysis_dir}")
    
    def load_all_analyses(self, analyses_dir: str = None):
        """
        Load all analyses from the analyses directory
        
        Args:
            analyses_dir: Path to analyses directory
        """
        if analyses_dir is None:
            analyses_dir = "output/analyses"
        
        if not os.path.exists(analyses_dir):
            logger.warning(f"Analyses directory not found: {analyses_dir}")
            return
        
        for analysis_name in os.listdir(analyses_dir):
            analysis_dir = os.path.join(analyses_dir, analysis_name)
            if os.path.isdir(analysis_dir):
                try:
                    self.load_analysis_from_directory(analysis_dir)
                except Exception as e:
                    logger.error(f"Error loading analysis {analysis_name}: {e}")
        
        logger.info("Loaded all analyses from directory")
    
    def generate_response(self, query: str, use_web_search: bool = True,
                        perplexity_api_key: str = None) -> str:
        """
        Generate response to user query using RAG
        
        Args:
            query: User question
            use_web_search: Whether to use Perplexity for real-time research
            perplexity_api_key: Perplexity API key for web search
        
        Returns:
            Generated response
        """
        # Retrieve relevant context
        context_docs = self.retrieve_relevant_context(query, top_k=3)
        insight_docs = self.retrieve_insights(query, top_k=3)
        
        # Build context string
        context = "Relevant Information:\n\n"
        
        if context_docs:
            context += "From Analysis Reports:\n"
            for doc in context_docs:
                context += f"- {doc['content'][:500]}...\n"
            context += "\n"
        
        if insight_docs:
            context += "From Insights:\n"
            for insight in insight_docs:
                context += f"- {insight['metadata'].get('title', 'Untitled')}: {insight['content'][:300]}...\n"
            context += "\n"
        
        # Build prompt
        prompt = f"""You are a data science assistant helping answer questions about data analyses.

{context}

User Question: {query}

Please provide a clear, helpful answer based on the information above. If the information is not available, say so. Use technical terms appropriately but explain them when needed.
"""
        
        # For now, return context-based answer
        # In production, this would call LLM with the prompt
        
        response = f"""Based on the analysis results, here's what I found:

"""
        
        # Add insights to response
        if insight_docs:
            response += "Relevant Insights:\n\n"
            for insight in insight_docs[:3]:
                response += f"**{insight['metadata'].get('title', 'Untitled')}**\n"
                response += f"{insight['content'][:400]}...\n\n"
        else:
            response += "I didn't find specific insights related to your question. "
            response += "You might want to rephrase or ask about different aspects of the analysis.\n\n"
        
        # If web search is enabled, add note
        if use_web_search:
            response += "\n---\n"
            response += "*Note: For real-time context and additional information about trends, I can search the web using Perplexity. "
            response += "This would provide current information to help explain why certain patterns might exist.*\n"
        
        return response
    
    def chat(self, query: str, history: List[Dict[str, str]] = None) -> str:
        """
        Interactive chat interface
        
        Args:
            query: User question
            history: Chat history (list of {role, message} tuples)
        
        Returns:
            Response
        """
        if history is None:
            history = []
        
        # Generate response
        response = self.generate_response(query)
        
        return response
    
    def clear_memory(self):
        """Clear all collections"""
        self.client.delete_collection("analysis_memory")
        self.client.delete_collection("insights_memory")
        
        # Recreate collections
        self.collection = self.client.get_or_create_collection(
            name="analysis_memory",
            metadata={"hnsw:space": "cosine"}
        )
        self.insight_collection = self.client.get_or_create_collection(
            name="insights_memory",
            metadata={"hnsw:space": "cosine"}
        )
        
        logger.info("Memory cleared")
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Get statistics about stored collections
        
        Returns:
            Dictionary with collection statistics
        """
        analysis_count = self.collection.count()
        insight_count = self.insight_collection.count()
        
        return {
            "analysis_documents": analysis_count,
            "insight_documents": insight_count,
            "total_documents": analysis_count + insight_count
        }


class SimpleChatbotUI:
    """Simple command-line UI for the chatbot"""
    
    def __init__(self, chatbot: RAGChatbot):
        """
        Initialize UI
        
        Args:
            chatbot: RAG chatbot instance
        """
        self.chatbot = chatbot
        self.history = []
    
    def run(self):
        """Run interactive chat loop"""
        print("="*60)
        print("Data Science Analysis Chatbot")
        print("="*60)
        print("Type your questions about the analysis.")
        print("Commands:")
        print("  /stats - Show collection statistics")
        print("  /clear - Clear chat history")
        print("  /reload - Reload analyses from directory")
        print("  /quit - Exit")
        print()
        
        while True:
            try:
                query = input("You: ").strip()
                
                if not query:
                    continue
                
                # Handle commands
                if query.lower() == '/quit':
                    print("Goodbye!")
                    break
                elif query.lower() == '/stats':
                    stats = self.chatbot.get_collection_stats()
                    print(f"\nCollection Stats:")
                    print(f"  Analysis Documents: {stats['analysis_documents']}")
                    print(f"  Insight Documents: {stats['insight_documents']}")
                    print(f"  Total: {stats['total_documents']}\n")
                    continue
                elif query.lower() == '/clear':
                    self.history = []
                    print("Chat history cleared.\n")
                    continue
                elif query.lower() == '/reload':
                    print("Reloading analyses...")
                    self.chatbot.load_all_analyses()
                    stats = self.chatbot.get_collection_stats()
                    print(f"Loaded {stats['total_documents']} documents.\n")
                    continue
                
                # Get response
                response = self.chatbot.chat(query, self.history)
                
                # Add to history
                self.history.append({"role": "user", "message": query})
                self.history.append({"role": "assistant", "message": response})
                
                # Print response
                print(f"\nAssistant: {response}\n")
                
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                print(f"\nError: {e}\n")


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Data Science Analysis Chatbot with RAG"
    )
    
    parser.add_argument(
        "--load-dir", "-l",
        help="Directory containing analyses to load"
    )
    
    parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="Run interactive chat mode"
    )
    
    args = parser.parse_args()
    
    # Initialize chatbot
    chatbot = RAGChatbot()
    
    # Load analyses
    if args.load_dir:
        chatbot.load_analysis_from_directory(args.load_dir)
    else:
        chatbot.load_all_analyses()
    
    # Print stats
    stats = chatbot.get_collection_stats()
    print(f"Loaded {stats['total_documents']} documents into memory.")
    print()
    
    # Run interactive mode
    if args.interactive:
        ui = SimpleChatbotUI(chatbot)
        ui.run()
    else:
        print("Use --interactive flag to start chat mode.")


if __name__ == "__main__":
    main()
