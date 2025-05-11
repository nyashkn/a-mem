import unittest
import os
import logging
import json
from amem.memory_system import AgenticMemorySystem, MemoryNote

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestAgenticMemoryWithCloudServices(unittest.TestCase):
    """Test the Agentic Memory System with actual cloud service calls.
    
    This test suite makes actual calls to:
    - LiteLLM for embeddings (AWS Bedrock with Cohere)
    - OpenRouter for LLM (Meta LLaMA)
    - Qdrant for vector storage
    
    Note: These tests require:
    1. Environment variables or .env file to be properly set up
    2. Docker to be running with Qdrant container started
    3. Internet connectivity for cloud service calls
    """
    
    @classmethod
    def setUpClass(cls):
        """Set up resources before any tests run"""
        # Check if Docker/Qdrant is running
        import requests
        try:
            # Use port from environment variable or default to 7333
            from os import environ
            port = int(environ.get("QDRANT_PORT", "7333"))
            host = environ.get("QDRANT_HOST", "localhost")
            response = requests.get(f"http://{host}:{port}/healthz")
            if response.status_code != 200:
                logger.warning("Qdrant might not be running properly. Tests may fail.")
        except Exception as e:
            logger.warning(f"Could not connect to Qdrant: {e}")
            logger.warning("Make sure to start Qdrant using 'docker-compose up -d'")
    
    def setUp(self):
        """Create a fresh memory system before each test"""
        # Initialize with default configuration (from .env)
        self.memory_system = AgenticMemorySystem()
        
        # Test data
        self.test_memories = [
            "Machine learning is a field of inquiry devoted to understanding and building methods that 'learn' from data to improve performance on some set of tasks.",
            "Deep learning is part of a broader family of machine learning methods based on artificial neural networks with representation learning.",
            "Natural language processing (NLP) is a subfield of linguistics, computer science, and artificial intelligence concerned with the interactions between computers and human language."
        ]
    
    def test_embedding_functionality(self):
        """Test that embedding provider is working with actual API calls"""
        # Get embedding for test text
        test_text = "This is a test for embedding functionality"
        
        # Direct embedding test
        embedding = self.memory_system.embedding_provider.get_embedding(test_text)
        
        # Verify embedding shape and type
        self.assertIsInstance(embedding, list)
        self.assertTrue(len(embedding) > 0, "Embedding should have non-zero length")
        self.assertIsInstance(embedding[0], float)
        
        logger.info(f"Embedding vector size: {len(embedding)}")
    
    def test_llm_functionality(self):
        """Test that LLM controller is working with actual API calls"""
        # Test simple prompt
        prompt = "Generate a one-sentence summary of machine learning"
        
        response = self.memory_system.llm_controller.get_completion(prompt)
        
        # Basic verification
        self.assertIsInstance(response, str)
        self.assertTrue(len(response) > 10, "Response should be meaningful")
        
        logger.info(f"LLM response: {response}")
    
    def test_content_analysis(self):
        """Test content analysis with LLM for metadata extraction"""
        # Text to analyze
        content = "Machine learning is revolutionizing artificial intelligence applications across many industries."
        
        # Get analysis
        analysis = self.memory_system.analyze_content(content)
        
        # Verify structure
        self.assertIn("keywords", analysis)
        self.assertIn("context", analysis)
        self.assertIn("tags", analysis)
        
        logger.info(f"Content analysis: {json.dumps(analysis, indent=2)}")
    
    def test_end_to_end_memory_flow(self):
        """Test complete workflow: add notes, search, and evolve memories"""
        # Add test memories
        memory_ids = []
        for content in self.test_memories:
            memory_id = self.memory_system.add_note(content)
            memory_ids.append(memory_id)
            logger.info(f"Added memory: {memory_id}")
        
        # Check memory count
        self.assertEqual(len(self.memory_system.memories), len(self.test_memories))
        
        # Search memories
        search_results = self.memory_system.search_agentic("artificial intelligence", k=2)
        self.assertTrue(len(search_results) > 0, "Should find results for 'artificial intelligence'")
        logger.info(f"Search returned {len(search_results)} results")
        
        # Add a related memory to trigger evolution
        evolution_content = "Reinforcement learning is an area of machine learning concerned with how intelligent agents ought to take actions in an environment to maximize cumulative reward."
        evolution_id = self.memory_system.add_note(evolution_content)
        
        # Get the evolved memory and check its properties
        evolved_memory = self.memory_system.read(evolution_id)
        logger.info(f"Evolved memory tags: {evolved_memory.tags}")
        logger.info(f"Evolved memory links: {evolved_memory.links}")
        
        # Clean up - delete all memories
        for memory_id in memory_ids + [evolution_id]:
            self.memory_system.delete(memory_id)
    
    def test_vector_db_storage(self):
        """Test that Qdrant is storing and retrieving vectors properly"""
        # Add a sample memory
        content = "This is a test memory for vector database functionality."
        memory_id = self.memory_system.add_note(content)
        
        # Retrieve the memory directly from DB
        retrieval_results = self.memory_system.retriever.search(content, k=1)
        
        # Verify results
        self.assertTrue(len(retrieval_results['ids']) > 0)
        self.assertEqual(retrieval_results['ids'][0][0], memory_id)
        
        # Clean up
        self.memory_system.delete(memory_id)


if __name__ == '__main__':
    unittest.main()
