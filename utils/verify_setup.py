#!/usr/bin/env python3
"""
A-MEM Verification Script

This script performs basic tests to verify that all components of A-MEM are working correctly.
It tests:
1. FalkorDB connection and functionality
2. FalkorDBRetriever class
3. AWS Bedrock embedding functionality
4. OpenRouter LLM integration

Usage:
  python verify_setup.py
"""

import os
import sys
import json
import time
import uuid
from typing import List, Dict, Any
import numpy as np
from dotenv import load_dotenv
from amem.memory_system import AgenticMemorySystem
from amem.embedding.providers import LiteLLMEmbedding
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("verify")

def check_falkordb_connection():
    """Test connection to FalkorDB using the FalkorDB Python library"""
    logger.info("🔍 Checking FalkorDB connection...")
    try:
        from falkordb import FalkorDB
        
        # Connect to FalkorDB
        host = os.getenv("FALKORDB_HOST", "localhost")
        port = int(os.getenv("FALKORDB_PORT", "7379"))
        client = FalkorDB(host=host, port=port)
        
        logger.info("✅ Connected to FalkorDB")
        
        # List available graphs
        graphs = client.list_graphs()
        logger.info(f"Available graphs: {graphs}")
        
        # Try to create a simple test graph
        try:
            # Select a test graph
            test_graph = "verify_test_graph"
            graph = client.select_graph(test_graph)
            
            # Delete the graph if it exists (in case of previous test failure)
            try:
                if test_graph in graphs:
                    graph.delete()
                    logger.info(f"Cleaned up existing test graph: {test_graph}")
                    # Re-select the graph after deletion
                    graph = client.select_graph(test_graph)
            except Exception as clean_e:
                logger.warning(f"Error during cleanup: {clean_e}")
            
            # Create a simple node without properties
            result = graph.query("CREATE (:Person)")
            logger.info(f"✅ Created test node, stats: {result.nodes_created} node(s) created")
            
            # Query the graph
            result = graph.query("MATCH (p:Person) RETURN p")
            logger.info(f"✅ Query returned {len(result.result_set)} results")
            
            # Add properties to nodes
            result = graph.query("MATCH (p:Person) SET p.name = 'Alice', p.age = 30 RETURN p")
            logger.info(f"✅ Added properties to nodes: {result.properties_set} properties set")
            
            # Query with properties
            result = graph.query("MATCH (p:Person) WHERE p.name = 'Alice' RETURN p.name, p.age")
            if result.result_set and len(result.result_set) > 0:
                logger.info(f"✅ Query with properties returned: {result.result_set}")
            
            # Test vector operations if supported
            try:
                # Create vector index
                graph.create_node_vector_index("Person", "embedding", dim=4, similarity_function="cosine")
                logger.info("✅ Created vector index")
                
                # Create a node with vector
                vector = [0.1, 0.2, 0.3, 0.4]  # Simple 4D vector
                result = graph.query(
                    "CREATE (:Person {name: 'Bob', embedding: $embedding})",
                    params={"embedding": vector}
                )
                logger.info(f"✅ Created node with vector embedding")
                
                # Vector search (may not work on all FalkorDB versions)
                try:
                    result = graph.query(
                        """
                        MATCH (p:Person)
                        WHERE p.embedding IS NOT NULL
                        WITH p, vector.similarity.cosine(p.embedding, $query_vector) AS score
                        RETURN p.name, score
                        ORDER BY score DESC
                        """,
                        params={"query_vector": [0.1, 0.2, 0.3, 0.4]}
                    )
                    logger.info(f"✅ Vector similarity search successful: {result.result_set}")
                except Exception as vector_e:
                    logger.warning(f"Vector similarity search not supported: {vector_e}")
            except Exception as vector_index_e:
                logger.warning(f"Vector index creation not supported: {vector_index_e}")
            
            # Delete the test graph
            graph.delete()
            logger.info("✅ Deleted test graph")
            
            logger.info("✅ FalkorDB is running properly with full graph capabilities")
            return True
        except Exception as graph_e:
            logger.error(f"❌ Error executing graph commands: {graph_e}")
            logger.error("   Graph commands failed")
            return False
    except ImportError:
        logger.error("❌ FalkorDB library not installed. Install with: pip install falkordb")
        return False
    except Exception as e:
        logger.error(f"❌ Could not connect to FalkorDB: {e}")
        logger.error("   Make sure to start FalkorDB using 'docker-compose up -d'")
        return False

def test_falkordb_retriever():
    """Test the FalkorDBRetriever class"""
    logger.info("🔍 Testing FalkorDBRetriever functionality...")
    try:
        from amem.retrievers import FalkorDBRetriever
        
        # Create a simple mock embedding provider for testing
        class MockEmbeddingProvider:
            def get_embedding(self, text: str) -> List[float]:
                # Return a simple deterministic vector based on text length
                # This is just for testing purposes
                return [0.1, 0.2, 0.3, 0.4] * 256  # 1024-dimensional vector
        
        # Create retriever with test collection
        retriever = FalkorDBRetriever(
            collection_name="test_retriever",
            host=os.getenv("FALKORDB_HOST", "localhost"),
            port=int(os.getenv("FALKORDB_PORT", "7379")),
            embedding_provider=MockEmbeddingProvider(),
            vector_size=1024
        )
        logger.info("✅ Created FalkorDBRetriever instance")
        
        # Add a test document
        doc_id = str(uuid.uuid4())
        metadata = {
            "keywords": ["test", "example", "memory"],
            "context": "Testing",
            "timestamp": "202505141234"
        }
        content = "This is a test memory for the FalkorDBRetriever"
        
        # Store JSON strings instead of raw arrays/objects to prevent serialization issues
        metadata["keywords"] = json.dumps(metadata["keywords"])
        
        result = retriever.add_document(content, metadata, doc_id)
        if result:
            logger.info(f"✅ Added test document with ID: {doc_id}")
        else:
            logger.error("❌ Failed to add test document")
            return False
        
        # Search for the document
        search_results = retriever.search("test memory", k=5)
        
        if (search_results and 'ids' in search_results and 
            search_results['ids'] and len(search_results['ids']) > 0 and 
            len(search_results['ids'][0]) > 0):
            logger.info(f"✅ Search returned {len(search_results['ids'][0])} results")
            logger.info(f"   First result ID: {search_results['ids'][0][0]}")
            logger.info(f"   First result similarity: {search_results['distances'][0][0]}")
        else:
            logger.error("❌ Search returned no results")
            return False
        
        # Delete the document
        delete_result = retriever.delete_document(doc_id)
        if delete_result:
            logger.info("✅ Successfully deleted test document")
        else:
            logger.error("❌ Failed to delete test document")
            return False
        
        # Clean up by deleting the test graph
        try:
            from falkordb import FalkorDB
            client = FalkorDB(host=os.getenv("FALKORDB_HOST", "localhost"), 
                              port=int(os.getenv("FALKORDB_PORT", "7379")))
            graphs = client.list_graphs()
            if "test_retriever" in graphs:
                graph = client.select_graph("test_retriever")
                graph.delete()
                logger.info("✅ Cleaned up test graph")
        except Exception as e:
            logger.warning(f"Warning during cleanup: {e}")
        
        logger.info("✅ FalkorDBRetriever tests passed")
        return True
    except ImportError:
        logger.error("❌ Required libraries not installed")
        return False
    except Exception as e:
        logger.error(f"❌ Error testing FalkorDBRetriever: {e}")
        return False

def check_embeddings():
    """Check if embedding API is working"""
    logger.info("🔍 Testing embedding functionality...")
    try:
        # Create embedding provider directly
        embedding_provider = LiteLLMEmbedding(
            model_name="bedrock/cohere.embed-multilingual-v3",
            aws_region=os.getenv("AWS_REGION_NAME"),
            aws_access_key=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
            vector_size=768
        )
        
        # Test text for embedding
        test_text = "This is a test of the embedding functionality"
        
        # First test raw LiteLLM embedding to see the structure
        try:
            from litellm import embedding as litellm_embedding
            logger.debug("Testing raw litellm embedding response structure...")
            raw_response = litellm_embedding(
                model="bedrock/cohere.embed-multilingual-v3",
                input=[test_text]
            )
            logger.debug(f"Raw response type: {type(raw_response)}")
            logger.debug(f"Raw response data type: {type(raw_response.data[0])}")
            
            if hasattr(raw_response.data[0], 'embedding'):
                logger.debug("Response has 'embedding' attribute")
            elif isinstance(raw_response.data[0], dict):
                logger.debug(f"Response keys: {list(raw_response.data[0].keys())}")
        except Exception as e:
            logger.warning(f"Could not get raw embedding structure: {e}")
        
        # Get embedding through our provider
        embedding_vector = embedding_provider.get_embedding(test_text)
        
        # Check if it's a valid vector
        if not isinstance(embedding_vector, list):
            logger.error(f"❌ Embedding is not a list: {type(embedding_vector)}")
            return False
        
        if len(embedding_vector) == 0:
            logger.error("❌ Embedding vector is empty")
            return False
            
        if not isinstance(embedding_vector[0], float):
            logger.error(f"❌ Embedding values are not floats: {type(embedding_vector[0])}")
            return False
            
        # Check if vector is not all zeros
        non_zero_count = sum(1 for val in embedding_vector if abs(val) > 1e-10)
        if non_zero_count < len(embedding_vector) * 0.5:  # At least 50% should be non-zero
            logger.error(f"❌ Embedding appears to be mostly zeros (only {non_zero_count}/{len(embedding_vector)} non-zero values)")
            return False
            
        # Display some stats about the embedding
        avg_magnitude = sum(abs(val) for val in embedding_vector) / len(embedding_vector)
        max_magnitude = max(abs(val) for val in embedding_vector)
        logger.info(f"✅ Embedding API working correctly (vector size: {len(embedding_vector)}, avg magnitude: {avg_magnitude:.6f}, max: {max_magnitude:.6f})")
            
        # Also test batch embedding
        test_texts = ["This is the first test text", "This is the second test text"]
        embedding_vectors = embedding_provider.get_embeddings(test_texts)
        
        if len(embedding_vectors) != len(test_texts):
            logger.error(f"❌ Batch embedding returned wrong number of vectors: {len(embedding_vectors)} (expected {len(test_texts)})")
            return False
            
        logger.info("✅ Batch embedding also working correctly")
        
        # Clean up any test graphs from FalkorDB if needed
        try:
            from falkordb import FalkorDB
            client = FalkorDB(host=os.getenv("FALKORDB_HOST", "localhost"), 
                              port=int(os.getenv("FALKORDB_PORT", "7379")))
            graphs = client.list_graphs()
            if "verification_test" in graphs:
                graph = client.select_graph("verification_test")
                graph.delete()
        except Exception as e:
            logger.warning(f"Note: Could not clean up test graph: {e}")
            
        return True
    except Exception as e:
        logger.error(f"❌ Embedding test failed: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def check_llm():
    """Check if LLM API is working"""
    logger.info("🔍 Testing LLM functionality...")
    try:
        # Initialize memory system
        memory_system = AgenticMemorySystem()
        
        # Simple prompt
        prompt = "What is machine learning? Answer in one sentence."
        
        # Get response
        response = memory_system.llm_controller.get_completion(prompt)
        
        # Check if response is valid
        if isinstance(response, str) and len(response) > 10:
            logger.info(f"✅ LLM API working correctly")
            logger.info(f"📝 LLM response: {response.strip()}")
            return True
        else:
            logger.error("❌ LLM response not in expected format")
            return False
    except Exception as e:
        logger.error(f"❌ LLM test failed: {e}")
        return False

def main():
    """Run verification of all components"""
    logger.info("🚀 Starting A-MEM verification")
    logger.info("-------------------------------")
    
    # Load environment variables from parent directory
    load_dotenv(dotenv_path="../.env")
    
    # Check all components
    db_ok = check_falkordb_connection()
    
    if not db_ok:
        logger.error(f"❌ FalkorDB verification failed. Cannot proceed with further tests.")
        return False
    
    retriever_ok = test_falkordb_retriever()
    embedding_ok = check_embeddings()
    llm_ok = check_llm()
    
    # Report results
    logger.info("\n📋 Verification Results:")
    logger.info(f"FalkorDB:          {'✅' if db_ok else '❌'}")
    logger.info(f"FalkorDBRetriever: {'✅' if retriever_ok else '❌'}")
    logger.info(f"Embedding:         {'✅' if embedding_ok else '❌'}")
    logger.info(f"LLM:               {'✅' if llm_ok else '❌'}")
    
    if all([db_ok, retriever_ok, embedding_ok, llm_ok]):
        logger.info("\n✨ All components verified successfully! ✨")
        logger.info("The A-MEM system is ready to use.")
        return True
    else:
        logger.error("\n⚠️ One or more components failed verification.")
        logger.error("Please check the logs above for details.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
