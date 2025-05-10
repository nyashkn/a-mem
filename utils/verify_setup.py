#!/usr/bin/env python3
"""
A-MEM Verification Script

This script performs basic tests to verify that all components of A-MEM are working correctly.
It tests:
1. Qdrant connection
2. AWS Bedrock embedding functionality
3. OpenRouter LLM integration

Usage:
  python verify_setup.py
"""

import os
import sys
import json
import time
from dotenv import load_dotenv
from amem.memory_system import AgenticMemorySystem
from amem.embedding.providers import LiteLLMEmbedding
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("verify")

def check_qdrant():
    """Check if Qdrant is running and accessible"""
    logger.info("🔍 Checking Qdrant connection...")
    try:
        import requests
        response = requests.get("http://localhost:6333/healthz")
        if response.status_code == 200:
            logger.info("✅ Qdrant is running properly")
            return True
        else:
            logger.error(f"❌ Qdrant is responding with unexpected status code: {response.status_code}")
            return False
    except Exception as e:
        logger.error(f"❌ Could not connect to Qdrant: {e}")
        logger.info("   Make sure to start Qdrant using 'docker-compose up -d'")
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
        
        # Clean up test collection
        try:
            from qdrant_client import QdrantClient
            client = QdrantClient(host="localhost", port=6333)
            if "verification_test" in [c.name for c in client.get_collections().collections]:
                client.delete_collection("verification_test")
        except Exception as e:
            logger.warning(f"Note: Could not clean up test collection: {e}")
            
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
    
    # Load environment variables
    load_dotenv()
    
    # Check all components
    qdrant_ok = check_qdrant()
    
    if not qdrant_ok:
        logger.error("❌ Qdrant verification failed. Cannot proceed with further tests.")
        return False
    
    embedding_ok = check_embeddings()
    llm_ok = check_llm()
    
    # Report results
    logger.info("\n📋 Verification Results:")
    logger.info(f"Qdrant:    {'✅' if qdrant_ok else '❌'}")
    logger.info(f"Embedding: {'✅' if embedding_ok else '❌'}")
    logger.info(f"LLM:       {'✅' if llm_ok else '❌'}")
    
    if all([qdrant_ok, embedding_ok, llm_ok]):
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
