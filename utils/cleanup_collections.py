#!/usr/bin/env python3
"""
Cleanup script for Qdrant collections.

This script removes Qdrant collections with the wrong vector dimensions.
"""

import logging
from qdrant_client import QdrantClient

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def cleanup_collections():
    """Remove existing collections in Qdrant"""
    try:
        # Connect to Qdrant
        # Use port from environment variable or default to 7333
        from os import environ
        port = int(environ.get("QDRANT_PORT", "7333"))
        host = environ.get("QDRANT_HOST", "localhost")
        client = QdrantClient(host=host, port=port)
        
        # Get list of collections
        collections = client.get_collections()
        
        # Delete these collections
        target_collections = ["test_memories", "example_memories", "memories", "verification_test"]
        for collection in collections.collections:
            if collection.name in target_collections:
                logger.info(f"Deleting collection: {collection.name}")
                client.delete_collection(collection.name)
                
        logger.info("Cleanup completed successfully")
        
    except Exception as e:
        logger.error(f"Error during cleanup: {e}")

if __name__ == "__main__":
    logger.info("🧹 Starting Qdrant collection cleanup")
    cleanup_collections()
