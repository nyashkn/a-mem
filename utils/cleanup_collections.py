#!/usr/bin/env python3
"""
Cleanup script for FalkorDB and Qdrant collections.

This script removes database collections/graphs to allow for fresh initialization.
It supports both FalkorDB graphs and Qdrant collections.
"""

import logging
import os
from dotenv import load_dotenv

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def cleanup_falkordb():
    """Remove existing graphs in FalkorDB"""
    try:
        # Connect to FalkorDB
        import redis
        port = int(os.environ.get("FALKORDB_PORT", ""))
        host = os.environ.get("FALKORDB_HOST", "localhost")
        client = redis.Redis(host=host, port=port, decode_responses=True)
        
        # Check connection
        if not client.ping():
            logger.error("❌ Cannot connect to FalkorDB")
            return False
            
        # Get list of graphs
        try:
            graphs = client.execute_command("GRAPH.LIST")
            
            # Target graphs to delete
            target_graphs = ["test_memories", "example_memories", "memories", "verification_test"]
            
            # Delete graphs
            deleted_count = 0
            for graph in graphs:
                if graph in target_graphs:
                    logger.info(f"Deleting graph: {graph}")
                    client.execute_command(f"GRAPH.DELETE {graph}")
                    deleted_count += 1
                    
            if deleted_count > 0:
                logger.info(f"Successfully deleted {deleted_count} FalkorDB graphs")
            else:
                logger.info("No target FalkorDB graphs found to delete")
                
            return True
        except Exception as e:
            logger.error(f"Error while working with FalkorDB graphs: {e}")
            return False
        
    except Exception as e:
        logger.error(f"Error connecting to FalkorDB: {e}")
        return False

def cleanup_qdrant():
    """Remove existing collections in Qdrant"""
    try:
        # Connect to Qdrant
        from qdrant_client import QdrantClient
        port = int(os.environ.get("QDRANT_PORT", "7333"))
        host = os.environ.get("QDRANT_HOST", "localhost")
        client = QdrantClient(host=host, port=port)
        
        # Get list of collections
        collections = client.get_collections()
        
        # Delete these collections
        target_collections = ["test_memories", "example_memories", "memories", "verification_test"]
        deleted_count = 0
        for collection in collections.collections:
            if collection.name in target_collections:
                logger.info(f"Deleting collection: {collection.name}")
                client.delete_collection(collection.name)
                deleted_count += 1
                
        if deleted_count > 0:
            logger.info(f"Successfully deleted {deleted_count} Qdrant collections")
        else:
            logger.info("No target Qdrant collections found to delete")
            
        return True
        
    except Exception as e:
        logger.error(f"Error during Qdrant cleanup: {e}")
        return False

def main():
    """Run appropriate cleanup based on configuration"""
    # Load environment variables
    load_dotenv()
    
    # Determine which database to clean based on environment variable
    db_type = os.environ.get("VECTOR_DB_TYPE", "falkordb").lower()
    
    if db_type == "falkordb":
        logger.info("🧹 Starting FalkorDB graph cleanup")
        cleanup_falkordb()
    elif db_type == "qdrant":
        logger.info("🧹 Starting Qdrant collection cleanup")
        cleanup_qdrant()
    else:
        logger.error(f"❌ Unknown database type: {db_type}")
        logger.info("   Set VECTOR_DB_TYPE to 'falkordb' or 'qdrant' in your .env file")
        return False
        
    logger.info("✨ Cleanup completed")
    return True

if __name__ == "__main__":
    main()
