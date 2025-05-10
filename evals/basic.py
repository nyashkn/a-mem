#!/usr/bin/env python3
"""
A-MEM Example Script

This script demonstrates how to use the A-MEM system with:
- Qdrant vector database
- OpenRouter API for LLM (meta-llama/llama-4-maverick)
- AWS Bedrock for embeddings (cohere.embed-multilingual-v3)

Usage:
  python example.py             # Run example
  python example.py --cleanup   # Run example and clean up test data after
"""

import os
import sys
import time
import argparse
from amem.memory_system import AgenticMemorySystem
from dotenv import load_dotenv

def check_qdrant_running():
    """Check if Qdrant is running via simple HTTP request"""
    try:
        import requests
        response = requests.get("http://localhost:6333/healthz")
        if response.status_code == 200:
            print("✅ Qdrant is running")
            return True
        else:
            print("❌ Qdrant is responding but may have issues (status code: {response.status_code})")
            return False
    except Exception as e:
        print(f"❌ Qdrant is not running or not reachable: {e}")
        print("   Please start Qdrant using 'docker-compose up -d'")
        return False

def main():
    """Main example function"""
    # Load environment variables
    load_dotenv()
    
    # Check if Qdrant is running
    if not check_qdrant_running():
        print("Please start Qdrant first using 'docker-compose up -d'")
        sys.exit(1)
    
    print("\n🚀 Initializing A-MEM system...")
    try:
        # Initialize the memory system with a specific test collection
        memory_system = AgenticMemorySystem(
            config={
                "vector_db": {
                    "type": "qdrant",
                    "qdrant": {
                        "collection": "example_memories",  # Use specific collection for example script
                        "host": "localhost",
                        "port": 6333
                    }
                }
                # Other config settings will be loaded from .env
            }
        )
        print("✅ Memory system initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize memory system: {e}")
        sys.exit(1)

    # Add example memories
    print("\n📝 Adding example memories...")
    try:
        # Memory 1
        memory1_id = memory_system.add_note(
            "Machine learning is a field of inquiry devoted to understanding "
            "and building methods that 'learn', that is, methods that leverage "
            "data to improve performance on some set of tasks.",
            tags=["machine learning", "AI", "data science"]
        )
        print(f"✅ Added memory 1: {memory1_id}")
        
        # Memory 2
        memory2_id = memory_system.add_note(
            "Deep learning is part of a broader family of machine learning methods "
            "based on artificial neural networks with representation learning. "
            "Learning can be supervised, semi-supervised or unsupervised.",
            tags=["deep learning", "neural networks", "AI"]
        )
        print(f"✅ Added memory 2: {memory2_id}")
        
        # Memory 3
        memory3_id = memory_system.add_note(
            "Natural language processing (NLP) is a subfield of linguistics, "
            "computer science, and artificial intelligence concerned with the "
            "interactions between computers and human language.",
            tags=["NLP", "AI", "language processing"]
        )
        print(f"✅ Added memory 3: {memory3_id}")
        
        # Give the system a moment to process and evolve memories
        time.sleep(2)
        
    except Exception as e:
        print(f"❌ Error adding memories: {e}")
        sys.exit(1)

    # Search for memories
    print("\n🔍 Searching for memories about 'machine learning'...")
    try:
        results = memory_system.search_agentic("machine learning", k=3)
        for i, result in enumerate(results):
            print(f"\n--- Result {i+1} ---")
            print(f"ID: {result['id']}")
            print(f"Content: {result['content'][:100]}...")
            print(f"Tags: {result['tags']}")
            print(f"Context: {result['context']}")
            if 'is_neighbor' in result and result['is_neighbor']:
                print("(This result is a linked neighbor memory)")
    except Exception as e:
        print(f"❌ Error searching memories: {e}")

    # Demonstrate memory evolution
    print("\n🧬 Adding related memory to demonstrate evolution...")
    try:
        memory4_id = memory_system.add_note(
            "Reinforcement learning (RL) is an area of machine learning concerned "
            "with how intelligent agents ought to take actions in an environment "
            "in order to maximize the notion of cumulative reward.",
            tags=["reinforcement learning", "machine learning", "AI"]
        )
        print(f"✅ Added memory 4: {memory4_id}")
        
        # Give the system a moment to process and evolve memories
        time.sleep(2)
        
        # Check if links have been created
        memory4 = memory_system.read(memory4_id)
        if memory4 and memory4.links:
            print(f"🔄 Memory has been evolved and linked to {len(memory4.links)} other memories")
            for link in memory4.links:
                linked_memory = memory_system.read(link)
                if linked_memory:
                    print(f"  - Linked to: '{linked_memory.content[:50]}...'")
        else:
            print("Memory hasn't formed connections yet (evolution can take time)")
    except Exception as e:
        print(f"❌ Error demonstrating evolution: {e}")

    print("\n✨ Example completed successfully! ✨")

def cleanup_example_data():
    """Clean up the example collection from Qdrant"""
    try:
        print("\n🧹 Cleaning up example data...")
        from qdrant_client import QdrantClient
        client = QdrantClient(host="localhost", port=6333)
        
        collections = [c.name for c in client.get_collections().collections]
        if "example_memories" in collections:
            client.delete_collection("example_memories")
            print("✅ Example collection deleted successfully")
        else:
            print("ℹ️ No example collection found to delete")
            
    except Exception as e:
        print(f"❌ Error cleaning up example data: {e}")

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Run A-MEM example")
    parser.add_argument("--cleanup", action="store_true", help="Clean up test data after running the example")
    args = parser.parse_args()
    
    # Run main example
    main()
    
    # Clean up if requested
    if args.cleanup:
        cleanup_example_data()
