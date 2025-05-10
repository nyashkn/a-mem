# A-MEM with Qdrant, LiteLLM, and Cohere Embeddings

This document outlines the enhancements made to the A-MEM system to use Qdrant for vector storage and LiteLLM for accessing cloud embedding services.

## Overview of Changes

1. **Removed SentenceTransformer Dependency**
   - Simplified the codebase to focus exclusively on cloud-based embedding services
   - Reduced package dependencies
   - Created a more efficient embedding pipeline

2. **Enhanced LiteLLM Integration**
   - Added support for AWS Bedrock's Cohere embedding models
   - Improved error handling and logging
   - Added vector size configuration option
   - Enhanced empty text handling

3. **Qdrant Vector Database**
   - Added Docker Compose configuration for easy setup
   - Created QdrantRetriever with ChromaDB-compatible interface
   - Configured automatic restart on system boot

4. **Environment-Based Configuration**
   - Centralized settings in `.env` file
   - Added configuration loader
   - Maintained backward compatibility with existing code

## Usage

### Starting Qdrant

```bash
docker-compose up -d
```

### Configuration

Create a `.env` file with your settings:

```
# Vector Database (qdrant)
VECTOR_DB_TYPE=qdrant
QDRANT_HOST=localhost
QDRANT_PORT=6333
QDRANT_COLLECTION=memories

# Embedding Provider (AWS Bedrock with Cohere)
EMBEDDING_PROVIDER=litellm
EMBEDDING_MODEL=bedrock/cohere.embed-multilingual-v3
AWS_REGION_NAME=us-west-2
AWS_ACCESS_KEY_ID=your-aws-access-key
AWS_SECRET_ACCESS_KEY=your-aws-secret-key

# LLM Configuration (OpenRouter with LLaMa)
LLM_BACKEND=openai
LLM_MODEL=meta-llama/llama-4-maverick
LLM_BASE_URL=https://openrouter.ai/api/v1
OPENROUTER_API_KEY=your-openrouter-key
```

### Basic Code Example

```python
from amem.memory_system import AgenticMemorySystem

# Initialize the memory system (config loaded from .env)
memory_system = AgenticMemorySystem()

# Add a memory
memory_id = memory_system.add_note(
    "Machine learning is a field of inquiry devoted to understanding and building methods that 'learn'."
)

# Search for memories
results = memory_system.search_agentic("machine learning", k=3)
for result in results:
    print(f"ID: {result['id']}")
    print(f"Content: {result['content'][:100]}...")
    print(f"Tags: {result['tags']}")
```

## Testing

Run the tests to verify functionality with cloud services:

```bash
# Start Qdrant first
docker-compose up -d

# Run tests
python -m unittest amem.test_memories
```

This will test actual API calls to embedding and LLM services, as well as Qdrant integration.

## Architecture

The enhanced system uses:

1. **LiteLLM** for accessing embedding and LLM services
   - Supports OpenAI, Anthropic, AWS Bedrock, etc.
   - Unified API for multiple providers

2. **Qdrant** for vector storage
   - High-performance vector database
   - Persistent storage
   - Docker-based deployment

3. **Environment Configuration**
   - Flexible settings via `.env`
   - Support for multiple deployment scenarios
