# A-MEM Testing Guide

This guide explains how to test the A-MEM system with the Qdrant vector store and cloud embedding services.

## Prerequisites

Before running any tests, make sure:

1. **Qdrant is running**:
   ```bash
   docker-compose up -d
   ```

2. **Environment variables are set**:
   All necessary API keys and configuration parameters should be in your `.env` file (AWS credentials, OpenRouter API key, etc.)

3. **Dependencies are installed**:
   ```bash
   uv pip install -e .
   ```

## Testing Options

### 1. Run the Example Script

This is the quickest way to see the system in action. It creates, searches, and evolves memories using a dedicated `example_memories` collection.

```bash
# Run the example
python example.py

# Run and clean up after completion
python example.py --cleanup
```

The example script demonstrates:
- Memory creation and storage
- Semantic search using Qdrant
- Memory evolution with LLM
- Relationship formation between memories

### 2. Run Unit Tests

For more comprehensive testing and verification:

```bash
# Run all tests
python -m unittest discover -s amem/tests

# Run only cloud service tests
python -m unittest amem.tests.test_memories

# Run only memory system tests
python -m unittest amem.tests.test_memory_system
```

#### Test Isolation

The tests are designed with isolation in mind:

- `test_memories.py` uses a `test_memories` collection
- `test_memory_system.py` uses a `test_memories_unit_tests` collection
- Both automatically clean up collections after testing

## Key Tests to Verify

### 1. Embedding Functionality

```bash
python -m unittest amem.tests.test_memories.TestAgenticMemoryWithCloudServices.test_embedding_functionality
```

This test:
- Verifies AWS Bedrock Cohere embedding API works
- Confirms embedding vector sizes and format

### 2. LLM Integration

```bash
python -m unittest amem.tests.test_memories.TestAgenticMemoryWithCloudServices.test_llm_functionality
```

This test:
- Confirms OpenRouter LLM API connectivity
- Tests response quality and format

### 3. End-to-End Memory Flow

```bash
python -m unittest amem.tests.test_memories.TestAgenticMemoryWithCloudServices.test_end_to_end_memory_flow
```

This test:
- Creates multiple memories
- Searches for related memories 
- Tests memory evolution
- Verifies relationship formation

## Troubleshooting

### Common Issues

1. **Qdrant Connection Errors**:
   - Verify Docker is running: `docker ps`
   - Check Qdrant container status: `docker logs a-mem-qdrant-1`
   - Test Qdrant API directly: `curl http://localhost:6333/healthz`

2. **Cloud API Failures**:
   - Check API keys in `.env` file
   - Verify network connectivity
   - Look for rate limiting or quota issues in error messages

3. **Missing Dependencies**:
   - Run `uv pip install -e .` to ensure all dependencies are installed
   - Check for any missing packages in error messages

### Logs

For more detailed debugging, enable debug logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Memory Collection Cleanup

To clean up all test collections and start fresh:

```python
from qdrant_client import QdrantClient
client = QdrantClient(host="localhost", port=6333)
for collection in ["test_memories", "test_memories_unit_tests", "example_memories"]:
    try:
        client.delete_collection(collection)
        print(f"Deleted {collection}")
    except:
        pass
