# A-MEM Utility Scripts

This directory contains utility scripts for maintaining and verifying the A-MEM memory system.

## Available Utilities

- `verify_setup.py`: Verifies that all components of the A-MEM system are working correctly.
  - Checks Qdrant connection
  - Tests embedding functionality
  - Verifies LLM integration

- `cleanup_collections.py`: Cleans up Qdrant collections with incorrect vector dimensions or for testing purposes.
  - Removes test collections
  - Useful after changing embedding providers or dimensions

## Usage

To verify your setup:

```bash
python utils/verify_setup.py
```

To clean up collections:

```bash
python utils/cleanup_collections.py
```

## Prerequisites

- Qdrant server running (start with `docker-compose up -d`)
- AWS Bedrock credentials set up in environment variables
- OpenRouter API key set up in environment variables
