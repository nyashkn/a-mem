# Agentic Memory 🧠

A novel agentic memory system for LLM agents that can dynamically organize memories in an agentic way.

## Introduction 🌟

Large Language Model (LLM) agents have demonstrated remarkable capabilities in handling complex real-world tasks through external tool usage. However, to effectively leverage historical experiences, they require sophisticated memory systems. Traditional memory systems, while providing basic storage and retrieval functionality, often lack advanced memory organization capabilities.

Our project introduces an innovative **Agentic Memory** system that revolutionizes how LLM agents manage and utilize their memories:

<div align="center">
  <img src="Figure/intro-a.jpg" alt="Traditional Memory System" width="600"/>
  <img src="Figure/intro-b.jpg" alt="Our Proposed Agentic Memory" width="600"/>
  <br>
  <em>Comparison between traditional memory system (top) and our proposed agentic memory (bottom). Our system enables dynamic memory operations and flexible agent-memory interactions.</em>
</div>

> **Note:** This repository provides a memory system to facilitate agent construction. If you want to reproduce the results presented in our paper, please refer to: [https://github.com/WujiangXu/AgenticMemory](https://github.com/WujiangXu/AgenticMemory)

For more details, please refer to our paper: [A-MEM: Agentic Memory for LLM Agents](https://arxiv.org/pdf/2502.12110)


## Key Features ✨

- 🔄 Dynamic memory organization based on Zettelkasten principles
- 🔍 Intelligent indexing and linking of memories via ChromaDB
- 📝 Comprehensive note generation with structured attributes
- 🌐 Interconnected knowledge networks
- 🧬 Continuous memory evolution and refinement
- 🤖 Agent-driven decision making for adaptive memory management

## Framework 🏗️

<div align="center">
  <img src="Figure/framework.jpg" alt="Agentic Memory Framework" width="800"/>
  <br>
  <em>The framework of our Agentic Memory system showing the dynamic interaction between LLM agents and memory components.</em>
</div>

## How It Works 🛠️

When a new memory is added to the system:
1. Generates comprehensive notes with structured attributes
2. Creates contextual descriptions and tags
3. Analyzes historical memories for relevant connections
4. Establishes meaningful links based on similarities
5. Enables dynamic memory evolution and updates

## Results 📊

Empirical experiments conducted on six foundation models demonstrate superior performance compared to existing SOTA baselines.

## Getting Started 🚀

1. Clone the repository:
```bash
git clone https://github.com/nyashkn/A-mem.git .
```

> **Note:** This is a fork of the original project which can be found at [https://github.com/agiresearch/A-mem.git](https://github.com/agiresearch/A-mem.git). We recommend checking out the original repository for the latest research developments.

2. Install dependencies using uv (recommended):
```bash
# Install uv if you don't have it
curl -sSf https://raw.githubusercontent.com/astral-sh/uv/main/install.sh | bash

# Create and activate virtual environment with uv
uv venv .venv
source .venv/bin/activate  # Linux/Mac
.venv\Scripts\activate     # Windows

# Install package in development mode with uv
uv pip install -e .
```

3. Set up Qdrant using Docker:
```bash
# Start Qdrant vector database
docker-compose up -d
```

4. Configure environment variables:
Create a `.env` file in the project root with the following settings (or modify the existing one):

```bash
# Vector Database (qdrant)
VECTOR_DB_TYPE=qdrant
QDRANT_HOST=localhost
QDRANT_PORT=6333

# Embedding Provider (AWS Bedrock with Cohere)
EMBEDDING_PROVIDER=litellm
EMBEDDING_MODEL=bedrock/cohere.embed-multilingual-v3
AWS_REGION_NAME=your_aws_region
AWS_ACCESS_KEY_ID=your_aws_access_key
AWS_SECRET_ACCESS_KEY=your_aws_secret_key

# LLM Configuration (OpenRouter with LLaMa)
LLM_BACKEND=openai
LLM_MODEL=meta-llama/llama-4-maverick
LLM_BASE_URL=https://openrouter.ai/api/v1
OPENROUTER_API_KEY=your_openrouter_api_key

# Collection name
QDRANT_COLLECTION=memories
```

5. Setup Verification ✅

Before using the system, you can verify that all components are working correctly:

```bash
python utils/verify_setup.py
```

This will check:
- Qdrant database connection
- Embedding functionality with AWS Bedrock
- LLM integration with OpenRouter

If you encounter dimension errors with Qdrant collections, you can clean them up:

```bash
python utils/cleanup_collections.py
```

6. Usage Examples 💡

Here's how to use the Agentic Memory system for basic operations:

```python
from amem.memory_system import AgenticMemorySystem

# Initialize the memory system 🚀
memory_system = AgenticMemorySystem(
    config={
        "embedding": {
            "model": "bedrock/cohere.embed-multilingual-v3"  # AWS Bedrock embedding model
        },
        "llm": {
            "backend": "openai",           # LLM backend (openai/ollama/litellm)
            "model": "meta-llama/llama-4-maverick"  # Model name in OpenRouter format
        }
    }
)

# For a basic example, you can run:
# python evals/basic.py

# Add Memories ➕
# Simple addition
memory_id = memory_system.add_note("Deep learning neural networks")

# Addition with metadata
memory_id = memory_system.add_note(
    content="Machine learning project notes",
    tags=["ml", "project"],
    category="Research",
    timestamp="202503021500"  # YYYYMMDDHHmm format
)

# Read (Retrieve) Memories 📖
# Get memory by ID
memory = memory_system.read(memory_id)
print(f"Content: {memory.content}")
print(f"Tags: {memory.tags}")
print(f"Context: {memory.context}")
print(f"Keywords: {memory.keywords}")

# Search memories
results = memory_system.search_agentic("neural networks", k=5)
for result in results:
    print(f"ID: {result['id']}")
    print(f"Content: {result['content']}")
    print(f"Tags: {result['tags']}")
    print("---")

# Update Memories 🔄
memory_system.update(memory_id, content="Updated content about deep learning")

# Delete Memories ❌
memory_system.delete(memory_id)

# Memory Evolution 🧬
# The system automatically evolves memories by:
# 1. Finding semantic relationships using ChromaDB
# 2. Updating metadata and context
# 3. Creating connections between related memories
# This happens automatically when adding or updating memories!
```

### Advanced Features 🌟

1. **ChromaDB Vector Storage** 📦
   - Efficient vector embedding storage and retrieval
   - Fast semantic similarity search
   - Automatic metadata handling
   - Persistent memory storage

2. **Memory Evolution** 🧬
   - Automatically analyzes content relationships
   - Updates tags and context based on related memories
   - Creates semantic connections between memories

3. **Flexible Metadata** 📋
   - Custom tags and categories
   - Automatic keyword extraction
   - Context generation
   - Timestamp tracking

4. **Multiple LLM Backends** 🤖
   - OpenAI (GPT-4, GPT-3.5)
   - Ollama (for local deployment)
   - LiteLLM (unified API for various providers)
   - AWS Bedrock for embeddings

### Best Practices 💪

1. **Memory Creation** ✨:
   - Provide clear, specific content
   - Add relevant tags for better organization
   - Let the system handle context and keyword generation

2. **Memory Retrieval** 🔍:
   - Use specific search queries
   - Adjust 'k' parameter based on needed results
   - Consider both exact and semantic matches

3. **Memory Evolution** 🧬:
   - Allow automatic evolution to organize memories
   - Review generated connections periodically
   - Use consistent tagging conventions

4. **Error Handling** ⚠️:
   - Always check return values
   - Handle potential KeyError for non-existent memories
   - Use try-except blocks for LLM operations

## Citation 📚

If you use this code in your research, please cite our work:

```bibtex
@article{xu2025mem,
  title={A-mem: Agentic memory for llm agents},
  author={Xu, Wujiang and Liang, Zujie and Mei, Kai and Gao, Hang and Tan, Juntao and Zhang, Yongfeng},
  journal={arXiv preprint arXiv:2502.12110},
  year={2025}
}
```

## License 📄

This project is licensed under the MIT License. See LICENSE for details.
