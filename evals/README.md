# A-MEM Evaluation Scripts

This directory contains scripts for evaluating the A-MEM memory system performance.

## Available Evaluations

- `basic.py`: Basic functionality demonstration and testing of the A-MEM system.
  - Tests adding memories, searching, and memory evolution
  - Demonstrates core API usage

## Future Plans

Additional evaluation scripts will be added to measure:

- Performance metrics (speed, memory usage)
- Accuracy of memory retrieval
- Quality of memory evolution
- Comparative benchmarks against different embedding models
- Stress testing with large memory stores

## Running Evaluations

To run the basic evaluation:

```bash
python evals/basic.py
```

To clean up after running evaluations:

```bash
python evals/basic.py --cleanup  # Will delete test collections after running
```

## Prerequisites

- Qdrant server running (start with `docker-compose up -d`)
- AWS Bedrock credentials set up in environment variables
- OpenRouter API key set up in environment variables
