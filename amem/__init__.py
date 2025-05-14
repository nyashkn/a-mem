"""A-Mem: An agentic memory system for AI applications"""

from .memory_system import AgenticMemorySystem
from .llm_controller import LLMController
from .retrievers import QdrantRetriever, FalkorDBRetriever
from .embedding.providers import EmbeddingProvider, LiteLLMEmbedding
from .factory import EmbeddingProviderFactory, RetrieverFactory, LLMControllerFactory

__version__ = "0.1.0"

__all__ = [
    'AgenticMemorySystem',
    'LLMController',
    'QdrantRetriever',
    'FalkorDBRetriever',
    'EmbeddingProvider',
    'LiteLLMEmbedding',
    'EmbeddingProviderFactory',
    'RetrieverFactory',
    'LLMControllerFactory'
]
