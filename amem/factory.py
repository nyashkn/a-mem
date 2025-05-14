"""
Factory module for creating A-MEM components.

This module provides factory classes for creating the various components
needed by the AgenticMemorySystem:
- EmbeddingProviderFactory: Creates embedding providers
- RetrieverFactory: Creates vector database retrievers
- LLMControllerFactory: Creates LLM controllers
"""

from typing import Dict, Optional, Any, Union
import os
import logging
from amem.embedding.providers import EmbeddingProvider, LiteLLMEmbedding
from amem.retrievers import FalkorDBRetriever
from amem.llm_controller import LLMController

logger = logging.getLogger(__name__)

class EmbeddingProviderFactory:
    """Factory for creating embedding providers."""
    
    @staticmethod
    def create(config: Dict[str, Any]) -> EmbeddingProvider:
        """Create an embedding provider based on configuration.
        
        Args:
            config: Configuration dictionary containing embedding settings
            
        Returns:
            An instance of EmbeddingProvider
        
        Raises:
            ValueError: If embedding provider cannot be initialized
        """
        # Handle case where config might be missing embedding section
        if "embedding" not in config or not config.get("embedding"):
            config["embedding"] = {
                "provider": "litellm",
                "model": "bedrock/cohere.embed-multilingual-v3"
            }
        
        embedding_config = config["embedding"]
        embedding_model = embedding_config.get("model", "bedrock/cohere.embed-multilingual-v3")
        
        try:
            if "bedrock" in embedding_model:
                # For AWS Bedrock support
                embedding_provider = LiteLLMEmbedding(
                    model_name=embedding_model,
                    api_key=embedding_config.get("api_key"),
                    aws_region=os.getenv("AWS_REGION_NAME"),
                    aws_access_key=os.getenv("AWS_ACCESS_KEY_ID"),
                    aws_secret_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
                    vector_size=1024  # Updated to match actual dimensions from AWS Bedrock
                )
                logger.info(f"Using AWS Bedrock embedding model: {embedding_model}")
            else:
                # Other LiteLLM supported providers
                embedding_provider = LiteLLMEmbedding(
                    model_name=embedding_model,
                    api_key=embedding_config.get("api_key"),
                    vector_size=1024  # Updated to match actual dimensions
                )
                logger.info(f"Using LiteLLM embedding model: {embedding_model}")
            
            # Test the embedding provider to verify it works
            try:
                test_embedding = embedding_provider.get_embedding("Test embedding functionality")
                if not test_embedding or not any(test_embedding):
                    raise ValueError("Embedding provider returned empty or all-zero embedding")
                logger.info(f"Embedding provider verified working (vector size: {len(test_embedding)})")
            except Exception as test_error:
                logger.error(f"Embedding provider test failed: {test_error}")
                raise ValueError(f"Embedding provider verification failed: {test_error}")
            
            return embedding_provider
        except Exception as e:
            logger.error(f"Failed to initialize embedding provider: {e}")
            raise ValueError(f"Could not initialize embedding provider: {e}")


class RetrieverFactory:
    """Factory for creating vector database retrievers."""
    
    @staticmethod
    def create(config: Dict[str, Any], embedding_provider: Optional[EmbeddingProvider] = None) -> FalkorDBRetriever:
        """Create a retriever based on configuration.
        
        Args:
            config: Configuration dictionary containing vector_db settings
            embedding_provider: Optional embedding provider to use
            
        Returns:
            An instance of FalkorDBRetriever
        """
        vector_db_config = config.get("vector_db", {})
        
        # Get FalkorDB configuration
        falkordb_config = vector_db_config.get("falkordb", {})
        
        # Use environment variable for collection if available
        collection_name = os.getenv("FALKORDB_COLLECTION") or falkordb_config.get("collection", "memories")
        host = falkordb_config.get("host", "localhost")
        port = falkordb_config.get("port", 7379)
        
        # Initialize FalkorDB retriever
        retriever = FalkorDBRetriever(
            collection_name=collection_name,
            host=host,
            port=port,
            embedding_provider=embedding_provider
        )
        logger.info(f"Created FalkorDB retriever with collection '{collection_name}'")
        return retriever


class LLMControllerFactory:
    """Factory for creating LLM controllers."""
    
    @staticmethod
    def create(config: Dict[str, Any]) -> LLMController:
        """Create an LLM controller based on configuration.
        
        Args:
            config: Configuration dictionary containing llm settings
            
        Returns:
            An instance of LLMController
        """
        llm_config = config.get("llm", {})
        
        # Get config values with defaults - only support openai, ollama, litellm
        backend = llm_config.get("backend", "openai")  # Default to openai instead of openrouter
        # Validate the backend
        if backend not in ["openai", "ollama", "litellm"]:
            logger.warning(f"Unsupported backend '{backend}', falling back to 'openai'")
            backend = "openai"
            
        model = llm_config.get("model", "meta-llama/llama-4-maverick")
        api_key = llm_config.get("api_key")
        
        # Set base_url for OpenRouter if using openai backend with OpenRouter model
        base_url = llm_config.get("base_url")
        if backend == "openai" and "openrouter.ai" not in str(base_url) and "meta-llama/" in model:
            logger.info(f"Using OpenRouter base URL for model {model}")
            base_url = "https://openrouter.ai/api/v1"
        
        # Initialize LLM controller
        llm_controller = LLMController(
            backend=backend,
            model=model,
            api_key=api_key,
            base_url=base_url
        )
        
        logger.info(f"Created LLM controller with backend '{backend}' and model '{model}'")
        return llm_controller
