from abc import ABC, abstractmethod
from typing import List, Union, Optional
import os
from litellm import embedding
import logging

logger = logging.getLogger(__name__)

class EmbeddingProvider(ABC):
    """Abstract class for embedding providers"""
    
    @abstractmethod
    def get_embedding(self, text: str) -> List[float]:
        """Get embedding for a single text"""
        pass
    
    @abstractmethod
    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for multiple texts"""
        pass

class LiteLLMEmbedding(EmbeddingProvider):
    """LiteLLM-based embedding provider for cloud embedding services"""
    
    def __init__(self, model_name: str, api_key: Optional[str] = None, 
                 aws_region: Optional[str] = None,
                 aws_access_key: Optional[str] = None, 
                 aws_secret_key: Optional[str] = None,
                 vector_size: int = 384):
        """Initialize LiteLLM embedding provider
        
        Args:
            model_name: Model name in LiteLLM format (e.g., 'bedrock/cohere.embed-multilingual-v3')
            api_key: API key for the embedding provider
            aws_region: AWS region name for Bedrock models
            aws_access_key: AWS access key ID for Bedrock models
            aws_secret_key: AWS secret access key for Bedrock models
            vector_size: Dimensionality of the embedding vectors (default: 384)
        """
        self.model_name = model_name
        self.vector_size = vector_size
        
        # Set up API keys based on the provider
        provider = model_name.split('/')[0] if '/' in model_name else model_name
        
        if provider == "bedrock" and aws_region and aws_access_key and aws_secret_key:
            os.environ["AWS_REGION_NAME"] = aws_region
            os.environ["AWS_ACCESS_KEY_ID"] = aws_access_key
            os.environ["AWS_SECRET_ACCESS_KEY"] = aws_secret_key
            logger.info(f"Using AWS Bedrock with region {aws_region}")
        elif api_key:
            # For other providers that use API keys
            if provider == "openai":
                os.environ["OPENAI_API_KEY"] = api_key
            elif provider == "cohere":
                os.environ["COHERE_API_KEY"] = api_key
            elif provider == "huggingface":
                os.environ["HUGGINGFACE_API_KEY"] = api_key
            else:
                # Generic case
                os.environ[f"{provider.upper()}_API_KEY"] = api_key
            
            logger.info(f"Using {provider} embedding provider")
    
    def get_embedding(self, text: str) -> List[float]:
        """Get embedding for a single text"""
        if not text or not text.strip():
            logger.warning("Attempted to get embedding for empty text")
            raise ValueError("Empty text cannot be embedded")
            
        try:
            response = embedding(
                model=self.model_name,
                input=[text]
            )
            
            # Output debug info about response structure
            logger.debug(f"Response type: {type(response)}")
            if hasattr(response, 'data') and len(response.data) > 0:
                logger.debug(f"Response data type: {type(response.data[0])}")
                if isinstance(response.data[0], dict):
                    logger.debug(f"Response data keys: {list(response.data[0].keys())}")
            
            # For AWS Bedrock, check the data structure directly
            if hasattr(response, 'data') and len(response.data) > 0:
                data = response.data[0]
                if isinstance(data, dict) and "embedding" in data:
                    embedding_vector = data["embedding"]
                    # Verify it's not empty or all zeros
                    if embedding_vector and any(embedding_vector):
                        return embedding_vector
            
            # Standard formats
            if hasattr(response.data[0], 'embedding'):
                # Object with embedding attribute
                embedding_vector = response.data[0].embedding
                if embedding_vector and any(embedding_vector):
                    return embedding_vector
            elif isinstance(response.data[0], dict):
                # Dictionary with embedding key (various providers)
                for key in ['embedding', 'embeddings', 'vector', 'values', 'data']:
                    if key in response.data[0]:
                        embedding_vector = response.data[0][key]
                        if embedding_vector and any(embedding_vector):
                            return embedding_vector
                
                # If it's another type of response, try to extract embedding
                if len(response.data[0]) == 1:  # If dict has only one key, use its value
                    embedding_vector = list(response.data[0].values())[0]
                    if embedding_vector and any(embedding_vector):
                        return embedding_vector
            
            # If we reach here, we couldn't find the embedding in the response
            logger.error(f"Unknown embedding response structure")
            logger.debug(f"Response type: {type(response)}")
            logger.debug(f"Response data: {response.data[0] if hasattr(response, 'data') else 'No data attribute'}")
            
            # Critical error - do not use fallback in production
            raise ValueError(f"Could not extract valid embedding from response")
            
        except Exception as e:
            logger.error(f"Error getting embedding: {e}")
            # Don't use fallback zeros - embeddings are critical
            raise ValueError(f"Failed to generate embedding: {e}")
    
    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for multiple texts"""
        # Check for empty texts
        if not texts:
            logger.warning("Empty texts list provided")
            raise ValueError("Cannot embed empty list of texts")
        
        # Filter empty texts
        empty_indices = [i for i, t in enumerate(texts) if not t or not t.strip()]
        if len(empty_indices) == len(texts):
            logger.warning("All texts are empty")
            raise ValueError("All texts are empty, cannot generate embeddings")
            
        # Get valid texts
        valid_texts = [t for t in texts if t and t.strip()]
        
        try:
            response = embedding(
                model=self.model_name,
                input=valid_texts
            )
            
            # Output debug info about response structure
            logger.debug(f"Response type: {type(response)}")
            if hasattr(response, 'data') and len(response.data) > 0:
                logger.debug(f"Response data type: {type(response.data[0])}")
                if isinstance(response.data[0], dict):
                    logger.debug(f"Response data keys: {list(response.data[0].keys())}")
            
            # Extract embeddings from response
            embeddings = []
            for i, data in enumerate(response.data):
                embedding_vector = None
                # Try to extract embedding from data
                if isinstance(data, dict) and "embedding" in data:
                    embedding_vector = data["embedding"]
                elif hasattr(data, 'embedding'):
                    embedding_vector = data.embedding
                elif isinstance(data, dict):
                    # Try common keys
                    for key in ['embedding', 'embeddings', 'vector', 'values', 'data']:
                        if key in data:
                            embedding_vector = data[key]
                            break
                    else:  # No break occurred - no common key found
                        if len(data) == 1:  # If dict has only one key, use its value
                            embedding_vector = list(data.values())[0]
                
                # Validate embedding vector
                if not embedding_vector or not any(embedding_vector):
                    # Critical error - invalid embedding
                    raise ValueError(f"Invalid embedding vector for text at index {i}")
                
                embeddings.append(embedding_vector)
            
            # Verify we got the right number of embeddings
            if len(embeddings) != len(valid_texts):
                logger.error(f"Embedding count mismatch: got {len(embeddings)}, expected {len(valid_texts)}")
                raise ValueError(f"Embedding count mismatch: {len(embeddings)} vs {len(valid_texts)}")
            
            # Map embeddings back to original text positions
            result = []
            valid_idx = 0
            
            for i, text in enumerate(texts):
                if i in empty_indices:
                    # Raise error for empty texts instead of using zero vectors
                    raise ValueError(f"Cannot include empty text at index {i} in embeddings")
                else:
                    result.append(embeddings[valid_idx])
                    valid_idx += 1
                    
            return result
        
        except Exception as e:
            logger.error(f"Error getting embeddings: {e}")
            # Don't use fallback zeros - embeddings are critical
            raise ValueError(f"Failed to generate embeddings: {e}")
