from typing import List, Dict, Any, Optional, Union
from rank_bm25 import BM25Okapi
import nltk
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import chromadb
from chromadb.config import Settings
import pickle
from nltk.tokenize import word_tokenize
import os
import json
import uuid
from amem.embedding.providers import EmbeddingProvider, LiteLLMEmbedding
import logging

logger = logging.getLogger(__name__)

def simple_tokenize(text):
    return word_tokenize(text)

class ChromaRetriever:
    """Vector database retrieval using ChromaDB"""
    def __init__(self, collection_name: str = "memories"):
        """Initialize ChromaDB retriever.
        
        Args:
            collection_name: Name of the ChromaDB collection
        """
        self.client = chromadb.Client(Settings(allow_reset=True))
        self.collection = self.client.get_or_create_collection(name=collection_name)
        
    def add_document(self, document: str, metadata: Dict, doc_id: str):
        """Add a document to ChromaDB.
        
        Args:
            document: Text content to add
            metadata: Dictionary of metadata
            doc_id: Unique identifier for the document
        """
        # Convert MemoryNote object to serializable format
        processed_metadata = {}
        for key, value in metadata.items():
            if isinstance(value, list):
                processed_metadata[key] = json.dumps(value)
            elif isinstance(value, dict):
                processed_metadata[key] = json.dumps(value)
            else:
                processed_metadata[key] = str(value)
                
        self.collection.add(
            documents=[document],
            metadatas=[processed_metadata],
            ids=[doc_id]
        )
        
    def delete_document(self, doc_id: str):
        """Delete a document from ChromaDB.
        
        Args:
            doc_id: ID of document to delete
        """
        self.collection.delete(ids=[doc_id])
        
    def search(self, query: str, k: int = 5):
        """Search for similar documents.
        
        Args:
            query: Query text
            k: Number of results to return
            
        Returns:
            Dict with documents, metadatas, ids, and distances
        """
        results = self.collection.query(
            query_texts=[query],
            n_results=k
        )
        
        # Convert string metadata back to original types
        if 'metadatas' in results and results['metadatas'] and len(results['metadatas']) > 0:
            # First level is a list with one item per query
            for i in range(len(results['metadatas'])):
                # Second level is a list of metadata dicts for each result
                if isinstance(results['metadatas'][i], list):
                    for j in range(len(results['metadatas'][i])):
                        # Process each metadata dict
                        if isinstance(results['metadatas'][i][j], dict):
                            metadata = results['metadatas'][i][j]
                            for key, value in metadata.items():
                                try:
                                    # Try to parse JSON for lists and dicts
                                    if isinstance(value, str) and (value.startswith('[') or value.startswith('{')):
                                        metadata[key] = json.loads(value)
                                    # Convert numeric strings back to numbers
                                    elif isinstance(value, str) and value.replace('.', '', 1).isdigit():
                                        if '.' in value:
                                            metadata[key] = float(value)
                                        else:
                                            metadata[key] = int(value)
                                except (json.JSONDecodeError, ValueError):
                                    # If parsing fails, keep the original string
                                    pass
                        
        return results

class QdrantRetriever:
    """Vector database retrieval using Qdrant"""
    
    def __init__(self, 
                collection_name: str = "memories", 
                host: str = "localhost", 
                port: int = 6333,
                embedding_provider: Optional[EmbeddingProvider] = None,
                vector_size: int = 1024):
        """Initialize Qdrant retriever.
        
        Args:
            collection_name: Name of the Qdrant collection
            host: Qdrant server host
            port: Qdrant server port
            embedding_provider: Provider for embeddings
            vector_size: Size of embedding vectors
        """
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.models import VectorParams, Distance
        except ImportError:
            raise ImportError("Qdrant client not found. Install it with: pip install qdrant-client")
        
        self.collection_name = collection_name
        
        # Set up embedding provider
        if embedding_provider is None:
            # Default to LiteLLM embedding provider with default settings
            self.embedding_provider = LiteLLMEmbedding(
                model_name="bedrock/cohere.embed-multilingual-v3",
                vector_size=vector_size
            )
            logger.info("Using default LiteLLM embedding provider with AWS Bedrock")
        else:
            self.embedding_provider = embedding_provider
            
        self.vector_size = vector_size
        
        # Connect to Qdrant
        self.client = QdrantClient(host=host, port=port)
        
        # Check if collection exists, create if it doesn't
        collections = self.client.get_collections()
        collection_exists = any(collection.name == collection_name for collection in collections.collections)
        
        if not collection_exists:
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
            )
    
    def add_document(self, document: str, metadata: Dict, doc_id: str):
        """Add a document to Qdrant.
        
        Args:
            document: Text content to add
            metadata: Dictionary of metadata
            doc_id: Unique identifier for the document
        """
        from qdrant_client.models import PointStruct
        
        # Process metadata to ensure it's serializable
        processed_metadata = {}
        for key, value in metadata.items():
            if key == 'content':
                # Make sure we preserve the document content in metadata
                processed_metadata[key] = document
            elif isinstance(value, (list, dict)):
                # Qdrant can handle these natively
                processed_metadata[key] = value
            else:
                processed_metadata[key] = str(value)
        
        # Get embedding for document
        try:
            embedding = self.embedding_provider.get_embedding(document)
            
            # Add point to Qdrant
            self.client.upsert(
                collection_name=self.collection_name,
                points=[
                    PointStruct(
                        id=doc_id,
                        vector=embedding,
                        payload=processed_metadata
                    )
                ]
            )
        except Exception as e:
            logger.error(f"Error adding document to Qdrant: {e}")
    
    def delete_document(self, doc_id: str):
        """Delete a document from Qdrant.
        
        Args:
            doc_id: ID of document to delete
        """
        try:
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=[doc_id]
            )
        except Exception as e:
            logger.error(f"Error deleting document from Qdrant: {e}")
    
    def search(self, query: str, k: int = 5):
        """Search for similar documents.
        
        Args:
            query: Query text
            k: Number of results to return
            
        Returns:
            Dict with documents, metadatas, ids, and distances (in ChromaDB-compatible format)
        """
        try:
            # Get embedding for query
            query_embedding = self.embedding_provider.get_embedding(query)
            
            # Search Qdrant
            results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=k
            )
            
            # Convert to format compatible with ChromaDB results
            ids = [[result.id for result in results]]
            metadatas = [[result.payload for result in results]]
            distances = [[result.score for result in results]]
            documents = [[result.payload.get('content', '') for result in results]]
            
            return {
                'ids': ids,
                'metadatas': metadatas,
                'distances': distances,
                'documents': documents
            }
        except Exception as e:
            logger.error(f"Error searching in Qdrant: {e}")
            # Return empty results in ChromaDB format
            return {
                'ids': [[]],
                'metadatas': [[]],
                'distances': [[]],
                'documents': [[]]
            }
