from typing import List, Dict, Any, Optional, Union
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import word_tokenize
from amem.embedding.providers import EmbeddingProvider, LiteLLMEmbedding
import logging

logger = logging.getLogger(__name__)


def simple_tokenize(text):
    return word_tokenize(text)

class FalkorDBRetriever:
    """Graph database retrieval using FalkorDB"""
    
    def __init__(self, 
                collection_name: str = "memories", 
                host: str = "localhost", 
                port: int = 7379,  # Using the non-default port for FalkorDB
                embedding_provider: Optional[EmbeddingProvider] = None,
                vector_size: int = 1024):
        """Initialize FalkorDB retriever.
        
        Args:
            collection_name: Name of the FalkorDB graph
            host: FalkorDB server host
            port: FalkorDB server port
            embedding_provider: Provider for embeddings
            vector_size: Size of embedding vectors
        """
        try:
            from falkordb import FalkorDB
        except ImportError:
            raise ImportError("FalkorDB client not found. Install it with: pip install falkordb")
        
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
        
        # Connect to FalkorDB
        self.client = FalkorDB(host=host, port=port)
        
        # Get graph instance
        self.graph = self.client.select_graph(self.collection_name)
        
        # Initialize graph if needed
        self._initialize_graph()
    
    def _initialize_graph(self):
        """Initialize graph schema if needed"""
        try:
            # Check if the graph exists (implicit in select_graph)
            logger.info(f"Initializing FalkorDB graph: {self.collection_name}")
            
            # Check for existing indices to avoid duplicates
            try:
                indices = self.graph.list_indices().result_set
                has_id_index = any('id' in str(idx) for idx in indices)
                has_vector_index = any('embedding_vector' in str(idx) for idx in indices)
                
                # Create index on Memory nodes by ID if not exists
                if not has_id_index:
                    self.graph.create_node_range_index("Memory", "id")
                    logger.info("Created index on Memory(id)")
                else:
                    logger.info("Index on Memory(id) already exists")
                
                # Create vector index for embeddings if supported and not exists
                if not has_vector_index:
                    try:
                        self.graph.create_node_vector_index("Memory", "embedding_vector", 
                                                          dim=self.vector_size, 
                                                          similarity_function="cosine")
                        logger.info(f"Created vector index on Memory(embedding_vector) with dimension {self.vector_size}")
                    except Exception as vector_e:
                        # Vector index might not be supported
                        logger.warning(f"Could not create vector index: {vector_e}")
                else:
                    logger.info("Vector index on Memory(embedding_vector) already exists")
                    
            except Exception as idx_e:
                # If listing indices fails, try creating them anyway
                logger.warning(f"Error checking existing indices: {idx_e}")
                try:
                    self.graph.create_node_range_index("Memory", "id")
                    logger.info("Created index on Memory(id)")
                    
                    self.graph.create_node_vector_index("Memory", "embedding_vector", 
                                                      dim=self.vector_size, 
                                                      similarity_function="cosine")
                    logger.info(f"Created vector index on Memory(embedding_vector) with dimension {self.vector_size}")
                except Exception as e:
                    logger.warning(f"Could not create indices: {e}")
                
        except Exception as e:
            logger.error(f"Error initializing FalkorDB graph: {e}")
    
    def add_document(self, document: str, metadata: Dict, doc_id: str):
        """Add a document to FalkorDB.
        
        Args:
            document: Text content to add
            metadata: Dictionary of metadata
            doc_id: Unique identifier for the document
        """
        try:
            # Get embedding for document
            embedding = self.embedding_provider.get_embedding(document)
            
            # Process metadata to ensure it's properly formatted
            processed_metadata = {
                'id': doc_id,
                'content': document
            }
            
            # Process complex types to ensure they're serializable
            import json
            for key, value in metadata.items():
                if key not in ['id', 'content']:  # Avoid duplication
                    # Check the type and convert as needed
                    if isinstance(value, (list, dict)):
                        if not isinstance(value, str):
                            processed_metadata[key] = json.dumps(value)
                        else:
                            processed_metadata[key] = value
                    else:
                        processed_metadata[key] = value
            
            # Instead of using parameters with complex structures, build a more compatible query
            # with explicitly formatted properties
            props_list = []
            for key, value in processed_metadata.items():
                if key != 'embedding_vector':  # Handle embedding vector separately
                    if isinstance(value, str):
                        # Escape quotes in strings
                        escaped = value.replace("'", "\\'")
                        props_list.append(f"{key}: '{escaped}'")
                    elif isinstance(value, (int, float, bool)):
                        props_list.append(f"{key}: {value}")
                    else:
                        # Fallback for any other types
                        props_list.append(f"{key}: '{str(value).replace('\'', '\\\'')}'")
            
            # Add embedding as vector property, but only store small number of dimensions in properties
            props_list.append(f"embedding_vector: {embedding[:4]}")  # Store just first 4 dims in properties
            
            # Create node with explicit properties
            props_str = "{" + ", ".join(props_list) + "}"
            query = f"CREATE (m:Memory {props_str})"
            
            # Execute query
            result = self.graph.query(query)
            logger.info(f"Added document {doc_id} to FalkorDB")
            
            # Store full embedding as a separate property directly
            # This circumvents any size limitations in the query parameters
            embedding_json = json.dumps(embedding)
            update_query = f"MATCH (m:Memory {{id: '{doc_id}'}}) SET m.embedding = '{embedding_json}'"
            self.graph.query(update_query)
            
            return True
        except Exception as e:
            logger.error(f"Error adding document to FalkorDB: {e}")
            return False
    
    def delete_document(self, doc_id: str):
        """Delete a document from FalkorDB.
        
        Args:
            doc_id: ID of document to delete
        """
        try:
            # Delete node and all its relationships
            query = "MATCH (m:Memory {id: $id}) DETACH DELETE m"
            self.graph.query(query, params={"id": doc_id})
            logger.info(f"Deleted document {doc_id} from FalkorDB")
            return True
        except Exception as e:
            logger.error(f"Error deleting document from FalkorDB: {e}")
            return False
    
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
            
            # Fallback: Just use standard graph query
            # We avoid vector search for now as it's not widely supported in all FalkorDB versions
            query_all = "MATCH (m:Memory) RETURN m"
            result = self.graph.query(query_all)
            
            if not result.result_set or len(result.result_set) == 0:
                # No results
                return {
                    'ids': [[]],
                    'metadatas': [[]],
                    'distances': [[]],
                    'documents': [[]]
                }
            
            # Calculate similarities manually
            node_similarities = []
            import json
            for record in result.result_set:
                node = record[0]  # First column contains node data
                try:
                    # Node objects have a 'properties' dictionary attribute
                    if not hasattr(node, 'properties'):
                        logger.warning(f"Node doesn't have properties attribute: {type(node)}")
                        continue
                        
                    # Extract embedding from node properties
                    node_embedding_str = node.properties.get('embedding', '[]')
                    node_embedding = json.loads(node_embedding_str) if isinstance(node_embedding_str, str) else []
                    
                    # Skip if no valid embedding
                    if not node_embedding:
                        logger.warning(f"Node {node.properties.get('id', 'unknown')} has no valid embedding")
                        continue
                    
                    # Calculate similarity
                    similarity = float(cosine_similarity(
                        np.array(query_embedding).reshape(1, -1),
                        np.array(node_embedding).reshape(1, -1)
                    )[0][0])
                    
                    node_similarities.append((node, similarity))
                except Exception as inner_e:
                    logger.warning(f"Error calculating similarity for node: {inner_e}")
            
            # Sort by similarity (highest first)
            node_similarities.sort(key=lambda x: x[1], reverse=True)
            
            # Take top k
            top_results = node_similarities[:k]
            
            # Format results
            doc_ids = []
            metadatas = []
            distances = []
            documents = []
            
            for node, similarity in top_results:
                doc_ids.append(node.properties.get('id', ''))
                
                # Extract and process metadata
                metadata = {}
                for key, value in node.properties.items():
                    if key not in ['embedding', 'embedding_vector']:  # Don't include the embeddings in metadata
                        if isinstance(value, str) and (value.startswith('[') or value.startswith('{')):
                            # Try to parse JSON strings back to objects
                            try:
                                metadata[key] = json.loads(value)
                            except:
                                metadata[key] = value
                        else:
                            metadata[key] = value
                
                metadatas.append(metadata)
                distances.append(similarity)
                documents.append(node.properties.get('content', ''))
            
            # Format in ChromaDB-compatible structure
            return {
                'ids': [doc_ids],
                'metadatas': [metadatas],
                'distances': [distances],
                'documents': [documents]
            }
                
        except Exception as e:
            logger.error(f"Error searching in FalkorDB: {e}")
            # Return empty results in ChromaDB format
            return {
                'ids': [[]],
                'metadatas': [[]],
                'distances': [[]],
                'documents': [[]]
            }
    
    def _process_search_results(self, result_set, k):
        """Process search results into ChromaDB-compatible format"""
        doc_ids = []
        metadatas = []
        distances = []
        documents = []
        
        for record in result_set:
            if len(record) >= 2:  # Should have node and score
                node = record[0]
                similarity = record[1]
                
                if not hasattr(node, 'properties'):
                    logger.warning(f"Node doesn't have properties attribute in _process_search_results: {type(node)}")
                    continue
                
                doc_ids.append(node.properties.get('id', ''))
                
                # Extract and process metadata
                metadata = {}
                for key, value in node.properties.items():
                    if key not in ['embedding', 'embedding_vector']:  # Don't include embeddings
                        metadata[key] = value
                
                metadatas.append(metadata)
                distances.append(similarity)
                documents.append(node.properties.get('content', ''))
        
        # Format in ChromaDB-compatible structure
        return {
            'ids': [doc_ids],
            'metadatas': [metadatas],
            'distances': [distances],
            'documents': [documents]
        }

class QdrantRetriever:
    """Vector database retrieval using Qdrant"""
    
    def __init__(self, 
                collection_name: str = "memories", 
                host: str = "localhost", 
                port: int = 7333,
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
