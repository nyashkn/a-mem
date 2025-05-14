import keyword
from typing import List, Dict, Optional, Any, Tuple, Union, Literal
import uuid
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field, ConfigDict
from amem.config import load_config
from amem.llm_controller import LLMController
from amem.retrievers import FalkorDBRetriever
from amem.embedding.providers import EmbeddingProvider
from amem.factory import EmbeddingProviderFactory, RetrieverFactory, LLMControllerFactory
import json
import logging
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os
from nltk.tokenize import word_tokenize
import time

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

class RelationshipType(str, Enum):
    """Base relationship types, extensible by the mem_librarian."""
    CONTAINS_ENTITY = "CONTAINS_ENTITY"  # Memory contains reference to entity
    MEMORY_LINKS = "MEMORY_LINKS"  # Memory links to another memory
    REASONING_PATH = "REASONING_PATH"  # Memory is part of a reasoning path
    SIMILAR_TO = "SIMILAR_TO"  # Memory is semantically similar to another
    CONTRADICTS = "CONTRADICTS"  # Memory contradicts another memory
    SUPPORTS = "SUPPORTS"  # Memory supports/reinforces another memory
    TEMPORAL_FOLLOWS = "TEMPORAL_FOLLOWS"  # Memory chronologically follows another
    QUERY_RETRIEVED = "QUERY_RETRIEVED"  # Memory was retrieved by this query


class MemoryNote(BaseModel):
    """A memory note that represents a single unit of information in the memory system.
    
    This class encapsulates all metadata associated with a memory, including:
    - Core content and identifiers
    - Temporal information (creation and access times)
    - Semantic metadata (keywords, context, tags)
    - Usage statistics (retrieval count)
    """
    
    # Core content and ID
    content: str
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    
    # Semantic metadata
    keywords: List[str] = Field(default_factory=list)
    links: List[str] = Field(default_factory=list)  # Legacy field - will use graph relationships
    context: str = "General"
    category: str = "Uncategorized"
    tags: List[str] = Field(default_factory=list)
    
    # Temporal information
    timestamp: str = Field(default_factory=lambda: datetime.now().strftime("%Y%m%d%H%M"))
    last_accessed: str = Field(default_factory=lambda: datetime.now().strftime("%Y%m%d%H%M"))
    
    # Usage data
    retrieval_count: int = 0
    
    # Make the model immutable
    model_config = ConfigDict(frozen=True)
    
    def to_metadata_dict(self) -> Dict[str, Any]:
        """Convert to a dictionary suitable for database storage."""
        return self.model_dump()
    
    @classmethod
    def from_metadata_dict(cls, metadata: Dict[str, Any]) -> "MemoryNote":
        """Create a MemoryNote from metadata dictionary."""
        # Process complex types if they're stored as JSON strings
        processed_metadata = {}
        for key, value in metadata.items():
            if isinstance(value, str) and (value.startswith('[') or value.startswith('{')):
                try:
                    processed_metadata[key] = json.loads(value)
                except:
                    processed_metadata[key] = value
            else:
                processed_metadata[key] = value
        
        return cls(**processed_metadata)


class Entity(BaseModel):
    """Entity identified across memories."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    type: str  # "PERSON", "ORGANIZATION", "PROJECT", "LOCATION", "CONCEPT"
    aliases: List[str] = Field(default_factory=list)
    first_seen: str = Field(default_factory=lambda: datetime.now().strftime("%Y%m%d%H%M"))
    confidence: float = 1.0
    
    # Make the model immutable
    model_config = ConfigDict(frozen=True)


class MemoryEvent(BaseModel):
    """Event log entry for memory system operations."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = Field(default_factory=lambda: datetime.now().strftime("%Y%m%d%H%M"))
    event_type: str  # "query", "evolution", "relationship_creation", etc.
    details: Dict[str, Any] = Field(default_factory=dict)  # Flexible event details
    agent_id: Optional[str] = None  # Which agent/user triggered this event
    
    # Make the model immutable
    model_config = ConfigDict(frozen=True)

class AgenticMemorySystem:
    """Core memory system that manages memory notes and their evolution.
    
    This system provides:
    - Memory creation, retrieval, update, and deletion
    - Content analysis and metadata extraction
    - Memory evolution and relationship management
    - Hybrid search capabilities
    """
    
    def __init__(self, 
                 config: Optional[Dict[str, Any]] = None,
                 embedding_provider: Optional[EmbeddingProvider] = None,
                 retriever: Optional[FalkorDBRetriever] = None,
                 llm_controller: Optional[LLMController] = None,
                 evo_threshold: int = 100):
        """Initialize the memory system.
        
        Args:
            config: Configuration dictionary (if None, will load from .env)
            embedding_provider: Optional pre-configured embedding provider 
            retriever: Optional pre-configured vector database retriever
            llm_controller: Optional pre-configured LLM controller
            model_name: Name of the embedding model (deprecated, use config)
            llm_backend: LLM backend to use (deprecated, use config)
            llm_model: Name of the LLM model (deprecated, use config)
            evo_threshold: Number of memories before triggering evolution
            api_key: API key for the LLM service (deprecated, use config)
            base_url: Base URL for LLM API (deprecated, use config)
        """
        self.memories = {}
        self.evo_cnt = 0
        self.evo_threshold = evo_threshold
        
        # Load configuration
        if config is None:
            config = load_config()
        self.config = config
        
        # Use provided embedding provider or create from factory
        try:
            self.embedding_provider = embedding_provider or EmbeddingProviderFactory.create(self.config)
        except Exception as e:
            logger.error(f"Error initializing embedding provider: {e}")
            raise RuntimeError(f"Failed to initialize embedding provider: {e}")
        
        # For retriever, we need to pass the embedding provider if it's not provided
        if retriever:
            self.retriever = retriever
        else:
            try:
                self.retriever = RetrieverFactory.create(
                    config=self.config, 
                    embedding_provider=self.embedding_provider
                )
            except Exception as e:
                logger.error(f"Error initializing retriever: {e}")
                # Raise error if retriever initialization fails
                logger.error("Failed to initialize FalkorDB retriever")
                raise e
            
        # For LLM controller
        self.llm_controller = llm_controller or LLMControllerFactory.create(self.config)
        
        # Event store
        self.events = []

        # Evolution system prompt
        self._evolution_system_prompt = '''
                                You are an AI memory evolution agent responsible for managing and evolving a knowledge base.
                                Analyze the the new memory note according to keywords and context, also with their several nearest neighbors memory.
                                Make decisions about its evolution.  

                                The new memory context:
                                {context}
                                content: {content}
                                keywords: {keywords}

                                The nearest neighbors memories:
                                {nearest_neighbors_memories}

                                Based on this information, determine:
                                1. Should this memory be evolved? Consider its relationships with other memories.
                                2. What specific actions should be taken (strengthen, update_neighbor)?
                                   2.1 If choose to strengthen the connection, which memory should it be connected to? Can you give the updated tags of this memory?
                                   2.2 If choose to update_neighbor, you can update the context and tags of these memories based on the understanding of these memories. If the context and the tags are not updated, the new context and tags should be the same as the original ones. Generate the new context and tags in the sequential order of the input neighbors.
                                Tags should be determined by the content of these characteristic of these memories, which can be used to retrieve them later and categorize them.
                                Note that the length of new_tags_neighborhood must equal the number of input neighbors, and the length of new_context_neighborhood must equal the number of input neighbors.
                                The number of neighbors is {neighbor_number}.
                                Return your decision in JSON format with the following structure:
                                {{
                                    "should_evolve": True or False,
                                    "actions": ["strengthen", "update_neighbor"],
                                    "suggested_connections": ["neighbor_memory_ids"],
                                    "tags_to_update": ["tag_1",..."tag_n"], 
                                    "new_context_neighborhood": ["new context",...,"new context"],
                                    "new_tags_neighborhood": [["tag_1",...,"tag_n"],...["tag_1",...,"tag_n"]],
                                }}
                                '''
        
    def analyze_content(self, content: str) -> Dict:            
        """Analyze content using LLM to extract semantic metadata.
        
        Uses a language model to understand the content and extract:
        - Keywords: Important terms and concepts
        - Context: Overall domain or theme
        - Tags: Classification categories
        
        Args:
            content (str): The text content to analyze
            
        Returns:
            Dict: Contains extracted metadata with keys:
                - keywords: List[str]
                - context: str
                - tags: List[str]
        """
        prompt = """Generate a structured analysis of the following content by:
            1. Identifying the most salient keywords (focus on nouns, verbs, and key concepts)
            2. Extracting core themes and contextual elements
            3. Creating relevant categorical tags

            Format the response as a JSON object:
            {
                "keywords": [
                    // several specific, distinct keywords that capture key concepts and terminology
                    // Order from most to least important
                    // Don't include keywords that are the name of the speaker or time
                    // At least three keywords, but don't be too redundant.
                ],
                "context": 
                    // one sentence summarizing:
                    // - Main topic/domain
                    // - Key arguments/points
                    // - Intended audience/purpose
                ,
                "tags": [
                    // several broad categories/themes for classification
                    // Include domain, format, and type tags
                    // At least three tags, but don't be too redundant.
                ]
            }

            Content for analysis:
            """ + content
        try:
            response = self.llm_controller.llm.get_completion(prompt, response_format={"type": "json_schema", "json_schema": {
                        "name": "response",
                        "schema": {
                            "type": "object",
                            "properties": {
                                "keywords": {
                                    "type": "array",
                                    "items": {
                                        "type": "string"
                                    }
                                },
                                "context": {
                                    "type": "string",
                                },
                                "tags": {
                                    "type": "array",
                                    "items": {
                                        "type": "string"
                                    }
                                }
                            }
                        }
                    }})
            return json.loads(response)
        except Exception as e:
            print(f"Error analyzing content: {e}")
            return {"keywords": [], "context": "General", "tags": []}

    def add_note(self, content: str, time: str = None, **kwargs) -> str:
        """Add a new memory note"""
        # Create MemoryNote using Pydantic
        note_params = {"content": content}
        if time is not None:
            note_params['timestamp'] = time
        note_params.update(kwargs)
        
        try:
            # Create immutable note
            note = MemoryNote(**note_params)
            
            # Extract metadata and analyze content if needed
            if not note.keywords and not kwargs.get('keywords'):
                # Analyze content using LLM
                analysis = self.analyze_content(content)
                # Create a new note with the analysis results
                note = MemoryNote(**{
                    **note.model_dump(),
                    "keywords": analysis.get("keywords", []),
                    "context": analysis.get("context", note.context),
                    "tags": analysis.get("tags", note.tags),
                })
            
            # Process for evolution
            evo_label, processed_note = self.process_memory(note)
            
            # Store the processed note
            self.memories[processed_note.id] = processed_note
            
            # Add to vector database
            metadata = processed_note.to_metadata_dict()
            self.retriever.add_document(processed_note.content, metadata, processed_note.id)
            
            # Record memory creation event
            self.add_memory_event(
                "memory_creation",
                processed_note.id,
                {}
            )
            
            # Check if evolution threshold reached
            if evo_label:
                self.evo_cnt += 1
                if self.evo_cnt % self.evo_threshold == 0:
                    self.consolidate_memories()
                    
            return processed_note.id
        except Exception as e:
            logger.error(f"Error adding memory note: {e}")
            # Remove from in-memory store if present but adding to vector DB failed
            if 'note' in locals() and hasattr(note, 'id') and note.id in self.memories:
                del self.memories[note.id]
            raise RuntimeError(f"Failed to add memory note: {e}")
    
    def consolidate_memories(self):
        """Consolidate memories: update retriever with new documents"""
        try:
            # Get current retriever configuration
            current_retriever_type = type(self.retriever)
            current_collection = self.retriever.collection_name
            current_host = self.retriever.client.host
            current_port = self.retriever.client.port
            
            # Create a new FalkorDB retriever
            self.retriever = FalkorDBRetriever(
                collection_name=current_collection,
                host=current_host,
                port=current_port,
                embedding_provider=self.embedding_provider
            )
            logger.info(f"Created new FalkorDB retriever for consolidation")
            
            # Re-add all memory documents with their complete metadata
            for memory in self.memories.values():
                # Convert Pydantic model to metadata dict
                metadata = memory.to_metadata_dict()
                self.retriever.add_document(memory.content, metadata, memory.id)
                
            logger.info(f"Successfully consolidated {len(self.memories)} memories")
        except Exception as e:
            logger.error(f"Error consolidating memories: {e}")
    
    def find_related_memories(self, query: str, k: int = 5) -> Tuple[str, List[int]]:
        """Find related memories using vector retrieval"""
        if not self.memories:
            return "", []
            
        try:
            # Get results from vector database
            results = self.retriever.search(query, k)
            
            # Convert to list of memories
            memory_str = ""
            indices = []
            
            if 'ids' in results and results['ids'] and len(results['ids']) > 0 and len(results['ids'][0]) > 0:
                for i, doc_id in enumerate(results['ids'][0]):
                    # Get metadata from search results
                    if i < len(results['metadatas'][0]):
                        metadata = results['metadatas'][0][i]
                        # Format memory string
                        memory_str += f"memory index:{i}\ttalk start time:{metadata.get('timestamp', '')}\tmemory content: {metadata.get('content', '')}\tmemory context: {metadata.get('context', '')}\tmemory keywords: {str(metadata.get('keywords', []))}\tmemory tags: {str(metadata.get('tags', []))}\n"
                        indices.append(i)
                    
            return memory_str, indices
        except Exception as e:
            logger.error(f"Error in find_related_memories: {str(e)}")
            return "", []

    def find_related_memories_raw(self, query: str, k: int = 5) -> str:
        """Find related memories using vector retrieval in raw format"""
        if not self.memories:
            return ""
        
        try:
            # Get results from vector database
            results = self.retriever.search(query, k)
            
            # Convert to list of memories
            memory_str = ""
            
            if 'ids' in results and results['ids'] and len(results['ids']) > 0:
                for i, doc_id in enumerate(results['ids'][0][:k]):
                    if i < len(results['metadatas'][0]):
                        # Get metadata from search results
                        metadata = results['metadatas'][0][i]
                        
                        # Add main memory info
                        memory_str += f"talk start time:{metadata.get('timestamp', '')}\tmemory content: {metadata.get('content', '')}\tmemory context: {metadata.get('context', '')}\tmemory keywords: {str(metadata.get('keywords', []))}\tmemory tags: {str(metadata.get('tags', []))}\n"
                        
                        # Add linked memories if available
                        links = metadata.get('links', [])
                        j = 0
                        for link_id in links:
                            if link_id in self.memories and j < k:
                                neighbor = self.memories[link_id]
                                memory_str += f"talk start time:{neighbor.timestamp}\tmemory content: {neighbor.content}\tmemory context: {neighbor.context}\tmemory keywords: {str(neighbor.keywords)}\tmemory tags: {str(neighbor.tags)}\n"
                                j += 1
                                
            return memory_str
        except Exception as e:
            logger.error(f"Error in find_related_memories_raw: {e}")
            return ""

    def read(self, memory_id: str) -> Optional[MemoryNote]:
        """Retrieve a memory note by its ID.
        
        Args:
            memory_id (str): ID of the memory to retrieve
            
        Returns:
            MemoryNote if found, None otherwise
        """
        return self.memories.get(memory_id)
    
    def update(self, memory_id: str, **kwargs) -> bool:
        """Update a memory note.
        
        Args:
            memory_id: ID of memory to update
            **kwargs: Fields to update
            
        Returns:
            bool: True if update successful
        """
        if memory_id not in self.memories:
            return False
            
        note = self.memories[memory_id]
        
        # Create a new immutable object with updated fields
        # First, get existing values
        update_dict = note.model_dump()
        
        # Then apply updates for fields that exist
        for key, value in kwargs.items():
            if key in update_dict:
                update_dict[key] = value
        
        # Create new immutable note with updates
        updated_note = MemoryNote(**update_dict)
        
        try:
            # Delete and re-add to update in vector database
            self.retriever.delete_document(memory_id)
            self.retriever.add_document(
                document=updated_note.content, 
                metadata=updated_note.to_metadata_dict(), 
                doc_id=memory_id
            )
            
            # Update in-memory storage
            self.memories[memory_id] = updated_note
            
            # Record update event
            self.add_memory_event(
                "memory_update",
                memory_id,
                {
                    "updated_fields": list(kwargs.keys())
                }
            )
            
            return True
        except Exception as e:
            logger.error(f"Error updating memory: {e}")
            return False
    
    def add_memory_event(self, event_type: str, memory_id: str, details: Dict[str, Any]) -> str:
        """Record a memory system event.
        
        Args:
            event_type: Type of event (e.g., "memory_creation", "query", "evolution")
            memory_id: ID of the related memory
            details: Additional event details
            
        Returns:
            str: ID of the created event
        """
        event = MemoryEvent(
            event_type=event_type,
            details={
                "memory_id": memory_id,
                **details
            }
        )
        
        # Add to in-memory event store
        self.events.append(event)
        
        # Log the event
        logger.debug(f"Event recorded: {event_type} for memory {memory_id}")
        
        return event.id
    
    def delete(self, memory_id: str) -> bool:
        """Delete a memory note by its ID.
        
        Args:
            memory_id (str): ID of the memory to delete
            
        Returns:
            bool: True if memory was deleted, False if not found
        """
        if memory_id in self.memories:
            try:
                # Delete from vector database
                self.retriever.delete_document(memory_id)
                # Delete from local storage
                del self.memories[memory_id]
                
                # Record deletion event
                self.add_memory_event("memory_deletion", memory_id, {})
                
                return True
            except Exception as e:
                logger.error(f"Error deleting memory: {e}")
                return False
        return False
    
    def _search_raw(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Internal search method that returns raw results from vector database.
        
        This is used internally by the memory evolution system to find
        related memories for potential evolution.
        
        Args:
            query (str): The search query text
            k (int): Maximum number of results to return
            
        Returns:
            List[Dict[str, Any]]: Raw search results from vector database
        """
        try:
            results = self.retriever.search(query, k)
            return [{'id': doc_id, 'score': score} 
                    for doc_id, score in zip(results['ids'][0], results['distances'][0])]
        except Exception as e:
            logger.error(f"Error in _search_raw: {e}")
            return []
                
    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Search for memories using vector retrieval.
        
        This method uses:
        1. FalkorDB for semantic similarity search
        
        The results are deduplicated and ranked by relevance.
        
        Args:
            query (str): The search query text
            k (int): Maximum number of results to return
            
        Returns:
            List[Dict[str, Any]]: List of search results, each containing:
                - id: Memory ID
                - content: Memory content
                - score: Similarity score
                - metadata: Additional memory metadata
        """
        try:
            # Get results from vector search
            search_results = self.retriever.search(query, k)
            memories = []
            
            # Process search results
            for i, doc_id in enumerate(search_results['ids'][0]):
                memory = self.memories.get(doc_id)
                if memory:
                    memories.append({
                        'id': doc_id,
                        'content': memory.content,
                        'context': memory.context,
                        'keywords': memory.keywords,
                        'score': search_results['distances'][0][i]
                    })
            
            # Return the vector search results
            return memories[:k]
        except Exception as e:
            logger.error(f"Error in search: {e}")
            return []
    
    def _search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Search for memories using vector retrieval.
        
        This method uses:
        1. FalkorDB for semantic similarity search
        
        The results are deduplicated and ranked by relevance.
        
        Args:
            query (str): The search query text
            k (int): Maximum number of results to return
            
        Returns:
            List[Dict[str, Any]]: List of search results, each containing:
                - id: Memory ID
                - content: Memory content
                - score: Similarity score
                - metadata: Additional memory metadata
        """
        try:
            # Get results from vector search
            search_results = self.retriever.search(query, k)
            memories = []
            
            # Process search results
            for i, doc_id in enumerate(search_results['ids'][0]):
                memory = self.memories.get(doc_id)
                if memory:
                    memories.append({
                        'id': doc_id,
                        'content': memory.content,
                        'context': memory.context,
                        'keywords': memory.keywords,
                        'score': search_results['distances'][0][i]
                    })
                    
            # Get results from embedding retriever
            embedding_results = self.retriever.search(query, k)
            
            # Combine results with deduplication
            seen_ids = set(m['id'] for m in memories)
            for result in embedding_results:
                memory_id = result.get('id')
                if memory_id and memory_id not in seen_ids:
                    memory = self.memories.get(memory_id)
                    if memory:
                        memories.append({
                            'id': memory_id,
                            'content': memory.content,
                            'context': memory.context,
                            'keywords': memory.keywords,
                            'score': result.get('score', 0.0)
                        })
                        seen_ids.add(memory_id)
                        
            return memories[:k]
        except Exception as e:
            logger.error(f"Error in _search: {e}")
            return []

    def search_agentic(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Search for memories using vector retrieval."""
        if not self.memories:
            return []
            
        try:
            # Get results from vector database
            results = self.retriever.search(query, k)
            
            # Process results
            memories = []
            seen_ids = set()
            
            # Check if we have valid results
            if ('ids' not in results or not results['ids'] or 
                len(results['ids']) == 0 or len(results['ids'][0]) == 0):
                return []
                
            # Process search results
            for i, doc_id in enumerate(results['ids'][0][:k]):
                if doc_id in seen_ids:
                    continue
                    
                if i < len(results['metadatas'][0]):
                    metadata = results['metadatas'][0][i]
                    
                    # Create result dictionary with all metadata fields
                    memory_dict = {
                        'id': doc_id,
                        'content': metadata.get('content', ''),
                        'context': metadata.get('context', ''),
                        'keywords': metadata.get('keywords', []),
                        'tags': metadata.get('tags', []),
                        'timestamp': metadata.get('timestamp', ''),
                        'category': metadata.get('category', 'Uncategorized'),
                        'is_neighbor': False
                    }
                    
                    # Add score if available
                    if 'distances' in results and len(results['distances']) > 0 and i < len(results['distances'][0]):
                        memory_dict['score'] = results['distances'][0][i]
                        
                    memories.append(memory_dict)
                    seen_ids.add(doc_id)
            
            # Add linked memories (neighbors)
            neighbor_count = 0
            for memory in list(memories):  # Use a copy to avoid modification during iteration
                if neighbor_count >= k:
                    break
                    
                # Get links from metadata
                links = memory.get('links', [])
                if not links and 'id' in memory:
                    # Try to get links from memory object
                    mem_obj = self.memories.get(memory['id'])
                    if mem_obj:
                        links = mem_obj.links
                        
                for link_id in links:
                    if link_id not in seen_ids and neighbor_count < k:
                        neighbor = self.memories.get(link_id)
                        if neighbor:
                            memories.append({
                                'id': link_id,
                                'content': neighbor.content,
                                'context': neighbor.context,
                                'keywords': neighbor.keywords,
                                'tags': neighbor.tags,
                                'timestamp': neighbor.timestamp,
                                'category': neighbor.category,
                                'is_neighbor': True
                            })
                            seen_ids.add(link_id)
                            neighbor_count += 1
            
            return memories[:k]
        except Exception as e:
            logger.error(f"Error in search_agentic: {str(e)}")
            return []

    def process_memory(self, note: MemoryNote) -> Tuple[bool, MemoryNote]:
        """Process a memory note and determine if it should evolve.
        
        Args:
            note: The memory note to process
            
        Returns:
            Tuple[bool, MemoryNote]: (should_evolve, processed_note)
        """
        # For first memory or testing, just return the note without evolution
        if not self.memories:
            return False, note
            
        try:
            # Get nearest neighbors
            neighbors_text, indices = self.find_related_memories(note.content, k=5)
            if not neighbors_text or not indices:
                return False, note
                
            # Format neighbors for LLM - in this case, neighbors_text is already formatted
            
            # Query LLM for evolution decision
            prompt = self._evolution_system_prompt.format(
                content=note.content,
                context=note.context,
                keywords=note.keywords,
                nearest_neighbors_memories=neighbors_text,
                neighbor_number=len(indices)
            )
            
            try:
                response = self.llm_controller.llm.get_completion(
                    prompt,
                    response_format={"type": "json_schema", "json_schema": {
                        "name": "response",
                        "schema": {
                            "type": "object",
                            "properties": {
                                "should_evolve": {
                                    "type": "boolean"
                                },
                                "actions": {
                                    "type": "array",
                                    "items": {
                                        "type": "string"
                                    }
                                },
                                "suggested_connections": {
                                    "type": "array",
                                    "items": {
                                        "type": "string"
                                    }
                                },
                                "new_context_neighborhood": {
                                    "type": "array",
                                    "items": {
                                        "type": "string"
                                    }
                                },
                                "tags_to_update": {
                                    "type": "array",
                                    "items": {
                                        "type": "string"
                                    }
                                },
                                "new_tags_neighborhood": {
                                    "type": "array",
                                    "items": {
                                        "type": "array",
                                        "items": {
                                            "type": "string"
                                        }
                                    }
                                }
                            },
                            "required": ["should_evolve", "actions", "suggested_connections", 
                                      "tags_to_update", "new_context_neighborhood", "new_tags_neighborhood"],
                            "additionalProperties": False
                        },
                        "strict": True
                    }}
                )
                
                response_json = json.loads(response)
                should_evolve = response_json["should_evolve"]
                
                # Only process if evolution is needed
                if should_evolve:
                    actions = response_json["actions"]
                    updated_note = note  # Start with original note
                    
                    for action in actions:
                        if action == "strengthen":
                            # Get suggested connections and tags
                            suggest_connections = response_json["suggested_connections"]
                            new_tags = response_json["tags_to_update"]
                            
                            # Create a new immutable note with updated connections and tags
                            # Combine existing links with new connections (deduplicate if needed)
                            combined_links = list(set(updated_note.links + suggest_connections))
                            
                            # Create new immutable object with updated fields
                            updated_note = MemoryNote(**{
                                **updated_note.model_dump(),
                                "links": combined_links,
                                "tags": new_tags
                            })
                            
                            # Log the strengthening event
                            self.add_memory_event(
                                "memory_evolution_strengthen",
                                updated_note.id,
                                {
                                    "added_connections": suggest_connections,
                                    "new_tags": new_tags
                                }
                            )
                            
                        elif action == "update_neighbor":
                            # Get context and tags for neighbors
                            new_context_neighborhood = response_json["new_context_neighborhood"]
                            new_tags_neighborhood = response_json["new_tags_neighborhood"]
                            
                            # Get the list of memories and their IDs
                            memory_list = list(self.memories.values())
                            memory_ids = list(self.memories.keys())
                            
                            # Update each neighbor with new context and tags
                            for i in range(min(len(indices), len(new_tags_neighborhood))):
                                # Skip if index is out of range
                                if i >= len(indices):
                                    continue
                                
                                # Get new tags and context for this neighbor
                                new_tags = new_tags_neighborhood[i]
                                if i < len(new_context_neighborhood):
                                    new_context = new_context_neighborhood[i]
                                else:
                                    # Keep existing context if no new one provided
                                    if i < len(memory_list):
                                        new_context = memory_list[i].context
                                    else:
                                        continue
                                
                                # Get the memory from indices
                                if i < len(indices):
                                    memory_idx = indices[i]
                                    # Check if index is valid
                                    if memory_idx < len(memory_list):
                                        old_memory = memory_list[memory_idx]
                                        
                                        # Create updated memory with immutable model
                                        updated_memory = MemoryNote(**{
                                            **old_memory.model_dump(),
                                            "tags": new_tags,
                                            "context": new_context
                                        })
                                        
                                        # Check if the ID is valid
                                        if memory_idx < len(memory_ids):
                                            memory_id = memory_ids[memory_idx]
                                            # Update in-memory store
                                            self.memories[memory_id] = updated_memory
                                            
                                            # Record the update
                                            self.add_memory_event(
                                                "memory_evolution_update_neighbor",
                                                memory_id,
                                                {
                                                    "new_tags": new_tags,
                                                    "new_context": new_context
                                                }
                                            )
                                
                return should_evolve, note
                
            except (json.JSONDecodeError, KeyError, Exception) as e:
                logger.error(f"Error in memory evolution: {str(e)}")
                return False, note
                
        except Exception as e:
            # For testing purposes, catch all exceptions and return the original note
            logger.error(f"Error in process_memory: {str(e)}")
            return False, note
