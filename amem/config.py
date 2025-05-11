import os
from typing import Dict, Any, Optional
from dotenv import load_dotenv

def load_config() -> Dict[str, Any]:
    """Load configuration from environment variables"""
    # Load .env file if it exists
    load_dotenv()
    
    config = {
        # Vector DB configuration
        "vector_db": {
            "type": os.getenv("VECTOR_DB_TYPE", "qdrant"),
            "qdrant": {
                "host": os.getenv("QDRANT_HOST", "localhost"),
                "port": int(os.getenv("QDRANT_PORT", "7333")),
                "collection": os.getenv("QDRANT_COLLECTION", "memories")
            }
        },
        
        # Embedding configuration
        "embedding": {
            "provider": os.getenv("EMBEDDING_PROVIDER", "sentence_transformer"),
            "model": os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2"),
            "api_key": os.getenv("EMBEDDING_API_KEY")
        },
        
        # LLM configuration
        "llm": {
            "backend": os.getenv("LLM_BACKEND", "openai"),
            "model": os.getenv("LLM_MODEL", "gpt-4o-mini"),
            "api_key": os.getenv("LLM_API_KEY"),
            "base_url": os.getenv("LLM_BASE_URL")
        }
    }
    
    return config
