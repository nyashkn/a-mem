from typing import Dict, Optional, Literal, Any, Union
import os
import json
from abc import ABC, abstractmethod
from litellm import completion

class BaseLLMController(ABC):
    @abstractmethod
    def get_completion(self, prompt: str, response_format: dict = None, temperature: float = 0.7) -> str:
        """Get completion from LLM"""
        pass

class OpenAIController(BaseLLMController):
    def __init__(self, model: str = "gpt-4", api_key: Optional[str] = None, base_url: Optional[str] = None):
        try:
            from openai import OpenAI
            self.model = model
            
            # Handle different API keys based on service
            if api_key is None:
                if base_url and "openrouter.ai" in base_url:
                    api_key = os.getenv('OPENROUTER_API_KEY')
                else:
                    api_key = os.getenv('OPENAI_API_KEY')
            
            if api_key is None:
                raise ValueError("API key not found. Set appropriate API key environment variable.")
            
            self.client = OpenAI(api_key=api_key, base_url=base_url)
        except ImportError:
            raise ImportError("OpenAI package not found. Install it with: pip install openai")
    
    def get_completion(self, prompt: str, response_format: dict = None, temperature: float = 0.7) -> str:
        messages = [
            {"role": "system", "content": "You must respond with a JSON object."},
            {"role": "user", "content": prompt}
        ]
        
        kwargs = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": 1000
        }
        
        if response_format:
            kwargs["response_format"] = response_format
            
        response = self.client.chat.completions.create(**kwargs)
        return response.choices[0].message.content

class LiteLLMController(BaseLLMController):
    def __init__(self, model: str = "gpt-4", api_key: Optional[str] = None, base_url: Optional[str] = None):
        """Initialize LiteLLM controller
        
        Args:
            model: Model identifier in LiteLLM format (e.g., 'openai/gpt-4', 'bedrock/anthropic.claude-v2')
            api_key: API key for the LLM provider
            base_url: Base URL for API endpoint
        """
        self.model = model
        
        # Set environment variables based on the model provider
        if api_key:
            provider = model.split('/')[0] if '/' in model else model
            if provider == "openai":
                os.environ["OPENAI_API_KEY"] = api_key
            elif provider == "anthropic":
                os.environ["ANTHROPIC_API_KEY"] = api_key
            elif provider == "openrouter":
                os.environ["OPENROUTER_API_KEY"] = api_key
            elif provider == "bedrock":
                os.environ["AWS_ACCESS_KEY_ID"] = os.getenv("AWS_ACCESS_KEY_ID")
                os.environ["AWS_SECRET_ACCESS_KEY"] = os.getenv("AWS_SECRET_ACCESS_KEY")
                os.environ["AWS_REGION_NAME"] = os.getenv("AWS_REGION_NAME")
            else:
                # Generic case
                os.environ[f"{provider.upper()}_API_KEY"] = api_key
        
        if base_url:
            os.environ[f"{provider.upper()}_API_BASE"] = base_url
    
    def _generate_empty_response(self, response_format: dict) -> dict:
        """Generate empty response based on the schema"""
        if "json_schema" not in response_format:
            return {}
            
        schema = response_format["json_schema"]["schema"]
        result = {}
        
        if "properties" in schema:
            for prop_name, prop_schema in schema["properties"].items():
                result[prop_name] = self._generate_empty_value(prop_schema.get("type", "string"), 
                                                            prop_schema.get("items"))
        
        return result
    
    def _generate_empty_value(self, schema_type: str, schema_items: dict = None) -> Any:
        """Generate empty value based on schema type"""
        if schema_type == "array":
            return []
        elif schema_type == "string":
            return ""
        elif schema_type == "object":
            return {}
        elif schema_type == "number":
            return 0
        elif schema_type == "boolean":
            return False
        return None
    
    def get_completion(self, prompt: str, response_format: dict = None, temperature: float = 0.7) -> str:
        try:
            kwargs = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": "You must respond with a JSON object."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": temperature
            }
            
            if response_format:
                kwargs["response_format"] = response_format
            
            response = completion(**kwargs)
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error in LiteLLM completion: {e}")
            if response_format:
                empty_response = self._generate_empty_response(response_format)
                return json.dumps(empty_response)
            return "{}"

class OllamaController(BaseLLMController):
    def __init__(self, model: str = "llama2"):
        from ollama import chat
        self.model = model
    
    def _generate_empty_value(self, schema_type: str, schema_items: dict = None) -> Any:
        if schema_type == "array":
            return []
        elif schema_type == "string":
            return ""
        elif schema_type == "object":
            return {}
        elif schema_type == "number":
            return 0
        elif schema_type == "boolean":
            return False
        return None

    def _generate_empty_response(self, response_format: dict) -> dict:
        if "json_schema" not in response_format:
            return {}
            
        schema = response_format["json_schema"]["schema"]
        result = {}
        
        if "properties" in schema:
            for prop_name, prop_schema in schema["properties"].items():
                result[prop_name] = self._generate_empty_value(prop_schema["type"], 
                                                            prop_schema.get("items"))
        
        return result

    def get_completion(self, prompt: str, response_format: dict = None, temperature: float = 0.7) -> str:
        try:
            response = completion(
                model="ollama_chat/{}".format(self.model),
                messages=[
                    {"role": "system", "content": "You must respond with a JSON object."},
                    {"role": "user", "content": prompt}
                ],
                response_format=response_format,
            )
            return response.choices[0].message.content
        except Exception as e:
            empty_response = self._generate_empty_response(response_format)
            return json.dumps(empty_response)

class LLMController:
    """LLM-based controller for memory metadata generation"""
    def __init__(self, 
                 backend: Literal["openai", "ollama", "litellm"] = "openai",
                 model: str = "gpt-4", 
                 api_key: Optional[str] = None,
                 base_url: Optional[str] = None):
        if backend == "openai":
            self.llm = OpenAIController(model, api_key, base_url)
        elif backend == "ollama":
            self.llm = OllamaController(model)
        elif backend == "litellm":
            self.llm = LiteLLMController(model, api_key, base_url)
        else:
            raise ValueError("Backend must be one of: 'openai', 'ollama', 'litellm'")
            
    def get_completion(self, prompt: str, response_format: dict = None, temperature: float = 0.7) -> str:
        return self.llm.get_completion(prompt, response_format, temperature)
