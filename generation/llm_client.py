# DEPENDENCIES
import os
import json
import time
import aiohttp
import requests
from typing import List
from typing import Dict
from typing import Optional
from typing import AsyncGenerator
from config.models import LLMProvider
from config.settings import get_settings
from config.logging_config import get_logger
from utils.error_handler import handle_errors
from utils.error_handler import LLMClientError


# Setup Settings and Logging
settings = get_settings()
logger   = get_logger(__name__)


class LLMClient:
    """
    Unified LLM client supporting multiple providers (Ollama, OpenAI): Provides consistent interface for text generation across different LLM services
    """
    def __init__(self, provider: LLMProvider = None, model_name: str = None, api_key: str = None, base_url: str = None):
        """
        Initialize LLM client
        
        Arguments:
        ----------
            provider   { LLMProvider } : LLM provider to use
            
            model_name { str }         : Model name to use
            
            api_key    { str }         : API key (for OpenAI)
            
            base_url   { str }         : Base URL for API (for Ollama)
        """
        self.logger     = logger
        self.settings   = get_settings()
        self.provider   = provider or LLMProvider.OLLAMA
        self.model_name = model_name or self.settings.OLLAMA_MODEL
        self.api_key    = api_key
        self.base_url   = base_url or self.settings.OLLAMA_BASE_URL
        self.timeout    = self.settings.OLLAMA_TIMEOUT
        
        # Initialize provider-specific configurations
        self._initialize_provider()
    

    def _initialize_provider(self):
        """
        Initialize provider-specific configurations
        """
        # Auto-detect provider if not explicitly set
        # if (self.settings.IS_HF_SPACE and not self.settings.OLLAMA_ENABLED):
        #     if (self.settings.USE_OPENAI and self.settings.OPENAI_API_KEY):
        #         self.provider = LLMProvider.OPENAI
        #         logger.info("HF Space detected: Using OpenAI API")
            
        #     else:
        #         raise LLMClientError("Running in HF Space without Ollama. Set OPENAI_API_KEY in Space secrets.")
        
        # Provider initialization
        if (self.provider == LLMProvider.OLLAMA):
            if not self.base_url:
                raise LLMClientError("Ollama base URL is required")
            
            self.logger.info(f"Initialized Ollama client: {self.base_url}, model: {self.model_name}")
        
        elif (self.provider == LLMProvider.OPENAI):
            if not self.api_key:
                # Try to get from environment
                self.api_key = os.getenv('OPENAI_API_KEY')
                if not self.api_key:
                    raise LLMClientError("OpenAI API key is required")
            
            self.base_url = "https://api.openai.com/v1"

            self.logger.info(f"Initialized OpenAI client, model: {self.model_name}")
        
        else:
            raise LLMClientError(f"Unsupported provider: {self.provider}")
    

    async def generate(self, messages: List[Dict], **generation_params) -> Dict:
        """
        Generate text completion (async)
        
        Arguments:
        ----------
            messages          { list } : List of message dictionaries
            
            **generation_params        : Generation parameters (temperature, max_tokens, etc.)
        
        Returns:
        --------
                   { dict }            : Generation response
        """
        try:
            if (self.provider == LLMProvider.OLLAMA):
                return await self._generate_ollama(messages, **generation_params)
            
            elif (self.provider == LLMProvider.OPENAI):
                return await self._generate_openai(messages, **generation_params)
            
            else:
                raise LLMClientError(f"Unsupported provider: {self.provider}")
        
        except Exception as e:
            self.logger.error(f"Generation failed: {repr(e)}")
            raise LLMClientError(f"Generation failed: {repr(e)}")
    

    async def generate_stream(self, messages: List[Dict], **generation_params) -> AsyncGenerator[str, None]:
        """
        Generate text completion with streaming (async)
        
        Arguments:
        ----------
            messages          { list } : List of message dictionaries
            
            **generation_params        : Generation parameters
        
        Returns:
        --------
                   { AsyncGenerator }  : Async generator yielding response chunks
        """
        try:
            if (self.provider == LLMProvider.OLLAMA):
                async for chunk in self._generate_ollama_stream(messages, **generation_params):
                    yield chunk
            
            elif (self.provider == LLMProvider.OPENAI):
                async for chunk in self._generate_openai_stream(messages, **generation_params):
                    yield chunk
            
            else:
                raise LLMClientError(f"Unsupported provider: {self.provider}")
        
        except Exception as e:
            self.logger.error(f"Stream generation failed: {repr(e)}")
            raise LLMClientError(f"Stream generation failed: {repr(e)}")
    

    async def _generate_ollama(self, messages: List[Dict], **generation_params) -> Dict:
        """
        Generate using Ollama API
        """
        url     = f"{self.base_url}/api/chat"
        
        # Prepare request payload
        payload = {"model"    : self.model_name,
                   "messages" : messages,
                   "stream"   : False,
                   "options"  : {"temperature" : generation_params.get("temperature", 0.1),
                                 "top_p"       : generation_params.get("top_p", 0.9),
                                 "num_predict" : generation_params.get("max_tokens", 1000),
                                }
                  }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json = payload, timeout = self.timeout) as response:
                if (response.status != 200):
                    error_text = await response.text()
                    raise LLMClientError(f"Ollama API error: {response.status} - {error_text}")
                
                result = await response.json()
                
                return {"content"       : result["message"]["content"],
                        "usage"         : {"prompt_tokens"     : result.get("prompt_eval_count", 0),
                                           "completion_tokens" : result.get("eval_count", 0),
                                           "total_tokens"      : result.get("prompt_eval_count", 0) + result.get("eval_count", 0),
                                          },
                        "finish_reason" : result.get("done_reason", "stop"),
                       }
    

    async def _generate_ollama_stream(self, messages: List[Dict], **generation_params) -> AsyncGenerator[str, None]:
        """
        Generate stream using Ollama API - FIXED VERSION
        """
        url     = f"{self.base_url}/api/chat"
        
        payload = {"model"    : self.model_name,
                   "messages" : messages,
                   "stream"   : True,
                   "options"  : {"temperature" : generation_params.get("temperature", 0.1),
                                 "top_p"       : generation_params.get("top_p", 0.9),
                                 "num_predict" : generation_params.get("max_tokens", 1000),
                                }
                  }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json = payload, timeout = self.timeout) as response:
                if (response.status != 200):
                    error_text = await response.text()
                    raise LLMClientError(f"Ollama API error: {response.status} - {error_text}")
                
                async for line in response.content:
                    line_str = line.decode('utf-8').strip()
                    
                    # Skip empty lines
                    if not line_str:
                        continue
                    
                    try:
                        chunk_data = json.loads(line_str)
                    
                        # Check if this is the final chunk
                        if (chunk_data.get("done", False)):
                            break
                        
                        # Extract content regardless of whether it's empty: Ollama sends incremental content in each chunk
                        if ("message" in chunk_data):
                            content = chunk_data["message"].get("content", "")
                            
                            # Only yield non-empty content
                            if content:  
                                yield content
                    
                    except json.JSONDecodeError as e:
                        self.logger.warning(f"Failed to parse streaming chunk: {line_str[:100]}")
                        continue
    

    async def _generate_openai(self, messages: List[Dict], **generation_params) -> Dict:
        """
        Generate using OpenAI API
        """
        url     = f"{self.base_url}/chat/completions"
        
        headers = {"Authorization" : f"Bearer {self.api_key}",
                   "Content-Type"  : "application/json",
                  }
        
        payload = {"model"       : self.model_name,
                   "messages"    : messages,
                   "temperature" : generation_params.get("temperature", 0.1),
                   "top_p"       : generation_params.get("top_p", 0.9),
                   "max_tokens"  : generation_params.get("max_tokens", 1000),
                   "stream"      : False,
                 }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers = headers, json = payload, timeout = self.timeout) as response:
                if (response.status != 200):
                    error_text = await response.text()
                    raise LLMClientError(f"OpenAI API error: {response.status} - {error_text}")
                
                result = await response.json()
                
                return {"content"       : result["choices"][0]["message"]["content"],
                        "usage"         : result["usage"],
                        "finish_reason" : result["choices"][0]["finish_reason"],
                       }
    

    async def _generate_openai_stream(self, messages: List[Dict], **generation_params) -> AsyncGenerator[str, None]:
        """
        Generate stream using OpenAI API
        """
        url     = f"{self.base_url}/chat/completions"
        
        headers = {"Authorization" : f"Bearer {self.api_key}",
                   "Content-Type"  : "application/json",
                  }
        
        payload = {"model"       : self.model_name,
                   "messages"    : messages,
                   "temperature" : generation_params.get("temperature", 0.1),
                   "top_p"       : generation_params.get("top_p", 0.9),
                   "max_tokens"  : generation_params.get("max_tokens", 1000),
                   "stream"      : True,
                  }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers = headers, json = payload, timeout = self.timeout) as response:
                if (response.status != 200):
                    error_text = await response.text()
                    
                    raise LLMClientError(f"OpenAI API error: {response.status} - {error_text}")
                
                async for line in response.content:
                    line = line.decode('utf-8').strip()
                    
                    if (line.startswith('data: ')):
                        # Remove 'data: ' prefix
                        data = line[6:]  
                        
                        if (data == '[DONE]'):
                            break
                        
                        try:
                            chunk_data = json.loads(data)
                            if ("choices" in chunk_data) and (chunk_data["choices"]):
                                delta = chunk_data["choices"][0].get("delta", {})
                                
                                if ("content" in delta):
                                    yield delta["content"]
                        
                        except json.JSONDecodeError:
                            continue
    

    def check_health(self) -> bool:
        """
        Check if LLM provider is healthy and accessible
        
        Returns:
        --------
            { bool }    : True if healthy
        """
        try:
            if (self.provider == LLMProvider.OLLAMA):
                response = requests.get(f"{self.base_url}/api/tags", timeout = 30)
                return (response.status_code == 200)
            
            elif (self.provider == LLMProvider.OPENAI):
                # Simple models list check
                headers  = {"Authorization": f"Bearer {self.api_key}"}
                response = requests.get(f"{self.base_url}/models", headers=headers, timeout=10)
                return (response.status_code == 200)
            
            return False
        
        except Exception as e:
            self.logger.warning(f"Health check failed: {repr(e)}")
            return False
    

    def get_provider_info(self) -> Dict:
        """
        Get provider information
        
        Returns:
        --------
            { dict }    : Provider information
        """
        return {"provider" : self.provider.value,
                "model"    : self.model_name,
                "base_url" : self.base_url,
                "healthy"  : self.check_health(),
                "timeout"  : self.timeout,
               }


# Global LLM client instance
_llm_client = None


def get_llm_client(provider: LLMProvider = None, **kwargs) -> LLMClient:
    """
    Get global LLM client instance (singleton)
    
    Arguments:
    ----------
        provider { LLMProvider } : LLM provider to use
        
        **kwargs                 : Additional client configuration
    
    Returns:
    --------
        { LLMClient }           : LLMClient instance
    """
    global _llm_client
    
    if _llm_client is None or (provider and _llm_client.provider != provider):
        _llm_client = LLMClient(provider, **kwargs)
    
    return _llm_client


@handle_errors(error_type = LLMClientError, log_error = True, reraise = False)
async def generate_text(messages: List[Dict], provider: LLMProvider = LLMProvider.OLLAMA, **kwargs) -> str:
    """
    Convenience function for text generation
    
    Arguments:
    ----------
        messages     { list }    : List of message dictionaries
        
        provider { LLMProvider } : LLM provider to use
        
        **kwargs                 : Generation parameters
    
    Returns:
    --------
             { str }             : Generated text
    """
    client   = get_llm_client(provider, **kwargs)
    response = await client.generate(messages, **kwargs)
    
    return response["content"]
