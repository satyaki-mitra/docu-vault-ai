# DEPENDENCIES
import random
import datetime
from typing import List
from typing import Dict
from config.models import LLMProvider
from config.settings import get_settings
from config.logging_config import get_logger
from utils.error_handler import handle_errors
from generation.llm_client import get_llm_client


# Setup Settings and Logging
settings = get_settings()
logger   = get_logger(__name__)


class GeneralResponder:
    """
    Handles general/conversational queries using LLM without RAG
    """
    def __init__(self, provider: LLMProvider = None, model_name: str = None):
        self.logger             = logger
        self.provider           = provider or LLMProvider.OLLAMA
        self.model_name         = model_name or settings.OLLAMA_MODEL
        
        # Initialize LLM client for general responses
        self.llm_client         = get_llm_client(provider   = self.provider, 
                                                 model_name = self.model_name,
                                                )
        
        # System prompt for general conversation
        self.system_prompt      = """
                                     You are a helpful, friendly AI assistant. You're part of a larger system called the "AI Universal Knowledge Ingestion System" which specializes in document analysis and retrieval.

                                     When users ask general questions, answer helpfully and conversationally. When they ask about your capabilities, explain that you can:
                                     1. Answer general knowledge questions
                                     2. Help with document analysis (when they upload documents)
                                     3. Provide explanations on various topics
                                     4. Engage in friendly conversation

                                     If a question is better answered by searching through documents, politely suggest uploading documents first.

                                     Be concise but thorough. Use a friendly, professional tone.
                                  """

        # Fallback responses (if LLM fails)
        self.fallback_responses = {"greeting" : ["Hello! 👋 I'm your AI assistant. How can I help you today?", "Hi there! I'm here to help with questions or document analysis. What's on your mind?", "Greetings! I can answer questions or help analyze documents. What would you like to know?"],
                                   "farewell" : ["Goodbye! Feel free to come back if you have more questions.", "See you later! Don't hesitate to ask if you need help with documents.", "Take care! Remember I'm here for document analysis too."],
                                   "thanks"   : ["You're welcome! Happy to help.", "My pleasure! Let me know if you need anything else.", "Glad I could help! Don't hesitate to ask more questions."],
                                   "default"  : ["I'm here to help! You can ask me general questions or upload documents for analysis.", "That's an interesting question! I'd be happy to discuss it with you.", "I can help with that. What specific aspect would you like to know about?"],
                                  }
    

    async def respond(self, query: str, conversation_history: List[Dict] = None) -> Dict:
        """
        Generate a response to a general query
        
        Arguments:
        ----------
            query                { str } : User query

            conversation_history { list } : Previous messages in conversation
        
        Returns:
        --------
                     { dict }             : Response dictionary
        """
        try:
            # Prepare messages for LLM
            messages = list()
            
            # Add system prompt
            messages.append({"role"    : "system", 
                             "content" : self.system_prompt,
                           })
            
            # Add conversation history if available
            if conversation_history:
                # Last 5 messages for context
                messages.extend(conversation_history[-5:])  
            
            # Add current query
            messages.append({"role"    : "user", 
                             "content" : query,
                           })
            
            # Generate response: Slightly higher temp for conversational
            llm_response  = await self.llm_client.generate(messages    = messages,
                                                           temperature = 0.7, 
                                                           max_tokens  = 500,
                                                          )
            
            response_text = llm_response.get("content", "").strip()
            
            if not response_text:
                response_text = self._get_fallback_response(query)
            
            return {"answer"       : response_text,
                    "is_general"   : True,
                    "requires_rag" : False,
                    "tokens_used"  : llm_response.get("usage", {}),
                    "model"        : self.model_name,
                   }
            
        except Exception as e:
            logger.error(f"General response generation failed: {e}")
            return {"answer"       : self._get_fallback_response(query),
                    "is_general"   : True,
                    "requires_rag" : False,
                    "error"        : str(e),
                    "model"        : self.model_name,
                   }
    

    def _get_fallback_response(self, query: str) -> str:
        """
        Get a fallback response if LLM fails
        """
        query_lower = query.lower()
        
        if (any(word in query_lower for word in ["hello", "hi", "hey", "greetings"])):
            return random.choice(self.fallback_responses["greeting"])
        
        elif (any(word in query_lower for word in ["thank", "thanks", "appreciate"])):
            return random.choice(self.fallback_responses["thanks"])
        
        elif (any(word in query_lower for word in ["bye", "goodbye", "see you"])):
            return random.choice(self.fallback_responses["farewell"])
        
        else:
            return random.choice(self.fallback_responses["default"])



# Global responder instance
_general_responder = None


def get_general_responder(provider: LLMProvider = None, model_name: str = None) -> GeneralResponder:
    """
    Get global general responder instance
    
    Returns:
    --------
        { GeneralResponder } : GeneralResponder instance
    """
    global _general_responder
    
    if _general_responder is None:
        _general_responder = GeneralResponder(provider   = provider, 
                                              model_name = model_name,
                                             )
    
    return _general_responder