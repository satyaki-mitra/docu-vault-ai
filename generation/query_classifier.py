# DEPENDENCIES
import json
from typing import Dict
from config.models import LLMProvider
from config.settings import get_settings
from config.logging_config import get_logger
from generation.llm_client import get_llm_client


# Setup Settings and Logging
settings = get_settings()
logger   = get_logger(__name__)


class QueryClassifier:
    """
    LLM-based query classifier that intelligently routes queries to:
    1. General/Conversational (no document context needed)
    2. RAG/Document-based (needs retrieval from documents)
    
    Uses the LLM itself for classification instead of hardcoded patterns.
    """
    def __init__(self, provider: LLMProvider = None, model_name: str = None):
        self.logger        = logger
        self.provider      = provider or LLMProvider.OLLAMA
        self.model_name    = model_name or settings.OLLAMA_MODEL
        
        # Initialize LLM client for classification
        self.llm_client    = get_llm_client(provider   = self.provider,
                                            model_name = self.model_name,
                                           )
        
        # Classification prompt
        self.system_prompt = """
                                You are a query classification system for a RAG (Retrieval-Augmented Generation) system. 
                                Your job is to determine if a user query should be answered using the user's uploaded documents.

                                **IMPORTANT CONTEXT**: The user has uploaded documents to the system. All queries related to the content of those uploaded documents should use RAG.

                                Classify queries into TWO categories:

                                **RAG (Document-based)** - Use when ANY of these are true:
                                1. Query asks about ANY content that could be in the uploaded documents
                                2. Query asks factual questions that could be answered from document content
                                3. Query asks for lists, summaries, or analysis of information
                                4. Query mentions specific details, data, statistics, names, dates, or facts
                                5. Query asks "what", "how", "list", "explain", "summarize", "compare", "analyze" about any topic
                                6. Query could reasonably be answered by searching through documents
                                7. **CRITICAL**: When documents are uploaded, DEFAULT TO RAG for most factual/content queries

                                **GENERAL (Conversational)** - Use ONLY when MOST of these are true:
                                1. Query is purely conversational (greetings, thanks, casual chat)
                                2. Query asks about the RAG system itself or its functionality
                                3. Query asks for general knowledge that is NOT specific to uploaded documents
                                4. Query is a meta-question about how to use the system
                                5. Query contains NO request for factual information from documents

                                **EXAMPLES FOR ANY DOCUMENT TYPE**:
                                - For business documents: "What sales channels does the company use?" → RAG
                                - For research papers: "What were the study's findings?" → RAG
                                - For legal documents: "What are the key clauses?" → RAG
                                - For technical manuals: "How do I configure the system?" → RAG
                                - For personal documents: "What dates are mentioned?" → RAG
                                - "Hi, how are you?" → GENERAL
                                - "How do I upload a document?" → GENERAL
                                - "What is the capital of France?" → GENERAL (unless geography documents were uploaded)

                                **KEY RULES**:
                                1. When documents exist, assume queries are about them unless clearly not
                                2. When in doubt, classify as RAG (safer to search than hallucinate)
                                3. If query could be answered from document content, use RAG
                                4. Only use GENERAL for purely conversational or system-related queries

                                Respond with ONLY a JSON object (no markdown, no extra text):
                                {
                                "type": "rag" or "general",
                                "confidence": 0.0 to 1.0,
                                "reason": "brief explanation"
                                }
                             """
    
    
    async def classify(self, query: str, has_documents: bool = True) -> Dict:
        """
        Classify a query using LLM
        
        Arguments:
        ----------
            query         { str }  : User query

            has_documents { bool } : Whether documents are available in the system
        
        Returns:
        --------
                 { dict }          : Classification result
        """
        try:
            # If no documents are available, everything should be general
            if not has_documents:
                return {"type"              : "general",
                        "confidence"        : 1.0,
                        "reason"            : "No documents available in system",
                        "suggested_action"  : "respond_with_general_llm",
                        "is_llm_classified" : False,
                       }
            
            # Build classification prompt
            user_prompt    = f"""
                                 Query: "{query}"

                                 System status: {"Documents are available" if has_documents else "No documents uploaded"}

                                 Classify this query. Remember: if uncertain, prefer RAG.
                              """
            
            messages       = [{"role"    : "system", 
                               "content" : self.system_prompt,
                              },
                              {"role"    : "user", 
                               "content" : user_prompt,
                              }
                             ]
            
            # Get LLM classification (use low temperature for consistency)
            llm_response   = await self.llm_client.generate(messages    = messages,
                                                            temperature = 0.1,  # Low temperature for consistent classification
                                                            max_tokens  = 150,
                                                           )
            
            response_text  = llm_response.get("content", "").strip()
            
            # Parse JSON response
            classification = self._parse_llm_response(response_text)
            
            # Add suggested action based on classification
            if (classification["type"] == "rag"):
                classification["suggested_action"] = "respond_with_rag"

            elif (classification["type"] == "general"):
                classification["suggested_action"] = "respond_with_general_llm"

            else:
                # Default to RAG if uncertain
                classification["suggested_action"] = "respond_with_rag"
            
            classification["is_llm_classified"] = True
            
            logger.info(f"LLM classified query as: {classification['type']} (confidence: {classification['confidence']:.2f})")
            logger.debug(f"Classification reason: {classification['reason']}")
            
            return classification
            
        except Exception as e:
            logger.error(f"LLM classification failed: {e}, defaulting to RAG")
            # On error, default to RAG (safer to try document search)
            return {"type"              : "rag",
                    "confidence"        : 0.5,
                    "reason"            : f"Classification failed: {str(e)}, defaulting to RAG",
                    "suggested_action"  : "respond_with_rag",
                    "is_llm_classified" : False,
                    "error"             : str(e)
                   }
    
    
    def _parse_llm_response(self, response_text: str) -> Dict:
        """
        Parse LLM JSON response
        
        Arguments:
        ----------
            response_text { str } : LLM response text
        
        Returns:
        --------
                 { dict }         : Parsed classification
        """
        try:
            # Remove markdown code blocks if present
            if ("```json" in response_text):
                response_text = response_text.split("```json")[1].split("```")[0].strip()
            
            elif ("```" in response_text):
                response_text = response_text.split("```")[1].split("```")[0].strip()
            
            # Parse JSON
            result = json.loads(response_text)
            
            # Validate required fields
            if ("type" not in result) or (result["type"] not in ["rag", "general"]):
                raise ValueError(f"Invalid type in response: {result.get('type')}")
            
            # Set defaults for missing fields
            result.setdefault("confidence", 0.8)
            result.setdefault("reason", "LLM classification")
            
            # Clamp confidence to valid range
            result["confidence"] = max(0.0, min(1.0, float(result["confidence"])))
            
            return result
            
        except (json.JSONDecodeError, ValueError, KeyError) as e:
            logger.warning(f"Failed to parse LLM response: {e}")
            logger.debug(f"Raw response: {response_text}")
            
            # Try to extract type from text if JSON parsing fails
            response_lower = response_text.lower()
            
            if (("general" in response_lower) and ("rag" not in response_lower)):
                return {"type"       : "general",
                        "confidence" : 0.6,
                        "reason"     : "Parsed from non-JSON response",
                       }

            else:
                # Default to RAG if parsing fails
                return {"type"       : "rag",
                        "confidence" : 0.6,
                        "reason"     : "Failed to parse response, defaulting to RAG",
                       }


# Global classifier instance
_query_classifier = None


def get_query_classifier(provider: LLMProvider = None, model_name: str = None) -> QueryClassifier:
    """
    Get global query classifier instance
    
    Arguments:
    ----------
        provider   { LLMProvider } : LLM provider

        model_name    { str }      : Model name
    
    Returns:
    --------
        { QueryClassifier }        : QueryClassifier instance
    """
    global _query_classifier
    
    if _query_classifier is None:
        _query_classifier = QueryClassifier(provider   = provider, 
                                            model_name = model_name,
                                           )
    
    return _query_classifier