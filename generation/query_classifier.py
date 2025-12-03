# DEPENDENCIES
import re
from typing import List
from typing import Dict
from typing import Tuple
from config.settings import get_settings
from config.logging_config import get_logger
from utils.error_handler import handle_errors


# Setup Settings and Logging
settings = get_settings()
logger   = get_logger(__name__)


class QueryClassifier:
    """
    Classifies queries as either:
    1. General/Conversational (no document context needed)
    2. RAG/Document-based (needs retrieval from documents)
    3. Mixed/Ambiguous (could be either)
    """
    def __init__(self):
        self.logger                = logger
        
        # Patterns that indicate document-related queries
        self.document_patterns     = [r'find\s+(in|from)',
                                      r'search\s+(for|in)',
                                      r'extract\s+(from|in)',
                                      r'analyze\s+(the\s+)?(document|file|text)',
                                      r'my\s+(document|file|pdf|doc|text|report)',
                                      r'summarize\s+(the\s+)?(document|file|text)',
                                      r'explain\s+(the\s+)?(document|content|text)',
                                      r'in\s+(the\s+)?(document|file|text|report|pdf)',
                                      r'according\s+to\s+(the\s+)?(document|text|file)',
                                      r'what\s+(does|says)\s+(the\s+)?(document|file|text)',
                                      r'from\s+(the\s+)?(document|file|text|upload|uploaded)',
                                      r'based\s+on\s+(the\s+)?(document|context|information|content)',
                                     ]
        
        # Patterns that indicate general knowledge queries
        self.general_patterns      = [r'^why\s+',
                                      r'^when\s+',
                                      r'^where\s+',
                                      r'^how to\s+',
                                      r'^define\s+',
                                      r'^can you\s+',
                                      r'^explain\s+',
                                      r'^could you\s+',
                                      r'^what (is|are)',
                                      r'^what (are|can) you',
                                      r'^what(\'s| is) your',
                                      r'^(bye|goodbye|see you)',
                                      r'^tell me about yourself',
                                      r'^who (made|created|built) you',
                                      r'^how (do|does) (you|this) work',
                                      r'^(thanks|thank you|appreciate)',
                                      r'^(hi|hello|hey|greetings|good\s+(morning|afternoon|evening))',                                
                                     ]
        
        # Whitelist: These are always general queries (even if they contain document-like words)
        self.general_whitelist     = ["hi",
                                      "hey",
                                      "bye",
                                      "hello",
                                      "thanks",
                                      "goodbye",
                                      "thank you",
                                      "how are you",
                                      "good morning",
                                      "good evening",
                                      "who made you",
                                      "what are you",
                                      "good afternoon",
                                      "who created you",
                                      "what can you do",
                                      "what is your name",
                                     ]
        
        # Blacklist: These always need RAG (even if they look general)
        self.rag_blacklist         = ["find in",
                                      "search for",
                                      "extract from",
                                      "from the file",
                                      "in my document",
                                      "based on the text",
                                      "according to the report",
                                     ]
        
        self.compiled_doc_patterns = [re.compile(p, re.IGNORECASE) for p in self.document_patterns]
        self.compiled_gen_patterns = [re.compile(p, re.IGNORECASE) for p in self.general_patterns]
    

    def classify(self, query: str) -> Dict:
        """
        Classify a query and return classification with confidence
        
        Arguments:
        ----------
            query { str } : User query
        
        Returns:
        --------
             { dict }     : Classification result
        """
        query_lower = query.lower().strip()
        
        # Check whitelist first (always general)
        for whitelist_query in self.general_whitelist:
            if whitelist_query in query_lower:
                return {"type"             : "general",
                        "confidence"       : 0.95,
                        "reason"           : "Matches general whitelist",
                        "suggested_action" : "respond_with_general_llm",
                       }
        
        # Check blacklist (always RAG)
        for blacklist_query in self.rag_blacklist:
            if blacklist_query in query_lower:
                return {"type"             : "rag",
                        "confidence"       : 0.95,
                        "reason"           : "Matches RAG blacklist",
                        "suggested_action" : "respond_with_rag"
                       }
        
        # Calculate document score
        doc_score = 0

        for pattern in self.compiled_doc_patterns:
            if pattern.search(query_lower):
                doc_score += 1
        
        # Calculate general score
        gen_score = 0
        for pattern in self.compiled_gen_patterns:
            if pattern.search(query_lower):
                gen_score += 1
        
        # Heuristics: Query length, question words, etc.
        heuristics  = self._apply_heuristics(query_lower)
        doc_score  += heuristics.get("doc_bonus", 0)
        gen_score  += heuristics.get("gen_bonus", 0)
        
        # Determine classification
        total_score = doc_score + gen_score
        
        if (total_score == 0):
            # No clear pattern - use length heuristic
            if (len(query_lower.split()) <= 4):
                classification = "general"
                confidence     = 0.6
                reason         = "Short query, likely conversational"
            
            else:
                classification = "rag"
                confidence     = 0.6
                reason         = "Longer query, may need document context"
        
        else:
            if (doc_score > gen_score):
                classification = "rag"
                confidence     = doc_score / total_score
                reason         = f"Document patterns: {doc_score}, General patterns: {gen_score}"
            
            else:
                classification = "general"
                confidence     = gen_score / total_score
                reason         = f"General patterns: {gen_score}, Document patterns: {doc_score}"
        
        # Adjust based on question words
        if (classification == "general") and (self._has_specific_question_word(query_lower)):
            # Reduce confidence for general classification
            confidence *= 0.8  
        
        # Determine action
        if ((classification == "rag") and (confidence > 0.7)):
            action = "respond_with_rag"

        elif ((classification == "general") and (confidence > 0.7)):
            action = "respond_with_general_llm"

        else:
            # Ambiguous - try RAG first, then fallback to general
            action     = "try_rag_then_general"
            # Low confidence for ambiguous
            confidence = 0.5  
        
        return {"type"             : classification,
                "confidence"       : round(confidence, 2),
                "reason"           : reason,
                "suggested_action" : action,
               }
    

    def _apply_heuristics(self, query: str) -> Dict:
        """
        Apply additional heuristics for classification
        """
        heuristics = {"doc_bonus" : 0, 
                      "gen_bonus" : 0,
                     }
        
        # Length-based: Very short queries are likely general
        words = query.split()
        
        if (len(words) <= 3):
            heuristics["gen_bonus"] += 1
        
        elif (len(words) > 10):
            heuristics["doc_bonus"] += 0.5
        
        # Question word analysis
        question_words = ["why", 
                          "how", 
                          "who",
                          "when",
                          "what",  
                          "where",
                          "which",
                         ]
        
        if any(query.startswith(word) for word in question_words):
            # Starting with question word - could be either
            pass
        
        # Check for possessive pronouns (suggests specific content)
        if (("my " in query) or ("our " in query)):
            heuristics["doc_bonus"] += 1
        
        # Check for commands
        if (query.startswith(("tell", "explain", "describe", "define"))):
            heuristics["gen_bonus"] += 0.5
        
        return heuristics
    

    def _has_specific_question_word(self, query: str) -> bool:
        """
        Check if query has specific question words that might need facts
        """
        specific_words = ["why is", 
                          "who is",
                          "what is", 
                          "what are", 
                          "when was", 
                          "where is",
                          "how many", 
                          "how much", 
                          "how does", 
                         ]
                         
        return any(word in query for word in specific_words)



# Global classifier instance
_query_classifier = None


def get_query_classifier() -> QueryClassifier:
    """
    Get global query classifier instance
    
    Returns:
    --------
        { QueryClassifier } : QueryClassifier instance
    """
    global _query_classifier
    
    if _query_classifier is None:
        _query_classifier = QueryClassifier()
    
    return _query_classifier