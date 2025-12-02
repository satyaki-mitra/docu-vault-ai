# DEPENDENCIES
import tiktoken
from typing import List
from typing import Dict
from typing import Optional
from config.settings import get_settings
from config.logging_config import get_logger
from utils.error_handler import handle_errors
from utils.error_handler import TokenManagementError


# Setup Settings and Logging
settings = get_settings()
logger   = get_logger(__name__)


class TokenManager:
    """
    Token management for LLM context windows: Handles token counting, context window management, and optimization for different LLM providers (Ollama, OpenAI)
    """
    def __init__(self, model_name: str = None):
        """
        Initialize token manager
        
        Arguments:
        ----------
            model_name { str } : Model name for tokenizer selection
        """
        self.logger         = logger
        self.settings       = get_settings()
        self.model_name     = model_name or self.settings.OLLAMA_MODEL
        self.encoding       = None
        self.context_window = self._get_context_window()
        
        self._initialize_tokenizer()
    

    def _initialize_tokenizer(self):
        """
        Initialize appropriate tokenizer based on model
        """
        try:
            # Determine tokenizer based on model
            if self.model_name.startswith(('gpt-3.5', 'gpt-4')):
                # OpenAI models
                self.encoding = tiktoken.encoding_for_model(self.model_name)
                self.logger.debug(f"Initialized tiktoken for {self.model_name}")
            
            else:
                # Default for Ollama/local models
                self.encoding = tiktoken.get_encoding("cl100k_base")
                self.logger.debug(f"Using cl100k_base tokenizer for local model {self.model_name}")
        
        except Exception as e:
            self.logger.warning(f"Failed to initialize specific tokenizer: {repr(e)}, using approximation")
            self.encoding = None
    

    def _get_context_window(self) -> int:
        """
        Get context window size based on model
        
        Returns:
        --------
            { int }    : Context window size in tokens
        """
        model_contexts = {"gpt-3.5-turbo" : 4096,
                          "mistral:7b"    : 8192,
                         }
        
        # Find matching model
        for model_pattern, context_size in model_contexts.items():
            if model_pattern in self.model_name.lower():
                return context_size
        
        # Default context window
        default_context = self.settings.CONTEXT_WINDOW
        self.logger.info(f"Using default context window {default_context} for model {self.model_name}")
        
        return default_context
    

    def count_tokens(self, text: str) -> int:
        """
        Count tokens in text
        
        Arguments:
        ----------
            text { str } : Input text
        
        Returns:
        --------
             { int }     : Number of tokens
        """
        if not text:
            return 0
        
        if self.encoding is not None:
            try:
                return len(self.encoding.encode(text))
            
            except Exception as e:
                self.logger.warning(f"Tokenizer failed, using approximation: {repr(e)}")
        
        # Fallback approximation
        return self._approximate_token_count(text = text)
    

    def _approximate_token_count(self, text: str) -> int:
        """
        Approximate token count when tokenizer is unavailable
        
        Arguments:
        ----------
            text { str } : Input text
        
        Returns:
        --------
             { int }     : Approximate token count
        """
        if not text:
            return 0
        
        # Use word-based approximation (more reliable than char-based)
        words            = text.split()
        
        # English text averages ~1.3 tokens per word : (accounting for punctuation and subword tokenization)
        estimated_tokens = int(len(words) * 1.3)
        
        # Add 5% buffer for punctuation and special tokens
        estimated_tokens = int(estimated_tokens * 1.05)
        
        return estimated_tokens
        

    def count_message_tokens(self, messages: List[Dict]) -> int:
        """
        Count tokens in chat messages
        
        Arguments:
        ----------
            messages { list } : List of message dictionaries
        
        Returns:
        --------
                 { int }      : Total tokens in messages
        """
        if not messages:
            return 0
        
        total_tokens = 0
        
        for message in messages:
            # Count content tokens
            content       = message.get('content', '')
            total_tokens += self.count_tokens(text = content)
            
            # Count role tokens (approximate)
            role          = message.get('role', '')
            total_tokens += self.count_tokens(text = role)
            
            # Add overhead for message structure: Approximate overhead per message
            total_tokens += 5  
        
        return total_tokens
    

    def fits_in_context(self, prompt: str, max_completion_tokens: int = 1000) -> bool:
        """
        Check if prompt fits in context window with room for completion
        
        Arguments:
        ----------
            prompt                { str } : Prompt text
            
            max_completion_tokens { int } : Tokens to reserve for completion
        
        Returns:
        --------
                      { bool }            : True if prompt fits
        """
        prompt_tokens  = self.count_tokens(text = prompt)
        total_required = prompt_tokens + max_completion_tokens
        
        return (total_required <= self.context_window)
    

    def truncate_to_fit(self, text: str, max_tokens: int, strategy: str = "end") -> str:
        """
        Truncate text to fit within token limit
        
        Arguments:
        ----------
            text       { str } : Text to truncate
            
            max_tokens { int } : Maximum tokens allowed
            
            strategy   { str } : Truncation strategy ('end', 'start', 'middle')
        
        Returns:
        --------
                { str }        : Truncated text
        """
        current_tokens = self.count_tokens(text = text)
        
        if (current_tokens <= max_tokens):
            return text
        
        if (strategy == "end"):
            return self._truncate_from_end(text       = text, 
                                           max_tokens = max_tokens,
                                          )
        
        elif (strategy == "start"):
            return self._truncate_from_start(text       = text, 
                                             max_tokens = max_tokens,
                                            )
        
        elif (strategy == "middle"):
            return self._truncate_from_middle(text       = text, 
                                              max_tokens = max_tokens,
                                             )
        
        else:
            self.logger.warning(f"Unknown truncation strategy: {strategy}, using 'end'")
            return self._truncate_from_end(text       = text, 
                                           max_tokens = max_tokens,
                                          )
    

    def _truncate_from_end(self, text: str, max_tokens: int) -> str:
        """
        Truncate from the end of the text
        """
        if self.encoding is not None:
            tokens           = self.encoding.encode(text)
            truncated_tokens = tokens[:max_tokens]
            return self.encoding.decode(truncated_tokens)
        
        # Approximate truncation
        words           = text.split()
        # Conservative estimate
        target_words    = int(max_tokens * 0.75)  
        truncated_words = words[:target_words]
        
        return " ".join(truncated_words)
    

    def _truncate_from_start(self, text: str, max_tokens: int) -> str:
        """
        Truncate from the start of the text
        """
        if self.encoding is not None:
            tokens           = self.encoding.encode(text)
            # Take from end
            truncated_tokens = tokens[-max_tokens:]  
            return self.encoding.decode(truncated_tokens)
        
        # Approximate truncation
        words           = text.split()
        target_words    = int(max_tokens * 0.75)

        # Take from end
        truncated_words = words[-target_words:]  
        
        return " ".join(truncated_words)
    

    def _truncate_from_middle(self, text: str, max_tokens: int) -> str:
        """
        Truncate from the middle of the text
        """
        if self.encoding is not None:
            tokens = self.encoding.encode(text)

            if (len(tokens) <= max_tokens):
                return text
            
            # Keep beginning and end, remove middle
            keep_start   = max_tokens // 3
            keep_end     = max_tokens - keep_start
            
            start_tokens = tokens[:keep_start]
            end_tokens   = tokens[-keep_end:]
            
            return self.encoding.decode(start_tokens) + " [...] " + self.encoding.decode(end_tokens)
        
        # Approximate truncation
        words       = text.split()

        if (len(words) <= max_tokens):
            return text
        
        keep_start  = max_tokens // 3
        keep_end    = max_tokens - keep_start
        
        start_words = words[:keep_start]
        end_words   = words[-keep_end:]
        
        return " ".join(start_words) + " [...] " + " ".join(end_words)
    

    def calculate_max_completion_tokens(self, prompt: str, reserve_tokens: int = 100) -> int:
        """
        Calculate maximum completion tokens given prompt length
        
        Arguments:
        ----------
            prompt         { str } : Prompt text
            
            reserve_tokens { int } : Tokens to reserve for safety
        
        Returns:
        --------
                   { int }         : Maximum completion tokens
        """
        prompt_tokens    = self.count_tokens(text = prompt)
        available_tokens = self.context_window - prompt_tokens - reserve_tokens
        
        return max(0, available_tokens)
    

    def optimize_context_usage(self, context: str, prompt: str, max_completion_tokens: int = 1000) -> str:
        """
        Optimize context to fit within context window
        
        Arguments:
        ----------
            context              { str } : Context text
            
            prompt               { str } : Prompt template
            
            max_completion_tokens { int } : Tokens needed for completion
        
        Returns:
        --------
                      { str }            : Optimized context
        """
        total_prompt_tokens   = self.count_tokens(text = prompt.format(context=""))
        available_for_context = self.context_window - total_prompt_tokens - max_completion_tokens
        
        if (available_for_context <= 0):
            self.logger.warning("Prompt too large for context window")
            return ""
        
        context_tokens = self.count_tokens(text = context)
        
        if (context_tokens <= available_for_context):
            return context
        
        # Truncate context to fit
        optimized_context = self.truncate_to_fit(context, available_for_context, strategy="end")
        
        reduction_pct     = ((context_tokens - self.count_tokens(text = optimized_context)) / context_tokens) * 100

        self.logger.info(f"Context reduced by {reduction_pct:.1f}% to fit context window")
        
        return optimized_context
    

    def get_token_stats(self, text: str) -> Dict:
        """
        Get detailed token statistics
        
        Arguments:
        ----------
            text { str } : Input text
        
        Returns:
        --------
            { dict }     : Token statistics
        """
        tokens = self.count_tokens(text = text)
        chars  = len(text)
        words  = len(text.split())
        
        return {"tokens"          : tokens,
                "characters"      : chars,
                "words"           : words,
                "chars_per_token" : chars / tokens if tokens > 0 else 0,
                "tokens_per_word" : tokens / words if words > 0 else 0,
                "context_window"  : self.context_window,
                "model"           : self.model_name,
               }


# Global token manager instance
_token_manager = None


def get_token_manager(model_name: str = None) -> TokenManager:
    """
    Get global token manager instance
    
    Arguments:
    ----------
        model_name { str } : Model name for tokenizer selection
    
    Returns:
    --------
        { TokenManager }   : TokenManager instance
    """
    global _token_manager
    
    if _token_manager is None or (model_name and _token_manager.model_name != model_name):
        _token_manager = TokenManager(model_name)
    
    return _token_manager


@handle_errors(error_type = TokenManagementError, log_error = True, reraise = False)
def count_tokens_safe(text: str, model_name: str = None) -> int:
    """
    Safe token counting with error handling
    
    Arguments:
    ----------
        text       { str } : Text to count tokens for
        
        model_name { str } : Model name for tokenizer
    
    Returns:
    --------
             { int }       : Token count (0 on error)
    """
    try:
        manager = get_token_manager(model_name = model_name)
        return manager.count_tokens(text = text)
    
    except Exception:
        return 0