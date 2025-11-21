# DEPENDENCIES
import re
import tiktoken
from enum import Enum
from typing import List
from typing import Optional
from config.settings import get_settings
from config.logging_config import get_logger


# Setup Logger and settings
logger   = get_logger(__name__)
settings = get_settings()


class TokenizerType(str, Enum):
    """
    Supported tokenizer types
    """
    CL100K      = "cl100k_base"  # GPT-4, GPT-3.5-turbo
    P50K        = "p50k_base"    # Codex, text-davinci-002/003
    R50K        = "r50k_base"    # GPT-3, text-davinci-001
    GPT2        = "gpt2"         # GPT-2
    APPROXIMATE = "approximate"  # Fast approximation


class TokenCounter:
    """
    Token counting utility with support for multiple tokenizers: Provides accurate token counts for chunking and context management
    """
    def __init__(self, tokenizer_type: str = "cl100k_base"):
        """
        Initialize token counter.
        
        Arguments:
        ----------
            tokenizer_type { str } : Type of tokenizer to use
        """
        self.tokenizer_type = tokenizer_type
        self.logger         = logger
        
        # Initialize tokenizer
        if (tokenizer_type != TokenizerType.APPROXIMATE):
            try:
                self.tokenizer = tiktoken.get_encoding(tokenizer_type)
                self.logger.debug(f"Initialized tiktoken tokenizer: {tokenizer_type}")

            except Exception as e:
                self.logger.warning(f"Failed to load tiktoken: {repr(e)}, using approximation")
                self.tokenizer      = None
                self.tokenizer_type = TokenizerType.APPROXIMATE
        
        else:
            self.tokenizer = None
    

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
        
        if self.tokenizer is not None:
            # Use tiktoken for accurate counting
            try:
                tokens = self.tokenizer.encode(text)
                return len(tokens)
            
            except Exception as e:
                self.logger.warning(f"Tokenizer error: {e}, falling back to approximation")
                return self._approximate_token_count(text)
        
        else:
            # Use approximation
            return self._approximate_token_count(text = text)
    

    def _approximate_token_count(self, text: str) -> int:
        """
        Approximate token count using heuristics: 
        - Rule -  ~4 characters per token for English text
        
        Arguments:
        ----------
            text { str } : Input text
        
        Returns:
        --------
             { int }     : Approximate token count
        """
        # Split by whitespace to get words
        words         = text.split()
        
        # Count characters
        char_count    = len(text)
        
        # Hybrid approach: average of word-based and char-based estimates - Words typically = 1.3 tokens (accounting for subword tokenization)
        word_estimate = len(words) * 1.3
        
        # Characters typically = 4 chars per token
        char_estimate = char_count / 4.0
        
        # Take average of both estimates
        estimate      = int((word_estimate + char_estimate) / 2)
        
        # Ensure at least 1 token
        return max(1, estimate)  
    

    def encode(self, text: str) -> List[int]:
        """
        Encode text to token IDs
        
        Arguments:
        ----------
            text { str } : Input text
        
        Returns:
        --------
             { list }    : List of token IDs
        """
        if self.tokenizer is None:
            raise ValueError("Cannot encode with approximate tokenizer")
        
        return self.tokenizer.encode(text)
    

    def decode(self, tokens: List[int]) -> str:
        """
        Decode token IDs to text
        
        Arguments:
        ----------
            tokens { list } : List of token IDs
        
        Returns:
        --------
              { str }       : Decoded text
        """
        if self.tokenizer is None:
            raise ValueError("Cannot decode with approximate tokenizer")
        
        return self.tokenizer.decode(tokens)
    

    def truncate_to_tokens(self, text: str, max_tokens: int, suffix: str = "") -> str:
        """
        Truncate text to maximum token count
        
        Arguments:
        ----------
            text       { str } : Input text

            max_tokens { int } : Maximum number of tokens
            
            suffix     { str } : Suffix to add (e.g., "...")
        
        Returns:
        --------
                { str }        : Truncated text
        """
        if self.tokenizer is not None:
            # Use precise token-based truncation
            tokens = self.encode(text)
            
            if (len(tokens) <= max_tokens):
                return text
            
            # Account for suffix tokens
            suffix_tokens    = len(self.encode(suffix)) if suffix else 0
            truncate_at      = max_tokens - suffix_tokens
            
            truncated_tokens = tokens[:truncate_at]
            truncated_text   = self.decode(truncated_tokens)
            
            return truncated_text + suffix

        else:
            # Use character-based approximation
            current_tokens = self.count_tokens(text = text)
            
            if (current_tokens <= max_tokens):
                return text
            
            # Estimate character position
            ratio         = max_tokens / current_tokens
            char_position = int(len(text) * ratio)
            
            # Find nearest word boundary
            truncated     = text[:char_position]
            last_space    = truncated.rfind(' ')
            
            if (last_space > 0):
                truncated = truncated[:last_space]
            
            return truncated + suffix
    

    def split_into_token_chunks(self, text: str, chunk_size: int, overlap: int = 0) -> List[str]:
        """
        Split text into chunks of approximately equal token count
        
        Arguments:
        ----------
            text       { str } : Input text

            chunk_size { int } : Target tokens per chunk
            
            overlap    { int } : Number of overlapping tokens between chunks
        
        Returns:
        --------
               { list }        : List of text chunks
        """
        if (overlap >= chunk_size):
            raise ValueError("Overlap must be less than chunk_size")
        
        if self.tokenizer is not None:
            precise_chunks = self._split_precise(text       = text, 
                                                 chunk_size = chunk_size, 
                                                 overlap    = overlap,
                                                )

            return precise_chunks
        
        else:
            approximate_chunks = self._split_approximate(text       = text, 
                                                         chunk_size = chunk_size, 
                                                         overlap    = overlap,
                                                        )

            return approximate_chunks

    
    def _split_precise(self, text: str, chunk_size: int, overlap: int) -> List[str]:
        """
        Split using precise token counts
        """
        tokens = self.encode(text)
        chunks = list()
        
        start  = 0

        while (start < len(tokens)):
            # Get chunk tokens
            end          = min(start + chunk_size, len(tokens))
            chunk_tokens = tokens[start:end]
            
            # Decode to text
            chunk_text   = self.decode(chunk_tokens)

            chunks.append(chunk_text)
            
            # Move to next chunk with overlap
            start        = end - overlap
            
            # Avoid infinite loop
            if ((start >= len(tokens)) or ((end == len(tokens)))):
                break
        
        return chunks
    

    def _split_approximate(self, text: str, chunk_size: int, overlap: int) -> List[str]:
        """
        Split using approximate token counts
        """
        # Estimate characters per chunk : Rule = ~4 chars per token
        chars_per_chunk = chunk_size * 4
        overlap_chars   = overlap * 4
        
        chunks          = list()
        sentences       = self._split_into_sentences(text = text)
        
        current_chunk   = list()
        current_tokens  = 0
        
        for sentence in sentences:
            sentence_tokens = self.count_tokens(text = sentence)
            
            if (((current_tokens + sentence_tokens) > chunk_size) and current_chunk):
                # Save current chunk
                chunk_text = " ".join(current_chunk)
                
                chunks.append(chunk_text)
                
                # Start new chunk with overlap
                if (overlap > 0):
                    # Keep last few sentences for overlap
                    overlap_text   = chunk_text[-overlap_chars:] if len(chunk_text) > overlap_chars else chunk_text
                    current_chunk  = [overlap_text, sentence]
                    current_tokens = self.count_tokens(text = " ".join(current_chunk))
                
                else:
                    current_chunk  = [sentence]
                    current_tokens = sentence_tokens
            
            else:
                current_chunk.append(sentence)
                
                current_tokens += sentence_tokens
        
        # Add final chunk
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        
        return chunks
    

    @staticmethod
    def _split_into_sentences(text: str) -> List[str]:
        """
        Simple sentence splitter
        """
        # Split on sentence boundaries
        sentences       = re.split(r'(?<=[.!?])\s+', text)
        final_sentences = [s.strip() for s in sentences if s.strip()]

        return final_sentences
    

    def get_token_stats(self, text: str) -> dict:
        """
        Get comprehensive token statistics
        
        Arguments:
        ----------
            text { str } : Input text
        
        Returns:
        --------
            { dict }     : Dictionary with statistics
        """
        token_count = self.count_tokens(text = text)
        char_count  = len(text)
        word_count  = len(text.split())
        
        stats       = {"tokens"          : token_count,
                       "characters"      : char_count,
                       "words"           : word_count,
                       "chars_per_token" : char_count / token_count if (token_count > 0) else 0,
                       "tokens_per_word" : token_count / word_count if (word_count > 0) else 0,
                       "tokenizer"       : self.tokenizer_type,
                      }
        
        return stats
    

    def estimate_cost(self, text: str, cost_per_1k_tokens: float = 0.002) -> float:
        """
        Estimate API cost for text.
        
        Arguments:
        ----------
            text                { str }  : Input text

            cost_per_1k_tokens { float } : Cost per 1000 tokens (default: GPT-4 input)
        
        Returns:
        --------
                    { float }            : Estimated cost in dollars
        """
        tokens = self.count_tokens(text = text)
        cost   = (tokens / 1000) * cost_per_1k_tokens

        return round(cost, 6)
    

    def batch_count_tokens(self, texts: List[str]) -> List[int]:
        """
        Count tokens for multiple texts efficiently
        
        Arguments:
        ----------
            texts { list } : List of texts
        
        Returns:
        --------
             { list }      : List of token counts
        """
        token_counts = [self.count_tokens(text = text) for text in texts]
        
        return token_counts 
    

    def find_token_boundaries(self, text: str, target_tokens: int) -> tuple[int, str]:
        """
        Find character position that gives approximately target tokens
        
        Arguments:
        ----------
            text          { str } : Input text

            target_tokens { int } : Target number of tokens
        
        Returns:
        --------
                 { tuple }        : Tuple of (character_position, text_up_to_position)
        """
        if self.tokenizer is not None:
            tokens = self.encode(text)

            if (len(tokens) <= target_tokens):
                return len(text), text
            
            target_tokens_subset = tokens[:target_tokens]
            result_text          = self.decode(target_tokens_subset)
            
            return len(result_text), result_text
        
        else:
            # Approximate
            total_tokens = self.count_tokens(text = text)
            
            if (total_tokens <= target_tokens):
                return len(text), text
            
            ratio    = target_tokens / total_tokens
            char_pos = int(len(text) * ratio)
            
            return char_pos, text[:char_pos]


# Global counter instance
_counter = None


def get_token_counter(tokenizer_type: str = "cl100k_base") -> TokenCounter:
    """
    Get global token counter instance
    
    Arguments:
    ----------
        tokenizer_type { str } : Tokenizer type
    
    Returns:
    --------
         { TokenCounter }      : TokenCounter instance
    """
    global _counter

    if _counter is None or _counter.tokenizer_type != tokenizer_type:
        _counter = TokenCounter(tokenizer_type)
    
    return _counter


# Convenience functions
def count_tokens(text: str, tokenizer_type: str = "cl100k_base") -> int:
    """
    Quick token count
    
    Arguments:
    ----------
        text           { str } : Input text

        tokenizer_type { str } : Tokenizer type
    
    Returns:
    --------
              { int }          : Token count
    """
    counter = get_token_counter(tokenizer_type)

    return counter.count_tokens(text)


def truncate_to_tokens(text: str, max_tokens: int, suffix: str = "...", tokenizer_type: str = "cl100k_base") -> str:
    """
    Truncate text to max tokens
    
    Arguments:
    ----------
        text           { str } : Input text
        
        max_tokens     { int } : Maximum tokens
        
        suffix         { str } : Suffix to add
        
        tokenizer_type { str } : Tokenizer type
    
    Returns:
    ---------
              { str }          : Truncated text
    """
    counter = get_token_counter(tokenizer_type)
    
    return counter.truncate_to_tokens(text, max_tokens, suffix)



if __name__ == "__main__":
    # Test token counter
    print("=== Token Counter Tests ===\n")
    
    # Test with different tokenizers
    test_text = """
                    This is a test document with multiple sentences. It contains various types of content.
                    We'll use this to test the token counter functionality and ensure it works correctly.
                    The quick brown fox jumps over the lazy dog.
                """
    
    print("Test 1: Count tokens with different tokenizers")
    for tokenizer in ["cl100k_base", "gpt2", "approximate"]:
        try:
            counter = TokenCounter(tokenizer)
            tokens  = counter.count_tokens(test_text)
            print(f"  {tokenizer}: {tokens} tokens")
        
        except Exception as e:
            print(f"  {tokenizer}: Error - {e}")
    
    print()
    
    # Test token stats
    print("Test 2: Token statistics")
    counter = TokenCounter("approximate")
    stats   = counter.get_token_stats(test_text)
    
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print()
    
    # Test truncation
    print("Test 3: Truncation to 20 tokens")
    truncated = counter.truncate_to_tokens(test_text, max_tokens=20, suffix="...")

    print(f"  Original: {counter.count_tokens(test_text)} tokens")
    print(f"  Truncated: {counter.count_tokens(truncated)} tokens")
    print(f"  Text: {truncated}")
    print()
    
    # Test splitting
    print("Test 4: Split into chunks (20 tokens each, 5 overlap)")
    chunks = counter.split_into_token_chunks(test_text, chunk_size=20, overlap=5)
    
    print(f"  Created {len(chunks)} chunks:")
    for i, chunk in enumerate(chunks, 1):
        tokens = counter.count_tokens(chunk)
        print(f"    Chunk {i}: {tokens} tokens - {chunk[:50]}...")
    
    print()
    
    # Test cost estimation
    print("Test 5: Cost estimation")
    cost = counter.estimate_cost(test_text, cost_per_1k_tokens=0.002)
    
    print(f"  Text tokens: {counter.count_tokens(test_text)}")
    print(f"  Estimated cost: ${cost:.6f}")
    print()
    
    # Test convenience functions
    print("Test 6: Convenience functions")
    quick_count = count_tokens("Hello world!")
    
    print(f"  Quick count: {quick_count} tokens")
    quick_truncate = truncate_to_tokens("This is a longer text that will be truncated", max_tokens=5)
    print(f"  Quick truncate: {quick_truncate}")
    print()
    
    print("✓ Token counter module created successfully!")