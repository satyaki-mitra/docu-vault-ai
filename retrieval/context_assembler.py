# DEPENDENCIES
from typing import List
from typing import Optional
from collections import defaultdict
from config.settings import get_settings
from config.models import ChunkWithScore
from config.models import DocumentChunk
from config.logging_config import get_logger
from utils.error_handler import handle_errors
from chunking.token_counter import get_token_counter
from utils.error_handler import ContextAssemblyError
from retrieval.citation_tracker import CitationTracker


# Setup Settings and Logging
settings = get_settings()
logger   = get_logger(__name__)


class ContextAssembler:
    """
    Context assembly and optimization: Assembles retrieved chunks into optimal context
    for LLM processing with token limits and quality optimization
    """
    def __init__(self, max_context_tokens: int = None, strategy: str = "score_based"):
        """
        Initialize context assembler
        
        Arguments:
        ----------
            max_context_tokens { int } : Maximum tokens for assembled context (default from settings)
            
            strategy           { str } : Assembly strategy ('score_based', 'diversity', 'sequential')
        """
        self.logger             = logger
        self.settings           = get_settings()
        self.max_context_tokens = max_context_tokens or (settings.CONTEXT_WINDOW - 1000)
        self.strategy           = strategy
        self.citation_tracker   = CitationTracker()
        
        # Strategy configurations
        self.strategy_configs   = {"score_based" : {"diversity_penalty": 0.1, "min_chunk_score": 0.1},
                                   "diversity"   : {"diversity_penalty": 0.3, "min_chunk_score": 0.05},
                                   "sequential"  : {"diversity_penalty": 0.0, "min_chunk_score": 0.0},
                                  }
        
        self.logger.info(f"ContextAssembler initialized: max_tokens={self.max_context_tokens}, strategy={strategy}")
    

    def assemble_context(self, chunks: List[ChunkWithScore], query: str = "", include_citations: bool = True, format_for_llm: bool = True) -> str:
        """
        Assemble context from retrieved chunks
        
        Arguments:
        ----------
            chunks             { list } : List of retrieved chunks

            query              { str }  : Original query (for relevance optimization)
            
            include_citations  { bool } : Include citation markers in context
            
            format_for_llm     { bool } : Format context for LLM consumption
        
        Returns:
        --------
                    { str }             : Assembled context string
        """
        if not chunks:
            self.logger.warning("No chunks provided for context assembly")
            return ""
        
        try:
            self.logger.info(f"Starting context assembly with {len(chunks)} chunks")
            
            # Filter and sort chunks based on strategy
            filtered_chunks = self._filter_chunks(chunks = chunks)
            self.logger.info(f"After filtering: {len(filtered_chunks)} chunks")
            
            # Ensure we have chunks after filtering
            if not filtered_chunks:
                self.logger.error("All chunks filtered out - using top chunk from original list")
                filtered_chunks = [chunks[0]]
            
            sorted_chunks = self._sort_chunks(chunks = filtered_chunks, 
                                              query  = query,
                                             )
            
            # Select chunks within token limit
            selected_chunks = self._select_chunks_by_tokens(chunks = sorted_chunks)
            self.logger.info(f"After token selection: {len(selected_chunks)} chunks selected")
            
            # Validate selection results with progressive fallback
            if not selected_chunks:
                self.logger.warning("Token selection returned 0 chunks - using progressive fallback")
                
                # Fallback 1 - Try with higher token budget (allow 10% overflow)
                if sorted_chunks:
                    overflow_budget         = int(self.max_context_tokens * 1.10)
                    self.logger.info(f"Fallback 1: Allowing {overflow_budget} tokens (10% overflow)")
                    
                    old_budget              = self.max_context_tokens
                    self.max_context_tokens = overflow_budget
                    
                    selected_chunks         = self._select_chunks_by_tokens(chunks = sorted_chunks)
                    self.max_context_tokens = old_budget
                
                # Fallback 2 - Force include top chunks regardless of token count
                if not selected_chunks and sorted_chunks:
                    self.logger.warning("Fallback 2: Force including top 3 chunks")

                    selected_chunks = sorted_chunks[:min(3, len(sorted_chunks))]
                
                # Fallback 3 - Use first available chunk
                if not selected_chunks and chunks:
                    self.logger.error("Fallback 3: Emergency - using only first chunk")
                    
                    selected_chunks = [chunks[0]]
                
                # Complete failure
                if not selected_chunks:
                    self.logger.error("All fallbacks exhausted - no chunks available")
                    
                    raise ContextAssemblyError("No valid chunks available after all selection attempts")
            
            # Assemble context
            if format_for_llm:
                context = self._format_for_llm(chunks            = selected_chunks, 
                                               include_citations = include_citations,
                                              )
            
            else:
                context = self._format_simple(chunks            = selected_chunks,       
                                              include_citations = include_citations,
                                             )
            
            # Validate assembled context
            if not context or not context.strip():
                self.logger.error("Assembled context is empty")
                
                raise ContextAssemblyError("Context assembly produced empty result")
            
            context_tokens = self._count_tokens(text = context)
            
            self.logger.info(f"Assembled context: {len(selected_chunks)} chunks, {context_tokens} tokens, {len(context)} chars")
            
            # Warn if context exceeds limit
            if (context_tokens > self.max_context_tokens):
                overflow_pct = ((context_tokens - self.max_context_tokens) / self.max_context_tokens) * 100

                self.logger.warning(f"Context exceeds limit by {overflow_pct:.1f}% ({context_tokens} > {self.max_context_tokens})")
            
            return context
        
        except ContextAssemblyError:
            # Re-raise context assembly errors
            raise
        
        except Exception as e:
            self.logger.error(f"Context assembly failed with unexpected error: {repr(e)}", exc_info = True)
            
            # Emergency fallback: return first chunk text only
            if (chunks and len(chunks) > 0):
                self.logger.warning("Emergency fallback: returning first chunk text only")
                
                return chunks[0].chunk.text
            
            # No chunks available at all
            raise ContextAssemblyError(f"Context assembly failed with no fallback available: {repr(e)}")


    def _filter_chunks(self, chunks: List[ChunkWithScore]) -> List[ChunkWithScore]:
        """
        Filter chunks based on quality and strategy
        """
        if not chunks:
            return []
        
        strategy_config = self.strategy_configs.get(self.strategy, self.strategy_configs["score_based"])
        min_score       = strategy_config["min_chunk_score"]
        
        # Don't filter if all scores are below threshold
        all_below_threshold = all(chunk.score < min_score for chunk in chunks)
        
        if all_below_threshold:
            self.logger.warning(f"All chunks below min_score {min_score}, keeping top chunks anyway")
            # Keep at least top 5 chunks regardless of score
            filtered = sorted(chunks, key = lambda x: x.score, reverse = True)[:5]
        
        else:
            filtered = [chunk for chunk in chunks if chunk.score >= min_score]
        
        self.logger.info(f"Filtered {len(chunks)} -> {len(filtered)} chunks (min_score={min_score})")
        
        # Remove very similar chunks if diversity is important
        if (strategy_config["diversity_penalty"] > 0):
            filtered = self._apply_diversity_filter(filtered, strategy_config["diversity_penalty"])
        
        return filtered
    

    def _apply_diversity_filter(self, chunks: List[ChunkWithScore], diversity_penalty: float) -> List[ChunkWithScore]:
        """
        Apply diversity filtering to reduce redundancy - FIXED
        
        Arguments:
        ----------
            chunks            { list }  : Chunks to filter
            
            diversity_penalty { float } : Penalty factor for same-document chunks
        
        Returns:
        --------
                       { list }         : Diversified chunks
        """
        if (len(chunks) <= 1):
            return chunks
        
        # Simple diversity: penalize chunks from same document
        document_scores = dict()
        
        for chunk in chunks:
            doc_id = chunk.chunk.document_id
            if doc_id not in document_scores:
                document_scores[doc_id] = []
            
            document_scores[doc_id].append(chunk.score)
        
        # Adjust scores based on document diversity
        diversified = list()

        for chunk in chunks:
            doc_id          = chunk.chunk.document_id
            doc_chunk_count = len(document_scores[doc_id])
            
            # Penalize if multiple chunks from same document
            penalty         = diversity_penalty * (doc_chunk_count - 1)
            adjusted_score  = max(0.0, chunk.score - penalty)
            
            if (adjusted_score > 0):
                diversified_chunk = ChunkWithScore(chunk            = chunk.chunk,
                                                   score            = adjusted_score,
                                                   rank             = chunk.rank,
                                                   retrieval_method = chunk.retrieval_method + "_diversified"
                                                  )

                diversified.append(diversified_chunk)
        
        # Re-sort by adjusted scores
        diversified.sort(key = lambda x: x.score, reverse = True)
        
        # Re-normalize scores to maintain 0-1 range after penalty
        if diversified:
            max_score = max(chunk.score for chunk in diversified)
            
            if (max_score > 0):
                for chunk in diversified:
                    chunk.score = chunk.score / max_score
        
        return diversified
    

    def _sort_chunks(self, chunks: List[ChunkWithScore], query: str) -> List[ChunkWithScore]:
        """
        Sort chunks based on strategy
        """
        if (self.strategy == "sequential"):
            # Sort by document order and position
            chunks.sort(key = lambda x: (x.chunk.document_id, 
                                         x.chunk.page_number or 0,
                                         x.chunk.chunk_index
                                        )
                       )

        else:
            # Default: sort by score (already sorted by retrieval)
            chunks.sort(key = lambda x: x.score, reverse = True)
        
        return chunks
    

    def _select_chunks_by_tokens(self, chunks: List[ChunkWithScore]) -> List[ChunkWithScore]:
        """
        Select chunks that fit within token limit - IMPROVED VERSION
        
        Arguments:
        ----------
            chunks { list } : Chunks to select from
        
        Returns:
        --------
            { list }        : Selected chunks within token limit
        """
        if not chunks:
            self.logger.error("No chunks provided to _select_chunks_by_tokens")
            return []
        
        selected     = list()
        total_tokens = 0
        
        self.logger.info(f"Token selection: {len(chunks)} chunks, max={self.max_context_tokens} tokens")
        
        # Reserve tokens for system prompt and formatting overhead
        formatting_overhead = 200  
        available_tokens    = self.max_context_tokens - formatting_overhead
        
        self.logger.debug(f"Available tokens after overhead: {available_tokens}")
        
        for i, chunk in enumerate(chunks):
            # Get or calculate chunk tokens
            chunk_tokens = chunk.chunk.token_count
            
            if (chunk_tokens is None) or (chunk_tokens <= 0):
                chunk_text = chunk.chunk.text if chunk.chunk.text else ""
                
                if not chunk_text:
                    self.logger.warning(f"Chunk {i} has no text content, skipping")
                    continue
                
                chunk_tokens = self._count_tokens(text = chunk_text)
                self.logger.debug(f"Chunk {i} calculated: {chunk_tokens} tokens from {len(chunk_text)} chars")
            
            # Reserve tokens for chunk separators and citations
            chunk_formatting = 25 
            total_needed     = total_tokens + chunk_tokens + chunk_formatting
            
            if (total_needed <= available_tokens):
                selected.append(chunk)

                total_tokens += chunk_tokens + chunk_formatting
                
                self.logger.debug(f"Chunk {i}: score={chunk.score:.3f}, tokens={chunk_tokens}, total={total_tokens}/{available_tokens}")
            
            else:
                # Calculate remaining space
                remaining = available_tokens - total_tokens - chunk_formatting
                
                self.logger.debug(f"Chunk {i} exceeds limit: needs {chunk_tokens}, have {remaining} remaining")
                
                # Try partial chunk if we have reasonable space (at least 200 tokens)
                if (remaining >= 200):
                    self.logger.info(f"Attempting partial chunk {i} with {remaining} available tokens")
                    
                    partial_chunk = self._create_partial_chunk(chunk             = chunk, 
                                                               availabale_tokens = remaining,
                                                              )
                    
                    if partial_chunk:
                        selected.append(partial_chunk)
                        partial_tokens = self._count_tokens(text = partial_chunk.chunk.text)
                        total_tokens  += partial_tokens + chunk_formatting
                        
                        self.logger.info(f"Added partial chunk {i}: {partial_tokens} tokens")
                
                # Stop adding chunks - no more space
                self.logger.info(f"Stopping chunk selection at index {i}")
                break
        
        # Log selection summary
        utilization = (total_tokens / self.max_context_tokens * 100) if (self.max_context_tokens > 0) else 0
        
        self.logger.info(f"Token selection complete:")
        self.logger.info(f"- Selected: {len(selected)}/{len(chunks)} chunks")
        self.logger.info(f"- Tokens: {total_tokens}/{self.max_context_tokens} ({utilization:.1f}% utilization)")
        
        # Warning if selection is poor
        if ((len(selected) == 0) and (len(chunks) > 0)):
            self.logger.error(f"- ZERO chunks selected from {len(chunks)} available!")
            self.logger.error(f"- Max tokens: {self.max_context_tokens}")
            self.logger.error(f"- Available after overhead: {available_tokens}")
            self.logger.error(f"- First chunk tokens: {chunks[0].chunk.token_count or 'unknown'}")
            
            # Diagnostic: check if first chunk is too large
            if chunks[0].chunk.text:
                first_chunk_tokens = self._count_tokens(chunks[0].chunk.text)
                self.logger.error(f"- First chunk actual tokens: {first_chunk_tokens}")
                
                if first_chunk_tokens > available_tokens:
                    self.logger.error(f"- First chunk ({first_chunk_tokens} tokens) exceeds available space ({available_tokens} tokens)")
        
        return selected
    

    def _create_partial_chunk(self, chunk: ChunkWithScore, available_tokens: int) -> Optional[ChunkWithScore]:
        """
        Create a partial chunk that fits within available tokens
        """
        full_text    = chunk.chunk.text
        
        # Try to truncate at sentence boundary
        sentences    = full_text.split('. ')
        partial_text = ""
        
        for sentence in sentences:
            test_text   = partial_text + sentence + ". "
            test_tokens = self._count_tokens(text = test_text)
            
            if (test_tokens <= available_tokens):
                partial_text = test_text

            else:
                break
        
        if partial_text:
            partial_chunk_obj = DocumentChunk(chunk_id      = chunk.chunk.chunk_id + "_partial",
                                              document_id   = chunk.chunk.document_id,
                                              text          = partial_text.strip(),
                                              embedding     = chunk.chunk.embedding,
                                              chunk_index   = chunk.chunk.chunk_index,
                                              start_char    = chunk.chunk.start_char,
                                              end_char      = chunk.chunk.start_char + len(partial_text.strip()),
                                              page_number   = chunk.chunk.page_number,
                                              section_title = chunk.chunk.section_title,
                                              token_count   = self._count_tokens(text = partial_text.strip()),
                                              metadata      = chunk.chunk.metadata,
                                             )
            # Create partial chunk
            partial_chunk     = ChunkWithScore(chunk            = partial_chunk_obj,
                                               score            = chunk.score * 0.8,
                                               rank             = chunk.rank,
                                               retrieval_method = chunk.retrieval_method + "_partial",
                                              )

            
            return partial_chunk
        
        return None
    

    def _format_for_llm(self, chunks: List[ChunkWithScore], include_citations: bool) -> str:
        """
        Format context for LLM consumption with citations
        """
        context_parts = list()

        for i, chunk_with_score in enumerate(chunks, 1):
            chunk           = chunk_with_score.chunk
            
            # Build citation marker
            citation_marker = f"[{i}]" if include_citations else ""
            
            # Build source info
            source_info     = list()

            if chunk.page_number:
                source_info.append(f"Page {chunk.page_number}")
            
            if chunk.section_title:
                source_info.append(f"Section: {chunk.section_title}")
            
            source_str = f"({', '.join(source_info)})" if source_info else ""
            
            # Format chunk
            if include_citations and source_info:
                chunk_text = f"{citation_marker} {source_str}\n{chunk.text}"
            
            elif include_citations:
                chunk_text = f"{citation_marker}\n{chunk.text}"
            
            else:
                chunk_text = chunk.text
            
            context_parts.append(chunk_text)
        
        return "\n\n".join(context_parts)
    

    def _format_simple(self, chunks: List[ChunkWithScore], include_citations: bool) -> str:
        """
        Simple formatting without extensive metadata
        """
        context_parts = list()

        for i, chunk_with_score in enumerate(chunks, 1):
            chunk = chunk_with_score.chunk
            
            if include_citations:
                context_parts.append(f"[{i}] {chunk.text}")
            
            else:
                context_parts.append(chunk.text)
        
        return "\n\n".join(context_parts)
    

    def _count_tokens(self, text: str) -> int:
        """
        Count tokens in text with conservative fallback
        
        Arguments:
        ----------
            text { str } : Text to count tokens for
        
        Returns:
        --------
            { int }      : Token count
        """
        if not text:
            return 0
        
        try:
            token_counter = get_token_counter()

            return token_counter.count_tokens(text)
        
        except Exception as e:
            # Conservative fallback calculation for technical text
            self.logger.debug(f"Token counter error, using conservative approximation: {repr(e)}")
            
            # More accurate approximation for technical/scientific text: Count words (split by whitespace)
            words = text.split()
            
            # Technical text has more subword tokenization: Average: 1 word ≈ 1.8 tokens for technical English (conservative)
            estimated_tokens = int(len(words) * 1.8)
            
            # Add overhead for punctuation, numbers, special chars (15%)
            estimated_tokens = int(estimated_tokens * 1.15)
            
            # Add safety margin (10%)
            estimated_tokens = int(estimated_tokens * 1.10)
            
            # Ensure minimum reasonable value
            estimated_tokens = max(10, estimated_tokens)
            
            self.logger.debug(f"Conservative estimate: {len(words)} words → {estimated_tokens} tokens")
            
            return estimated_tokens
    

    def optimize_context_quality(self, context: str, chunks: List[ChunkWithScore]) -> str:
        """
        Optimize context quality by removing redundancies and improving flow
        """
        # Remove duplicate sentences
        sentences        = context.split('. ')
        unique_sentences = list()

        for sentence in sentences:
            sentence_clean = sentence.strip()
            
            if (sentence_clean and (sentence_clean not in unique_sentences)):
                unique_sentences.append(sentence_clean)
        
        optimized = '. '.join(unique_sentences)
        
        # Ensure proper citation consistency
        if '[' in optimized:
            optimized = self.citation_tracker.ensure_citation_consistency(optimized, chunks)
        
        return optimized
    

    def get_context_statistics(self, context: str, chunks: List[ChunkWithScore]) -> dict:
        """
        Get statistics about assembled context
        """
        token_count    = self._count_tokens(text = context)
        char_count     = len(context)
        
        # Citation statistics
        citation_stats = self.citation_tracker.get_citation_statistics(context, chunks)
        
        # Source diversity
        source_docs    = set(chunk.chunk.document_id for chunk in chunks)
        source_pages   = set(chunk.chunk.page_number for chunk in chunks if chunk.chunk.page_number)
        
        return {"total_tokens"      : token_count,
                "total_chars"       : char_count,
                "chunk_count"       : len(chunks),
                "source_documents"  : len(source_docs),
                "source_pages"      : len(source_pages),
                "token_utilization" : (token_count / self.max_context_tokens * 100) if self.max_context_tokens > 0 else 0,
                "citation_stats"    : citation_stats,
                "strategy"          : self.strategy,
               }


# Global context assembler instance
_context_assembler = None


def get_context_assembler() -> ContextAssembler:
    """
    Get global context assembler instance
    """
    global _context_assembler
    
    if _context_assembler is None:
        _context_assembler = ContextAssembler()
    
    return _context_assembler


@handle_errors(error_type = ContextAssemblyError, log_error = True, reraise = False)
def assemble_context_for_llm(chunks: List[ChunkWithScore], query: str = "", **kwargs) -> str:
    """
    Convenience function for context assembly
    """
    assembler = get_context_assembler()
    return assembler.assemble_context(chunks, query, **kwargs)