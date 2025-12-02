# DEPENDENCIES
import re
import time
import asyncio
from typing import Dict
from typing import List
from typing import Optional
from datetime import datetime
from config.models import QueryRequest
from config.models import QueryResponse
from config.models import ChunkWithScore
from config.settings import get_settings
from config.models import EvaluationResult
from config.logging_config import get_logger
from utils.error_handler import RAGException
from utils.error_handler import handle_errors


# Setup Settings and Logging
settings = get_settings()
logger   = get_logger(__name__)


class RAGASEvaluator:
    """
    RAGAS (Retrieval-Augmented Generation Assessment) evaluator for RAG pipeline quality metrics
    
    Evaluates:
    - Answer Relevancy: How well the answer addresses the question
    - Faithfulness: Whether the answer is grounded in the retrieved context
    - Context Precision: Whether relevant chunks are ranked high
    - Context Recall: Whether all necessary information was retrieved
    """
    def __init__(self):
        """
        Initialize RAGAS evaluator
        """
        self.logger             = logger
        self.settings           = get_settings()
        
        # Evaluation history for session tracking: session_id -> List[EvaluationResult]
        self.evaluation_history = dict()  
        
        self.logger.info("Initialized RAGASEvaluator")
    
    
    @handle_errors(error_type = RAGException, log_error = True, reraise = False)
    async def evaluate_response(self, query: str, answer: str, context_chunks: List[ChunkWithScore], session_id: Optional[str] = None, reference_answer: Optional[str] = None) -> EvaluationResult:
        """
        Evaluate a RAG response using RAGAS metrics
        
        Arguments:
        ----------
            query            { str }  : User query

            answer           { str }  : Generated answer
            
            context_chunks   { list } : Retrieved context chunks
            
            session_id       { str }  : Session ID for tracking
            
            reference_answer { str }  : Ground truth answer (optional)
        
        Returns:
        --------
            { EvaluationResult }      : Evaluation metrics
        """
        start_time = time.time()
        
        self.logger.info(f"Evaluating response for query: '{query[:50]}...'")
        
        try:
            # Calculate individual metrics
            answer_relevancy  = await self._calculate_answer_relevancy(query, answer)
            faithfulness      = await self._calculate_faithfulness(answer, context_chunks)
            context_precision = await self._calculate_context_precision(query, context_chunks)
            context_recall    = await self._calculate_context_recall(query, context_chunks, reference_answer) if reference_answer else None
                                        
            # Calculate overall score (weighted average)
            weights           = {'answer_relevancy'  : 0.35,
                                 'faithfulness'      : 0.35,
                                 'context_precision' : 0.30,
                                }
            
            overall_score     = (answer_relevancy * weights['answer_relevancy'] + faithfulness * weights['faithfulness'] + context_precision * weights['context_precision'])
            
            # Convert to ms
            evaluation_time   = (time.time() - start_time) * 1000 
            
            result            = EvaluationResult(answer_relevancy   = answer_relevancy,
                                                 faithfulness       = faithfulness,
                                                 context_precision  = context_precision,
                                                 context_recall     = context_recall,
                                                 overall_score      = overall_score,
                                                 evaluation_time_ms = evaluation_time,
                                                 model_used         = "ragas_heuristic",
                                                 timestamp          = datetime.now(),
                                                )
            
            # Store in history if session_id provided
            if session_id:
                if session_id not in self.evaluation_history:
                    self.evaluation_history[session_id] = []

                self.evaluation_history[session_id].append(result)
            
            self.logger.info(f"Evaluation complete: overall_score={overall_score:.3f}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Evaluation failed: {repr(e)}", exc_info = True)
            # Return default metrics on failure
            return EvaluationResult(answer_relevancy   = 0.0,
                                    faithfulness       = 0.0,
                                    context_precision  = 0.0,
                                    context_recall     = None,
                                    overall_score      = 0.0,
                                    evaluation_time_ms = 0.0,
                                    model_used         = "ragas_heuristic_failed",
                                    timestamp          = datetime.now(),
                                   )
    
    
    async def _calculate_answer_relevancy(self, query: str, answer: str) -> float:
        """
        Calculate how well the answer addresses the question: Uses keyword overlap and length heuristics
        """
        try:
            # Tokenize query and answer
            query_tokens    = set(self._tokenize(query.lower()))
            answer_tokens   = set(self._tokenize(answer.lower()))
            
            if not query_tokens or not answer_tokens:
                return 0.0
            
            # Calculate keyword overlap
            overlap         = len(query_tokens & answer_tokens)
            overlap_ratio   = overlap / len(query_tokens)
            
            # Penalize very short answers
            answer_words    = len(answer.split())

            # Optimal around 20+ words
            length_penalty  = min(1.0, answer_words / 20)  
            
            # Penalize very long answers (possibly off-topic)
            if (answer_words > 200):
                length_penalty *= 0.9
            
            # Check if answer is a refusal/no-answer
            refusal_phrases = ['cannot answer', 'not found', 'no information', 'unable to']
            is_refusal      = any(phrase in answer.lower() for phrase in refusal_phrases)
            
            if is_refusal:
                # Low but not zero score for honest refusal
                return 0.2  
            
            # Combine metrics
            relevancy      = (overlap_ratio * 0.6 + length_penalty * 0.4)
            
            # Normalize to [0, 1]
            relevancy      = max(0.0, min(1.0, relevancy))
            
            return relevancy
            
        except Exception as e:
            self.logger.warning(f"Answer relevancy calculation failed: {repr(e)}")
            # Neutral score on error
            return 0.5  
    
    
    async def _calculate_faithfulness(self, answer: str, context_chunks: List[ChunkWithScore]) -> float:
        """
        Calculate whether the answer is grounded in the retrieved context: Checks if answer statements can be found in context
        """
        try:
            if not context_chunks:
                return 0.0
            
            # Combine all context
            full_context        = " ".join([chunk.chunk.text for chunk in context_chunks])
            full_context_lower  = full_context.lower()
            
            # Split answer into sentences
            answer_sentences    = self._split_sentences(answer)
            
            if not answer_sentences:
                return 0.0
            
            # Check how many answer sentences have support in context
            supported_sentences = 0
            total_sentences     = len(answer_sentences)
            
            for sentence in answer_sentences:
                sentence_lower = sentence.lower()
                
                # Skip very short sentences
                if (len(sentence.split()) < 3):
                    continue
                
                # Check for keyword overlap with context
                sentence_tokens = set(self._tokenize(sentence_lower))
                context_tokens  = set(self._tokenize(full_context_lower))
                
                overlap         = len(sentence_tokens & context_tokens)
                overlap_ratio   = overlap / len(sentence_tokens) if sentence_tokens else 0
                
                # Consider sentence supported if >40% keyword overlap
                if (overlap_ratio > 0.4):
                    supported_sentences += 1
            
            # Calculate faithfulness score
            faithfulness   = supported_sentences / total_sentences if total_sentences > 0 else 0.0
            
            # Boost score if citations are present
            citation_count = answer.count('[') + answer.count('(')
            
            if (citation_count > 0):
                citation_boost = min(0.15, citation_count * 0.03)
                faithfulness   = min(1.0, faithfulness + citation_boost)
            
            return faithfulness
            
        except Exception as e:
            self.logger.warning(f"Faithfulness calculation failed: {repr(e)}")
            # Neutral score on error
            return 0.5 
    
    
    async def _calculate_context_precision(self, query: str, context_chunks: List[ChunkWithScore]) -> float:
        """
        Calculate whether relevant chunks are ranked high: Checks if top-ranked chunks have high relevance scores
        """
        try:
            if not context_chunks:
                return 0.0
            
            # Extract scores from top chunks: Top 5
            scores         = [chunk.score for chunk in context_chunks[:5]] 
            
            if not scores:
                return 0.0
            
            # Calculate weighted precision: Higher weight for top-ranked chunks
            weights        = [0.4, 0.3, 0.2, 0.1, 0.0] if (len(scores) >= 5) else [1.0 / len(scores)] * len(scores)
            
            weighted_score = sum(score * weight for score, weight in zip(scores, weights[:len(scores)]))
            
            # Normalize by checking score distribution (Good precision: scores decrease gradually & Bad precision: flat or random distribution)
            if (len(scores) > 1):
                score_variance  = sum((s - sum(scores)/len(scores))**2 for s in scores) / len(scores)

                # Reward variance
                variance_factor = min(1.0, score_variance * 2)  
                weighted_score  = weighted_score * (0.7 + 0.3 * variance_factor)
            
            return min(1.0, max(0.0, weighted_score))
            
        except Exception as e:
            self.logger.warning(f"Context precision calculation failed: {repr(e)}")
            # Neutral score on error
            return 0.5  
    
    
    async def _calculate_context_recall(self, query: str, context_chunks: List[ChunkWithScore], reference_answer: Optional[str]) -> Optional[float]:
        """
        Calculate whether all necessary information was retrieved: Requires reference answer for accurate calculation
        """
        try:
            if not reference_answer or not context_chunks:
                return None
            
            # Combine all context
            full_context     = " ".join([chunk.chunk.text for chunk in context_chunks])
            
            # Tokenize reference answer and context
            reference_tokens = set(self._tokenize(reference_answer.lower()))
            context_tokens   = set(self._tokenize(full_context.lower()))
            
            if not reference_tokens:
                return None
            
            # Calculate recall: what % of reference tokens are in context
            overlap          = len(reference_tokens & context_tokens)
            recall           = overlap / len(reference_tokens)
            
            # Boost recall if we have many chunks (more coverage)
            coverage_boost   = min(0.1, len(context_chunks) * 0.01)
            recall           = min(1.0, recall + coverage_boost)
            
            return recall
            
        except Exception as e:
            self.logger.warning(f"Context recall calculation failed: {repr(e)}")
            return None
    
    
    def _tokenize(self, text: str) -> List[str]:
        """
        Simple tokenization (split on whitespace and punctuation)
        """
        # Remove punctuation and split
        tokens = re.findall(r'\b\w+\b', text.lower())

        # Filter very short tokens
        return [t for t in tokens if len(t) > 2]  
    
    
    def _split_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences
        """
        # Simple sentence splitter
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    
    def get_session_metrics(self, session_id: str) -> Dict:
        """
        Get aggregated metrics for a session
        
        Arguments:
        ----------
            session_id { str } : Session ID
        
        Returns:
        --------
            { dict }           : Aggregated metrics
        """
        if session_id not in self.evaluation_history:
            return {'total_evaluations'     : 0,
                    'avg_answer_relevancy'  : 0.0,
                    'avg_faithfulness'      : 0.0,
                    'avg_context_precision' : 0.0,
                    'avg_overall_score'     : 0.0,
                    'evaluations'           : []
                   }
         
        evaluations = self.evaluation_history[session_id]
        
        if not evaluations:
            return {'total_evaluations'     : 0,
                    'avg_answer_relevancy'  : 0.0,
                    'avg_faithfulness'      : 0.0,
                    'avg_context_precision' : 0.0,
                    'avg_overall_score'     : 0.0,
                    'evaluations'           : [],
                   }
        
        # Calculate averages
        avg_answer_relevancy  = sum(e.answer_relevancy for e in evaluations) / len(evaluations)
        avg_faithfulness      = sum(e.faithfulness for e in evaluations) / len(evaluations)
        avg_context_precision = sum(e.context_precision for e in evaluations) / len(evaluations)
        avg_overall_score     = sum(e.overall_score for e in evaluations) / len(evaluations)
        
        # Get individual evaluations
        eval_list             = list()

        for i, eval_result in enumerate(evaluations, 1):
            eval_list.append({'query_number'      : i,
                              'answer_relevancy'  : round(eval_result.answer_relevancy, 3),
                              'faithfulness'      : round(eval_result.faithfulness, 3),
                              'context_precision' : round(eval_result.context_precision, 3),
                              'overall_score'     : round(eval_result.overall_score, 3),
                              'timestamp'         : eval_result.timestamp.isoformat(),
                            })
        
        return {'total_evaluations'     : len(evaluations),
                'avg_answer_relevancy'  : round(avg_answer_relevancy, 3),
                'avg_faithfulness'      : round(avg_faithfulness, 3),
                'avg_context_precision' : round(avg_context_precision, 3),
                'avg_overall_score'     : round(avg_overall_score, 3),
                'evaluations'           : eval_list,
               }
    
    
    def clear_session(self, session_id: str):
        """
        Clear evaluation history for a session
        
        Arguments:
        ----------
            session_id { str } : Session ID to clear
        """
        if session_id in self.evaluation_history:
            del self.evaluation_history[session_id]
            self.logger.info(f"Cleared evaluation history for session: {session_id}")
    
    
    def get_all_sessions(self) -> List[str]:
        """
        Get all active session IDs
        
        Returns:
        --------
            { list } : List of session IDs
        """
        return list(self.evaluation_history.keys())


# Global evaluator instance
_ragas_evaluator = None


def get_ragas_evaluator() -> RAGASEvaluator:
    """
    Get global RAGAS evaluator instance
    
    Returns:
    --------
        { RAGASEvaluator } : RAGASEvaluator instance
    """
    global _ragas_evaluator
    
    if _ragas_evaluator is None:
        _ragas_evaluator = RAGASEvaluator()
    
    return _ragas_evaluator


@handle_errors(error_type=RAGException, log_error=True, reraise=False)
async def evaluate_rag_response(query: str, answer: str, context_chunks: List[ChunkWithScore], session_id: Optional[str] = None) -> EvaluationResult:
    """
    Convenience function for evaluating RAG responses
    
    Arguments:
    ----------
        query          { str }  : User query

        answer         { str }  : Generated answer
        
        context_chunks { list } : Retrieved context chunks
        
        session_id     { str }  : Session ID for tracking
    
    Returns:
    --------
        { EvaluationResult }    : Evaluation metrics
    """
    evaluator = get_ragas_evaluator()
    
    return await evaluator.evaluate_response(query, answer, context_chunks, session_id)