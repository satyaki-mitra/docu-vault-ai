# DEPENDENCIES
import os
import math
import logging
import statistics
from typing import Any
from typing import List
from typing import Dict
from ragas import evaluate
from typing import Optional
from datasets import Dataset
from datetime import datetime
from ragas.metrics import faithfulness
from config.settings import get_settings
from ragas.metrics import context_recall
from config.models import RAGASStatistics
from config.models import RAGASExportData
from ragas.metrics import answer_relevancy
from ragas.metrics import context_precision
from ragas.metrics import context_relevancy
from ragas.metrics import answer_similarity
from ragas.metrics import answer_correctness
from config.logging_config import get_logger
from ragas.metrics import context_utilization
from config.models import RAGASEvaluationResult


# Setup Logging
settings = get_settings()
logger   = get_logger(__name__)


# Set OpenAI API key from settings
if (hasattr(settings, 'OPENAI_API_KEY') and settings.OPENAI_API_KEY):
    os.environ["OPENAI_API_KEY"] = settings.OPENAI_API_KEY
    logger.info("OpenAI API key loaded from settings")

else:
    logger.warning("OPENAI_API_KEY not found in settings. Please add it to your .env file.")

# Supressing Warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def sanitize_ragas_score(value: Any, metric_name: str = "unknown") -> float:
    """
    Sanitize a single RAGAS score to handle NaN, None, and invalid values
    
    Arguments:
    ----------
        value       { Any } : Raw score value
        
        metric_name { str } : Name of the metric (for logging)
    
    Returns:
    --------
          { float }         : Valid float between 0.0 and 1.0
    """
    # Handle None
    if value is None:
        return 0.0
    
    # Handle NaN and infinity
    try:
        float_val = float(value)
        
        if math.isnan(float_val) or math.isinf(float_val):
            logger.warning(f"Invalid RAGAS score for {metric_name}: {value}, defaulting to 0.0")
            return 0.0
        
        # Clamp between 0 and 1
        return max(0.0, min(1.0, float_val))
    
    except (ValueError, TypeError):
        logger.warning(f"Could not convert RAGAS score for {metric_name}: {value}, defaulting to 0.0")
        return 0.0


class RAGASEvaluator:
    """
    RAGAS evaluation module for RAG system quality assessment
    """
    def __init__(self, enable_ground_truth_metrics: bool = False):
        """
        Initialize RAGAS evaluator
        
        Arguments:
        ----------
            enable_ground_truth_metrics { bool } : Whether to compute metrics requiring ground truth
        """
        self.enable_ground_truth                              = enable_ground_truth_metrics
        
        # Metrics that don't require ground truth (UPDATED)
        self.base_metrics                                     = [answer_relevancy, 
                                                                 faithfulness,
                                                                 context_utilization, 
                                                                 context_relevancy,
                                                                ]
        
        # Metrics requiring ground truth
        self.ground_truth_metrics                             = [context_precision,
                                                                 context_recall,
                                                                 answer_similarity,
                                                                 answer_correctness,
                                                                ]
        
        # Store evaluation history
        self.evaluation_history : List[RAGASEvaluationResult] = list()
        self.session_start                                    = datetime.now()
        
        logger.info(f"RAGAS Evaluator initialized (ground_truth_metrics: {enable_ground_truth_metrics})")
    
    
    def evaluate_single(self, query: str, answer: str, contexts: List[str], ground_truth: Optional[str] = None, retrieval_time_ms: int = 0,
                        generation_time_ms: int = 0, total_time_ms: int = 0, chunks_retrieved: int = 0, query_type: str = "rag") -> RAGASEvaluationResult:
        """
        Evaluate a single query-answer pair using RAGAS metrics
        
        Arguments:
        ----------
            query              { str }  : User query
            
            answer             { str }  : Generated answer
            
            contexts           { list } : Retrieved context chunks
            
            ground_truth       { str }  : Reference answer (optional)
            
            retrieval_time_ms  { int }  : Retrieval time in milliseconds
            
            generation_time_ms { int }  : Generation time in milliseconds
            
            total_time_ms      { int }  : Total time in milliseconds
            
            chunks_retrieved   { int }  : Number of chunks retrieved

            query_type         { str }  : Type of the query : RAG or non-RAG
            
        Returns:
        --------
            { RAGASEvaluationResult }   : RAGASEvaluationResult object
        """
        try:
            logger.info(f"Evaluating {query_type.upper()}, query: {query[:100]}...")

            if ((query_type == "general") or (query_type == "non-rag")):
                logger.info(f"Skipping detailed RAGAS evaluation for {query_type} query")
            
                return RAGASEvaluationResult(query               = query,
                                             answer              = answer,
                                             contexts            = contexts,
                                             ground_truth        = ground_truth,
                                             timestamp           = datetime.now().isoformat(),
                                             answer_relevancy    = 0.0,  # N/A for non-RAG
                                             faithfulness        = 0.0,  # N/A for non-RAG
                                             context_utilization = None,
                                             context_precision   = None,
                                             context_relevancy   = 0.0,  # N/A for non-RAG
                                             context_recall      = None,
                                             answer_similarity   = None,
                                             answer_correctness  = None,
                                             retrieval_time_ms   = retrieval_time_ms,
                                             generation_time_ms  = generation_time_ms,
                                             total_time_ms       = total_time_ms,
                                             chunks_retrieved    = chunks_retrieved,
                                             query_type          = query_type,
                                            )
            
            # Only for RAG queries : Validate inputs
            if not contexts or not any(c.strip() for c in contexts):
                logger.warning("No valid contexts provided for RAGAS evaluation")
                raise ValueError("No valid contexts for evaluation")
            
            # Prepare dataset for RAGAS
            eval_data = {"question" : [query],
                         "answer"   : [answer],
                         "contexts" : [contexts],
                        }
            
            # Add ground truth if available
            if ground_truth and self.enable_ground_truth:
                eval_data["ground_truth"] = [ground_truth]
            
            # Create dataset
            dataset = Dataset.from_dict(eval_data)
            
            # Select metrics based on ground truth availability
            if (ground_truth and self.enable_ground_truth):
                metrics_to_use = self.base_metrics + self.ground_truth_metrics
            
            else:
                metrics_to_use = self.base_metrics
            
            # Run evaluation
            logger.info(f"Running RAGAS evaluation with {len(metrics_to_use)} metrics...")

            results                 = evaluate(dataset, metrics = metrics_to_use)
            
            # Extract scores
            scores                  = results.to_pandas().iloc[0].to_dict()
            
            # Sanitize all scores to handle NaN values
            answer_relevancy        = sanitize_ragas_score(scores.get('answer_relevancy'), 'answer_relevancy')
            
            faithfulness            = sanitize_ragas_score(scores.get('faithfulness'), 'faithfulness')
            
            context_utilization_val = sanitize_ragas_score(scores.get('context_utilization'), 'context_utilization') if not ground_truth else None
            
            context_relevancy_val   = sanitize_ragas_score(scores.get('context_relevancy'), 'context_relevancy')
            
            # Ground truth metrics (sanitized)
            context_precision_val   = None
            context_recall_val      = None
            answer_similarity_val   = None
            answer_correctness_val  = None

            if (ground_truth and ('context_precision' in scores)):
                context_precision_val  = sanitize_ragas_score(scores.get('context_precision'), 'context_precision')
            
            

            if (ground_truth and ('context_recall' in scores)):
                context_recall_val     = sanitize_ragas_score(scores.get('context_recall'), 'context_recall')
            
            
            if ground_truth and 'answer_similarity' in scores:
                answer_similarity_val  = sanitize_ragas_score(scores.get('answer_similarity'), 'answer_similarity')
            
            
            if ground_truth and 'answer_correctness' in scores:
                answer_correctness_val = sanitize_ragas_score(scores.get('answer_correctness'), 'answer_correctness')
            
            # Create result object with sanitized values
            result = RAGASEvaluationResult(query               = query,
                                           answer              = answer,
                                           contexts            = contexts,
                                           ground_truth        = ground_truth,
                                           timestamp           = datetime.now().isoformat(),
                                           answer_relevancy    = answer_relevancy,
                                           faithfulness        = faithfulness,
                                           context_utilization = context_utilization_val,
                                           context_precision   = context_precision_val,
                                           context_relevancy   = context_relevancy_val,
                                           context_recall      = context_recall_val,
                                           answer_similarity   = answer_similarity_val,
                                           answer_correctness  = answer_correctness_val,
                                           retrieval_time_ms   = retrieval_time_ms,
                                           generation_time_ms  = generation_time_ms,
                                           total_time_ms       = total_time_ms,
                                           chunks_retrieved    = chunks_retrieved,
                                           query_type          = query_type,
                                          )
            
            # Store in history
            self.evaluation_history.append(result)
            
            # Log results
            if ground_truth:
                logger.info(f"Evaluation complete: relevancy={result.answer_relevancy:.3f}, faithfulness={result.faithfulness:.3f}, precision={result.context_precision:.3f}, overall={result.overall_score:.3f}")
            
            else:
                logger.info(f"Evaluation complete: relevancy={result.answer_relevancy:.3f}, faithfulness={result.faithfulness:.3f}, utilization={result.context_utilization:.3f}, overall={result.overall_score:.3f}")
            
            return result
            
        except Exception as e:
            logger.error(f"RAGAS evaluation failed for {query_type} query: {e}", exc_info = True)
            
            # Return zero metrics on failure (all sanitized)
            return RAGASEvaluationResult(query               = query,
                                         answer              = answer,
                                         contexts            = contexts,
                                         ground_truth        = ground_truth,
                                         timestamp           = datetime.now().isoformat(),
                                         answer_relevancy    = 0.0,
                                         faithfulness        = 0.0,
                                         context_utilization = 0.0 if not ground_truth else None,
                                         context_precision   = None if not ground_truth else 0.0,
                                         context_relevancy   = 0.0,
                                         context_recall      = None,
                                         answer_similarity   = None,
                                         answer_correctness  = None,
                                         retrieval_time_ms   = retrieval_time_ms,
                                         generation_time_ms  = generation_time_ms,
                                         total_time_ms       = total_time_ms,
                                         chunks_retrieved    = chunks_retrieved,
                                         query_type          = query_type
                                        )
    

    def evaluate_query_response(self, query_response: Any) -> Dict:
        """
        Evaluate based on actual response characteristics, not predictions
        
        Arguments:
        ----------
            query_response { Any } : QueryResponse object with metadata
        
        Returns:
        --------
                { dict }           : RAGAS evaluation results
        """
        try:
            # Extract necessary data from response object: Check if it has the attributes we need
            if (hasattr(query_response, 'sources')):
                sources = query_response.sources

            elif hasattr(query_response, 'contexts'):
                sources = query_response.contexts

            else:
                sources = []
            
            # Extract context from sources
            contexts = list()

            if (sources and len(sources) > 0):
                if (hasattr(sources[0], 'content')):
                    contexts = [s.content for s in sources]

                elif ((isinstance(sources[0], dict)) and ('content' in sources[0])):
                    contexts = [s['content'] for s in sources]

                elif (isinstance(sources[0], str)):
                    contexts = sources
            
            # Check if this is actually a RAG response
            is_actual_rag = ((sources and len(sources) > 0) or (contexts and len(contexts) > 0) or (hasattr(query_response, 'metrics') and  query_response.metrics and query_response.metrics.get("execution_path") == "rag_pipeline"))
            
            if not is_actual_rag:
                logger.info(f"Non-RAG response, skipping RAGAS evaluation")
                return {"evaluated" : False,
                        "reason"    : "Not a RAG response",
                        "is_rag"    : False,
                       }
            
            # Get query and answer
            query  = getattr(query_response, 'query', '')
            answer = getattr(query_response, 'answer', '')
            
            if not query or not answer:
                logger.warning("Missing query or answer for evaluation")
                return {"evaluated" : False,
                        "reason"    : "Missing query or answer",
                        "is_rag"    : True,
                       }
            
            # Check if context exists in metrics
            if (hasattr(query_response, 'metrics') and query_response.metrics):
                if (query_response.metrics.get("context_for_evaluation")):
                    contexts = [query_response.metrics["context_for_evaluation"]]
            
            if ((not contexts) or (not any(c.strip() for c in contexts))):
                logger.warning("No context available for RAGAS evaluation")
                return {"evaluated" : False,
                        "reason"    : "No context available",
                        "is_rag"    : True,
                       }
            
            # Try to get query_type from query_response
            if (hasattr(query_response, 'query_type')):
                detected_query_type = query_response.query_type
            
            elif (hasattr(query_response, 'metrics') and query_response.metrics):
                detected_query_type = query_response.metrics.get("query_type", "rag")
            
            else:
                # Determine based on contexts
                detected_query_type = "rag" if (contexts and (len(contexts) > 0)) else "general"

            # Now use the existing evaluate_single method
            result                       = self.evaluate_single(query              = query,
                                                                answer             = answer,
                                                                contexts           = contexts,
                                                                ground_truth       = None,
                                                                retrieval_time_ms  = getattr(query_response, 'retrieval_time_ms', 0),
                                                                generation_time_ms = getattr(query_response, 'generation_time_ms', 0),
                                                                total_time_ms      = getattr(query_response, 'total_time_ms', 0),
                                                                chunks_retrieved   = len(sources) if sources else len(contexts),
                                                                query_type         = detected_query_type,
                                                               )
            
            # Convert to dict and add metadata
            result_dict                  = result.to_dict() if hasattr(result, 'to_dict') else vars(result)
            
            # Add evaluation metadata
            result_dict["evaluated"]     = True
            result_dict["is_rag"]        = True
            result_dict["context_count"] = len(contexts)
            
            # Add prediction vs reality info if available
            if ((hasattr(query_response, 'metrics')) and query_response.metrics):
                result_dict["predicted_type"]      = query_response.metrics.get("predicted_type", "unknown")
                result_dict["actual_type"]         = query_response.metrics.get("actual_type", "unknown")
                result_dict["confidence_mismatch"] = (query_response.metrics.get("predicted_type") != query_response.metrics.get("actual_type"))
            
            logger.info(f"RAGAS evaluation completed for RAG response")
            return result_dict
            
        except Exception as e:
            logger.error(f"Query response evaluation failed: {repr(e)}", exc_info = True)
            
            return {"evaluated" : False,
                    "error"     : str(e),
                    "is_rag"    : True,
                   }

    
    def evaluate_batch(self, queries: List[str], answers: List[str], contexts_list: List[List[str]], ground_truths: Optional[List[str]] = None,
                       query_types: Optional[List[str]] = None) -> List[RAGASEvaluationResult]:
        """
        Evaluate multiple query-answer pairs in batch
        
        Arguments:
        ----------
            queries       { list } : List of user queries

            answers       { list } : List of generated answers
            
            contexts_list { list } : List of context lists
            
            ground_truths { list } : List of reference answers (optional)

            query_types   { list } : List of query types RAG / non-RAG
            
        Returns:
        --------
                  { list }         : List of RAGASEvaluationResult objects
        """
        try:
            logger.info(f"Batch evaluating {len(queries)} queries...")
            
            # Prepare dataset
            eval_data = {"question" : queries,
                         "answer"   : answers,
                         "contexts" : contexts_list,
                        }
            
            if ground_truths and self.enable_ground_truth:
                eval_data["ground_truth"] = ground_truths
            
            # Create dataset
            dataset = Dataset.from_dict(eval_data)
            
            # Select metrics
            if (ground_truths and self.enable_ground_truth):
                metrics_to_use = self.base_metrics + self.ground_truth_metrics

            else:
                metrics_to_use = self.base_metrics
            
            # Run evaluation
            results            = evaluate(dataset, metrics = metrics_to_use)
            results_df         = results.to_pandas()
            
            # Create result objects
            evaluation_results = list()

            for idx, row in results_df.iterrows():
                # Determine query_type for this item
                if query_types and idx < len(query_types):
                    current_query_type = query_types[idx]

                else:
                    # Default based on whether contexts are available
                    current_query_type = "rag" if contexts_list[idx] and len(contexts_list[idx]) > 0 else "general"
            
                # Sanitize all scores
                answer_relevancy_val    = sanitize_ragas_score(row.get('answer_relevancy', 0.0), f'answer_relevancy_{idx}')
                
                faithfulness_val        = sanitize_ragas_score(row.get('faithfulness', 0.0), f'faithfulness_{idx}')
                
                context_relevancy_val   = sanitize_ragas_score(row.get('context_relevancy', 0.0), f'context_relevancy_{idx}')
                
                # Handle context_utilization vs context_precision
                context_utilization_val = sanitize_ragas_score(row.get('context_utilization'), f'context_utilization_{idx}') if not ground_truths else None
                
                context_precision_val   = sanitize_ragas_score(row.get('context_precision'), f'context_precision_{idx}') if (ground_truths and 'context_precision' in row) else None
            
                # Ground truth metrics
                context_recall_val      = sanitize_ragas_score(row.get('context_recall'), f'context_recall_{idx}') if (ground_truths and 'context_recall' in row) else None
                
                answer_similarity_val   = sanitize_ragas_score(row.get('answer_similarity'), f'answer_similarity_{idx}') if (ground_truths and 'answer_similarity' in row) else None
                
                answer_correctness_val  = sanitize_ragas_score(row.get('answer_correctness'), f'answer_correctness_{idx}') if (ground_truths and 'answer_correctness' in row) else None
                
                # For non-RAG queries, set appropriate scores
                if ((current_query_type == "general") or (current_query_type == "non-rag")):
                    # Non-RAG queries shouldn't have RAGAS metrics
                    answer_relevancy_val    = 0.0
                    faithfulness_val        = 0.0
                    context_relevancy_val   = 0.0
                    context_utilization_val = None
                    context_precision_val   = None
            
                result                  = RAGASEvaluationResult(query               = queries[idx],
                                                                answer              = answers[idx],
                                                                contexts            = contexts_list[idx],
                                                                ground_truth        = ground_truths[idx] if ground_truths else None,
                                                                timestamp           = datetime.now().isoformat(),
                                                                answer_relevancy    = answer_relevancy_val,
                                                                faithfulness        = faithfulness_val,
                                                                context_precision   = context_precision_val,
                                                                context_utilization = context_utilization_val,
                                                                context_relevancy   = context_relevancy_val,
                                                                context_recall      = context_recall_val,
                                                                answer_similarity   = answer_similarity_val,
                                                                answer_correctness  = answer_correctness_val,
                                                                retrieval_time_ms   = 0,
                                                                generation_time_ms  = 0,
                                                                total_time_ms       = 0,
                                                                chunks_retrieved    = len(contexts_list[idx]),
                                                                query_type          = current_query_type,
                                                               )

                evaluation_results.append(result)

                self.evaluation_history.append(result)
            
            logger.info(f"Batch evaluation complete for {len(evaluation_results)} queries")
            return evaluation_results
            
        except Exception as e:
            logger.error(f"Batch evaluation failed: {e}", exc_info = True)
            return []
    
    
    def get_session_statistics(self) -> RAGASStatistics:
        """
        Get aggregate statistics for the current evaluation session
        
        Returns:
        ---------
            { RAGASStatistics } : RAGASStatistics object with aggregate metrics
        """
        if not self.evaluation_history:
            # Return empty statistics
            return RAGASStatistics(total_evaluations       = 0,
                                   avg_answer_relevancy    = 0.0,
                                   avg_faithfulness        = 0.0,
                                   avg_context_precision   = 0.0,
                                   avg_context_utilization = 0.0,
                                   avg_context_relevancy   = 0.0,
                                   avg_overall_score       = 0.0,
                                   avg_retrieval_time_ms   = 0.0,
                                   avg_generation_time_ms  = 0.0,
                                   avg_total_time_ms       = 0.0,
                                   min_score               = 0.0,
                                   max_score               = 0.0,
                                   std_dev                 = 0.0,
                                   session_start           = self.session_start,
                                   last_updated            = datetime.now(),
                                  )
        
        n                  = len(self.evaluation_history)
        
        # Calculate averages
        avg_relevancy      = sum(r.answer_relevancy for r in self.evaluation_history) / n
        avg_faithfulness   = sum(r.faithfulness for r in self.evaluation_history) / n
        
        # Calculate context_precision and context_utilization separately
        precision_values   = [r.context_precision for r in self.evaluation_history if r.context_precision is not None]
        utilization_values = [r.context_utilization for r in self.evaluation_history if r.context_utilization is not None]
        
        avg_precision      = sum(precision_values) / len(precision_values) if precision_values else 0.0
        avg_utilization    = sum(utilization_values) / len(utilization_values) if utilization_values else 0.0
        
        avg_relevancy_ctx  = sum(r.context_relevancy for r in self.evaluation_history) / n
        
        # Overall scores
        overall_scores     = [r.overall_score for r in self.evaluation_history]
        avg_overall        = sum(overall_scores) / n
        min_score          = min(overall_scores)
        max_score          = max(overall_scores)
        std_dev            = statistics.stdev(overall_scores) if n > 1 else 0.0
        
        # Performance averages
        avg_retrieval      = sum(r.retrieval_time_ms for r in self.evaluation_history) / n
        avg_generation     = sum(r.generation_time_ms for r in self.evaluation_history) / n
        avg_total          = sum(r.total_time_ms for r in self.evaluation_history) / n
        
        # Ground truth metrics (if available)
        recall_values      = [r.context_recall for r in self.evaluation_history if r.context_recall is not None]
        similarity_values  = [r.answer_similarity for r in self.evaluation_history if r.answer_similarity is not None]
        correctness_values = [r.answer_correctness for r in self.evaluation_history if r.answer_correctness is not None]
        
        return RAGASStatistics(total_evaluations       = n,
                               avg_answer_relevancy    = round(avg_relevancy, 3),
                               avg_faithfulness        = round(avg_faithfulness, 3),
                               avg_context_precision   = round(avg_precision, 3) if precision_values else None,
                               avg_context_utilization = round(avg_utilization, 3) if utilization_values else None,
                               avg_context_relevancy   = round(avg_relevancy_ctx, 3),
                               avg_overall_score       = round(avg_overall, 3),
                               avg_context_recall      = round(sum(recall_values) / len(recall_values), 3) if recall_values else None,
                               avg_answer_similarity   = round(sum(similarity_values) / len(similarity_values), 3) if similarity_values else None,
                               avg_answer_correctness  = round(sum(correctness_values) / len(correctness_values), 3) if correctness_values else None,
                               avg_retrieval_time_ms   = round(avg_retrieval, 2),
                               avg_generation_time_ms  = round(avg_generation, 2),
                               avg_total_time_ms       = round(avg_total, 2),
                               min_score               = round(min_score, 3),
                               max_score               = round(max_score, 3),
                               std_dev                 = round(std_dev, 3),
                               session_start           = self.session_start,
                               last_updated            = datetime.now(),
                              )


    def get_evaluation_history(self) -> List[Dict]:
        """
        Get full evaluation history as list of dictionaries
        
        Returns:
        --------
            { list }    : List of evaluation results as dictionaries
        """
        return [result.to_dict() for result in self.evaluation_history]
    
    
    def clear_history(self):
        """
        Clear evaluation history and reset session
        """
        self.evaluation_history.clear()
        self.session_start = datetime.now()

        logger.info("Evaluation history cleared, new session started")
    

    def export_to_dict(self) -> RAGASExportData:
        """
        Export all evaluations to structured format
        
        Returns:
        --------
            { RAGASExportData }    : RAGASExportData object with complete evaluation data
        """
        return RAGASExportData(export_timestamp     = datetime.now().isoformat(),
                               total_evaluations    = len(self.evaluation_history),
                               statistics           = self.get_session_statistics(),
                               evaluations          = self.evaluation_history,
                               ground_truth_enabled = self.enable_ground_truth,
                              )



# Global evaluator instance
_ragas_evaluator : Optional[RAGASEvaluator] = None


def get_ragas_evaluator(enable_ground_truth_metrics: bool = False) -> RAGASEvaluator:
    """
    Get or create global RAGAS evaluator instance
    
    Arguments:
    ----------
        enable_ground_truth_metrics { bool } : Whether to enable ground truth metrics
        
    Returns:
    --------
               { RAGASEvaluator }            : RAGASEvaluator instance
    """
    global _ragas_evaluator
    
    if _ragas_evaluator is None:
        _ragas_evaluator = RAGASEvaluator(enable_ground_truth_metrics)
    
    return _ragas_evaluator