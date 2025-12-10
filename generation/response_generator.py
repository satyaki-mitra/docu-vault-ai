# DEPENDENCIES
import time
import asyncio
from typing import Dict
from typing import List
from typing import Optional
from datetime import datetime
from typing import AsyncGenerator
from config.models import PromptType
from config.models import LLMProvider
from config.models import QueryRequest
from config.models import QueryResponse
from config.models import ChunkWithScore
from config.settings import get_settings
from config.logging_config import get_logger
from utils.error_handler import handle_errors
from generation.llm_client import get_llm_client
from utils.error_handler import ResponseGenerationError
from generation.prompt_builder import get_prompt_builder
from retrieval.hybrid_retriever import get_hybrid_retriever
from generation.query_classifier import get_query_classifier
from generation.general_responder import get_general_responder
from generation.citation_formatter import get_citation_formatter
from generation.temperature_controller import get_temperature_controller


# Setup Settings and Logging
settings = get_settings()
logger   = get_logger(__name__)


class ResponseGenerator:
    """
    Main orchestrator for RAG response generation with LLM-based intelligent query routing
    
    Handles both:
    1. Generic/conversational queries (greetings, system info, general knowledge)
    2. Document-based RAG queries (retrieval + generation)
    
    Pipeline: Query → LLM Classifier → Route to (General LLM | RAG Pipeline) → Response
    """
    def __init__(self, provider: LLMProvider = None, model_name: str = None):
        """
        Initialize response generator with LLM-based query routing capabilities
        
        Arguments:
        ----------
            provider   { LLMProvider } : LLM provider (Ollama/OpenAI)

            model_name { str }         : Model name to use
        """
        self.logger                 = logger
        self.settings               = get_settings()
        self.provider               = provider or LLMProvider.OLLAMA
        self.model_name             = model_name or self.settings.OLLAMA_MODEL
        
        # Initialize components
        self.llm_client             = get_llm_client(provider   = self.provider, 
                                                     model_name = self.model_name,
                                                    )
        
        # Query routing components
        self.query_classifier       = get_query_classifier(provider   = self.provider,
                                                           model_name = self.model_name,
                                                          )
        
        self.general_responder      = get_general_responder(provider   = self.provider,
                                                            model_name = self.model_name,
                                                           )
        
        # RAG components
        self.hybrid_retriever       = get_hybrid_retriever()
        self.prompt_builder         = get_prompt_builder(model_name = self.model_name)
        self.citation_formatter     = get_citation_formatter()
        self.temperature_controller = get_temperature_controller()
        
        # Statistics
        self.generation_count       = 0
        self.total_generation_time  = 0.0
        self.general_query_count    = 0
        self.rag_query_count        = 0
        
        self.logger.info(f"Initialized ResponseGenerator with LLM-Based Query Routing: provider={self.provider.value}, model={self.model_name}")
    

    @handle_errors(error_type = ResponseGenerationError, log_error = True, reraise = True)
    async def generate_response(self, request: QueryRequest, conversation_history: List[Dict] = None, has_documents: bool = True) -> QueryResponse:
        """
        Generate response with LLM-based intelligent query routing
        
        Arguments:
        ----------
            request              { QueryRequest } : Query request object

            conversation_history { list }        : Previous conversation messages
            
            has_documents        { bool }        : Whether documents are available in the system
        
        Returns:
        --------
            { QueryResponse }                    : Complete query response
        """
        start_time = time.time()
        
        self.logger.info(f"Processing query: '{request.query[:100]}...'")
        
        try:
            # Classify query using LLM
            classification = await self.query_classifier.classify(query         = request.query,
                                                                  has_documents = has_documents,
                                                                 )
            
            self.logger.info(f"Query classified as: {classification['type']} (confidence: {classification['confidence']:.2f}, LLM-based: {classification.get('is_llm_classified', False)})")
            self.logger.debug(f"Classification reason: {classification['reason']}")
            
            # Route based on classification
            if (classification['suggested_action'] == 'respond_with_general_llm'):
                # Handle as general query
                response                  = await self._handle_general_query(request        = request,
                                                                             classification = classification,
                                                                             start_time     = start_time,
                                                                             history        = conversation_history,
                                                                            )
                                    
                self.general_query_count += 1
                
                return response
            
            elif (classification['suggested_action'] == 'respond_with_rag'):
                # Handle as RAG query
                response              = await self._handle_rag_query(request        = request,
                                                                     classification = classification,
                                                                     start_time     = start_time,
                                                                    )
                
                self.rag_query_count += 1
                
                return response
            
            else:
                # Default to RAG if unclear
                self.logger.info("Unclear classification - defaulting to RAG...")
                
                try:
                    response = await self._handle_rag_query(request        = request,
                                                            classification = classification,
                                                            start_time     = start_time,
                                                            allow_fallback = True,
                                                           )
                    
                    # If no results from RAG, fall back to general
                    if ((not response.sources) or (len(response.sources) == 0)):
                        self.logger.info("No RAG results - falling back to general response")

                        response                  = await self._handle_general_query(request        = request,
                                                                                     classification = classification,
                                                                                     start_time     = start_time,
                                                                                     history        = conversation_history,
                                                                                    )
                        self.general_query_count += 1
                    
                    else:
                        self.rag_query_count     += 1
                    
                    return response
                    
                except Exception as e:
                    self.logger.warning(f"RAG attempt failed, falling back to general: {e}")
                    response                  = await self._handle_general_query(request        = request,
                                                                                 classification = classification,
                                                                                 start_time     = start_time,
                                                                                 history        = conversation_history,
                                                                                )
                    self.general_query_count += 1
                    
                    return response
        
        except Exception as e:
            self.logger.error(f"Response generation failed: {repr(e)}", exc_info = True)
            raise ResponseGenerationError(f"Response generation failed: {repr(e)}")
    

    async def _handle_general_query(self, request: QueryRequest, classification: Dict, start_time: float, history: List[Dict] = None) -> QueryResponse:
        """
        Handle general/conversational queries without RAG
        
        Arguments:
        ----------
            request        { QueryRequest } : Original request

            classification { dict }         : Classification result
            
            start_time     { float }        : Start timestamp
        
            history        { list }         : Conversation history
        
        Returns:
        --------
            { QueryResponse }               : Response without RAG
        """
        self.logger.debug("Handling as general query...")
        
        # Use general responder
        general_response = await self.general_responder.respond(query                = request.query,
                                                                conversation_history = history,
                                                               )
        
        answer           = general_response.get("answer", "I'm here to help! Please let me know how I can assist you.")
        total_time       = (time.time() - start_time) * 1000
        
        # Create QueryResponse object
        response         = QueryResponse(query              = request.query,
                                         answer             = answer,
                                         sources            = [],  # No sources for general queries
                                         retrieval_time_ms  = 0.0,
                                         generation_time_ms = total_time,
                                         total_time_ms      = total_time,
                                         tokens_used        = general_response.get("tokens_used", {"input": 0, "output": 0, "total": 0}),
                                         model_used         = self.model_name,
                                         timestamp          = datetime.now(),
                                        )
        
        # Add metadata about query type
        if request.include_metrics:
            response.metrics = {"query_type"        : "general",
                                "classification"    : classification['type'],
                                "confidence"        : classification['confidence'],
                                "requires_rag"      : False,
                                "conversation_mode" : True,
                                "llm_classified"    : classification.get('is_llm_classified', False),
                               }
        
        self.logger.info(f"General response generated in {total_time:.0f}ms")
        
        return response
    

    async def _handle_rag_query(self, request: QueryRequest, classification: Dict, start_time: float, allow_fallback: bool = False) -> QueryResponse:
        """
        Handle RAG-based queries with document retrieval
        
        Arguments:
        ----------
            request        { QueryRequest } : Original request

            classification { dict }         : Classification result
            
            start_time     { float }        : Start timestamp
            
            allow_fallback { bool }         : Whether to allow fallback to general
        
        Returns:
        --------
            { QueryResponse }               : RAG response
        """
        self.logger.debug("Handling as RAG query...")
        
        try:
            # Retrieve relevant context
            self.logger.debug("Retrieving context...")
            retrieval_start  = time.time()
            
            retrieval_result = self.hybrid_retriever.retrieve_with_context(query             = request.query,
                                                                           top_k             = request.top_k or self.settings.TOP_K_RETRIEVE,
                                                                           enable_reranking  = request.enable_reranking,
                                                                           include_citations = request.include_sources,
                                                                          )
            
            retrieval_time   = (time.time() - retrieval_start) * 1000
            
            chunks           = retrieval_result["chunks"]
            context          = retrieval_result["context"]
            
            if not chunks:
                self.logger.warning("No relevant context found for query")
                
                if allow_fallback:
                    # Return empty response to trigger fallback
                    return QueryResponse(query              = request.query,
                                         answer             = "",
                                         sources            = [],
                                         retrieval_time_ms  = retrieval_time,
                                         generation_time_ms = 0.0,
                                         total_time_ms      = retrieval_time,
                                         tokens_used        = {"input": 0, "output": 0, "total": 0},
                                         model_used         = self.model_name,
                                         timestamp          = datetime.now(),
                                        )

                else:
                    return self._create_no_results_response(request           = request, 
                                                            retrieval_time_ms = retrieval_time,
                                                           )
            
            self.logger.info(f"Retrieved {len(chunks)} chunks in {retrieval_time:.0f}ms")
            
            # Determine prompt type and temperature
            self.logger.debug("Determining prompt strategy...")

            prompt_type = self._infer_prompt_type(query = request.query)
            
            temperature = self._get_adaptive_temperature(request          = request,
                                                         query            = request.query,
                                                         context          = context,
                                                         retrieval_scores = [chunk.score for chunk in chunks],
                                                         prompt_type      = prompt_type,
                                                        )
            
            self.logger.debug(f"Prompt type: {prompt_type.value}, Temperature: {temperature}")
            
            # Build optimized prompt
            self.logger.debug("Building prompt...")

            prompt      = self.prompt_builder.build_prompt(query                 = request.query,
                                                           context               = context,
                                                           sources               = chunks,
                                                           prompt_type           = prompt_type,
                                                           include_citations     = request.include_sources,
                                                           max_completion_tokens = request.max_tokens or self.settings.MAX_TOKENS,
                                                          )
            
            # Generate LLM response
            self.logger.debug("Generating LLM response...")
            generation_start = time.time()
            
            messages         = [{"role"    : "system", 
                                 "content" : prompt["system"]
                                },
                                {"role"    : "user", 
                                 "content" : prompt["user"],
                                }
                               ]
            
            llm_response     = await self.llm_client.generate(messages    = messages,
                                                              temperature = temperature,
                                                              top_p       = request.top_p or self.settings.TOP_P,
                                                              max_tokens  = request.max_tokens or self.settings.MAX_TOKENS,
                                                             )
            
            generation_time  = (time.time() - generation_start) * 1000
            
            answer           = llm_response["content"]
            
            self.logger.info(f"Generated response in {generation_time:.0f}ms ({llm_response['usage']['completion_tokens']} tokens)")
            
            # Format citations (if enabled)
            if request.include_sources:
                self.logger.debug("Formatting citations...")
                answer = self._post_process_citations(answer  = answer, 
                                                      sources = chunks,
                                                     )
            
            # Create response object
            total_time = (time.time() - start_time) * 1000
            
            response   = QueryResponse(query              = request.query,
                                       answer             = answer,
                                       sources            = chunks if request.include_sources else [],
                                       retrieval_time_ms  = retrieval_time,
                                       generation_time_ms = generation_time,
                                       total_time_ms      = total_time,
                                       tokens_used        = {"input"  : llm_response["usage"]["prompt_tokens"],
                                                             "output" : llm_response["usage"]["completion_tokens"],
                                                             "total"  : llm_response["usage"]["total_tokens"],
                                                            },
                                       model_used         = self.model_name,
                                       timestamp          = datetime.now(),
                                      )
            
            # Add quality metrics if requested
            if request.include_metrics:
                response.metrics                           = self._calculate_quality_metrics(query    = request.query,
                                                                                             answer   = answer,
                                                                                             context  = context,
                                                                                             sources  = chunks,
                                                                                            )
                # Track both: prediction & reality 
                response.metrics["predicted_type"]         = classification.get('type', 'unknown')
                response.metrics["predicted_confidence"]   = classification.get('confidence', 0.0)
                response.metrics["actual_type"]            = "rag"  # Always rag if we're here
                response.metrics["execution_path"]         = "rag_pipeline"
                response.metrics["has_context"]            = len(chunks) > 0
                response.metrics["context_chunks"]         = len(chunks)
                response.metrics["rag_confidence"]         = min(1.0, sum(c.score for c in chunks) / len(chunks) if chunks else 0.0)
                response.metrics["is_forced_rag"]          = classification.get('is_forced_rag', False)
                response.metrics["llm_classified"]         = classification.get('is_llm_classified', False)

                # Add context for evaluation
                response.metrics["context_for_evaluation"] = context
            
            # Update statistics
            self.generation_count      += 1
            self.total_generation_time += total_time
            
            self.logger.info(f"RAG response generated successfully in {total_time:.0f}ms")
            
            return response
            
        except Exception as e:
            self.logger.error(f"RAG query handling failed: {repr(e)}", exc_info = True)
            
            if allow_fallback:
                # Return empty to trigger fallback
                return QueryResponse(query              = request.query,
                                     answer             = "",
                                     sources            = [],
                                     retrieval_time_ms  = 0.0,
                                     generation_time_ms = 0.0,
                                     total_time_ms      = 0.0,
                                     tokens_used        = {"input": 0, "output": 0, "total": 0},
                                     model_used         = self.model_name,
                                     timestamp          = datetime.now(),
                                    )
            else:
                raise
    

    @handle_errors(error_type = ResponseGenerationError, log_error = True, reraise = True)
    async def generate_response_stream(self, request: QueryRequest, has_documents: bool = True) -> AsyncGenerator[str, None]:
        """
        Generate streaming RAG response
        
        Arguments:
        ----------
            request       { QueryRequest } : Query request object
            
            has_documents { bool }         : Whether documents are available
        
        Yields:
        -------
            { str }                        : Response chunks (tokens)
        """
        self.logger.info(f"Generating streaming response for query: '{request.query[:100]}...'")
        
        try:
            # Classify query first
            classification = await self.query_classifier.classify(query         = request.query,
                                                                  has_documents = has_documents,
                                                                 )
            
            if (classification['suggested_action'] == 'respond_with_general_llm'):
                # Stream general response
                general_response = await self.general_responder.respond(query = request.query)
                yield general_response.get("answer", "")
                
                return
            
            # Otherwise proceed with RAG streaming - Procced with Retrieving context
            retrieval_result = self.hybrid_retriever.retrieve_with_context(query             = request.query,
                                                                           top_k             = request.top_k or self.settings.TOP_K_RETRIEVE,
                                                                           enable_reranking  = request.enable_reranking,
                                                                           include_citations = request.include_sources,
                                                                          )
            
            chunks           = retrieval_result["chunks"]
            context          = retrieval_result["context"]
            
            if not chunks:
                yield "I couldn't find relevant information to answer your question."
                return
            
            # Determine strategy
            prompt_type = self._infer_prompt_type(query = request.query)
            temperature = self._get_adaptive_temperature(request          = request,
                                                         query            = request.query,
                                                         context          = context,
                                                         retrieval_scores = [chunk.score for chunk in chunks],
                                                         prompt_type      = prompt_type,
                                                        )
            
            # Build prompt
            prompt      = self.prompt_builder.build_prompt(query                 = request.query,
                                                           context               = context,
                                                           sources               = chunks,
                                                           prompt_type           = prompt_type,
                                                           include_citations     = request.include_sources,
                                                           max_completion_tokens = request.max_tokens or self.settings.MAX_TOKENS,
                                                          )
            
            # Stream LLM response
            messages    = [{"role"    : "system", 
                            "content" : prompt["system"],
                           },
                           {"role"    : "user", 
                            "content" : prompt["user"],
                           },
                          ]
            
            async for chunk_text in self.llm_client.generate_stream(messages    = messages,
                                                                    temperature = temperature,
                                                                    top_p       = request.top_p or self.settings.TOP_P,
                                                                    max_tokens  = request.max_tokens or self.settings.MAX_TOKENS,
                                                                   ):
                yield chunk_text
            
            self.logger.info("Streaming response completed")
            
        except Exception as e:
            self.logger.error(f"Streaming generation failed: {repr(e)}", exc_info = True)
            
            yield f"\n\n[Error: {str(e)}]"
    

    def _infer_prompt_type(self, query: str) -> PromptType:
        """
        Infer appropriate prompt type from query
        
        Arguments:
        ----------
            query { str }  : User query
        
        Returns:
        --------
            { PromptType } : Inferred prompt type
        """
        query_lower = query.lower()
        
        # Summary indicators
        if (any(word in query_lower for word in ['summarize', 'summary', 'overview', 'tldr', 'brief'])):
            return PromptType.SUMMARY
        
        # Comparison indicators
        if (any(word in query_lower for word in ['compare', 'contrast', 'difference', 'versus', 'vs'])):
            return PromptType.COMPARISON
        
        # Analytical indicators
        if (any(word in query_lower for word in ['analyze', 'analysis', 'evaluate', 'assess', 'examine'])):
            return PromptType.ANALYTICAL
        
        # Extraction indicators
        if (any(word in query_lower for word in ['extract', 'list', 'find all', 'identify', 'enumerate'])):
            return PromptType.EXTRACTION
        
        # Creative indicators
        if (any(word in query_lower for word in ['create', 'write', 'compose', 'generate', 'imagine'])):
            return PromptType.CREATIVE
        
        # Conversational indicators
        if (any(word in query_lower for word in ['tell me about', 'explain', 'discuss', 'talk about'])):
            return PromptType.CONVERSATIONAL
        
        # Default to QA
        return PromptType.QA
    

    def _get_adaptive_temperature(self, request: QueryRequest, query: str, context: str, retrieval_scores: List[float], prompt_type: PromptType) -> float:
        """
        Get adaptive temperature based on query characteristics
        
        Arguments:
        ----------
            request          { QueryRequest } : Original request

            query            { str }          : User query
            
            context          { str }          : Retrieved context
            
            retrieval_scores { list }         : Retrieval scores
            
            prompt_type      { PromptType }   : Inferred prompt type
        
        Returns:
        --------
                  { float }                   : Temperature value
        """
        # Use request temperature if explicitly provided
        if (request.temperature is not None):
            self.logger.debug(f"Using request temperature: {request.temperature}")
            
            return request.temperature
        
        # Otherwise, use adaptive temperature controller
        temperature = self.temperature_controller.get_temperature(query            = query,
                                                                  context          = context,
                                                                  retrieval_scores = retrieval_scores,
                                                                  query_type       = prompt_type.value,
                                                                 )
        
        return temperature
    

    def _post_process_citations(self, answer: str, sources: List[ChunkWithScore]) -> str:
        """
        Post-process answer to format citations
        
        Arguments:
        ----------
            answer  { str }  : Generated answer with citation markers

            sources { list } : Source chunks
        
        Returns:
        --------
              { str }        : Answer with formatted citations
        """
        try:
            # Validate citations
            is_valid, invalid = self.citation_formatter.validate_citations(answer, sources)
            
            if not is_valid:
                self.logger.warning(f"Invalid citations found: {invalid}")
                # Normalize to fix issues
                answer = self.citation_formatter.normalize_citations(answer, sources)
            
            # Format citations according to style
            formatted_answer = self.citation_formatter.format_citations_in_text(answer, sources)
            
            return formatted_answer
            
        except Exception as e:
            self.logger.error(f"Citation post-processing failed: {repr(e)}")
            # Return original answer if formatting fails
            return answer
    

    def _create_no_results_response(self, request: QueryRequest, retrieval_time_ms: float) -> QueryResponse:
        """
        Create response when no results are found
        
        Arguments:
        ----------
            request           { QueryRequest } : Original request

            retrieval_time_ms { float }        : Time spent on retrieval
        
        Returns:
        --------
            { QueryResponse }                  : Response indicating no results
        """
        no_results_answer = ("I couldn't find relevant information in the available documents to answer your question. "
                             "This could mean:\n"
                             "1. The information is not present in the indexed documents\n"
                             "2. The question may need to be rephrased for better matching\n"
                             "3. The relevant documents haven't been uploaded yet\n\n"
                             "Please try:\n"
                             "- Rephrasing your question with different keywords\n"
                             "- Asking a more specific or general question\n"
                             "- Ensuring the relevant documents are uploaded\n"
                            )
        
        return QueryResponse(query              = request.query,
                             answer             = no_results_answer,
                             sources            = [],
                             retrieval_time_ms  = retrieval_time_ms,
                             generation_time_ms = 0.0,
                             total_time_ms      = retrieval_time_ms,
                             tokens_used        = {"input": 0, "output": 0, "total": 0},
                             model_used         = self.model_name,
                             timestamp          = datetime.now(),
                            )
    

    def _calculate_quality_metrics(self, query: str, answer: str, context: str, sources: List[ChunkWithScore]) -> Dict[str, float]:
        """
        Calculate quality metrics for the response
        
        Arguments:
        ----------
            query   { str }  : User query

            answer  { str }  : Generated answer
            
            context { str }  : Retrieved context
            
            sources { list } : Source chunks
        
        Returns:
        --------
             { dict }        : Quality metrics
        """
        metrics = dict()
        
        try:
            # Answer length metrics
            metrics["answer_length"]       = len(answer.split())
            metrics["answer_char_length"]  = len(answer)
            
            # Citation metrics
            citation_stats                 = self.citation_formatter.get_citation_statistics(answer, sources)
            metrics["citations_used"]      = citation_stats.get("total_citations", 0)
            metrics["unique_citations"]    = citation_stats.get("unique_citations", 0)
            metrics["citation_density"]    = citation_stats.get("citation_density", 0.0)
            
            # Context utilization
            context_length                 = len(context.split())
            metrics["context_utilization"] = min(1.0, metrics["answer_length"] / max(1, context_length))
            
            # Retrieval quality
            if sources:
                avg_score                      = sum(s.score for s in sources) / len(sources)
                metrics["avg_retrieval_score"] = avg_score
                metrics["top_retrieval_score"] = sources[0].score if sources else 0.0
            
            # Query-answer alignment (simple keyword overlap)
            query_words                     = set(query.lower().split())
            answer_words                    = set(answer.lower().split())
            overlap                         = len(query_words & answer_words)
            metrics["query_answer_overlap"] = overlap / max(1, len(query_words))
            
        except Exception as e:
            self.logger.warning(f"Failed to calculate some quality metrics: {repr(e)}")
        
        return metrics
    

    async def generate_batch_responses(self, requests: List[QueryRequest], has_documents: bool = True) -> List[QueryResponse]:
        """
        Generate responses for multiple queries in batch
        
        Arguments:
        ----------
            requests      { list } : List of query requests
            
            has_documents { bool } : Whether documents are available
        
        Returns:
        --------
                  { list }         : List of query responses
        """
        self.logger.info(f"Generating batch responses for {len(requests)} queries")
        
        tasks     = [self.generate_response(request       = request,
                                            has_documents = has_documents) for request in requests]
        
        responses = await asyncio.gather(*tasks, return_exceptions = True)
        
        # Handle exceptions
        results   = list()

        for i, response in enumerate(responses):
            if isinstance(response, Exception):
                self.logger.error(f"Batch query {i} failed: {repr(response)}")
                # Create error response
                error_response = self._create_error_response(requests[i], str(response))
                results.append(error_response)
            
            else:
                results.append(response)
        
        self.logger.info(f"Completed batch generation: {len(results)} responses")
        
        return results
    

    def _create_error_response(self, request: QueryRequest, error_message: str) -> QueryResponse:
        """
        Create error response for failed generation
        
        Arguments:
        ----------
            request       { QueryRequest } : Original request

            error_message { str }          : Error message
        
        Returns:
        --------
            { QueryResponse }              : Error response
        """
        return QueryResponse(query              = request.query,
                             answer             = f"An error occurred while generating the response: {error_message}",
                             sources            = [],
                             retrieval_time_ms  = 0.0,
                             generation_time_ms = 0.0,
                             total_time_ms      = 0.0,
                             tokens_used        = {"input": 0, "output": 0, "total": 0},
                             model_used         = self.model_name,
                             timestamp          = datetime.now(),
                            )
    

    def get_generation_stats(self) -> Dict:
        """
        Get generation statistics including query type breakdown
        
        Returns:
        --------
            { dict }    : Generation statistics
        """
        avg_time = (self.total_generation_time / self.generation_count) if self.generation_count > 0 else 0
        
        return {"total_generations"      : self.generation_count,
                "general_queries"        : self.general_query_count,
                "rag_queries"            : self.rag_query_count,
                "total_generation_time"  : self.total_generation_time,
                "avg_generation_time_ms" : avg_time,
                "provider"               : self.provider.value,
                "model"                  : self.model_name,
                "llm_health"             : self.llm_client.check_health(),
                "query_routing_enabled"  : True,
                "llm_based_routing"      : True,
               }
    

    def reset_stats(self):
        """
        Reset generation statistics
        """
        self.generation_count      = 0
        self.general_query_count   = 0
        self.rag_query_count       = 0
        self.total_generation_time = 0.0
        
        self.logger.info("Generation statistics reset")


# Global response generator instance
_response_generator = None


def get_response_generator(provider: LLMProvider = None, model_name: str = None) -> ResponseGenerator:
    """
    Get global response generator instance (singleton)
    
    Arguments:
    ----------
        provider   { LLMProvider } : LLM provider

        model_name { str }         : Model name
    
    Returns:
    --------
        { ResponseGenerator }      : ResponseGenerator instance
    """
    global _response_generator
    
    if _response_generator is None or (provider and _response_generator.provider != provider):
        _response_generator = ResponseGenerator(provider, model_name)
    
    return _response_generator


@handle_errors(error_type = ResponseGenerationError, log_error = True, reraise = False)
async def generate_answer(query: str, top_k: int = 5, temperature: float = None, has_documents: bool = True, **kwargs) -> str:
    """
    Convenience function for quick answer generation
    
    Arguments:
    ----------
        query         { str }   : User query

        top_k         { int }   : Number of chunks to retrieve
        
        temperature   { float } : Temperature for generation
        
        has_documents { bool }  : Whether documents are available
        
        **kwargs                : Additional parameters
    
    Returns:
    --------
             { str }            : Generated answer
    """
    request   = QueryRequest(query       = query,
                             top_k       = top_k,
                             temperature = temperature,
                             **kwargs
                            )
    
    generator = get_response_generator()
    response  = await generator.generate_response(request       = request,
                                                  has_documents = has_documents,
                                                 )
    
    return response.answer