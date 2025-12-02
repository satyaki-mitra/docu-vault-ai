# DEPENDENCIES
from typing import List
from typing import Dict
from typing import Optional
from config.models import PromptType
from config.settings import get_settings
from config.models import ChunkWithScore
from config.logging_config import get_logger
from utils.error_handler import handle_errors
from generation.token_manager import TokenManager
from utils.error_handler import PromptBuildingError


# Setup Settings and Logging
settings = get_settings()
logger   = get_logger(__name__)


class PromptBuilder:
    """
    Intelligent prompt building for LLM generation: Constructs optimized prompts for different task types with proper context management and citation handling
    """
    def __init__(self, model_name: str = None):
        """
        Initialize prompt builder
        
        Arguments:
        ----------
            model_name { str } : Model name for token management
        """
        self.logger           = logger
        self.settings         = get_settings()
        self.model_name       = model_name or self.settings.OLLAMA_MODEL
        self.token_manager    = TokenManager(model_name)
        
        # Prompt templates for different tasks
        self.prompt_templates = {PromptType.QA             : {"system": self._get_qa_system_prompt(), "user": self._get_qa_user_template(), "max_context_ratio": 0.6},
                                 PromptType.SUMMARY        : {"system": self._get_summary_system_prompt(), "user": self._get_summary_user_template(), "max_context_ratio": 0.8,},
                                 PromptType.ANALYTICAL     : {"system": self._get_analytical_system_prompt(), "user": self._get_analytical_user_template(), "max_context_ratio": 0.7},
                                 PromptType.COMPARISON     : {"system": self._get_comparison_system_prompt(), "user": self._get_comparison_user_template(), "max_context_ratio": 0.5},
                                 PromptType.EXTRACTION     : {"system": self._get_extraction_system_prompt(), "user": self._get_extraction_user_template(), "max_context_ratio": 0.6},
                                 PromptType.CREATIVE       : {"system": self._get_creative_system_prompt(), "user": self._get_creative_user_template(), "max_context_ratio": 0.4},
                                 PromptType.CONVERSATIONAL : {"system": self._get_conversational_system_prompt(), "user": self._get_conversational_user_template(), "max_context_ratio": 0.5}
                                }
    

    def build_prompt(self, query: str, context: str, sources: List[ChunkWithScore], prompt_type: PromptType = PromptType.QA,  include_citations: bool = True,
                     max_completion_tokens: int = 1000) -> Dict[str, str]:
        """
        Build complete prompt for LLM generation
        
        Arguments:
        ----------
            query                      { str }   : User query
            
            context                    { str }   : Retrieved context
            
            sources                   { list }   : Source chunks
            
            prompt_type           { PromptType } : Type of prompt to build
            
            include_citations         { bool }   : Whether to include citation instructions
            
            max_completion_tokens     { int }    : Tokens to reserve for completion
        
        Returns:
        --------
                         { dict }                : Dictionary with 'system' and 'user' prompts
        """
        if not query or not context:
            raise PromptBuildingError("Query and context cannot be empty")
        
        try:
            # Get template for prompt type
            template          = self.prompt_templates.get(prompt_type, self.prompt_templates[PromptType.QA])
            
            # Optimize context to fit within token limits
            optimized_context = self._optimize_context(context               = context, 
                                                       template              = template, 
                                                       max_completion_tokens = max_completion_tokens,
                                                      )
            
            # Build system prompt
            system_prompt     = self._build_system_prompt(template          = template, 
                                                          include_citations = include_citations, 
                                                          prompt_type       = prompt_type,
                                                         )
            
            # Build user prompt
            user_prompt       = self._build_user_prompt(template    = template, 
                                                        query       = query, 
                                                        context     = optimized_context, 
                                                        sources     = sources, 
                                                        prompt_type = prompt_type,
                                                       )
            
            # Validate token usage
            self._validate_prompt_length(system_prompt         = system_prompt, 
                                         user_prompt           = user_prompt, 
                                         max_completion_tokens = max_completion_tokens,
                                        )
            
            self.logger.debug(f"Built {prompt_type.value} prompt: {self.token_manager.count_tokens(system_prompt + user_prompt)} tokens")
            
            return {"system"   : system_prompt,
                    "user"     : user_prompt,
                    "metadata" : {"prompt_type"    : prompt_type.value,
                                  "context_tokens" : self.token_manager.count_tokens(optimized_context),
                                  "total_tokens"   : self.token_manager.count_tokens(system_prompt + user_prompt),
                                  "sources_count"  : len(sources),
                                 }
                  }
        
        except Exception as e:
            self.logger.error(f"Prompt building failed: {repr(e)}")
            raise PromptBuildingError(f"Prompt building failed: {repr(e)}")
    

    def _optimize_context(self, context: str, template: Dict, max_completion_tokens: int) -> str:
        """
        Optimize context to fit within token limits
        
        Arguments:
        ----------
            context               { str }  : Context text to optimize

            template              { Dict } : Prompt template
            
            max_completion_tokens { int }  : Tokens to reserve for completion
        
        Returns:
        --------
                        { str }            : Optimized context
        """
        # Calculate tokens for system prompt
        system_tokens          = self.token_manager.count_tokens(template["system"])
        
        # Estimate user template tokens by removing placeholders
        user_template_clean    = template["user"]
        
        # Remove all known placeholders
        placeholders_to_remove = ["{query}", "{context}", "{sources_info}", "{focus}"]
        
        for placeholder in placeholders_to_remove:
            user_template_clean = user_template_clean.replace(placeholder, "")
        
        user_template_tokens = self.token_manager.count_tokens(user_template_clean)
        
        # Calculate available tokens for context - Reserve: system + user template + completion + buffer
        reserved_tokens      = system_tokens + user_template_tokens + max_completion_tokens + 100
        max_context_tokens   = self.token_manager.context_window - reserved_tokens
        
        # Ensure we don't exceed zero
        if (max_context_tokens <= 0):
            self.logger.warning(f"No tokens available for context. Reserved: {reserved_tokens}, Window: {self.token_manager.context_window}")
            return ""
        
        # Apply max context ratio from template config
        ratio_limit        = int(self.token_manager.context_window * template["max_context_ratio"])
        max_context_tokens = min(max_context_tokens, ratio_limit)
        
        # Log optimization details
        self.logger.debug(f"Context optimization: max_tokens={max_context_tokens}, ratio={template['max_context_ratio']}")
        
        # Truncate context to fit
        optimized_context  = self.token_manager.truncate_to_fit(context, max_context_tokens, strategy="end")
        
        # Calculate reduction percentage
        original_tokens    = self.token_manager.count_tokens(context)
        optimized_tokens   = self.token_manager.count_tokens(optimized_context)
        
        if (original_tokens > optimized_tokens):
            reduction_pct = ((original_tokens - optimized_tokens) / original_tokens) * 100
            
            self.logger.info(f"Context reduced by {reduction_pct:.1f}% ({original_tokens} → {optimized_tokens} tokens)")
        
        return optimized_context
    

    def _build_system_prompt(self, template: Dict, include_citations: bool, prompt_type: PromptType) -> str:
        """
        Build system prompt
        """
        system_prompt = template["system"]
        
        # Add citation instructions if needed
        if (include_citations and (prompt_type != PromptType.CREATIVE)):
            citation_instructions = self._get_citation_instructions()
            system_prompt        += "\n\n" + citation_instructions
        
        return system_prompt
    

    def _build_user_prompt(self, template: Dict, query: str, context: str, sources: List[ChunkWithScore], prompt_type: PromptType) -> str:
        """
        Build user prompt
        """
        # Format sources information
        sources_info = self._format_sources_info(sources) if sources else ""
        
        # Build user prompt using template
        user_prompt  = template["user"].format(query        = query,
                                               context      = context,
                                               sources_info = sources_info,
                                             )
        
        # Add task-specific formatting
        if (prompt_type == PromptType.COMPARISON):
            user_prompt = self._enhance_comparison_prompt(user_prompt = user_prompt, 
                                                          sources     = sources,
                                                         )

        elif (prompt_type == PromptType.ANALYTICAL):
            user_prompt = self._enhance_analytical_prompt(user_prompt = user_prompt, 
                                                          sources     = sources,
                                                         )
        
        return user_prompt
    

    def _format_sources_info(self, sources: List[ChunkWithScore]) -> str:
        """
        Format sources information for the prompt
        """
        if not sources:
            return ""
        
        sources_list = list()

        for i, source in enumerate(sources, 1):
            chunk       = source.chunk
            source_info = f"Source [{i}]:"
            
            if hasattr(chunk, 'metadata') and 'filename' in chunk.metadata:
                source_info += f" {chunk.metadata['filename']}"
            
            if chunk.page_number:
                source_info += f" (page {chunk.page_number})"
            
            if chunk.section_title:
                source_info += f" - {chunk.section_title}"
            
            sources_list.append(source_info)
        
        return "Available sources:\n" + "\n".join(sources_list)
    

    def _enhance_comparison_prompt(self, user_prompt: str, sources: List[ChunkWithScore]) -> str:
        """
        Enhance prompt for comparison tasks
        
        Arguments:
        ----------
            user_prompt { str }  : Base user prompt

            sources     { list } : Source chunks
        
        Returns:
        --------
                { str }          : Enhanced prompt
        """
        if (len(sources) < 2):
            return user_prompt
        
        enhancement  = "\n\nPlease compare and contrast the information from different sources. "
        enhancement += "Highlight agreements, disagreements, and complementary information. "
        enhancement += "If sources conflict, present both perspectives clearly."
        
        return user_prompt + enhancement
    

    def _enhance_analytical_prompt(self, user_prompt: str, sources: List[ChunkWithScore]) -> str:
        """
        Enhance prompt for analytical tasks - FIXED SIGNATURE
        
        Arguments:
        ----------
            user_prompt { str }  : Base user prompt

            sources     { list } : Source chunks
        
        Returns:
        --------
                { str }          : Enhanced prompt
        """
        enhancement  = "\n\nProvide analytical insights by:"
        enhancement += "\n1. Identifying patterns and relationships in the information"
        enhancement += "\n2. Analyzing implications and consequences"
        enhancement += "\n3. Evaluating the strength of evidence from different sources"
        enhancement += "\n4. Drawing well-reasoned conclusions"
        
        return user_prompt + enhancement
    

    def _validate_prompt_length(self, system_prompt: str, user_prompt: str, max_completion_tokens: int):
        """
        Validate that prompt fits within context window
        
        Arguments:
        ----------
            system_prompt         { str } : System prompt

            user_prompt           { str } : User prompt
            
            max_completion_tokens { int } : Tokens needed for completion
        
        Raises:
        -------
            PromptBuildingError           : If prompt exceeds context window
        """
        system_tokens  = self.token_manager.count_tokens(system_prompt)
        user_tokens    = self.token_manager.count_tokens(user_prompt)
        total_tokens   = system_tokens + user_tokens
        total_required = total_tokens + max_completion_tokens
        
        if (total_required > self.token_manager.context_window):
            error_msg = (f"Prompt exceeds context window:\n"
                         f"- System prompt: {system_tokens} tokens\n"
                         f"- User prompt: {user_tokens} tokens\n"
                         f"- Completion reserve: {max_completion_tokens} tokens\n"
                         f"- Total required: {total_required} tokens\n"
                         f"- Context window: {self.token_manager.context_window} tokens\n"
                         f"- Overflow: {total_required - self.token_manager.context_window} tokens"
                        )
            self.logger.error(error_msg)

            raise PromptBuildingError(error_msg)
        
        # Log successful validation
        utilization = (total_required / self.token_manager.context_window) * 100

        self.logger.debug(f"Prompt validation passed: {total_required}/{self.token_manager.context_window} tokens ({utilization:.1f}% utilization)")


    def _get_citation_instructions(self) -> str:
        """
        Get citation instructions for system prompt
        """
        return ("CITATION INSTRUCTIONS:\n"
                "1. Always cite your sources using [number] notation\n"
                "2. Use the citation number that corresponds to the source in the context\n"
                "3. Cite sources for all factual claims and specific information\n"
                "4. If information appears in multiple sources, cite the most relevant one\n"
                "5. Do not make up information not present in the provided context\n"
                "6. If the context doesn't contain the answer, explicitly state this"
               )
    

    # Template methods for different prompt types
    def _get_qa_system_prompt(self) -> str:
        return ("You are a precise and helpful AI assistant that answers questions based solely on the provided context.\n\n"
                "Core Principles:\n"
                "1. ONLY use information from the provided context\n"
                "2. Be concise but complete - don't omit important details\n"
                "3. Structure complex answers clearly\n"
                "4. If information is ambiguous or conflicting, acknowledge this\n"
                "5. Never make up or infer information not present in the context"
               )
    

    def _get_qa_user_template(self) -> str:
        return ("Context Information:\n{context}\n\n"
                "{sources_info}\n\n"
                "Question: {query}\n\n"
                "Instructions: Answer the question using ONLY the information provided in the context above. "
                "Cite your sources using [number] notation. If the context doesn't contain enough information "
                "to answer fully, state this clearly.\n\n"
                "Answer:"
               )
    

    def _get_summary_system_prompt(self) -> str:
        return ("You are a thorough AI assistant that provides comprehensive summaries based on the provided context.\n\n"
                "Summary Guidelines:\n"
                "1. Capture all key points and main ideas\n"
                "2. Maintain the original meaning and intent\n"
                "3. Organize information logically\n"
                "4. Be comprehensive but concise\n"
                "5. Highlight important findings and conclusions"
               )
    

    def _get_summary_user_template(self) -> str:
        return ("Content to summarize:\n{context}\n\n"
                "Please provide a comprehensive summary that captures all key points and main ideas. "
                "Organize the summary logically and ensure it reflects the original content accurately.\n\n"
                "Summary:"
               )
    

    def _get_analytical_system_prompt(self) -> str:
        return ("You are an analytical AI assistant that provides insights based on the provided context.\n\n"
                "Analytical Guidelines:\n"
                "1. Analyze patterns and connections in the information\n"
                "2. Compare different perspectives if multiple sources exist\n"
                "3. Highlight key findings and implications\n"
                "4. Identify gaps or limitations in the available information\n"
                "5. Provide well-reasoned analysis and conclusions"
               )

    
    def _get_analytical_user_template(self) -> str:
        return ("Context for analysis:\n{context}\n\n"
                "{sources_info}\n\n"
                "Analytical task: {query}\n\n"
                "Please provide a detailed analysis based on the context above. "
                "Identify patterns, relationships, and implications. "
                "Cite sources for all analytical claims.\n\n"
                "Analysis:"
               )

    
    def _get_comparison_system_prompt(self) -> str:
        return ("You are an AI assistant that compares information across multiple sources.\n\n"
                "Comparison Guidelines:\n"
                "1. Identify similarities and differences between sources\n"
                "2. Note if sources agree or disagree on specific points\n"
                "3. Highlight complementary information\n"
                "4. Present conflicting perspectives fairly\n"
                "5. Draw conclusions about the overall consensus or disagreement"
               )

    
    def _get_comparison_user_template(self) -> str:
        return ("Context from multiple sources:\n{context}\n\n"
                "{sources_info}\n\n"
                "Comparison task: {query}\n\n"
                "Please compare how different sources address this topic. "
                "Identify agreements, disagreements, and complementary information. "
                "Cite specific sources for each point of comparison.\n\n"
                "Comparison:"
               )

    
    def _get_extraction_system_prompt(self) -> str:
        return ("You are a precise AI assistant that extracts specific information from context.\n\n"
                "Extraction Guidelines:\n"
                "1. Extract only the requested information\n"
                "2. Be thorough and complete\n"
                "3. Maintain accuracy and precision\n"
                "4. Organize extracted information clearly\n"
                "5. Cite sources for all extracted information"
               )

    
    def _get_extraction_user_template(self) -> str:
        return ("Context:\n{context}\n\n"
                "{sources_info}\n\n"
                "Extraction task: {query}\n\n"
                "Please extract the requested information from the context above. "
                "Be thorough and precise. Cite sources for all extracted information.\n\n"
                "Extracted Information:"
               )

    
    def _get_creative_system_prompt(self) -> str:
        return ("You are a creative AI assistant that generates content based on provided context.\n\n"
                "Creative Guidelines:\n"
                "1. Use the context as inspiration and foundation\n"
                "2. Be creative and engaging\n"
                "3. Maintain coherence with the source material\n"
                "4. You may extrapolate and build upon the provided information\n"
                "5. Clearly distinguish between source-based content and creative additions"
               )

    
    def _get_creative_user_template(self) -> str:
        return ("Context and inspiration:\n{context}\n\n"
                "Creative task: {query}\n\n"
                "Please create content based on the context above. "
                "You may use the context as inspiration and build upon it creatively. "
                "If you add information beyond what's in the context, make this clear.\n\n"
                "Creative Response:"
               )

    
    def _get_conversational_system_prompt(self) -> str:
        return ("You are a helpful and engaging conversational AI assistant.\n\n"
                "Conversational Guidelines:\n"
                "1. Be natural and engaging in conversation\n"
                "2. Use the provided context to inform your responses\n"
                "3. Maintain a friendly and helpful tone\n"
                "4. Ask clarifying questions when needed\n"
                "5. Cite sources when providing specific information"
               )

    
    def _get_conversational_user_template(self) -> str:
        return ("Context for our conversation:\n{context}\n\n"
                "Current message: {query}\n\n"
                "Please respond naturally and helpfully based on the context above. "
                "If providing specific information, cite your sources.\n\n"
                "Response:"
               )


# Global prompt builder instance
_prompt_builder = None


def get_prompt_builder(model_name: str = None) -> PromptBuilder:
    """
    Get global prompt builder instance (singleton)
    
    Arguments:
    ----------
        model_name { str } : Model name for token management
    
    Returns:
    --------
        { PromptBuilder }  : PromptBuilder instance
    """
    global _prompt_builder
    
    if _prompt_builder is None or (model_name and _prompt_builder.model_name != model_name):
        _prompt_builder = PromptBuilder(model_name)
    
    return _prompt_builder


@handle_errors(error_type = PromptBuildingError, log_error = True, reraise = False)
def build_qa_prompt(query: str, context: str, sources: List[ChunkWithScore], **kwargs) -> Dict[str, str]:
    """
    Convenience function for building QA prompts
    
    Arguments:
    ----------
        query   { str }  : User query
        
        context { str }  : Retrieved context
        
        sources { list } : Source chunks
        
        **kwargs         : Additional prompt building arguments
    
    Returns:
    --------
             { dict }    : Dictionary with system and user prompts
    """
    builder = get_prompt_builder()
    
    return builder.build_prompt(query, context, sources, PromptType.QA, **kwargs)