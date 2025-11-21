# DEPENDENCIES
from typing import Any
from typing import List
from typing import Dict
from config.models import ChunkWithScore


class PromptTemplates:
    """
    Collection of prompt templates for different tasks
    """
    # System Prompts
    SYSTEM_BASE       = """
                           You are a precise and helpful AI assistant that answers questions based solely on the provided context.

                           Core Principles:
                           1. ONLY use information from the provided context
                           2. If the context doesn't contain the answer, explicitly say so
                           3. Cite sources using the format [Source N] where N is the source number
                           4. Be concise but complete - don't omit important details
                           5. If information is ambiguous or conflicting, acknowledge this
                           6. Never make up or infer information not present in the context

                           Response Format:
                           - Directly answer the question
                           - Support claims with source citations [Source N]
                           - Use clear, professional language
                           - Structure complex answers with bullet points when appropriate
                        """


    SYSTEM_CONCISE    = """
                            You are a precise AI assistant that provides brief, accurate answers based on the provided context.

                            Rules:
                            - Answer in 2-3 sentences maximum
                            - Use only information from the context
                            - Cite sources as [Source N]
                            - Say "Not found in context" if information is unavailable
                        """


    SYSTEM_DETAILED   = """
                            You are a thorough AI assistant that provides comprehensive answers based on the provided context.

                            Rules:
                            - Provide detailed, well-structured answers
                            - Include all relevant information from the context
                            - Cite sources extensively using [Source N]
                            - Organize information logically with sections if needed
                            - Explain technical terms when appropriate
                            - Compare and contrast information from different sources if relevant
                        """


    SYSTEM_ANALYTICAL = """
                            You are an analytical AI assistant that provides insights based on the provided context.

                            Rules:
                            - Analyze patterns and connections in the information
                            - Compare different perspectives if multiple sources exist
                            - Highlight key findings and implications
                            - Identify gaps or limitations in the available information
                            - Cite sources rigorously using [Source N]
                            - Structure analysis logically
                        """

    # User Prompt Templates
    
    @staticmethod
    def build_qa_prompt(query: str, context_chunks: List[ChunkWithScore], style: str = "base") -> str:
        """
        Build complete question-answering prompt.
        
        Arguments:
        ----------
            query           { str } : User's question
            
            context_chunks { list } : Retrieved chunks with scores
            
            style           { str } : Prompt style (base, concise, detailed, analytical)
        
        Returns:
        --------
                  { str }           : Formatted prompt string
        """
        
        # Format context with citations
        context_parts = list()

        for i, chunk_with_score in enumerate(context_chunks, 1):
            chunk = chunk_with_score.chunk
            
            # Build source info
            source_info = f"[Source {i}]"
            
            if chunk.page_number:
                source_info += f" (Page {chunk.page_number})"
            
            if chunk.section_title:
                source_info += f" - {chunk.section_title}"
            
            context_parts.append(f"{source_info}\n{chunk.text}")
        
        context_text = "\n\n".join(context_parts)
        
        # Build full prompt
        prompt       = f"""
                            Context Information:
                            {context_text}

                            Question: {query}

                            Instructions: Answer the question using ONLY the information provided in the context above. Cite your sources using [Source N] notation. If the context doesn't contain enough information to answer fully, state this clearly.

                            Answer:
                        """
        
        return prompt
    

    @staticmethod
    def build_comparison_prompt(query: str, context_chunks: List[ChunkWithScore]) -> str:
        """
        Build prompt for comparing information across sources
        """
        context_parts = list()

        for i, chunk_with_score in enumerate(context_chunks, 1):
            chunk = chunk_with_score.chunk
            context_parts.append(f"[Source {i}] {chunk.text}")
        
        context_text      = "\n\n".join(context_parts)

        comparison_prompt = f"""
                                 You are comparing information across multiple sources.

                                 Context from multiple sources:
                                 {context_text}

                                 Question: {query}

                                 Instructions:
                                 1. Compare how different sources address the question
                                 2. Identify similarities and differences
                                 3. Note if sources agree or disagree
                                 4. Cite each source as [Source N]
                                 5. If sources conflict, present both perspectives

                                 Comparison:
                             """

        return comparison_prompt

    
    @staticmethod
    def build_summary_prompt(context_chunks: List[ChunkWithScore], focus: str = "general") -> str:
        """
        Build prompt for summarizing documents
        """
        context_parts = list()

        for chunk_with_score in context_chunks:
            context_parts.append(chunk_with_score.chunk.text)
        
        context_text       = "\n\n".join(context_parts)
        
        focus_instructions = {"general"       : "Provide a comprehensive summary of the key points.",
                              "key_points"    : "Extract and list the main key points only.",
                              "technical"     : "Focus on technical details and specifications.",
                              "risks"         : "Focus on risks, challenges, and potential issues.",
                              "opportunities" : "Focus on opportunities and positive aspects.",
                             }
        
        instruction        = focus_instructions.get(focus, focus_instructions["general"])
        
        summary_prompt     = f"""
                                  Content to summarize:
                                  {context_text}

                                  Task: {instruction}

                                  Summary:
                              """
        
        return summary_prompt

    
    @staticmethod
    def build_extraction_prompt(context_chunks: List[ChunkWithScore], extract_type: str) -> str:
        """
        Build prompt for extracting specific information
        """
        context_parts = list()

        for i, chunk_with_score in enumerate(context_chunks, 1):
            chunk = chunk_with_score.chunk
            context_parts.append(f"[Source {i}] {chunk.text}")
        
        context_text         = "\n\n".join(context_parts)
        
        extract_instructions = {"dates"        : "Extract all dates and their associated events. Format: Date - Event",
                                "names"        : "Extract all person names and their roles/titles.",
                                "numbers"      : "Extract all numerical data with context (e.g., '$5M revenue', '20% growth').",
                                "locations"    : "Extract all locations mentioned and their context.",
                                "action_items" : "Extract all action items, tasks, or to-dos mentioned.",
                                "definitions"  : "Extract all term definitions or explanations.",
                               }
        
        instruction          = extract_instructions.get(extract_type, f"Extract all {extract_type}")

        extraction_prompt    = f"""
                                    Context:
                                    {context_text}

                                    Task: {instruction}
                                    Cite sources using [Source N] notation.

                                    Extracted Information:
                                """
        
        return extraction_prompt

    
    # Few-Shot Examples
    FEW_SHOT_EXAMPLES = """
                            Example 1:
                            Context: [Source 1] The company reported Q3 revenue of $5.2M, up 23% year-over-year.
                            Question: What was the Q3 revenue?
                            Answer: The Q3 revenue was $5.2M, representing a 23% increase compared to the previous year [Source 1].

                            Example 2:
                            Context: [Source 1] The contract includes a 60-day payment term. [Source 2] Standard terms are NET 30.
                            Question: What are the payment terms?
                            Answer: The contract specifies 60-day payment terms [Source 1], which differs from the standard NET 30 terms mentioned elsewhere [Source 2].

                            Example 3:
                            Context: [Source 1] The product supports Windows and macOS.
                            Question: Does it support Linux?
                            Answer: The context does not mention Linux support. Only Windows and macOS are listed as supported platforms [Source 1].
                        """
    
    @staticmethod
    def build_few_shot_prompt(query: str, context_chunks: List[ChunkWithScore]) -> str:
        """
        Build prompt with few-shot examples
        """
        context_parts = list()

        for i, chunk_with_score in enumerate(context_chunks, 1):
            chunk = chunk_with_score.chunk
            context_parts.append(f"[Source {i}] {chunk.text}")
        
        context_text    = "\n\n".join(context_parts)
        
        few_shot_prompt = f"""
                              {PromptTemplates.FEW_SHOT_EXAMPLES}

                              Now answer this question:

                              Context:
                              {context_text}

                              Question: {query}

                              Answer:
                           """

        return few_shot_prompt
    

    # Specialized Prompts
    @staticmethod
    def build_legal_analysis_prompt(query: str, context_chunks: List[ChunkWithScore]) -> str:
        """
        Build prompt for legal document analysis
        """
        context_parts = list()

        for i, chunk_with_score in enumerate(context_chunks, 1):
            chunk      = chunk_with_score.chunk
            source_ref = f"[Source {i}]"

            if chunk.page_number:
                source_ref += f" (Page {chunk.page_number})"

            context_parts.append(f"{source_ref}\n{chunk.text}")
        
        context_text          = "\n\n".join(context_parts)

        legal_analysis_prompt = f"""
                                    You are analyzing legal documents. Be precise and cite sources meticulously.

                                    Legal Context:
                                    {context_text}

                                    Question: {query}

                                    Instructions:
                                    1. Identify relevant clauses and terms
                                    2. Note any unusual or non-standard provisions
                                    3. Highlight potential risks or concerns
                                    4. Cite specific sources and page numbers
                                    5. Use clear, professional legal language

                                    Analysis:
                                 """

        return legal_analysis_prompt

    
    @staticmethod
    def build_technical_doc_prompt(query: str, context_chunks: List[ChunkWithScore]) -> str:
        """
        Build prompt for technical documentation queries
        """
        context_parts = list()

        for i, chunk_with_score in enumerate(context_chunks, 1):
            chunk = chunk_with_score.chunk
            context_parts.append(f"[Source {i}] {chunk.text}")
        
        context_text         = "\n\n".join(context_parts)
        
        technical_doc_prompt = f"""
                                    You are answering a technical question based on documentation.

                                    Technical Documentation:
                                    {context_text}

                                    Question: {query}

                                    Instructions:
                                    1. Provide accurate technical details
                                    2. Include code examples if present
                                    3. Note version-specific information if mentioned
                                    4. Cite sources for all technical claims
                                    5. If information is incomplete, state what's missing

                                    Technical Answer:
                                """

        return technical_doc_prompt
    

    # Utility Methods
    @staticmethod
    def get_system_prompt(style: str = "base") -> str:
        """
        Get system prompt by style
        """
        prompts = {"base"       : PromptTemplates.SYSTEM_BASE,
                   "concise"    : PromptTemplates.SYSTEM_CONCISE,
                   "detailed"   : PromptTemplates.SYSTEM_DETAILED,
                   "analytical" : PromptTemplates.SYSTEM_ANALYTICAL,
                  }

        return prompts.get(style, PromptTemplates.SYSTEM_BASE)

    
    @staticmethod
    def format_sources_list(context_chunks: List[ChunkWithScore]) -> str:
        """
        Format list of sources for reference
        """
        sources = list()

        for i, chunk_with_score in enumerate(context_chunks, 1):
            chunk      = chunk_with_score.chunk
            
            source_str = f"[{i}] Document: {chunk.document_id}"
            
            if chunk.page_number:
                source_str += f", Page: {chunk.page_number}"
            
            if chunk.section_title:
                source_str += f", Section: {chunk.section_title}"
            
            sources.append(source_str)
        
        return "\n".join(sources)
    

    @staticmethod
    def validate_prompt_length(prompt: str, max_tokens: int = 8000) -> bool:
        """
        Validate prompt doesn't exceed token limit: 
        
        - Rough estimate: 1 token ≈ 4 characters
        """
        estimated_tokens = len(prompt) // 4
        
        return estimated_tokens <= max_tokens


class PromptBuilder:
    """
    Fluent interface for building prompts
    
    Example:
        prompt = (PromptBuilder()
                  .set_style("detailed")
                  .add_query(query)
                  .add_context(chunks)
                  .build())
    """
    
    def __init__(self):
        self.style          = "base"
        self.query          = None
        self.context_chunks = []
        self.task_type      = "qa"  # qa, comparison, summary, extraction
        self.extract_type   = None
        self.summary_focus  = "general"

    
    def set_style(self, style: str) -> 'PromptBuilder':
        """
        Set prompt style
        """
        self.style = style
        return self
    

    def add_query(self, query: str) -> 'PromptBuilder':
        """
        Add user query
        """
        self.query = query
        return self
    

    def add_context(self, context_chunks: List[ChunkWithScore]) -> 'PromptBuilder':
        """
        Add context chunks
        """
        self.context_chunks = context_chunks
        return self
    

    def set_task(self, task_type: str, **kwargs) -> 'PromptBuilder':
        """
        Set task type and parameters
        """
        self.task_type     = task_type
        self.extract_type  = kwargs.get("extract_type")
        self.summary_focus = kwargs.get("summary_focus", "general")
        
        return self
    

    def build(self) -> tuple[str, str]:
        """
        Build final prompt
        
        Returns:
        --------
            { tuple }    : (system_prompt, user_prompt)
        """
        system_prompt = PromptTemplates.get_system_prompt(style = self.style)
        
        if (self.task_type == "qa"):
            user_prompt = PromptTemplates.build_qa_prompt(query          = self.query, 
                                                          context_chunks = self.context_chunks, 
                                                          style          = self.style,
                                                         )

        elif (self.task_type == "comparison"):
            user_prompt = PromptTemplates.build_comparison_prompt(query          = self.query, 
                                                                  context_chunks = self.context_chunks,
                                                                 )

        elif (self.task_type == "summary"):
            user_prompt = PromptTemplates.build_summary_prompt(context_chunks = self.context_chunks, 
                                                               focus          = self.summary_focus,
                                                              )

        elif (self.task_type == "extraction"):
            user_prompt = PromptTemplates.build_extraction_prompt(context_chunks = self.context_chunks, 
                                                                  extract_type   = self.extract_type,
                                                                 )

        elif (self.task_type == "legal"):
            user_prompt = PromptTemplates.build_legal_analysis_prompt(query          = self.query, 
                                                                      context_chunks = self.context_chunks,
                                                                     )

        elif (self.task_type == "technical"):
            user_prompt = PromptTemplates.build_technical_doc_prompt(query          = self.query, 
                                                                     context_chunks = self.context_chunks,
                                                                    )
        
        else:
            user_prompt = PromptTemplates.build_qa_prompt(query          = self.query, 
                                                          context_chunks = self.context_chunks,
                                                         )
        
        return system_prompt, user_prompt



if __name__ == "__main__":
    # Test prompt templates
    from config.models import DocumentChunk
    from config.models import ChunkWithScore

    # Create test chunks
    chunk1             = DocumentChunk(chunk_id    = "chunk_doc_123_0",
                                       document_id = "doc_123",
                                       text        = "The Q3 revenue was $5.2M, up 23% year-over-year.",
                                       chunk_index = 0,
                                       start_char  = 0,
                                       end_char    = 100,
                                       page_number = 1,
                                       token_count = 20,
                                      )
    
    chunk2             = DocumentChunk(chunk_id    = "chunk_doc_123_1",
                                       document_id = "doc_123",
                                       text        = "The company projects Q4 revenue of $6M based on current pipeline.",
                                       chunk_index = 1,
                                       start_char  = 100,
                                       end_char    = 200,
                                       page_number = 2,
                                       token_count = 18,
                                      )
    
    chunks_with_scores = [ChunkWithScore(chunk = chunk1, score = 0.95, rank = 1, retrieval_method = "hybrid"),
                          ChunkWithScore(chunk = chunk2, score = 0.87, rank = 2, retrieval_method = "hybrid"),
                         ]
    
    # Test basic QA prompt
    print("=== Basic QA Prompt ===")
    prompt = PromptTemplates.build_qa_prompt(query          = "What was the Q3 revenue?", 
                                             context_chunks = chunks_with_scores,
                                            )

    print(prompt[:500] + "...\n")
    
    # Test fluent interface
    print("=== Fluent Builder ===")
    system, user = (PromptBuilder().set_style("detailed").add_query("What is the revenue trend?").add_context(chunks_with_scores).build())
    
    print("System:", system[:200] + "...")
    print("\nUser:", user[:300] + "...\n")
    
    # Test sources formatting
    print("=== Sources List ===")
    print(PromptTemplates.format_sources_list(chunks_with_scores))