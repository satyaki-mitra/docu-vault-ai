# DEPENDENCIES
import re
import numpy as np
from typing import List
from typing import Tuple
from typing import Optional
from config.models import DocumentChunk
from config.settings import get_settings
from config.models import DocumentMetadata
from config.models import ChunkingStrategy
from config.logging_config import get_logger
from chunking.base_chunker import BaseChunker
from chunking.base_chunker import ChunkerConfig
from chunking.token_counter import TokenCounter
from chunking.fixed_chunker import FixedChunker
from chunking.overlap_manager import OverlapManager
from sentence_transformers import SentenceTransformer


# Setup Settings and Logging
logger   = get_logger(__name__)
settings = get_settings()


class SemanticChunker(BaseChunker):
    """
    Semantic chunking strategy with section-aware splitting:
    - Detects section boundaries and NEVER crosses them
    - Creates chunks based on semantic similarity within sections
    - Preserves hierarchical structure (sections → subsections → content)
    
    Best for:
    - Medium documents (50K-500K tokens)
    - Documents with clear topics/sections
    - When context coherence is critical
    """
    def __init__(self, chunk_size: int = None, overlap: int = None, similarity_threshold: float = None, min_chunk_size: int = 100, 
                 embedding_model: Optional[SentenceTransformer] = None, respect_section_boundaries: bool = True):
        """
        Initialize semantic chunker
        
        Arguments:
        ----------
            chunk_size                  { int }                  : Target tokens per chunk (soft limit)

            overlap                     { int }                  : Overlap tokens between chunks
            
            similarity_threshold        { float }                : Threshold for semantic breakpoints (0-1)
            
            min_chunk_size              { int }                  : Minimum chunk size in tokens
            
            embedding_model             { SentenceTransformer }  : Pre-loaded embedding model (optional)
            
            respect_section_boundaries  { bool }                 : Detect and respect section headers
        """
        super().__init__(ChunkingStrategy.SEMANTIC)
        
        self.chunk_size                  = chunk_size or settings.FIXED_CHUNK_SIZE
        self.overlap                     = overlap or settings.FIXED_CHUNK_OVERLAP
        self.similarity_threshold        = similarity_threshold or settings.SEMANTIC_BREAKPOINT_THRESHOLD
        self.min_chunk_size              = min_chunk_size
        self.respect_section_boundaries  = respect_section_boundaries
        
        # Initialize token counter and overlap manager
        self.token_counter               = TokenCounter()
        self.overlap_manager             = OverlapManager(overlap_tokens = self.overlap)
        
        # Initialize or use provided embedding model
        if embedding_model is not None:
            self.embedding_model = embedding_model

        else:
            try:
                self.logger.info(f"Loading embedding model: {settings.EMBEDDING_MODEL}")
                self.embedding_model = SentenceTransformer(settings.EMBEDDING_MODEL)
                self.logger.info("Embedding model loaded successfully")
           
            except Exception as e:
                self.logger.error(f"Failed to load embedding model: {repr(e)}")
                self.embedding_model = None
        
        self.logger.info(f"Initialized SemanticChunker: chunk_size={self.chunk_size}, threshold={self.similarity_threshold}, "
                         f"model_loaded={self.embedding_model is not None}, section_aware={self.respect_section_boundaries}")
    

    def chunk_text(self, text: str, metadata: Optional[DocumentMetadata] = None) -> List[DocumentChunk]:
        """
        Chunk text based on semantic similarity AND section structure
        
        Arguments:
        ----------
            text            { str }                : Input text

            metadata        { DocumentMetadata }   : Document metadata
        
        Returns:
        --------
                            { list }               : List of DocumentChunk objects
        """
        if not text or not text.strip():
            return []
        
        document_id = metadata.document_id if metadata else "unknown"
        
        # If embedding model not available, fall back to fixed chunking
        if self.embedding_model is None:
            self.logger.warning("Embedding model not available, using sentence-based chunking")
            return self._fallback_chunking(text=text, document_id=document_id)
        
        # Detect section headers if enabled
        if self.respect_section_boundaries:
            headers = self._detect_section_headers(text)
            
            if headers:
                self.logger.info(f"Detected {len(headers)} section headers - using section-aware chunking")
                chunks = self._chunk_by_sections(text        = text,
                                                 headers     = headers, 
                                                 document_id = document_id,
                                                )
            
            else:
                self.logger.info("No section headers detected - using standard semantic chunking")
                chunks = self._chunk_semantic(text        = text, 
                                              document_id = document_id,
                                             )

        else:
            chunks = self._chunk_semantic(text        = text, 
                                          document_id = document_id,
                                         )
        
        # Filter out chunks that are too small
        chunks = [c for c in chunks if (c.token_count >= self.min_chunk_size)]
        
        # Use OverlapManager to add proper overlap between semantic chunks
        if ((len(chunks) > 1) and (self.overlap > 0)):
            chunks = self.overlap_manager.add_overlap(chunks         = chunks, 
                                                      overlap_tokens = self.overlap,
                                                     )
        
        self.logger.debug(f"Created {len(chunks)} semantic chunks")
        
        return chunks
    

    def _detect_section_headers(self, text: str) -> List[Tuple[int, str, str, int]]:
        """
        Detect section headers in text to preserve document structure and returns a list of (line_index, header_type, header_text, char_position)
        
        Detects:
        - Project headers 
        - Subsection headers
        - Major section headers
        """
        headers       = list()
        lines         = text.split('\n')
        char_position = 0
        
        for i, line in enumerate(lines):
            line_stripped = line.strip()
            
            # Pattern 1: Headers - "a) Name" or "b) Name"
            if (re.match(r'^[a-z]\)\s+[A-Z]', line_stripped)):
                headers.append((i, 'section', line_stripped, char_position))
                self.logger.debug(f"Detected section header at line {i}: {line_stripped[:60]}")
            
            # Pattern 2: Subsection headers - "● Subsection:" (bullet with colon)
            elif ((line_stripped.startswith('●')) and (':' in line_stripped)):
                headers.append((i, 'subsection', line_stripped, char_position))
                self.logger.debug(f"Detected subsection header at line {i}: {line_stripped[:60]}")
            
            # Pattern 3: Major section headers - "1. SECTION NAME" or all caps with numbers
            elif (re.match(r'^\d+\.\s+[A-Z\s&]+:', line_stripped)):
                headers.append((i, 'section', line_stripped, char_position))
                self.logger.debug(f"Detected major section at line {i}: {line_stripped[:60]}")
            
            # Pattern 4: All caps headers (must be substantial)
            elif (line_stripped.isupper() and (len(line_stripped) > 15) and (not line_stripped.startswith('●'))):
                headers.append((i, 'category', line_stripped, char_position))
                self.logger.debug(f"Detected category header at line {i}: {line_stripped[:60]}")
            
            # +1 for newline
            char_position += len(line) + 1  
        
        return headers
    

    def _chunk_by_sections(self, text: str, headers: List[Tuple], document_id: str) -> List[DocumentChunk]:
        """
        Create chunks that never cross section boundaries: Each chunk preserves its parent section in metadata
        """
        lines                     = text.split('\n')
        chunks                    = list()
        
        # Group lines by their parent section
        current_section_lines     = list()
        current_section_header    = None
        current_subsection_header = None
        start_char                = 0
        
        for line_idx, line in enumerate(lines):
            # Check if this line is a header
            matching_headers = [h for h in headers if (h[0] == line_idx)]
            
            if matching_headers:
                header_info = matching_headers[0]
                header_type = header_info[1]
                header_text = header_info[2]
                
                # If we hit a Header, save previous section
                if (header_type == 'section'):
                    if current_section_lines:
                        # Create chunks from previous section
                        section_text   = '\n'.join(current_section_lines)
                        section_chunks = self._split_section_if_large(text              = section_text,
                                                                      document_id       = document_id,
                                                                      start_index       = len(chunks),
                                                                      start_char        = start_char,
                                                                      section_header    = current_section_header,
                                                                      subsection_header = current_subsection_header,
                                                                     )
                        chunks.extend(section_chunks)
                        start_char    += len(section_text) + 1
                    
                    # Start new section
                    current_section_header    = header_text
                    current_subsection_header = None
                    current_section_lines     = [line]
                
                # If we hit a SUBSECTION header within a section
                elif (header_type == 'subsection'):
                    if (current_section_lines and current_subsection_header):
                        # Save previous subsection
                        section_text          = '\n'.join(current_section_lines)
                        section_chunks        = self._split_section_if_large(text              = section_text,
                                                                             document_id       = document_id,
                                                                             start_index       = len(chunks),
                                                                             start_char        = start_char,
                                                                             section_header    = current_section_header,
                                                                             subsection_header = current_subsection_header,
                                                                            )
                        chunks.extend(section_chunks)
                        start_char           += len(section_text) + 1
                        current_section_lines = list()
                    
                    # Update subsection
                    current_subsection_header = header_text
                    current_section_lines.append(line)
                
                else:
                    current_section_lines.append(line)
            
            else:
                current_section_lines.append(line)
        
        # Process final section
        if current_section_lines:
            section_text   = '\n'.join(current_section_lines)

            section_chunks = self._split_section_if_large(text              = section_text,
                                                          document_id       = document_id,
                                                          start_index       = len(chunks),
                                                          start_char        = start_char,
                                                          section_header    = current_section_header,
                                                          subsection_header = current_subsection_header,
                                                         )
            chunks.extend(section_chunks)
        
        return chunks
    

    def _split_section_if_large(self, text: str, document_id: str, start_index: int, start_char: int, section_header: Optional[str],
                                subsection_header: Optional[str]) -> List[DocumentChunk]:
        """
        Split a section if it's too large, while preserving section context: Always stores section info in metadata
        """
        token_count   = self.token_counter.count_tokens(text)
        
        # Build section title for metadata
        section_parts = list()

        if section_header:
            section_parts.append(section_header)

        if subsection_header:
            section_parts.append(subsection_header)
            
        section_title = " | ".join(section_parts) if section_parts else None
        
        # If section fits in one chunk, keep it whole
        if (token_count <= self.chunk_size * 1.5):
            chunk = self._create_chunk(text          = self._clean_chunk_text(text),
                                       chunk_index   = start_index,
                                       document_id   = document_id,
                                       start_char    = start_char,
                                       end_char      = start_char + len(text),
                                       section_title = section_title, 
                                       metadata      = {"section_header"    : section_header,
                                                        "subsection_header" : subsection_header,
                                                        "semantic_chunk"    : True,
                                                        "section_aware"     : True,
                                                       }
                                      )
            return [chunk]
        
        # Section too large - split by bullet points or sentences: But always keep section context in metadata
        if '❖' in text or '●' in text:
            # Split by bullet points (Interactive Demo Features style)
            parts = re.split(r'(❖[^\n]+)', text)
            parts = [p for p in parts if p.strip()]

        else:
            # Split by sentences within this section
            parts = self._split_sentences(text)
        
        sub_chunks  = []
        current_pos = start_char
        
        for part in parts:
            if not part.strip():
                continue
            
            part_tokens = self.token_counter.count_tokens(part)
            
            # Create chunk with preserved section context
            chunk       = self._create_chunk(text          = self._clean_chunk_text(part),
                                             chunk_index   = start_index + len(sub_chunks),
                                             document_id   = document_id,
                                             start_char    = current_pos,
                                             end_char      = current_pos + len(part),
                                             section_title = section_title, 
                                             metadata      = {"section_header"     : section_header,
                                                              "subsection_header"  : subsection_header,
                                                              "parent_section"     : section_title,
                                                              "semantic_chunk"     : True,
                                                              "section_aware"      : True,
                                                              "is_subsection_part" : True,
                                                             }
                                            )
            sub_chunks.append(chunk)
            current_pos += len(part)
        
        if sub_chunks:
            return sub_chunks

        else:
            chunks_list = [self._create_chunk(text          = self._clean_chunk_text(text),
                                              chunk_index   = start_index,
                                              document_id   = document_id,
                                              start_char    = start_char,
                                              end_char      = start_char + len(text),
                                              section_title = section_title,
                                              metadata      = {"section_header"    : section_header,
                                                               "subsection_header" : subsection_header,
                                                               "semantic_chunk"    : True,
                                                              }
                                             )
                          ]

            return chunks_list
    

    def _chunk_semantic(self, text: str, document_id: str) -> List[DocumentChunk]:
        """
        Standard semantic chunking (when no headers detected)
        """
        # Split into sentences
        sentences = self._split_sentences(text = text)
        
        if (len(sentences) < 2):
            return self._create_single_chunk(text=text, document_id=document_id)
        
        # Calculate semantic similarities
        similarities = self._calculate_similarities(sentences=sentences)
        
        # Find breakpoints
        breakpoints  = self._find_breakpoints(similarities=similarities)
        
        # Create chunks WITHOUT overlap
        chunks       = self._create_chunks_from_breakpoints(sentences   = sentences,
                                                            breakpoints = breakpoints,
                                                            document_id = document_id,
                                                           )
        
        return chunks
    

    def _split_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences
        """
        # Protect abbreviations
        protected        = text
        abbreviations    = ['Dr.', 'Mr.', 'Mrs.', 'Ms.', 'Jr.', 'Sr.', 'Prof.', 'Inc.', 'Ltd.', 'Corp.', 'Co.', 'vs.', 'etc.', 'e.g.', 'i.e.', 'Ph.D.', 'M.D.', 'B.A.', 'M.A.', 'U.S.', 'U.K.']
        
        for abbr in abbreviations:
            protected = protected.replace(abbr, abbr.replace('.', '<DOT>'))
        
        # Split on sentence boundaries
        sentence_pattern = r'(?<=[.!?])\s+(?=[A-Z])'
        sentences        = re.split(sentence_pattern, protected)
        
        # Restore abbreviations
        sentences        = [s.replace('<DOT>', '.').strip() for s in sentences]
        
        # Filter empty
        sentences        = [s for s in sentences if s]
        
        return sentences
    

    def _calculate_similarities(self, sentences: List[str]) -> List[float]:
        """
        Calculate cosine similarity between adjacent sentences
        """
        if (len(sentences) < 2):
            return []
        
        self.logger.debug(f"Generating embeddings for {len(sentences)} sentences")
        
        embeddings   = self.embedding_model.encode(sentences,
                                                   show_progress_bar = False,
                                                   convert_to_numpy  = True,
                                                  )
        
        similarities = list()

        for i in range(len(embeddings) - 1):
            similarity = self._cosine_similarity(vec1 = embeddings[i], 
                                                 vec2 = embeddings[i + 1],
                                                )
            similarities.append(similarity)
        
        return similarities
    

    @staticmethod
    def _cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two vectors
        """
        dot_product = np.dot(vec1, vec2)
        norm1       = np.linalg.norm(vec1)
        norm2       = np.linalg.norm(vec2)
        
        if ((norm1 == 0) or (norm2 == 0)):
            return 0.0
        
        return dot_product / (norm1 * norm2)
    

    def _find_breakpoints(self, similarities: List[float]) -> List[int]:
        """
        Find breakpoints where semantic similarity drops significantly
        """
        if not similarities:
            return []
        
        similarities_array = np.array(similarities)
        threshold          = np.percentile(similarities_array, (1 - self.similarity_threshold) * 100)
        
        breakpoints        = [0]

        for i, sim in enumerate(similarities):
            if (sim < threshold):
                breakpoints.append(i + 1)
        
        self.logger.debug(f"Found {len(breakpoints)} breakpoints with threshold {threshold:.3f}")
        
        return breakpoints
    

    def _create_chunks_from_breakpoints(self, sentences: List[str], breakpoints: List[int], document_id: str) -> List[DocumentChunk]:
        """
        Create chunks from sentences and breakpoints WITHOUT overlap
        """
        chunks      = list()
        breakpoints = sorted(set(breakpoints))
        
        if (breakpoints[-1] != len(sentences)):
            breakpoints.append(len(sentences))
        
        current_pos = 0
        
        for i in range(len(breakpoints) - 1):
            start_idx = breakpoints[i]
            end_idx   = breakpoints[i + 1]
            
            chunk_sentences = sentences[start_idx:end_idx]
            
            if not chunk_sentences:
                continue
            
            chunk_text  = " ".join(chunk_sentences)
            token_count = self.token_counter.count_tokens(chunk_text)
            
            if (token_count > self.chunk_size * 1.5):
                sub_chunks = self._split_large_chunk_simple(chunk_sentences = chunk_sentences,
                                                            document_id     = document_id,
                                                            start_index     = len(chunks),
                                                            start_char      = current_pos,
                                                           )
                chunks.extend(sub_chunks)
           
            else:
                chunk = self._create_chunk(text        = self._clean_chunk_text(chunk_text),
                                           chunk_index = len(chunks),
                                           document_id = document_id,
                                           start_char  = current_pos,
                                           end_char    = current_pos + len(chunk_text),
                                           metadata    = {"sentences"      : len(chunk_sentences),
                                                          "semantic_chunk" : True,
                                                         }
                                          )

                chunks.append(chunk)
            
            current_pos += len(chunk_text)
        
        return chunks
    

    def _split_large_chunk_simple(self, chunk_sentences: List[str], document_id: str, start_index: int, start_char: int) -> List[DocumentChunk]:
        """
        Split a large chunk into smaller pieces without overlap
        """
        sub_chunks        = list()
        current_sentences = list()
        current_tokens    = 0
        current_pos       = start_char
        
        for sentence in chunk_sentences:
            sentence_tokens = self.token_counter.count_tokens(sentence)
            
            if (((current_tokens + sentence_tokens) > self.chunk_size) and current_sentences):
                chunk_text        = " ".join(current_sentences)
                chunk             = self._create_chunk(text        = self._clean_chunk_text(chunk_text),
                                                       chunk_index = start_index + len(sub_chunks),
                                                       document_id = document_id,
                                                       start_char  = current_pos,
                                                       end_char    = current_pos + len(chunk_text),
                                                      )
                sub_chunks.append(chunk)
                
                current_sentences = [sentence]
                current_tokens    = sentence_tokens
                current_pos      += len(chunk_text)

            else:
                current_sentences.append(sentence)
                current_tokens += sentence_tokens
        
        if current_sentences:
            chunk_text = " ".join(current_sentences)
            chunk      = self._create_chunk(text        = self._clean_chunk_text(chunk_text),
                                            chunk_index = start_index + len(sub_chunks),
                                            document_id = document_id,
                                            start_char  = current_pos,
                                            end_char    = current_pos + len(chunk_text),
                                           )
            sub_chunks.append(chunk)
        
        return sub_chunks
    

    def _create_single_chunk(self, text: str, document_id: str) -> List[DocumentChunk]:
        """
        Create a single chunk for short text
        """
        chunk = self._create_chunk(text        = self._clean_chunk_text(text),
                                   chunk_index = 0,
                                   document_id = document_id,
                                   start_char  = 0,
                                   end_char    = len(text),
                                  )
        return [chunk]
    

    def _fallback_chunking(self, text: str, document_id: str) -> List[DocumentChunk]:
        """
        Fallback to sentence-based chunking when embeddings unavailable
        """
        fallback_chunker = FixedChunker(chunk_size                  = self.chunk_size,
                                        overlap                     = self.overlap,
                                        respect_sentence_boundaries = True,
                                       )
        
        metadata         = DocumentMetadata(document_id     = document_id,
                                            filename        = "fallback",
                                            document_type   = "txt",
                                            file_size_bytes = len(text),
                                           )
        
        return fallback_chunker.chunk_text(text, metadata)
    

    @classmethod
    def from_config(cls, config: ChunkerConfig) -> 'SemanticChunker':
        """
        Create SemanticChunker from configuration
        """
        return cls(chunk_size                 = config.chunk_size,
                   overlap                    = config.overlap,
                   similarity_threshold       = config.extra.get('semantic_threshold', settings.SEMANTIC_BREAKPOINT_THRESHOLD),
                   min_chunk_size             = config.min_chunk_size,
                   respect_section_boundaries = config.extra.get('respect_section_boundaries', True),
                  )