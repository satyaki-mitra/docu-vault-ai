# DEPENDENCIES
import os
import cv2
import fitz
import easyocr
import numpy as np
from PIL import Image
from io import BytesIO
from typing import List
from typing import Dict
from typing import Tuple
from pathlib import Path
from PIL import ImageFilter
from typing import Optional
from PIL import ImageEnhance
from paddleocr import PaddleOCR
from config.settings import get_settings
from utils.error_handler import OCRException
from config.logging_config import get_logger
from utils.error_handler import handle_errors


# Setup Settings and Logging
settings = get_settings()
logger   = get_logger(__name__)


class OCREngine:
    """
    OCR engine with layout preservation - maintains document structure and formatting
    """
    def __init__(self, use_paddle: bool = True, lang: str = 'en', gpu: bool = False):
        """
        Initialize OCR engine
        
        Arguments:
        ----------
            use_paddle  { bool } : Use PaddleOCR as primary (better accuracy)

            lang        { str }  : Language code ('en', 'es', 'fr', 'de', etc.)
            
            gpu         { bool } : Use GPU acceleration if available
        """
        self.logger       = logger
        self.use_paddle   = use_paddle
        self.lang         = lang
        self.gpu          = gpu
        self.paddle_ocr   = None
        self.easy_ocr     = None
        self._initialized = False
        
        self._initialize_engines()
    

    def _initialize_engines(self):
        """
        Initialize OCR engines with proper error handling
        """
        if self.use_paddle:
            try:
                self.paddle_ocr = PaddleOCR(use_angle_cls     = True,
                                            lang              = self.lang,
                                            use_gpu           = self.gpu,
                                            show_log          = False,
                                            det_db_thresh     = 0.3,
                                            det_db_box_thresh = 0.5,
                                           )

                self.logger.info("PaddleOCR initialized successfully")

            except Exception as e:
                self.logger.warning(f"PaddleOCR not available: {repr(e)}. Falling back to EasyOCR.")
                self.use_paddle = False
        
        if not self.use_paddle:
            try: 
                self.easy_ocr = easyocr.Reader([self.lang], gpu = self.gpu)
                self.logger.info("EasyOCR initialized successfully")

            except Exception as e:
                self.logger.error(f"Failed to initialize EasyOCR: {repr(e)}")
                raise OCRException(f"OCR engine initialization failed: {repr(e)}")
        
        self._initialized = True
    

    @handle_errors(error_type=OCRException, log_error=True, reraise=True)
    def extract_text_from_pdf(self, pdf_path: Path, pages: Optional[List[int]] = None, preserve_layout: bool = True) -> str:
        """
        Extract text from PDF using OCR with layout preservation
        
        Arguments:
        ----------
            pdf_path        { Path } : Path to PDF file

            pages           { list } : Specific pages to OCR (None = all pages)
            
            preserve_layout { bool } : Preserve document layout and structure
            
        Returns:
        --------
               { str }               : Extracted text with preserved formatting
        """
        pdf_path = Path(pdf_path)

        self.logger.info(f"Starting OCR extraction from PDF: {pdf_path}")
        
        if not pdf_path.exists():
            raise OCRException(f"PDF file not found: {pdf_path}")
        
        # Convert PDF pages to high-quality images
        images = self._pdf_to_images(pdf_path = pdf_path, 
                                     pages    = pages, 
                                     dpi      = 300,
                                    )

        self.logger.info(f"Converted {len(images)} pages to images for OCR")
        
        # OCR each image with layout preservation
        all_text = list()

        for i, image in enumerate(images):
            page_num = pages[i] if pages else i + 1

            self.logger.info(f"Processing page {page_num}...")
            
            try:
                if preserve_layout:
                    # Extract text with layout information
                    page_text = self._extract_text_with_layout(image    = image, 
                                                               page_num = page_num,
                                                              )
                
                else:
                    # Simple extraction without layout
                    img_array = np.array(image)
                    page_text = self._ocr_image(img_array)

                if page_text and page_text.strip():
                    all_text.append(f"[PAGE {page_num}]\n{page_text}")
                    self.logger.info(f"✓ Extracted {len(page_text)} characters from page {page_num}")
                
                else:
                    self.logger.warning(f"No text extracted from page {page_num}")
            
            except Exception as e:
                self.logger.error(f"OCR failed for page {page_num}: {repr(e)}")
                all_text.append(f"[PAGE {page_num}]\n[OCR FAILED: {str(e)}]")
        
        combined_text = "\n\n".join(all_text)
        self.logger.info(f"OCR completed: {len(combined_text)} total characters extracted")
        
        return combined_text
    

    def _extract_text_with_layout(self, image: Image.Image, page_num: int) -> str:
        """
        Extract text while preserving document layout and structure
        
        Arguments:
        ----------
            image    { Image.Image } : PIL Image
            
            page_num { int }         : Page number
            
        Returns:
        --------
                 { str }             : Formatted text with layout preserved
        """
        img_array = np.array(image)
        
        # Get OCR results with bounding boxes
        if (self.use_paddle and self.paddle_ocr):
            text_blocks = self._ocr_with_layout_paddle(image_array = img_array)
        
        elif self.easy_ocr:
            text_blocks = self._ocr_with_layout_easyocr(image_array = img_array)
        
        else:
            return ""
        
        if not text_blocks:
            return ""
        
        # Organize text blocks into reading order with layout preservation
        formatted_text = self._reconstruct_layout(text_blocks = text_blocks, 
                                                  image_size  = image.size,
                                                 )
        
        return formatted_text
    

    def _ocr_with_layout_paddle(self, image_array: np.ndarray) -> List[Dict]:
        """
        OCR using PaddleOCR and return structured text blocks with positions
        
        Returns:
        --------
            { list } : {'text': str, 'bbox': [...], 'confidence': float}
        """
        try:
            result = self.paddle_ocr.ocr(image_array, cls=True)
            
            if not result or not result[0]:
                return []
            
            text_blocks = list()
            
            for line in result[0]:
                if (line and (len(line) >= 2)):
                    bbox      = line[0]  # [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
                    text_info = line[1]
                    
                    if (isinstance(text_info, (list, tuple)) and (len(text_info) >= 2)):
                        text       = text_info[0]
                        confidence = text_info[1]

                    elif isinstance(text_info, str):
                        text       = text_info
                        confidence = 1.0

                    else:
                        continue
                    
                    if ((confidence > 0.5) and text and text.strip()):
                        # Calculate bounding box coordinates
                        x_coords = [point[0] for point in bbox]
                        y_coords = [point[1] for point in bbox]
                        
                        text_blocks.append({'text'       : text.strip(),
                                            'bbox'       : {'x1': min(x_coords),
                                                            'y1': min(y_coords),
                                                            'x2': max(x_coords),
                                                            'y2': max(y_coords)
                                                           },
                                            'confidence' : confidence,
                                            'center_y'   : (min(y_coords) + max(y_coords)) / 2,
                                            'center_x'   : (min(x_coords) + max(x_coords)) / 2,
                                          })
            
            return text_blocks
            
        except Exception as e:
            self.logger.error(f"PaddleOCR layout extraction failed: {repr(e)}")
            return []
    

    def _ocr_with_layout_easyocr(self, image_array: np.ndarray) -> List[Dict]:
        """
        OCR using EasyOCR and return structured text blocks with positions
        """
        try:
            result = self.easy_ocr.readtext(image_array, paragraph=False)
            
            if not result:
                return []
            
            text_blocks = list()
            
            for detection in result:
                bbox       = detection[0]  # [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
                text       = detection[1]
                confidence = detection[2]
                
                if ((confidence > 0.5) and text and text.strip()):
                    x_coords = [point[0] for point in bbox]
                    y_coords = [point[1] for point in bbox]
                    
                    text_blocks.append({'text'       : text.strip(),
                                        'bbox'       : {'x1' : min(x_coords),
                                                        'y1' : min(y_coords),
                                                        'x2' : max(x_coords),
                                                        'y2' : max(y_coords),
                                                       },
                                        'confidence' : confidence,
                                        'center_y'   : (min(y_coords) + max(y_coords)) / 2,
                                        'center_x'   : (min(x_coords) + max(x_coords)) / 2,
                                      })
            
            return text_blocks
            
        except Exception as e:
            self.logger.error(f"EasyOCR layout extraction failed: {repr(e)}")
            return []
    

    def _reconstruct_layout(self, text_blocks: List[Dict], image_size: Tuple[int, int]) -> str:
        """
        Reconstruct document layout from text blocks
        
        Strategy:
        1. Group text blocks into lines (similar Y coordinates)
        2. Detect columns, tables, lists
        3. Sort lines top to bottom
        4. Within each line, sort left to right
        5. Detect paragraphs, headings, and lists
        6. Add appropriate spacing and formatting
        """
        if not text_blocks:
            return ""
        
        # Sort all blocks by Y position first
        sorted_blocks         = sorted(text_blocks, key = lambda x: (x['center_y'], x['center_x']))
        
        # Detect multi-column layout
        columns               = self._detect_columns(text_blocks = text_blocks, 
                                                     image_size  = image_size,
                                                    )
        
        # Group into lines (blocks with similar Y coordinates)
        lines                 = list()

        current_line          = [sorted_blocks[0]]

        # pixels
        line_height_threshold = 25  
        
        for block in sorted_blocks[1:]:
            # Check if this block is on the same line as the previous one
            y_diff = abs(block['center_y'] - current_line[-1]['center_y'])
            
            if (y_diff < line_height_threshold):
                current_line.append(block)

            else:
                # Sort current line by X position and add to lines
                current_line.sort(key = lambda x: x['center_x'])
                lines.append(current_line)
                
                current_line = [block]
        
        # Don't forget the last line
        if current_line:
            current_line.sort(key = lambda x: x['center_x'])
            lines.append(current_line)
        
        # Reconstruct text with formatting
        formatted_lines = list()
        prev_y          = 0
        prev_indent     = 0
        
        for i, line_blocks in enumerate(lines):
            # Calculate line metrics
            current_y        = line_blocks[0]['center_y']
            vertical_gap     = current_y - prev_y if (prev_y > 0) else 0
            
            # Detect indentation (left margin)
            line_left_margin = line_blocks[0]['bbox']['x1']
            
            # Combine text blocks in this line with proper spacing
            line_text        = self._combine_line_blocks(line_blocks = line_blocks)
            
            # Clean the text
            line_text        = self._clean_ocr_text(text = line_text)
            
            # Skip if empty after cleaning
            if not line_text.strip():
                continue
            
            # Skip likely page numbers or artifacts (single numbers, very short text)
            if self._is_page_artifact(line_text):
                continue
            
            # Add extra newline for paragraph breaks (large vertical gaps)
            # Threshold for paragraph break
            if (vertical_gap > 35):  
                formatted_lines.append("")
            
            # Detect and format different line types
            if (self._is_heading(line_text, line_blocks)):
                # Heading - add extra spacing
                formatted_lines.append(f"\n{line_text}")
            
            elif (self._is_bullet_point(line_text)):
                # Bullet point or list item
                formatted_lines.append(f"  {line_text}")
            
            elif (self._is_table_row(line_blocks)):
                # Table row - preserve spacing between columns
                formatted_lines.append(self._format_table_row(line_blocks))
            
            else:
                # Regular paragraph text
                formatted_lines.append(line_text)
            
            prev_y      = current_y
            prev_indent = line_left_margin
        
        return "\n".join(formatted_lines)
    

    def _combine_line_blocks(self, line_blocks: List[Dict]) -> str:
        """
        Combine text blocks in a line with intelligent spacing
        """
        if (len(line_blocks) == 1):
            return line_blocks[0]['text']
        
        result = list()

        for i, block in enumerate(line_blocks):
            result.append(block['text'])
            
            # Add space between blocks if they're not touching
            if (i < len(line_blocks) - 1):
                next_block = line_blocks[i + 1]
                gap        = next_block['bbox']['x1'] - block['bbox']['x2']
                
                # If gap is significant, add spacing
                if (gap > 20):  # Threshold for adding extra space
                    # Double space for columns/tables
                    result.append("  ")  

                elif (gap > 5):
                    # Normal space
                    result.append(" ")  
        
        return "".join(result)
    

    def _clean_ocr_text(self, text: str) -> str:
        """
        Clean OCR artifacts and normalize text
        """
        # Replace common OCR errors
        replacements = {'''      : "'",  # Smart quote to regular quote
                        '''      : "'",
                        '"'      : '"',
                        '"'      : '"',
                        '—'      : '-',
                        '–'      : '-',
                        '…'      : '...',
                        '\u00a0' : ' ',  # Non-breaking space
                       }
        
        for old, new in replacements.items():
            text = text.replace(old, new)
        
        # Fix common OCR mistakes
        text = text.replace('l ', 'I ')    # lowercase L to I at start of sentence
        text = text.replace(' l ', ' I ')  # lowercase L to I
        
        # Remove extra spaces
        text = ' '.join(text.split())
        
        return text
    

    def _is_page_artifact(self, text: str) -> bool:
        """
        Detect page numbers, headers, footers, and other artifacts
        """
        text = text.strip()
        
        # Empty or very short
        if (len(text) < 2):
            return True
        
        # Just a number (likely page number)
        if (text.isdigit() and (len(text) <= 3)):
            return True
        
        # Common footer patterns
        footer_patterns = ['page', 'of', 'for informational purposes', 'confidential', 'draft', 'version']
        text_lower      = text.lower()

        if ((len(text) < 50) and (any(pattern in text_lower for pattern in footer_patterns))):
            # This is actually useful - don't skip
            return False
        
        # Very short isolated text (likely artifact)
        if ((len(text) <= 3) and not text.isalnum()):
            return True
        
        return False
    

    def _is_bullet_point(self, text: str) -> bool:
        """
        Detect if text is a bullet point or list item
        """
        text           = text.strip()
        
        # Check for common bullet markers
        bullet_markers = ['•', '·', '-', '○', '◦', '*', '►', '▪']
       
        if (text and (text[0] in bullet_markers)):
            return True
        
        # Check for numbered lists
        if (len(text) > 2):
            
            # Pattern: "1. ", "a) ", "i. "
            if (text[0].isdigit() and text[1] in '.):'):
                return True

            if (text[0].isalpha() and len(text) > 1 and text[1] in '.):'):
                return True
        
        return False
    

    def _is_table_row(self, line_blocks: List[Dict]) -> bool:
        """
        Detect if a line is part of a table (multiple separated columns)
        """
        if (len(line_blocks) < 2):
            return False
        
        # Calculate gaps between blocks
        gaps = list()

        for i in range(len(line_blocks) - 1):
            gap = line_blocks[i + 1]['bbox']['x1'] - line_blocks[i]['bbox']['x2']
            gaps.append(gap)
        
        # If there are significant gaps, likely a table
        significant_gaps = sum(1 for gap in gaps if gap > 30)
        
        return (significant_gaps >= 1) and (len(line_blocks) >= 2)
    

    def _format_table_row(self, line_blocks: List[Dict]) -> str:
        """
        Format a table row with proper column alignment
        """
        cells = list()

        for block in line_blocks:
            cells.append(block['text'].strip())
        
        # Join with tab or multiple spaces for better readability
        return ("  |  ".join(cells))
    

    def _detect_columns(self, text_blocks: List[Dict], image_size: Tuple[int, int]) -> List[Dict]:
        """
        Detect multi-column layout
        """
        # Group blocks by X position to detect columns
        if not text_blocks:
            return []
        
        # Return single column
        return [{'x_start': 0, 'x_end': image_size[0]}]
    

    def _is_heading(self, text: str, blocks: List[Dict]) -> bool:
        """
        Detect if a line is likely a heading
        
        Heuristics:
        - All uppercase or Title Case
        - Shorter than typical paragraph lines
        - Often centered or left-aligned
        - Larger font (if detectable from bbox height)
        """
        words = text.split()
        if not words:
            return False
        
        # Skip very short text (likely artifacts)
        if len(text) < 3:
            return False
        
        # Check for common heading keywords
        heading_keywords    = ['summary', 'introduction', 'conclusion', 'analysis', 'report', 'overview', 'chapter', 'section', 'terms', 'points', 'protections', 'category', 'breakdown', 'recommendation', 'clause']
        text_lower          = text.lower()
        has_heading_keyword = any(keyword in text_lower for keyword in heading_keywords)
        
        # All caps or mostly caps
        caps_ratio          = sum(1 for w in words if w.isupper() and len(w) > 1) / len(words)
        
        # Title case (each word starts with capital)
        title_case_ratio    = sum(1 for w in words if w and w[0].isupper()) / len(words)
        
        # Short lines might be headings
        is_short            = len(text) < 100
        
        # Check if text is likely a heading
        is_likely_heading   = ((caps_ratio > 0.7 and is_short) or                                # Mostly uppercase and short
                               (title_case_ratio > 0.8 and is_short and has_heading_keyword) or  # Title case with keywords
                               (has_heading_keyword and is_short and title_case_ratio > 0.5)     # Keywords + some capitals
                              )
        
        # Check font size (larger bounding box height indicates heading)
        if blocks:
            avg_height = sum(b['bbox']['y2'] - b['bbox']['y1'] for b in blocks) / len(blocks)
            
            # Headings often have larger font (taller bbox)
            if (avg_height > 25):  # Threshold for heading font size
                is_likely_heading = is_likely_heading or (is_short and title_case_ratio > 0.5)
        
        return is_likely_heading
    

    def _pdf_to_images(self, pdf_path: Path, pages: Optional[List[int]] = None, dpi: int = 300) -> List[Image.Image]:
        """
        Convert PDF pages to high-quality images
        """
        try:
            doc    = fitz.open(str(pdf_path))
            images = list()
            
            if pages is None:
                pages_to_process = range(len(doc))

            else:
                pages_to_process = [p-1 for p in pages if (0 < p <= len(doc))]
            
            for page_num in pages_to_process:
                page     = doc[page_num]
                
                # High-quality conversion
                zoom     = dpi / 72.0
                mat      = fitz.Matrix(zoom, zoom)
                pix      = page.get_pixmap(matrix = mat, alpha = False)
                
                # Convert to PIL Image
                img_data = pix.tobytes("png")
                image    = Image.open(BytesIO(img_data))
                
                if (image.mode != 'RGB'):
                    image = image.convert('RGB')
                
                images.append(image)
            
            doc.close()
            return images
            
        except Exception as e:
            raise OCRException(f"Failed to convert PDF to images: {repr(e)}")
    

    def _ocr_image(self, image_array: np.ndarray) -> str:
        """
        Simple OCR without layout preservation
        """
        if self.use_paddle and self.paddle_ocr:
            try:
                result = self._ocr_with_paddle_simple(image_array)
                
                if result:
                    return result
            
            except Exception as e:
                self.logger.debug(f"PaddleOCR failed: {repr(e)}")
        
        if self.easy_ocr:
            try:
                result = self._ocr_with_easyocr_simple(image_array)
                
                if result:
                    return result
            
            except Exception as e:
                self.logger.debug(f"EasyOCR failed: {repr(e)}")
        
        return ""
    

    def _ocr_with_paddle_simple(self, image_array: np.ndarray) -> str:
        """
        Simple PaddleOCR extraction
        """
        result = self.paddle_ocr.ocr(image_array, cls=True)
        
        if not result or not result[0]:
            return ""
        
        texts = list()

        for line in result[0]:
            if (line and (len(line) >= 2)):
                text_info = line[1]
                if isinstance(text_info, (list, tuple)):
                    text, conf = text_info[0], text_info[1]
                
                else:
                    text, conf = text_info, 1.0
                
                if ((conf > 0.5) and text):
                    texts.append(text.strip())
        
        return "\n".join(texts)
    

    def _ocr_with_easyocr_simple(self, image_array: np.ndarray) -> str:
        """
        Simple EasyOCR extraction
        """
        result = self.easy_ocr.readtext(image_array)
        
        if not result:
            return ""
        
        texts  = list()

        for detection in result:
            text, conf = detection[1], detection[2]
            if ((conf > 0.5) and text):
                texts.append(text.strip())
        
        return "\n".join(texts)
    

    @handle_errors(error_type = OCRException, log_error = True, reraise = True)
    def extract_text_from_image(self, image_path: Path, preserve_layout: bool = True) -> str:
        """
        Extract text from image file
        """
        image_path = Path(image_path)

        self.logger.info(f"Extracting text from image: {image_path}")
        
        if not image_path.exists():
            raise OCRException(f"Image file not found: {image_path}")
        
        image = Image.open(image_path)

        if (image.mode != 'RGB'):
            image = image.convert('RGB')
        
        if preserve_layout:
            text = self._extract_text_with_layout(image, page_num=1)
        
        else:
            img_array = np.array(image)
            text      = self._ocr_image(img_array)
        
        self.logger.info(f"Image OCR completed: {len(text)} characters extracted")
        return text
    

    def get_supported_languages(self) -> List[str]:
        """
        Get list of supported languages
        """
        return ['en', 'es', 'fr', 'de', 'it', 'pt', 'ru', 'zh', 'ja', 'ko', 'ar']
    

    def get_engine_info(self) -> dict:
        """
        Get information about OCR engine configuration
        """
        return {"primary_engine"      : "PaddleOCR" if self.use_paddle else "EasyOCR",
                "language"            : self.lang,
                "gpu_enabled"         : self.gpu,
                "initialized"         : self._initialized,
                "layout_preservation" : True,
                "supported_languages" : self.get_supported_languages(),
               }


# Global OCR instance
_global_ocr_engine = None


def get_ocr_engine() -> OCREngine:
    """
    Get global OCR engine instance (singleton)
    """
    global _global_ocr_engine
    
    if _global_ocr_engine is None:
        _global_ocr_engine = OCREngine()
    
    return _global_ocr_engine


def extract_text_with_ocr(file_path: Path, preserve_layout: bool = True, **kwargs) -> str:
    """
    Convenience function for OCR text extraction with layout preservation
    """
    ocr_engine = get_ocr_engine()
    
    if (file_path.suffix.lower() == '.pdf'):
        return ocr_engine.extract_text_from_pdf(file_path, preserve_layout=preserve_layout, **kwargs)

    else:
        return ocr_engine.extract_text_from_image(file_path, preserve_layout=preserve_layout, **kwargs)