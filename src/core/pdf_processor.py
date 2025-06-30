"""PDF processing module for extracting text from PDF documents."""

import fitz  # PyMuPDF
from typing import List, Tuple
import logging

logger = logging.getLogger(__name__)


class PDFProcessor:
    """Handles PDF text extraction and processing."""
    
    def __init__(self):
        self.logger = logger
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from PDF file.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Extracted text as a single string
            
        Raises:
            FileNotFoundError: If PDF file doesn't exist
            Exception: If PDF processing fails
        """
        try:
            doc = fitz.open(pdf_path)
            all_paragraphs = []
            
            for page_num, page in enumerate(doc):
                try:
                    blocks = page.get_text("blocks")
                    for block in blocks:
                        paragraph_text = block[4].replace('\n', ' ').strip()
                        if paragraph_text:
                            all_paragraphs.append(paragraph_text)
                except Exception as e:
                    self.logger.warning(f"Error processing page {page_num + 1}: {e}")
                    continue
            
            doc.close()
            extracted_text = "\n\n".join(all_paragraphs)
            
            self.logger.info(f"Successfully extracted {len(extracted_text)} characters from PDF")
            return extracted_text
            
        except FileNotFoundError:
            self.logger.error(f"PDF file not found: {pdf_path}")
            raise
        except Exception as e:
            self.logger.error(f"Failed to extract text from PDF: {e}")
            raise
    
    def get_pdf_info(self, pdf_path: str) -> dict:
        """Get basic information about the PDF file.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Dictionary containing PDF metadata
        """
        try:
            doc = fitz.open(pdf_path)
            info = {
                'page_count': len(doc),
                'metadata': doc.metadata,
                'file_size': doc.tobytes().__sizeof__(),
                'is_encrypted': doc.is_encrypted,
                'is_pdf': doc.is_pdf
            }
            doc.close()
            return info
        except Exception as e:
            self.logger.error(f"Failed to get PDF info: {e}")
            return {}
    
    def extract_text_by_pages(self, pdf_path: str) -> List[Tuple[int, str]]:
        """Extract text from PDF page by page.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            List of tuples (page_number, page_text)
        """
        try:
            doc = fitz.open(pdf_path)
            pages = []
            
            for page_num, page in enumerate(doc):
                page_text = page.get_text()
                pages.append((page_num + 1, page_text))
            
            doc.close()
            return pages
            
        except Exception as e:
            self.logger.error(f"Failed to extract text by pages: {e}")
            return [] 