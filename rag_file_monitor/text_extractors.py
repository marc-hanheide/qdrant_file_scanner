"""
Text extraction utilities for various file formats
"""

import logging
import magic
import chardet
from pathlib import Path
from typing import Optional

# PDF extraction
try:
    import PyPDF2
except ImportError:
    PyPDF2 = None

# DOCX extraction
try:
    from docx import Document
except ImportError:
    Document = None

# HTML extraction
try:
    from bs4 import BeautifulSoup
except ImportError:
    BeautifulSoup = None


class TextExtractor:
    """Extract text from various file formats"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    def extract_text(self, file_path: str) -> str:
        """Extract text from file based on its type"""
        path = Path(file_path)
        extension = path.suffix.lower()
        
        try:
            if extension == '.pdf':
                return self.extract_pdf(file_path)
            elif extension == '.docx':
                return self.extract_docx(file_path)
            elif extension in ['.html', '.htm']:
                return self.extract_html(file_path)
            elif extension in ['.txt', '.md', '.rtf']:
                return self.extract_text_file(file_path)
            else:
                # Try as text file
                return self.extract_text_file(file_path)
                
        except Exception as e:
            self.logger.error(f"Error extracting text from {file_path}: {str(e)}")
            return ""
            
    def extract_pdf(self, file_path: str) -> str:
        """Extract text from PDF file"""
        if PyPDF2 is None:
            raise ImportError("PyPDF2 is required for PDF extraction")
            
        text = ""
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
        except Exception as e:
            self.logger.error(f"Error reading PDF {file_path}: {str(e)}")
            
        return text.strip()
        
    def extract_docx(self, file_path: str) -> str:
        """Extract text from DOCX file"""
        if Document is None:
            raise ImportError("python-docx is required for DOCX extraction")
            
        try:
            doc = Document(file_path)
            text = []
            for paragraph in doc.paragraphs:
                text.append(paragraph.text)
            return "\n".join(text)
        except Exception as e:
            self.logger.error(f"Error reading DOCX {file_path}: {str(e)}")
            return ""
            
    def extract_html(self, file_path: str) -> str:
        """Extract text from HTML file"""
        if BeautifulSoup is None:
            raise ImportError("beautifulsoup4 is required for HTML extraction")
            
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
                
            soup = BeautifulSoup(content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
                
            # Get text and clean it up
            text = soup.get_text()
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = '\n'.join(chunk for chunk in chunks if chunk)
            
            return text
            
        except Exception as e:
            self.logger.error(f"Error reading HTML {file_path}: {str(e)}")
            return ""
            
    def extract_text_file(self, file_path: str) -> str:
        """Extract text from plain text file with memory optimization"""
        try:
            # Detect encoding first with limited data
            with open(file_path, 'rb') as file:
                # Read only a sample for encoding detection to save memory
                raw_data = file.read(10000)  # Read first 10KB for encoding detection
                
            result = chardet.detect(raw_data)
            encoding = result['encoding'] or 'utf-8'
            
            # For very large files, implement size limit
            with open(file_path, 'r', encoding=encoding, errors='ignore') as file:
                # Read in chunks for very large files
                max_size = 10 * 1024 * 1024  # 10MB limit
                content = file.read(max_size)
                
                # Check if file was truncated
                if len(content) == max_size:
                    self.logger.warning(f"Large file {file_path} was truncated to {max_size} bytes")
                    
            return content
                
        except Exception as e:
            self.logger.error(f"Error reading text file {file_path}: {str(e)}")
            return ""
