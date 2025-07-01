"""
Integration tests for the RAG file scanner system
"""

import pytest
import tempfile
import os
import yaml
import shutil
from pathlib import Path
import sys
import time
from unittest.mock import patch, Mock

# Add the project root to sys.path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def _has_all_dependencies():
    """Check if all required dependencies are available"""
    try:
        import sentence_transformers
        import qdrant_client
        return True
    except ImportError:
        return False


@pytest.fixture
def temp_directory():
    """Create a temporary directory for testing"""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def sample_files(temp_directory):
    """Create sample files for testing"""
    files = {}
    
    # Text file
    txt_file = Path(temp_directory) / "sample.txt"
    txt_file.write_text("This is a sample text file for testing.")
    files["txt"] = str(txt_file)
    
    # Markdown file
    md_file = Path(temp_directory) / "sample.md"
    md_file.write_text("# Sample Markdown\n\nThis is markdown content.")
    files["md"] = str(md_file)
    
    # HTML file
    html_file = Path(temp_directory) / "sample.html"
    html_file.write_text("<html><body><h1>Test</h1><p>HTML content</p></body></html>")
    files["html"] = str(html_file)
    
    # Files to ignore
    tmp_file = Path(temp_directory) / "temp.tmp"
    tmp_file.write_text("Temporary file")
    files["tmp"] = str(tmp_file)
    
    log_file = Path(temp_directory) / "app.log"
    log_file.write_text("Log file content")
    files["log"] = str(log_file)
    
    return files


@pytest.fixture
def integration_config(temp_directory):
    """Create configuration for integration testing"""
    return {
        "directories": {
            temp_directory: {"ignore_extensions": [], "max_filesize": 0}
        },
        "file_extensions": [".txt", ".md", ".html", ".pdf", ".docx"],
        "qdrant": {
            "host": "localhost",
            "port": 6333,
            "collection_name": "test_integration",
            "vector_size": 384,
            "vector_name": "test-vector"
        },
        "embedding": {
            "model_name": "all-MiniLM-L6-v2",
            "chunk_size": 512,
            "chunk_overlap": 100
        },
        "processing": {
            "max_file_size_mb": 5,
            "exclude_patterns": ["*.tmp", "*.log"],
            "delete_embeddings_on_file_deletion": False
        },
        "memory": {
            "chunk_batch_size": 10,
            "hash_loading_batch_size": 100,
            "unload_model_after_idle_minutes": 30,
            "force_gc_after_operations": 100
        },
        "logging": {
            "level": "WARNING",
            "file": "test.log"
        }
    }


class TestTextExtractionIntegration:
    """Integration tests for text extraction from various file formats"""

    def test_text_extraction_pipeline(self, sample_files):
        """Test the complete text extraction pipeline"""
        from rag_file_monitor.text_extractors import TextExtractor
        
        extractor = TextExtractor()
        
        # Test text file extraction
        txt_content = extractor.extract_text(sample_files["txt"])
        assert "sample text file" in txt_content
        
        # Test markdown file extraction
        md_content = extractor.extract_text(sample_files["md"])
        assert "Sample Markdown" in md_content
        assert "markdown content" in md_content
        
        # Test HTML file extraction
        html_content = extractor.extract_text(sample_files["html"])
        assert "Test" in html_content
        assert "HTML content" in html_content


@pytest.mark.skipif(not _has_all_dependencies(), reason="Required dependencies not available")
class TestEndToEndIntegration:
    """End-to-end integration tests"""

    def test_file_processing_workflow(self, integration_config, sample_files, temp_directory):
        """Test the complete file processing workflow"""
        with patch('qdrant_client.QdrantClient') as mock_qdrant_client:
            # Mock Qdrant client
            mock_client = Mock()
            mock_qdrant_client.return_value = mock_client
            mock_client.search.return_value = []
            mock_client.upsert.return_value = Mock()
            mock_client.get_collections.return_value = Mock()
            
            try:
                from rag_file_monitor.file_monitor import FileMonitor
                
                # Initialize file monitor
                monitor = FileMonitor(integration_config)
                
                # Test configuration parsing
                directories_config = monitor.get_directories_config()
                assert temp_directory in directories_config
                
                # Test file filtering
                txt_file = sample_files["txt"]
                tmp_file = sample_files["tmp"]
                
                assert not monitor.should_ignore_file(txt_file)
                assert monitor.should_ignore_file(tmp_file)
                
                # Test directory-specific extension handling
                dir_config = directories_config[temp_directory]
                effective_extensions = monitor.get_effective_extensions_for_directory(
                    temp_directory, dir_config
                )
                expected_extensions = {".txt", ".md", ".html", ".pdf", ".docx"}
                assert effective_extensions == expected_extensions
                
            except ImportError as e:
                pytest.skip(f"Required dependencies not available: {e}")

    def test_scan_directory_workflow(self, integration_config, sample_files, temp_directory):
        """Test directory scanning workflow"""
        with patch('qdrant_client.QdrantClient') as mock_qdrant_client:
            # Mock Qdrant client
            mock_client = Mock()
            mock_qdrant_client.return_value = mock_client
            mock_client.search.return_value = []
            mock_client.upsert.return_value = Mock()
            mock_client.get_collection.return_value = Mock()
            
            try:
                from rag_file_monitor.file_monitor import FileMonitor
                
                monitor = FileMonitor(integration_config)
                
                # Get files that should be processed
                processed_files = []
                all_files = list(Path(temp_directory).rglob("*"))
                
                for file_path in all_files:
                    if file_path.is_file():
                        if not monitor.should_ignore_file(str(file_path)):
                            processed_files.append(str(file_path))
                
                # Should have processed .txt, .md, .html files but not .tmp, .log
                expected_processed = {sample_files["txt"], sample_files["md"], sample_files["html"]}
                assert set(processed_files) == expected_processed
                
            except ImportError as e:
                pytest.skip(f"Required dependencies not available: {e}")


class TestConfigurationIntegration:
    """Integration tests for configuration handling"""

    def test_config_file_loading(self, integration_config, temp_directory):
        """Test loading configuration from YAML file"""
        config_file = Path(temp_directory) / "test_config.yaml"
        
        with open(config_file, 'w') as f:
            yaml.dump(integration_config, f)
        
        try:
            with patch('rag_file_monitor.file_monitor.EmbeddingManager'), \
                 patch('rag_file_monitor.file_monitor.TextExtractor'):
                from rag_file_monitor.file_monitor import FileMonitor
                
                # Test loading from file path
                monitor = FileMonitor(str(config_file))
                assert monitor.config == integration_config
                
        except ImportError as e:
            pytest.skip(f"Required dependencies not available: {e}")

    def test_directory_config_variations(self, temp_directory):
        """Test different directory configuration formats"""
        # Test new format
        new_format_config = {
            "directories": {
                temp_directory: {"ignore_extensions": [".log"], "max_filesize": 10}
            },
            "file_extensions": [".txt", ".pdf"],
            "processing": {"max_file_size_mb": 5, "exclude_patterns": []},
            "qdrant": {"host": "localhost", "port": 6333, "collection_name": "test"},
            "embedding": {"model_name": "test-model", "chunk_size": 512, "chunk_overlap": 100},
            "memory": {"chunk_batch_size": 10},
            "logging": {"level": "WARNING", "file": "test.log"}
        }
        
        # Test legacy format
        legacy_format_config = {
            "directories": [temp_directory],
            "file_extensions": [".txt", ".pdf"],
            "processing": {"max_file_size_mb": 5, "exclude_patterns": []},
            "qdrant": {"host": "localhost", "port": 6333, "collection_name": "test"},
            "embedding": {"model_name": "test-model", "chunk_size": 512, "chunk_overlap": 100},
            "memory": {"chunk_batch_size": 10},
            "logging": {"level": "WARNING", "file": "test.log"}
        }
        
        try:
            with patch('rag_file_monitor.file_monitor.EmbeddingManager'), \
                 patch('rag_file_monitor.file_monitor.TextExtractor'):
                from rag_file_monitor.file_monitor import FileMonitor
                
                # Test new format
                monitor_new = FileMonitor(new_format_config)
                dirs_config_new = monitor_new.get_directories_config()
                assert temp_directory in dirs_config_new
                assert dirs_config_new[temp_directory]["ignore_extensions"] == [".log"]
                assert dirs_config_new[temp_directory]["max_filesize"] == 10
                
                # Test legacy format
                monitor_legacy = FileMonitor(legacy_format_config)
                dirs_config_legacy = monitor_legacy.get_directories_config()
                assert temp_directory in dirs_config_legacy
                assert dirs_config_legacy[temp_directory]["ignore_extensions"] == []
                assert dirs_config_legacy[temp_directory]["max_filesize"] == 0
                
        except ImportError as e:
            pytest.skip(f"Required dependencies not available: {e}")


class TestErrorHandling:
    """Integration tests for error handling"""

    def test_invalid_file_handling(self, integration_config):
        """Test handling of invalid or corrupted files"""
        from rag_file_monitor.text_extractors import TextExtractor
        
        extractor = TextExtractor()
        
        # Test non-existent file
        result = extractor.extract_text("/nonexistent/file.txt")
        assert result == ""
        
        # Test binary file (should not crash)
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(b"\x00\x01\x02\x03")  # Binary content
            tmp_path = tmp.name
        
        try:
            result = extractor.extract_text(tmp_path)
            assert isinstance(result, str)  # Should return string, even if empty
        finally:
            os.unlink(tmp_path)

    def test_config_validation_errors(self):
        """Test configuration validation and error handling"""
        invalid_configs = [
            {},  # Empty config
            {"directories": {}},  # Missing required sections
            {"directories": {"invalid": None}},  # Invalid directory config
        ]
        
        for config in invalid_configs:
            with pytest.raises((KeyError, ValueError, ImportError, Exception)):
                with patch('rag_file_monitor.file_monitor.EmbeddingManager'), \
                     patch('rag_file_monitor.file_monitor.TextExtractor'):
                    from rag_file_monitor.file_monitor import FileMonitor
                    FileMonitor(config)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
