"""
Unit tests for file monitor module
"""

import pytest
import tempfile
import os
import yaml
from pathlib import Path
import sys
from unittest.mock import Mock, patch, MagicMock

# Add the project root to sys.path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from rag_file_monitor.file_monitor import FileMonitor, FileMonitorHandler


@pytest.fixture
def sample_config():
    """Sample configuration for testing"""
    return {
        "directories": {
            "/test/dir1": {"ignore_extensions": [], "max_filesize": 0},
            "/test/dir2": {"ignore_extensions": [".xlsx"], "max_filesize": 2}
        },
        "file_extensions": [".txt", ".pdf", ".docx", ".xlsx"],
        "qdrant": {
            "host": "localhost",
            "port": 6333,
            "collection_name": "test_collection",
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
            "exclude_patterns": ["*.tmp", "*.log", ".git/*"],
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


class TestFileMonitor:
    """Test cases for FileMonitor class"""

    def test_init_with_config_dict(self, sample_config):
        """Test initialization with configuration dictionary"""
        try:
            with patch('rag_file_monitor.file_monitor.EmbeddingManager'), \
                 patch('rag_file_monitor.file_monitor.TextExtractor'):
                monitor = FileMonitor(sample_config)
                assert monitor.config == sample_config
        except ImportError:
            pytest.skip("Required dependencies not available")

    def test_init_with_config_file(self, sample_config):
        """Test initialization with configuration file"""
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as tmp:
                yaml.dump(sample_config, tmp)
                tmp_path = tmp.name

            try:
                with patch('rag_file_monitor.file_monitor.EmbeddingManager'), \
                     patch('rag_file_monitor.file_monitor.TextExtractor'):
                    monitor = FileMonitor(tmp_path)
                    assert monitor.config == sample_config
            finally:
                os.unlink(tmp_path)
        except ImportError:
            pytest.skip("Required dependencies not available")

    def test_get_directories_config_new_format(self, sample_config):
        """Test parsing of new directory configuration format"""
        try:
            with patch('rag_file_monitor.file_monitor.EmbeddingManager'), \
                 patch('rag_file_monitor.file_monitor.TextExtractor'):
                monitor = FileMonitor(sample_config)
                directories_config = monitor.get_directories_config()
                
                assert isinstance(directories_config, dict)
                assert "/test/dir1" in directories_config
                assert "/test/dir2" in directories_config
                
                # Check default values are set
                for dir_path, dir_config in directories_config.items():
                    assert "ignore_extensions" in dir_config
                    assert "max_filesize" in dir_config
        except ImportError:
            pytest.skip("Required dependencies not available")

    def test_get_directories_config_legacy_format(self):
        """Test parsing of legacy directory configuration format"""
        legacy_config = {
            "directories": ["/test/dir1", "/test/dir2"],
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
                monitor = FileMonitor(legacy_config)
                directories_config = monitor.get_directories_config()
                
                assert isinstance(directories_config, dict)
                assert "/test/dir1" in directories_config
                assert "/test/dir2" in directories_config
                
                # Check that legacy format is converted properly
                for dir_path, dir_config in directories_config.items():
                    assert dir_config["ignore_extensions"] == []
                    assert dir_config["max_filesize"] == 0
        except ImportError:
            pytest.skip("Required dependencies not available")

    def test_get_effective_extensions_for_directory(self, sample_config):
        """Test effective file extension calculation per directory"""
        try:
            with patch('rag_file_monitor.file_monitor.EmbeddingManager'), \
                 patch('rag_file_monitor.file_monitor.TextExtractor'):
                monitor = FileMonitor(sample_config)
                
                # Test directory with no ignored extensions
                dir1_config = {"ignore_extensions": [], "max_filesize": 0}
                extensions1 = monitor.get_effective_extensions_for_directory("/test/dir1", dir1_config)
                expected1 = {".txt", ".pdf", ".docx", ".xlsx"}
                assert extensions1 == expected1
                
                # Test directory with ignored extensions
                dir2_config = {"ignore_extensions": [".xlsx"], "max_filesize": 2}
                extensions2 = monitor.get_effective_extensions_for_directory("/test/dir2", dir2_config)
                expected2 = {".txt", ".pdf", ".docx"}
                assert extensions2 == expected2
        except ImportError:
            pytest.skip("Required dependencies not available")

    def test_get_effective_max_filesize_for_directory(self, sample_config):
        """Test effective file size limit calculation per directory"""
        try:
            with patch('rag_file_monitor.file_monitor.EmbeddingManager'), \
                 patch('rag_file_monitor.file_monitor.TextExtractor'):
                monitor = FileMonitor(sample_config)
                
                # Test directory using global default (max_filesize = 0)
                dir1_config = {"ignore_extensions": [], "max_filesize": 0}
                max_size1 = monitor.get_effective_max_filesize_for_directory("/test/dir1", dir1_config)
                assert max_size1 == 5  # Global default from processing.max_file_size_mb
                
                # Test directory with specific limit
                dir2_config = {"ignore_extensions": [".xlsx"], "max_filesize": 2}
                max_size2 = monitor.get_effective_max_filesize_for_directory("/test/dir2", dir2_config)
                assert max_size2 == 2  # Directory-specific limit
        except ImportError:
            pytest.skip("Required dependencies not available")

    def test_should_ignore_file(self, sample_config):
        """Test file filtering logic"""
        try:
            with patch('rag_file_monitor.file_monitor.EmbeddingManager'), \
                 patch('rag_file_monitor.file_monitor.TextExtractor'):
                monitor = FileMonitor(sample_config)
                
                # Test exclude patterns
                assert monitor.should_ignore_file("/path/to/file.tmp") == True
                assert monitor.should_ignore_file("/path/to/file.log") == True
                assert monitor.should_ignore_file("/path/to/.git/config") == True
                assert monitor.should_ignore_file("/path/to/file.txt") == False
        except ImportError:
            pytest.skip("Required dependencies not available")

    def test_is_supported_file_type(self, sample_config):
        """Test file type support checking"""
        try:
            with patch('rag_file_monitor.file_monitor.EmbeddingManager'), \
                 patch('rag_file_monitor.file_monitor.TextExtractor'):
                monitor = FileMonitor(sample_config)
                
                supported_extensions = {".txt", ".pdf", ".docx"}
                
                assert monitor.is_supported_file_type("/path/to/file.txt", supported_extensions) == True
                assert monitor.is_supported_file_type("/path/to/file.pdf", supported_extensions) == True
                assert monitor.is_supported_file_type("/path/to/file.xlsx", supported_extensions) == False
                assert monitor.is_supported_file_type("/path/to/file.unknown", supported_extensions) == False
        except ImportError:
            pytest.skip("Required dependencies not available")


class TestFileMonitorHandler:
    """Test cases for FileMonitorHandler class"""

    def test_handler_init(self):
        """Test FileMonitorHandler initialization"""
        mock_embedding_manager = Mock()
        mock_text_extractor = Mock()
        file_extensions = {".txt", ".pdf"}
        exclude_patterns = ["*.tmp"]
        max_file_size = 5
        delete_embeddings = False
        directory_path = "/test/dir"
        directory_config = {"ignore_extensions": [], "max_filesize": 0}
        
        handler = FileMonitorHandler(
            embedding_manager=mock_embedding_manager,
            text_extractor=mock_text_extractor,
            file_extensions=file_extensions,
            exclude_patterns=exclude_patterns,
            max_file_size=max_file_size,
            delete_embeddings_on_deletion=delete_embeddings,
            directory_path=directory_path,
            directory_config=directory_config
        )
        
        assert handler.embedding_manager == mock_embedding_manager
        assert handler.text_extractor == mock_text_extractor
        assert handler.file_extensions == file_extensions
        assert handler.exclude_patterns == exclude_patterns
        assert handler.max_file_size == max_file_size
        assert handler.delete_embeddings_on_deletion == delete_embeddings
        assert handler.directory_path == directory_path
        assert handler.directory_config == directory_config

    def test_get_effective_extensions_for_file(self):
        """Test effective extensions calculation for specific files"""
        mock_embedding_manager = Mock()
        mock_text_extractor = Mock()
        file_extensions = {".txt", ".pdf", ".xlsx"}
        directory_config = {"ignore_extensions": [".xlsx"], "max_filesize": 0}
        
        handler = FileMonitorHandler(
            embedding_manager=mock_embedding_manager,
            text_extractor=mock_text_extractor,
            file_extensions=file_extensions,
            exclude_patterns=[],
            max_file_size=5,
            delete_embeddings_on_deletion=False,
            directory_path="/test/dir",
            directory_config=directory_config
        )
        
        # Should return effective extensions (global minus ignored)
        effective = handler.get_effective_extensions_for_file("/test/dir/file.txt")
        expected = {".txt", ".pdf"}  # .xlsx is ignored
        assert effective == expected
        
        # Test with no directory config
        handler.directory_config = None
        effective = handler.get_effective_extensions_for_file("/test/dir/file.txt")
        assert effective == file_extensions


def test_config_validation():
    """Test configuration validation"""
    # Test with missing required sections
    invalid_configs = [
        {},  # Empty config
        {"directories": {}},  # Missing other required sections
    ]
    
    for config in invalid_configs:
        with pytest.raises((KeyError, ImportError, Exception)):
            with patch('rag_file_monitor.file_monitor.EmbeddingManager'), \
                 patch('rag_file_monitor.file_monitor.TextExtractor'):
                FileMonitor(config)


if __name__ == "__main__":
    pytest.main([__file__])
