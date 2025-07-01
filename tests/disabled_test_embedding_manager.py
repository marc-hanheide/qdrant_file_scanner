"""
Unit tests for embedding manager module - Simple tests without complex mocking
"""

import pytest
import tempfile
import os
from pathlib import Path
import sys

# Add the project root to sys.path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from rag_file_monitor.embedding_manager import EmbeddingManager


def _has_sentence_transformers_support():
    """Check if sentence-transformers is available"""
    try:
        import sentence_transformers

        return True
    except ImportError:
        return False


@pytest.fixture
def sample_config():
    """Sample configuration for testing"""
    return {
        "qdrant": {
            "host": "localhost",
            "port": 6333,
            "collection_name": "test_collection",
            "vector_size": 384,
            "vector_name": "test-vector",
        },
        "embedding": {"model_name": "all-MiniLM-L6-v2", "chunk_size": 512, "chunk_overlap": 100},
        "processing": {"max_file_size_mb": 5, "exclude_patterns": ["*.tmp", "*.log"]},
        "memory": {
            "chunk_batch_size": 10,
            "hash_loading_batch_size": 100,
            "unload_model_after_idle_minutes": 30,
            "force_gc_after_operations": 100,
        },
    }


class TestEmbeddingManager:
    """Test cases for EmbeddingManager class - focused on simple, non-mocking tests"""

    @pytest.mark.skipif(not _has_sentence_transformers_support(), reason="sentence-transformers not available")
    def test_init_with_valid_config(self, sample_config):
        """Test initialization with valid configuration"""
        try:
            manager = EmbeddingManager(sample_config, slim_mode=True)
            assert manager.config == sample_config
            assert manager.model_name == "all-MiniLM-L6-v2"
            assert manager.embedding_model is None  # Lazy loading
            assert manager.config["embedding"]["chunk_size"] == 512
            assert manager.config["embedding"]["chunk_overlap"] == 100
        except ImportError:
            pytest.skip("Required dependencies not available")

    @pytest.mark.skipif(not _has_sentence_transformers_support(), reason="sentence-transformers not available")
    def test_chunk_text_simple_cases(self, sample_config):
        """Test text chunking functionality with simple cases"""
        try:
            manager = EmbeddingManager(sample_config, slim_mode=True)

            # Test empty text
            chunks = manager.chunk_text("")
            assert chunks == [""]

            # Test short text (should return as single chunk)
            short_text = "This is a short text."
            chunks = manager.chunk_text(short_text)
            assert len(chunks) == 1
            assert chunks[0] == short_text

            # Test text that's exactly the chunk size
            exact_size_text = "A" * sample_config["embedding"]["chunk_size"]
            chunks = manager.chunk_text(exact_size_text)
            assert len(chunks) == 1
            assert chunks[0] == exact_size_text

        except ImportError:
            pytest.skip("Required dependencies not available")

    @pytest.mark.skipif(not _has_sentence_transformers_support(), reason="sentence-transformers not available")
    def test_chunk_text_long_text(self, sample_config):
        """Test text chunking with long text"""
        try:
            manager = EmbeddingManager(sample_config, slim_mode=True)

            # Create text longer than chunk_size
            long_text = "This is a test sentence. " * 50  # Should be > 512 chars
            chunks = manager.chunk_text(long_text)

            # Should create multiple chunks
            assert len(chunks) > 1

            # Each chunk should not exceed chunk_size + chunk_overlap
            max_chunk_size = sample_config["embedding"]["chunk_size"] + sample_config["embedding"]["chunk_overlap"]
            for chunk in chunks:
                assert len(chunk) <= max_chunk_size

        except ImportError:
            pytest.skip("Required dependencies not available")

    def test_file_hash_calculation(self, sample_config):
        """Test file hash calculation - no mocking needed"""
        try:
            manager = EmbeddingManager(sample_config, slim_mode=True)

            # Create a temporary file
            test_content = "This is test content for hashing."
            with tempfile.NamedTemporaryFile(mode="w", delete=False) as tmp:
                tmp.write(test_content)
                tmp_path = tmp.name

            try:
                hash1 = manager.get_file_hash(tmp_path)
                hash2 = manager.get_file_hash(tmp_path)

                # Same file should produce same hash
                assert hash1 == hash2
                assert len(hash1) == 32  # MD5 hex digest length
                assert isinstance(hash1, str)

                # Different content should produce different hash
                with open(tmp_path, "w") as f:
                    f.write("Different content")
                hash3 = manager.get_file_hash(tmp_path)
                assert hash1 != hash3
                assert len(hash3) == 32

            finally:
                os.unlink(tmp_path)
        except ImportError:
            pytest.skip("Required dependencies not available")

    def test_file_hash_nonexistent_file(self, sample_config):
        """Test file hash calculation for non-existent file"""
        try:
            manager = EmbeddingManager(sample_config, slim_mode=True)
            hash_result = manager.get_file_hash("/nonexistent/file.txt")
            # Should return empty string for missing files
            assert hash_result == ""
        except ImportError:
            pytest.skip("Required dependencies not available")

    def test_file_hash_empty_file(self, sample_config):
        """Test file hash calculation for empty file"""
        try:
            manager = EmbeddingManager(sample_config, slim_mode=True)

            # Create an empty temporary file
            with tempfile.NamedTemporaryFile(mode="w", delete=False) as tmp:
                tmp_path = tmp.name

            try:
                hash_result = manager.get_file_hash(tmp_path)
                # Should return valid hash for empty file
                assert isinstance(hash_result, str)
                assert len(hash_result) == 32  # MD5 hex digest length

            finally:
                os.unlink(tmp_path)
        except ImportError:
            pytest.skip("Required dependencies not available")

    def test_configuration_access(self, sample_config):
        """Test that configuration values are accessible"""
        try:
            manager = EmbeddingManager(sample_config, slim_mode=True)

            # Test that config sections are accessible
            assert "qdrant" in manager.config
            assert "embedding" in manager.config
            assert "processing" in manager.config
            assert "memory" in manager.config

            # Test specific config values
            assert manager.config["qdrant"]["host"] == "localhost"
            assert manager.config["qdrant"]["port"] == 6333
            assert manager.config["embedding"]["chunk_size"] == 512
            assert manager.config["embedding"]["chunk_overlap"] == 100

        except ImportError:
            pytest.skip("Required dependencies not available")

    def test_config_validation_missing_sections(self):
        """Test configuration validation with missing sections"""
        # Test completely empty config
        with pytest.raises((KeyError, ImportError)):
            EmbeddingManager({})

        # Test config missing qdrant section
        incomplete_config = {
            "embedding": {"model_name": "test", "chunk_size": 512, "chunk_overlap": 100},
            "processing": {"max_file_size_mb": 5, "exclude_patterns": []},
            "memory": {"chunk_batch_size": 10},
        }
        with pytest.raises((KeyError, ImportError)):
            EmbeddingManager(incomplete_config)

        # Test config missing embedding section
        incomplete_config2 = {
            "qdrant": {"host": "localhost", "port": 6333, "collection_name": "test"},
            "processing": {"max_file_size_mb": 5, "exclude_patterns": []},
            "memory": {"chunk_batch_size": 10},
        }
        with pytest.raises((KeyError, ImportError)):
            EmbeddingManager(incomplete_config2)

    def test_slim_mode_initialization(self, sample_config):
        """Test that slim_mode parameter works (doesn't throw errors)"""
        try:
            # Test with slim_mode=True - should work without errors
            manager1 = EmbeddingManager(sample_config, slim_mode=True)
            assert manager1 is not None

            # Test with slim_mode=False (default behavior should still work)
            manager2 = EmbeddingManager(sample_config, slim_mode=False)
            assert manager2 is not None

        except ImportError:
            pytest.skip("Required dependencies not available")


if __name__ == "__main__":
    pytest.main([__file__])
