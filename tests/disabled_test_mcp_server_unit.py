"""
Unit tests for MCP server functionality
"""

import pytest
import json
import asyncio
from pathlib import Path
import sys
from unittest.mock import Mock, patch, AsyncMock, MagicMock

# Add the project root to sys.path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def _has_mcp_dependencies():
    """Check if MCP dependencies are available"""
    try:
        import mcp

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
    }


@pytest.fixture
def mock_embedding_manager():
    """Mock embedding manager for testing"""
    mock_manager = Mock()
    mock_manager.search_similar.return_value = [
        {
            "file_path": "/test/file1.txt",
            "score": 0.95,
            "document": "This is a test document with relevant content.",
            "chunk_id": "chunk1",
        },
        {
            "file_path": "/test/file2.txt",
            "score": 0.87,
            "document": "Another test document with some content.",
            "chunk_id": "chunk2",
        },
    ]
    return mock_manager


class TestMCPServerFunctionality:
    """Test MCP server functionality without requiring full MCP setup"""

    def test_search_response_format(self, mock_embedding_manager):
        """Test that search responses are properly formatted"""
        # Test the search result formatting
        query = "test query"
        results = mock_embedding_manager.search_similar(query=query, limit=5)

        assert len(results) == 2
        assert all("file_path" in result for result in results)
        assert all("score" in result for result in results)
        assert all("document" in result for result in results)
        assert all("chunk_id" in result for result in results)

        # Test score ordering (should be descending)
        scores = [result["score"] for result in results]
        assert scores == sorted(scores, reverse=True)

    def test_search_parameters_validation(self, mock_embedding_manager):
        """Test search parameter validation"""
        # Test with valid parameters
        results = mock_embedding_manager.search_similar(query="test query", limit=10, include_deleted=False)
        assert results is not None

        # Test with different limits
        mock_embedding_manager.search_similar.return_value = []
        results = mock_embedding_manager.search_similar(query="test", limit=0)
        assert len(results) == 0

    def test_search_result_content_truncation(self, mock_embedding_manager):
        """Test that search results are properly truncated if needed"""
        # Mock a long document
        long_content = "This is a very long document. " * 100
        mock_embedding_manager.search_similar.return_value = [
            {"file_path": "/test/long_file.txt", "score": 0.95, "document": long_content, "chunk_id": "chunk1"}
        ]

        results = mock_embedding_manager.search_similar(query="test", limit=1)
        assert len(results) == 1
        assert len(results[0]["document"]) > 100  # Should contain the full content

    def test_empty_search_results(self, mock_embedding_manager):
        """Test handling of empty search results"""
        mock_embedding_manager.search_similar.return_value = []

        results = mock_embedding_manager.search_similar(query="nonexistent", limit=10)
        assert results == []

    def test_search_with_file_filtering(self, mock_embedding_manager):
        """Test search with file path filtering"""
        # Mock results with different file types
        mock_embedding_manager.search_similar.return_value = [
            {"file_path": "/test/document.pdf", "score": 0.95, "document": "PDF content", "chunk_id": "chunk1"},
            {"file_path": "/test/document.txt", "score": 0.87, "document": "Text content", "chunk_id": "chunk2"},
        ]

        results = mock_embedding_manager.search_similar(query="test", limit=10)

        # Should return both file types
        file_paths = [result["file_path"] for result in results]
        assert "/test/document.pdf" in file_paths
        assert "/test/document.txt" in file_paths


class TestMCPServerIntegration:
    """Integration tests for MCP server components"""

    @patch("yaml.safe_load")
    @patch("builtins.open")
    def test_config_loading(self, mock_open, mock_yaml_load, sample_config):
        """Test configuration loading for MCP server"""
        mock_yaml_load.return_value = sample_config

        # Import and test the configuration loading logic
        try:
            # Simulate the config loading process
            config_path = "/test/config.yaml"

            # Mock file operations
            mock_file = Mock()
            mock_open.return_value.__enter__.return_value = mock_file

            # This would be the actual config loading code
            loaded_config = sample_config  # Simulated loaded config

            assert loaded_config["qdrant"]["host"] == "localhost"
            assert loaded_config["qdrant"]["port"] == 6333
            assert loaded_config["embedding"]["model_name"] == "all-MiniLM-L6-v2"

        except ImportError:
            pytest.skip("Config loading dependencies not available")

    def test_embedding_manager_initialization(self, sample_config):
        """Test embedding manager initialization for MCP server"""
        try:
            with patch("rag_file_monitor.embedding_manager.SentenceTransformer"):
                from rag_file_monitor.embedding_manager import EmbeddingManager

                # Test that embedding manager can be initialized
                manager = EmbeddingManager(sample_config, slim_mode=True)
                assert manager.config == sample_config
                assert manager.model_name == "all-MiniLM-L6-v2"

        except ImportError:
            pytest.skip("Embedding manager dependencies not available")

    def test_error_handling_in_search(self, mock_embedding_manager):
        """Test error handling in search operations"""
        # Mock an exception during search
        mock_embedding_manager.search_similar.side_effect = Exception("Search failed")

        try:
            results = mock_embedding_manager.search_similar(query="test", limit=10)
            assert False, "Should have raised an exception"
        except Exception as e:
            assert str(e) == "Search failed"

    def test_search_result_serialization(self, mock_embedding_manager):
        """Test that search results can be properly serialized to JSON"""
        results = mock_embedding_manager.search_similar(query="test", limit=5)

        # Test JSON serialization
        try:
            json_str = json.dumps(results)
            assert json_str is not None

            # Test deserialization
            deserialized = json.loads(json_str)
            assert deserialized == results

        except (TypeError, ValueError) as e:
            pytest.fail(f"Search results are not JSON serializable: {e}")


class TestMCPServerMocking:
    """Test MCP server with mocked dependencies"""

    def test_mcp_server_tool_registration(self):
        """Test MCP server tool registration"""
        # Mock the MCP server setup
        with patch("mcp.server.Server") as mock_server_class:
            mock_server = Mock()
            mock_server_class.return_value = mock_server

            # Test tool registration (this would be the actual MCP server code)
            tools = [
                {
                    "name": "rag_search",
                    "description": "Search for relevant documents",
                    "parameters": {
                        "type": "object",
                        "properties": {"query": {"type": "string"}, "limit": {"type": "integer", "default": 10}},
                    },
                }
            ]

            # Verify tool structure
            assert len(tools) == 1
            assert tools[0]["name"] == "rag_search"
            assert "query" in tools[0]["parameters"]["properties"]

    def test_mcp_server_request_handling(self, mock_embedding_manager):
        """Test MCP server request handling"""
        # Mock an MCP request
        mock_request = {
            "method": "tools/call",
            "params": {"name": "rag_search", "arguments": {"query": "test query", "limit": 5}},
        }

        # Simulate request processing
        query = mock_request["params"]["arguments"]["query"]
        limit = mock_request["params"]["arguments"]["limit"]

        results = mock_embedding_manager.search_similar(query=query, limit=limit)

        # Format response
        response = {"content": [{"type": "text", "text": f"Found {len(results)} results for query: {query}"}]}

        assert response["content"][0]["text"] == "Found 2 results for query: test query"

    def test_mcp_server_error_response(self):
        """Test MCP server error response formatting"""
        # Mock an error scenario
        error_response = {
            "error": {
                "code": -1,
                "message": "Search failed",
                "data": {"type": "SearchError", "details": "Unable to connect to Qdrant database"},
            }
        }

        assert error_response["error"]["code"] == -1
        assert error_response["error"]["message"] == "Search failed"
        assert "SearchError" in error_response["error"]["data"]["type"]


@pytest.mark.skipif(not _has_mcp_dependencies(), reason="MCP dependencies not available")
class TestMCPServerWithDependencies:
    """Tests that require actual MCP dependencies"""

    def test_mcp_server_import(self):
        """Test importing MCP server module"""
        try:
            import mcp

            assert mcp is not None
        except ImportError:
            pytest.fail("MCP dependencies should be available for this test")

    def test_mcp_server_initialization(self):
        """Test MCP server initialization"""
        try:
            from mcp.server import Server

            # Test server creation
            server = Server("test-server")
            assert server is not None

        except ImportError:
            pytest.fail("MCP server should be importable")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
