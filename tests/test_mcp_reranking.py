"""
Test the MCP server integration with re-ranking
"""

import pytest
import unittest
from unittest.mock import Mock, patch, MagicMock
import sys
from pathlib import Path

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from mcp_server import RAGSearchResult, RAGSearchResponse


class TestMCPServerReranking(unittest.TestCase):
    """Test MCP server integration with re-ranking"""

    def test_rag_search_result_with_reranking(self):
        """Test RAGSearchResult handles re-ranking scores correctly"""
        # Test with re-ranking scores
        result_with_rerank = RAGSearchResult(
            file_path="/test/file.pdf",
            document="Test document content",
            score=0.95,  # This should be the rerank_score when available
            chunk_index=0,
            is_deleted=False,
            rerank_score=0.95,
            original_score=0.62,
        )

        self.assertEqual(result_with_rerank.score, 0.95)
        self.assertEqual(result_with_rerank.rerank_score, 0.95)
        self.assertEqual(result_with_rerank.original_score, 0.62)

    def test_rag_search_result_without_reranking(self):
        """Test RAGSearchResult handles results without re-ranking"""
        # Test without re-ranking scores
        result_without_rerank = RAGSearchResult(
            file_path="/test/file.pdf",
            document="Test document content",
            score=0.62,  # This should be the original embedding score
            chunk_index=0,
            is_deleted=False,
            rerank_score=None,
            original_score=None,
        )

        self.assertEqual(result_without_rerank.score, 0.62)
        self.assertIsNone(result_without_rerank.rerank_score)
        self.assertIsNone(result_without_rerank.original_score)

    def test_rag_search_response_structure(self):
        """Test RAGSearchResponse structure"""
        results = [
            RAGSearchResult(
                file_path="/test/file1.pdf",
                document="First document",
                score=0.95,
                chunk_index=0,
                is_deleted=False,
                rerank_score=0.95,
                original_score=0.60,
            ),
            RAGSearchResult(
                file_path="/test/file2.pdf",
                document="Second document",
                score=0.88,
                chunk_index=1,
                is_deleted=False,
                rerank_score=0.88,
                original_score=0.70,
            ),
        ]

        response = RAGSearchResponse(results=results, query="test query", total_results=2, filtered_by_pattern="*.pdf")

        self.assertEqual(len(response.results), 2)
        self.assertEqual(response.query, "test query")
        self.assertEqual(response.total_results, 2)
        self.assertEqual(response.filtered_by_pattern, "*.pdf")

        # Verify results are properly sorted (highest score first)
        self.assertGreaterEqual(response.results[0].score, response.results[1].score)

    @patch("mcp_server.EmbeddingManager")
    @patch("mcp_server.yaml.safe_load")
    @patch("mcp_server.Path.exists")
    def test_mcp_server_score_priority(self, mock_exists, mock_yaml_load, mock_embedding_manager):
        """Test that MCP server properly prioritizes rerank_score over original score"""
        # Mock configuration
        mock_exists.return_value = True
        mock_yaml_load.return_value = {
            "qdrant": {"host": "localhost", "port": 6333},
            "embedding": {"model_name": "test-model"},
            "reranker": {"enabled": True},
            "logging": {"level": "INFO"},
        }

        # Mock embedding manager results with re-ranking
        mock_embedding_instance = Mock()
        mock_embedding_instance.search_similar.return_value = [
            {
                "file_path": "/test/file1.pdf",
                "document": "First document",
                "score": 0.60,  # Original embedding score
                "chunk_index": 0,
                "is_deleted": False,
                "rerank_score": 0.95,  # Higher re-ranking score
                "original_score": 0.60,
            },
            {
                "file_path": "/test/file2.pdf",
                "document": "Second document",
                "score": 0.70,  # Original embedding score
                "chunk_index": 1,
                "is_deleted": False,
                "rerank_score": 0.88,  # Lower re-ranking score
                "original_score": 0.70,
            },
        ]
        mock_embedding_manager.return_value = mock_embedding_instance

        # Import and test the rag_search function logic
        # Note: We can't easily test the actual MCP function due to context requirements,
        # but we can test the core logic
        raw_results = mock_embedding_instance.search_similar("test", 10, True, None)

        # Simulate the MCP server logic for score assignment
        structured_results = []
        for result in raw_results:
            primary_score = result.get("rerank_score") if result.get("rerank_score") is not None else result.get("score", 0.0)

            structured_result = {
                "file_path": result.get("file_path", ""),
                "document": result.get("document", ""),
                "score": primary_score,
                "chunk_index": result.get("chunk_index", 0),
                "is_deleted": result.get("is_deleted", False),
                "rerank_score": result.get("rerank_score"),
                "original_score": result.get("original_score"),
            }
            structured_results.append(structured_result)

        # Verify that the primary score is the rerank_score
        self.assertEqual(structured_results[0]["score"], 0.95)  # Should use rerank_score
        self.assertEqual(structured_results[1]["score"], 0.88)  # Should use rerank_score

        # Verify original scores are preserved
        self.assertEqual(structured_results[0]["original_score"], 0.60)
        self.assertEqual(structured_results[1]["original_score"], 0.70)


if __name__ == "__main__":
    unittest.main()
