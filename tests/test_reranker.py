"""
Tests for the re-ranking functionality
"""

import pytest
import unittest
from unittest.mock import Mock, patch, MagicMock
from rag_file_monitor.reranker import ReRanker


class TestReRanker(unittest.TestCase):
    """Test cases for ReRanker class"""

    def setUp(self):
        """Set up test configuration"""
        self.config_disabled = {
            "reranker": {
                "enabled": False,
                "model_name": "cross-encoder/ms-marco-MiniLM-L-6-v2",
                "top_k_retrieve": 50,
                "score_threshold": 0.0,
                "unload_after_idle_minutes": 15
            },
            "embedding": {
                "cache_folder": "/tmp/test-models"
            }
        }
        
        self.config_enabled = {
            "reranker": {
                "enabled": True,
                "model_name": "cross-encoder/ms-marco-MiniLM-L-6-v2",
                "top_k_retrieve": 50,
                "score_threshold": 0.5,
                "unload_after_idle_minutes": 15
            },
            "embedding": {
                "cache_folder": "/tmp/test-models"
            }
        }

    def test_reranker_disabled(self):
        """Test re-ranker when disabled"""
        reranker = ReRanker(self.config_disabled)
        
        self.assertFalse(reranker.enabled)
        
        # Test that disabled re-ranker returns original results
        results = [
            {"document": "test doc 1", "score": 0.8},
            {"document": "test doc 2", "score": 0.6}
        ]
        
        reranked = reranker.rerank_results("test query", results, 2)
        self.assertEqual(reranked, results)

    def test_calculate_retrieval_limit_disabled(self):
        """Test retrieval limit calculation when disabled"""
        reranker = ReRanker(self.config_disabled)
        
        # When disabled, should return requested limit
        self.assertEqual(reranker.calculate_retrieval_limit(10), 10)
        self.assertEqual(reranker.calculate_retrieval_limit(5), 5)

    def test_calculate_retrieval_limit_enabled(self):
        """Test retrieval limit calculation when enabled"""
        reranker = ReRanker(self.config_enabled)
        
        # When enabled, should return higher limit for re-ranking
        self.assertEqual(reranker.calculate_retrieval_limit(10), 50)  # Uses top_k_retrieve
        self.assertEqual(reranker.calculate_retrieval_limit(20), 60)  # 20 * 3 = 60
        self.assertEqual(reranker.calculate_retrieval_limit(50), 100)  # max(50*3, 50) = 100 (capped)

    @patch('rag_file_monitor.reranker.CrossEncoder')
    def test_reranker_enabled_with_mock(self, mock_cross_encoder_class):
        """Test re-ranker when enabled with mocked CrossEncoder"""
        # Mock the CrossEncoder instance
        mock_model = Mock()
        mock_model.predict.return_value = [0.9, 0.3, 0.7]  # Re-ranking scores
        mock_cross_encoder_class.return_value = mock_model
        
        reranker = ReRanker(self.config_enabled)
        reranker.enabled = True  # Force enable for test
        
        results = [
            {"document": "relevant document", "score": 0.6},
            {"document": "less relevant", "score": 0.8},
            {"document": "moderately relevant", "score": 0.7}
        ]
        
        reranked = reranker.rerank_results("test query", results, 3)
        
        # Should be re-ordered by re-ranking scores: 0.9, 0.7, 0.3
        # But filtered by threshold (0.5), so only first two
        self.assertEqual(len(reranked), 2)
        self.assertEqual(reranked[0]["document"], "relevant document")
        self.assertEqual(reranked[1]["document"], "moderately relevant")
        
        # Check scores are properly assigned
        self.assertEqual(reranked[0]["rerank_score"], 0.9)
        self.assertEqual(reranked[0]["original_score"], 0.6)

    def test_empty_results(self):
        """Test re-ranker with empty results"""
        reranker = ReRanker(self.config_enabled)
        
        reranked = reranker.rerank_results("test query", [], 10)
        self.assertEqual(reranked, [])

    def test_unload_model(self):
        """Test model unloading"""
        reranker = ReRanker(self.config_enabled)
        
        # Mock a loaded model
        reranker.cross_encoder = Mock()
        reranker.model_last_used = Mock()
        
        reranker.unload_model()
        
        self.assertIsNone(reranker.cross_encoder)
        self.assertIsNone(reranker.model_last_used)


if __name__ == "__main__":
    unittest.main()
