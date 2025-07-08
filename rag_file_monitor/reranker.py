"""
Re-ranking module for improving RAG search results

This module implements a two-stage retrieval system:
1. Initial retrieval using embedding similarity (current system)
2. Re-ranking using cross-encoder models for better relevance scoring
"""

import logging
from typing import List, Dict, Optional, Tuple
from datetime import datetime

try:
    from sentence_transformers import CrossEncoder
except ImportError:
    CrossEncoder = None


class ReRanker:
    """Re-rank search results using cross-encoder models for improved relevance"""

    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Re-ranker configuration
        reranker_config = config.get("reranker", {})
        self.enabled = reranker_config.get("enabled", False)
        self.model_name = reranker_config.get("model_name", "cross-encoder/ms-marco-MiniLM-L-6-v2")
        self.top_k_retrieve = reranker_config.get("top_k_retrieve", 50)  # Retrieve more, re-rank to fewer
        self.score_threshold = reranker_config.get("score_threshold", 0.0)

        # Model management
        self.cross_encoder = None
        self.model_last_used = None
        self.unload_after_minutes = reranker_config.get("unload_after_idle_minutes", 15)

        if self.enabled and CrossEncoder is None:
            self.logger.error("sentence-transformers with CrossEncoder support required for re-ranking")
            self.enabled = False

        if self.enabled:
            self.logger.info(f"Re-ranker configured: {self.model_name} (enabled: {self.enabled})")

    def _get_cross_encoder(self):
        """Lazy loading of cross-encoder model"""
        if not self.enabled:
            return None

        if self.cross_encoder is None:
            self.logger.info(f"Loading cross-encoder model: {self.model_name}")
            cache_folder = self.config.get("embedding", {}).get("cache_folder", "/tmp/rag-models-cache")
            self.cross_encoder = CrossEncoder(self.model_name, cache_folder=cache_folder)

        self.model_last_used = datetime.now()
        return self.cross_encoder

    def rerank_results(self, query: str, results: List[Dict], target_limit: int) -> List[Dict]:
        """
        Re-rank search results using cross-encoder for better relevance

        Args:
            query: The search query
            results: List of initial search results from embedding similarity
            target_limit: Final number of results to return after re-ranking

        Returns:
            Re-ranked and filtered results
        """
        if not self.enabled or not results:
            return results[:target_limit]

        try:
            cross_encoder = self._get_cross_encoder()
            if cross_encoder is None:
                return results[:target_limit]

            # Prepare query-document pairs for cross-encoder
            pairs = []
            for result in results:
                document = result.get("document", "")
                pairs.append([query, document])

            # Get relevance scores from cross-encoder
            self.logger.debug(f"Re-ranking {len(pairs)} results with cross-encoder")
            scores = cross_encoder.predict(pairs)

            # Combine original results with new scores
            enhanced_results = []
            for i, result in enumerate(results):
                enhanced_result = result.copy()
                enhanced_result["rerank_score"] = float(scores[i])
                enhanced_result["original_score"] = result.get("score", 0.0)
                enhanced_results.append(enhanced_result)

            # Sort by re-ranking score (descending)
            enhanced_results.sort(key=lambda x: x["rerank_score"], reverse=True)

            # Apply score threshold if configured
            if self.score_threshold > 0:
                enhanced_results = [r for r in enhanced_results if r["rerank_score"] >= self.score_threshold]

            # Return top results up to target limit
            final_results = enhanced_results[:target_limit]

            self.logger.info(
                f"Re-ranked {len(results)} -> {len(final_results)} results " f"(threshold: {self.score_threshold})"
            )

            return final_results

        except Exception as e:
            self.logger.error(f"Error during re-ranking: {e}")
            # Fallback to original results on error
            return results[:target_limit]

    def calculate_retrieval_limit(self, requested_limit: int) -> int:
        """
        Calculate how many results to retrieve initially before re-ranking

        Args:
            requested_limit: Final number of results wanted

        Returns:
            Number of results to retrieve for re-ranking
        """
        if not self.enabled:
            return requested_limit

        # Retrieve more results than needed to improve re-ranking quality
        return min(max(requested_limit * 4, self.top_k_retrieve), 100)

    def unload_model(self):
        """Unload cross-encoder model to free memory"""
        if self.cross_encoder is not None:
            self.logger.info("Unloading cross-encoder model")
            del self.cross_encoder
            self.cross_encoder = None
            self.model_last_used = None
            import gc

            gc.collect()

    def check_and_unload_idle_model(self) -> bool:
        """Check if model has been idle and unload if necessary"""
        if (
            self.cross_encoder is not None
            and self.model_last_used is not None
            and (datetime.now() - self.model_last_used).total_seconds() > self.unload_after_minutes * 60
        ):

            self.logger.info(f"Unloading idle cross-encoder after {self.unload_after_minutes} minutes")
            self.unload_model()
            return True
        return False

    def preload_model(self):
        """Pre-load the cross-encoder model for faster response times"""
        if not self.enabled:
            self.logger.info("Re-ranker disabled, skipping model pre-loading")
            return

        try:
            self.logger.info(f"Pre-loading cross-encoder model: {self.model_name}")
            _ = self._get_cross_encoder()
            self.logger.info("Cross-encoder model pre-loaded successfully")
        except Exception as e:
            self.logger.error(f"Failed to pre-load cross-encoder model: {e}")
            raise
