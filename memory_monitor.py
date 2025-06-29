#!/usr/bin/env python3
"""
Memory monitoring tool for RAG system
"""

import psutil
import os
import yaml
import time
from rag_file_monitor.embedding_manager import EmbeddingManager


def get_process_memory_mb():
    """Get current process memory usage in MB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024


def main():
    """Monitor memory usage and demonstrate cleanup"""
    print("RAG Memory Monitor")
    print("=" * 50)

    # Load config
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    print(f"Initial memory usage: {get_process_memory_mb():.1f} MB")

    # Initialize embedding manager
    print("\nInitializing embedding manager...")
    em = EmbeddingManager(config, slim_mode=True)  # Use slim mode to avoid loading all hashes
    print(f"Memory after initialization: {get_process_memory_mb():.1f} MB")

    # Get memory stats
    stats = em.get_memory_stats()
    print("\nMemory Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")

    # Test embedding generation (this loads the model)
    print("\nGenerating test embedding...")
    embeddings = em.generate_embeddings(["This is a test document for memory monitoring"])
    print(f"Memory after loading model: {get_process_memory_mb():.1f} MB")

    # Show updated stats
    stats = em.get_memory_stats()
    print("\nUpdated Memory Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")

    # Manually unload model
    print("\nManually unloading model...")
    with em.model_lock:
        em.unload_embedding_model()
    print(f"Memory after unloading model: {get_process_memory_mb():.1f} MB")

    # Final stats
    stats = em.get_memory_stats()
    print("\nFinal Memory Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")

    print("\nMemory monitoring complete!")


if __name__ == "__main__":
    main()
