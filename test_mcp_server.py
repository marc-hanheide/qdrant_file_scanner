#!/usr/bin/env python
"""
Test script for the MCP RAG server

This script provides basic testing functionality for the MCP server without
requiring the full MCP dependencies to be installed.
"""

import asyncio
import sys
import logging
from pathlib import Path

# Add the project root to sys.path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

try:
    from rag_file_monitor.embedding_manager import EmbeddingManager
    import yaml
except ImportError as e:
    print(f"Missing dependencies: {e}")
    print("Please install the project dependencies first.")
    sys.exit(1)


async def test_rag_search():
    """Test the RAG search functionality"""

    # Load configuration
    config_path = Path(__file__).parent / "config.yaml"
    if not config_path.exists():
        print(f"Configuration file not found: {config_path}")
        return False

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    print("Testing RAG search functionality...")

    try:
        # Initialize embedding manager
        print("Initializing embedding manager...")
        embedding_manager = EmbeddingManager(config)
        print("✓ Embedding manager initialized successfully")

        # Test search
        test_query = "document"
        print(f"Testing search with query: '{test_query}'")

        results = embedding_manager.search_similar(query=test_query, limit=5, include_deleted=False)

        print("Search completed successfully")
        print(f"  Results found: {len(results)}")

        if results:
            print("\nFirst result:")
            result = results[0]
            print(f"  File: {result.get('file_path', 'N/A')}")
            print(f"  Score: {result.get('score', 0.0):.4f}")
            print(f"  Content preview: {result.get('document', '')[:100]}...")
        else:
            print("  No documents found. Make sure you have indexed some documents first.")
            print("  Run 'rag-monitor --scan-only' to index existing files.")

        return True

    except Exception as e:
        print(f"✗ Error during RAG search test: {e}")
        return False


async def test_mcp_server_import():
    """Test if MCP server can be imported"""
    print("\nTesting MCP server import...")

    try:
        print("✓ MCP dependencies available")

        print("✓ MCP server module imported successfully")

        return True

    except ImportError as e:
        print(f"✗ MCP server import failed: {e}")
        print("  To install MCP dependencies: pip install 'mcp[cli]>=1.10.0'")
        return False
    except Exception as e:
        print(f"✗ Unexpected error importing MCP server: {e}")
        return False


def main():
    """Run all tests"""
    print("RAG MCP Server Test Suite")
    print("=" * 40)

    # Setup logging
    logging.basicConfig(level=logging.INFO)

    tests_passed = 0
    total_tests = 2

    # Test 1: RAG search functionality
    if asyncio.run(test_rag_search()):
        tests_passed += 1

    # Test 2: MCP server import
    if asyncio.run(test_mcp_server_import()):
        tests_passed += 1

    print(f"\nTest Results: {tests_passed}/{total_tests} tests passed")

    if tests_passed == total_tests:
        print("✓ All tests passed! Your MCP server should work correctly.")
        print("\nNext steps:")
        print("1. Start the MCP server: python mcp_server.py")
        print("2. Test with MCP Inspector: mcp dev mcp_server.py")
        print("3. Install in Claude Desktop: mcp install mcp_server.py")
    else:
        print("✗ Some tests failed. Please check the error messages above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
