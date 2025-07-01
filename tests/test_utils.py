"""
Test utilities and fixtures for the test suite
"""

import tempfile
import shutil
from pathlib import Path
import yaml


def create_test_config(temp_dir=None):
    """Create a test configuration"""
    if temp_dir is None:
        temp_dir = "/tmp/test"

    return {
        "directories": {str(temp_dir): {"ignore_extensions": [], "max_filesize": 0}},
        "file_extensions": [".txt", ".md", ".pdf", ".docx", ".html"],
        "qdrant": {
            "host": "localhost",
            "port": 6333,
            "collection_name": "test_collection",
            "vector_size": 384,
            "vector_name": "test-vector",
        },
        "embedding": {"model_name": "all-MiniLM-L6-v2", "chunk_size": 512, "chunk_overlap": 100},
        "processing": {
            "max_file_size_mb": 5,
            "exclude_patterns": ["*.tmp", "*.log", ".git/*", "__pycache__/*"],
            "delete_embeddings_on_file_deletion": False,
        },
        "memory": {
            "chunk_batch_size": 10,
            "hash_loading_batch_size": 100,
            "unload_model_after_idle_minutes": 30,
            "force_gc_after_operations": 100,
        },
        "logging": {"level": "WARNING", "file": "test.log", "mcp_logfile": "/tmp/mcp_test.log"},
    }


def create_test_files(directory):
    """Create test files in the given directory"""
    dir_path = Path(directory)
    dir_path.mkdir(parents=True, exist_ok=True)

    files = {}

    # Text file
    txt_file = dir_path / "test.txt"
    txt_file.write_text("This is a test text file with some content.")
    files["txt"] = str(txt_file)

    # Markdown file
    md_file = dir_path / "test.md"
    md_file.write_text("# Test Markdown\n\nThis is markdown content with **bold** text.")
    files["md"] = str(md_file)

    # HTML file
    html_file = dir_path / "test.html"
    html_file.write_text("<html><body><h1>Test HTML</h1><p>HTML content</p></body></html>")
    files["html"] = str(html_file)

    # Files that should be ignored
    tmp_file = dir_path / "temp.tmp"
    tmp_file.write_text("Temporary file content")
    files["tmp"] = str(tmp_file)

    log_file = dir_path / "test.log"
    log_file.write_text("Log file content")
    files["log"] = str(log_file)

    # Python cache directory (should be ignored)
    cache_dir = dir_path / "__pycache__"
    cache_dir.mkdir(exist_ok=True)
    cache_file = cache_dir / "test.pyc"
    cache_file.write_bytes(b"Python bytecode")
    files["pyc"] = str(cache_file)

    return files


def cleanup_test_files(directory):
    """Clean up test files and directories"""
    if Path(directory).exists():
        shutil.rmtree(directory)


class MockQdrantClient:
    """Mock Qdrant client for testing"""

    def __init__(self, host="localhost", port=6333):
        self.host = host
        self.port = port
        self.collections = {}
        self.points = {}

    def get_collections(self):
        """Mock get_collections"""
        return {"collections": [{"name": name} for name in self.collections.keys()]}

    def create_collection(self, collection_name, vectors_config):
        """Mock create_collection"""
        self.collections[collection_name] = {"vectors_config": vectors_config, "points": []}

    def get_collection(self, collection_name):
        """Mock get_collection"""
        if collection_name in self.collections:
            return {"name": collection_name}
        else:
            raise Exception(f"Collection {collection_name} not found")

    def upsert(self, collection_name, points):
        """Mock upsert"""
        if collection_name not in self.collections:
            raise Exception(f"Collection {collection_name} not found")

        self.collections[collection_name]["points"].extend(points)
        return {"operation_id": "mock-operation-id"}

    def search(self, collection_name, query_vector, limit=10, query_filter=None):
        """Mock search"""
        if collection_name not in self.collections:
            return []

        # Return mock search results
        return [
            {
                "id": f"doc_{i}",
                "score": 0.9 - i * 0.1,
                "payload": {
                    "file_path": f"/test/file_{i}.txt",
                    "document": f"This is test document {i} content.",
                    "chunk_id": f"chunk_{i}",
                },
            }
            for i in range(min(limit, 3))
        ]

    def delete(self, collection_name, points_selector):
        """Mock delete"""
        return {"operation_id": "mock-delete-operation-id"}


class MockSentenceTransformer:
    """Mock SentenceTransformer for testing"""

    def __init__(self, model_name):
        self.model_name = model_name

    def encode(self, texts, convert_to_tensor=False):
        """Mock encode method"""
        if isinstance(texts, str):
            texts = [texts]

        # Return mock embeddings (384 dimensions for all-MiniLM-L6-v2)
        embeddings = []
        for i, text in enumerate(texts):
            # Create a simple mock embedding based on text length and content
            embedding = [0.1 * (j + len(text) + i) % 2 - 1 for j in range(384)]
            embeddings.append(embedding)

        return embeddings
