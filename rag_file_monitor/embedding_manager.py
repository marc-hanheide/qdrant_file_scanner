"""
Embedding generation and Qdrant database management
"""

import logging
import uuid
from typing import List, Dict, Optional
from datetime import datetime

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, VectorParams, PointStruct
except ImportError:
    QdrantClient = None
    Distance = None
    VectorParams = None
    PointStruct = None


class EmbeddingManager:
    """Manage embeddings and Qdrant vector database operations"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize embedding model
        if SentenceTransformer is None:
            raise ImportError("sentence-transformers is required")
            
        model_name = config['embedding']['model_name']
        self.logger.info(f"Loading embedding model: {model_name}")
        self.embedding_model = SentenceTransformer(model_name)
        
        # Initialize Qdrant client
        if QdrantClient is None:
            raise ImportError("qdrant-client is required")
            
        qdrant_config = config['qdrant']
        self.client = QdrantClient(
            host=qdrant_config['host'],
            port=qdrant_config['port']
        )
        
        self.collection_name = qdrant_config['collection_name']
        self.vector_size = qdrant_config['vector_size']
        
        # Ensure collection exists
        self._ensure_collection_exists()
        
        # File tracking for change detection
        self.file_hashes = {}
        self._load_file_hashes()
        
    def _ensure_collection_exists(self):
        """Ensure the Qdrant collection exists"""
        try:
            collections = self.client.get_collections()
            collection_names = [col.name for col in collections.collections]
            
            if self.collection_name not in collection_names:
                self.logger.info(f"Creating collection: {self.collection_name}")
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config={
                        "fast-all-minilm-l6-v2": VectorParams(
                            size=self.vector_size,
                            distance=Distance.COSINE
                        )
                    }
                )
            else:
                self.logger.info(f"Collection {self.collection_name} already exists")
                
        except Exception as e:
            self.logger.error(f"Error ensuring collection exists: {str(e)}")
            raise
            
    def _load_file_hashes(self):
        """Load existing file hashes from Qdrant metadata"""
        try:
            # Get all points to build file hash cache
            scroll_result = self.client.scroll(
                collection_name=self.collection_name,
                limit=10000,  # Adjust based on your needs
                with_payload=True
            )
            
            for point in scroll_result[0]:
                if 'file_path' in point.payload and 'file_hash' in point.payload:
                    self.file_hashes[point.payload['file_path']] = point.payload['file_hash']
                    
        except Exception as e:
            self.logger.error(f"Error loading file hashes: {str(e)}")
            
    def is_file_unchanged(self, file_path: str, current_hash: str) -> bool:
        """Check if file has not changed since last indexing"""
        return self.file_hashes.get(file_path) == current_hash
    
    def get_file_hash(self, file_path: str) -> str:
        """Get MD5 hash of file for change detection"""
        import hashlib
        try:
            with open(file_path, 'rb') as f:
                return hashlib.md5(f.read()).hexdigest()
        except Exception:
            return ""
        
    def chunk_text(self, text: str) -> List[str]:
        """Split text into chunks for embedding"""
        chunk_size = self.config['embedding']['chunk_size']
        chunk_overlap = self.config['embedding']['chunk_overlap']
        
        if len(text) <= chunk_size:
            return [text]
            
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            
            # Try to break at word boundary
            if end < len(text):
                # Find the last space within the chunk
                last_space = text.rfind(' ', start, end)
                if last_space > start:
                    end = last_space
                    
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
                
            # Move start position with overlap
            start = end - chunk_overlap
            if start >= len(text):
                break
                
        return chunks
        
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for list of texts"""
        try:
            embeddings = self.embedding_model.encode(texts, convert_to_tensor=False)
            return embeddings.tolist()
        except Exception as e:
            self.logger.error(f"Error generating embeddings: {str(e)}")
            return []
            
    def index_document(self, file_path: str, text_content: str, file_hash: str):
        """Index a document in Qdrant"""
        try:
            # First, delete existing documents for this file (including those marked as deleted)
            self.delete_document(file_path, force_delete=True)
            
            # Chunk the text
            chunks = self.chunk_text(text_content)
            if not chunks:
                self.logger.warning(f"No chunks generated for {file_path}")
                return
                
            # Generate embeddings
            embeddings = self.generate_embeddings(chunks)
            if not embeddings:
                self.logger.error(f"No embeddings generated for {file_path}")
                return
                
            # Create points for Qdrant
            points = []
            timestamp = datetime.now().isoformat()
            
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                point_id = str(uuid.uuid4())
                
                payload = {
                    'document': chunk,
                    'metadata': {
                        'file_path': file_path,
                        'file_hash': file_hash,
                        'chunk_index': i,
                        'timestamp': timestamp,
                        'file_size': len(text_content),
                        'is_deleted': False  # Mark as active file
                    },
                    'file_path': file_path,
                    'file_hash': file_hash,
                    'chunk_index': i,
                    'timestamp': timestamp,
                    'file_size': len(text_content),
                    'is_deleted': False  # Mark as active file

                }
                
                point = PointStruct(
                    id=point_id,
                    vector={
                       'fast-all-minilm-l6-v2': embedding
                    },
                    payload=payload
                )
                points.append(point)
                
            # Upload to Qdrant
            self.client.upsert(
                collection_name=self.collection_name,
                points=points
            )
            
            # Update file hash cache
            self.file_hashes[file_path] = file_hash
            
            self.logger.info(f"Indexed {len(chunks)} chunks from {file_path}")
            
        except Exception as e:
            self.logger.error(f"Error indexing document {file_path}: {str(e)}")
            
    def delete_document(self, file_path: str, force_delete: bool = False):
        """Delete all chunks for a document from Qdrant"""
        try:
            # Find all points for this file
            search_result = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter={
                    "must": [
                        {
                            "key": "file_path",
                            "match": {"value": file_path}
                        }
                    ]
                },
                limit=10000,
                with_payload=True
            )
            
            # Extract point IDs
            point_ids = [point.id for point in search_result[0]]
            
            if point_ids:
                # Delete points
                self.client.delete(
                    collection_name=self.collection_name,
                    points_selector=point_ids
                )
                
                self.logger.info(f"Deleted {len(point_ids)} chunks for {file_path}")
                
            # Remove from file hash cache
            self.file_hashes.pop(file_path, None)
            
        except Exception as e:
            self.logger.error(f"Error deleting document {file_path}: {str(e)}")
    
    def mark_document_as_deleted(self, file_path: str):
        """Mark all chunks for a document as deleted without removing from Qdrant"""
        try:
            # Find all points for this file
            search_result = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter={
                    "must": [
                        {
                            "key": "file_path",
                            "match": {"value": file_path}
                        }
                    ]
                },
                limit=10000,
                with_payload=True
            )
            
            # Update points to mark as deleted
            points_to_update = []
            deletion_timestamp = datetime.now().isoformat()
            
            for point in search_result[0]:
                # Update payload to mark as deleted
                updated_payload = point.payload.copy()
                updated_payload['is_deleted'] = True
                updated_payload['deletion_timestamp'] = deletion_timestamp
                
                updated_point = PointStruct(
                    id=point.id,
                    vector={
                       'fast-all-minilm-l6-v2': point.vector
                    },
                    payload=updated_payload
                )
                points_to_update.append(updated_point)
            
            if points_to_update:
                # Update points in Qdrant
                self.client.upsert(
                    collection_name=self.collection_name,
                    points=points_to_update
                )
                
                self.logger.info(f"Marked {len(points_to_update)} chunks as deleted for {file_path}")
            
            # Remove from file hash cache (so it will be reprocessed if the file reappears)
            self.file_hashes.pop(file_path, None)
            
        except Exception as e:
            self.logger.error(f"Error marking document as deleted {file_path}: {str(e)}")
            
    def search_similar(self, query: str, limit: int = 5, include_deleted: bool = False) -> List[Dict]:
        """Search for similar documents (for testing)"""
        try:
            query_embedding = self.generate_embeddings([query])[0]
            
            # Create filter to exclude deleted documents unless specifically requested
            search_filter = None
            if not include_deleted:
                search_filter = {
                    "should": [
                        {
                            "key": "is_deleted",
                            "match": {"value": False}
                        },
                        {
                            "must_not": [
                                {
                                    "key": "is_deleted",
                                    "match": {"any": [True, False]}
                                }
                            ]
                        }
                    ]
                }
            
            search_result = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=limit,
                query_filter=search_filter,
                with_payload=True
            )
            
            results = []
            for hit in search_result:
                result = {
                    'file_path': hit.payload.get('file_path'),
                    'document': hit.payload.get('document'),
                    'score': hit.score,
                    'chunk_index': hit.payload.get('chunk_index'),
                    'is_deleted': hit.payload.get('is_deleted', False)
                }
                
                if hit.payload.get('deletion_timestamp'):
                    result['deletion_timestamp'] = hit.payload.get('deletion_timestamp')
                    
                results.append(result)
                
            return results
            
        except Exception as e:
            self.logger.error(f"Error searching: {str(e)}")
            return []
    
    def get_deleted_documents(self) -> List[Dict]:
        """Get list of all documents marked as deleted"""
        try:
            search_result = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter={
                    "must": [
                        {
                            "key": "is_deleted",
                            "match": {"value": True}
                        }
                    ]
                },
                limit=10000,
                with_payload=True
            )
            
            # Group by file_path to avoid duplicates
            deleted_files = {}
            for point in search_result[0]:
                file_path = point.payload.get('file_path')
                if file_path not in deleted_files:
                    deleted_files[file_path] = {
                        'file_path': file_path,
                        'deletion_timestamp': point.payload.get('deletion_timestamp'),
                        'original_timestamp': point.payload.get('timestamp'),
                        'chunk_count': 0
                    }
                deleted_files[file_path]['chunk_count'] += 1
            
            return list(deleted_files.values())
            
        except Exception as e:
            self.logger.error(f"Error getting deleted documents: {str(e)}")
            return []
