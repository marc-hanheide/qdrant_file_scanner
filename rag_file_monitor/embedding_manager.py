"""
Embedding generation and Qdrant database management
"""

import logging
import uuid
import fnmatch
import threading
from typing import List, Dict, Optional, Any
from datetime import datetime, timedelta

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue, MatchText
except ImportError:
    QdrantClient = None
    Distance = None
    VectorParams = None
    PointStruct = None
    Filter = None
    FieldCondition = None
    MatchValue = None
    MatchText = None

#from ..memory_utils import ModelManager


class EmbeddingManager:
    """Manage embeddings and Qdrant vector database operations"""
    
    def __init__(self, config: Dict, slim_mode: bool = False):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize embedding model lazily
        if SentenceTransformer is None:
            raise ImportError("sentence-transformers is required")
            
        self.model_name = config['embedding']['model_name']
        self.embedding_model = None  # Load lazily
        self.model_last_used = None  # Track when model was last used
        self.model_lock = threading.Lock()  # Thread safety for model operations
        self.logger.info(f"Embedding model configured: {self.model_name}")
        
        # Memory optimization settings
        self.unload_after_minutes = config.get('memory', {}).get('unload_model_after_idle_minutes', 30)
        self.operation_counter = 0
        self.gc_frequency = config.get('memory', {}).get('force_gc_after_operations', 100)
        
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
        # Get vector name from config with default fallback
        self.vector_name = qdrant_config.get('vector_name', 'fast-all-minilm-l6-v2')
        # Ensure collection exists
        self._ensure_collection_exists()
        
        # File tracking for change detection
        self.file_hashes = {}
        if not slim_mode:
            self.logger.info("Loading existing file hashes from Qdrant")
            # Load existing file hashes from Qdrant metadata
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
                        self.vector_name: VectorParams(
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
        """Load existing file hashes from Qdrant metadata using pagination"""
        try:
            self.logger.info(f"Loading existing file hashes from Qdrant")
            
            # Use pagination to avoid loading all points at once
            offset = None
            batch_size = self.config.get('memory', {}).get('hash_loading_batch_size', 1000)  # Use config value with default
            total_loaded = 0
            
            while True:
                scroll_result = self.client.scroll(
                    collection_name=self.collection_name,
                    limit=batch_size,
                    offset=offset,
                    with_payload=["file_path", "file_hash"]  # Only load needed fields
                )
                
                points, next_offset = scroll_result
                
                if not points:
                    break
                    
                # Process batch and extract unique file hashes
                for point in points:
                    if 'file_path' in point.payload and 'file_hash' in point.payload:
                        self.file_hashes[point.payload['file_path']] = point.payload['file_hash']
                
                total_loaded += len(points)
                
                # Break if no more results
                if next_offset is None:
                    break
                    
                offset = next_offset
                
                # Optional: Yield control to prevent blocking
                if total_loaded % 5000 == 0:
                    self.logger.info(f"Loaded {total_loaded} file hashes so far...")
            
            self.logger.info(f"Loaded {len(self.file_hashes)} unique file hashes from {total_loaded} points")
                    
        except Exception as e:
            self.logger.error(f"Error loading file hashes: {str(e)}")
            
    def is_file_unchanged(self, file_path: str, current_hash: str) -> bool:
        """Check if file has not changed since last indexing"""
        return self.file_hashes.get(file_path) == current_hash
    
    def get_file_hash(self, file_path: str) -> str:
        """Get MD5 hash of file for change detection using streaming"""
        import hashlib
        try:
            hash_md5 = hashlib.md5()
            # Read file in chunks to avoid loading entire file into memory
            with open(file_path, 'rb') as f:
                for chunk in iter(lambda: f.read(8192), b""):
                    hash_md5.update(chunk)
            return hash_md5.hexdigest()
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
            original_end = end
            
            # Try to break at word boundary
            if end < len(text):
                # Find the last space within the chunk
                last_space = text.rfind(' ', start, end)
                if last_space > start:
                    end = last_space
                    
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
                
            # Move start position with overlap, but ensure we always advance
            next_start = end - chunk_overlap
            if next_start <= start:
                # Fallback: advance by at least chunk_size to avoid infinite loop
                next_start = start + max(chunk_size, 1)
            
            start = next_start
            if start >= len(text):
                break
                
        return chunks
        
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for list of texts with memory optimization"""
        try:
            # Increment operation counter and check for GC
            self.operation_counter += 1
            if self.operation_counter % self.gc_frequency == 0:
                import gc
                gc.collect()
                self.logger.debug(f"Forced garbage collection after {self.operation_counter} operations")
            
            # Process in smaller batches to reduce memory usage
            batch_size = self.config.get('memory', {}).get('embedding_batch_size', 32)  # Adjust based on your system's memory
            all_embeddings = []
            
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                # Ensure all text chunks are properly UTF-8 encoded
                batch = [self._ensure_utf8_encoding(text) for text in batch]
                batch_embeddings = self._get_embedding_model().encode(
                    batch, 
                    convert_to_tensor=False,
                    show_progress_bar=False,
                    batch_size=min(len(batch), 32)  # Internal batch size
                )
                
                # Validate and clean embeddings
                validated_embeddings = []
                for j, embedding in enumerate(batch_embeddings.tolist()):
                    validated_embedding = self._validate_embedding(embedding, texts[i + j])
                    if validated_embedding is not None:
                        validated_embeddings.append(validated_embedding)
                    else:
                        # Skip this text if embedding validation fails
                        self.logger.warning(f"Skipping text chunk due to invalid embedding: '{texts[i + j][:100]}...'")
                
                all_embeddings.extend(validated_embeddings)
                
                # Optional: Force garbage collection for large batches
                if len(texts) > 100 and i % (batch_size * 4) == 0:
                    import gc
                    gc.collect()
            
            return all_embeddings
        except Exception as e:
            self.logger.error(f"Error generating embeddings: {str(e)}")
            return []
            
    def index_document(self, file_path: str, text_content: str, file_hash: str):
        """Index a document in Qdrant with memory optimization"""
        try:
            # First, delete existing documents for this file (including those marked as deleted)
            self.delete_document(file_path, force_delete=True)
            
            # Chunk the text
            chunks = self.chunk_text(text_content)
            if not chunks:
                self.logger.warning(f"No chunks generated for {file_path}")
                return
            self.logger.info(f"Indexing document {file_path} with {len(chunks)} chunks")

            timestamp = datetime.now().isoformat()
            # Process chunks in batches to reduce memory usage
            chunk_batch_size = self.config['memory'].get('chunk_batch_size', 50)  # Process chunks in smaller batches
            timestamp = datetime.now().isoformat()
            total_indexed = 0
            
            for batch_start in range(0, len(chunks), chunk_batch_size):
                batch_end = min(batch_start + chunk_batch_size, len(chunks))
                chunk_batch = chunks[batch_start:batch_end]
                
                # Generate embeddings for this batch
                embeddings = self.generate_embeddings(chunk_batch)
                if not embeddings:
                    error_msg = f"No embeddings generated for batch {batch_start}-{batch_end} of {file_path}"
                    self.logger.error(error_msg)
                    raise RuntimeError(error_msg)
                
                # Ensure embeddings match chunks after validation
                if len(embeddings) != len(chunk_batch):
                    self.logger.warning(f"Embedding count mismatch for {file_path}: {len(embeddings)} embeddings vs {len(chunk_batch)} chunks. Adjusting chunks to match embeddings.")
                    # Adjust chunks to match validated embeddings
                    chunk_batch = chunk_batch[:len(embeddings)]
                
                # Create points for Qdrant
                points = []
                
                for i, (chunk, embedding) in enumerate(zip(chunk_batch, embeddings)):
                    point_id = str(uuid.uuid4())
                    chunk_index = batch_start + i
                    
                    payload = {
                        'document': chunk,
                        'metadata': {
                            'file_path': file_path,
                            'file_hash': file_hash,
                            'chunk_index': chunk_index,
                            'timestamp': timestamp,
                            'file_size': len(text_content),
                            'is_deleted': False  # Mark as active file
                        },
                        'file_path': file_path,
                        'file_hash': file_hash,
                        'chunk_index': chunk_index,
                        'timestamp': timestamp,
                        'file_size': len(text_content),
                        'is_deleted': False  # Mark as active file
                    }
                    
                    point = PointStruct(
                        id=point_id,
                        vector={
                           self.vector_name: embedding
                        },
                        payload=payload
                    )
                    points.append(point)
                
                # Upload batch to Qdrant
                try:
                    self.client.upsert(
                        collection_name=self.collection_name,
                        points=points
                    )
                except Exception as e:
                    error_msg = f"Failed to upsert points to Qdrant for {file_path}: {str(e)}"
                    self.logger.error(error_msg)
                    raise RuntimeError(error_msg)
                
                total_indexed += len(points)
                
                # Clear batch data from memory
                del chunk_batch, embeddings, points
                
                # Optional: Force garbage collection between batches
                if batch_start % (chunk_batch_size * 4) == 0:
                    import gc
                    gc.collect()
            
            # Update file hash cache
            self.file_hashes[file_path] = file_hash
            
            self.logger.info(f"Indexed {total_indexed} chunks from {file_path}")
            
        except Exception as e:
            error_msg = f"Error indexing document {file_path}: {str(e)}"
            self.logger.error(error_msg)
            raise RuntimeError(error_msg)
            
    def delete_document(self, file_path: str, force_delete: bool = False):
        """Delete all chunks for a document from Qdrant"""
        try:
            # Find all points for this file
            search_result = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=Filter(
                    must=[
                        FieldCondition(
                            key="file_path",
                            match=MatchValue(value=file_path)
                        )
                    ]
                ),
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
                scroll_filter=Filter(
                    must=[
                        FieldCondition(
                            key="file_path",
                            match=MatchValue(value=file_path)
                        )
                    ]
                ),
                limit=10000,
                with_payload=True,
                with_vectors=True  # No need to load vectors for deletion
            )
            
            # Update points to mark as deleted
            points_to_update = []
            deletion_timestamp = datetime.now().isoformat()
            
            for point in search_result[0]:
                # Update payload to mark as deleted
                updated_payload = point.payload.copy()
                updated_payload['is_deleted'] = True
                updated_payload['metadata']['is_deleted'] = True
                updated_payload['deletion_timestamp'] = deletion_timestamp
                updated_point = PointStruct(
                    id=point.id,
                    vector={
                       self.vector_name: point.vector[self.vector_name]
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
            
    def _build_glob_filter_conditions(self, glob_pattern: str) -> List[Any]:
        """
        Convert glob pattern to Qdrant filter conditions.
        
        This method handles common glob patterns and converts them to efficient Qdrant filters:
        - *.ext -> files ending with .ext
        - *keyword* -> files containing keyword
        - prefix* -> files starting with prefix
        - Complex patterns fall back to post-processing
        """
        if not glob_pattern:
            return []
            
        filter_conditions = []
        pattern_lower = glob_pattern.lower()
        
        try:
            # Handle file extension patterns like "*.pdf", "*.txt"
            if pattern_lower.startswith('*.') and '*' not in pattern_lower[2:]:
                extension = pattern_lower[1:]  # Include the dot
                # Use MatchText to find files ending with the extension
                if MatchText is not None:
                    filter_conditions.append(
                        FieldCondition(
                            key="file_path",
                            match=MatchText(text=extension)
                        )
                    )
                return filter_conditions
            
            # Handle simple prefix patterns like "report*"
            elif pattern_lower.endswith('*') and '*' not in pattern_lower[:-1]:
                prefix = pattern_lower[:-1]
                if MatchText is not None:
                    filter_conditions.append(
                        FieldCondition(
                            key="file_path", 
                            match=MatchText(text=prefix)
                        )
                    )
                return filter_conditions
            
            # Handle simple contains patterns like "*keyword*"
            elif pattern_lower.startswith('*') and pattern_lower.endswith('*') and pattern_lower.count('*') == 2:
                keyword = pattern_lower[1:-1]
                if keyword and MatchText is not None:
                    filter_conditions.append(
                        FieldCondition(
                            key="file_path",
                            match=MatchText(text=keyword)
                        )
                    )
                return filter_conditions
                
        except Exception as e:
            self.logger.warning(f"Error building glob filter conditions: {e}")
        
        # For complex patterns, return empty list and fall back to post-processing
        # This includes patterns with multiple *, path separators, etc.
        return []

    def search_similar(self, query: str, limit: int = 5, include_deleted: bool = True, glob_pattern: Optional[str] = None) -> List[Dict]:
        """Search for similar documents with optional glob pattern filtering"""
        try:
            query_embedding = self.generate_embeddings([query])[0]
            
            # Build search filter conditions using proper Qdrant filter format
            filter_conditions = []
            
            # Handle deleted documents filter
            if not include_deleted:
                filter_conditions.append(
                    FieldCondition(
                        key="is_deleted",
                        match=MatchValue(value=False)
                    )
                )
            
            # Try to add glob pattern filter conditions at Qdrant level
            use_post_processing = False
            if glob_pattern:
                glob_filter_conditions = self._build_glob_filter_conditions(glob_pattern)
                if glob_filter_conditions:
                    self.logger.info(f"Direct filtering on qdrant: {glob_filter_conditions}")
                    # We can filter at Qdrant level
                    filter_conditions.extend(glob_filter_conditions)
                else:
                    # Complex pattern - need post-processing
                    self.logger.info(f"needs post processing: {glob_pattern}")
                    use_post_processing = True
                    
            # Set search limit based on whether we need post-processing
            effective_limit = limit
            if use_post_processing:
                # Increase limit to account for post-processing filtering
                effective_limit = min(limit * 100, 1000)
            
            # Combine all filter conditions
            search_filter = None
            if filter_conditions:
                search_filter = Filter(must=filter_conditions)
            
            search_result = self.client.search(
                collection_name=self.collection_name,
                query_vector=(self.vector_name, query_embedding),
                limit=effective_limit,
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
                
                # Apply post-processing glob filter if needed
                if glob_pattern:
                    file_path = result.get('file_path', '')
                    if not self._matches_glob_pattern(file_path, glob_pattern):
                        continue
                
                results.append(result)
                
                # Stop once we have enough results (for post-processing case)
                if len(results) >= limit:
                    break
                
            return results
            
        except Exception as e:
            self.logger.error(f"Error searching: {str(e)}")
            return []
    
    def _matches_glob_pattern(self, file_path: str, glob_pattern: str) -> bool:
        """
        Check if a file path matches the given glob pattern.
        This is used for complex patterns that can't be handled by Qdrant filters.
        """
        if not file_path or not glob_pattern:
            return True
        
        # Case insensitive matching
        file_path_lower = file_path.lower()
        glob_pattern_lower = glob_pattern.lower()
        
        # Use fnmatch for the actual pattern matching
        return fnmatch.fnmatch(file_path_lower, glob_pattern_lower)

    def get_deleted_documents(self) -> List[Dict]:
        """Get list of all documents marked as deleted"""
        try:
            search_result = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=Filter(
                    must=[
                        FieldCondition(
                            key="is_deleted",
                            match=MatchValue(value=True)
                        )
                    ]
                ),
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
    
    def _get_embedding_model(self):
        """Lazy loading of embedding model with automatic unloading after idle time"""
        with self.model_lock:
            # Check if model should be unloaded due to idle time
            if (self.embedding_model is not None and 
                self.model_last_used is not None and 
                datetime.now() - self.model_last_used > timedelta(minutes=self.unload_after_minutes)):
                self.logger.info(f"Unloading embedding model after {self.unload_after_minutes} minutes of inactivity")
                self.unload_embedding_model()
            
            # Load model if not already loaded
            if self.embedding_model is None:
                self.logger.info(f"Loading embedding model: {self.model_name}")
                # Use local cache directory to avoid downloading model every time
                cache_folder = self.config.get('embedding', {}).get('cache_folder', './models')
                self.embedding_model = SentenceTransformer(self.model_name, cache_folder=cache_folder)
            
            # Update last used timestamp
            self.model_last_used = datetime.now()
            return self.embedding_model
    
    def unload_embedding_model(self):
        """Unload embedding model to free memory when not needed"""
        # Note: This method should be called with model_lock held
        if self.embedding_model is not None:
            self.logger.info("Unloading embedding model to free memory")
            del self.embedding_model
            self.embedding_model = None
            self.model_last_used = None
            import gc
            gc.collect()
    
    def _validate_embedding(self, embedding: List[float], text_sample: str = "") -> Optional[List[float]]:
        """Validate embedding vector to ensure it's suitable for Qdrant"""
        try:
            import math
            
            # Check if embedding is None or empty
            if embedding is None or len(embedding) == 0:
                self.logger.warning(f"Empty or None embedding for text: '{text_sample[:50]}...'")
                return None
            
            # Check expected dimensions
            if len(embedding) != self.vector_size:
                self.logger.warning(f"Embedding dimension mismatch: expected {self.vector_size}, got {len(embedding)} for text: '{text_sample[:50]}...'")
                return None
            
            # Check for NaN or infinity values and provide detailed analysis
            invalid_indices = []
            for i, value in enumerate(embedding):
                if not isinstance(value, (int, float)):
                    self.logger.warning(f"Non-numeric value at index {i} in embedding for text: '{text_sample[:50]}...'")
                    return None
                if math.isnan(value):
                    invalid_indices.append(f"NaN at index {i}")
                elif math.isinf(value):
                    invalid_indices.append(f"Inf at index {i}")
            
            if invalid_indices:
                self.logger.error(f"Invalid embedding values found: {invalid_indices[:5]} for text: '{text_sample[:100]}...'")
                # Log additional debug info for problematic text
                return None
            
            # Check for suspicious patterns (all zeros, all same values)
            unique_values = set(embedding)
            if len(unique_values) == 1:
                self.logger.warning(f"Embedding contains only one unique value ({list(unique_values)[0]}) for text: '{text_sample[:50]}...'")
                # Don't reject, but log for investigation
            
            # Convert to regular floats to ensure JSON serialization
            validated_embedding = [float(v) for v in embedding]
            
            return validated_embedding
            
        except Exception as e:
            self.logger.error(f"Error validating embedding: {str(e)} for text: '{text_sample[:50]}...'")
            return None
        
    def _ensure_utf8_encoding(self, text: str) -> str:
        """Ensure text is properly UTF-8 encoded and clean"""
        if not isinstance(text, str):
            return str(text)
        
        try:
            # Remove or replace problematic characters
            # Replace null bytes and other control characters
            cleaned_text = text.replace('\x00', '').replace('\ufffd', '')
            
            # Ensure the text can be encoded as UTF-8
            cleaned_text.encode('utf-8')
            
            return cleaned_text
        except (UnicodeEncodeError, UnicodeDecodeError) as e:
            self.logger.warning(f"Encoding issue with text, attempting to fix: {str(e)}")
            # Try to encode and decode to fix encoding issues
            try:
                return text.encode('utf-8', errors='ignore').decode('utf-8')
            except Exception:
                # Last resort: return a safe placeholder
                return "[TEXT_ENCODING_ERROR]"

    def check_and_unload_idle_model(self):
        """Check if model has been idle and unload it if necessary"""
        with self.model_lock:
            if (self.embedding_model is not None and 
                self.model_last_used is not None and 
                datetime.now() - self.model_last_used > timedelta(minutes=self.unload_after_minutes)):
                self.logger.info(f"Unloading idle embedding model after {self.unload_after_minutes} minutes")
                self.unload_embedding_model()
                return True
        return False
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory usage statistics"""
        return {
            'file_hashes_count': len(self.file_hashes),
            'embedding_model_loaded': self.embedding_model is not None,
            'model_last_used': self.model_last_used.isoformat() if self.model_last_used else None,
            'operation_counter': self.operation_counter
        }

    def _debug_text_content(self, text_content: str, file_path: str) -> Dict[str, Any]:
        """Debug helper to analyze text content that might cause embedding issues"""
        debug_info = {
            'file_path': file_path,
            'content_length': len(text_content),
            'content_preview': text_content[:200] + '...' if len(text_content) > 200 else text_content,
            'encoding_issues': [],
            'special_chars': []
        }
        
        try:
            # Check for encoding issues
            text_content.encode('utf-8')
        except UnicodeEncodeError as e:
            debug_info['encoding_issues'].append(str(e))
        
        # Check for unusual characters that might cause issues
        import unicodedata
        unusual_chars = set()
        for char in text_content:
            if unicodedata.category(char) in ['Cc', 'Cf', 'Cs', 'Co', 'Cn']:  # Control and other problematic categories
                unusual_chars.add(f"'{char}' ({unicodedata.name(char, 'UNKNOWN')})")
        
        debug_info['special_chars'] = list(unusual_chars)[:10]  # Limit to first 10
        
        return debug_info
