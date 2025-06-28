"""
Memory monitoring and optimization utilities
"""

import gc
import logging
import psutil
import threading
import time
from typing import Optional, Callable
from functools import wraps


class MemoryMonitor:
    """Monitor and manage memory usage"""
    
    def __init__(self, config: dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.process = psutil.Process()
        self.operation_count = 0
        self.last_gc_time = time.time()
        
        # Configuration from config
        self.force_gc_threshold = config.get('memory', {}).get('force_gc_after_operations', 100)
        
    def get_memory_usage(self) -> dict:
        """Get current memory usage statistics"""
        memory_info = self.process.memory_info()
        memory_percent = self.process.memory_percent()
        
        return {
            'rss_mb': memory_info.rss / 1024 / 1024,  # Resident Set Size
            'vms_mb': memory_info.vms / 1024 / 1024,  # Virtual Memory Size
            'percent': memory_percent,
            'available_mb': psutil.virtual_memory().available / 1024 / 1024
        }
    
    def log_memory_usage(self, context: str = ""):
        """Log current memory usage"""
        usage = self.get_memory_usage()
        self.logger.info(
            f"Memory usage {context}: "
            f"RSS={usage['rss_mb']:.1f}MB, "
            f"VMS={usage['vms_mb']:.1f}MB, "
            f"Percent={usage['percent']:.1f}%, "
            f"Available={usage['available_mb']:.1f}MB"
        )
    
    def check_memory_threshold(self, threshold_percent: float = 80.0) -> bool:
        """Check if memory usage exceeds threshold"""
        usage = self.get_memory_usage()
        return usage['percent'] > threshold_percent
    
    def force_garbage_collection(self):
        """Force garbage collection and log memory freed"""
        before = self.get_memory_usage()
        gc.collect()
        after = self.get_memory_usage()
        
        freed_mb = before['rss_mb'] - after['rss_mb']
        if freed_mb > 1.0:  # Only log if significant memory was freed
            self.logger.info(f"Garbage collection freed {freed_mb:.1f}MB")
        
        self.last_gc_time = time.time()
    
    def increment_operation_count(self):
        """Increment operation counter and check if GC is needed"""
        self.operation_count += 1
        
        if self.operation_count % self.force_gc_threshold == 0:
            self.force_garbage_collection()
    
    def memory_usage_decorator(self, context: str = ""):
        """Decorator to monitor memory usage of functions"""
        def decorator(func: Callable):
            @wraps(func)
            def wrapper(*args, **kwargs):
                # Log memory before
                before = self.get_memory_usage()
                
                try:
                    result = func(*args, **kwargs)
                    return result
                finally:
                    # Log memory after and increment counter
                    after = self.get_memory_usage()
                    memory_diff = after['rss_mb'] - before['rss_mb']
                    
                    if abs(memory_diff) > 10:  # Log significant changes
                        self.logger.debug(
                            f"{func.__name__} {context}: "
                            f"Memory change: {memory_diff:+.1f}MB, "
                            f"Current: {after['rss_mb']:.1f}MB"
                        )
                    
                    self.increment_operation_count()
            
            return wrapper
        return decorator


class ModelManager:
    """Manage ML model lifecycle to optimize memory"""
    
    def __init__(self, config: dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.model = None
        self.last_used = time.time()
        self.idle_timeout = config.get('memory', {}).get('unload_model_after_idle_minutes', 30) * 60
        self.cleanup_timer = None
        
    def get_model(self, model_loader: Callable):
        """Get model with lazy loading and automatic cleanup"""
        if self.model is None:
            self.logger.info("Loading model on demand")
            self.model = model_loader()
        
        self.last_used = time.time()
        self._schedule_cleanup()
        return self.model
    
    def _schedule_cleanup(self):
        """Schedule model cleanup after idle timeout"""
        if self.cleanup_timer:
            self.cleanup_timer.cancel()
        
        self.cleanup_timer = threading.Timer(self.idle_timeout, self._cleanup_if_idle)
        self.cleanup_timer.start()
    
    def _cleanup_if_idle(self):
        """Clean up model if it's been idle"""
        if time.time() - self.last_used >= self.idle_timeout:
            self.unload_model()
    
    def unload_model(self):
        """Unload model to free memory"""
        if self.model is not None:
            self.logger.info("Unloading model due to inactivity")
            del self.model
            self.model = None
            gc.collect()
        
        if self.cleanup_timer:
            self.cleanup_timer.cancel()
            self.cleanup_timer = None
    
    def __del__(self):
        """Cleanup on destruction"""
        self.unload_model()


def memory_efficient_batch_processor(items: list, batch_size: int, processor: Callable, 
                                    memory_monitor: Optional[MemoryMonitor] = None):
    """
    Process items in batches with memory monitoring
    
    Args:
        items: List of items to process
        batch_size: Size of each batch
        processor: Function to process each batch
        memory_monitor: Optional memory monitor for tracking
    """
    results = []
    
    for i in range(0, len(items), batch_size):
        batch = items[i:i + batch_size]
        
        # Process batch
        batch_results = processor(batch)
        results.extend(batch_results)
        
        # Clear batch from memory
        del batch, batch_results
        
        # Optional memory monitoring
        if memory_monitor:
            memory_monitor.increment_operation_count()
            
            # Force GC if memory usage is high
            if memory_monitor.check_memory_threshold(70.0):
                memory_monitor.force_garbage_collection()
    
    return results


def stream_large_file(file_path: str, chunk_size: int = 8192):
    """Generator to stream large files in chunks"""
    try:
        with open(file_path, 'rb') as f:
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                yield chunk
    except Exception as e:
        logging.getLogger(__name__).error(f"Error streaming file {file_path}: {e}")
