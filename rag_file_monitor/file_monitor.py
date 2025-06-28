#!/usr/bin/env python3
"""
RAG File Monitor - Monitors directories and indexes files for RAG system
"""

import os
import sys
import time
import logging
import hashlib
import yaml
import click
from pathlib import Path
from typing import List, Dict, Set
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

from .text_extractors import TextExtractor
from .embedding_manager import EmbeddingManager


class FileMonitorHandler(FileSystemEventHandler):
    """Handler for file system events"""
    
    def __init__(self, embedding_manager: EmbeddingManager, text_extractor: TextExtractor, 
                 file_extensions: Set[str], exclude_patterns: List[str], max_file_size: int,
                 delete_embeddings_on_deletion: bool):
        self.embedding_manager = embedding_manager
        self.text_extractor = text_extractor
        self.file_extensions = file_extensions
        self.exclude_patterns = exclude_patterns
        self.max_file_size = max_file_size
        self.delete_embeddings_on_deletion = delete_embeddings_on_deletion
        self.logger = logging.getLogger(__name__)
        
    def should_process_file(self, file_path: str) -> bool:
        """Check if file should be processed"""
        path = Path(file_path)
        
        # Check extension
        if path.suffix.lower() not in self.file_extensions:
            return False
            
        # Check exclude patterns
        for pattern in self.exclude_patterns:
            if path.match(pattern):
                return False
                
        # Check file size
        try:
            if path.stat().st_size > self.max_file_size:
                return False
        except OSError:
            return False
            
        return True
        
    def get_file_hash(self, file_path: str) -> str:
        """Get MD5 hash of file for change detection using streaming"""
        try:
            hash_md5 = hashlib.md5()
            # Read file in chunks to avoid loading entire file into memory
            with open(file_path, 'rb') as f:
                for chunk in iter(lambda: f.read(8192), b""):
                    hash_md5.update(chunk)
            return hash_md5.hexdigest()
        except Exception:
            return ""
            
    def on_created(self, event):
        if not event.is_directory and self.should_process_file(event.src_path):
            self.logger.info(f"New file detected: {event.src_path}")
            self.process_file(event.src_path)
            
    def on_modified(self, event):
        if not event.is_directory and self.should_process_file(event.src_path):
            self.logger.info(f"File modified: {event.src_path}")
            self.process_file(event.src_path)
            
    def on_deleted(self, event):
        if not event.is_directory:
            self.logger.info(f"File deleted: {event.src_path}")
            if self.delete_embeddings_on_deletion:
                self.embedding_manager.delete_document(event.src_path)
            else:
                self.embedding_manager.mark_document_as_deleted(event.src_path)
            
    def on_moved(self, event):
        if not event.is_directory:
            self.logger.info(f"File moved: {event.src_path} -> {event.dest_path}")
            if self.delete_embeddings_on_deletion:
                self.embedding_manager.delete_document(event.src_path)
            else:
                self.embedding_manager.mark_document_as_deleted(event.src_path)
            
            if self.should_process_file(event.dest_path):
                self.process_file(event.dest_path)
                
    def process_file(self, file_path: str):
        """Process a single file"""
        try:
            # Check if file has changed
            current_hash = self.get_file_hash(file_path)
            if self.embedding_manager.is_file_unchanged(file_path, current_hash):
                return
                
            self.logger.info(f"Document {file_path} needs indexing, extract text")
            # Extract text
            text_content = self.text_extractor.extract_text(file_path)
            if not text_content.strip():
                self.logger.warning(f"No text extracted from {file_path}")
                return
                
            # Index the document
            self.embedding_manager.index_document(file_path, text_content, current_hash)
            self.logger.info(f"Successfully indexed: {file_path}")
            
        except Exception as e:
            self.logger.error(f"Error processing {file_path}: {str(e)}")


class FileMonitor:
    """Main file monitoring class"""
    
    def __init__(self, config_path: str = "config.yaml"):
        self.config = self.load_config(config_path)
        self.setup_logging()
        
        self.embedding_manager = EmbeddingManager(self.config)
        self.text_extractor = TextExtractor()
        
        self.observers = []
        
    def load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file"""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            print(f"Error loading config: {e}")
            sys.exit(1)
            
    def setup_logging(self):
        """Setup logging configuration"""
        log_level = getattr(logging, self.config['logging']['level'].upper())
        log_file = self.config['logging']['file']
        
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        
        self.logger = logging.getLogger(__name__)
        
    def scan_existing_files(self):
        """Scan existing files in monitored directories"""
        self.logger.info("Scanning existing files...")
        
        file_extensions = set(self.config['file_extensions'])
        exclude_patterns = self.config['processing']['exclude_patterns']
        max_file_size = self.config['processing']['max_file_size_mb'] * 1024 * 1024
        delete_embeddings_on_deletion = self.config['processing']['delete_embeddings_on_file_deletion']
        
        handler = FileMonitorHandler(
            self.embedding_manager, 
            self.text_extractor,
            file_extensions,
            exclude_patterns,
            max_file_size,
            delete_embeddings_on_deletion
        )
        
        total_files = 0
        for directory in self.config['directories']:
            if not os.path.exists(directory):
                self.logger.warning(f"Directory does not exist: {directory}")
                continue
                
            self.logger.info(f"Scanning directory: {directory}")
            
            for root, dirs, files in os.walk(directory):
                for file in files:
                    file_path = os.path.join(root, file)
                    if handler.should_process_file(file_path):
                        self.logger.info(f"Processing existing file: {file_path}")
                        handler.process_file(file_path)
                        total_files += 1
                    else:
                        self.logger.debug(f"Skipping file: {file_path} (does not match criteria)")
                        
        self.logger.info(f"Finished scanning. Processed {total_files} files.")
        
    def start_monitoring(self):
        """Start monitoring directories for changes"""
        self.logger.info("Starting file monitoring...")
        
        file_extensions = set(self.config['file_extensions'])
        exclude_patterns = self.config['processing']['exclude_patterns']
        max_file_size = self.config['processing']['max_file_size_mb'] * 1024 * 1024
        delete_embeddings_on_deletion = self.config['processing']['delete_embeddings_on_file_deletion']
        
        event_handler = FileMonitorHandler(
            self.embedding_manager,
            self.text_extractor,
            file_extensions,
            exclude_patterns,
            max_file_size,
            delete_embeddings_on_deletion
        )
        
        for directory in self.config['directories']:
            if not os.path.exists(directory):
                self.logger.warning(f"Directory does not exist: {directory}")
                continue
                
            observer = Observer()
            observer.schedule(event_handler, directory, recursive=True)
            observer.start()
            self.observers.append(observer)
            self.logger.info(f"Monitoring directory: {directory}")
            
        if not self.observers:
            self.logger.error("No valid directories to monitor")
            return
            
        try:
            self.logger.info("File monitoring active. Press Ctrl+C to stop.")
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            self.logger.info("Stopping file monitoring...")
            for observer in self.observers:
                observer.stop()
                observer.join()


@click.command()
@click.option('--config', '-c', default='config.yaml', help='Path to config file')
@click.option('--scan-only', is_flag=True, help='Only scan existing files, don\'t monitor')
@click.option('--monitor-only', is_flag=True, help='Only monitor for changes, skip initial scan')
def main(config, scan_only, monitor_only):
    """RAG File Monitor - Index files for RAG system"""
    
    monitor = FileMonitor(config)
    
    if scan_only:
        monitor.scan_existing_files()
    elif monitor_only:
        monitor.start_monitoring()
    else:
        # Default: scan existing files then start monitoring
        monitor.scan_existing_files()
        monitor.start_monitoring()


if __name__ == "__main__":
    main()
