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
from typing import List, Dict, Set, Union
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

from .text_extractors import TextExtractor
from .embedding_manager import EmbeddingManager
from tqdm import tqdm


class FileMonitorHandler(FileSystemEventHandler):
    """Handler for file system events"""

    def __init__(
        self,
        embedding_manager: EmbeddingManager,
        text_extractor: TextExtractor,
        file_extensions: Set[str],
        exclude_patterns: List[str],
        max_file_size: int,
        delete_embeddings_on_deletion: bool,
        directory_path: str = None,
        directory_config: Dict = None,
    ):
        self.embedding_manager = embedding_manager
        self.text_extractor = text_extractor
        self.file_extensions = file_extensions
        self.exclude_patterns = exclude_patterns
        self.max_file_size = max_file_size
        self.delete_embeddings_on_deletion = delete_embeddings_on_deletion
        self.directory_path = directory_path
        self.directory_config = directory_config or {}
        self.logger = logging.getLogger(__name__)

    def get_effective_extensions_for_file(self, file_path: str) -> Set[str]:
        """Get effective file extensions for a file, considering directory-specific ignore rules"""
        if not self.directory_config or not self.directory_path:
            return self.file_extensions

        # Check if this file is in the configured directory
        if not file_path.startswith(self.directory_path):
            return self.file_extensions

        # Get ignore extensions for this directory
        ignore_extensions = set(self.directory_config.get("ignore_extensions", []))

        # Return global extensions minus ignored ones
        return self.file_extensions - ignore_extensions

    def get_effective_max_filesize_for_file(self, file_path: str) -> int:
        """Get effective max file size for a file, considering directory-specific settings"""
        if not self.directory_config or not self.directory_path:
            return self.max_file_size

        # Check if this file is in the configured directory
        if not file_path.startswith(self.directory_path):
            return self.max_file_size

        # Get max filesize for this directory (in MB, 0 means use global default)
        directory_max_mb = self.directory_config.get("max_filesize", 0)

        if directory_max_mb <= 0:
            return self.max_file_size
        else:
            return directory_max_mb * 1024 * 1024  # Convert MB to bytes

    def should_process_file(self, file_path: str) -> bool:
        """Check if file should be processed"""
        path = Path(file_path)
        self.logger.debug(f"Checking file: {file_path}")
        # Get effective extensions for this file
        effective_extensions = self.get_effective_extensions_for_file(file_path)
        self.logger.debug(f"Effective extensions for {file_path}: {effective_extensions}")

        # Check extension
        if path.suffix.lower() not in effective_extensions:
            self.logger.debug(f"File {file_path} has unsupported extension: {path.suffix.lower()}, skipping")
            return False

        # Check exclude patterns
        for pattern in self.exclude_patterns:
            if path.match(pattern):
                self.logger.debug(f"File {file_path} matches exclude pattern: {pattern}, skipping")
                return False

        # Check file size with directory-specific limits
        try:
            file_size = path.stat().st_size
            effective_max_size = self.get_effective_max_filesize_for_file(file_path)
            if file_size > effective_max_size:
                self.logger.info(
                    f"File {file_path} exceeds max size limit ({file_size} > {effective_max_size} bytes) and is skipped"
                )
                return False
        except OSError:
            return False
        self.logger.debug(f"File {file_path} passed all checks, will be processed")
        return True

    def should_ignore_file(self, file_path: str) -> bool:
        """Check if file should be ignored based on patterns and file size"""
        path = Path(file_path)

        # Check exclude patterns
        exclude_patterns = self.config.get("processing", {}).get("exclude_patterns", [])
        for pattern in exclude_patterns:
            # Convert glob pattern to Path.match format
            if "*" in pattern or "?" in pattern:
                if path.match(pattern):
                    return True
            elif pattern in str(path):
                return True

        # Check file size
        try:
            max_size_mb = self.config.get("processing", {}).get("max_file_size_mb", 5)
            max_size_bytes = max_size_mb * 1024 * 1024
            if path.stat().st_size > max_size_bytes:
                return True
        except (OSError, FileNotFoundError):
            return True

        return False

    def is_supported_file_type(self, file_path: str, supported_extensions: Set[str]) -> bool:
        """Check if file type is supported based on given extensions"""
        path = Path(file_path)
        return path.suffix.lower() in supported_extensions

    def get_file_hash(self, file_path: str) -> str:
        """Get MD5 hash of file for change detection using streaming"""
        try:
            hash_md5 = hashlib.md5()
            # Read file in chunks to avoid loading entire file into memory
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(8192), b""):
                    hash_md5.update(chunk)
            return hash_md5.hexdigest()
        except Exception:
            return ""

    def on_created(self, event):
        if not event.is_directory and self.should_process_file(event.src_path):
            self.logger.info(f"New file detected: {event.src_path}")
            print(f"New file detected: {event.src_path}", file=sys.stderr)
            self.process_file(event.src_path)

    def on_modified(self, event):
        if not event.is_directory and self.should_process_file(event.src_path):
            self.logger.info(f"File modified: {event.src_path}")
            print(f"File modified: {event.src_path}", file=sys.stderr)
            self.process_file(event.src_path)

    def on_deleted(self, event):
        if not event.is_directory:
            self.logger.info(f"File deleted: {event.src_path}")
            print(f"File deleted: {event.src_path}", file=sys.stderr)
            if self.delete_embeddings_on_deletion:
                self.embedding_manager.delete_document(event.src_path)
            else:
                self.embedding_manager.mark_document_as_deleted(event.src_path)

    def on_moved(self, event):
        if not event.is_directory:
            self.logger.info(f"File moved: {event.src_path} -> {event.dest_path}")
            print(f"File moved: {event.src_path} -> {event.dest_path}", file=sys.stderr)
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
                self.logger.info(f"File {file_path} has not changed, skipping indexing")
                return

            self.logger.info(f"Document {file_path} needs indexing, extract text")
            # Extract text
            text_content = self.text_extractor.extract_text(file_path)
            if not text_content.strip():
                self.logger.warning(f"No text extracted from {file_path}")
                return

            # Index the document - this now raises exceptions on failure
            self.embedding_manager.index_document(file_path, text_content, current_hash)
            self.logger.info(f"Successfully indexed: {file_path}")

        except Exception as e:
            self.logger.error(f"Failed to process {file_path}: {str(e)}")
            raise e  # Raise the exception to log it properly
            # Don't raise the exception to avoid crashing the monitor for single file failures


class FileMonitor:
    """Main file monitoring class"""

    def __init__(self, config_or_path: Union[str, Dict] = "config.yaml"):
        if isinstance(config_or_path, dict):
            self.config = config_or_path
        else:
            self.config = self.load_config(config_or_path)
        self.setup_logging()

        self.embedding_manager = EmbeddingManager(self.config)
        self.text_extractor = TextExtractor()

        self.observers = []

    def load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file with smart discovery"""
        # If the provided path is just "config.yaml" (default), try smart discovery
        if config_path == "config.yaml":
            config_candidates = [Path.cwd() / "config.yaml", Path(__file__).parent.parent / "config.yaml"]
            actual_config_path = None
            for candidate in config_candidates:
                if candidate.exists():
                    actual_config_path = candidate
                    break

            if actual_config_path is None:
                print(f"Configuration file not found. Tried: {config_candidates}")
                print("Please ensure config.yaml exists in the current directory or specify the path with --config")
                sys.exit(1)

            config_path = str(actual_config_path)

        # Load the configuration file
        try:
            with open(config_path, "r") as f:
                return yaml.safe_load(f)
        except Exception as e:
            print(f"Error loading config from {config_path}: {e}")
            sys.exit(1)

    def setup_logging(self):
        """Setup logging configuration"""
        log_level = getattr(logging, self.config["logging"]["level"].upper())
        log_file = self.config["logging"]["file"]

        logging.basicConfig(
            level=log_level,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[logging.FileHandler(log_file, mode="w", encoding="utf-8"), logging.StreamHandler(sys.stdout)],
        )

        self.logger = logging.getLogger(__name__)

    def is_supported_file_type(self, file_path: str, supported_extensions: Set[str]) -> bool:
        """Check if file type is supported based on given extensions"""
        path = Path(file_path)
        return path.suffix.lower() in supported_extensions

    def scan_existing_files(self):
        """Scan existing files in monitored directories"""
        self.logger.info("Scanning existing files...")

        # Get base configuration
        global_file_extensions = set(self.config["file_extensions"])
        exclude_patterns = self.config["processing"]["exclude_patterns"]
        max_file_size = self.config["processing"]["max_file_size_mb"] * 1024 * 1024
        delete_embeddings_on_deletion = self.config["processing"]["delete_embeddings_on_file_deletion"]

        # Get directories configuration
        directories_config = self.get_directories_config()

        total_files = 0
        for directory, directory_config in directories_config.items():
            if not os.path.exists(directory):
                self.logger.warning(f"Directory does not exist: {directory}")
                continue

            self.logger.info(f"Scanning directory: {directory}")

            # Get effective extensions for this directory
            effective_extensions = self.get_effective_extensions_for_directory(directory, directory_config)

            if not effective_extensions:
                self.logger.info(f"No file extensions to process in directory: {directory}")
                continue

            # Create handler for this directory
            handler = FileMonitorHandler(
                self.embedding_manager,
                self.text_extractor,
                effective_extensions,
                exclude_patterns,
                max_file_size,
                delete_embeddings_on_deletion,
                directory,
                directory_config,
            )

            # First pass: count total files to process for progress bar
            files_to_process = []
            for root, dirs, files in os.walk(directory):
                for file in files:
                    file_path = os.path.join(root, file)
                    if handler.should_process_file(file_path):
                        files_to_process.append(file_path)

            # Second pass: process files with progress bar
            if files_to_process:
                with tqdm(
                    files_to_process,
                    desc=f"Processing {os.path.basename(directory)}",
                    unit="files",
                    file=sys.stderr,
                    colour="green",
                ) as pbar:
                    for file_path in pbar:
                        # Update progress bar with current file (last 30 chars)
                        current_file = file_path[-30:] if len(file_path) > 30 else file_path
                        pbar.set_postfix_str(f'"...{current_file}"')
                        self.logger.debug(f"Processing existing file: {file_path}")
                        handler.process_file(file_path)
                        total_files += 1
            else:
                self.logger.info(f"No files to process in {directory}")

        self.logger.info(f"Finished scanning. Processed {total_files} files.")

    def start_monitoring(self):
        """Start monitoring directories for changes"""
        self.logger.info("Starting file monitoring...")

        # Get base configuration
        global_file_extensions = set(self.config["file_extensions"])
        exclude_patterns = self.config["processing"]["exclude_patterns"]
        max_file_size = self.config["processing"]["max_file_size_mb"] * 1024 * 1024
        delete_embeddings_on_deletion = self.config["processing"]["delete_embeddings_on_file_deletion"]

        # Get directories configuration
        directories_config = self.get_directories_config()

        for directory, directory_config in directories_config.items():
            if not os.path.exists(directory):
                self.logger.warning(f"Directory does not exist: {directory}")
                continue
            if directory_config is None:
                directory_config = {}

            # Get effective extensions for this directory
            effective_extensions = self.get_effective_extensions_for_directory(directory, directory_config)

            if not effective_extensions:
                self.logger.info(f"No file extensions to process in directory: {directory}")
                continue

            # Create handler for this directory
            event_handler = FileMonitorHandler(
                self.embedding_manager,
                self.text_extractor,
                effective_extensions,
                exclude_patterns,
                max_file_size,
                delete_embeddings_on_deletion,
                directory,
                directory_config,
            )

            observer = Observer()
            observer.schedule(event_handler, directory, recursive=True)
            observer.start()
            self.observers.append(observer)

            # Log directory configuration
            max_filesize_mb = directory_config.get("max_filesize", 0)
            if max_filesize_mb > 0:
                size_info = f"max_size={max_filesize_mb}MB"
            else:
                size_info = f"max_size={self.config['processing']['max_file_size_mb']}MB(global)"

            self.logger.info(f"Monitoring directory: {directory} (extensions: {sorted(effective_extensions)}, {size_info})")

        if not self.observers:
            self.logger.error("No valid directories to monitor")
            return

        try:
            self.logger.info("File monitoring active. Press Ctrl+C to stop.")
            idle_check_counter = 0
            while True:
                time.sleep(1)
                idle_check_counter += 1

                # Check for idle model every 1 minutes (60 seconds)
                if idle_check_counter % 60 == 0:
                    self.embedding_manager.check_and_unload_idle_model()
                    idle_check_counter = 0  # Reset counter

        except KeyboardInterrupt:
            self.logger.info("Stopping file monitoring...")
            for observer in self.observers:
                observer.stop()
                observer.join()

    def get_directories_config(self) -> Dict[str, Dict]:
        """Parse directories configuration to handle both old and new formats"""
        directories_config = self.config.get("directories", {})

        # Handle legacy format (list of directories)
        if isinstance(directories_config, list):
            self.logger.warning(
                "Using legacy directory configuration format. Consider upgrading to the new per-directory format."
            )
            # Convert to new format with empty configurations
            return {directory: {"ignore_extensions": [], "max_filesize": 0} for directory in directories_config}

        # Handle new format (dictionary with per-directory settings)
        if isinstance(directories_config, dict):
            # Ensure all directories have default values
            for directory, config in directories_config.items():
                if not config:
                    config = {}
                if config is None or "max_filesize" not in config:
                    config["max_filesize"] = 0  # 0 means use global default
                if "ignore_extensions" not in config:
                    config["ignore_extensions"] = []
            return directories_config

        self.logger.error("Invalid directories configuration format")
        return {}

    def get_effective_extensions_for_directory(self, directory: str, directory_config: Dict) -> Set[str]:
        """Get effective file extensions for a directory"""
        global_extensions = set(self.config.get("file_extensions", []))
        if directory_config:
            ignore_extensions = set(directory_config.get("ignore_extensions", []))
        else:
            ignore_extensions = set()

        effective_extensions = global_extensions - ignore_extensions
        self.logger.debug(
            f"Directory {directory}: global={global_extensions}, ignore={ignore_extensions}, effective={effective_extensions}"
        )

        return effective_extensions

    def get_effective_max_filesize_for_directory(self, directory: str, directory_config: Dict) -> int:
        """Get effective max file size for a directory (in MB)"""
        global_max_mb = self.config.get("processing", {}).get("max_file_size_mb", 5)
        directory_max_mb = directory_config.get("max_filesize", 0)

        if directory_max_mb <= 0:
            return global_max_mb
        else:
            return directory_max_mb

    def scan_adhoc_items(self, files: tuple, directories: tuple, recursive: bool = True):
        """Scan specific files and directories provided via command line"""
        self.logger.info("Starting ad-hoc file/directory scanning...")

        # Get base configuration
        global_file_extensions = set(self.config["file_extensions"])
        exclude_patterns = self.config["processing"]["exclude_patterns"]
        max_file_size = self.config["processing"]["max_file_size_mb"] * 1024 * 1024
        delete_embeddings_on_deletion = self.config["processing"]["delete_embeddings_on_file_deletion"]

        # Use default configuration for ad-hoc scanning
        default_config = {"ignore_extensions": [], "max_filesize": 0}

        # Create a single handler for all ad-hoc items
        handler = FileMonitorHandler(
            self.embedding_manager,
            self.text_extractor,
            global_file_extensions,
            exclude_patterns,
            max_file_size,
            delete_embeddings_on_deletion,
            "ad-hoc",  # directory name for logging
            default_config,
        )

        files_to_process = []

        # Process individual files
        for file_path in files:
            file_path = os.path.abspath(file_path)
            if not os.path.exists(file_path):
                self.logger.warning(f"File does not exist: {file_path}")
                continue

            if not os.path.isfile(file_path):
                self.logger.warning(f"Path is not a file: {file_path}")
                continue

            if handler.should_process_file(file_path):
                files_to_process.append(file_path)
                self.logger.info(f"Added file for processing: {file_path}")
            else:
                self.logger.info(f"Skipping file (doesn't meet criteria): {file_path}")

        # Process directories
        for directory in directories:
            directory = os.path.abspath(directory)
            if not os.path.exists(directory):
                self.logger.warning(f"Directory does not exist: {directory}")
                continue

            if not os.path.isdir(directory):
                self.logger.warning(f"Path is not a directory: {directory}")
                continue

            self.logger.info(f"Scanning directory: {directory} (recursive: {recursive})")

            if recursive:
                # Recursive scan
                for root, dirs, files in os.walk(directory):
                    for file in files:
                        file_path = os.path.join(root, file)
                        if handler.should_process_file(file_path):
                            files_to_process.append(file_path)
            else:
                # Non-recursive scan (only immediate files)
                try:
                    for item in os.listdir(directory):
                        file_path = os.path.join(directory, item)
                        if os.path.isfile(file_path) and handler.should_process_file(file_path):
                            files_to_process.append(file_path)
                except PermissionError:
                    self.logger.warning(f"Permission denied accessing directory: {directory}")
                    continue

        # Process all collected files
        if files_to_process:
            self.logger.info(f"Processing {len(files_to_process)} files...")

            with tqdm(
                files_to_process,
                desc="Processing ad-hoc files",
                unit="files",
                file=sys.stderr,
                colour="blue",
            ) as pbar:
                processed_count = 0
                error_count = 0

                for file_path in pbar:
                    # Update progress bar with current file (last 30 chars)
                    current_file = file_path[-30:] if len(file_path) > 30 else file_path
                    pbar.set_postfix_str(f'"...{current_file}"')

                    try:
                        self.logger.debug(f"Processing ad-hoc file: {file_path}")
                        handler.process_file(file_path)
                        processed_count += 1
                    except Exception as e:
                        error_count += 1
                        self.logger.error(f"Failed to process {file_path}: {str(e)}")
                        # Continue processing other files
                        continue

                self.logger.info(f"Ad-hoc scanning completed. Processed: {processed_count}, Errors: {error_count}")
        else:
            self.logger.info("No files found to process in specified files/directories")


@click.command()
@click.option("--config", "-c", default="config.yaml", help="Path to config file")
@click.option("--scan-only", is_flag=True, help="Only scan existing files, don't monitor")
@click.option("--monitor-only", is_flag=True, help="Only monitor for changes, skip initial scan")
@click.option("--add-file", "-f", multiple=True, help="Add specific file(s) to scan (can be used multiple times)")
@click.option(
    "--add-directory", "-d", multiple=True, help="Add specific directory/directories to scan (can be used multiple times)"
)
@click.option("--no-recursive", is_flag=True, help="Scan directories non-recursively (only immediate files)")
def main(config, scan_only, monitor_only, add_file, add_directory, no_recursive):
    """RAG File Monitor - Index files for RAG system"""

    monitor = FileMonitor(config)

    # Handle ad-hoc file/directory scanning
    if add_file or add_directory:
        recursive = not no_recursive  # Default is recursive unless --no-recursive is specified
        monitor.scan_adhoc_items(add_file, add_directory, recursive)
        return

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
