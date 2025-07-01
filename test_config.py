#!/usr/bin/env python3
"""
Test script to verify the new directory configuration format works correctly.
"""

import tempfile
import os
from pathlib import Path

# Test configurations
legacy_config = {
    "directories": [
        "/path/to/dir1",
        "/path/to/dir2"
    ],
    "file_extensions": [".txt", ".pdf", ".docx"],
    "processing": {"exclude_patterns": [], "max_file_size_mb": 5},
    "logging": {"level": "WARNING", "file": "test.log"}
}

new_config = {
    "directories": {
        "/path/to/dir1": {"ignore_extensions": [], "max_filesize": 0},
        "/path/to/dir2": {"ignore_extensions": [".docx"], "max_filesize": 2},
        "/path/to/dir3": {"ignore_extensions": [".pdf", ".docx"], "max_filesize": 10}
    },
    "file_extensions": [".txt", ".pdf", ".docx"],
    "processing": {"exclude_patterns": [], "max_file_size_mb": 5},
    "logging": {"level": "WARNING", "file": "test.log"}
}

# Mock the FileMonitor class with just the methods we need to test
class MockFileMonitor:
    def __init__(self, config):
        self.config = config

    def get_directories_config(self):
        """Parse directories configuration to handle both old and new formats"""
        directories_config = self.config.get("directories", {})
        
        # Handle legacy format (list of directories)
        if isinstance(directories_config, list):
            print("Using legacy directory configuration format")
            return {directory: {"ignore_extensions": [], "max_filesize": 0} for directory in directories_config}
        
        # Handle new format (dictionary with per-directory settings)
        if isinstance(directories_config, dict):
            # Ensure all directories have default values
            for directory, config in directories_config.items():
                if "max_filesize" not in config:
                    config["max_filesize"] = 0  # 0 means use global default
                if "ignore_extensions" not in config:
                    config["ignore_extensions"] = []
            return directories_config
        
        print("Invalid directories configuration format")
        return {}

    def get_effective_extensions_for_directory(self, directory, directory_config):
        """Get effective file extensions for a directory"""
        global_extensions = set(self.config.get("file_extensions", []))
        ignore_extensions = set(directory_config.get("ignore_extensions", []))
        
        effective_extensions = global_extensions - ignore_extensions
        print(f"Directory {directory}: global={global_extensions}, ignore={ignore_extensions}, effective={effective_extensions}")
        
        return effective_extensions
    
    def get_effective_max_filesize_for_directory(self, directory, directory_config):
        """Get effective max file size for a directory"""
        global_max_mb = self.config.get("processing", {}).get("max_file_size_mb", 5)
        directory_max_mb = directory_config.get("max_filesize", 0)
        
        if directory_max_mb <= 0:
            effective_max_mb = global_max_mb
            source = "global"
        else:
            effective_max_mb = directory_max_mb
            source = "directory"
        
        print(f"Directory {directory}: max_filesize={effective_max_mb}MB ({source})")
        return effective_max_mb

def test_legacy_format():
    print("=== Testing Legacy Format ===")
    monitor = MockFileMonitor(legacy_config)
    directories_config = monitor.get_directories_config()
    
    print(f"Parsed directories: {directories_config}")
    
    for directory, dir_config in directories_config.items():
        effective_ext = monitor.get_effective_extensions_for_directory(directory, dir_config)
        effective_max = monitor.get_effective_max_filesize_for_directory(directory, dir_config)
        print(f"Effective extensions for {directory}: {effective_ext}")
    
    print()

def test_new_format():
    print("=== Testing New Format ===")
    monitor = MockFileMonitor(new_config)
    directories_config = monitor.get_directories_config()
    
    print(f"Parsed directories: {directories_config}")
    
    for directory, dir_config in directories_config.items():
        effective_ext = monitor.get_effective_extensions_for_directory(directory, dir_config)
        effective_max = monitor.get_effective_max_filesize_for_directory(directory, dir_config)
        print(f"Effective extensions for {directory}: {effective_ext}")
    
    print()

def test_config_yaml():
    print("=== Testing Actual config.yaml ===")
    config_path = Path(__file__).parent / "config.yaml"
    
    if config_path.exists():
        try:
            import yaml
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)
        except ImportError:
            print("PyYAML not available, skipping config.yaml test")
            return
        
        monitor = MockFileMonitor(config)
        directories_config = monitor.get_directories_config()
        
        print(f"Found {len(directories_config)} directories in config.yaml")
        
        for directory, dir_config in directories_config.items():
            effective_ext = monitor.get_effective_extensions_for_directory(directory, dir_config)
            effective_max = monitor.get_effective_max_filesize_for_directory(directory, dir_config)
            print(f"Directory: {directory}")
            print(f"  Ignore: {dir_config.get('ignore_extensions', [])}")
            print(f"  Max size: {dir_config.get('max_filesize', 0)}MB")
            print(f"  Effective extensions: {sorted(effective_ext)}")
    else:
        print("config.yaml not found")

if __name__ == "__main__":
    test_legacy_format()
    test_new_format()
    test_config_yaml()
