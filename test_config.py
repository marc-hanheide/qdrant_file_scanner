#!/usr/bin/env python3
"""
Test script to verify the new directory configuration format works correctly.
"""

import yaml
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
    "processing": {"exclude_patterns": []},
    "logging": {"level": "WARNING", "file": "test.log"}
}

new_config = {
    "directories": {
        "/path/to/dir1": {"ignore_extensions": []},
        "/path/to/dir2": {"ignore_extensions": [".docx"]},
        "/path/to/dir3": {"ignore_extensions": [".pdf", ".docx"]}
    },
    "file_extensions": [".txt", ".pdf", ".docx"],
    "processing": {"exclude_patterns": []},
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
            return {directory: {"ignore_extensions": []} for directory in directories_config}
        
        # Handle new format (dictionary with per-directory settings)
        if isinstance(directories_config, dict):
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

def test_legacy_format():
    print("=== Testing Legacy Format ===")
    monitor = MockFileMonitor(legacy_config)
    directories_config = monitor.get_directories_config()
    
    print(f"Parsed directories: {directories_config}")
    
    for directory, dir_config in directories_config.items():
        effective_ext = monitor.get_effective_extensions_for_directory(directory, dir_config)
        print(f"Effective extensions for {directory}: {effective_ext}")
    
    print()

def test_new_format():
    print("=== Testing New Format ===")
    monitor = MockFileMonitor(new_config)
    directories_config = monitor.get_directories_config()
    
    print(f"Parsed directories: {directories_config}")
    
    for directory, dir_config in directories_config.items():
        effective_ext = monitor.get_effective_extensions_for_directory(directory, dir_config)
        print(f"Effective extensions for {directory}: {effective_ext}")
    
    print()

def test_config_yaml():
    print("=== Testing Actual config.yaml ===")
    config_path = Path(__file__).parent / "config.yaml"
    
    if config_path.exists():
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        
        monitor = MockFileMonitor(config)
        directories_config = monitor.get_directories_config()
        
        print(f"Found {len(directories_config)} directories in config.yaml")
        
        for directory, dir_config in directories_config.items():
            effective_ext = monitor.get_effective_extensions_for_directory(directory, dir_config)
            print(f"Directory: {directory}")
            print(f"  Ignore: {dir_config.get('ignore_extensions', [])}")
            print(f"  Effective: {sorted(effective_ext)}")
    else:
        print("config.yaml not found")

if __name__ == "__main__":
    test_legacy_format()
    test_new_format()
    test_config_yaml()
