#!/usr/bin/env python3
"""
Test runner for RAG File Scanner
"""

import sys
import os
import subprocess
from pathlib import Path

# Add the project root to sys.path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def check_dependencies():
    """Check if required dependencies are available"""
    dependencies = {
        "pytest": "pytest",
        "sentence-transformers": "sentence_transformers", 
        "qdrant-client": "qdrant_client",
        "python-docx": "docx",
        "python-pptx": "pptx",
        "openpyxl": "openpyxl",
        "PyPDF2": "PyPDF2",
        "beautifulsoup4": "bs4",
        "pyyaml": "yaml",
        "watchdog": "watchdog",
        "tqdm": "tqdm",
    }
    
    available = {}
    missing = []
    
    for name, module in dependencies.items():
        try:
            __import__(module)
            available[name] = True
        except ImportError:
            available[name] = False
            missing.append(name)
    
    return available, missing


def run_tests(test_type="all", verbose=False):
    """Run tests with different configurations"""
    
    # Check dependencies
    available, missing = check_dependencies()
    
    print("RAG File Scanner Test Suite")
    print("=" * 50)
    print("\nDependency Status:")
    for dep, status in available.items():
        status_str = "✓" if status else "✗"
        print(f"  {status_str} {dep}")
    
    if missing:
        print(f"\nMissing dependencies: {', '.join(missing)}")
        print("Some tests may be skipped. Install with:")
        print(f"pip install {' '.join(missing)}")
    
    print("\n" + "=" * 50)
    
    # Determine which tests to run
    test_files = []
    
    if test_type == "all" or test_type == "unit":
        test_files.extend([
            "tests/test_text_extractors_unit.py",
            "tests/test_embedding_manager.py", 
            "tests/test_file_monitor.py",
            "tests/test_mcp_server_unit.py"
        ])
    
    if test_type == "all" or test_type == "integration":
        test_files.append("tests/test_integration.py")
    
    if test_type == "all" or test_type == "legacy":
        test_files.extend([
            "tests/test_config.py",
            "tests/test_extractors.py",
            "tests/test_mcp_server.py"
        ])
    
    # Run pytest
    pytest_args = [
        "python", "-m", "pytest",
        "-v" if verbose else "",
        "--tb=short",
        "--strict-markers",
        "--disable-warnings"
    ]
    
    # Filter out empty strings
    pytest_args = [arg for arg in pytest_args if arg]
    
    # Add test files
    pytest_args.extend(test_files)
    
    print(f"Running tests: {test_type}")
    print(f"Command: {' '.join(pytest_args)}")
    print("-" * 50)
    
    try:
        result = subprocess.run(pytest_args, cwd=project_root)
        return result.returncode == 0
    except FileNotFoundError:
        print("Error: pytest not found. Install with: pip install pytest")
        return False


def run_legacy_tests():
    """Run the legacy test scripts directly"""
    print("Running legacy test scripts...")
    print("-" * 50)
    
    legacy_tests = [
        "tests/test_config.py",
        "tests/test_extractors.py", 
        "tests/test_mcp_server.py"
    ]
    
    success_count = 0
    
    for test_file in legacy_tests:
        test_path = project_root / test_file
        if test_path.exists():
            print(f"\nRunning {test_file}...")
            try:
                result = subprocess.run([sys.executable, str(test_path)], 
                                      cwd=project_root,
                                      capture_output=True,
                                      text=True)
                
                if result.returncode == 0:
                    print(f"✓ {test_file} passed")
                    success_count += 1
                else:
                    print(f"✗ {test_file} failed")
                    print("STDOUT:", result.stdout)
                    print("STDERR:", result.stderr)
            except Exception as e:
                print(f"✗ Error running {test_file}: {e}")
        else:
            print(f"✗ {test_file} not found")
    
    print(f"\nLegacy tests completed: {success_count}/{len(legacy_tests)} passed")
    return success_count == len(legacy_tests)


def main():
    """Main test runner"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run RAG File Scanner tests")
    parser.add_argument("--type", choices=["all", "unit", "integration", "legacy"], 
                       default="all", help="Type of tests to run")
    parser.add_argument("--verbose", "-v", action="store_true", 
                       help="Verbose output")
    parser.add_argument("--legacy-only", action="store_true",
                       help="Run only legacy test scripts")
    parser.add_argument("--check-deps", action="store_true",
                       help="Only check dependencies")
    
    args = parser.parse_args()
    
    if args.check_deps:
        available, missing = check_dependencies()
        for dep, status in available.items():
            status_str = "Available" if status else "Missing"
            print(f"{dep}: {status_str}")
        return 0 if not missing else 1
    
    if args.legacy_only:
        success = run_legacy_tests()
        return 0 if success else 1
    
    success = run_tests(args.type, args.verbose)
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
