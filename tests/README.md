# RAG File Scanner Tests

This directory contains comprehensive tests for the RAG File Scanner project.

## Test Structure

### Unit Tests
- `test_text_extractors_unit.py` - Unit tests for text extraction functionality
- `test_embedding_manager.py` - Unit tests for embedding generation and Qdrant operations
- `test_file_monitor.py` - Unit tests for file monitoring and directory configuration
- `test_mcp_server_unit.py` - Unit tests for MCP server functionality

### Integration Tests
- `test_integration.py` - End-to-end integration tests for the complete system

### Legacy Tests (Migrated)
- `test_config.py` - Configuration format testing (migrated from root)
- `test_extractors.py` - Text extractor integration tests (migrated from root)
- `test_mcp_server.py` - MCP server integration tests (migrated from root)

### Test Utilities
- `test_utils.py` - Common test utilities, fixtures, and mock classes
- `run_tests.py` - Test runner script with dependency checking

## Running Tests

### Using the Test Runner (Recommended)
```bash
# Run all tests
python tests/run_tests.py

# Run only unit tests
python tests/run_tests.py --type unit

# Run only integration tests
python tests/run_tests.py --type integration

# Run legacy tests only
python tests/run_tests.py --legacy-only

# Check dependencies
python tests/run_tests.py --check-deps

# Verbose output
python tests/run_tests.py --verbose
```

### Using pytest directly
```bash
# Run all tests
pytest

# Run specific test files
pytest tests/test_text_extractors_unit.py

# Run with specific markers
pytest -m unit
pytest -m integration
pytest -m "not slow"

# Run with verbose output
pytest -v

# Run with coverage
pytest --cov=rag_file_monitor --cov-report=html
```

### Using the legacy test scripts directly
```bash
# Configuration tests
python tests/test_config.py

# Text extractor tests
python tests/test_extractors.py

# MCP server tests
python tests/test_mcp_server.py
```

## Dependencies

### Required Dependencies
- `pytest` - Test framework
- `pyyaml` - Configuration file parsing
- `pathlib` - Path handling (usually built-in)

### Optional Dependencies (some tests will be skipped if not available)
- `sentence-transformers` - Embedding model support
- `qdrant-client` - Vector database client
- `python-docx` - DOCX file support
- `python-pptx` - PPTX file support
- `openpyxl` - XLSX file support
- `PyPDF2` - PDF file support
- `beautifulsoup4` - HTML parsing
- `watchdog` - File system monitoring
- `tqdm` - Progress bars

### Installing Test Dependencies
```bash
# Install basic test dependencies
pip install pytest pyyaml

# Install all optional dependencies
pip install -r requirements.txt

# Install development dependencies
pip install -r requirements-dev.txt
```

## Test Features

### Dependency Detection
Tests automatically detect which dependencies are available and skip tests that require missing dependencies. This ensures tests can run in minimal environments.

### Mock Support
Tests use extensive mocking to avoid requiring external services (like Qdrant) to be running during testing.

### Temporary File Handling
Integration tests create temporary files and directories that are automatically cleaned up after each test.

### Configuration Testing
Tests verify both legacy and new configuration formats work correctly.

### Error Handling
Tests verify that the system handles various error conditions gracefully.

## Test Markers

Tests are marked with the following pytest markers:

- `@pytest.mark.unit` - Unit tests
- `@pytest.mark.integration` - Integration tests
- `@pytest.mark.slow` - Slow-running tests
- `@pytest.mark.requires_deps` - Tests requiring optional dependencies
- `@pytest.mark.requires_qdrant` - Tests requiring Qdrant server
- `@pytest.mark.requires_models` - Tests requiring ML models

## Test Coverage

The test suite covers:

1. **Text Extraction**
   - All supported file formats (TXT, MD, HTML, PDF, DOCX, PPTX, XLSX)
   - Error handling for corrupted/missing files
   - Character encoding detection

2. **Embedding Management**
   - Model loading and lazy initialization
   - Text chunking with overlap
   - Embedding generation
   - Qdrant database operations
   - File hash calculation

3. **File Monitoring**
   - Directory configuration parsing (both formats)
   - File filtering and pattern matching
   - File size limits
   - Extension-based filtering per directory

4. **Configuration Management**
   - YAML configuration loading
   - Legacy vs. new format support
   - Default value handling
   - Configuration validation

5. **MCP Server**
   - Search request handling
   - Response formatting
   - Error handling
   - Tool registration

6. **Integration**
   - End-to-end file processing
   - Configuration-driven behavior
   - Error recovery

## Continuous Integration

The test suite is designed to work in CI environments:

- Tests that require optional dependencies are automatically skipped
- Temporary files are properly cleaned up
- Tests avoid network dependencies
- Mock objects simulate external services

## Adding New Tests

When adding new tests:

1. Place unit tests in the appropriate `test_*_unit.py` file
2. Add integration tests to `test_integration.py`
3. Use appropriate pytest markers
4. Add mock dependencies for external services
5. Clean up any temporary files/directories
6. Update this README if adding new test categories

## Troubleshooting

### Common Issues

1. **ImportError for optional dependencies**
   - This is expected - tests will be skipped automatically
   - Install missing dependencies if you want to run those tests

2. **Temporary file cleanup issues**
   - Tests should clean up automatically
   - Manually remove `/tmp/test_*` directories if needed

3. **Model download in tests**
   - Some tests may download ML models on first run
   - Use `slim_mode=True` in tests to avoid this

4. **Qdrant connection errors**
   - Tests use mocks by default
   - Only integration tests with `@pytest.mark.requires_qdrant` need real Qdrant

### Getting Help

If tests fail unexpectedly:

1. Run with verbose output: `pytest -v`
2. Check dependency status: `python tests/run_tests.py --check-deps`
3. Run individual test files to isolate issues
4. Check the test output for specific error messages
