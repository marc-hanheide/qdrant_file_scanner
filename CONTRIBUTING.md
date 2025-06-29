# Contributing to RAG File Monitor

Thank you for your interest in contributing to RAG File Monitor! This document provides guidelines and instructions for contributing to the project.

## Development Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd qdrant_file_scanner
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -e .
   pip install -r requirements-dev.txt
   ```

4. **Install system dependencies**
   - **macOS**: `brew install libmagic`
   - **Ubuntu/Debian**: `sudo apt-get install libmagic1 libmagic-dev`
   - **Windows**: Follow the python-magic installation guide

## Running Tests

### Quick Test Run
```bash
./run_tests.sh
```

### Manual Testing
```bash
# Run all tests
python -m pytest test_*.py -v

# Run tests with coverage
python -m pytest test_*.py -v --cov=rag_file_monitor --cov-report=term-missing

# Run specific test file
python -m pytest test_extractors.py -v
```

### Code Quality Checks
```bash
# Format code
black .

# Check formatting
black --check --diff .

# Lint code
flake8 rag_file_monitor/ mcp_server.py

# Type checking
mypy rag_file_monitor/ --ignore-missing-imports
```

## Testing Your Changes

1. **Unit Tests**: Add tests for new functionality in `test_*.py` files
2. **Integration Tests**: Test the complete workflow
3. **CLI Testing**: Verify CLI commands work correctly
4. **Build Testing**: Ensure the package builds correctly

## Continuous Integration

The project uses GitHub Actions for CI/CD:

- **CI Pipeline** (`.github/workflows/ci.yml`): Runs on every push and PR
  - Tests across multiple Python versions (3.8-3.12)
  - Tests on Ubuntu, macOS, and Windows
  - Builds and tests the package
  - Runs security checks

- **Code Quality** (`.github/workflows/code-quality.yml`): Runs on every PR
  - Code formatting checks
  - Linting
  - Type checking
  - Security scanning

- **Release Pipeline** (`.github/workflows/release.yml`): Runs on releases
  - Builds and publishes to PyPI
  - Creates release artifacts

## Pull Request Process

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature-name`
3. Make your changes
4. Add tests for your changes
5. Run the test suite: `./run_tests.sh`
6. Commit your changes: `git commit -am 'Add some feature'`
7. Push to the branch: `git push origin feature/your-feature-name`
8. Create a Pull Request

## Code Style

- Follow PEP 8 style guidelines
- Use Black for code formatting (line length: 127)
- Add type hints where appropriate
- Write descriptive docstrings for functions and classes
- Keep functions focused and modular

## Reporting Issues

Use the GitHub issue templates:
- **Bug Report**: For reporting bugs
- **Feature Request**: For suggesting new features

## Security

If you discover a security vulnerability, please report it privately by emailing the maintainers rather than opening a public issue.

## License

By contributing to this project, you agree that your contributions will be licensed under the same license as the project.
