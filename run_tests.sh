#!/bin/bash
# Test runner script for local development

set -e

echo "🧪 Running RAG File Monitor Tests"
echo "================================="

# Check if virtual environment is activated
if [[ "$VIRTUAL_ENV" == "" ]]; then
    echo "⚠️  Warning: No virtual environment detected. Consider activating one."
fi

# Install dev dependencies if not already installed
echo "📦 Installing development dependencies..."
pip install -r requirements-dev.txt

# Run code formatting check
echo "🎨 Checking code formatting with black..."
black --check --diff .

# Run linting
echo "🔍 Running linting with flake8..."
flake8 rag_file_monitor/ mcp_server.py --count --select=E9,F63,F7,F82 --show-source --statistics
flake8 rag_file_monitor/ mcp_server.py --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics

# Run tests
echo "🧪 Running tests with pytest..."
python -m pytest test_*.py -v --tb=short --cov=rag_file_monitor --cov-report=term-missing

# Test package build
echo "📦 Testing package build..."
pip install build
python -m build

# Test CLI commands
echo "🎯 Testing CLI commands..."
rag-monitor --help
rag-search --help 
rag-manage --help

echo "✅ All tests passed!"
