#!/bin/bash

# Setup script for RAG MCP Server
# This script installs the additional dependencies needed for the MCP server

set -e

echo "RAG MCP Server Setup"
echo "===================="

# Check if we're in a virtual environment
if [[ "$VIRTUAL_ENV" != "" ]]; then
    echo "✓ Virtual environment detected: $VIRTUAL_ENV"
else
    echo "⚠ Warning: No virtual environment detected"
    echo "  It's recommended to run this in a virtual environment"
    read -p "Continue anyway? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Exiting. Please activate a virtual environment first."
        exit 1
    fi
fi

# Install MCP dependencies
echo "Installing MCP dependencies..."
pip install "mcp[cli]>=1.10.0"

# Verify installation
echo "Verifying installation..."
python -c "from mcp.server.fastmcp import FastMCP; print('✓ MCP installed successfully')"

echo ""
echo "✓ MCP server setup complete!"
echo ""
echo "Next steps:"
echo "1. Make sure your RAG system is set up and Qdrant is running"
echo "2. Test the MCP server: python test_mcp_server.py"
echo "3. Start the MCP server: python mcp_server.py"
echo "4. Test with MCP Inspector: mcp dev mcp_server.py"
echo "5. Install in Claude Desktop: mcp install mcp_server.py"
