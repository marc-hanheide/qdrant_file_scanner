#!/bin/bash

# Setup script for RAG Search CLI tool
# This script will install the rag-search command-line tool

set -e

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RAG_SEARCH_SCRIPT="$SCRIPT_DIR/rag-search"

# Check if the rag-search script exists
if [ ! -f "$RAG_SEARCH_SCRIPT" ]; then
    echo "Error: rag-search script not found at $RAG_SEARCH_SCRIPT"
    exit 1
fi

# Make sure the script is executable
chmod +x "$RAG_SEARCH_SCRIPT"

# Determine the best location to install the symlink
if [ -d "/usr/local/bin" ] && [ -w "/usr/local/bin" ]; then
    INSTALL_DIR="/usr/local/bin"
elif [ -d "$HOME/.local/bin" ]; then
    INSTALL_DIR="$HOME/.local/bin"
    # Make sure ~/.local/bin is in PATH
    if [[ ":$PATH:" != *":$HOME/.local/bin:"* ]]; then
        echo "Adding $HOME/.local/bin to PATH in ~/.bashrc and ~/.zshrc"
        echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc 2>/dev/null || true
        echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.zshrc 2>/dev/null || true
    fi
else
    # Create ~/.local/bin if it doesn't exist
    mkdir -p "$HOME/.local/bin"
    INSTALL_DIR="$HOME/.local/bin"
    echo "Created $HOME/.local/bin directory"
    
    # Add to PATH
    echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc 2>/dev/null || true
    echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.zshrc 2>/dev/null || true
    echo "Added $HOME/.local/bin to PATH in shell configuration files"
fi

# Create or update the symlink
SYMLINK_PATH="$INSTALL_DIR/rag-search"

if [ -L "$SYMLINK_PATH" ]; then
    echo "Updating existing symlink at $SYMLINK_PATH"
    rm "$SYMLINK_PATH"
elif [ -f "$SYMLINK_PATH" ]; then
    echo "Warning: $SYMLINK_PATH already exists and is not a symlink"
    echo "Please remove it manually if you want to install rag-search"
    exit 1
fi

# Create the symlink
ln -s "$RAG_SEARCH_SCRIPT" "$SYMLINK_PATH"
echo "Installed rag-search to $SYMLINK_PATH"

# Test the installation
if command -v rag-search >/dev/null 2>&1; then
    echo "✓ rag-search is now available in your PATH"
    echo "You can now run: rag-search --help"
else
    echo "⚠ rag-search installed but not found in PATH"
    echo "You may need to:"
    echo "1. Restart your terminal, or"
    echo "2. Run: source ~/.bashrc (or ~/.zshrc)"
    echo "3. Or add $INSTALL_DIR to your PATH manually"
fi

echo ""
echo "Installation complete!"
echo ""
echo "Examples:"
echo "  rag-search --query 'machine learning'"
echo "  rag-search --example ~/Documents/report.pdf"
echo "  rag-search --glob '*.pdf' --start-date '2024-01-01'"
echo "  rag-search --help"
