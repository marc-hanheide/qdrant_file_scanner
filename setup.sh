#!/bin/bash

set -e -x

# Setup script for RAG File Monitor using pip

echo "Setting up RAG File Monitor with pip..."

# Create virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3.12 -m venv .venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install PyTorch CPU version from PyTorch index
#echo "Installing PyTorch (CPU version)..."
#pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install main dependencies
echo "Installing main dependencies..."
pip install -r requirements.txt

# Install development dependencies
echo "Installing development dependencies..."
pip install -r requirements-dev.txt

# Install the project in development mode
echo "Installing project in development mode..."
pip install -e .

echo "Setup complete! To activate the environment, run:"
echo "source .venv/bin/activate"
