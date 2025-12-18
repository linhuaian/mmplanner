#!/bin/bash

# MMPlanner Environment Setup Script
# This script creates a local virtual environment and installs all required dependencies

set -e  # Exit on any error

echo "=========================================="
echo "  MMPlanner Environment Setup"
echo "=========================================="

# Define venv directory
VENV_DIR="venv"

# Check if venv already exists
if [ -d "$VENV_DIR" ]; then
    echo "Virtual environment already exists at ./$VENV_DIR"
    read -p "Do you want to recreate it? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Removing existing virtual environment..."
        rm -rf "$VENV_DIR"
    else
        echo "Using existing virtual environment."
        source "$VENV_DIR/bin/activate"
        echo "Environment activated. Run 'deactivate' to exit."
        exit 0
    fi
fi

# Create virtual environment
echo ""
echo "[1/5] Creating virtual environment..."
python3 -m venv "$VENV_DIR"

# Activate virtual environment
echo "[2/5] Activating virtual environment..."
source "$VENV_DIR/bin/activate"

# Upgrade pip
echo "[3/5] Upgrading pip..."
pip install --upgrade pip

# Install PyTorch with CUDA 12.1 support
echo "[4/5] Installing PyTorch with CUDA 12.1 support..."
pip install torch==2.2.1 torchvision==0.17.1 --index-url https://download.pytorch.org/whl/cu121

# Install all other dependencies
echo "[5/5] Installing dependencies..."
pip install \
    openai \
    python-dotenv \
    pandas \
    pillow \
    tqdm \
    requests \
    accelerate \
    diffusers>=0.30.3 \
    peft>=0.17.0 \
    transformers>=4.45.0 \
    qwen-vl-utils

echo ""
echo "=========================================="
echo "  Setup Complete!"
echo "=========================================="
echo ""
echo "To activate the environment, run:"
echo "  source venv/bin/activate"
echo ""
echo "To deactivate, run:"
echo "  deactivate"
echo ""
echo "Note: Make sure to create a .env file with your OPENAI_TOKEN:"
echo "  echo 'OPENAI_TOKEN=your_api_key_here' > .env"
echo ""
