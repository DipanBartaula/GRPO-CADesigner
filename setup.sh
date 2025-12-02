#!/bin/bash

# CAD RL Training Project Setup Script

echo "=========================================="
echo "CAD RL Training Project Setup"
echo "=========================================="

# Create directories
echo "Creating directories..."
mkdir -p checkpoints
mkdir -p data
mkdir -p outputs
mkdir -p logs

echo "✓ Directories created"

# Create virtual environment
echo ""
echo "Creating virtual environment..."
python3 -m venv venv

echo "✓ Virtual environment created"

# Activate virtual environment
echo ""
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo ""
echo "Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo ""
echo "Installing dependencies..."
pip install -r requirements.txt

echo "✓ Dependencies installed"

# Install CLIP
echo ""
echo "Installing CLIP..."
pip install git+https://github.com/openai/CLIP.git

echo "✓ CLIP installed"

# Generate test data
echo ""
echo "Generating test data..."
python create_test_data.py

echo "✓ Test data generated"

# Run core tests
echo ""
echo "Running core component tests..."
python test.py

# Run mesh conversion tests
echo ""
echo "Running mesh conversion tests..."
python run_all_mesh_tests.py

# Setup complete
echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "To start training:"
echo "  1. Activate the environment: source venv/bin/activate"
echo "  2. Login to WandB: wandb login"
echo "  3. Start training: python train.py"
echo ""
echo "To run inference:"
echo "  python inference.py --prompt 'Your prompt here'"
echo ""
echo "=========================================="