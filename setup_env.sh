#!/bin/bash
# Setup script for the image annotation project

echo "Setting up Python environment for Image Annotation project..."

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

echo "Setup complete! To use the project:"
echo "1. Run: source venv/bin/activate"
echo "2. Then run your Python scripts"
echo ""
echo "For your IDE, point it to: $(pwd)/venv/bin/python"