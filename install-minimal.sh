#!/bin/bash
# Minimal installation script for PAROL6 Python API
# This script installs the package and removes the unnecessary rtb-data package
# to save ~200MB of disk space.

set -e

echo "Installing PAROL6 Python API..."
pip install .

echo ""
echo "Checking if rtb-data was installed as a dependency..."
if pip show rtb-data > /dev/null 2>&1; then
    echo "Found rtb-data package. Removing it (not needed for PAROL6)..."
    pip uninstall -y rtb-data
    echo "✓ rtb-data removed successfully"
else
    echo "rtb-data was not installed"
fi

echo ""
echo "✓ Installation complete!"
echo ""
echo "To launch the controller:"
echo "  parol6-server --log-level=INFO"
