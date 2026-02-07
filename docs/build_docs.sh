#!/bin/bash

# Build script for TensorToolkit documentation with syntax highlighting
# This script generates HTML documentation using Doxygen
# 
# CROSS-PLATFORM SUPPORT:
# ✅ macOS: Uses 'open' command to open browser
# ✅ Linux: Uses 'xdg-open' command to open browser
# ✅ Both: Same Doxygen installation and build process

echo "Building TensorToolkit documentation..."

# Check if Doxygen is installed
if ! command -v doxygen &> /dev/null; then
    echo "Error: Doxygen is not installed. Please install it first:"
    echo "  macOS: brew install doxygen"
    echo "  Ubuntu/Debian: sudo apt-get install doxygen"
    echo "  CentOS/RHEL: sudo yum install doxygen"
    exit 1
fi

# Graphviz is optional, but recommended for diagrams (HAVE_DOT=YES in Doxyfile).
if ! command -v dot &> /dev/null; then
    echo "Warning: Graphviz 'dot' not found. Diagrams will be skipped."
    echo "  macOS: brew install graphviz"
    echo "  Ubuntu/Debian: sudo apt-get install graphviz"
    echo "  CentOS/RHEL: sudo yum install graphviz"
fi

# Create build directory if it doesn't exist
mkdir -p build

# Generate documentation
echo "Running Doxygen..."
doxygen Doxyfile

if [ $? -eq 0 ]; then
    echo "✅ Documentation built successfully!"
    echo "📁 HTML output: build/html/index.html"
    echo "🌐 Open build/html/index.html in your browser to view the documentation"
    
    # Cross-platform browser opening
    # macOS: uses 'open' command
    # Linux: uses 'xdg-open' command
    if command -v open &> /dev/null; then
        echo "🔗 Opening documentation in default browser (macOS)..."
        open build/html/index.html
    elif command -v xdg-open &> /dev/null; then
        echo "🔗 Opening documentation in default browser (Linux)..."
        xdg-open build/html/index.html
    else
        echo "⚠️  Could not automatically open browser. Please open build/html/index.html manually."
    fi
else
    echo "❌ Error building documentation"
    exit 1
fi
