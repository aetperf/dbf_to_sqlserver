#!/bin/bash
# Build script for creating Linux executable with Nuitka using pip
# Run this on Linux to test the build process before building for Windows

set -e  # Exit on error

echo "========================================"
echo "Building dbf_to_sqlserver with Nuitka (Linux, using pip)"
echo "========================================"
echo

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python 3 is not installed or not in PATH"
    exit 1
fi

echo "Python version: $(python3 --version)"

# Check if pip is installed
if ! command -v pip3 &> /dev/null && ! python3 -m pip --version &> /dev/null; then
    echo "ERROR: pip is not installed"
    echo "Please install pip first"
    exit 1
fi

echo

echo "Step 1: Installing dependencies with pip..."
echo

# Install dependencies using pip
pip3 install -r requirements.txt -r requirements-build.txt

echo
echo "Step 2: Checking Nuitka installation..."
if ! python3 -c "import nuitka" &> /dev/null; then
    echo "ERROR: Nuitka is not available after installation"
    echo "Please check the installation logs above"
    exit 1
fi

echo "Nuitka is installed"

echo
echo "Step 3: Cleaning previous build artifacts..."
rm -rf dbf_to_sqlserver.dist
rm -rf dbf_to_sqlserver.build
rm -rf dbf_to_sqlserver.onefile-build
rm -rf dbf_to_sqlserver

echo "Step 4: Running Nuitka compilation..."
echo "This may take several minutes on first run..."
echo

python3 -m nuitka \
    --standalone \
    --onefile \
    --no-deployment-flag=self-execution \
    --enable-plugin=anti-bloat \
    --assume-yes-for-downloads \
    --follow-imports \
    --include-package=sqlalchemy.dialects.mssql \
    --include-package=dbfread \
    --include-module=pymssql \
    --include-module=pyodbc \
    --nofollow-import-to=pytest \
    --nofollow-import-to=setuptools \
    --nofollow-import-to=distutils \
    --nofollow-import-to=sqlalchemy.testing \
    --output-filename=dbf_to_sqlserver \
    dbf_to_sqlserver.py

echo
echo "========================================"
echo "Build completed successfully!"
echo "========================================"
echo
echo "Executable location: ./dbf_to_sqlserver"
echo "File size: $(du -h dbf_to_sqlserver | cut -f1)"
echo
echo "Make executable (if needed):"
echo "  chmod +x dbf_to_sqlserver"
echo
echo "Test the executable with:"
echo "  ./dbf_to_sqlserver --help"
echo
echo "NOTE: This is a Linux binary and will NOT run on Windows."
echo "Use build.bat or build-uv.bat on Windows to create a Windows .exe"
echo
