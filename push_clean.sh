#!/bin/bash

# Push clean CSO code to fresh GitHub repository
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=========================================="
echo "Pushing Clean CSO Code to GitHub"
echo "=========================================="

# Remove old git history if exists
if [ -d ".git" ]; then
    echo "Removing old git history..."
    rm -rf .git
fi

# Initialize new repository
echo "Initializing git repository..."
git init
git branch -M main

# Add all files
echo "Adding files..."
git add .

# Show what will be committed
echo ""
echo "Files to be committed:"
git status --short

# Create initial commit
echo ""
echo "Creating commit..."
git commit -m "Initial commit: CSO - Critical Step Optimization for Agent Alignment

- Clean repository with all sensitive data removed
- Baseline and DPO inference scripts
- PRM resampling and verified DPO generation tools
- Complete documentation and setup guide"

# Add remote
echo "Adding remote..."
git remote add origin https://github.com/kiaia/CSO.git

# Push to GitHub
echo ""
echo "Pushing to GitHub..."
git push -u origin main

echo ""
echo "=========================================="
echo "✅ Successfully pushed to https://github.com/kiaia/CSO"
echo "=========================================="
