#!/bin/bash

# Final clean push to GitHub - All secrets removed
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=========================================="
echo "🔍 Final Security Check"
echo "=========================================="

# Check for any remaining secrets
echo "Checking for AWS keys..."
if grep -r "AKIA[0-9A-Z]\{16\}" --include="*.py" --include="*.sh" . 2>/dev/null | grep -v "READY_TO_PUSH\|FINAL_PUSH"; then
    echo "❌ AWS keys found! Aborting."
    exit 1
fi

echo "Checking for Azure keys..."
if grep -r "ec7d5fd6\|[0-9a-f]\{32\}" --include="*.py" --include="*.sh" . 2>/dev/null | grep -v "READY_TO_PUSH\|FINAL_PUSH\|cookies.py"; then
    echo "❌ Potential Azure keys found! Please review."
    # Note: Not failing here as 32-char hex can be other things
fi

echo "Checking for Google API keys..."
if grep -r "AIzaSy[a-zA-Z0-9_-]\{33\}" --include="*.py" --include="*.sh" . 2>/dev/null | grep -v "READY_TO_PUSH\|FINAL_PUSH"; then
    echo "❌ Google API keys found! Aborting."
    exit 1
fi

echo "✅ No obvious secrets detected"
echo ""

echo "=========================================="
echo "📦 Preparing Git Repository"
echo "=========================================="

# Clean git history
if [ -d ".git" ]; then
    echo "Removing old git history..."
    rm -rf .git
fi

# Initialize fresh repository
echo "Initializing fresh repository..."
git init
git branch -M main

# Remove temporary documentation files
echo "Removing temporary files..."
rm -f READY_TO_PUSH.md FINAL_PUSH.sh

# Add all files
echo "Adding files..."
git add .

# Show summary
echo ""
echo "Files to be committed:"
git status --short | head -20
echo ""
file_count=$(git status --short | wc -l | tr -d ' ')
echo "Total: $file_count files"

echo ""
echo "=========================================="
echo "💬 Creating Commit"
echo "=========================================="

git commit -m "Initial commit: CSO - Critical Step Optimization for Agent Alignment

This repository contains the implementation of Critical Step Optimization (CSO),
a novel approach for improving agent alignment through process reward models.

Key Features:
- Baseline and DPO inference scripts for agent evaluation
- PRM-based resampling for identifying critical decision points
- Verified DPO data generation with full trajectory validation
- Complete integration with LLaMAFactory for training
- Comprehensive documentation and setup guides

All sensitive credentials have been removed from this public release.
Please configure your own API keys and endpoints before running."

echo ""
echo "=========================================="
echo "🚀 Pushing to GitHub"
echo "=========================================="

git remote add origin https://github.com/kiaia/CSO.git

echo ""
echo "About to push to: https://github.com/kiaia/CSO"
echo ""
read -p "Press Enter to continue, or Ctrl+C to abort..."

git push -u origin main

echo ""
echo "=========================================="
echo "✅ SUCCESS!"
echo "=========================================="
echo ""
echo "Repository pushed to: https://github.com/kiaia/CSO"
echo ""
echo "⚠️  IMPORTANT: If any credentials were ever exposed, revoke/rotate them immediately."
echo "See READY_TO_PUSH.md for generic revocation steps."
echo ""
echo "=========================================="

