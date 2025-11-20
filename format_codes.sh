#!/bin/zsh

pip install -q black flake8

echo "Using black to format and lint Python files."
black .

echo "Using flake8 to lint all Python files..."
flake8 .
