#!/bin/bash

# Activate the virtual environment if using one
# source venv/bin/activate

echo "Running unit tests..."
python -m unittest discover -s tests

# Check the exit code of the test run
if [ $? -ne 0 ]; then
    echo "Unit tests failed!"
    exit 1
fi

echo "Checking static typing with mypy..."
mypy .

# Check the exit code of mypy
if [ $? -ne 0 ]; then
    echo "Static typing check failed!"
    exit 1
fi

# Auto format the code
black src tests --line-length 120
if [ $? -ne 0 ]; then
	echo "Formatting with black failed!"
	exit 1 
fi

echo "All checks passed successfully!"

