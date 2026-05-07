#!/bin/bash

# Activate virtual environment
source .venv/bin/activate

# Initialize database before starting the server
echo "Initializing backend database..."
python -m app.core.database

# Start the backend server
uvicorn app.main:app --reload
