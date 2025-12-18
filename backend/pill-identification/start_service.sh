#!/bin/bash
# Startup script for Pill Identification FastAPI service

# Set default environment variables if not set
export PILL_MODEL_PATH=${PILL_MODEL_PATH:-"./models/pill_model.pth"}
export PILL_INDEX_PATH=${PILL_INDEX_PATH:-"./data/pill_index.index"}
export PILL_METADATA_PATH=${PILL_METADATA_PATH:-"./data/pill_metadata.json"}
export PILL_NETWORK=${PILL_NETWORK:-"resnet18"}
export PILL_EMBEDDING_DIM=${PILL_EMBEDDING_DIM:-"2048"}
export PILL_DEVICE=${PILL_DEVICE:-"cpu"}
export PILL_METRIC=${PILL_METRIC:-"cosine"}

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Check if virtual environment exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Start the service
echo "Starting Pill Identification API on http://127.0.0.1:8005"
echo "Model: $PILL_MODEL_PATH"
echo "Index: $PILL_INDEX_PATH"
echo "Device: $PILL_DEVICE"
echo ""

uvicorn api.app:app --host 127.0.0.1 --port 8005 --reload







