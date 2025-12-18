# Quick Setup Guide

## Prerequisites

- Python 3.8+
- Trained model weights (`.pth` file) from ePillID training
- Reference pill images for building the index

## Quick Start

### 1. Install Dependencies

```bash
cd backend/pill-identification
pip install -r requirements.txt
```

### 2. Prepare Model Weights

Place your trained model file (`.pth`) in:
```
backend/pill-identification/models/pill_model.pth
```

Or set `PILL_MODEL_PATH` environment variable to point to your model file.

### 3. Build Reference Index

```bash
python generate_reference_index.py \
    --data_dir /path/to/reference/images \
    --output_dir ./data \
    --model_path ./models/pill_model.pth
```

This will create:
- `./data/pill_index.index` (FAISS index)
- `./data/pill_metadata.json` (metadata)

### 4. Start Service

```bash
# Option 1: Use startup script
./start_service.sh

# Option 2: Direct uvicorn
uvicorn api.app:app --host 127.0.0.1 --port 8005 --reload
```

### 5. Test the Service

```bash
# Health check
curl http://127.0.0.1:8005/health

# Get service info
curl http://127.0.0.1:8005/info
```

## Environment Variables

All configuration is done via environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `PILL_MODEL_PATH` | `./models/pill_model.pth` | Path to model weights |
| `PILL_INDEX_PATH` | `./data/pill_index.index` | Path to FAISS index |
| `PILL_METADATA_PATH` | `./data/pill_metadata.json` | Path to metadata JSON |
| `PILL_NETWORK` | `resnet18` | CNN backbone architecture |
| `PILL_EMBEDDING_DIM` | `2048` | Embedding dimension |
| `PILL_DEVICE` | `cpu` | Device (`cpu` or `cuda`) |
| `PILL_METRIC` | `cosine` | Distance metric (`cosine` or `l2`) |

## File Structure After Setup

```
pill-identification/
├── models/
│   └── pill_model.pth          # Your trained model
├── data/
│   ├── pill_index.index        # Generated FAISS index
│   └── pill_metadata.json       # Generated metadata
├── api/
│   └── app.py                  # FastAPI service
└── ...
```

## Troubleshooting

### Model file not found
- Check that `PILL_MODEL_PATH` points to the correct file
- Verify the file exists and is readable

### Index not found
- Run `generate_reference_index.py` first
- Check that `PILL_INDEX_PATH` is correct

### CUDA errors
- Set `PILL_DEVICE=cpu` to use CPU
- Install `faiss-gpu` for GPU support

### Import errors
- Ensure all dependencies are installed: `pip install -r requirements.txt`
- Check Python version: `python --version` (should be 3.8+)

## Next Steps

See `README.md` for detailed API documentation and usage examples.







