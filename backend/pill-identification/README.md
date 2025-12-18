# Pill Identification Module

A production-ready pill identification system based on the ePillID-benchmark architecture. This module provides:

- **CNN-based embedding generation** using ResNet backbones with bilinear pooling
- **FAISS vector search** for fast similarity matching
- **FastAPI microservice** for REST API access
- **Reference embedding generation** script for building search indices

## Architecture

The system follows the ePillID-benchmark architecture:

1. **EmbeddingModel**: CNN encoder (ResNet18/50) → Global Average Pooling → Embedding head
2. **Preprocessing**: ImageNet normalization (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
3. **Embedding Generation**: L2-normalized 2048-dimensional vectors
4. **Vector Search**: FAISS-based cosine similarity search

## Installation

### 1. Install Dependencies

```bash
cd backend/pill-identification
pip install -r requirements.txt
```

**Note**: For GPU support, replace `faiss-cpu` with `faiss-gpu` in requirements.txt.

### 2. Model Weights

You need trained model weights (`.pth` file) from the ePillID training process. Place the model file in a location accessible to the service.

**Default location**: `backend/pill-identification/models/pill_model.pth`

You can specify a custom path using the `PILL_MODEL_PATH` environment variable.

## Setup

### Step 1: Generate Reference Embeddings

Before using the identification service, you need to build a FAISS index of reference pill embeddings.

```bash
python generate_reference_index.py \
    --data_dir /path/to/reference/pill/images \
    --output_dir ./data \
    --model_path ./models/pill_model.pth \
    --network resnet18 \
    --embedding_dim 2048 \
    --batch_size 32
```

**Options**:
- `--data_dir`: Directory containing reference pill images
- `--metadata_csv`: (Optional) CSV file with image paths and metadata columns
- `--output_dir`: Where to save the index and metadata
- `--model_path`: Path to trained model weights
- `--network`: CNN backbone (resnet18, resnet50, etc.)
- `--embedding_dim`: Embedding dimension (default: 2048)
- `--batch_size`: Batch size for processing (default: 32)
- `--metric`: Distance metric ('cosine' or 'l2', default: 'cosine')

**CSV Format** (if using `--metadata_csv`):
```csv
image_path,pill_id,medication_name,shape,color,imprint
/path/to/image1.jpg,pill_001,Aspirin,round,white,ASA
/path/to/image2.jpg,pill_002,Ibuprofen,oval,white,IBU
```

### Step 2: Configure Environment Variables

Set environment variables for the FastAPI service:

```bash
export PILL_MODEL_PATH=./models/pill_model.pth
export PILL_INDEX_PATH=./data/pill_index.index
export PILL_METADATA_PATH=./data/pill_metadata.json
export PILL_NETWORK=resnet18
export PILL_EMBEDDING_DIM=2048
export PILL_DEVICE=cuda  # or 'cpu'
export PILL_METRIC=cosine
```

### Step 3: Run the FastAPI Service

```bash
cd backend/pill-identification
uvicorn api.app:app --host 127.0.0.1 --port 8005 --reload
```

The service will be available at `http://127.0.0.1:8005`

## API Endpoints

### `GET /`
Service information and available endpoints.

### `GET /health`
Health check endpoint. Returns service status.

### `GET /info`
Get model and index information.

### `POST /identify`
Identify a pill from a base64-encoded image.

**Request Body**:
```json
{
  "image": "data:image/jpeg;base64,/9j/4AAQSkZJRg...",
  "k": 5,
  "min_confidence": 0.5
}
```

**Response**:
```json
{
  "top_match": {
    "rank": 1,
    "index": 42,
    "confidence": 0.95,
    "metadata": {
      "pill_id": "pill_001",
      "medication_name": "Aspirin",
      "image_path": "/path/to/reference.jpg"
    }
  },
  "candidates": [
    {
      "rank": 1,
      "index": 42,
      "confidence": 0.95,
      "metadata": {...}
    },
    ...
  ],
  "num_results": 5,
  "embedding_dim": 2048
}
```

### `POST /embed`
Generate embedding vector for an image.

**Request Body**:
```json
{
  "image": "data:image/jpeg;base64,/9j/4AAQSkZJRg..."
}
```

**Response**:
```json
{
  "embedding": [0.123, -0.456, ...],
  "dimension": 2048
}
```

### `POST /identify/file`
Identify a pill from an uploaded image file.

**Form Data**:
- `file`: Image file (multipart/form-data)
- `k`: Number of results (optional, default: 5)
- `min_confidence`: Minimum confidence threshold (optional, default: 0.0)

### `POST /embed/file`
Generate embedding for an uploaded image file.

**Form Data**:
- `file`: Image file (multipart/form-data)

## Usage Examples

### Python Client

```python
import requests
import base64

# Read image
with open('pill_image.jpg', 'rb') as f:
    image_data = base64.b64encode(f.read()).decode('utf-8')

# Identify pill
response = requests.post(
    'http://127.0.0.1:8005/identify',
    json={
        'image': f'data:image/jpeg;base64,{image_data}',
        'k': 5,
        'min_confidence': 0.5
    }
)

result = response.json()
print(f"Top match: {result['top_match']['metadata']['medication_name']}")
print(f"Confidence: {result['top_match']['confidence']:.2f}")
```

### cURL

```bash
# Identify pill
curl -X POST http://127.0.0.1:8005/identify \
  -H "Content-Type: application/json" \
  -d '{
    "image": "data:image/jpeg;base64,...",
    "k": 5
  }'

# Generate embedding
curl -X POST http://127.0.0.1:8005/embed \
  -H "Content-Type: application/json" \
  -d '{
    "image": "data:image/jpeg;base64,..."
  }'
```

## Directory Structure

```
pill-identification/
├── __init__.py
├── inference.py              # Main inference class
├── generate_reference_index.py  # Script to build FAISS index
├── requirements.txt
├── README.md
├── models/
│   ├── __init__.py
│   └── embedding_model.py   # Model architecture
├── utils/
│   ├── __init__.py
│   ├── preprocessing.py     # Image preprocessing
│   ├── embedding.py         # Embedding generation
│   └── vector_search.py     # FAISS vector search
├── api/
│   ├── __init__.py
│   └── app.py               # FastAPI service
└── data/                    # Generated index files
    ├── pill_index.index
    └── pill_metadata.json
```

## Error Handling

The service includes comprehensive error handling for:

- Invalid base64 images
- Unreadable image files
- Missing FAISS index
- Missing model weights
- Mismatched embedding dimensions
- Empty search results

All errors return structured JSON responses with error messages.

## Integration with Existing Backend

This module is self-contained and does not modify existing medication tracking or metric logging systems. To integrate:

1. The FastAPI service runs independently on port 8005
2. Your existing backend can call the service via HTTP
3. No changes to existing routes or controllers are required

## Model Training

This module is for **inference only**. Model training should be done using the original ePillID-benchmark repository:

1. Train the model using the ePillID training scripts
2. Save the model weights (`.pth` file)
3. Use those weights with this inference module

## Performance Notes

- **CPU**: Suitable for development and low-volume usage
- **GPU**: Recommended for production (use `faiss-gpu` and set `PILL_DEVICE=cuda`)
- **Batch Processing**: The embedding generation script processes images in batches for efficiency
- **Index Size**: FAISS indices are memory-efficient and support millions of vectors

## Troubleshooting

### "Identifier not initialized"
- Check that model weights file exists at the specified path
- Verify FAISS index and metadata files exist
- Check environment variables are set correctly

### "Index is empty"
- Run `generate_reference_index.py` to build the index
- Verify image paths in your data directory

### "CUDA out of memory"
- Reduce batch size in `generate_reference_index.py`
- Use CPU mode: `export PILL_DEVICE=cpu`

### Import errors
- Ensure all dependencies are installed: `pip install -r requirements.txt`
- Check Python version (3.8+ required)

## License

This module is based on the ePillID-benchmark architecture. Please refer to the original repository for licensing information.

## References

- ePillID-benchmark: https://github.com/usuyama/ePillID-benchmark
- Paper: "ePillID Dataset: A Low-Shot Fine-Grained Benchmark for Pill Identification" (CVPR 2020)







