# Pill Identification System - Complete Implementation

## 🎯 Overview

A complete, production-ready pill identification system based on the ePillID-benchmark architecture. This implementation provides:

- ✅ **Automatic dataset preparation** - Detects and organizes ePillID dataset
- ✅ **Flexible model training** - Train new models or use existing ones
- ✅ **Automatic model detection** - Finds and loads models automatically
- ✅ **FAISS vector search** - Fast similarity matching
- ✅ **FastAPI microservice** - REST API on localhost:8005
- ✅ **Complete pipeline** - End-to-end automation

## 🚀 Quick Start

### 1. Install Dependencies

```bash
cd backend/pill-identification
pip install -r requirements.txt
```

### 2. Run Complete Pipeline

```bash
python pipeline.py --data_root /path/to/epillid/data --start_service
```

This single command will:
1. Prepare the dataset
2. Train a model (or use existing)
3. Build the FAISS index
4. Start the inference service

## 📁 Project Structure

```
pill-identification/
├── dataset/
│   ├── __init__.py
│   └── prepare.py              # Dataset preparation & organization
├── training/
│   ├── __init__.py
│   ├── trainer.py              # Training logic (Triplet/ArcFace)
│   └── train.py                # Training script
├── models/
│   ├── embedding_model.py      # CNN encoder + embedding head
│   └── margin_linear.py        # ArcFace margin layer
├── utils/
│   ├── preprocessing.py        # Image preprocessing
│   ├── embedding.py            # Embedding generation
│   ├── vector_search.py        # FAISS search
│   └── model_detector.py       # Auto-detect models/indices
├── api/
│   └── app.py                  # FastAPI service
├── pipeline.py                 # Complete automation script
├── build_index.py              # Index generation
├── inference.py                # Main inference class
├── generate_reference_index.py # Legacy index script
├── test_service.py             # Test script
├── start_service.sh            # Startup script
├── requirements.txt            # Dependencies
├── README.md                   # Basic documentation
├── SETUP.md                    # Quick setup guide
├── COMPLETE_GUIDE.md           # Detailed guide
└── README_FULL.md              # This file
```

## 🔧 Components

### 1. Dataset Preparation (`dataset/prepare.py`)

**Features:**
- Automatically detects dataset structure (CSV or folder-based)
- Organizes reference and consumer images
- Creates train/val/test splits
- Generates metadata and label encoders

**Usage:**
```python
from dataset.prepare import DatasetPreparer

preparer = DatasetPreparer('/path/to/epillid_data')
result = preparer.prepare_dataset(output_dir='./prepared')
```

### 2. Model Training (`training/trainer.py`)

**Features:**
- Modern PyTorch 2.x implementation
- Triplet loss or ArcFace loss
- Automatic checkpointing
- Learning rate scheduling
- GPU/CPU support

**Architecture:**
- CNN Backbone: ResNet18/34/50
- Pooling: Global Average Pooling
- Embedding: 2048-dimensional L2-normalized vectors
- Loss: Triplet (metric learning) or ArcFace (classification)

**Usage:**
```bash
python training/train.py \
    --data_root /path/to/data \
    --prepared_dir ./prepared \
    --output_dir ./models \
    --network resnet18 \
    --num_epochs 50
```

### 3. Model Detection (`utils/model_detector.py`)

**Features:**
- Automatically searches for existing models
- Validates model compatibility
- Extracts model metadata

**Search Locations:**
- `pill-identification/models/`
- `models/`
- `ml-service/pretrained-models/`

### 4. Index Building (`build_index.py`)

**Features:**
- Generates embeddings for reference images
- Builds FAISS index
- Saves metadata

**Usage:**
```bash
python build_index.py \
    --prepared_dir ./prepared \
    --model_path ./models/model_final.pth \
    --output_dir ./data
```

### 5. Inference (`inference.py`)

**Features:**
- Automatic model and index detection
- Image preprocessing
- Embedding generation
- Similarity search

**Usage:**
```python
from inference import PillIdentifier

identifier = PillIdentifier(auto_detect=True)
result = identifier.identify('pill_image.jpg', k=5)
```

### 6. FastAPI Service (`api/app.py`)

**Endpoints:**
- `POST /identify` - Identify pill from image
- `POST /embed` - Generate embedding
- `GET /health` - Health check
- `GET /info` - Model/index info

**Features:**
- Automatic initialization
- Base64 and file upload support
- Comprehensive error handling
- Localhost-only (127.0.0.1:8005)

## 📖 Usage Examples

### Complete Pipeline

```bash
# Run everything
python pipeline.py --data_root /path/to/data --start_service

# Skip training (use existing model)
python pipeline.py --data_root /path/to/data --skip_training --start_service

# Skip index building
python pipeline.py --data_root /path/to/data --skip_index --start_service
```

### Step-by-Step

```bash
# 1. Prepare dataset
python -c "from dataset.prepare import prepare_dataset; prepare_dataset('/path/to/data', './prepared')"

# 2. Train model
python training/train.py --data_root /path/to/data --prepared_dir ./prepared --output_dir ./models

# 3. Build index
python build_index.py --prepared_dir ./prepared --output_dir ./data

# 4. Start service
uvicorn api.app:app --host 127.0.0.1 --port 8005
```

### Python API

```python
from inference import PillIdentifier

# Auto-detect everything
identifier = PillIdentifier()

# Identify pill
result = identifier.identify('pill.jpg', k=5)
print(f"Top match: {result['top_match']['metadata']['label']}")
print(f"Confidence: {result['top_match']['confidence']:.2f}")

# Generate embedding
embedding = identifier.embed('pill.jpg')
print(f"Embedding shape: {embedding.shape}")
```

### REST API

```python
import requests
import base64

# Read image
with open('pill.jpg', 'rb') as f:
    img_data = base64.b64encode(f.read()).decode('utf-8')

# Identify
response = requests.post(
    'http://127.0.0.1:8005/identify',
    json={'image': f'data:image/jpeg;base64,{img_data}', 'k': 5}
)

result = response.json()
print(result['top_match'])
```

## 🔍 Automatic Detection

The system automatically detects:

1. **Models**: Searches common directories for `.pth` files
2. **Indices**: Looks for `pill_index.index` in data directories
3. **Metadata**: Finds corresponding `pill_metadata.json` files
4. **Architecture**: Extracts network and embedding_dim from checkpoints

**Enable/Disable:**
```bash
export PILL_AUTO_DETECT=true  # Default: true
```

## ⚙️ Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `PILL_MODEL_PATH` | Auto-detect | Path to model weights |
| `PILL_INDEX_PATH` | Auto-detect | Path to FAISS index |
| `PILL_METADATA_PATH` | Auto-detect | Path to metadata JSON |
| `PILL_NETWORK` | `resnet18` | CNN backbone |
| `PILL_EMBEDDING_DIM` | `2048` | Embedding dimension |
| `PILL_DEVICE` | Auto | Device (`cpu` or `cuda`) |
| `PILL_METRIC` | `cosine` | Distance metric |
| `PILL_AUTO_DETECT` | `true` | Enable auto-detection |

### Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--network` | `resnet18` | CNN backbone |
| `--embedding_dim` | `2048` | Embedding dimension |
| `--loss_type` | `triplet` | Loss function |
| `--margin` | `1.0` | Triplet loss margin |
| `--num_epochs` | `50` | Training epochs |
| `--batch_size` | `32` | Batch size |
| `--learning_rate` | `1e-4` | Initial LR |

## 📊 Architecture Details

### Model Architecture

```
Input Image (224×224×3)
    ↓
ResNet18/50 Backbone
    ↓
Global Average Pooling
    ↓
Dropout (0.5)
    ↓
Embedding Head:
  Linear(512/2048 → 1000)
  BatchNorm + ReLU
  Linear(1000 → 2048)
  Tanh
    ↓
L2-Normalized Embedding (2048-dim)
```

### Training

- **Loss Functions:**
  - Triplet Loss: Metric learning approach
  - ArcFace Loss: Classification with margin

- **Optimization:**
  - Optimizer: Adam
  - Weight Decay: 1e-5
  - Scheduler: ReduceLROnPlateau

- **Augmentation:**
  - Random horizontal flip
  - Random rotation (±10°)
  - Color jitter

### Inference

1. **Preprocessing:**
   - Resize to 224×224
   - ImageNet normalization
   - Convert to tensor

2. **Embedding:**
   - Forward pass through model
   - L2 normalization

3. **Search:**
   - FAISS cosine similarity
   - Top-k retrieval
   - Confidence scoring

## 🧪 Testing

```bash
# Test service
python test_service.py /path/to/test/image.jpg

# Health check
curl http://127.0.0.1:8005/health

# Get info
curl http://127.0.0.1:8005/info
```

## 🐛 Troubleshooting

### Model Not Found
- Run training: `python training/train.py ...`
- Check model path in environment variable
- Verify model file exists

### Index Not Found
- Build index: `python build_index.py ...`
- Check prepared dataset exists
- Verify model is loaded

### CUDA Errors
- Use CPU: `export PILL_DEVICE=cpu`
- Reduce batch size
- Check GPU availability

### Import Errors
- Install dependencies: `pip install -r requirements.txt`
- Check Python version (3.8+)
- Verify module paths

## 📝 Documentation

- **README.md** - Basic overview
- **SETUP.md** - Quick setup guide
- **COMPLETE_GUIDE.md** - Detailed step-by-step guide
- **README_FULL.md** - This comprehensive guide

## 🎓 References

- **ePillID-benchmark**: https://github.com/usuyama/ePillID-benchmark
- **Paper**: "ePillID Dataset: A Low-Shot Fine-Grained Benchmark for Pill Identification" (CVPR 2020)

## ✅ Features Checklist

- [x] Dataset preparation with auto-detection
- [x] Model training (Triplet/ArcFace)
- [x] Automatic model detection
- [x] FAISS index building
- [x] Automatic index detection
- [x] Inference pipeline
- [x] FastAPI service
- [x] Complete automation
- [x] Comprehensive documentation
- [x] Error handling
- [x] Localhost-only service
- [x] No external dependencies
- [x] No GitHub pushes
- [x] No cloud deployment

## 🚦 Next Steps

1. **Download Dataset**: Get ePillID dataset from releases
2. **Run Pipeline**: Execute `pipeline.py` with your data
3. **Test Service**: Use `test_service.py` to verify
4. **Integrate**: Call API from your application

## 📄 License

Based on ePillID-benchmark (MIT License). See original repository for details.







