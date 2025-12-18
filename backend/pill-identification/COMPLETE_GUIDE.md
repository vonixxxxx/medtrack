# Complete Pill Identification Pipeline Guide

This guide covers the entire pipeline from dataset to running inference service.

## Overview

The pill identification system consists of:

1. **Dataset Preparation** - Organize and prepare ePillID dataset
2. **Model Training** - Train embedding model (or use existing)
3. **Index Building** - Generate FAISS index from reference images
4. **Inference Service** - FastAPI service for pill identification

## Quick Start

### Option 1: Automated Pipeline

Run the complete pipeline in one command:

```bash
cd backend/pill-identification
python pipeline.py --data_root /path/to/epillid/data --start_service
```

This will:
- Prepare the dataset
- Train a model (if none exists)
- Build the FAISS index
- Start the inference service

### Option 2: Step-by-Step

Follow the steps below for more control.

## Step 1: Dataset Preparation

### Download ePillID Dataset

Download the dataset from the [ePillID releases page](https://github.com/usuyama/ePillID-benchmark/releases).

Extract it to a directory, e.g., `/path/to/epillid_data`.

### Prepare Dataset

```bash
python -c "from dataset.prepare import prepare_dataset; prepare_dataset('/path/to/epillid_data', './output/prepared')"
```

Or use the preparer directly:

```python
from dataset.prepare import DatasetPreparer

preparer = DatasetPreparer('/path/to/epillid_data')
result = preparer.prepare_dataset(output_dir='./output/prepared')
```

The preparer will:
- Automatically detect dataset structure (CSV or folder-based)
- Organize reference and consumer images
- Create train/val/test splits
- Generate metadata files

**Output files:**
- `prepared/train.csv` - Training images
- `prepared/val.csv` - Validation images
- `prepared/test.csv` - Test images
- `prepared/metadata.json` - Dataset statistics
- `prepared/label_encoder.pkl` - Label encoder

## Step 2: Model Training

### Automatic Model Detection

The system automatically detects existing models. If a model is found, training is skipped.

### Train New Model

```bash
python training/train.py \
    --data_root /path/to/epillid_data \
    --prepared_dir ./output/prepared \
    --output_dir ./output/models \
    --network resnet18 \
    --embedding_dim 2048 \
    --num_epochs 50 \
    --batch_size 32 \
    --loss_type triplet
```

**Training options:**
- `--network`: CNN backbone (`resnet18`, `resnet34`, `resnet50`)
- `--embedding_dim`: Embedding dimension (default: 2048)
- `--loss_type`: Loss function (`triplet` or `arcface`)
- `--margin`: Margin for triplet loss (default: 1.0)
- `--num_epochs`: Number of training epochs
- `--batch_size`: Batch size
- `--learning_rate`: Initial learning rate (default: 1e-4)

**Output:**
- `models/checkpoint_best.pth` - Best model checkpoint
- `models/checkpoint_latest.pth` - Latest checkpoint
- `models/model_final.pth` - Final model for inference

### Resume Training

```bash
python training/train.py \
    --prepared_dir ./output/prepared \
    --checkpoint ./output/models/checkpoint_latest.pth \
    --num_epochs 50
```

## Step 3: Build FAISS Index

Build the search index from reference images:

```bash
python build_index.py \
    --prepared_dir ./output/prepared \
    --model_path ./output/models/model_final.pth \
    --output_dir ./output/data \
    --use_reference_only
```

**Options:**
- `--prepared_dir`: Directory with prepared dataset
- `--model_path`: Path to trained model (auto-detected if not provided)
- `--output_dir`: Output directory for index
- `--use_reference_only`: Only use reference images (recommended)
- `--batch_size`: Batch size for embedding generation
- `--metric`: Distance metric (`cosine` or `l2`)

**Output:**
- `data/pill_index.index` - FAISS index file
- `data/pill_metadata.json` - Metadata for each indexed image

## Step 4: Run Inference Service

### Start Service

```bash
# Option 1: Use startup script
./start_service.sh

# Option 2: Direct uvicorn
uvicorn api.app:app --host 127.0.0.1 --port 8005 --reload
```

### Environment Variables

Set these to customize paths (optional, auto-detection enabled by default):

```bash
export PILL_MODEL_PATH=./output/models/model_final.pth
export PILL_INDEX_PATH=./output/data/pill_index.index
export PILL_METADATA_PATH=./output/data/pill_metadata.json
export PILL_NETWORK=resnet18
export PILL_EMBEDDING_DIM=2048
export PILL_DEVICE=cuda  # or 'cpu'
export PILL_METRIC=cosine
export PILL_AUTO_DETECT=true  # Enable auto-detection
```

### Test Service

```bash
# Health check
curl http://127.0.0.1:8005/health

# Get info
curl http://127.0.0.1:8005/info

# Test with image
python test_service.py /path/to/test/image.jpg
```

## API Usage

### Identify Pill

```python
import requests
import base64

# Read image
with open('pill.jpg', 'rb') as f:
    image_data = base64.b64encode(f.read()).decode('utf-8')

# Identify
response = requests.post(
    'http://127.0.0.1:8005/identify',
    json={
        'image': f'data:image/jpeg;base64,{image_data}',
        'k': 5,
        'min_confidence': 0.5
    }
)

result = response.json()
print(f"Top match: {result['top_match']['metadata']['label']}")
print(f"Confidence: {result['top_match']['confidence']:.2f}")
```

### Generate Embedding

```python
response = requests.post(
    'http://127.0.0.1:8005/embed',
    json={'image': base64_image}
)

embedding = response.json()['embedding']
print(f"Embedding dimension: {len(embedding)}")
```

## Architecture

### Model Architecture

```
Input Image (224x224x3)
    ↓
ResNet18/50 Backbone
    ↓
Global Average Pooling
    ↓
Embedding Head (MLP)
    ↓
L2-Normalized Embedding (2048-dim)
```

### Training

- **Loss**: Triplet loss or ArcFace loss
- **Optimizer**: Adam with weight decay
- **Scheduler**: ReduceLROnPlateau
- **Augmentation**: Random flip, rotation, color jitter

### Inference

1. Preprocess image (resize, normalize)
2. Generate embedding via model
3. Search FAISS index for similar pills
4. Return top-k matches with confidence scores

## File Structure

```
pill-identification/
├── dataset/
│   ├── __init__.py
│   └── prepare.py          # Dataset preparation
├── training/
│   ├── __init__.py
│   ├── trainer.py          # Training logic
│   └── train.py            # Training script
├── models/
│   ├── embedding_model.py  # Model architecture
│   └── margin_linear.py    # ArcFace layer
├── utils/
│   ├── preprocessing.py    # Image preprocessing
│   ├── embedding.py        # Embedding generation
│   ├── vector_search.py    # FAISS search
│   └── model_detector.py   # Auto-detect models
├── api/
│   └── app.py              # FastAPI service
├── pipeline.py              # Complete pipeline
├── build_index.py           # Index building
├── inference.py             # Inference class
└── ...
```

## Troubleshooting

### Model Not Found

- Check that model file exists
- Verify path in environment variable
- Run training if no model exists

### Index Not Found

- Run `build_index.py` to create index
- Check that prepared dataset exists
- Verify model is loaded correctly

### CUDA Out of Memory

- Reduce batch size
- Use CPU: `export PILL_DEVICE=cpu`
- Use smaller network (resnet18 instead of resnet50)

### Import Errors

- Install dependencies: `pip install -r requirements.txt`
- Check Python version (3.8+)
- Verify all modules are in path

## Performance Tips

1. **Use GPU**: Set `PILL_DEVICE=cuda` for faster inference
2. **Batch Processing**: Process multiple images together
3. **Index Size**: Use reference-only images for smaller index
4. **Model Size**: ResNet18 is faster, ResNet50 is more accurate

## Limitations

- Dataset: Requires ePillID dataset (not included)
- Training: GPU recommended for training
- Index: Large datasets may require significant memory
- Accuracy: Depends on training data quality

## Next Steps

1. **Fine-tune**: Adjust hyperparameters for your dataset
2. **Evaluate**: Test on held-out test set
3. **Optimize**: Quantize model for faster inference
4. **Scale**: Use GPU cluster for large-scale training

## References

- ePillID-benchmark: https://github.com/usuyama/ePillID-benchmark
- Paper: "ePillID Dataset: A Low-Shot Fine-Grained Benchmark for Pill Identification" (CVPR 2020)







