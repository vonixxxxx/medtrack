# Starting Training and Testing

## Current Status

The system is ready for training, but we need the ePillID dataset. Here's how to proceed:

## Option 1: Download Dataset Automatically

The system can automatically download the ePillID dataset:

```bash
cd backend/pill-identification
python3 -c "
from dataset.download import download_dataset
dataset_path = download_dataset('./data/epillid')
print(f'Dataset ready at: {dataset_path}')
"
```

## Option 2: Use Existing Dataset

If you have the ePillID dataset already:

```bash
cd backend/pill-identification

# Prepare dataset
python3 -c "
from dataset.prepare import prepare_dataset
result = prepare_dataset('/path/to/epillid/data', './output/prepared')
print(f'Prepared: {len(result[\"train\"])} train, {len(result[\"val\"])} val, {len(result[\"test\"])} test')
"

# Train model
python3 training/train.py \
    --prepared_dir ./output/prepared \
    --output_dir ./output/models \
    --network resnet18 \
    --embedding_dim 2048 \
    --num_epochs 10 \
    --batch_size 16 \
    --loss_type triplet

# Build index
python3 build_index.py \
    --prepared_dir ./output/prepared \
    --model_path ./output/models/model_final.pth \
    --output_dir ./output/data \
    --use_reference_only

# Test model
python3 evaluate_test_set.py \
    --prepared_dir ./output/prepared \
    --model_path ./output/models/model_final.pth \
    --index_path ./output/data/pill_index.index \
    --metadata_path ./output/data/pill_metadata.json
```

## Option 3: Quick Script

Use the quick training script:

```bash
python3 quick_train_test.py
```

This will:
1. Check for prepared dataset
2. Train model (or use existing)
3. Build index
4. Test on reserved test set

## Notes

- The dataset needs to have reference and consumer images
- Training requires at least 10-20 images per class for good results
- The test set is automatically reserved (20% of classes)
- Training uses only train/val sets, never the test set

## Troubleshooting

If you see "No training images found":
- The dataset preparation reserved all images for testing
- You need a larger dataset with more images
- Download the full ePillID dataset from GitHub releases





