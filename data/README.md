# Data Structure - Vi-VQA Dataset

## Dataset Source

The Vi-VQA dataset is stored on HuggingFace Hub:
- **URL**: `Ouyangxuelili/vlsp2023-vqa-dataset`
- **Type**: Vision Language datasets (Images + Questions + Answers)

## Folder Structure

```
data/
├── raw/                    # Raw data (downloaded from HuggingFace)
│   ├── train/              # Training images
│   └── dev/                # Validation images (supplied separately)
│
├── processed/              # Processed/cleaned data
│   ├── train_processed.json
│   └── dev_processed.json
│
└── cache/                  # Cache for Hugging Face datasets
    └── (auto-generated)
```

## Data Loading

### Option 1: Load from HuggingFace Hub (RECOMMENDED)
```python
from datasets import load_dataset

# Load directly from Hub
dataset = load_dataset("Ouyangxuelili/vlsp2023-vqa-dataset")

# Access splits
train_dataset = dataset["train"]
dev_dataset = dataset["dev"]
```

### Option 2: Load Local JSON Files
```python
import json

with open("raw/train/vlsp2023_train_data.json", "r", encoding="utf-8") as f:
    train_data = json.load(f)
```

## Dataset Format

Expected format from HuggingFace:
```json
{
  "image_id": 12345,
  "question": "Hãy mô tả bức ảnh này",
  "answer": "Một người đi xe đạp trên đường",
  "image_path": "training-images/000000012345.jpg"
}
```

## Data Statistics

| Split | Samples | Images |
|-------|---------|--------|
| Train | 30,833 | 9,129 |
| Dev   | 3,545  | 1,070 referenced images |

## Important Notes

- Dataset is hosted on HuggingFace - **no need to download manually**
- Images are typically 256×256 or 512×512
- Questions are in Vietnamese
- Answers are short Vietnamese phrases (1-20 words)
- The public repository has train and dev annotations, but no test split.
- The 9,129 image files on the Hub cover the train split. The 1,070 images
  referenced by dev annotations must be supplied separately for evaluation.
- The repository is public, so a Hugging Face token is not required.
