# **filename: docs/ai_handbook/01_onboarding/02_data_overview.md**

# **Onboarding: Data Overview**

This document provides a high-level overview of the dataset structure, annotation format, and processing pipeline used in this project.

## **1. Dataset Structure**

The project follows the standard ICDAR competition format for data organization.

```
data/
└── datasets/
    ├── images/
    │   ├── train/      # Training images
    │   ├── val/        # Validation images
    │   └── test/       # Test images
    └── jsons/
        ├── train.json  # Training annotations
        ├── val.json    # Validation annotations
        └── test.json   # Test annotations (no ground truth)
```

## **2. Annotation Format**

Annotations are provided in a JSON file that maps image filenames to the words they contain. Each word is defined by a quadrilateral polygon.

```json
{
  "images": {
    "image1.jpg": {
      "words": {
        "word_1": {
          "points": [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
        },
        "word_2": {
          // ... more words
        }
      }
    }
    // ... more images
  }
}
```

## **3. Data Processing Pipeline**

The data pipeline transforms raw images and annotations into batches ready for model training.

1. **OCRDataset Class**: The primary torch.utils.data.Dataset implementation. It loads images, parses JSON annotations, and handles basic validation (e.g., filtering images without annotations).
2. **Transform Pipeline**: Uses the albumentations library to apply augmentations (e.g., rotation, scaling) and normalization to the images and their corresponding polygons.
3. **Collate Function**: A custom collate function (db_collate_fn) batches the individual samples into a format suitable for the DBNet model.
4. **DataLoader**: The standard PyTorch DataLoader manages batching, shuffling, and parallel data loading.

## **4. Evaluation Metrics**

The primary metric for this project is **CLEval (Character-Level Evaluation)**, which is the industry standard for OCR tasks. It calculates precision, recall, and F1-score at a character level, providing a more granular assessment than simple bounding box IoU.
