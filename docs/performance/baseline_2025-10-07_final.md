# Performance Baseline Report

**Generated:** 2025-10-07 17:00:39
**WandB Run:** [baseline_profiling_2025_10_07_v4](https://wandb.ai/runs/zr90z4cu)
**Run ID:** `zr90z4cu`
**Status:** finished

---

## Validation Performance

| Metric | Value |
|--------|-------|
| **Total Validation Time** | 16.29s |
| **Number of Batches** | 34 |
| **Mean Batch Time** | 436.2ms |
| **Median Batch Time** | 422.9ms |
| **P95 Batch Time** | 617.1ms |
| **P99 Batch Time** | 669.4ms |
| **Batch Time Std Dev** | 81.6ms |

## Memory Usage

| Resource | Usage |
|----------|-------|
| **GPU Memory** | 0.06 GB |
| **GPU Memory Reserved** | 5.62 GB |
| **CPU Memory** | 7.8% |

## Training vs Validation Comparison

- **Note:** Training batch time not available in this run for comparison. Based on the performance plan, validation is typically ~10x slower than training due to PyClipper bottleneck.

## Identified Bottlenecks

No significant bottlenecks detected.

## Additional Analysis

Based on the performance optimization plan documented in the project handbook, the following issues are likely present:

- **PyClipper Polygon Processing**: Known bottleneck causing ~10x validation slowdown
- **Memory Usage**: Check for potential memory leaks during validation
- **Batch Variance**: High variance in processing times indicating inconsistent performance

## Recommendations

Based on the baseline analysis and the performance optimization plan:

1. **PyClipper Caching** (Phase 1.1) - Implement caching for polygon processing operations
2. **Parallel Processing** (Phase 1.2) - Use multiprocessing for preprocessing steps
3. **Memory-Mapped Caching** (Phase 1.3) - Optimize memory usage with memory-mapped files
4. **Memory Optimization** (Phase 2) - Reduce memory footprint for larger batches
5. **Automated Profiling** (Phase 3) - Establish continuous performance monitoring

For more details on the optimization roadmap, see the performance optimization plan in the documentation.

## Raw Metrics Summary

### Configuration
```json
{
  "data": {
    "batch_size": 12
  },
  "seed": 42,
  "debug": {
    "verbose": false,
    "profiling": false
  },
  "model": {
    "head": {
      "k": 50,
      "bias": false,
      "smooth": false,
      "upscale": 4,
      "_target_": "ocr.models.head.DBHead",
      "in_channels": 256,
      "postprocess": {
        "thresh": 0.3,
        "box_thresh": 0.4,
        "use_polygon": false,
        "max_candidates": 300
      }
    },
    "loss": {
      "eps": 1e-06,
      "_target_": "ocr.models.loss.DBLoss",
      "negative_ratio": 3,
      "prob_map_loss_weight": 5,
      "binary_map_loss_weight": 1,
      "thresh_map_loss_weight": 10
    },
    "decoder": {
      "bias": false,
      "strides": [
        4,
        8,
        16,
        32
      ],
      "_target_": "ocr.models.decoder.UNet",
      "in_channels": [
        64,
        128,
        256,
        512
      ],
      "inner_channels": 256,
      "output_channels": 256
    },
    "encoder": {
      "_target_": "ocr.models.encoder.TimmBackbone",
      "model_name": "resnet18",
      "pretrained": true,
      "select_features": [
        1,
        2,
        3,
        4
      ]
    },
    "_target_": "ocr.models.architecture.OCRModel",
    "optimizer": {
      "lr": 0.001,
      "eps": 1e-08,
      "betas": [
        0.9,
        0.999
      ],
      "_target_": "torch.optim.AdamW",
      "weight_decay": 0.0001
    },
    "scheduler": {
      "gamma": 0.1,
      "_target_": "torch.optim.lr_scheduler.StepLR",
      "step_size": 100
    },
    "architecture_name": "dbnet",
    "component_overrides": {
      "head": {
        "name": "db_head",
        "params": {
          "k": 50,
          "postprocess": {
            "thresh": 0.2,
            "box_thresh": 0.3,
            "use_polygon": true,
            "max_candidates": 300
          }
        }
      },
      "loss": {
        "name": "db_loss",
        "params": {
          "negative_ratio": 3,
          "prob_map_loss_weight": 5,
          "binary_map_loss_weight": 1,
          "thresh_map_loss_weight": 10
        }
      },
      "decoder": {
        "name": "fpn_decoder",
        "params": {
          "out_channels": 256,
          "inner_channels": 256,
          "output_channels": 256
        }
      },
      "encoder": {
        "model_name": "resnet18",
        "pretrained": true,
        "select_features": [
          1,
          2,
          3,
          4
        ]
      }
    }
  },
  "paths": {
    "log_dir": "outputs/baseline_profiling_2025_10_07_v4/logs",
    "checkpoint_dir": "outputs/baseline_profiling_2025_10_07_v4/checkpoints",
    "submission_dir": "outputs/baseline_profiling_2025_10_07_v4/submissions"
  },
  "wandb": true,
  "logger": {
    "wandb": {
      "enabled": true
    },
    "exp_version": "v1.0",
    "project_name": "receipt-text-recognition-ocr-project",
    "per_batch_image_logging": {
      "enabled": false,
      "image_format": "jpeg",
      "max_image_side": 640,
      "recall_threshold": 0.4,
      "max_images_per_batch": 4,
      "max_batches_per_epoch": 2,
      "use_transformed_batch": true
    }
  },
  "metrics": {
    "eval": {
      "_target_": "ocr.metrics.cleval_metric.CLEvalMetric",
      "scale_bins": [
        0,
        0.005,
        0.01,
        0.015,
        0.02,
        0.025,
        0.1,
        0.5,
        1
      ],
      "scale_wise": false,
      "scale_range": [
        0,
        1
      ],
      "max_polygons": 500,
      "ap_constraint": 0.3,
      "case_sensitive": true,
      "dist_sync_on_step": false,
      "recall_gran_penalty": 1,
      "precision_gran_penalty": 1,
      "vertical_aspect_ratio_thresh": 0.5
    }
  },
  "modules": {
    "lightning_module": {
      "_target_": "ocr.lightning_modules.OCRPLModule"
    },
    "lightning_data_module": {
      "_target_": "ocr.lightning_modules.OCRDataPLModule"
    }
  },
  "trainer": {
    "devices": 1,
    "strategy": "auto",
    "benchmark": true,
    "max_steps": -1,
    "precision": 32,
    "max_epochs": 200,
    "accelerator": "gpu",
    "deterministic": true,
    "gradient_clip_val": 5,
    "log_every_n_steps": 10,
    "val_check_interval": null,
    "num_sanity_val_steps": 1,
    "accumulate_grad_batches": 2,
    "check_val_every_n_epoch": 1
  },
  "datasets": {
    "val_dataset": {
      "_target_": "ocr.datasets.OCRDataset",
      "transform": {
        "_target_": "ocr.datasets.DBTransforms",
        "transforms": [
          {
            "p": 1,
            "_target_": "albumentations.LongestMaxSize",
            "max_size": 640
          },
          {
            "p": 1,
            "_target_": "albumentations.PadIfNeeded",
            "min_width": 640,
            "min_height": 640,
            "border_mode": 0
          },
          {
            "std": [
              0.229,
              0.224,
              0.225
            ],
            "mean": [
              0.485,
              0.456,
              0.406
            ],
            "_target_": "albumentations.Normalize"
          }
        ],
        "keypoint_params": {
          "format": "xy",
          "_target_": "albumentations.KeypointParams",
          "remove_invisible": true
        }
      },
      "image_path": "/home/vscode/workspace/upstageailab-ocr-recsys-competition-ocr-2/data/datasets/images_val_canonical",
      "annotation_path": "/home/vscode/workspace/upstageailab-ocr-recsys-competition-ocr-2/data/datasets/jsons/val.json"
    },
    "test_dataset": {
      "_target_": "ocr.datasets.OCRDataset",
      "transform": {
        "_target_": "ocr.datasets.DBTransforms",
        "transforms": [
          {
            "p": 1,
            "_target_": "albumentations.LongestMaxSize",
            "max_size": 640
          },
          {
            "p": 1,
            "_target_": "albumentations.PadIfNeeded",
            "min_width": 640,
            "min_height": 640,
            "border_mode": 0
          },
          {
            "std": [
              0.229,
              0.224,
              0.225
            ],
            "mean": [
              0.485,
              0.456,
              0.406
            ],
            "_target_": "albumentations.Normalize"
          }
        ],
        "keypoint_params": {
          "format": "xy",
          "_target_": "albumentations.KeypointParams",
          "remove_invisible": true
        }
      },
      "image_path": "/home/vscode/workspace/upstageailab-ocr-recsys-competition-ocr-2/data/datasets/images_val_canonical",
      "annotation_path": "/home/vscode/workspace/upstageailab-ocr-recsys-competition-ocr-2/data/datasets/jsons/val.json"
    },
    "train_dataset": {
      "_target_": "ocr.datasets.OCRDataset",
      "transform": {
        "_target_": "ocr.datasets.DBTransforms",
        "transforms": [
          {
            "p": 1,
            "_target_": "albumentations.LongestMaxSize",
            "max_size": 640
          },
          {
            "p": 1,
            "_target_": "albumentations.PadIfNeeded",
            "min_width": 640,
            "min_height": 640,
            "border_mode": 0
          },
          {
            "p": 0.5,
            "_target_": "albumentations.HorizontalFlip"
          },
          {
            "std": [
              0.229,
              0.224,
              0.225
            ],
            "mean": [
              0.485,
              0.456,
              0.406
            ],
            "_target_": "albumentations.Normalize"
          }
        ],
        "keypoint_params": {
          "format": "xy",
          "_target_": "albumentations.KeypointParams",
          "remove_invisible": true
        }
      },
      "image_path": "/home/vscode/workspace/upstageailab-ocr-recsys-competition-ocr-2/data/datasets/images/train",
      "annotation_path": "/home/vscode/workspace/upstageailab-ocr-recsys-competition-ocr-2/data/datasets/jsons/train.json"
    },
    "predict_dataset": {
      "_target_": "ocr.datasets.OCRDataset",
      "transform": {
        "_target_": "ocr.datasets.DBTransforms",
        "transforms": [
          {
            "p": 1,
            "_target_": "albumentations.LongestMaxSize",
            "max_size": 640
          },
          {
            "p": 1,
            "_target_": "albumentations.PadIfNeeded",
            "min_width": 640,
            "min_height": 640,
            "border_mode": 0
          },
          {
            "std": [
              0.229,
              0.224,
              0.225
            ],
            "mean": [
              0.485,
              0.456,
              0.406
            ],
            "_target_": "albumentations.Normalize"
          }
        ],
        "keypoint_params": {
          "format": "xy",
          "_target_": "albumentations.KeypointParams",
          "remove_invisible": true
        }
      },
      "image_path": "/home/vscode/workspace/upstageailab-ocr-recsys-competition-ocr-2/data/datasets/images/test",
      "annotation_path": null
    }
  },
  "exp_name": "baseline_profiling_2025_10_07_v4",
  "callbacks": {
    "performance_profiler": {
      "enabled": true,
      "verbose": true,
      "_target_": "ocr.lightning_modules.callbacks.PerformanceProfilerCallback",
      "log_interval": 10,
      "profile_memory": true
    }
  },
  "head_path": "ocr.models.head",
  "loss_path": "ocr.models.loss",
  "collate_fn": {
    "_target_": "ocr.datasets.DBCollateFN",
    "thresh_max": 0.7,
    "thresh_min": 0.3,
    "shrink_ratio": 0.4
  },
  "model_path": "ocr.models",
  "transforms": {
    "val_transform": {
      "_target_": "ocr.datasets.DBTransforms",
      "transforms": [
        {
          "p": 1,
          "_target_": "albumentations.LongestMaxSize",
          "max_size": 640
        },
        {
          "p": 1,
          "_target_": "albumentations.PadIfNeeded",
          "min_width": 640,
          "min_height": 640,
          "border_mode": 0
        },
        {
          "std": [
            0.229,
            0.224,
            0.225
          ],
          "mean": [
            0.485,
            0.456,
            0.406
          ],
          "_target_": "albumentations.Normalize"
        }
      ],
      "keypoint_params": {
        "format": "xy",
        "_target_": "albumentations.KeypointParams",
        "remove_invisible": true
      }
    },
    "test_transform": {
      "_target_": "ocr.datasets.DBTransforms",
      "transforms": [
        {
          "p": 1,
          "_target_": "albumentations.LongestMaxSize",
          "max_size": 640
        },
        {
          "p": 1,
          "_target_": "albumentations.PadIfNeeded",
          "min_width": 640,
          "min_height": 640,
          "border_mode": 0
        },
        {
          "std": [
            0.229,
            0.224,
            0.225
          ],
          "mean": [
            0.485,
            0.456,
            0.406
          ],
          "_target_": "albumentations.Normalize"
        }
      ],
      "keypoint_params": {
        "format": "xy",
        "_target_": "albumentations.KeypointParams",
        "remove_invisible": true
      }
    },
    "train_transform": {
      "_target_": "ocr.datasets.DBTransforms",
      "transforms": [
        {
          "p": 1,
          "_target_": "albumentations.LongestMaxSize",
          "max_size": 640
        },
        {
          "p": 1,
          "_target_": "albumentations.PadIfNeeded",
          "min_width": 640,
          "min_height": 640,
          "border_mode": 0
        },
        {
          "p": 0.5,
          "_target_": "albumentations.HorizontalFlip"
        },
        {
          "std": [
            0.229,
            0.224,
            0.225
          ],
          "mean": [
            0.485,
            0.456,
            0.406
          ],
          "_target_": "albumentations.Normalize"
        }
      ],
      "keypoint_params": {
        "format": "xy",
        "_target_": "albumentations.KeypointParams",
        "remove_invisible": true
      }
    },
    "predict_transform": {
      "_target_": "ocr.datasets.DBTransforms",
      "transforms": [
        {
          "p": 1,
          "_target_": "albumentations.LongestMaxSize",
          "max_size": 640
        },
        {
          "p": 1,
          "_target_": "albumentations.PadIfNeeded",
          "min_width": 640,
          "min_height": 640,
          "border_mode": 0
        },
        {
          "std": [
            0.229,
            0.224,
            0.225
          ],
          "mean": [
            0.485,
            0.456,
            0.406
          ],
          "_target_": "albumentations.Normalize"
        }
      ],
      "keypoint_params": null
    }
  },
  "dataloaders": {
    "val_dataloader": {
      "shuffle": false,
      "batch_size": 12,
      "pin_memory": true,
      "num_workers": 4,
      "prefetch_factor": 2,
      "persistent_workers": true
    },
    "test_dataloader": {
      "shuffle": false,
      "batch_size": 12,
      "pin_memory": true,
      "num_workers": 4,
      "prefetch_factor": 2,
      "persistent_workers": true
    },
    "train_dataloader": {
      "shuffle": true,
      "batch_size": 12,
      "pin_memory": true,
      "num_workers": 4,
      "prefetch_factor": 2,
      "persistent_workers": true
    },
    "predict_dataloader": {
      "shuffle": false,
      "batch_size": 12,
      "pin_memory": true,
      "num_workers": 2,
      "prefetch_factor": 2,
      "persistent_workers": true
    }
  },
  "dataset_path": "ocr.datasets",
  "decoder_path": "ocr.models.decoder",
  "encoder_path": "ocr.models.encoder",
  "project_name": "OCR_Performance_Baseline",
  "lightning_path": "ocr.lightning_modules",
  "checkpoint_path": "/home/vscode/workspace/upstageailab-ocr-recsys-competition-ocr-2/outputs/canonical-fix2-dbnet-fpn_decoder-mobilenetv3_small_050/checkpoints/last.ckpt",
  "dataset_base_path": "/home/vscode/workspace/upstageailab-ocr-recsys-competition-ocr-2/data/datasets/"
}
```

### Summary Values
```json
{
  "_runtime": 148,
  "_step": 0,
  "_timestamp": 1759824027.1123283,
  "_wandb": {
    "runtime": 148
  },
  "epoch": 0,
  "performance/cpu_memory_percent": 7.800000190734863,
  "performance/gpu_memory_gb": 0.0623164176940918,
  "performance/gpu_memory_reserved_gb": 5.6171875,
  "performance/val_batch_idx": 30,
  "performance/val_batch_mean": 0.4362019896507263,
  "performance/val_batch_median": 0.422916054725647,
  "performance/val_batch_p95": 0.6171090602874756,
  "performance/val_batch_p99": 0.669394314289093,
  "performance/val_batch_std": 0.0816410705447197,
  "performance/val_batch_time": 0.4456822872161865,
  "performance/val_epoch_time": 16.291444778442383,
  "performance/val_num_batches": 34,
  "test/hmean": 0.9561273455619812,
  "test/precision": 0.9537277817726136,
  "test/recall": 0.9594830870628356,
  "trainer/global_step": 0
}
```
