# Performance Baseline Report

**Generated:** 2025-10-13 23:58:03
**WandB Run:** [wchoi189_dbnet-resnet18-unet-dbhead-dbloss-bs16-lr1e-3_hmean0.596](https://wandb.ai/runs/9jpspsrq)
**Run ID:** `9jpspsrq`
**Status:** finished

---

## Training Metrics Summary

**Note:** This run did not have performance profiling enabled. Showing available training metrics instead.

| Metric | Validation | Test |
|--------|------------|------|
| **H-Mean** | 0.7429 | 0.5958 |
| **Precision** | 0.8624 | 0.6896 |
| **Recall** | 0.7239 | 0.5785 |

| Training Details | Value |
|------------------|-------|
| **Training Loss** | 2.2910 |
| **Validation Loss** | 1.9898 |
| **Epoch** | 3 |
| **Global Step** | 309 |

## Training vs Validation Comparison

- **Note:** Performance profiling not enabled - cannot compare training vs validation timing.

## Identified Issues

### 1. Low validation performance (HIGH)

Validation H-mean (0.743) is below 0.8, indicating potential model issues

## Recommendations

To get detailed performance analysis, enable performance profiling in future runs:

1. **Add Performance Profiler Callback**: Include `performance_profiler` in your training configuration
2. **Re-run Training**: Execute training with performance monitoring enabled
3. **Generate Full Report**: Use this script again on the profiled run

Example config addition:
```yaml
callbacks:
  performance_profiler:
    _target_: ocr.lightning_modules.callbacks.performance_profiler.PerformanceProfilerCallback
    enabled: true
    log_interval: 10
```

## Raw Metrics Summary

### Configuration
```json
{
  "data": {
    "val_num_samples": null,
    "test_num_samples": null,
    "train_num_samples": null
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
        "use_polygon": true,
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
      "T_max": 1000,
      "eta_min": 0,
      "_target_": "torch.optim.lr_scheduler.CosineAnnealingLR"
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
        "pretrained": false,
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
    "log_dir": "outputs/benchmark_optimized_32bit_full_cache/logs",
    "output_dir": "outputs/benchmark_optimized_32bit_full_cache",
    "checkpoint_dir": "outputs/benchmark_optimized_32bit_full_cache/checkpoints",
    "submission_dir": "outputs/benchmark_optimized_32bit_full_cache/submissions"
  },
  "wandb": true,
  "extras": {
    "enforce_tags": true,
    "print_config": true,
    "ignore_warnings": false
  },
  "logger": {
    "csv": {
      "name": "csv/",
      "prefix": "",
      "_target_": "lightning.pytorch.loggers.csv_logs.CSVLogger",
      "save_dir": "outputs/benchmark_optimized_32bit_full_cache"
    },
    "wandb": {
      "enabled": true
    },
    "settings": {
      "offline": false,
      "sync_dir": "outputs/wandb_sync",
      "save_code": false
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
  "resume": null,
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
  "runtime": {
    "ddp_strategy": "ddp_find_unused_parameters_false",
    "auto_gpu_devices": true,
    "min_auto_devices": 2
  },
  "trainer": {
    "devices": 1,
    "strategy": "auto",
    "benchmark": true,
    "max_steps": -1,
    "precision": "32-true",
    "max_epochs": 3,
    "accelerator": "gpu",
    "deterministic": true,
    "gradient_clip_val": 5,
    "limit_val_batches": 50,
    "log_every_n_steps": 20,
    "limit_test_batches": null,
    "val_check_interval": null,
    "limit_train_batches": null,
    "num_sanity_val_steps": 1,
    "accumulate_grad_batches": 2,
    "check_val_every_n_epoch": 1
  },
  "datasets": {
    "val_dataset": {
      "config": {
        "_target_": "ocr.datasets.schemas.DatasetConfig",
        "load_maps": true,
        "image_path": "/home/vscode/workspace/upstageailab-ocr-recsys-competition-ocr-2/data/datasets/images_val_canonical",
        "cache_config": {
          "_target_": "ocr.datasets.schemas.CacheConfig",
          "cache_maps": true,
          "cache_images": true,
          "log_statistics_every_n": 100,
          "cache_transformed_tensors": true
        },
        "preload_images": true,
        "annotation_path": "/home/vscode/workspace/upstageailab-ocr-recsys-competition-ocr-2/data/datasets/jsons/val.json"
      },
      "_target_": "ocr.datasets.ValidatedOCRDataset",
      "transform": {
        "_target_": "ocr.datasets.DBTransforms",
        "transforms": [
          {
            "p": 1,
            "_target_": "albumentations.LongestMaxSize",
            "max_size": 640,
            "interpolation": 1
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
      }
    },
    "test_dataset": {
      "config": {
        "_target_": "ocr.datasets.schemas.DatasetConfig",
        "image_path": "/home/vscode/workspace/upstageailab-ocr-recsys-competition-ocr-2/data/datasets/images/val",
        "annotation_path": "/home/vscode/workspace/upstageailab-ocr-recsys-competition-ocr-2/data/datasets/jsons/val.json"
      },
      "_target_": "ocr.datasets.ValidatedOCRDataset",
      "transform": {
        "_target_": "ocr.datasets.DBTransforms",
        "transforms": [
          {
            "p": 1,
            "_target_": "albumentations.LongestMaxSize",
            "max_size": 640,
            "interpolation": 1
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
      }
    },
    "train_dataset": {
      "config": {
        "_target_": "ocr.datasets.schemas.DatasetConfig",
        "image_path": "/home/vscode/workspace/upstageailab-ocr-recsys-competition-ocr-2/data/datasets/images/train",
        "annotation_path": "/home/vscode/workspace/upstageailab-ocr-recsys-competition-ocr-2/data/datasets/jsons/train.json"
      },
      "_target_": "ocr.datasets.ValidatedOCRDataset",
      "transform": {
        "_target_": "ocr.datasets.DBTransforms",
        "transforms": [
          {
            "p": 1,
            "_target_": "albumentations.LongestMaxSize",
            "max_size": 640,
            "interpolation": 1
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
      }
    },
    "predict_dataset": {
      "config": {
        "_target_": "ocr.datasets.schemas.DatasetConfig",
        "image_path": "/home/vscode/workspace/upstageailab-ocr-recsys-competition-ocr-2/data/datasets/images/test",
        "annotation_path": null
      },
      "_target_": "ocr.datasets.ValidatedOCRDataset",
      "transform": {
        "_target_": "ocr.datasets.DBTransforms",
        "transforms": [
          {
            "p": 1,
            "_target_": "albumentations.LongestMaxSize",
            "max_size": 640,
            "interpolation": 1
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
      }
    }
  },
  "exp_name": "benchmark_optimized_32bit_full_cache",
  "callbacks": {
    "early_stopping": {
      "mode": "max",
      "strict": true,
      "monitor": "val/hmean",
      "verbose": false,
      "_target_": "lightning.pytorch.callbacks.EarlyStopping",
      "patience": 5,
      "min_delta": 0,
      "check_finite": true,
      "stopping_threshold": null,
      "divergence_threshold": null,
      "check_on_train_epoch_end": false
    },
    "model_checkpoint": {
      "mode": "max",
      "dirpath": "outputs/benchmark_optimized_32bit_full_cache/checkpoints",
      "monitor": "val/hmean",
      "verbose": true,
      "_target_": "ocr.lightning_modules.callbacks.unique_checkpoint.UniqueModelCheckpoint",
      "filename": "{val/hmean:.4f}-best",
      "save_last": true,
      "save_top_k": 3,
      "add_timestamp": true,
      "every_n_epochs": 1,
      "save_weights_only": false,
      "every_n_train_steps": null,
      "train_time_interval": null,
      "auto_insert_metric_name": true,
      "save_on_train_epoch_end": false
    },
    "wandb_completion": {
      "_target_": "ocr.lightning_modules.callbacks.wandb_completion.WandbCompletionCallback"
    },
    "rich_progress_bar": {
      "theme": {
        "time": "cyan",
        "metrics": "white",
        "_target_": "lightning.pytorch.callbacks.progress.rich_progress.RichProgressBarTheme",
        "description": "bold white",
        "progress_bar": "red",
        "processing_speed": "cyan",
        "progress_bar_pulse": "yellow",
        "progress_bar_finished": "green"
      },
      "_target_": "lightning.pytorch.callbacks.RichProgressBar",
      "refresh_rate": 2
    },
    "wandb_image_logging": {
      "_target_": "ocr.lightning_modules.callbacks.wandb_image_logging.WandbImageLoggingCallback",
      "log_every_n_epochs": 1
    }
  },
  "head_path": "ocr.models.head",
  "loss_path": "ocr.models.loss",
  "task_name": "train",
  "batch_size": 16,
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
          "max_size": 640,
          "interpolation": 1
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
          "max_size": 640,
          "interpolation": 1
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
          "max_size": 640,
          "interpolation": 1
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
          "max_size": 640,
          "interpolation": 1
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
      "batch_size": 16,
      "pin_memory": true,
      "num_workers": 8,
      "prefetch_factor": 2,
      "persistent_workers": true
    },
    "test_dataloader": {
      "shuffle": false,
      "batch_size": 16,
      "pin_memory": true,
      "num_workers": 4,
      "prefetch_factor": 2,
      "persistent_workers": true
    },
    "train_dataloader": {
      "shuffle": true,
      "batch_size": 16,
      "pin_memory": true,
      "num_workers": 8,
      "prefetch_factor": 2,
      "persistent_workers": true
    },
    "predict_dataloader": {
      "shuffle": false,
      "batch_size": 16,
      "pin_memory": true,
      "num_workers": 4,
      "prefetch_factor": 2,
      "persistent_workers": true
    }
  },
  "dataset_path": "ocr.datasets",
  "decoder_path": "ocr.models.decoder",
  "encoder_path": "ocr.models.encoder",
  "project_name": null,
  "compile_model": true,
  "dataset_module": "ocr.datasets",
  "experiment_tag": null,
  "lightning_path": "ocr.lightning_modules",
  "dataset_base_path": "/home/vscode/workspace/upstageailab-ocr-recsys-competition-ocr-2/data/datasets/",
  "dataset_config_path": "ocr.datasets.schemas",
  "dataset_config_module": "ocr.datasets.schemas",
  "default_interpolation": 1
}
```

### Summary Values
```json
{
  "_runtime": 1185,
  "_step": 58,
  "_timestamp": 1760365410.4868536,
  "_wandb": {
    "runtime": 1185
  },
  "batch_0/hmean": 0.7813015580177307,
  "batch_0/precision": 0.923706591129303,
  "batch_0/recall": 0.74324631690979,
  "batch_1/hmean": 0.6520382761955261,
  "batch_1/precision": 0.7553050518035889,
  "batch_1/recall": 0.6469141244888306,
  "batch_10/hmean": 0.44550666213035583,
  "batch_10/precision": 0.715289831161499,
  "batch_10/recall": 0.417751282453537,
  "batch_11/hmean": 0.6651273965835571,
  "batch_11/precision": 0.8436368107795715,
  "batch_11/recall": 0.6577637195587158,
  "batch_12/hmean": 0.6875515580177307,
  "batch_12/precision": 0.8238400816917419,
  "batch_12/recall": 0.6784278750419617,
  "batch_13/hmean": 0.7471318244934082,
  "batch_13/precision": 0.8951807618141174,
  "batch_13/recall": 0.7239457964897156,
  "batch_14/hmean": 0.7852034568786621,
  "batch_14/precision": 0.8953279852867126,
  "batch_14/recall": 0.7710926532745361,
  "batch_15/hmean": 0.7226901054382324,
  "batch_15/precision": 0.8691636323928833,
  "batch_15/recall": 0.7041911482810974,
  "batch_16/hmean": 0.8529596924781799,
  "batch_16/precision": 0.9124583005905152,
  "batch_16/recall": 0.8376986384391785,
  "batch_17/hmean": 0.8029521107673645,
  "batch_17/precision": 0.8927215933799744,
  "batch_17/recall": 0.7906105518341064,
  "batch_18/hmean": 0.7221348881721497,
  "batch_18/precision": 0.8714715242385864,
  "batch_18/recall": 0.6992204189300537,
  "batch_19/hmean": 0.898589551448822,
  "batch_19/precision": 0.9524576663970948,
  "batch_19/recall": 0.8949865698814392,
  "batch_2/hmean": 0.8245001435279846,
  "batch_2/precision": 0.9333723187446594,
  "batch_2/recall": 0.810445249080658,
  "batch_20/hmean": 0.8976646065711975,
  "batch_20/precision": 0.9143017530441284,
  "batch_20/recall": 0.8846062421798706,
  "batch_21/hmean": 0.8521004915237427,
  "batch_21/precision": 0.8863224387168884,
  "batch_21/recall": 0.8284599184989929,
  "batch_22/hmean": 0.8760983943939209,
  "batch_22/precision": 0.9401735067367554,
  "batch_22/recall": 0.8570154905319214,
  "batch_23/hmean": 0.8149441480636597,
  "batch_23/precision": 0.9423416256904602,
  "batch_23/recall": 0.7990562319755554,
  "batch_24/hmean": 0.7995675802230835,
  "batch_24/precision": 0.8503684997558594,
  "batch_24/recall": 0.7650166749954224,
  "batch_25/hmean": 0.7289026975631714,
  "batch_25/precision": 0.721946120262146,
  "batch_25/recall": 0.7368543148040771,
  "batch_3/hmean": 0.6859691739082336,
  "batch_3/precision": 0.7388917803764343,
  "batch_3/recall": 0.6699752807617188,
  "batch_4/hmean": 0.9281824827194214,
  "batch_4/precision": 0.9546517729759216,
  "batch_4/recall": 0.916383683681488,
  "batch_5/hmean": 0.7877220511436462,
  "batch_5/precision": 0.9041764140129088,
  "batch_5/recall": 0.7637583017349243,
  "batch_6/hmean": 0.48811694979667664,
  "batch_6/precision": 0.7505792379379272,
  "batch_6/recall": 0.4600569605827331,
  "batch_7/hmean": 0.6814373731613159,
  "batch_7/precision": 0.8114643096923828,
  "batch_7/recall": 0.661275327205658,
  "batch_8/hmean": 0.675280749797821,
  "batch_8/precision": 0.8773285150527954,
  "batch_8/recall": 0.6331835985183716,
  "batch_9/hmean": 0.5012422800064087,
  "batch_9/precision": 0.7423297762870789,
  "batch_9/recall": 0.4786299169063568,
  "checkpoint_dir": "/home/vscode/workspace/upstageailab-ocr-recsys-competition-ocr-2/outputs/benchmark_optimized_32bit_full_cache/checkpoints",
  "epoch": 3,
  "final_mean_loss": 0,
  "final_run_name": "wchoi189_dbnet-resnet18-unet-dbhead-dbloss-bs16-lr1e-3_hmean0.596",
  "final_status": "success",
  "lr-AdamW": 0.001,
  "test/hmean": 0.5957932472229004,
  "test/precision": 0.6895830631256104,
  "test/recall": 0.5784671902656555,
  "train/loss": 2.2909860610961914,
  "train/loss_binary": 0.2714528441429138,
  "train/loss_prob": 0.3321135640144348,
  "train/loss_thresh": 0.03589652478694916,
  "trainer/global_step": 309,
  "val/hmean": 0.7428612112998962,
  "val/precision": 0.8623776435852051,
  "val/recall": 0.7238782644271851,
  "val_loss": 1.9897648096084597,
  "val_loss_binary": 0.2471596747636795,
  "val_loss_prob": 0.30108165740966797,
  "val_loss_thresh": 0.023719701915979385,
  "validation_images": {
    "_type": "images/separated",
    "captions": [
      "drp.en_ko.in_house.selectstar_000058.jpg (960x1280) | Ep 2 | GT=115 Pred=136",
      "drp.en_ko.in_house.selectstar_000032.jpg (959x1280) | Ep 2 | GT=155 Pred=15",
      "drp.en_ko.in_house.selectstar_000007.jpg (960x1280) | Ep 2 | GT=76 Pred=76",
      "drp.en_ko.in_house.selectstar_000067.jpg (1006x1280) | Ep 2 | GT=81 Pred=81",
      "drp.en_ko.in_house.selectstar_000064.jpg (682x1280) | Ep 2 | GT=106 Pred=101",
      "drp.en_ko.in_house.selectstar_000030.jpg (927x1280) | Ep 2 | GT=127 Pred=94",
      "drp.en_ko.in_house.selectstar_000046.jpg (920x1280) | Ep 2 | GT=104 Pred=110",
      "drp.en_ko.in_house.selectstar_000056.jpg (960x1280) | Ep 2 | GT=196 Pred=74"
    ],
    "count": 8,
    "filenames": [
      "media/images/validation_images_54_31128b8992913e2bb8b6.png",
      "media/images/validation_images_54_707924c915b64b517097.png",
      "media/images/validation_images_54_67e9476acecad6208054.png",
      "media/images/validation_images_54_e951bdb3d01e99b05a96.png",
      "media/images/validation_images_54_88e24dddde8491ee1c89.png",
      "media/images/validation_images_54_551e588da1608e9b8aab.png",
      "media/images/validation_images_54_284b9cc2789480308ddd.png",
      "media/images/validation_images_54_d3e4e96cebf38b00547d.png"
    ],
    "format": "png",
    "height": 1280,
    "width": 1006
  }
}
```
