# Performance Baseline Report

**Generated:** 2025-10-11 12:59:31
**WandB Run:** [wchoi189_dbnet-resnet18-pan-decoder-db-head-db-loss-bs16-lr1e-3_hmean0.953](https://wandb.ai/runs/niuscv84)
**Run ID:** `niuscv84`
**Status:** finished

---

## Training Metrics Summary

**Note:** This run did not have performance profiling enabled. Showing available training metrics instead.

| Metric | Validation | Test |
|--------|------------|------|
| **H-Mean** | 0.9527 | 0.9527 |
| **Precision** | 0.9514 | 0.9514 |
| **Recall** | 0.9561 | 0.9561 |

| Training Details | Value |
|------------------|-------|
| **Training Loss** | 1.1902 |
| **Validation Loss** | 1.4441 |
| **Epoch** | 20 |
| **Global Step** | 2060 |

## Training vs Validation Comparison

- **Note:** Performance profiling not enabled - cannot compare training vs validation timing.

## Identified Issues

No major issues detected in available metrics.

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
        "name": "pan_decoder",
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
    "log_dir": "outputs/pan_resnet18_add_polygons_canonical/logs",
    "output_dir": "outputs/pan_resnet18_add_polygons_canonical",
    "checkpoint_dir": "outputs/pan_resnet18_add_polygons_canonical/checkpoints",
    "submission_dir": "outputs/pan_resnet18_add_polygons_canonical/submissions"
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
      "save_dir": "outputs/pan_resnet18_add_polygons_canonical"
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
  "resume": "/home/vscode/workspace/upstageailab-ocr-recsys-competition-ocr-2/outputs/pan_resnet18_no_polygons_canonical/checkpoints/last_20251011_004442.ckpt",
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
    "precision": "16-mixed",
    "max_epochs": 20,
    "accelerator": "gpu",
    "deterministic": true,
    "gradient_clip_val": 5,
    "limit_val_batches": null,
    "log_every_n_steps": 50,
    "limit_test_batches": null,
    "val_check_interval": null,
    "limit_train_batches": null,
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
      "image_path": "/home/vscode/workspace/upstageailab-ocr-recsys-competition-ocr-2/data/datasets/images/test",
      "annotation_path": null
    }
  },
  "exp_name": "pan_resnet18_add_polygons_canonical",
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
      "dirpath": "outputs/pan_resnet18_add_polygons_canonical/checkpoints",
      "monitor": "val/hmean",
      "verbose": true,
      "_target_": "ocr.lightning_modules.callbacks.unique_checkpoint.UniqueModelCheckpoint",
      "filename": "epoch_{epoch:02d}_step_{step:06d}",
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
      "log_every_n_epochs": 5
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
  "experiment_tag": null,
  "lightning_path": "ocr.lightning_modules",
  "dataset_base_path": "/home/vscode/workspace/upstageailab-ocr-recsys-competition-ocr-2/data/datasets/",
  "default_interpolation": 1
}
```

### Summary Values
```json
{
  "_runtime": 2989,
  "_step": 74,
  "_timestamp": 1760123547.2355924,
  "_wandb": {
    "runtime": 2989
  },
  "batch_0/hmean": 0.971099615097046,
  "batch_0/precision": 0.9740210771560668,
  "batch_0/recall": 0.96881765127182,
  "batch_1/hmean": 0.9695701003074646,
  "batch_1/precision": 0.9612401127815248,
  "batch_1/recall": 0.9783722758293152,
  "batch_10/hmean": 0.9087637066841124,
  "batch_10/precision": 0.904483199119568,
  "batch_10/recall": 0.914007008075714,
  "batch_11/hmean": 0.978593409061432,
  "batch_11/precision": 0.9699248671531676,
  "batch_11/recall": 0.98771870136261,
  "batch_12/hmean": 0.9704917073249816,
  "batch_12/precision": 0.9653931260108948,
  "batch_12/recall": 0.9760769605636596,
  "batch_13/hmean": 0.9839971661567688,
  "batch_13/precision": 0.9802684783935548,
  "batch_13/recall": 0.987856149673462,
  "batch_14/hmean": 0.9747289419174194,
  "batch_14/precision": 0.9675267934799194,
  "batch_14/recall": 0.9822521805763244,
  "batch_15/hmean": 0.9768607020378112,
  "batch_15/precision": 0.9714381098747252,
  "batch_15/recall": 0.9826997518539428,
  "batch_16/hmean": 0.9617422819137572,
  "batch_16/precision": 0.9590545892715454,
  "batch_16/recall": 0.9655871987342834,
  "batch_17/hmean": 0.9791979789733888,
  "batch_17/precision": 0.9736434817314148,
  "batch_17/recall": 0.9850071668624878,
  "batch_18/hmean": 0.967177391052246,
  "batch_18/precision": 0.9568549990653992,
  "batch_18/recall": 0.9779912829399108,
  "batch_19/hmean": 0.9848189353942872,
  "batch_19/precision": 0.987313747406006,
  "batch_19/recall": 0.9826082587242126,
  "batch_2/hmean": 0.9478448629379272,
  "batch_2/precision": 0.9532461762428284,
  "batch_2/recall": 0.9493290781974792,
  "batch_20/hmean": 0.9201071858406068,
  "batch_20/precision": 0.9301711320877076,
  "batch_20/recall": 0.9116042256355286,
  "batch_21/hmean": 0.9817984104156494,
  "batch_21/precision": 0.9818720817565918,
  "batch_21/recall": 0.9818623661994934,
  "batch_22/hmean": 0.987366795539856,
  "batch_22/precision": 0.9859328866004944,
  "batch_22/recall": 0.9888481497764589,
  "batch_23/hmean": 0.9841288924217224,
  "batch_23/precision": 0.9799659848213196,
  "batch_23/recall": 0.9887354373931884,
  "batch_24/hmean": 0.855994462966919,
  "batch_24/precision": 0.8621127605438232,
  "batch_24/recall": 0.8512176275253296,
  "batch_25/hmean": 0.9884366989135742,
  "batch_25/precision": 0.986172616481781,
  "batch_25/recall": 0.9908307790756226,
  "batch_3/hmean": 0.7925618290901184,
  "batch_3/precision": 0.7948464751243591,
  "batch_3/recall": 0.7912514805793762,
  "batch_4/hmean": 0.9817115664482116,
  "batch_4/precision": 0.980058491230011,
  "batch_4/recall": 0.9838101863861084,
  "batch_5/hmean": 0.982292115688324,
  "batch_5/precision": 0.982518196105957,
  "batch_5/recall": 0.9823874235153198,
  "batch_6/hmean": 0.9645936489105223,
  "batch_6/precision": 0.9679576754570008,
  "batch_6/recall": 0.9624186754226683,
  "batch_7/hmean": 0.9700977802276612,
  "batch_7/precision": 0.9588280916213988,
  "batch_7/recall": 0.982513666152954,
  "batch_8/hmean": 0.9581848978996276,
  "batch_8/precision": 0.9551894068717957,
  "batch_8/recall": 0.9625146985054016,
  "batch_9/hmean": 0.856143057346344,
  "batch_9/precision": 0.8768925666809082,
  "batch_9/recall": 0.8680057525634766,
  "checkpoint_dir": "/home/vscode/workspace/upstageailab-ocr-recsys-competition-ocr-2/outputs/pan_resnet18_add_polygons_canonical/checkpoints",
  "epoch": 20,
  "final_mean_loss": 0,
  "final_run_name": "wchoi189_dbnet-resnet18-pan-decoder-db-head-db-loss-bs16-lr1e-3_hmean0.953",
  "final_status": "success",
  "lr-AdamW": 0.001,
  "test/hmean": 0.9526997804641724,
  "test/precision": 0.9514414668083192,
  "test/recall": 0.9560871720314026,
  "train/loss": 1.1902351379394531,
  "train/loss_binary": 0.15532761812210083,
  "train/loss_prob": 0.17814670503139496,
  "train/loss_thresh": 0.01441738847643137,
  "trainer/global_step": 2060,
  "val/hmean": 0.9526997804641724,
  "val/precision": 0.9514414668083192,
  "val/recall": 0.9560871720314026,
  "val_loss": 1.4440712928771973,
  "val_loss_binary": 0.17774108052253723,
  "val_loss_prob": 0.22416652739048004,
  "val_loss_thresh": 0.014549781568348408,
  "validation_images": {
    "_type": "images/separated",
    "captions": [
      "drp.en_ko.in_house.selectstar_000030.jpg (927x1280) | Ep 15 | GT=127 Pred=109",
      "drp.en_ko.in_house.selectstar_000046.jpg (920x1280) | Ep 15 | GT=104 Pred=101",
      "drp.en_ko.in_house.selectstar_000056.jpg (960x1280) | Ep 15 | GT=196 Pred=131",
      "drp.en_ko.in_house.selectstar_000032.jpg (959x1280) | Ep 15 | GT=155 Pred=131",
      "drp.en_ko.in_house.selectstar_000007.jpg (960x1280) | Ep 15 | GT=76 Pred=77",
      "drp.en_ko.in_house.selectstar_000058.jpg (960x1280) | Ep 15 | GT=115 Pred=116",
      "drp.en_ko.in_house.selectstar_000064.jpg (682x1280) | Ep 15 | GT=106 Pred=102",
      "drp.en_ko.in_house.selectstar_000067.jpg (1006x1280) | Ep 15 | GT=81 Pred=75"
    ],
    "count": 8,
    "filenames": [
      "media/images/validation_images_33_9b81cdb0a120f0e282aa.png",
      "media/images/validation_images_33_70dbf5b3e5dd396f7a91.png",
      "media/images/validation_images_33_388510e7aa34c0391302.png",
      "media/images/validation_images_33_ad1a42328fbc16e360ae.png",
      "media/images/validation_images_33_a87f92bd19374c67d286.png",
      "media/images/validation_images_33_7b4a00117079b45f9b39.png",
      "media/images/validation_images_33_bd152d28d3f85b894647.png",
      "media/images/validation_images_33_2bff899e3522ed69b937.png"
    ],
    "format": "png",
    "height": 1280,
    "width": 1006
  }
}
```
