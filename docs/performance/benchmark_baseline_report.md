# Performance Baseline Report

**Generated:** 2025-10-13 23:58:12
**WandB Run:** [wchoi189_dbnet-resnet18-unet-dbhead-dbloss-bs16-lr1e-3_hmean0.700](https://wandb.ai/runs/humu1mxg)
**Run ID:** `humu1mxg`
**Status:** finished

---

## Training Metrics Summary

**Note:** This run did not have performance profiling enabled. Showing available training metrics instead.

| Metric | Validation | Test |
|--------|------------|------|
| **H-Mean** | 0.8784 | 0.6997 |
| **Precision** | 0.9269 | 0.7439 |
| **Recall** | 0.8483 | 0.6727 |

| Training Details | Value |
|------------------|-------|
| **Training Loss** | 2.1726 |
| **Validation Loss** | 2.0524 |
| **Epoch** | 3 |
| **Global Step** | 309 |

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
    "log_dir": "outputs/benchmark_baseline_32bit_no_cache/logs",
    "output_dir": "outputs/benchmark_baseline_32bit_no_cache",
    "checkpoint_dir": "outputs/benchmark_baseline_32bit_no_cache/checkpoints",
    "submission_dir": "outputs/benchmark_baseline_32bit_no_cache/submissions"
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
      "save_dir": "outputs/benchmark_baseline_32bit_no_cache"
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
          "cache_transformed_tensors": false
        },
        "preload_images": false,
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
  "exp_name": "benchmark_baseline_32bit_no_cache",
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
      "dirpath": "outputs/benchmark_baseline_32bit_no_cache/checkpoints",
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
  "_runtime": 1217,
  "_step": 58,
  "_timestamp": 1760366972.0893502,
  "_wandb": {
    "runtime": 1217
  },
  "batch_0/hmean": 0.885769784450531,
  "batch_0/precision": 0.9427141547203064,
  "batch_0/recall": 0.848908007144928,
  "batch_1/hmean": 0.8907831907272339,
  "batch_1/precision": 0.94380784034729,
  "batch_1/recall": 0.8614925742149353,
  "batch_10/hmean": 0.7876716256141663,
  "batch_10/precision": 0.8808034062385559,
  "batch_10/recall": 0.7413176894187927,
  "batch_11/hmean": 0.9232850670814514,
  "batch_11/precision": 0.9443398118019104,
  "batch_11/recall": 0.909799575805664,
  "batch_12/hmean": 0.9106990098953248,
  "batch_12/precision": 0.9324162006378174,
  "batch_12/recall": 0.8919405341148376,
  "batch_13/hmean": 0.9033551812171936,
  "batch_13/precision": 0.9609553813934326,
  "batch_13/recall": 0.8612326979637146,
  "batch_14/hmean": 0.8764419555664062,
  "batch_14/precision": 0.9431482553482056,
  "batch_14/recall": 0.8403711915016174,
  "batch_15/hmean": 0.8996005654335022,
  "batch_15/precision": 0.9481329321861268,
  "batch_15/recall": 0.8690060377120972,
  "batch_16/hmean": 0.8775591850280762,
  "batch_16/precision": 0.943559467792511,
  "batch_16/recall": 0.8324763774871826,
  "batch_17/hmean": 0.8903751373291016,
  "batch_17/precision": 0.9422723650932312,
  "batch_17/recall": 0.8589824438095093,
  "batch_18/hmean": 0.9154051542282104,
  "batch_18/precision": 0.9437695145606996,
  "batch_18/recall": 0.8961372971534729,
  "batch_19/hmean": 0.9192081093788148,
  "batch_19/precision": 0.9609350562095642,
  "batch_19/recall": 0.890880823135376,
  "batch_2/hmean": 0.896049439907074,
  "batch_2/precision": 0.9310941696166992,
  "batch_2/recall": 0.8758753538131714,
  "batch_20/hmean": 0.8748457431793213,
  "batch_20/precision": 0.9202761650085448,
  "batch_20/recall": 0.8402521014213562,
  "batch_21/hmean": 0.840972900390625,
  "batch_21/precision": 0.9347290992736816,
  "batch_21/recall": 0.7835084795951843,
  "batch_22/hmean": 0.9033992290496826,
  "batch_22/precision": 0.9475210905075072,
  "batch_22/recall": 0.8733051419258118,
  "batch_23/hmean": 0.9139199256896972,
  "batch_23/precision": 0.9457998275756836,
  "batch_23/recall": 0.8888741731643677,
  "batch_24/hmean": 0.7817736268043518,
  "batch_24/precision": 0.8540991544723511,
  "batch_24/recall": 0.7348809242248535,
  "batch_25/hmean": 0.9651669263839722,
  "batch_25/precision": 0.970099151134491,
  "batch_25/recall": 0.9608613848686218,
  "batch_3/hmean": 0.7223608493804932,
  "batch_3/precision": 0.7963598370552063,
  "batch_3/recall": 0.6710794568061829,
  "batch_4/hmean": 0.9125792980194092,
  "batch_4/precision": 0.96073317527771,
  "batch_4/recall": 0.8894285559654236,
  "batch_5/hmean": 0.9102669954299928,
  "batch_5/precision": 0.957526445388794,
  "batch_5/recall": 0.8752723336219788,
  "batch_6/hmean": 0.9108353853225708,
  "batch_6/precision": 0.9188198447227478,
  "batch_6/recall": 0.9067775011062622,
  "batch_7/hmean": 0.906515896320343,
  "batch_7/precision": 0.9371632933616638,
  "batch_7/recall": 0.8836649060249329,
  "batch_8/hmean": 0.8733475804328918,
  "batch_8/precision": 0.9308459162712096,
  "batch_8/recall": 0.857177734375,
  "batch_9/hmean": 0.8115342855453491,
  "batch_9/precision": 0.8376935720443726,
  "batch_9/recall": 0.7954960465431213,
  "checkpoint_dir": "/home/vscode/workspace/upstageailab-ocr-recsys-competition-ocr-2/outputs/benchmark_baseline_32bit_no_cache/checkpoints",
  "epoch": 3,
  "final_mean_loss": 0,
  "final_run_name": "wchoi189_dbnet-resnet18-unet-dbhead-dbloss-bs16-lr1e-3_hmean0.700",
  "final_status": "success",
  "lr-AdamW": 0.001,
  "test/hmean": 0.6996931433677673,
  "test/precision": 0.7438960671424866,
  "test/recall": 0.6727182269096375,
  "train/loss": 2.1725826263427734,
  "train/loss_binary": 0.2646055221557617,
  "train/loss_prob": 0.3229055106639862,
  "train/loss_thresh": 0.02934494987130165,
  "trainer/global_step": 309,
  "val/hmean": 0.8784475326538086,
  "val/precision": 0.9268598556518556,
  "val/recall": 0.8482516407966614,
  "val_loss": 2.052438974380493,
  "val_loss_binary": 0.25442084670066833,
  "val_loss_prob": 0.31045880913734436,
  "val_loss_thresh": 0.024572424590587616,
  "validation_images": {
    "_type": "images/separated",
    "captions": [
      "drp.en_ko.in_house.selectstar_000058.jpg (960x1280) | Ep 2 | GT=115 Pred=138",
      "drp.en_ko.in_house.selectstar_000032.jpg (959x1280) | Ep 2 | GT=155 Pred=99",
      "drp.en_ko.in_house.selectstar_000007.jpg (960x1280) | Ep 2 | GT=76 Pred=78",
      "drp.en_ko.in_house.selectstar_000067.jpg (1006x1280) | Ep 2 | GT=81 Pred=86",
      "drp.en_ko.in_house.selectstar_000064.jpg (682x1280) | Ep 2 | GT=106 Pred=101",
      "drp.en_ko.in_house.selectstar_000030.jpg (927x1280) | Ep 2 | GT=127 Pred=98",
      "drp.en_ko.in_house.selectstar_000046.jpg (920x1280) | Ep 2 | GT=104 Pred=101",
      "drp.en_ko.in_house.selectstar_000056.jpg (960x1280) | Ep 2 | GT=196 Pred=67"
    ],
    "count": 8,
    "filenames": [
      "media/images/validation_images_54_fd6bb6628093307247ab.png",
      "media/images/validation_images_54_93b32bf3210d516408f0.png",
      "media/images/validation_images_54_8652ad32093e9dbb614e.png",
      "media/images/validation_images_54_523f92ac46c8f62af8b8.png",
      "media/images/validation_images_54_5cd800a8ad1cb63aa8b4.png",
      "media/images/validation_images_54_1cdab0e7f4f2506c0335.png",
      "media/images/validation_images_54_5727571798bfeefe5774.png",
      "media/images/validation_images_54_5a592987570a61e6d55f.png"
    ],
    "format": "png",
    "height": 1280,
    "width": 1006
  }
}
```
