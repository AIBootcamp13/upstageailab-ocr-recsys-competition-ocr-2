
## Wandb metadata (raw text)
Run path
ocr-team2/receipt-text-recognition-ocr-project/wj0l1mw3
Hostname
instance-15983
OS
Linux-5.4.0-166-generic-x86_64-with-glibc2.31
Python version
CPython 3.10.18
Python executable
/home/vscode/workspace/upstageailab-ocr-recsys-competition-ocr-2/.venv/bin/python3
Git repository
git clone
Git state
git checkout -b "wchoi189_debug_dbnet-resnet18-fpn-decoder-db-head-db-loss-bs8-lr1e-3_hmean0.000" 953ce079afcc39ded5ca46a1e55766c78014fdd5
Command
/home/vscode/workspace/upstageailab-ocr-recsys-competition-ocr-2/runners/train.py trainer.max_epochs=2 callbacks.throughput_monitor.enabled=true callbacks.profiler.enabled=true callbacks.profiler.profile_epochs=[1,2] callbacks.resource_monitor.enabled=true

---

  "test/hmean": 0,
  "test/precision": 0,
  "test/recall": 0,
  "trainer/global_step": 4,
  "val/hmean": 0,
  "val/precision": 0,
  "val/recall": 0,
  "val_loss": 60.96722030639648,
  "val_loss_binary": 0.8898123502731323,
  "val_loss_prob": 10.933058738708496,
  "val_loss_thresh": 0.5412113666534424

---

# Separate training run shows Prediction GT labels are missing.
```bash
andb: WARNING Changes to your `wandb` environment variables will be ignored because your `wandb` session has already started. For more information on how to modify your settings with `wandb.init()` arguments, please refer to https://wandb.me/wandb-init.
wandb: Tracking run with wandb version 0.22.0
wandb: Run data is saved locally in ./wandb/run-20251008_133123-wisr44wq
wandb: Run `wandb offline` to turn off syncing.
wandb: Syncing run wchoi189_debug_dbnet-resnet18-fpn-decoder-db-head-db-loss-bs8-lr3e-4_SCORE_PLACEHOLDER
wandb: ⭐️ View project at https://wandb.ai/ocr-team2/receipt-text-recognition-ocr-project
wandb: 🚀 View run at https://wandb.ai/ocr-team2/receipt-text-recognition-ocr-project/runs/wisr44wq
LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]

  | Name   | Type         | Params | Mode
------------------------------------------------
0 | model  | OCRModel     | 16.5 M | train
1 | metric | CLEvalMetric | 0      | train
------------------------------------------------
16.5 M    Trainable params
0         Non-trainable params
16.5 M    Total params
65.897    Total estimated model params size (MB)
155       Modules in train mode
0         Modules in eval mode
Sanity Checking: |                                                                   | 0/? [00:00<?, ?it/s]✅ PolygonCache enabled: max_size=False, persist=True
Cache stats: hits=0, misses=156, hit_rate=0.00%, size=0
Cache stats: hits=0, misses=160, hit_rate=0.00%, size=0
Cache stats: hits=0, misses=160, hit_rate=0.00%, size=0
Sanity Checking DataLoader 0: 100%|██████████████████████████████████████████| 1/1 [00:03<00:00,  0.27it/s[2025-10-08 13:31:31,045][root][WARNING] - Missing predictions for ground truth file 'drp.en_ko.in_house.selectstar_000082.jpg' during validation epoch end. This may indicate a data loading or prediction issue.
[2025-10-08 13:31:31,045][root][WARNING] - Missing predictions for ground truth file 'drp.en_ko.in_house.selectstar_000127.jpg' during validation epoch end. This may indicate a data loading or prediction issue.
[2025-10-08 13:31:31,045][root][WARNING] - Missing predictions for ground truth file 'drp.en_ko.in_house.selectstar_000130.jpg' during validation epoch end. This may indicate a data loading or prediction issue.
[2025-10-08 13:31:31,046][root][WARNING] - Missing predictions for ground truth file 'drp.en_ko.in_house.selectst

```
