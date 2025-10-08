## Wandb metadata (raw text)
Start time
October 6th, 2025 7:42:11 PM
Runtime
7m 38s
Tracked hours
7m 36s
Run path
ocr-team2/receipt-text-recognition-ocr-project/9gket21o
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
git checkout -b "wchoi189_dbnet-resnet18-fpn-decoder-db-head-db-loss-bs8-lr3e-4_SCORE_PLACEHOLDER" 3f96e50d5d44e8d046827c9cc75d0b1f01f973d5
Command
/home/vscode/workspace/upstageailab-ocr-recsys-competition-ocr-2/runners/train.py exp_name=canonical-fix2-dbnet-fpn_decoder-mobilenetv3_small_050 logger.wandb.enabled=true model.architecture_name=dbnet model/architectures=dbnet model.encoder.model_name=mobilenetv3_small_050 model.component_overrides.decoder.name=fpn_decoder model.component_overrides.head.name=db_head model.component_overrides.loss.name=db_loss model/optimizers=adamw model.optimizer.lr=0.000305 model.optimizer.weight_decay=0.0001 dataloaders.train_dataloader.batch_size=8 dataloaders.val_dataloader.batch_size=8 trainer.max_epochs=15 trainer.accumulate_grad_batches=1 trainer.gradient_clip_val=5.0 trainer.precision=32


---

  "train/loss": 2.836628675460815,
  "train/loss_binary": 0.3608019351959229,
  "train/loss_prob": 0.4344182312488556,
  "train/loss_thresh": 0.03037354163825512,
  "trainer/global_step": 408,
  "val/hmean": 0.8906151652336121,
  "val/precision": 0.9102451801300048,
  "val/recall": 0.8842350840568542,
  "val_loss": 1.9595088958740232,
  "val_loss_binary": 0.24381934106349945,
  "val_loss_prob": 0.29272517561912537,
  "val_loss_thresh": 0.025206292048096657

----
