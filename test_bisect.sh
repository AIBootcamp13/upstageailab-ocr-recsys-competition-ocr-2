#!/bin/bash
# Test script for git bisect: check if missing GT labels warning appears
# Exit 0 if good (no warning), 1 if bad (warning present)

# Run a quick training for 1 epoch
uv run python runners/train.py trainer.max_epochs=1 logger.wandb.enabled=false dataloaders.train_dataloader.batch_size=8 dataloaders.val_dataloader.batch_size=8 model.architecture_name=dbnet model/architectures=dbnet model.encoder.model_name=resnet18 model/optimizers=adamw model.optimizer.lr=0.000305 model.optimizer.weight_decay=0.0001 trainer.accumulate_grad_batches=1 trainer.gradient_clip_val=5.0 trainer.precision=32 2>&1 | tee /tmp/bisect_log.txt

# Check if warning is present
if grep -q "Missing predictions for ground truth file" /tmp/bisect_log.txt; then
    echo "Bad: Missing GT labels warning found"
    exit 1
else
    echo "Good: No missing GT labels warning"
    exit 0
fi
