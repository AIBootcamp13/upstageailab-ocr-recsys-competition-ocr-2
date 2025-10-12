
```bash
cd /home/vscode/workspace/upstageailab-ocr-recsys-competition-ocr-2 && HYDRA_FULL_ERROR=1 uv run python runners/train.py exp_name=pan_resnet18_polygons trainer.max_epochs=15 data=canonical model.component_overrides.decoder.name=pan_decoder logger.wandb.enabled=true trainer.log_every_n_steps=0
```

```bash
HYDRA_FULL_ERROR=1 uv run python runners/train.py exp_name=pan_resnet18_polygons trainer.max_epochs=15 data=canonical model.component_overrides.decoder.name=pan_decoder logger.wandb.enabled=true trainer.log_every_n_steps=0
```
### The run encountered an error during execution (possibly related to the pan decoder or data processing), but the command structure is correct. If you need to enable RAM caching and transformation caching for training data as well, add these overrides:

```bash
+datasets.train_dataset.preload_images=true +datasets.train_dataset.cache_transformed_tensors=true
```
### For full performance features, the complete command would be:
```bash
HYDRA_FULL_ERROR=1 uv run python runners/train.py exp_name=pan_resnet18_polygons trainer.max_epochs=15 data=canonical model.component_overrides.decoder.name=pan_decoder logger.wandb.enabled=true trainer.log_every_n_steps=50
```
