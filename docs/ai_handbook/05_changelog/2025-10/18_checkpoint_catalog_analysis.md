# Checkpoint Catalog System Analysis

**Date**: 2025-10-18
**Status**: Analysis Complete
**Related**: [checkpoint_catalog_refactor_plan.md](../../../../checkpoint_catalog_refactor_plan.md)

## Executive Summary

The current checkpoint catalog system ([checkpoint_catalog.py](../../../../ui/apps/inference/services/checkpoint_catalog.py)) suffers from significant performance and complexity issues due to:
1. **Repeated checkpoint loading** - Full PyTorch checkpoint loaded multiple times per checkpoint
2. **Complex fallback chains** - Metadata resolution cascades through 5+ sources
3. **Monolithic architecture** - 1,136 lines in a single file with tightly coupled concerns
4. **No caching** - Every catalog build re-processes all checkpoints from scratch
5. **Blocking I/O** - Sequential processing of all checkpoints with no parallelization

**Estimated Impact**: Current implementation takes ~10-30 seconds for catalogs with 20+ checkpoints. Target: <1-2 seconds.

---

## Current Architecture Analysis

### Data Flow Overview

```
build_catalog()
    ↓
_list_checkpoints() → [checkpoint_paths]
    ↓
For each checkpoint:
    _collect_metadata()
        ↓
        _load_metadata_dict()        # Read .metadata.json/yaml
        _resolve_config_path()        # Search for config files
        _load_config_dict()           # Read config sidecar
        _load_hydra_config()          # Read .hydra/config.yaml
        ↓
        _extract_epoch()              # Parse metadata
        _extract_created_timestamp()
        _extract_metric()
        _resolve_model_details()      # Parse config for architecture
        _extract_checkpointing_settings()
        ↓
        _load_checkpoint()            # ⚠️ BOTTLENECK: Load full .ckpt with PyTorch
            ↓
            torch.load() → ~500MB-2GB checkpoint
        ↓
        _extract_training_from_checkpoint()
        _extract_cleval_metric()
        _extract_checkpoint_metrics()
        _extract_checkpointing_from_checkpoint()
        ↓
        _ensure_resolved_config()
            ↓
            _infer_names_from_path()
            _load_checkpoint()        # ⚠️ LOADED AGAIN if not already loaded
            _infer_encoder_from_checkpoint()
            _extract_state_signatures_from_checkpoint()
            _build_config_dict()
            → Write .resolved.config.json
    ↓
schema.validate()
    ↓
Sort and return
```

### Key Bottlenecks Identified

#### 1. **Checkpoint Loading (CRITICAL)**
- **Location**: [checkpoint_catalog.py:1037-1061](../../../../ui/apps/inference/services/checkpoint_catalog.py#L1037-L1061)
- **Issue**: `torch.load()` called up to 2 times per checkpoint
- **Cost**:
  - Each checkpoint is 500MB-2GB
  - Loading takes 2-5 seconds per checkpoint
  - No caching between calls
- **Trigger Conditions**: Missing metadata fields (lines 211-222, 977-978)
- **Impact**: Dominates catalog build time (>90% of total time)

```python
# Current implementation - called conditionally
checkpoint_data: dict[str, Any] | None = None
if (metadata.validation_loss is None or metadata.hmean is None or ...):
    checkpoint_data = _load_checkpoint(checkpoint_path)  # 2-5 seconds!
```

#### 2. **Metadata Resolution Complexity**
- **Fallback Hierarchy** (in order):
  1. `.metadata.json/.metadata.yaml` files
  2. `.config.json/.config.yaml` sidecars
  3. `.hydra/config.yaml` from parent directories
  4. Checkpoint file itself (PyTorch tensor loading)
  5. Filename pattern inference
  6. State dict signature inference

- **Issue**: Each fallback requires disk I/O and parsing
- **Impact**: Even with metadata files, we check 4+ sources per checkpoint

#### 3. **Config Resolution & Inference**
- **Function**: `_ensure_resolved_config()` (lines 955-1034)
- **Complexity**:
  - Infers architecture from: path patterns → config → state dict
  - Infers encoder from: metadata → config → state dict weights
  - Extracts decoder/head signatures from state dict
  - Generates and writes `.resolved.config.json` files
- **Issue**: Runs for every checkpoint, even if config is valid
- **Cost**: Additional file I/O + checkpoint loading

#### 4. **State Dict Signature Extraction**
- **Function**: `_extract_state_signatures_from_checkpoint()` (lines 701-834)
- **Complexity**: 133 lines of pattern matching on PyTorch state dict keys
- **Issue**: Requires full checkpoint load to read weight tensor shapes
- **Examples**:
  ```python
  # Decoder signature extraction
  if any(key.startswith(f"{decoder_prefix}bottom_up") for key in state_dict):
      decoder_name = "pan_decoder"
      # Extract tensor shapes from state dict
      shape = tuple(weight.shape)
  ```

#### 5. **No Caching Layer**
- No memoization between catalog builds
- UI rebuilds catalog on every page refresh/interaction
- No incremental updates for new checkpoints

#### 6. **Sequential Processing**
- Single-threaded loop over all checkpoints
- No parallelization despite independent operations
- Each checkpoint processed in series

---

## Dependencies & Integration Points

### External Dependencies
1. **PyTorch**: For `torch.load()` - heavyweight dependency
2. **OmegaConf**: For DictConfig/ListConfig handling
3. **Pydantic**: For validation (light usage currently)
4. **PyYAML**: For config file parsing

### Internal Dependencies
1. **Models**:
   - [CheckpointMetadata](../../../../ui/apps/inference/models/checkpoint.py#L68-L217)
   - [CheckpointInfo](../../../../ui/apps/inference/models/checkpoint.py#L220-L366) (lightweight variant)
   - DecoderSignature, HeadSignature (dataclasses)
   - CheckpointMetadataSchema (Pydantic model - underutilized)

2. **Schema Validation**:
   - [ModelCompatibilitySchema](../../../../ui/apps/inference/services/schema_validator.py)
   - Used for final validation only
   - Not leveraged for metadata structure

3. **UI Components**:
   - [checkpoint_catalog.py](../../../../ui/apps/inference/services/checkpoint_catalog.py) service
   - [inference_runner.py](../../../../ui/apps/inference/services/inference_runner.py)
   - Streamlit UI components

### File Structure Dependencies
```
outputs/
└── experiment_name/
    ├── checkpoints/
    │   ├── epoch=10.ckpt              # 500MB-2GB PyTorch checkpoint
    │   ├── epoch=10.metadata.json     # NEW: Proposed metadata
    │   ├── epoch=10.config.yaml       # Optional config sidecar
    │   └── epoch=10.resolved.config.json  # Generated by catalog
    └── .hydra/
        └── config.yaml                # Hydra configuration
```

---

## Redundant Operations

### 1. **Duplicate Checkpoint Loading**
- Loaded in `_collect_metadata()` (line 222)
- Loaded again in `_ensure_resolved_config()` (line 978)
- Same data, no sharing between calls

### 2. **Repeated File Searches**
- Config path resolution searches 6+ directories per checkpoint
- No caching of search results
- Same searches repeated for metadata, config, hydra files

### 3. **Redundant Parsing**
- Metadata extracted from checkpoint even when `.metadata.json` exists
- Full config parsing even when only specific fields needed
- State dict traversal for signatures when config has the info

### 4. **Unnecessary Inference**
- Architecture inference from path when metadata has it
- Encoder inference from state dict when config has it
- Decoder signature extraction when metadata could provide it

### 5. **Validation Overhead**
- Schema validation at the end of entire pipeline
- No early validation to fail fast
- No validation of metadata files themselves

---

## Performance Metrics (Estimated)

### Current Implementation
| Operation | Time per Checkpoint | % of Total |
|-----------|---------------------|------------|
| Checkpoint loading | 2-5 seconds | 85-90% |
| Config file I/O | 50-200ms | 5-8% |
| Metadata parsing | 10-50ms | 2-3% |
| State dict analysis | 200-500ms | 3-5% |
| Path inference | 5-10ms | <1% |

**Total for 20 checkpoints**: 40-100 seconds (worst case with missing metadata)

### Target Implementation (Post-Refactor)
| Operation | Time per Checkpoint | % of Total |
|-----------|---------------------|------------|
| Metadata YAML load | 5-10ms | 40% |
| Config resolution | 5-10ms | 30% |
| Validation | 2-5ms | 20% |
| Path inference | 1-2ms | 10% |
| Checkpoint loading | 0ms (fallback only) | 0% |

**Target for 20 checkpoints**: 0.5-1 second

**Expected Speedup**: **40-100x faster**

---

## Opportunities for Modularization

### Proposed Module Structure

```
ui/apps/inference/services/checkpoint/
├── __init__.py
├── catalog.py                 # Main catalog builder (orchestration)
├── metadata_loader.py         # YAML metadata loading
├── config_resolver.py         # Config file resolution & loading
├── validator.py               # Pydantic-based validation
├── wandb_client.py            # Wandb API fallback
├── inference_engine.py        # Checkpoint inference (state dict analysis)
├── cache.py                   # LRU cache for catalog results
└── types.py                   # Pydantic models for all data structures
```

### Module Responsibilities

#### 1. **metadata_loader.py**
- Load `.metadata.yaml` files
- Parse and validate structure
- Return Pydantic models
- **No fallbacks** - just metadata files

#### 2. **config_resolver.py**
- Resolve config file paths
- Load Hydra configs
- Merge overrides
- Return structured config data

#### 3. **validator.py**
- Pydantic V2 validation
- Schema compatibility checks
- Batch validation support
- Error reporting

#### 4. **wandb_client.py**
- Query Wandb API for run metadata
- Download Hydra configs
- Handle authentication & offline mode
- Cache results

#### 5. **inference_engine.py**
- Load checkpoints (last resort)
- Extract state dict signatures
- Infer architecture from weights
- **Only called when metadata unavailable**

#### 6. **catalog.py**
- Orchestrate metadata collection
- Apply fallback hierarchy
- Cache results
- Expose simple API

---

## Recommended Approach

### Phase 1: Add Metadata Generation (Week 1)
1. Create `MetadataCallback` for Lightning
2. Generate `.metadata.yaml` during training
3. Test on new training runs

### Phase 2: Implement Core Modules (Week 2-3)
1. Build `metadata_loader.py` + `validator.py`
2. Implement `config_resolver.py`
3. Add `wandb_client.py` fallback
4. Create `cache.py` layer

### Phase 3: Refactor Catalog (Week 3-4)
1. Simplify `catalog.py` to use new modules
2. Implement fallback hierarchy
3. Add caching
4. Maintain backward compatibility

### Phase 4: Migration (Week 4-5)
1. Create legacy conversion tool
2. Convert existing checkpoints
3. Performance testing
4. Gradual rollout

---

## Risk Assessment

### Technical Risks
1. **Backward Compatibility**: Existing checkpoints lack metadata files
   - **Mitigation**: Conversion tool + fallback to current inference logic

2. **Wandb API Rate Limits**: Batch metadata lookups may hit limits
   - **Mitigation**: Caching + exponential backoff + offline mode

3. **YAML Format Changes**: Future schema evolution
   - **Mitigation**: Version field in metadata + migration scripts

### Operational Risks
1. **Training Pipeline Changes**: Requires callback integration
   - **Mitigation**: Backward compatible, optional initially

2. **Storage Overhead**: Metadata files add ~5-10KB per checkpoint
   - **Impact**: Negligible compared to 500MB-2GB checkpoints

---

## Next Steps

1. ✅ **Analysis Complete** (this document)
2. ⏭️ **Design Modular Architecture** (Task 1.2)
   - Define Pydantic models for metadata
   - Specify module interfaces
   - Design YAML schema
3. **Implement Metadata Generation** (Task 2.1)
4. **Build Conversion Tool** (Task 2.2)
5. **Refactor Catalog Service** (Task 3.2)

---

## References

- Current Implementation: [checkpoint_catalog.py:39-1136](../../../../ui/apps/inference/services/checkpoint_catalog.py#L39-L1136)
- Data Models: [checkpoint.py](../../../../ui/apps/inference/models/checkpoint.py)
- Schema Validator: [schema_validator.py](../../../../ui/apps/inference/services/schema_validator.py)
- Master Plan: [checkpoint_catalog_refactor_plan.md](../../../../checkpoint_catalog_refactor_plan.md)
