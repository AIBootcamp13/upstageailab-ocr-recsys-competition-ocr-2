# Master Prompt
You are an autonomous AI agent, my Chief of Staff for implementing the **Checkpoint Catalog Refactor**. Your primary responsibility is to execute the "Living Implementation Blueprint" systematically, handle outcomes, and keep track of our progress. Do not ask for clarification on what to do next; your next task is always explicitly defined.

# Living Implementation Blueprint: Checkpoint Catalog Refactor

## Updated Living Blueprint
## Progress Tracker
- **STATUS:** Phase 3 Complete
- **CURRENT STEP:** Phase 3, Task 3.2 - Refactor Catalog Service
- **LAST COMPLETED TASK:** Task 3.1 - Add Wandb Fallback Logic ✅
- **NEXT TASK:** Refactor checkpoint_catalog.py to use new modules

### Latest Discoveries (2025-10-18)
- **Critical Bottleneck Identified**: `torch.load()` called up to 2x per checkpoint (2-5 sec each)
- **Performance Impact**: Current implementation 40-100x slower than target
- **Key Finding**: Checkpoint loading accounts for 85-90% of catalog build time
- **Opportunity**: YAML metadata files can eliminate checkpoint loading entirely
- **Analysis Document**: [docs/ai_handbook/05_changelog/2025-10/18_checkpoint_catalog_analysis.md](docs/ai_handbook/05_changelog/2025-10/18_checkpoint_catalog_analysis.md)
- **Architecture Design**: [docs/ai_handbook/03_references/architecture/checkpoint_catalog_v2_design.md](docs/ai_handbook/03_references/architecture/checkpoint_catalog_v2_design.md)
- **Module Implementation**: Complete Pydantic V2 models and module skeleton created in `ui/apps/inference/services/checkpoint/`
- **Metadata Requirements**: Confirmed inclusion of precision, recall, hmean, epoch per user requirements
- **Metadata Callback**: MetadataCallback implemented for automatic YAML generation during training
- **Conversion Tool**: Legacy checkpoint conversion tool completed with multi-source extraction strategy
- **Metadata Extraction**: Successfully extracts from cleval_metrics, Hydra config, and callback state
- **Wandb Fallback**: Implemented Wandb API client with caching for metadata retrieval when YAML files unavailable
- **Fallback Hierarchy**: YAML → Wandb API → Inference (3-tier strategy for maximum compatibility)

### Implementation Outline (Checklist)

#### **Phase 1: Analysis & Design (Week 1)**
1. [x] **Task 1.1: Analyze Current System** ✅
   - [x] Review `checkpoint_catalog.py` for performance bottlenecks and complexity
   - [x] Document current data flow and dependencies
   - [x] Identify redundant operations (e.g., repeated checkpoint loading)

2. [x] **Task 1.2: Design Modular Architecture**
   - [x] Define modules: `metadata_loader.py`, `config_resolver.py`, `validator.py`, `wandb_client.py`
   - [x] Specify interfaces using Pydantic models
   - [x] Plan YAML-based metadata structure

#### **Phase 2: Core Implementation (Week 2-3)**
3. [x] ✅ **Task 2.1: Implement Metadata Generation**
   - [x] Create `MetadataCallback` for Lightning training
   - [x] Generate `.metadata.yaml` files during training
   - [x] Update training configs to include callback

4. [x] ✅ **Task 2.2: Build Conversion Tool**
   - [x] Develop `legacy_config_converter.py` script
   - [x] Convert existing checkpoints to YAML metadata
   - [x] Test conversion on sample checkpoints

5. [x] ✅ **Task 2.3: Implement Scalable Validation**
   - [x] Add Pydantic-based validation in `validator.py`
   - [x] Support batch validation for large catalogs
   - [x] Integrate with UI inference compatibility schema

#### **Phase 3: Integration & Fallbacks (Week 4)**
6. [ ] **Task 3.1: Add Wandb Fallback Logic**
   - [ ] Implement `wandb_client.py` for run ID lookups
   - [ ] Add fallback hierarchy: YAML → Wandb → Inference
   - [ ] Handle offline scenarios gracefully

7. [ ] **Task 3.2: Refactor Catalog Service**
   - [ ] Simplify `checkpoint_catalog.py` to use new modules
   - [ ] Add caching layer for performance
   - [ ] Maintain backward compatibility

#### **Phase 4: Testing & Deployment (Week 5)**
8. [ ] **Task 4.1: Comprehensive Testing**
   - [ ] Unit tests for all new modules
   - [ ] Integration tests with UI components
   - [ ] Performance benchmarks (aim for 5-10x speedup)

9. [ ] **Task 4.2: Migration & Rollout**
   - [ ] Run conversion tool on all existing checkpoints
   - [ ] Update documentation and training workflows
   - [ ] Deploy with feature flags for gradual rollout

---

## 📋 **Technical Requirements Checklist**

### **Architecture & Design**
- [ ] Modular Design: Separate concerns into focused modules
- [ ] Pydantic V2 Integration: Use for all data models and validation
- [ ] YAML-Driven Configuration: Primary metadata format
- [ ] Caching Strategy: LRU cache for repeated catalog builds

### **Integration Points**
- [ ] Hydra Config Integration: Direct loading of resolved configs
- [ ] Wandb API Integration: Fallback for metadata retrieval
- [ ] Lightning Callback Integration: Automatic metadata generation
- [ ] UI Inference Compatibility: Schema-based validation

### **Quality Assurance**
- [ ] Unit Test Coverage Goal (> 90% for new modules)
- [ ] Integration Test Requirement: End-to-end catalog building
- [ ] Performance Test Requirement: <1s for small catalogs, <5s for large
- [ ] Backward Compatibility Test: Existing UI components unaffected

---

## 🎯 **Success Criteria Validation**

### **Functional Requirements**
- [ ] Checkpoint catalog builds 5-10x faster than current implementation
- [ ] YAML metadata files generated automatically during training
- [ ] Legacy conversion tool successfully migrates all existing checkpoints
- [ ] Wandb fallback loads configs when local metadata unavailable
- [ ] UI inference works seamlessly with new system

### **Technical Requirements**
- [ ] Code Quality Standard is Met: Fully typed, documented, and linted
- [ ] Resource Usage is Within Limits: <100MB memory for catalog operations
- [ ] Compatibility with Hydra/Lightning/Wandb is Confirmed
- [ ] Maintainability Goal is Met: Modular design enables easy extensions

---

## 📊 **Risk Mitigation & Fallbacks**

### **Current Risk Level**: MEDIUM
### **Active Mitigation Strategies**:
1. Incremental Development: Implement in phases with full testing at each step
2. Comprehensive Testing: Extensive unit and integration tests before deployment
3. Backward Compatibility: Maintain existing API until new system is validated

### **Fallback Options**:
1. If Wandb API fails: Fall back to local inference (current behavior)
2. If YAML generation fails: Skip metadata file, use runtime inference
3. If performance regression: Revert to original catalog with optimizations only

---

## 🔄 **Blueprint Update Protocol**

**Update Triggers:**
- Task completion (move to next task)
- Blocker encountered (document and propose solution)
- Technical discovery (update approach if needed)
- Quality gate failure (address issues before proceeding)

**Update Format:**
1. Update Progress Tracker (STATUS, CURRENT STEP, LAST COMPLETED TASK, NEXT TASK)
2. Mark completed items with [x]
3. Add any new discoveries or changes to approach
4. Update risk assessment if needed

---

## 🚀 **Immediate Next Action**

**TASK:** Add Wandb Fallback Logic (Task 3.1)

**OBJECTIVE:** Implement Wandb API client for metadata fallback when local YAML files are unavailable

**APPROACH:**
1. Create `wandb_client.py` module for Wandb API interactions
2. Implement run ID lookup and config retrieval
3. Add fallback hierarchy: YAML → Wandb → Inference
4. Handle offline scenarios gracefully with proper error messages
5. Cache Wandb responses to minimize API calls

**SUCCESS CRITERIA:**
- Wandb client can fetch configs by run ID
- Fallback chain works seamlessly (tries YAML first, then Wandb)
- Offline mode doesn't crash (falls back to inference)
- API responses are cached for performance
- Integration with existing checkpoint catalog
