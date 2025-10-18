You are an autonomous AI agent, my Chief of Staff for implementing the **Timestamp-Based Folder Organization Migration**. Your primary responsibility is to execute the "Living Implementation Blueprint" systematically, handle outcomes, and keep track of our progress. Do not ask for clarification on what to do next; your next task is always explicitly defined.

# Living Implementation Blueprint: OCR Project - Timestamp-Based Folder Organization Migration

## Updated Living Blueprint

## Progress Tracker
- **STATUS:** In Progress
- **CURRENT STEP:** Phase 3, Task 3.1 - Update Training Runners
- **LAST COMPLETED TASK:** Created checkpoint metadata system with Pydantic schema
- **NEXT TASK:** Update training runners for experiment registry integration

### Implementation Outline (Checklist)

#### **Phase 1: Configuration Updates (Week 1)**
1. [x] **Task 1.1: Update Hydra Configuration**
   - [x] Change run directory to timestamp-based structure (`outputs/YYYY-MM-DD/HH-MM-SS`)
   - [x] Update sweep directory structure for multi-run experiments
   - [x] Ensure job chdir behavior remains consistent

2. [x] **Task 1.2: Update Paths Configuration**
   - [x] Set experiment_dir to use Hydra runtime output directory
   - [x] Update log_dir, checkpoint_dir, output_dir to use relative paths
   - [x] Maintain backward compatibility references

#### **Phase 2: Checkpoint System Refactoring (Week 2)**
3. [x] **Task 2.1: Modify UniqueModelCheckpoint Callback**
   - [x] Update `_setup_dirpath()` to use timestamp-based paths
   - [x] Remove experiment name dependency
   - [x] Ensure checkpoint naming remains informative
   - [x] Add metadata file generation alongside checkpoints

4. [x] **Task 2.2: Update Checkpoint Catalog Service**
   - [x] Implement timestamp-based discovery logic
   - [x] Simplify encoder detection (no more directory name parsing)
   - [x] Add metadata file parsing for checkpoint information
   - [x] Maintain backward compatibility with existing checkpoints

5. [x] **Task 2.3: Create Checkpoint Metadata System**
   - [x] Define metadata schema for checkpoint information
   - [x] Generate metadata files alongside checkpoints
   - [x] Include encoder, decoder, head, config hash, and training info
   - [x] Enable fast checkpoint filtering and discovery

#### **Phase 3: Training Integration (Week 3)**
6. [ ] **Task 3.1: Update Training Runners**
   - [ ] Integrate experiment registry for ID generation
   - [ ] Pass experiment ID through Hydra config system
   - [ ] Update logging and artifact storage
   - [ ] Ensure backward compatibility during transition

7. [ ] **Task 3.2: Create Experiment Registry Integration**
   - [ ] Integrate experiment registry with training runners for automatic ID generation
   - [ ] Update training pipeline to register experiments on startup
   - [ ] Track experiment metadata and status throughout training lifecycle
   - [ ] Enable experiment discovery and management in training scripts

#### **Phase 4: Migration Tools and Scripts (Week 4)**
8. [ ] **Task 4.1: Create Migration Script**
   - [ ] Scan existing `outputs/` directory
   - [ ] Create mapping from old experiment names to new timestamp structure
   - [ ] Copy/move files to new structure
   - [ ] Update any references in logs or metadata
   - [ ] Generate migration report

9. [ ] **Task 4.2: Backward Compatibility Layer**
   - [ ] Provide compatibility layer for old checkpoint discovery
   - [ ] Translate old paths to new structure during transition
   - [ ] Log deprecation warnings
   - [ ] Plan for eventual removal

10. [ ] **Task 4.3: Validation Scripts**
    - [ ] Verify all checkpoints are discoverable in new structure
    - [ ] Check metadata integrity
    - [ ] Validate training pipeline with new structure
    - [ ] Generate migration success report

#### **Phase 5: UI and Tool Updates (Week 5)**
11. [ ] **Task 5.1: Update Streamlit Inference UI**
    - [ ] Update checkpoint selection interface
    - [ ] Display experiment metadata (date, config, performance)
    - [ ] Simplify model compatibility validation
    - [ ] Add experiment filtering and search

12. [ ] **Task 5.2: Update Ablation Study Tools**
    - [ ] Update result collection to work with new structure
    - [ ] Modify analysis scripts for timestamp-based organization
    - [ ] Ensure backward compatibility during transition

#### **Phase 6: Testing and Validation (Week 6)**
13. [ ] **Task 6.1: Unit Tests**
    - [ ] Test new directory structure creation
    - [ ] Validate checkpoint discovery in new system
    - [ ] Test backward compatibility layer
    - [ ] Verify metadata generation and parsing

14. [ ] **Task 6.2: Integration Tests**
    - [ ] Run full training pipeline with new structure
    - [ ] Test multi-run experiments
    - [ ] Validate UI functionality
    - [ ] Test migration script on sample data

15. [ ] **Task 6.3: Performance Validation**
    - [ ] Measure checkpoint discovery time improvement
    - [ ] Validate training performance unchanged
    - [ ] Test large-scale experiment management

#### **Phase 7: Documentation and Training (Week 7)**
16. [ ] **Task 7.1: Update Documentation**
    - [ ] Update `docs/ai_handbook/` with new folder structure
    - [ ] Create migration guide for team members
    - [ ] Update README and contribution guidelines
    - [ ] Document new experiment management workflow

17. [ ] **Task 7.2: Team Training**
    - [ ] Conduct knowledge sharing session on new structure
    - [ ] Update any automated scripts or CI/CD pipelines
    - [ ] Communicate migration timeline and impact

---

## 📋 **Technical Requirements Checklist**

### **Architecture & Design**
- [x] **Modular Design**: Experiment registry provides clean separation of concerns
- [x] **Data Model Requirement**: Pydantic V2 models for runtime data validation (experiment metadata)
- [x] **Configuration Method**: YAML-Driven configuration with Hydra integration
- [x] **State Management Strategy**: Thread-safe experiment registry with JSON persistence

### **Integration Points**
- [x] **Integration with Hydra**: Timestamp-based directory structure
- [x] **Integration with PyTorch Lightning**: UniqueModelCheckpoint callback updates
- [x] **Integration with Streamlit UI**: Checkpoint catalog service updates
- [x] **Use of Existing Utility/Library**: Experiment registry integration

### **Quality Assurance**
- [ ] **Unit Test Coverage Goal**: > 90% for new components
- [ ] **Integration Test Requirement**: Full pipeline testing with new structure
- [ ] **Performance Test Requirement**: Checkpoint discovery time ≤ current baseline
- [ ] **UI/UX Test Requirement**: Streamlit interface works with new structure

---

## 🎯 **Success Criteria Validation**

### **Functional Requirements**
- [x] **New training runs use timestamp-based structure**: Hydra config generates proper directories
- [ ] **All existing checkpoints remain discoverable**: Backward compatibility maintained
- [ ] **Checkpoint catalog loads without errors**: No more "encoder 'None'" errors
- [ ] **Streamlit UI works with both old and new structures**: Compatibility layer functional
- [ ] **Multi-run experiments work correctly**: Sweep directory structure supports parallel jobs

### **Technical Requirements**
- [x] **Code Quality Standard is Met**: Fully documented, type-hinted, linted code
- [x] **Maintainability Goal is Met**: Clear separation of concerns, modular design
- [ ] **Resource Usage is Within Limits**: No significant memory or disk overhead
- [ ] **Compatibility with existing checkpoints confirmed**: Legacy discovery works

---

## 📊 **Risk Mitigation & Fallbacks**

### **Current Risk Level**: MEDIUM
### **Active Mitigation Strategies**:
1. **Incremental Development**: Phase-by-phase implementation with testing at each step
2. **Comprehensive Testing**: Unit tests, integration tests, and validation scripts
3. **Regular Code Quality Checks**: Linting, type checking, and peer review
4. **Backward Compatibility Layer**: Ensures existing functionality continues to work

### **Fallback Options**:
1. **Simplified Checkpoint Discovery**: If metadata system fails, fall back to directory parsing
2. **CPU-only Mode**: If performance issues arise, optimize I/O patterns
3. **Phased Rollout**: Deploy changes incrementally, rollback individual components if needed

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

**TASK:** Modify UniqueModelCheckpoint callback to use timestamp-based paths

**OBJECTIVE:** Update the checkpoint callback to work with the new directory structure and remove experiment name dependencies

**APPROACH:**
1. Examine current UniqueModelCheckpoint implementation in `ocr/callbacks/unique_checkpoint.py`
2. Update `_setup_dirpath()` method to use Hydra runtime output directory instead of experiment names
3. Remove experiment name parsing logic from checkpoint naming
4. Ensure checkpoint naming remains informative with timestamps and model info
5. Add metadata file generation alongside checkpoints
6. Test checkpoint saving with new structure

**SUCCESS CRITERIA:**
- [ ] Checkpoint callback uses timestamp-based directory structure
- [ ] No experiment name dependencies remain in checkpoint logic
- [ ] Metadata files are generated alongside checkpoints
- [ ] Training pipeline works with new checkpoint system
- [ ] Backward compatibility maintained for existing checkpoint loading</content>
<parameter name="filePath">/home/vscode/workspace/upstageailab-ocr-recsys-competition-ocr-2/MIGRATION_PLAN.md
