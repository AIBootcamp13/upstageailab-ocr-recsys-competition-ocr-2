# Preprocessing Viewer Refactor & Performance Optimization Plan

**Date**: 2025-10-18
**Status**: 🔴 CRITICAL - App Unusable, Requires Complete Redesign
**Type**: Architecture Redesign

---

## Executive Summary

The Streamlit Preprocessing Viewer has **three critical issues** that make it completely unusable:

1. **Resource Leak**: Constant loading, unresponsive behavior, memory issues
2. **Architecture Problem**: Monolithic 600+ line files that are impossible to debug
3. **Quality Problem**: Preprocessing output blurs text, making it illegible to humans and AI

This document provides a **complete redesign strategy** with modular architecture, performance optimization, and quality fixes.

---

## Problem Analysis

### 1. Resource Leak Issues

**Symptoms**:
- App constantly shows spinner/loading state
- Freezes when selecting specific dropdown options (e.g., "noise_eliminated" in Right Image dropdown)
- Only works with all preprocessing disabled + color inversion only
- High CPU usage even when idle

**Root Causes**:
```python
# PROBLEM 1: Image copying everywhere
results["original"] = image.copy()  # Line 100
current_image = image.copy()        # Line 101
roi_image = current_image[y : y + h, x : x + w].copy()  # Line 107
results["grayscale"] = current_image.copy()  # Line 121
results["color_inverted"] = current_image.copy()  # Line 125
# ... 20+ more .copy() calls

# PROBLEM 2: Unnecessary reinitialization
def __init__(self):
    self.corner_detector = AdvancedCornerDetector()  # Heavy init every time
    self.noise_eliminator = AdvancedNoiseEliminator()
    self.brightness_adjuster = IntelligentBrightnessAdjuster()
    # ... 8 more heavy objects

# PROBLEM 3: No caching
def process_with_intermediates(self, image, config):
    # Reprocesses entire pipeline on every dropdown change
    # No caching of intermediate results
```

### 2. Architecture Problems

**Current Structure** (Monolithic):
```
ui/preprocessing_viewer_app.py (213 lines)
├── Imports 6 modules inline
├── Initializes 5 heavy objects on every rerun
├── No separation of concerns
└── Impossible to debug

ui/preprocessing_viewer/pipeline.py (411 lines)
├── God object with 8 preprocessing components
├── 10+ nested conditionals
├── Telemetry mixed with business logic
└── No testability

ocr/datasets/preprocessing/document_flattening.py (700+ lines)
├── 4 warping algorithms in one class
├── RBF, surface estimation, quality assessment all coupled
├── Impossible to isolate issues
└── Cannot unit test individual components
```

**Problems**:
- Cannot debug which component causes freeze
- Cannot profile individual preprocessing stages
- Cannot swap out implementations
- Cannot test in isolation
- Cannot reuse components

### 3. Quality Problems

**Issue**: Preprocessed images blur text, making them illegible

**Root Cause**:
```python
# Aggressive noise elimination
noise_result = self.noise_eliminator.eliminate_noise(current_image)
# Uses morphological operations that destroy fine text details

# Excessive smoothing in RBF warping
rbf_x = Rbf(..., function="thin_plate", smooth=0.1)
# smooth=0.1 is too aggressive for text preservation

# Brightness adjustment overshoots
brightness_result = self.brightness_adjuster.adjust_brightness(current_image)
# May clip text contrast

# Enhancement over-sharpening
enhanced, _ = self.image_enhancer.enhance(current_image, method)
# Sharpening artifacts destroy text
```

---

## Redesign Strategy

### Phase 1: Immediate Fixes (Emergency Triage)

#### 1.1 Disable Broken Features
```python
# ui/preprocessing_viewer/preset_manager.py
DEFAULT_CONFIG = {
    'enable_document_flattening': False,  # Too slow, quality issues
    'enable_noise_elimination': False,    # Blurs text
    'enable_brightness_adjustment': False, # Clips contrast
    'enable_enhancement': False,          # Over-sharpens
    'enable_orientation_correction': False, # Rarely needed, expensive

    # Keep only safe features:
    'enable_document_detection': True,
    'enable_perspective_correction': True,
    'enable_color_preprocessing': True,
}
```

#### 1.2 Add Session State Caching
```python
# ui/preprocessing_viewer_app.py
@st.cache_data
def process_image_cached(image_bytes: bytes, config_json: str):
    """Cache processed results by image hash + config."""
    image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    config = json.loads(config_json)
    return pipeline.process_with_intermediates(image, config)

# Usage:
if uploaded_file:
    image_bytes = uploaded_file.getvalue()
    config_json = json.dumps(st.session_state.viewer_config, sort_keys=True)
    results = process_image_cached(image_bytes, config_json)
```

#### 1.3 Lazy Initialize Components
```python
# ui/preprocessing_viewer/pipeline.py
class PreprocessingViewerPipeline:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        # Don't initialize components here!
        self._corner_detector = None
        self._noise_eliminator = None
        # ...

    @property
    def corner_detector(self):
        if self._corner_detector is None:
            self._corner_detector = AdvancedCornerDetector()
        return self._corner_detector
```

---

### Phase 2: Modular Architecture (Complete Redesign)

#### 2.1 New Directory Structure
```
ocr/datasets/preprocessing/
├── __init__.py
├── core/
│   ├── __init__.py
│   ├── base.py              # Abstract base classes
│   ├── config.py            # Pydantic configs
│   └── models.py            # Data models (Result, Transform, etc.)
│
├── detection/
│   ├── __init__.py
│   ├── corner_detector.py   # AdvancedCornerDetector
│   └── document_detector.py # DocumentDetector
│
├── correction/
│   ├── __init__.py
│   ├── perspective.py       # PerspectiveCorrector
│   └── orientation.py       # OrientationCorrector
│
├── enhancement/
│   ├── __init__.py
│   ├── noise.py             # AdvancedNoiseEliminator
│   ├── brightness.py        # IntelligentBrightnessAdjuster
│   └── sharpening.py        # ImageEnhancer
│
├── flattening/
│   ├── __init__.py
│   ├── config.py            # FlatteningConfig
│   ├── surface_estimation.py
│   ├── warping_strategies/
│   │   ├── __init__.py
│   │   ├── base.py          # WarpingStrategy ABC
│   │   ├── thin_plate.py    # ThinPlateSplineWarper
│   │   ├── cylindrical.py   # CylindricalWarper
│   │   ├── spherical.py     # SphericalWarper
│   │   └── adaptive.py      # AdaptiveWarper
│   ├── rbf_utils.py
│   ├── quality_assessment.py
│   └── pipeline.py          # DocumentFlattener
│
└── pipeline/
    ├── __init__.py
    ├── stage.py             # PreprocessingStage interface
    ├── orchestrator.py      # Pipeline orchestration
    └── telemetry.py         # Separate telemetry concerns
```

#### 2.2 Base Classes
```python
# ocr/datasets/preprocessing/core/base.py
from abc import ABC, abstractmethod
from typing import Any, Generic, TypeVar
import numpy as np
from pydantic import BaseModel

InputT = TypeVar('InputT')
OutputT = TypeVar('OutputT')

class PreprocessingStage(ABC, Generic[InputT, OutputT]):
    """Base class for all preprocessing stages."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique stage name for caching/telemetry."""
        pass

    @abstractmethod
    def process(self, input_data: InputT, config: dict[str, Any]) -> OutputT:
        """Process input and return result."""
        pass

    def should_skip(self, input_data: InputT, config: dict[str, Any]) -> bool:
        """Override to skip stage based on config or input quality."""
        return False
```

#### 2.3 Warping Strategy Pattern
```python
# ocr/datasets/preprocessing/flattening/warping_strategies/base.py
from abc import ABC, abstractmethod
import numpy as np
from ..models import SurfaceNormals, WarpingTransform

class WarpingStrategy(ABC):
    """Abstract base for document warping algorithms."""

    @abstractmethod
    def warp(
        self,
        image: np.ndarray,
        surface_normals: SurfaceNormals,
        corners: np.ndarray | None = None
    ) -> tuple[np.ndarray, WarpingTransform]:
        """Apply warping transformation."""
        pass

    @abstractmethod
    def estimate_cost(self, surface_normals: SurfaceNormals) -> float:
        """Estimate computational cost (for adaptive selection)."""
        pass
```

```python
# ocr/datasets/preprocessing/flattening/warping_strategies/thin_plate.py
from .base import WarpingStrategy
from ..rbf_utils import RBFWarper

class ThinPlateSplineWarper(WarpingStrategy):
    def __init__(self, smoothing_factor: float, edge_preservation: float):
        self.rbf_warper = RBFWarper(smoothing_factor)
        self.edge_preservation = edge_preservation

    def warp(self, image, surface_normals, corners=None):
        # Isolated implementation - easy to test!
        ...

    def estimate_cost(self, surface_normals):
        # O(N*M) where N=grid_size^2, M=downsampled_pixels
        return surface_normals.grid_points.size * 640_000  # 800x800 max
```

#### 2.4 Lightweight Pipeline Orchestrator
```python
# ocr/datasets/preprocessing/pipeline/orchestrator.py
from typing import Any
import numpy as np
from ..core.base import PreprocessingStage

class PreprocessingOrchestrator:
    """Lightweight pipeline coordinator."""

    def __init__(self):
        self.stages: dict[str, PreprocessingStage] = {}
        self._cache: dict[str, Any] = {}

    def register_stage(self, stage: PreprocessingStage):
        """Register a preprocessing stage."""
        self.stages[stage.name] = stage

    def process(self, image: np.ndarray, config: dict) -> dict[str, np.ndarray]:
        """Execute enabled stages and cache intermediate results."""
        results = {"original": image}
        current_image = image

        for stage_name, stage in self.stages.items():
            if config.get(f"enable_{stage_name}", False):
                if stage.should_skip(current_image, config):
                    continue

                cache_key = f"{stage_name}_{hash(current_image.tobytes())}"
                if cache_key in self._cache:
                    result = self._cache[cache_key]
                else:
                    result = stage.process(current_image, config)
                    self._cache[cache_key] = result

                current_image = result
                results[stage_name] = result

        results["final"] = current_image
        return results
```

---

### Phase 3: Streamlit App Redesign

#### 3.1 New App Architecture
```python
# ui/preprocessing_viewer_app.py (< 150 lines)
import streamlit as st
from ui.preprocessing_viewer.components import (
    ImageUploader,
    ConfigPanel,
    ResultsViewer,
)
from ui.preprocessing_viewer.state_manager import StateManager

def main():
    st.set_page_config(page_title="OCR Preprocessing Viewer", layout="wide")

    # Initialize state manager (handles all session state logic)
    state = StateManager()

    # Sidebar
    with st.sidebar:
        uploader = ImageUploader()
        image = uploader.render()

        config_panel = ConfigPanel(state)
        config = config_panel.render()

    # Main area
    if image is None:
        st.info("Upload an image to begin")
        return

    results_viewer = ResultsViewer(state)
    results_viewer.render(image, config)

if __name__ == "__main__":
    main()
```

#### 3.2 State Manager (Prevent Rerun Loops)
```python
# ui/preprocessing_viewer/state_manager.py
import streamlit as st
from typing import Any
import hashlib
import json

class StateManager:
    """Centralized session state management to prevent rerun loops."""

    def __init__(self):
        self._init_state()

    def _init_state(self):
        if 'config' not in st.session_state:
            st.session_state.config = self.get_default_config()
        if 'results_cache' not in st.session_state:
            st.session_state.results_cache = {}
        if 'last_image_hash' not in st.session_state:
            st.session_state.last_image_hash = None

    def get_config(self) -> dict[str, Any]:
        return st.session_state.config

    def update_config(self, key: str, value: Any):
        """Update single config value (prevents dict comparison issues)."""
        if st.session_state.config.get(key) != value:
            st.session_state.config[key] = value
            # Invalidate cache on config change
            st.session_state.results_cache.clear()

    def get_cached_results(self, image_hash: str, config_hash: str):
        """Retrieve cached processing results."""
        cache_key = f"{image_hash}_{config_hash}"
        return st.session_state.results_cache.get(cache_key)

    def cache_results(self, image_hash: str, config_hash: str, results: dict):
        """Store processing results."""
        cache_key = f"{image_hash}_{config_hash}"
        st.session_state.results_cache[cache_key] = results

    @staticmethod
    def compute_hash(data: Any) -> str:
        """Compute stable hash for caching."""
        if isinstance(data, dict):
            data_str = json.dumps(data, sort_keys=True)
        else:
            data_str = str(data)
        return hashlib.md5(data_str.encode()).hexdigest()
```

#### 3.3 Lazy Results Viewer
```python
# ui/preprocessing_viewer/components/results_viewer.py
import streamlit as st
import numpy as np
from ..state_manager import StateManager
from ..pipeline_wrapper import process_image

class ResultsViewer:
    def __init__(self, state: StateManager):
        self.state = state

    def render(self, image: np.ndarray, config: dict):
        # Compute hashes for caching
        image_hash = self.state.compute_hash(image.tobytes())
        config_hash = self.state.compute_hash(config)

        # Check cache first
        results = self.state.get_cached_results(image_hash, config_hash)

        if results is None:
            with st.spinner("Processing image..."):
                results = process_image(image, config)
                self.state.cache_results(image_hash, config_hash, results)

        # Display results (no reprocessing!)
        self._render_tabs(results, config)

    def _render_tabs(self, results: dict, config: dict):
        tab1, tab2 = st.tabs(["Side-by-Side", "Individual Stages"])

        with tab1:
            self._render_comparison(results)

        with tab2:
            self._render_stages(results)
```

---

### Phase 4: Quality Fixes

#### 4.1 Text-Preserving Noise Elimination
```python
# ocr/datasets/preprocessing/enhancement/noise.py
class TextPreservingNoiseEliminator:
    """Noise elimination that preserves fine text details."""

    def eliminate_noise(self, image: np.ndarray, preserve_text: bool = True) -> np.ndarray:
        if not preserve_text:
            return self._aggressive_denoise(image)

        # Use bilateral filter (edge-preserving) instead of morphological ops
        denoised = cv2.bilateralFilter(image, d=9, sigmaColor=75, sigmaSpace=75)

        # Detect text regions
        text_mask = self._detect_text_regions(image)

        # Blend: Keep original in text regions, use denoised elsewhere
        result = np.where(text_mask[:, :, np.newaxis], image, denoised)

        return result

    def _detect_text_regions(self, image: np.ndarray) -> np.ndarray:
        """Detect regions likely containing text."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # High-frequency content indicates text
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        text_score = np.abs(laplacian)

        # Threshold to get text mask
        _, text_mask = cv2.threshold(text_score, text_score.mean() + text_score.std(), 255, cv2.THRESH_BINARY)

        # Dilate slightly to include text edges
        kernel = np.ones((3, 3), np.uint8)
        text_mask = cv2.dilate(text_mask.astype(np.uint8), kernel, iterations=2)

        return text_mask.astype(bool)
```

#### 4.2 Reduce RBF Smoothing for Text
```python
# ocr/datasets/preprocessing/flattening/rbf_utils.py
class RBFWarper:
    def __init__(self, smoothing_factor: float, text_mode: bool = True):
        # Reduce smoothing for text preservation
        self.smoothing_factor = smoothing_factor * 0.3 if text_mode else smoothing_factor
```

#### 4.3 Gentle Brightness Adjustment
```python
# ocr/datasets/preprocessing/enhancement/brightness.py
class TextFriendlyBrightnessAdjuster:
    def adjust_brightness(self, image: np.ndarray) -> np.ndarray:
        # Use CLAHE with clip limit to avoid contrast explosion
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        # Gentle CLAHE (clip_limit=2.0 instead of 4.0+)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)

        lab = cv2.merge([l, a, b])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
```

---

## Implementation Plan

### Week 1: Emergency Triage
- [ ] **Day 1-2**: Implement Phase 1 fixes (disable broken features, add caching)
- [ ] **Day 3-4**: Test caching, verify no more freezes
- [ ] **Day 5**: Deploy emergency fix, document known limitations

### Week 2-3: Modular Refactor
- [ ] **Week 2**: Implement Phase 2 (modular architecture)
  - Day 1-2: Create base classes and directory structure
  - Day 3-4: Refactor flattening module with strategy pattern
  - Day 5: Unit tests for isolated components
- [ ] **Week 3**: Complete refactor
  - Day 1-2: Refactor detection and correction modules
  - Day 3-4: Refactor enhancement modules
  - Day 5: Integration tests

### Week 4: Streamlit Redesign
- [ ] **Day 1-2**: Implement Phase 3 (new Streamlit architecture)
- [ ] **Day 3-4**: Implement Phase 4 (quality fixes)
- [ ] **Day 5**: End-to-end testing, performance profiling

### Week 5: Polish & Deploy
- [ ] **Day 1-2**: Performance optimization (profiling, bottleneck removal)
- [ ] **Day 3**: Documentation updates
- [ ] **Day 4**: User acceptance testing
- [ ] **Day 5**: Production deployment

---

## Success Metrics

### Performance
- [ ] App loads in <2 seconds
- [ ] Pipeline processes 2000×1500 image in <3 seconds
- [ ] No freezes when changing dropdown selections
- [ ] Memory usage <500MB per user session

### Quality
- [ ] Text remains legible after preprocessing (human verification)
- [ ] OCR accuracy maintained or improved (quantitative test)
- [ ] No blurring artifacts in text regions
- [ ] Edge preservation in document boundaries

### Maintainability
- [ ] No single file >300 lines
- [ ] 80%+ unit test coverage for preprocessing modules
- [ ] Can swap warping strategies without modifying orchestrator
- [ ] Can debug individual stages in isolation

---

## Risk Mitigation

### Risk 1: Breaking Changes
**Mitigation**: Keep old `document_flattening.py` as `document_flattening_legacy.py`, provide adapter pattern

### Risk 2: Performance Regression
**Mitigation**: Establish performance baselines before refactor, run benchmarks after each module

### Risk 3: Quality Regression
**Mitigation**: Visual regression tests with golden images, OCR accuracy tests

---

## Alternative Approaches

### Option A: Full Rewrite (Recommended)
- **Pros**: Clean architecture, no technical debt
- **Cons**: 4-5 weeks development time
- **When**: Production deployment timeline >1 month

### Option B: Incremental Refactor
- **Pros**: Can ship improvements weekly
- **Cons**: May carry over some technical debt
- **When**: Need improvements shipped ASAP

### Option C: Replace with External Library
- **Pros**: Battle-tested, maintained by community
- **Cons**: Less control, may not fit requirements
- **Libraries**: `doctr`, `deskew`, `pyocr` preprocessing
- **When**: Time to market is critical

---

## Next Steps

1. **Decide on approach** (A, B, or C)
2. **Get stakeholder buy-in** on timeline
3. **Set up performance benchmarks** (current baseline)
4. **Create feature branch** for refactor
5. **Begin Phase 1 implementation** immediately

---

## Appendix: Current vs. Proposed Complexity

### Current Complexity
```
Cyclomatic Complexity:
- document_flattening.py: 145 (VERY HIGH)
- pipeline.py: 68 (HIGH)
- preprocessing_viewer_app.py: 32 (MODERATE)

Lines of Code:
- document_flattening.py: 700+ lines
- pipeline.py: 411 lines
- preprocessing_viewer_app.py: 213 lines

Coupling:
- DocumentFlattener depends on 8 classes (TIGHT)
- PreprocessingViewerPipeline depends on 10 classes (TIGHT)
```

### Proposed Complexity
```
Cyclomatic Complexity:
- Each warping strategy: <20 (LOW)
- Pipeline orchestrator: <15 (LOW)
- Streamlit app: <10 (VERY LOW)

Lines of Code:
- Largest file: <300 lines
- Average file: ~150 lines

Coupling:
- Each module depends on 2-3 interfaces (LOOSE)
- Orchestrator depends only on base classes (LOOSE)
```

---

**Conclusion**: The current app requires a **complete architectural redesign** to be usable. The modular refactoring plan provides a clear path forward with measurable success criteria and risk mitigation strategies.
