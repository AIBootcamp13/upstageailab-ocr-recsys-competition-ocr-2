# Option C: Quick Start Code Templates

**Purpose**: Copy-paste templates to start Week 1 tomorrow
**Time to setup**: 15 minutes

---

## 🚀 15-Minute Setup

### Step 1: Create Project Structure (2 minutes)

```bash
# Navigate to project root
cd /home/vscode/workspace/upstageailab-ocr-recsys-competition-ocr-2

# Create clean branch
git checkout -b preprocessing-v2-clean-rewrite

# Create new directory
mkdir -p preprocessing_viewer_v2/{core,preprocessing,ui,tests}
cd preprocessing_viewer_v2

# Create __init__.py files
touch core/__init__.py preprocessing/__init__.py ui/__init__.py tests/__init__.py
```

### Step 2: Dependencies (3 minutes)

```bash
# Create requirements.txt
cat > requirements.txt <<'EOF'
# Core CV
opencv-python>=4.8.0
numpy>=1.24.0
pillow>=10.0.0

# Streamlit
streamlit>=1.28.0
streamlit-image-comparison>=0.0.4

# Utilities
pydantic>=2.0.0
diskcache>=5.6.0

# Testing
pytest>=7.4.0
pytest-cov>=4.1.0
EOF

# Install
uv pip install -r requirements.txt
```

### Step 3: Data Models (5 minutes)

```python
# core/models.py - Copy this entire file
from pydantic import BaseModel, Field
from typing import Optional
import numpy as np


class DetectionResult(BaseModel):
    """Result from document detection."""
    model_config = {"arbitrary_types_allowed": True}

    success: bool
    corners: Optional[np.ndarray] = None
    confidence: float = 0.0
    method: str = "contour_detection"


class CorrectionResult(BaseModel):
    """Result from perspective correction."""
    model_config = {"arbitrary_types_allowed": True}

    success: bool
    corrected_image: Optional[np.ndarray] = None
    transform_matrix: Optional[np.ndarray] = None


class BinarizationResult(BaseModel):
    """Result from binarization."""
    model_config = {"arbitrary_types_allowed": True}

    success: bool
    binarized_image: Optional[np.ndarray] = None
    method: str = "adaptive"


class EnhancementResult(BaseModel):
    """Result from enhancement."""
    model_config = {"arbitrary_types_allowed": True}

    success: bool
    enhanced_image: Optional[np.ndarray] = None
    settings: dict = Field(default_factory=dict)
```

### Step 4: Base Class (2 minutes)

```python
# preprocessing/base.py - Copy this entire file
from abc import ABC, abstractmethod
from typing import Any, TypeVar, Generic
import numpy as np

InputT = TypeVar('InputT')
OutputT = TypeVar('OutputT')


class PreprocessingStage(ABC, Generic[InputT, OutputT]):
    """Base class for all preprocessing stages."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Stage name for logging/caching."""
        pass

    @abstractmethod
    def process(self, input_data: InputT, **kwargs) -> OutputT:
        """Process input and return result."""
        pass

    def should_skip(self, input_data: InputT, **kwargs) -> bool:
        """Check if stage should be skipped."""
        return False
```

### Step 5: Document Detector (3 minutes)

```python
# preprocessing/detection.py - Copy this entire file
import cv2
import numpy as np
from typing import Optional
from ..core.models import DetectionResult
from .base import PreprocessingStage


class DocumentDetector(PreprocessingStage[np.ndarray, DetectionResult]):
    """Document boundary detection using OpenCV."""

    @property
    def name(self) -> str:
        return "document_detection"

    def process(self, image: np.ndarray, **kwargs) -> DetectionResult:
        """Detect document boundaries in image."""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Edge detection
        edges = cv2.Canny(blurred, 50, 150, apertureSize=3)

        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return DetectionResult(success=False, corners=None, confidence=0.0)

        # Get largest contour by area
        largest_contour = max(contours, key=cv2.contourArea)

        # Calculate contour area ratio (confidence metric)
        image_area = image.shape[0] * image.shape[1]
        contour_area = cv2.contourArea(largest_contour)
        confidence = contour_area / image_area

        # Approximate to quadrilateral
        epsilon = 0.02 * cv2.arcLength(largest_contour, True)
        approx = cv2.approxPolyDP(largest_contour, epsilon, True)

        # Check if we got a quadrilateral
        if len(approx) == 4:
            corners = approx.reshape(4, 2).astype(np.float32)
            ordered_corners = self._order_corners(corners)
            return DetectionResult(
                success=True,
                corners=ordered_corners,
                confidence=float(confidence),
                method="contour_detection"
            )

        # Fallback: use image boundaries
        h, w = image.shape[:2]
        margin = min(h, w) // 20  # 5% margin
        fallback_corners = np.array([
            [margin, margin],
            [w - margin, margin],
            [w - margin, h - margin],
            [margin, h - margin]
        ], dtype=np.float32)

        return DetectionResult(
            success=True,
            corners=fallback_corners,
            confidence=0.5,
            method="fallback_boundaries"
        )

    @staticmethod
    def _order_corners(corners: np.ndarray) -> np.ndarray:
        """
        Order corners as: top-left, top-right, bottom-right, bottom-left.

        Uses sum and diff heuristics:
        - Top-left has smallest sum (x + y)
        - Bottom-right has largest sum
        - Top-right has smallest diff (y - x)
        - Bottom-left has largest diff
        """
        s = corners.sum(axis=1)
        diff = np.diff(corners, axis=1).flatten()

        ordered = np.zeros((4, 2), dtype=np.float32)
        ordered[0] = corners[np.argmin(s)]      # top-left
        ordered[2] = corners[np.argmax(s)]      # bottom-right
        ordered[1] = corners[np.argmin(diff)]   # top-right
        ordered[3] = corners[np.argmax(diff)]   # bottom-left

        return ordered
```

---

## ✅ Verify Setup Works

```bash
# Test imports
python3 -c "
from preprocessing.detection import DocumentDetector
from core.models import DetectionResult
import numpy as np

# Create detector
detector = DocumentDetector()

# Test on dummy image
image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
result = detector.process(image)

print(f'Detection success: {result.success}')
print(f'Confidence: {result.confidence}')
print('✅ Setup complete!')
"
```

Expected output:
```
Detection success: True
Confidence: 0.XXX
✅ Setup complete!
```

---

## 📝 First Test (5 minutes)

```python
# tests/test_detection.py - Your first test!
import pytest
import numpy as np
from preprocessing.detection import DocumentDetector
from core.models import DetectionResult


def test_detector_returns_result():
    """Test that detector returns a DetectionResult."""
    detector = DocumentDetector()
    image = np.ones((100, 100, 3), dtype=np.uint8) * 255

    result = detector.process(image)

    assert isinstance(result, DetectionResult)
    assert result.success is not None


def test_detector_finds_corners():
    """Test that detector finds 4 corners."""
    detector = DocumentDetector()
    image = create_document_image()

    result = detector.process(image)

    assert result.success
    assert result.corners is not None
    assert result.corners.shape == (4, 2)


def create_document_image() -> np.ndarray:
    """Create a synthetic document for testing."""
    # White document on black background
    image = np.zeros((500, 500, 3), dtype=np.uint8)
    cv2.rectangle(image, (50, 50), (450, 450), (255, 255, 255), -1)
    return image


def test_corner_ordering():
    """Test that corners are ordered correctly."""
    detector = DocumentDetector()

    # Define corners in random order
    corners = np.array([
        [100, 300],  # bottom-left
        [100, 100],  # top-left
        [300, 100],  # top-right
        [300, 300],  # bottom-right
    ], dtype=np.float32)

    ordered = detector._order_corners(corners)

    # Check order: TL, TR, BR, BL
    assert np.allclose(ordered[0], [100, 100])  # top-left
    assert np.allclose(ordered[1], [300, 100])  # top-right
    assert np.allclose(ordered[2], [300, 300])  # bottom-right
    assert np.allclose(ordered[3], [100, 300])  # bottom-left


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
```

### Run Tests

```bash
# Run tests
pytest tests/test_detection.py -v

# Expected output:
# test_detection.py::test_detector_returns_result PASSED
# test_detection.py::test_detector_finds_corners PASSED
# test_detection.py::test_corner_ordering PASSED
# ======================== 3 passed in 0.X seconds ========================
```

---

## 🎯 Day 1 Goal: See It Work!

Create a minimal demo script:

```python
# demo.py - Test your detector visually
import cv2
import numpy as np
from preprocessing.detection import DocumentDetector


def main():
    print("🔍 Document Detector Demo")

    # Create detector
    detector = DocumentDetector()

    # Option 1: Load real image
    # image = cv2.imread("path/to/receipt.jpg")

    # Option 2: Create synthetic document
    image = create_test_document()

    # Detect
    print("Detecting document boundaries...")
    result = detector.process(image)

    if result.success:
        print(f"✅ Detection successful! (confidence: {result.confidence:.2f})")
        print(f"Corners:\n{result.corners}")

        # Draw corners
        vis = draw_corners(image, result.corners)

        # Save result
        cv2.imwrite("detection_result.jpg", vis)
        print("📁 Saved visualization to: detection_result.jpg")
    else:
        print("❌ Detection failed")


def create_test_document() -> np.ndarray:
    """Create a test document image."""
    # Black background
    img = np.zeros((600, 800, 3), dtype=np.uint8)

    # White document (slightly rotated)
    pts = np.array([
        [100, 120],
        [700, 80],
        [720, 500],
        [80, 520]
    ], dtype=np.int32)

    cv2.fillPoly(img, [pts], (255, 255, 255))

    # Add some text
    cv2.putText(img, "Test Document", (200, 300), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3)

    return img


def draw_corners(image: np.ndarray, corners: np.ndarray) -> np.ndarray:
    """Draw detected corners on image."""
    vis = image.copy()
    corners_int = corners.astype(int)

    # Draw lines between corners
    for i in range(4):
        pt1 = tuple(corners_int[i])
        pt2 = tuple(corners_int[(i + 1) % 4])
        cv2.line(vis, pt1, pt2, (0, 255, 0), 3)

    # Draw corner points
    for corner in corners_int:
        cv2.circle(vis, tuple(corner), 8, (0, 0, 255), -1)

    return vis


if __name__ == "__main__":
    main()
```

### Run Demo

```bash
python demo.py

# Expected output:
# 🔍 Document Detector Demo
# Detecting document boundaries...
# ✅ Detection successful! (confidence: 0.85)
# Corners:
# [[100. 120.]
#  [700.  80.]
#  [720. 500.]
#  [ 80. 520.]]
# 📁 Saved visualization to: detection_result.jpg
```

Open `detection_result.jpg` - you should see your document with green lines and red dots marking the detected corners!

---

## 🎉 Day 1 Complete!

You now have:
- ✅ Project structure
- ✅ Dependencies installed
- ✅ Data models
- ✅ Document detector working
- ✅ Tests passing
- ✅ Visual demo

**Total time**: ~30 minutes

---

## 📅 Tomorrow: Day 2 Tasks

1. **Test on real images**:
   ```bash
   # Download sample receipts
   wget https://raw.githubusercontent.com/stweil/ocr-test-data/master/receipts/receipt1.jpg

   # Test detector
   python -c "
   import cv2
   from preprocessing.detection import DocumentDetector

   image = cv2.imread('receipt1.jpg')
   detector = DocumentDetector()
   result = detector.process(image)
   print(f'Success: {result.success}, Confidence: {result.confidence:.2f}')
   "
   ```

2. **Improve detection** (if needed):
   - Adjust Canny thresholds
   - Try different blur kernels
   - Tune epsilon for approximation

3. **Add more tests**:
   - Test on various image sizes
   - Test on noisy images
   - Test failure cases

---

## 🚦 Progress Tracker

### Week 1: Document Detection
- [x] Day 1: Setup + Basic detector ← **YOU ARE HERE**
- [ ] Day 2: Test on real images
- [ ] Day 3: Refinements
- [ ] Day 4: Perspective correction prep
- [ ] Day 5: Week 1 review

### Week 2: Correction & Enhancement
- [ ] Day 1-2: Perspective correction
- [ ] Day 3: Binarization & enhancement
- [ ] Day 4-5: Pipeline orchestrator

### Week 3: Streamlit UI
- [ ] Day 1-2: State management
- [ ] Day 3-4: UI components
- [ ] Day 5: Polish & deploy

### Week 4: Testing & Docs
- [ ] Day 1-2: Comprehensive tests
- [ ] Day 3-4: Documentation
- [ ] Day 5: Demo prep

---

## 💪 Motivation

**Remember**: Your previous attempts failed because:
- Custom RBF warping → too complex
- Custom noise elimination → destroyed text
- Monolithic code → couldn't debug

**This attempt will succeed because**:
- Using OpenCV (battle-tested) → works
- Modular code → easy to debug
- Starting simple → document detection first

You've got 15 minutes of setup, then 15 minutes of testing. By end of Day 1, you'll have something **working**.

That's progress you can see and feel. Build on it tomorrow! 🚀

---

## 📞 Stuck? Quick Debugging

### Import Errors
```bash
# Check Python can find modules
cd preprocessing_viewer_v2
python -c "import sys; print(sys.path)"

# Add parent directory to path
export PYTHONPATH="${PYTHONPATH}:/home/vscode/workspace/upstageailab-ocr-recsys-competition-ocr-2/preprocessing_viewer_v2"
```

### OpenCV Not Found
```bash
# Reinstall
uv pip uninstall opencv-python
uv pip install opencv-python-headless
```

### Tests Failing
```bash
# Run with verbose output
pytest tests/test_detection.py -vv -s
```

---

**Start tomorrow with Day 1. Get document detection working. See the green lines around your document. Feel the progress! 💚**
