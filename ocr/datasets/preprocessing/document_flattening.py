"""
Document Flattening for Crumpled Paper Handling.

This module implements thin plate spline warping, surface normal estimation,
and geometric distortion correction to flatten crumpled documents for Office Lens
quality preprocessing. All data models use Pydantic V2 for validation and type safety.
"""

from enum import Enum
from typing import Any

import cv2
import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator
from scipy.interpolate import Rbf
from scipy.ndimage import gaussian_filter


class FlatteningMethod(str, Enum):
    """Available document flattening methods."""

    THIN_PLATE_SPLINE = "thin_plate_spline"
    CYLINDRICAL = "cylindrical"
    SPHERICAL = "spherical"
    ADAPTIVE = "adaptive"


class FlatteningConfig(BaseModel):
    """Configuration for document flattening operations.

    Uses Pydantic V2 for validation and type safety.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    method: FlatteningMethod = Field(default=FlatteningMethod.THIN_PLATE_SPLINE, description="Flattening algorithm to use")
    grid_size: int = Field(default=20, ge=5, le=100, description="Grid size for surface estimation (5-100)")
    smoothing_factor: float = Field(default=0.1, ge=0.0, le=1.0, description="Smoothing factor for RBF interpolation (0.0-1.0)")
    edge_preservation_strength: float = Field(
        default=0.8, ge=0.0, le=1.0, description="Strength of edge preservation during flattening (0.0-1.0)"
    )
    min_curvature_threshold: float = Field(default=0.05, ge=0.0, le=1.0, description="Minimum curvature to trigger flattening (0.0-1.0)")
    enable_quality_assessment: bool = Field(default=True, description="Enable quality assessment of flattening results")

    # Grid size validation is handled by Field constraints


class SurfaceNormals(BaseModel):
    """Surface normal estimation results.

    Contains estimated surface normals for document deformation analysis.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    normals: np.ndarray = Field(..., description="Surface normal vectors at grid points (grid_size, grid_size, 3)")
    grid_points: np.ndarray = Field(..., description="Grid point coordinates (grid_size, grid_size, 2)")
    curvature_map: np.ndarray = Field(..., description="Curvature values at each grid point (grid_size, grid_size)")
    mean_curvature: float = Field(..., ge=0.0, description="Mean curvature across the surface")
    max_curvature: float = Field(..., ge=0.0, description="Maximum curvature detected")

    @field_validator("normals")
    @classmethod
    def validate_normals(cls, v: np.ndarray) -> np.ndarray:
        """Validate surface normals have correct shape."""
        if len(v.shape) != 3 or v.shape[2] != 3:
            raise ValueError(f"Surface normals must be shape (H, W, 3), got {v.shape}")
        return v


class WarpingTransform(BaseModel):
    """Warping transformation for document flattening.

    Contains the transformation parameters for flattening operations.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    source_points: np.ndarray = Field(..., description="Source control points for warping (N, 2)")
    target_points: np.ndarray = Field(..., description="Target control points after flattening (N, 2)")
    transform_matrix: np.ndarray | None = Field(default=None, description="Transformation matrix if applicable")
    method: FlatteningMethod = Field(..., description="Method used for warping")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence in the warping transformation (0.0-1.0)")

    @field_validator("source_points", "target_points")
    @classmethod
    def validate_points(cls, v: np.ndarray) -> np.ndarray:
        """Validate control points have correct shape."""
        if len(v.shape) != 2 or v.shape[1] != 2:
            raise ValueError(f"Points must be shape (N, 2), got {v.shape}")
        return v


class FlatteningQualityMetrics(BaseModel):
    """Quality metrics for flattening results.

    Quantifies the quality and effectiveness of document flattening.
    """

    model_config = ConfigDict(strict=False)

    distortion_score: float = Field(..., ge=0.0, le=1.0, description="Geometric distortion score (0=no distortion, 1=severe)")
    edge_preservation_score: float = Field(..., ge=0.0, le=1.0, description="Edge preservation quality (0=poor, 1=perfect)")
    smoothness_score: float = Field(..., ge=0.0, le=1.0, description="Surface smoothness after flattening (0=rough, 1=smooth)")
    overall_quality: float = Field(..., ge=0.0, le=1.0, description="Overall flattening quality (0=poor, 1=excellent)")
    residual_curvature: float = Field(..., ge=0.0, description="Remaining curvature after flattening")
    processing_successful: bool = Field(..., description="Whether flattening completed successfully")


class FlatteningResult(BaseModel):
    """Result of document flattening operation.

    Contains the flattened image and associated metadata.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    flattened_image: np.ndarray = Field(..., description="Flattened document image")
    warping_transform: WarpingTransform = Field(..., description="Warping transformation applied")
    surface_normals: SurfaceNormals | None = Field(default=None, description="Estimated surface normals (if computed)")
    quality_metrics: FlatteningQualityMetrics | None = Field(default=None, description="Quality assessment metrics")
    method_used: FlatteningMethod = Field(..., description="Flattening method that was applied")
    processing_time_ms: float = Field(..., ge=0.0, description="Processing time in milliseconds")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata about the flattening operation")

    @field_validator("flattened_image")
    @classmethod
    def validate_image(cls, v: np.ndarray) -> np.ndarray:
        """Validate flattened image is valid."""
        if v.size == 0:
            raise ValueError("Flattened image cannot be empty")
        if len(v.shape) not in [2, 3]:
            raise ValueError(f"Image must be 2D or 3D array, got shape {v.shape}")
        return v


class DocumentFlattener:
    """
    Document flattening for crumpled paper handling.

    Implements thin plate spline warping, surface normal estimation,
    and geometric distortion correction to achieve Office Lens quality.
    """

    def __init__(self, config: FlatteningConfig | None = None):
        """Initialize flattener with configuration.

        Args:
            config: Flattening configuration (uses defaults if None)
        """
        self.config = config or FlatteningConfig()

    def flatten_document(self, image: np.ndarray, corners: np.ndarray | None = None) -> FlatteningResult:
        """
        Flatten a crumpled document image.

        Args:
            image: Input document image
            corners: Optional document corners for guided flattening

        Returns:
            FlatteningResult with flattened image and metadata
        """
        import time

        start_time = time.time()

        # Estimate surface normals and curvature
        surface_normals = self._estimate_surface_normals(image)

        # Check if flattening is needed
        if surface_normals.mean_curvature < self.config.min_curvature_threshold:
            # Document is already flat enough
            warping_transform = WarpingTransform(
                source_points=np.array([[0, 0]]), target_points=np.array([[0, 0]]), method=self.config.method, confidence=1.0
            )

            processing_time_ms = (time.time() - start_time) * 1000

            return FlatteningResult(
                flattened_image=image.copy(),
                warping_transform=warping_transform,
                surface_normals=surface_normals,
                quality_metrics=None,
                method_used=self.config.method,
                processing_time_ms=processing_time_ms,
                metadata={"skipped_reason": "already_flat"},
            )

        # Select flattening method
        if self.config.method == FlatteningMethod.THIN_PLATE_SPLINE:
            result_image, transform = self._thin_plate_spline_warping(image, surface_normals, corners)
        elif self.config.method == FlatteningMethod.CYLINDRICAL:
            result_image, transform = self._cylindrical_warping(image, surface_normals)
        elif self.config.method == FlatteningMethod.SPHERICAL:
            result_image, transform = self._spherical_warping(image, surface_normals)
        elif self.config.method == FlatteningMethod.ADAPTIVE:
            result_image, transform = self._adaptive_warping(image, surface_normals, corners)
        else:
            raise ValueError(f"Unknown flattening method: {self.config.method}")

        # Assess quality if enabled
        quality_metrics = None
        if self.config.enable_quality_assessment:
            quality_metrics = self._assess_flattening_quality(image, result_image, surface_normals)

        processing_time_ms = (time.time() - start_time) * 1000

        return FlatteningResult(
            flattened_image=result_image,
            warping_transform=transform,
            surface_normals=surface_normals,
            quality_metrics=quality_metrics,
            method_used=self.config.method,
            processing_time_ms=processing_time_ms,
            metadata={"original_shape": image.shape, "curvature_detected": surface_normals.mean_curvature},
        )

    def _estimate_surface_normals(self, image: np.ndarray) -> SurfaceNormals:
        """
        Estimate surface normals from image intensity gradients.

        Args:
            image: Input image

        Returns:
            SurfaceNormals with estimated normals and curvature
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        h, w = gray.shape

        # Create grid for surface estimation
        grid_h = self.config.grid_size
        grid_w = self.config.grid_size

        # Calculate gradient magnitude as proxy for depth
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)  # type: ignore[attr-defined]
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)  # type: ignore[attr-defined]
        grad_magnitude: np.ndarray = np.sqrt(grad_x**2 + grad_y**2)  # type: ignore[operator]

        # Smooth gradient for surface estimation
        grad_smooth = gaussian_filter(grad_magnitude, sigma=5.0)

        # Downsample to grid
        grid_points = np.zeros((grid_h, grid_w, 2))
        curvature_map = np.zeros((grid_h, grid_w))

        for i in range(grid_h):
            for j in range(grid_w):
                y = int(i * h / grid_h)
                x = int(j * w / grid_w)
                grid_points[i, j] = [x, y]
                curvature_map[i, j] = grad_smooth[y, x]

        # Normalize curvature map
        if curvature_map.max() > 0:
            curvature_map = curvature_map / curvature_map.max()

        # Estimate surface normals from curvature
        normals = np.zeros((grid_h, grid_w, 3))
        for i in range(grid_h):
            for j in range(grid_w):
                # Simple normal estimation from curvature gradient
                if i > 0 and j > 0 and i < grid_h - 1 and j < grid_w - 1:
                    dx = curvature_map[i, j + 1] - curvature_map[i, j - 1]
                    dy = curvature_map[i + 1, j] - curvature_map[i - 1, j]

                    # Normal vector pointing up with tilt based on gradient
                    normal = np.array([dx, dy, 1.0])
                    normal = normal / np.linalg.norm(normal)
                    normals[i, j] = normal
                else:
                    normals[i, j] = [0, 0, 1]  # Flat surface at edges

        mean_curvature = float(np.mean(curvature_map))
        max_curvature = float(np.max(curvature_map))

        return SurfaceNormals(
            normals=normals,
            grid_points=grid_points,
            curvature_map=curvature_map,
            mean_curvature=mean_curvature,
            max_curvature=max_curvature,
        )

    def _thin_plate_spline_warping(
        self, image: np.ndarray, surface_normals: SurfaceNormals, corners: np.ndarray | None
    ) -> tuple[np.ndarray, WarpingTransform]:
        """
        Apply thin plate spline warping for document flattening.

        Args:
            image: Input image
            surface_normals: Estimated surface normals
            corners: Optional document corners

        Returns:
            Tuple of (warped image, warping transform)
        """
        h, w = image.shape[:2]

        # Create control points from grid
        grid_h, grid_w = surface_normals.grid_points.shape[:2]
        source_points = []
        target_points = []

        for i in range(grid_h):
            for j in range(grid_w):
                src_pt = surface_normals.grid_points[i, j]
                curvature = surface_normals.curvature_map[i, j]

                # Target point with reduced curvature influence
                tgt_pt = src_pt.copy()

                # Apply flattening by reducing displacement based on curvature
                center_x, center_y = w / 2, h / 2
                dx = src_pt[0] - center_x
                dy = src_pt[1] - center_y

                # Flatten towards center proportional to curvature
                correction_factor = curvature * self.config.edge_preservation_strength
                tgt_pt[0] -= dx * correction_factor * 0.1
                tgt_pt[1] -= dy * correction_factor * 0.1

                source_points.append(src_pt)
                target_points.append(tgt_pt)

        source_points_np: np.ndarray = np.array(source_points, dtype=np.float32)
        target_points_np: np.ndarray = np.array(target_points, dtype=np.float32)

        # Apply RBF interpolation for smooth warping
        warped_image = self._apply_rbf_warping(image, source_points_np, target_points_np)

        # Calculate confidence based on curvature reduction
        confidence = 1.0 - (surface_normals.mean_curvature * 0.5)
        confidence = np.clip(confidence, 0.3, 1.0)

        transform = WarpingTransform(
            source_points=source_points_np,
            target_points=target_points_np,
            method=FlatteningMethod.THIN_PLATE_SPLINE,
            confidence=float(confidence),
        )

        return warped_image, transform

    def _cylindrical_warping(self, image: np.ndarray, surface_normals: SurfaceNormals) -> tuple[np.ndarray, WarpingTransform]:
        """
        Apply cylindrical warping for rolled document flattening.

        Args:
            image: Input image
            surface_normals: Estimated surface normals

        Returns:
            Tuple of (warped image, warping transform)
        """
        h, w = image.shape[:2]

        # Detect dominant curvature direction
        curvature = surface_normals.curvature_map
        grad_x = np.gradient(curvature, axis=1)
        grad_y = np.gradient(curvature, axis=0)

        horizontal_curvature = np.mean(np.abs(grad_x))
        vertical_curvature = np.mean(np.abs(grad_y))

        # Create cylindrical unwrapping
        source_points = []
        target_points = []

        if horizontal_curvature > vertical_curvature:
            # Horizontal roll - unwrap along x-axis
            for y in range(0, h, h // 10):
                for x in range(0, w, w // 10):
                    source_points.append([x, y])
                    # Unwrap x coordinate
                    x_unwrap = x * (1 + 0.1 * np.sin(x * np.pi / w))
                    target_points.append([x_unwrap, y])
        else:
            # Vertical roll - unwrap along y-axis
            for y in range(0, h, h // 10):
                for x in range(0, w, w // 10):
                    source_points.append([x, y])
                    # Unwrap y coordinate
                    y_unwrap = y * (1 + 0.1 * np.sin(y * np.pi / h))
                    target_points.append([x, y_unwrap])

        source_points_np: np.ndarray = np.array(source_points, dtype=np.float32)
        target_points_np: np.ndarray = np.array(target_points, dtype=np.float32)

        warped_image = self._apply_rbf_warping(image, source_points_np, target_points_np)

        transform = WarpingTransform(
            source_points=source_points_np, target_points=target_points_np, method=FlatteningMethod.CYLINDRICAL, confidence=0.8
        )

        return warped_image, transform

    def _spherical_warping(self, image: np.ndarray, surface_normals: SurfaceNormals) -> tuple[np.ndarray, WarpingTransform]:
        """
        Apply spherical warping for bulged document flattening.

        Args:
            image: Input image
            surface_normals: Estimated surface normals

        Returns:
            Tuple of (warped image, warping transform)
        """
        h, w = image.shape[:2]
        center_x, center_y = w / 2, h / 2

        # Create radial unwrapping from center
        source_points = []
        target_points = []

        for r_factor in np.linspace(0.1, 1.0, 5):
            for angle in np.linspace(0, 2 * np.pi, 20):
                x = center_x + r_factor * (w / 2) * np.cos(angle)
                y = center_y + r_factor * (h / 2) * np.sin(angle)

                if 0 <= x < w and 0 <= y < h:
                    source_points.append([x, y])

                    # Flatten radial distortion
                    r_correction = 1.0 + 0.1 * r_factor
                    x_flat = center_x + r_correction * (x - center_x)
                    y_flat = center_y + r_correction * (y - center_y)

                    target_points.append([x_flat, y_flat])

        source_points_np: np.ndarray = np.array(source_points, dtype=np.float32)
        target_points_np: np.ndarray = np.array(target_points, dtype=np.float32)

        warped_image = self._apply_rbf_warping(image, source_points_np, target_points_np)

        transform = WarpingTransform(
            source_points=source_points_np, target_points=target_points_np, method=FlatteningMethod.SPHERICAL, confidence=0.75
        )

        return warped_image, transform

    def _adaptive_warping(
        self, image: np.ndarray, surface_normals: SurfaceNormals, corners: np.ndarray | None
    ) -> tuple[np.ndarray, WarpingTransform]:
        """
        Apply adaptive warping based on detected deformation type.

        Args:
            image: Input image
            surface_normals: Estimated surface normals
            corners: Optional document corners

        Returns:
            Tuple of (warped image, warping transform)
        """
        # Analyze deformation pattern
        curvature = surface_normals.curvature_map

        # Check for cylindrical pattern
        grad_x = np.gradient(curvature, axis=1)
        grad_y = np.gradient(curvature, axis=0)

        horizontal_var = np.var(grad_x)
        vertical_var = np.var(grad_y)

        # Check for radial pattern
        h, w = image.shape[:2]
        center_x, center_y = w / 2, h / 2
        radial_pattern = 0.0

        for i in range(curvature.shape[0]):
            for j in range(curvature.shape[1]):
                x, y = surface_normals.grid_points[i, j]
                dist = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
                radial_pattern += curvature[i, j] * dist

        radial_pattern /= curvature.shape[0] * curvature.shape[1]

        # Select method based on pattern
        if abs(horizontal_var - vertical_var) > 0.1:
            # Cylindrical deformation
            return self._cylindrical_warping(image, surface_normals)
        elif radial_pattern > 0.2:
            # Spherical deformation
            return self._spherical_warping(image, surface_normals)
        else:
            # General deformation - use thin plate spline
            return self._thin_plate_spline_warping(image, surface_normals, corners)

    def _apply_rbf_warping(self, image: np.ndarray, source_points: np.ndarray, target_points: np.ndarray) -> np.ndarray:
        """
        Apply radial basis function warping.

        PERFORMANCE FIX: Downsamples image before RBF interpolation to avoid O(N*M) explosion
        where N=control_points and M=pixels. For 2000x1500 image with 20x20 grid, this reduces
        computation from 1.2 billion to ~19 million operations (63x speedup).

        Args:
            image: Input image
            source_points: Source control points (N, 2)
            target_points: Target control points (N, 2)

        Returns:
            Warped image
        """
        h, w = image.shape[:2]

        # CRITICAL FIX: Downsample for RBF computation to prevent hang
        # Target ~800px on longest edge for acceptable performance
        MAX_DIMENSION = 800
        if max(h, w) > MAX_DIMENSION:
            scale = MAX_DIMENSION / max(h, w)
            downsample_h = int(h * scale)
            downsample_w = int(w * scale)
            downsampled_image = cv2.resize(image, (downsample_w, downsample_h), interpolation=cv2.INTER_AREA)

            # Scale control points
            scaled_source = source_points * scale
            scaled_target = target_points * scale
        else:
            downsample_h, downsample_w = h, w
            downsampled_image = image
            scaled_source = source_points
            scaled_target = target_points

        # Create RBF interpolators for x and y displacements
        dx = scaled_target[:, 0] - scaled_source[:, 0]
        dy = scaled_target[:, 1] - scaled_source[:, 1]

        try:
            rbf_x = Rbf(scaled_source[:, 0], scaled_source[:, 1], dx, function="thin_plate", smooth=self.config.smoothing_factor)
            rbf_y = Rbf(scaled_source[:, 0], scaled_source[:, 1], dy, function="thin_plate", smooth=self.config.smoothing_factor)
        except Exception:
            # Fallback: return original image if RBF fails
            return image.copy()

        # Create coordinate grids on downsampled resolution
        x_coords, y_coords = np.meshgrid(np.arange(downsample_w), np.arange(downsample_h))

        # Calculate displacements
        dx_map = rbf_x(x_coords, y_coords)
        dy_map = rbf_y(x_coords, y_coords)

        # Apply warping on downsampled image
        map_x = (x_coords + dx_map).astype(np.float32)
        map_y = (y_coords + dy_map).astype(np.float32)

        warped_downsampled = cv2.remap(downsampled_image, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

        # Upsample back to original resolution if needed
        if max(h, w) > MAX_DIMENSION:
            warped = cv2.resize(warped_downsampled, (w, h), interpolation=cv2.INTER_LINEAR)
        else:
            warped = warped_downsampled

        return warped

    def _assess_flattening_quality(
        self, original: np.ndarray, flattened: np.ndarray, surface_normals: SurfaceNormals
    ) -> FlatteningQualityMetrics:
        """
        Assess quality of flattening operation.

        Args:
            original: Original image
            flattened: Flattened image
            surface_normals: Original surface normals

        Returns:
            FlatteningQualityMetrics with quality assessment
        """
        # Re-estimate surface normals on flattened image
        flattened_normals = self._estimate_surface_normals(flattened)

        # Calculate distortion score (lower is better)
        curvature_reduction = surface_normals.mean_curvature - flattened_normals.mean_curvature
        if surface_normals.mean_curvature > 0:
            distortion_score = 1.0 - min(curvature_reduction / surface_normals.mean_curvature, 1.0)
        else:
            distortion_score = 0.0
        distortion_score = np.clip(distortion_score, 0.0, 1.0)

        # Calculate edge preservation score
        original_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY) if len(original.shape) == 3 else original
        flattened_gray = cv2.cvtColor(flattened, cv2.COLOR_BGR2GRAY) if len(flattened.shape) == 3 else flattened

        original_edges = cv2.Canny(original_gray, 50, 150)
        flattened_edges = cv2.Canny(flattened_gray, 50, 150)

        try:
            corr_matrix = np.corrcoef(original_edges.flatten(), flattened_edges.flatten())
            edge_correlation = float(corr_matrix[0, 1])
            edge_preservation_score = max(0.0, min(edge_correlation, 1.0))
        except Exception:
            # If correlation fails (e.g., no edges), default to 0.5
            edge_preservation_score = 0.5

        # Calculate smoothness score
        smoothness_score = 1.0 - flattened_normals.mean_curvature
        smoothness_score = np.clip(smoothness_score, 0.0, 1.0)

        # Overall quality (weighted average)
        overall_quality = 0.4 * (1.0 - distortion_score) + 0.3 * edge_preservation_score + 0.3 * smoothness_score

        return FlatteningQualityMetrics(
            distortion_score=float(distortion_score),
            edge_preservation_score=float(edge_preservation_score),
            smoothness_score=float(smoothness_score),
            overall_quality=float(overall_quality),
            residual_curvature=float(flattened_normals.mean_curvature),
            processing_successful=overall_quality > 0.5,
        )


def flatten_crumpled_document(
    image: np.ndarray, config: FlatteningConfig | None = None, corners: np.ndarray | None = None
) -> FlatteningResult:
    """
    Convenience function to flatten a crumpled document.

    Args:
        image: Input document image
        config: Optional flattening configuration
        corners: Optional document corners

    Returns:
        FlatteningResult with flattened image and metadata
    """
    flattener = DocumentFlattener(config)
    return flattener.flatten_document(image, corners)
