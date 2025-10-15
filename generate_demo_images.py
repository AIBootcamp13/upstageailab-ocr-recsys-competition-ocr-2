"""
Demo Image Generation for Advanced Document Preprocessing.

This script generates synthetic demo images showcasing Phase 1 Foundation
and Phase 2 Enhancement capabilities of the preprocessing pipeline.
"""

import random
from pathlib import Path

import cv2
import numpy as np


class DemoImageGenerator:
    """Generate demo images for preprocessing pipeline phases."""

    def __init__(self, output_dir: str = "demo_images"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

    def generate_phase1_foundation_images(self):
        """Generate demo images for Phase 1: Foundation capabilities."""
        print("Generating Phase 1 Foundation demo images...")

        # Perfect rectangle document
        self._create_perfect_rectangle_document()

        # Skewed quadrilateral document
        self._create_skewed_quadrilateral_document()

        # Complex document with perspective
        self._create_perspective_document()

        # Low contrast document
        self._create_low_contrast_document()

        # Document with background noise
        self._create_noisy_background_document()

        # Multiple documents in frame
        self._create_multiple_documents()

        print("Phase 1 demo images generated successfully!")

    def generate_phase2_enhancement_images(self):
        """Generate demo images for Phase 2: Enhancement capabilities."""
        print("Generating Phase 2 Enhancement demo images...")

        # Shadow removal scenarios
        self._create_shadow_document()
        self._create_multiple_shadows_document()

        # Background noise elimination
        self._create_complex_background_noise()
        self._create_textured_background_document()

        # Document flattening scenarios
        self._create_crumpled_paper_effect()
        self._create_perspective_distortion()
        self._create_curved_document()

        # Brightness adjustment scenarios
        self._create_uneven_lighting()
        self._create_low_light_document()
        self._create_overexposed_document()

        # Combined challenges
        self._create_combined_challenges_document()

        print("Phase 2 demo images generated successfully!")

    def _create_perfect_rectangle_document(self):
        """Create a perfect rectangular document (Phase 1 baseline)."""
        img = np.ones((400, 600, 3), dtype=np.uint8) * 255  # White background

        # Draw perfect rectangle document
        cv2.rectangle(img, (150, 100), (450, 300), (240, 240, 240), -1)  # Light gray document

        # Add subtle border
        cv2.rectangle(img, (150, 100), (450, 300), (200, 200, 200), 2)

        # Add some text-like content
        self._add_document_content(img, (150, 100), (450, 300))

        cv2.imwrite(str(self.output_dir / "phase1_foundation" / "perfect_rectangle.png"), img)

    def _create_skewed_quadrilateral_document(self):
        """Create a skewed quadrilateral document."""
        img = np.ones((400, 600, 3), dtype=np.uint8) * 255

        # Define skewed quadrilateral points
        pts = np.array([[120, 80], [480, 120], [440, 320], [160, 280]], dtype=np.int32)

        # Fill document area
        cv2.fillPoly(img, [pts], (240, 240, 240))

        # Draw border
        cv2.polylines(img, [pts], True, (200, 200, 200), 2)

        # Add content
        self._add_document_content(img, (120, 80), (480, 320))

        cv2.imwrite(str(self.output_dir / "phase1_foundation" / "skewed_quadrilateral.png"), img)

    def _create_perspective_document(self):
        """Create document with perspective distortion."""
        img = np.ones((400, 600, 3), dtype=np.uint8) * 255

        # Define perspective points (trapezoid)
        pts = np.array([[100, 150], [500, 100], [520, 300], [80, 350]], dtype=np.int32)

        cv2.fillPoly(img, [pts], (240, 240, 240))
        cv2.polylines(img, [pts], True, (200, 200, 200), 2)

        self._add_document_content(img, (80, 100), (520, 350))

        cv2.imwrite(str(self.output_dir / "phase1_foundation" / "perspective_document.png"), img)

    def _create_low_contrast_document(self):
        """Create document with low contrast."""
        img = np.ones((400, 600, 3), dtype=np.uint8) * 220  # Light gray background

        # Very subtle document
        cv2.rectangle(img, (150, 100), (450, 300), (200, 200, 200), -1)
        cv2.rectangle(img, (150, 100), (450, 300), (180, 180, 180), 2)

        self._add_document_content(img, (150, 100), (450, 300), contrast=0.3)

        cv2.imwrite(str(self.output_dir / "phase1_foundation" / "low_contrast.png"), img)

    def _create_noisy_background_document(self):
        """Create document with background noise."""
        img = np.ones((400, 600, 3), dtype=np.uint8) * 255

        # Add background noise
        noise = np.random.normal(0, 10, img.shape).astype(np.uint8)
        img = cv2.add(img, noise)

        # Draw document
        cv2.rectangle(img, (150, 100), (450, 300), (220, 220, 220), -1)
        cv2.rectangle(img, (150, 100), (450, 300), (180, 180, 180), 2)

        self._add_document_content(img, (150, 100), (450, 300))

        cv2.imwrite(str(self.output_dir / "phase1_foundation" / "noisy_background.png"), img)

    def _create_multiple_documents(self):
        """Create image with multiple documents."""
        img = np.ones((400, 600, 3), dtype=np.uint8) * 255

        # Document 1 - main document
        cv2.rectangle(img, (100, 80), (350, 250), (240, 240, 240), -1)
        cv2.rectangle(img, (100, 80), (350, 250), (200, 200, 200), 2)
        self._add_document_content(img, (100, 80), (350, 250))

        # Document 2 - smaller overlapping
        cv2.rectangle(img, (300, 200), (500, 320), (230, 230, 230), -1)
        cv2.rectangle(img, (300, 200), (500, 320), (190, 190, 190), 2)
        self._add_document_content(img, (300, 200), (500, 320), font_scale=0.5)

        cv2.imwrite(str(self.output_dir / "phase1_foundation" / "multiple_documents.png"), img)

    def _create_shadow_document(self):
        """Create document with shadow effects."""
        img = np.ones((400, 600, 3), dtype=np.uint8) * 255

        # Create shadow mask
        shadow_mask = np.zeros((400, 600), dtype=np.uint8)
        shadow_pts = np.array([[180, 130], [480, 90], [500, 330], [200, 370]], dtype=np.int32)
        cv2.fillPoly(shadow_mask, [shadow_pts], (255,))

        # Apply shadow
        img[shadow_mask > 0] = [220, 220, 240]  # Slight blue tint for shadow

        # Draw document on top
        doc_pts = np.array([[150, 100], [450, 60], [470, 300], [170, 340]], dtype=np.int32)
        cv2.fillPoly(img, [doc_pts], (240, 240, 240))
        cv2.polylines(img, [doc_pts], True, (200, 200, 200), 2)

        self._add_document_content(img, (150, 60), (470, 340))

        cv2.imwrite(str(self.output_dir / "phase2_enhancement" / "shadow_document.png"), img)

    def _create_multiple_shadows_document(self):
        """Create document with multiple shadow sources."""
        img = np.ones((400, 600, 3), dtype=np.uint8) * 255

        # Multiple shadow sources
        shadow_sources = [
            ([200, 150], [500, 100], [520, 350], [220, 400]),  # Main shadow
            ([100, 200], [300, 180], [320, 280], [120, 300]),  # Secondary shadow
        ]

        for shadow_pts in shadow_sources:
            shadow_mask = np.zeros((400, 600), dtype=np.uint8)
            cv2.fillPoly(shadow_mask, [np.array(shadow_pts, dtype=np.int32)], (255,))
            img[shadow_mask > 0] = np.clip(img[shadow_mask > 0] - 30, 0, 255)

        # Draw document
        cv2.rectangle(img, (150, 100), (450, 300), (240, 240, 240), -1)
        cv2.rectangle(img, (150, 100), (450, 300), (200, 200, 200), 2)

        self._add_document_content(img, (150, 100), (450, 300))

        cv2.imwrite(str(self.output_dir / "phase2_enhancement" / "multiple_shadows.png"), img)

    def _create_complex_background_noise(self):
        """Create document with complex background patterns."""
        img = np.ones((400, 600, 3), dtype=np.uint8) * 255

        # Add geometric patterns to background
        for _ in range(20):
            center = (random.randint(0, 600), random.randint(0, 400))
            cv2.circle(img, center, random.randint(10, 50), (220, 220, 220), -1)

        # Add some lines
        for _ in range(10):
            pt1 = (random.randint(0, 600), random.randint(0, 400))
            pt2 = (random.randint(0, 600), random.randint(0, 400))
            cv2.line(img, pt1, pt2, (210, 210, 210), 1)

        # Draw document
        cv2.rectangle(img, (150, 100), (450, 300), (240, 240, 240), -1)
        cv2.rectangle(img, (150, 100), (450, 300), (200, 200, 200), 2)

        self._add_document_content(img, (150, 100), (450, 300))

        cv2.imwrite(str(self.output_dir / "phase2_enhancement" / "complex_background.png"), img)

    def _create_textured_background_document(self):
        """Create document with textured background."""
        img = np.ones((400, 600, 3), dtype=np.uint8) * 255

        # Create wood grain texture
        for y in range(400):
            for x in range(600):
                noise = np.sin(x * 0.01 + y * 0.05) * 10 + np.random.normal(0, 5)
                img[y, x] = np.clip(img[y, x] + noise, 200, 255)

        # Draw document
        cv2.rectangle(img, (150, 100), (450, 300), (240, 240, 240), -1)
        cv2.rectangle(img, (150, 100), (450, 300), (200, 200, 200), 2)

        self._add_document_content(img, (150, 100), (450, 300))

        cv2.imwrite(str(self.output_dir / "phase2_enhancement" / "textured_background.png"), img)

    def _create_crumpled_paper_effect(self):
        """Create crumpled paper effect for flattening demo."""
        img = np.ones((400, 600, 3), dtype=np.uint8) * 255

        # Create crumple effect using sine waves
        crumple_mask = np.zeros((400, 600), dtype=np.float32)
        for i in range(3):
            freq_x = 0.02 + i * 0.01
            freq_y = 0.015 + i * 0.008
            amplitude = 20 / (i + 1)
            x_wave = amplitude * np.sin(np.arange(600) * freq_x)
            y_wave = amplitude * np.sin(np.arange(400) * freq_y)
            crumple_mask += x_wave[None, :] + y_wave[:, None]

        # Apply crumple to document area
        doc_mask = np.zeros((400, 600), dtype=np.uint8)
        cv2.rectangle(doc_mask, (150, 100), (450, 300), 255, -1)

        # Simulate crumpled surface
        crumpled_region = crumple_mask[100:300, 150:450]
        crumpled_region = cv2.GaussianBlur(crumpled_region.astype(np.uint8), (5, 5), 0)

        # Draw crumpled document
        cv2.rectangle(img, (150, 100), (450, 300), (240, 240, 240), -1)
        cv2.rectangle(img, (150, 100), (450, 300), (200, 200, 200), 2)

        # Add crumple shading
        crumple_overlay = np.zeros((200, 300, 3), dtype=np.uint8)
        crumple_overlay[:, :] = [240, 240, 240]
        crumple_overlay[crumpled_region.astype(np.uint8) > 128] = [220, 220, 220]
        crumple_overlay[crumpled_region.astype(np.uint8) < 128] = [250, 250, 250]

        img[100:300, 150:450] = crumple_overlay

        self._add_document_content(img, (150, 100), (450, 300))

        cv2.imwrite(str(self.output_dir / "phase2_enhancement" / "crumpled_paper.png"), img)

    def _create_perspective_distortion(self):
        """Create document with strong perspective distortion."""
        img = np.ones((400, 600, 3), dtype=np.uint8) * 255

        # Extreme perspective points
        pts = np.array([[50, 200], [550, 150], [570, 350], [30, 400]], dtype=np.int32)

        cv2.fillPoly(img, [pts], (240, 240, 240))
        cv2.polylines(img, [pts], True, (200, 200, 200), 2)

        self._add_document_content(img, (30, 150), (570, 400))

        cv2.imwrite(str(self.output_dir / "phase2_enhancement" / "extreme_perspective.png"), img)

    def _create_curved_document(self):
        """Create document with curved surface effect."""
        img = np.ones((400, 600, 3), dtype=np.uint8) * 255

        # Create curved surface using polynomial
        y_coords = np.arange(100, 301)
        x_left = 150 + 50 * np.sin((y_coords - 100) * 0.02)
        x_right = 450 + 30 * np.cos((y_coords - 100) * 0.015)

        # Draw curved document
        for i, y in enumerate(y_coords):
            cv2.line(img, (int(x_left[i]), y), (int(x_right[i]), y), (240, 240, 240), 1)

        # Add border
        cv2.polylines(img, [np.column_stack((x_left.astype(int), y_coords))], False, (200, 200, 200), 2)
        cv2.polylines(img, [np.column_stack((x_right.astype(int), y_coords))], False, (200, 200, 200), 2)

        self._add_document_content(img, (150, 100), (450, 300))

        cv2.imwrite(str(self.output_dir / "phase2_enhancement" / "curved_surface.png"), img)

    def _create_uneven_lighting(self):
        """Create document with uneven lighting conditions."""
        img = np.ones((400, 600, 3), dtype=np.uint8) * 255

        # Create lighting gradient
        gradient = np.zeros((400, 600), dtype=np.float32)
        for y in range(400):
            for x in range(600):
                # Radial gradient from top-left
                dist = np.sqrt((x - 100) ** 2 + (y - 100) ** 2)
                gradient[y, x] = 1.0 - min(dist / 400, 1.0)

        # Apply lighting to document area
        doc_mask = np.zeros((400, 600), dtype=np.uint8)
        cv2.rectangle(doc_mask, (150, 100), (450, 300), 255, -1)

        lighting_factor = 0.5 + 0.5 * gradient
        lit_region = (240 * lighting_factor[100:300, 150:450]).astype(np.uint8)

        img[100:300, 150:450] = np.stack([lit_region] * 3, axis=-1)
        cv2.rectangle(img, (150, 100), (450, 300), (200, 200, 200), 2)

        self._add_document_content(img, (150, 100), (450, 300))

        cv2.imwrite(str(self.output_dir / "phase2_enhancement" / "uneven_lighting.png"), img)

    def _create_low_light_document(self):
        """Create document in low light conditions."""
        img = np.ones((400, 600, 3), dtype=np.uint8) * 50  # Very dark background

        # Draw document (also dark)
        cv2.rectangle(img, (150, 100), (450, 300), (80, 80, 80), -1)
        cv2.rectangle(img, (150, 100), (450, 300), (60, 60, 60), 2)

        self._add_document_content(img, (150, 100), (450, 300), brightness=0.3)

        cv2.imwrite(str(self.output_dir / "phase2_enhancement" / "low_light.png"), img)

    def _create_overexposed_document(self):
        """Create overexposed document."""
        img = np.ones((400, 600, 3), dtype=np.uint8) * 200  # Bright background

        # Draw document (washed out)
        cv2.rectangle(img, (150, 100), (450, 300), (250, 250, 250), -1)
        cv2.rectangle(img, (150, 100), (450, 300), (240, 240, 240), 2)

        self._add_document_content(img, (150, 100), (450, 300), brightness=1.5)

        cv2.imwrite(str(self.output_dir / "phase2_enhancement" / "overexposed.png"), img)

    def _create_combined_challenges_document(self):
        """Create document with multiple combined challenges."""
        img = np.ones((400, 600, 3), dtype=np.uint8) * 255

        # Add background texture
        for y in range(400):
            for x in range(600):
                img[y, x] = np.clip(img[y, x] + np.random.normal(0, 15), 0, 255)

        # Add shadows
        shadow_mask = np.zeros((400, 600), dtype=np.uint8)
        cv2.ellipse(shadow_mask, (300, 200), (100, 150), 45, 0, 360, (255,), -1)
        img[shadow_mask > 0] = np.clip(img[shadow_mask > 0] - 40, 0, 255)

        # Create skewed document with perspective
        pts = np.array([[120, 120], [480, 80], [500, 280], [140, 320]], dtype=np.int32)
        cv2.fillPoly(img, [pts], (220, 220, 220))
        cv2.polylines(img, [pts], True, (180, 180, 180), 2)

        # Add uneven lighting
        lighting = np.random.normal(1.0, 0.2, (400, 600))
        lighting = np.clip(lighting, 0.5, 1.5)
        img = (img.astype(np.float32) * lighting[:, :, None]).astype(np.uint8)

        self._add_document_content(img, (120, 80), (500, 320))

        cv2.imwrite(str(self.output_dir / "phase2_enhancement" / "combined_challenges.png"), img)

    def _add_document_content(
        self,
        img: np.ndarray,
        top_left: tuple[int, int],
        bottom_right: tuple[int, int],
        font_scale: float = 0.7,
        contrast: float = 1.0,
        brightness: float = 1.0,
    ):
        """Add simulated document content (text lines, etc.)."""
        x1, y1 = top_left
        x2, y2 = bottom_right

        # Add some horizontal lines (text lines)
        line_height = 25
        for i in range(3, int((y2 - y1) / line_height) - 1):
            y_pos = y1 + i * line_height
            if y_pos < y2 - 10:
                line_start = (x1 + 20, y_pos)
                line_end = (x2 - 20, y_pos)
                color_val = int(150 * contrast * brightness)
                color_val = max(0, min(255, color_val))
                cv2.line(img, line_start, line_end, (color_val, color_val, color_val), 1)

        # Add some vertical lines (margins)
        cv2.line(img, (x1 + 15, y1 + 10), (x1 + 15, y2 - 10), (180, 180, 180), 1)
        cv2.line(img, (x2 - 15, y1 + 10), (x2 - 15, y2 - 10), (180, 180, 180), 1)


def main():
    """Generate all demo images."""
    generator = DemoImageGenerator()

    print("Starting demo image generation...")
    generator.generate_phase1_foundation_images()
    generator.generate_phase2_enhancement_images()

    print("\nDemo images generated successfully!")
    print("Phase 1 images saved to: demo_images/phase1_foundation/")
    print("Phase 2 images saved to: demo_images/phase2_enhancement/")

    # List generated files
    print("\nGenerated Phase 1 images:")
    for img_file in sorted((Path("demo_images") / "phase1_foundation").glob("*.png")):
        print(f"  - {img_file.name}")

    print("\nGenerated Phase 2 images:")
    for img_file in sorted((Path("demo_images") / "phase2_enhancement").glob("*.png")):
        print(f"  - {img_file.name}")


if __name__ == "__main__":
    main()
