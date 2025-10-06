#!/usr/bin/env python3
"""
Comparison script between office-lens scanner and our preprocessing module
"""

import os

import cv2
import matplotlib.pyplot as plt
import numpy as np

# Import our preprocessing module
from ocr.utils.path_utils import setup_project_paths

setup_project_paths()
from ocr.datasets.preprocessing import DocumentPreprocessor


def test_office_lens_scanner(image_path):
    """Test the office-lens scanner implementation"""
    print("=== Testing Office Lens Scanner ===")

    im = cv2.imread(image_path)
    if im is None:
        print("Error: Could not load image")
        return None

    print(f"Image shape: {im.shape}")

    # Convert to grayscale
    grayscale = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

    # Apply blur
    blurred = cv2.GaussianBlur(grayscale, (3, 3), 0)

    # Edge detection
    edges = cv2.Canny(blurred, 50, 150)

    # Find contours
    contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

    vertices = None
    for cnt in contours:
        epsilon = 0.02 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        if len(approx) == 4:
            vertices = approx.reshape(4, 2)
            break

    # Fallback to convex hull
    if vertices is None and contours:
        hull = cv2.convexHull(contours[0])
        epsilon = 0.02 * cv2.arcLength(hull, True)
        approx = cv2.approxPolyDP(hull, epsilon, True)
        if len(approx) == 4:
            vertices = approx.reshape(4, 2)

    # Final fallback to image corners
    if vertices is None:
        height, width = im.shape[:2]
        vertices = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float32)

    # Reorder vertices
    reordered = np.zeros_like(vertices, dtype=np.float32)
    add = vertices.sum(1)
    reordered[0] = vertices[np.argmin(add)]  # top-left
    reordered[2] = vertices[np.argmax(add)]  # bottom-right
    diff = np.diff(vertices, axis=1)
    reordered[1] = vertices[np.argmin(diff)]  # top-right
    reordered[3] = vertices[np.argmax(diff)]  # bottom-left

    vertices = reordered
    print(f"Detected vertices: {vertices}")

    # Apply perspective transform
    width, height = 600, 850
    target = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float32)
    transform = cv2.getPerspectiveTransform(vertices.astype(np.float32), target)
    warped = cv2.warpPerspective(im, transform, (width, height))

    # Simple enhancement
    enhanced = cv2.convertScaleAbs(warped, alpha=1.1, beta=10)

    return enhanced, vertices


def test_our_preprocessor(image_path):
    """Test our DocumentPreprocessor"""
    print("\n=== Testing Our DocumentPreprocessor ===")

    im = cv2.imread(image_path)
    if im is None:
        print("Error: Could not load image")
        return None

    print(f"Image shape: {im.shape}")

    # Initialize preprocessor
    preprocessor = DocumentPreprocessor(
        enable_document_detection=True,
        enable_perspective_correction=True,
        enable_enhancement=True,
        enable_text_enhancement=False,
        target_size=(640, 640),
    )

    # Process image
    result = preprocessor(im)

    processed_image = result["image"]
    metadata = result["metadata"]

    print(f"Processing steps: {metadata['processing_steps']}")
    print(f"Document corners detected: {metadata['document_corners'] is not None}")
    if metadata["document_corners"] is not None:
        print(f"Corners: {metadata['document_corners']}")
    print(f"Final shape: {processed_image.shape}")

    return processed_image, metadata


def main():
    image_path = "data/datasets/images/test/drp.en_ko.in_house.selectstar_000017.jpg"

    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        return

    print(f"Testing on image: {image_path}")

    # Test both methods
    office_lens_result = test_office_lens_scanner(image_path)
    our_result = test_our_preprocessor(image_path)

    if office_lens_result is None or our_result is None:
        print("Error: One or both methods failed")
        return

    office_lens_img, office_lens_vertices = office_lens_result
    our_img, our_metadata = our_result

    # Create comparison visualization
    plt.figure(figsize=(15, 10))

    # Original image
    plt.subplot(2, 3, 1)
    original = cv2.imread(image_path)
    plt.imshow(original[:, :, ::-1])
    plt.title("Original Image")
    plt.axis("off")

    # Office Lens result
    plt.subplot(2, 3, 2)
    plt.imshow(office_lens_img[:, :, ::-1])
    plt.title("Office Lens Scanner")
    plt.axis("off")

    # Our preprocessor result
    plt.subplot(2, 3, 3)
    plt.imshow(our_img[:, :, ::-1])
    plt.title("Our DocumentPreprocessor")
    plt.axis("off")

    # Show detected corners on original
    plt.subplot(2, 3, 4)
    plt.imshow(original[:, :, ::-1])
    if office_lens_vertices is not None:
        plt.scatter(
            [x for x, y in office_lens_vertices],
            [y for x, y in office_lens_vertices],
            c="red",
            s=50,
            label="Office Lens",
        )
    if our_metadata["document_corners"] is not None:
        plt.scatter(
            [x for x, y in our_metadata["document_corners"]],
            [y for x, y in our_metadata["document_corners"]],
            c="blue",
            s=50,
            marker="x",
            label="Our Preprocessor",
        )
    plt.title("Detected Document Corners")
    plt.legend()
    plt.axis("off")

    # Processing info
    plt.subplot(2, 3, 5)
    info_text = (
        ".1f"
        ".1f"
        f"""
Office Lens Scanner:
- Detected: {"Yes" if not np.array_equal(office_lens_vertices, np.array([[0, 0], [648, 0], [648, 1280], [0, 1280]])) else "No"}
- Output size: {office_lens_img.shape}

Our Preprocessor:
- Steps: {", ".join(our_metadata["processing_steps"])}
- Detected: {"Yes" if our_metadata["document_corners"] is not None else "No"}
- Output size: {our_img.shape}
"""
    )
    plt.text(
        0.1,
        0.5,
        info_text,
        transform=plt.gca().transAxes,
        fontsize=10,
        verticalalignment="center",
        fontfamily="monospace",
    )
    plt.title("Processing Comparison")
    plt.axis("off")

    plt.tight_layout()
    plt.savefig("preprocessor_comparison.png", dpi=150, bbox_inches="tight")
    plt.show()

    print("\n=== Comparison Results ===")
    office_lens_detected = "Yes" if not np.array_equal(office_lens_vertices, np.array([[0, 0], [648, 0], [648, 1280], [0, 1280]])) else "No"
    our_detected = "Yes" if our_metadata["document_corners"] is not None else "No"
    print(f"Office Lens: Detected document corners: {office_lens_detected}")
    print(f"Our Preprocessor: Detected document corners: {our_detected}")
    print(f"Processing steps applied: {', '.join(our_metadata['processing_steps'])}")

    # Save individual results
    cv2.imwrite("office_lens_result.jpg", office_lens_img)
    cv2.imwrite("our_preprocessor_result.jpg", our_img)

    print("\nSaved results:")
    print("- preprocessor_comparison.png (visual comparison)")
    print("- office_lens_result.jpg")
    print("- our_preprocessor_result.jpg")


if __name__ == "__main__":
    main()
    main()
