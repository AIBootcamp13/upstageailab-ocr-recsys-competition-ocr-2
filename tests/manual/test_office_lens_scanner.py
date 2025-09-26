#!/usr/bin/env python3
"""
Test script for the document scanner implementation from documment_scanner_nb.py
"""

import os

import cv2
import matplotlib.pyplot as plt
import numpy as np


def imshow(im, title="Image"):
    """Display image using matplotlib"""
    plt.figure(figsize=(10, 8))
    width, height, *channels = im.shape
    if channels:
        # Convert BGR to RGB for matplotlib
        plt.imshow(im[:, :, ::-1])
    else:
        plt.imshow(im, cmap="gray")
    plt.title(title)
    plt.axis("off")
    plt.show()


def reorder(vertices):
    """Reorder vertices in consistent order: top-left, top-right, bottom-right, bottom-left"""
    reordered = np.zeros_like(vertices, dtype=np.float32)
    add = vertices.sum(1)
    reordered[0] = vertices[np.argmin(add)]  # top-left
    reordered[2] = vertices[np.argmax(add)]  # bottom-right
    diff = np.diff(vertices, axis=1)
    reordered[1] = vertices[np.argmin(diff)]  # top-right
    reordered[3] = vertices[np.argmax(diff)]  # bottom-left
    return reordered


def to_grayscale(im):
    """Convert image to grayscale"""
    return cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)


def blur(im):
    """Apply Gaussian blur"""
    return cv2.GaussianBlur(im, (3, 3), 0)


def to_edges(im):
    """Apply Canny edge detection"""
    edges = cv2.Canny(im, 50, 150)
    return edges


def find_vertices(im):
    """Find document vertices using contour detection"""
    contours, _ = cv2.findContours(im.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Sort contours by area and keep the largest 5
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

    vertices = None
    for cnt in contours:
        # Try to approximate to a quadrilateral
        epsilon = 0.02 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)

        if len(approx) == 4:
            vertices = approx.reshape(4, 2)
            break

    # Fallback: if no quadrilateral found, try the largest contour's convex hull
    if vertices is None and contours:
        hull = cv2.convexHull(contours[0])
        epsilon = 0.02 * cv2.arcLength(hull, True)
        approx = cv2.approxPolyDP(hull, epsilon, True)
        if len(approx) == 4:
            vertices = approx.reshape(4, 2)

    # If still not found, fallback to image corners
    if vertices is None:
        height, width = im.shape[:2]
        vertices = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float32)

    vertices = reorder(vertices)
    return vertices


def crop_out(im, vertices):
    """Apply perspective transform to crop out document"""
    width = 600
    height = 850

    target = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float32)

    transform = cv2.getPerspectiveTransform(vertices, target)
    cropped = cv2.warpPerspective(im, transform, (width, height))

    return cropped


def enhance(im):
    """Apply image enhancement"""
    # Mild color correction
    gamma = 1.1
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    corrected = cv2.LUT(im, table)

    # LAB Color Space
    lab = cv2.cvtColor(corrected, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    # Gentle CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(12, 12))
    l = clahe.apply(l)

    # Merge channels
    lab = cv2.merge([l, a, b])
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    # Smart saturation boost
    hsv = cv2.cvtColor(enhanced, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    # Saturation mask
    s_clean = cv2.bilateralFilter(s, 9, 75, 75)
    s = cv2.addWeighted(s, 1.2, s_clean, 0.3, 0)
    s = np.clip(s, 20, 230).astype(np.uint8)

    hsv = cv2.merge([h, s, v])
    enhanced = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    # Controlled sharpening
    blurred = cv2.GaussianBlur(enhanced, (0, 0), 2.0)
    sharpened = cv2.addWeighted(enhanced, 1.5, blurred, -0.5, 0)

    # Noise reduction
    final = cv2.bilateralFilter(sharpened, 9, 75, 75)
    return final


def scan(im):
    """Complete document scanning pipeline"""
    print("Step 1: Converting to grayscale...")
    grayscale = to_grayscale(im)

    print("Step 2: Applying blur...")
    blurred = blur(grayscale)

    print("Step 3: Detecting edges...")
    edges = to_edges(blurred)

    print("Step 4: Finding document vertices...")
    vertices = find_vertices(edges)
    print(f"Found vertices: {vertices}")

    print("Step 5: Applying perspective correction...")
    cropped = crop_out(im, vertices)

    print("Step 6: Enhancing image...")
    enhanced_img = enhance(cropped)

    return enhanced_img, vertices


def main():
    # Test image path
    image_path = "data/datasets/images/test/drp.en_ko.in_house.selectstar_000017.jpg"

    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        return

    print(f"Loading image: {image_path}")
    im = cv2.imread(image_path)

    if im is None:
        print("Error: Could not load image")
        return

    print(f"Image shape: {im.shape}")

    # Run document scanning
    print("\n=== Starting Document Scanning ===")
    scanned, vertices = scan(im)

    # Save result
    output_path = "scanned_office_lens_test.jpg"
    cv2.imwrite(output_path, scanned)
    print(f"\nSaved scanned image to: {output_path}")

    # Display results if matplotlib is available
    try:
        print("\nDisplaying results...")
        plt.figure(figsize=(15, 5))

        plt.subplot(1, 3, 1)
        plt.imshow(im[:, :, ::-1])
        plt.title("Original")
        plt.axis("off")

        plt.subplot(1, 3, 2)
        plt.imshow(im[:, :, ::-1])
        plt.scatter([x for x, y in vertices], [y for x, y in vertices], c="red", s=50)
        plt.title("Detected Corners")
        plt.axis("off")

        plt.subplot(1, 3, 3)
        plt.imshow(scanned[:, :, ::-1])
        plt.title("Scanned Result")
        plt.axis("off")

        plt.tight_layout()
        plt.savefig("scanning_comparison.png", dpi=150, bbox_inches="tight")
        plt.show()

        print("Saved comparison image to: scanning_comparison.png")

    except ImportError:
        print("Matplotlib not available for visualization")

    print("\n=== Document Scanning Complete ===")


if __name__ == "__main__":
    main()
