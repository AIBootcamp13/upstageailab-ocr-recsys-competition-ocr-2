# %% [markdown]
# # Imports

import logging
import os

# %%
import cv2
import matplotlib.pyplot as plt
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# %% [markdown]
# # Utilities


# %%
def imshow(im):
    # If the image is grayscale (2D), show it with a grayscale colormap.
    if len(im.shape) == 2:
        plt.imshow(im, cmap="gray")
    else:
        # If the image is color (3D), convert from BGR (OpenCV's default) to RGB for Matplotlib.
        plt.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
    plt.axis("off")


# %%
def reorder(vertices):
    # Reorders the 4 vertices into a consistent order: top-left, top-right, bottom-right, bottom-left
    assert vertices.shape == (
        4,
        2,
    ), f"Input vertices must have shape (4, 2), got {vertices.shape}"

    reordered = np.zeros_like(vertices, dtype=np.float32)
    add = vertices.sum(1)
    reordered[0] = vertices[np.argmin(add)]
    reordered[2] = vertices[np.argmax(add)]
    diff = np.diff(vertices, axis=1)
    reordered[1] = vertices[np.argmin(diff)]
    reordered[3] = vertices[np.argmax(diff)]
    return reordered


# %% [markdown]
# # Image

# %%
# Load the image
script_dir = os.path.dirname(os.path.abspath(__file__))
image_path = os.path.join(script_dir, "../../data/datasets/images/test/drp.en_ko.in_house.selectstar_000126.jpg")
im = cv2.imread(image_path)
if im is None:
    raise ValueError(f"Could not load image from {image_path}")
imshow(im)  # We'll show the final result at the end

# %% [markdown]
# # Process

# %% [markdown]
# ## Grayscale Transform


# %%
def to_grayscale(im):
    return cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)


# %% [markdown]
# ## Blurring the Image


# %%
def blur(im):
    # Using a slightly larger kernel can help reduce noise for edge detection
    return cv2.GaussianBlur(im, (5, 5), 0)


# %% [markdown]
# ## Edge Detection


# %%
def to_edges(im):
    """
    Apply multiple edge detection strategies optimized for document boundaries.
    """
    # Convert to grayscale if needed
    if len(im.shape) == 3:
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    else:
        gray = im

    # Apply bilateral filter to reduce noise while preserving edges
    filtered = cv2.bilateralFilter(gray, 9, 75, 75)

    # Try different Canny thresholds optimized for documents
    edges1 = cv2.Canny(filtered, 30, 100)  # Lower thresholds for document edges
    edges2 = cv2.Canny(filtered, 50, 150)  # Standard thresholds
    edges3 = cv2.Canny(filtered, 75, 200)  # Higher thresholds for cleaner edges

    # Combine edges
    edges = cv2.bitwise_or(edges1, edges2)
    edges = cv2.bitwise_or(edges, edges3)

    # Apply morphological operations to connect document boundaries
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    # Dilate to connect broken edges
    edges = cv2.dilate(edges, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=2)

    return edges


# %% [markdown]
# ## Contour Detection


# %%
def find_best_quadrilateral_corners(points):
    """
    From a polygon with more than 4 points, find the best 4 corners for a quadrilateral.
    Uses the points with maximum distances from each other.
    """
    if len(points) <= 4:
        return points

    # Find the convex hull to get the outermost points
    hull = cv2.convexHull(points)

    # If hull has exactly 4 points, use it
    if len(hull) == 4:
        return hull.reshape(4, 2)

    # If hull has more than 4 points, select the 4 most distant points
    if len(hull) > 4:
        hull_points = hull.reshape(-1, 2)

        # Find the point with maximum distance from centroid
        centroid = np.mean(hull_points, axis=0)
        distances = [np.linalg.norm(p - centroid) for p in hull_points]
        center_idx = np.argmax(distances)

        # Find the 4 points most distant from each other
        selected = [hull_points[center_idx]]
        candidates = np.delete(hull_points, center_idx, axis=0)

        for _ in range(3):
            if len(candidates) == 0:
                break
            # Find point most distant from already selected points
            max_dist = 0
            best_idx = 0
            for i, candidate in enumerate(candidates):
                min_dist_to_selected = min(np.linalg.norm(candidate - s) for s in selected)
                if min_dist_to_selected > max_dist:
                    max_dist = min_dist_to_selected
                    best_idx = i
            selected.append(candidates[best_idx])
            candidates = np.delete(candidates, best_idx, axis=0)

        if len(selected) == 4:
            return np.array(selected)

    return None


def find_vertices(im, kernel_size=5, iterations=3, epsilon_factor=0.02):
    """
    Find document vertices using improved contour detection optimized for documents.
    """
    edges = to_edges(im)

    # Try multiple dilation strategies
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilated1 = cv2.dilate(edges, kernel, iterations=iterations)
    dilated2 = cv2.dilate(
        edges,
        cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size)),
        iterations=iterations,
    )

    # Find contours on both dilated versions
    contours1, _ = cv2.findContours(dilated1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours2, _ = cv2.findContours(dilated2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Combine and sort contours by area
    all_contours = list(contours1) + list(contours2)
    all_contours = sorted(all_contours, key=cv2.contourArea, reverse=True)

    # Try different epsilon factors for approximation
    epsilon_factors = [epsilon_factor * 2, epsilon_factor * 4, epsilon_factor * 6]
    image_area = im.shape[0] * im.shape[1]

    for contour in all_contours[:15]:  # Check top 15 contours
        area = cv2.contourArea(contour)
        logger.debug(f"Checking contour with area: {area}, image area: {image_area}")

        # Skip contours that are too small or too large
        if area < image_area * 0.05 or area > image_area * 0.95:
            logger.debug(f"Contour area {area} is outside valid range")
            continue

        # Check if contour is roughly rectangular by looking at bounding box
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / h
        logger.debug(f"Contour bounding box: {x},{y},{w},{h}, aspect_ratio: {aspect_ratio}")

        if aspect_ratio < 0.1 or aspect_ratio > 10:  # Too extreme aspect ratio
            logger.debug(f"Contour aspect ratio {aspect_ratio} is too extreme")
            continue

        for eps_factor in epsilon_factors:
            epsilon = eps_factor * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            logger.debug(f"Approximated to {len(approx)} points with epsilon factor {eps_factor}")

            # Look for approximations with 4-6 points (allowing some flexibility)
            if 4 <= len(approx) <= 6:
                vertices = approx.reshape(len(approx), 2)
                # If we have more than 4 points, try to find the best 4 corners
                if len(vertices) > 4:
                    vertices = find_best_quadrilateral_corners(vertices)
                    if vertices is None:
                        continue

                logger.debug(f"Found {len(vertices)}-point approximation: {vertices}")
                if _is_valid_quadrilateral(vertices):
                    logger.debug("Quadrilateral passed validation")
                    return vertices
                else:
                    logger.debug("Quadrilateral failed validation")

    # If no valid quadrilateral found, return None
    return None


def _is_reasonable_document(vertices, img_height, img_width):
    """
    Check if detected quadrilateral is a reasonable document candidate.

    Parameters
    ----------
    vertices : np.ndarray
        4x2 array of corners
    img_height, img_width : int
        Image dimensions

    Returns
    -------
    bool
        True if quadrilateral seems like a reasonable document
    """
    # Calculate bounding box
    x_coords = vertices[:, 0]
    y_coords = vertices[:, 1]

    width = np.max(x_coords) - np.min(x_coords)
    height = np.max(y_coords) - np.min(y_coords)

    # Check minimum size (at least 10% of image dimensions)
    min_width = img_width * 0.1
    min_height = img_height * 0.1

    if width < min_width or height < min_height:
        return False

    # Check aspect ratio (not too extreme)
    aspect_ratio = width / height
    if aspect_ratio < 0.1 or aspect_ratio > 10:
        return False

    # Check that it's not too thin (height/width ratio reasonable)
    if min(width, height) / max(width, height) < 0.05:
        return False

    return True


# %% [markdown]
# ## Perspective Transform


# %%
def fallback_crop(im):
    """
    Fallback cropping strategy when document detection fails.
    Uses text region detection to find the main content area.
    """
    try:
        # Convert to grayscale
        if len(im.shape) == 3:
            gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        else:
            gray = im

        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Use adaptive thresholding to find text regions
        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

        # Apply morphological operations to clean up noise
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

        # Find contours of text regions
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            logger.warning("No text contours found, returning original image")
            return im

        # Filter contours by area (ignore very small noise)
        min_area = (im.shape[0] * im.shape[1]) * 0.001  # At least 0.1% of image area
        valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]

        if not valid_contours:
            logger.warning("No valid text contours found, returning original image")
            return im

        # Find bounding box of all valid text regions
        all_points = np.concatenate([cnt.reshape(-1, 2) for cnt in valid_contours])
        x, y, w, h = cv2.boundingRect(all_points)

        # Add some padding around the text region
        padding = 30
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = min(im.shape[1] - x, w + 2 * padding)
        h = min(im.shape[0] - y, h + 2 * padding)

        # Ensure minimum size and reasonable aspect ratio
        if w < 200 or h < 200:
            logger.warning("Text region too small, returning original image")
            return im

        aspect_ratio = w / h
        if aspect_ratio < 0.2 or aspect_ratio > 5.0:
            logger.warning(f"Text region aspect ratio {aspect_ratio:.2f} seems unreasonable, returning original image")
            return im

        # Crop the region
        cropped = im[y : y + h, x : x + w]

        logger.info(f"Fallback cropping applied. Region: ({x}, {y}, {w}, {h}) -> Output size: {cropped.shape}")
        return cropped

    except Exception as e:
        logger.error(f"Fallback cropping failed: {e}, returning original image")
        return im


def crop_out(im, vertices):
    """
    Apply perspective correction to crop out the document.
    If vertices are invalid, use fallback cropping strategy.

    Parameters
    ----------
    im : np.ndarray
        Input image
    vertices : np.ndarray or None
        4x2 array of document corners [tl, tr, br, bl], or None

    Returns
    -------
    cropped : np.ndarray
        Perspective-corrected document image or fallback crop
    """
    # Check if vertices are valid
    if vertices is None or vertices.shape != (4, 2):
        logger.warning("Invalid or missing vertices, using fallback cropping")
        return fallback_crop(im)

    # Check for degenerate quadrilateral
    if not _is_valid_quadrilateral(vertices):
        logger.warning("Detected quadrilateral appears degenerate, using fallback cropping")
        return fallback_crop(im)

    (tl, tr, br, bl) = vertices

    # Calculate the width of the new image
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    # Calculate the height of the new image
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    # Ensure minimum dimensions
    maxWidth = max(maxWidth, 50)
    maxHeight = max(maxHeight, 50)

    # Define the destination points for the warp based on the calculated dimensions
    target = np.array(
        [[0, 0], [maxWidth - 1, 0], [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]],
        dtype=np.float32,
    )

    try:
        transform = cv2.getPerspectiveTransform(vertices.astype(np.float32), target)
        cropped = cv2.warpPerspective(im, transform, (maxWidth, maxHeight))
        return cropped
    except Exception as e:
        logger.error(f"Perspective transform failed: {e}, using fallback cropping")
        return fallback_crop(im)


def _is_valid_quadrilateral(vertices):
    """
    Check if vertices form a valid document quadrilateral with document-specific criteria.

    Parameters
    ----------
    vertices : np.ndarray
        4x2 array of points
    image_shape : tuple
        Shape of the original image (height, width)

    Returns
    -------
    bool
        True if quadrilateral is valid for document scanning
    """
    if vertices is None or len(vertices) != 4:
        return False

    # Check for degenerate cases (duplicate points)
    unique_points = np.unique(vertices.reshape(-1, 2), axis=0)
    if len(unique_points) < 4:
        return False

    # Calculate bounding box
    x_coords = vertices[:, 0]
    y_coords = vertices[:, 1]
    width = np.max(x_coords) - np.min(x_coords)
    height = np.max(y_coords) - np.min(y_coords)

    # Check minimum size
    min_dimension = min(width, height)
    if min_dimension < 50:  # Too small
        return False

    # Check aspect ratio (documents are usually between 0.5 and 3.0 aspect ratio)
    if width > 0 and height > 0:
        aspect_ratio = max(width, height) / min(width, height)
        if aspect_ratio > 5.0:  # Too elongated
            return False

    # Check that points are reasonably spread out
    distances = []
    for i in range(4):
        for j in range(i + 1, 4):
            dist = np.linalg.norm(vertices[i] - vertices[j])
            distances.append(dist)

    min_dist = min(distances)

    # Points should not be too close together relative to the overall size
    if min_dist < min_dimension * 0.1:
        return False

    # Check area using shoelace formula
    x, y = vertices[:, 0], vertices[:, 1]
    area = 0.5 * abs(x[0] * y[1] + x[1] * y[2] + x[2] * y[3] + x[3] * y[0] - (x[1] * y[0] + x[2] * y[1] + x[3] * y[2] + x[0] * y[3]))
    return area > 100  # Minimum area threshold


# %% [markdown]
# ## Image Enhancement


# %%
def enhance(im, block_size=21, C=10):
    """
    Enhance the input image for OCR readability.

    Parameters
    ----------
    im : np.ndarray
        Input BGR image.
    block_size : int, optional
        Size of a pixel neighborhood that is used to calculate a threshold value for the pixel.
        Must be odd and greater than 1. Default is 21.
    C : int, optional
        Constant subtracted from the mean or weighted mean. Default is 10.

    Returns
    -------
    enhanced : np.ndarray
        Enhanced binary image.
    """
    # Validate parameters
    if block_size % 2 == 0 or block_size < 3:
        block_size = 21  # Fallback to safe default
        logger.warning(f"Invalid block_size {block_size}, using default 21")

    # 1. Convert to grayscale
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

    # 2. Apply a median blur to reduce noise before thresholding
    gray = cv2.medianBlur(gray, 3)

    # 3. Use adaptive thresholding to create a clean, high-contrast binary image.
    # This is highly effective for documents with varying lighting.
    enhanced = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        block_size,  # Configurable block size
        C,  # Configurable constant
    )

    return enhanced


# %% [markdown]
# ## Result


# %%
def scan(im, logger=None):
    """
    Complete document scanning pipeline.

    Parameters
    ----------
    im : np.ndarray
        Input image
    logger : logging.Logger, optional
        Logger instance. If None, uses default logger.

    Returns
    -------
    result : np.ndarray
        Processed document image
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    # Fallback in case document is not found
    original_im = im.copy()

    try:
        grayscale = to_grayscale(im)
        blurred = blur(grayscale)
        edges = to_edges(blurred)
        vertices = find_vertices(edges)

        # If vertices are not found, return an enhanced version of the original
        if vertices is None:
            logger.warning("Document outline not found, enhancing original image.")
            # Convert original to grayscale and apply adaptive threshold
            gray_original = to_grayscale(original_im)
            return cv2.adaptiveThreshold(
                gray_original,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                21,
                10,
            )

        logger.info(f"Document detected with vertices: {vertices}")
        cropped = crop_out(im, vertices)
        enhanced_result = enhance(cropped)

        logger.info(f"Document processed successfully. Output size: {enhanced_result.shape}")
        return enhanced_result

    except Exception as e:
        logger.error(f"Error during document scanning: {e}")
        # Return enhanced original as fallback
        logger.warning("Returning enhanced original image due to processing error")
        gray_original = to_grayscale(original_im)
        return cv2.adaptiveThreshold(
            gray_original,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            21,
            10,
        )


# %%
scanned = scan(im)
imshow(scanned)

# %%
cv2.imwrite("scanned_improved2.jpg", scanned)
