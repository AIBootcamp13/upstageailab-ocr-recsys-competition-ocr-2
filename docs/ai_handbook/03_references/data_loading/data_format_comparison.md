
## Data Format Comparison for Optimized Data Loading in OCR and Recommendation Systems

This reference document explores strategies for optimizing data loading pipelines by converting JPEG images to faster on-disk formats, focusing on reducing CPU bottlenecks during training. It compares options from simplest to most scalable, emphasizing the trade-off between disk space and performance.

### Key Options:
- **Uncompressed NumPy Arrays** (`.npy`): Eliminates JPEG decoding by saving raw pixel data, offering significant speed gains at the cost of increased disk space (e.g., 12x for a 640x640x3 image).
- **Lossless Compressed Formats** (e.g., `PNG`): Provides a middle-ground with faster decoding than `JPEG` but still involves a decompression step, resulting in moderate space savings.
- S**pecialized ML Data Formats** (e.g., `WebDataset`): Ideal for large-scale projects, packing data into archives for efficient sequential I/O, requiring pipeline changes.

Recommendation: For datasets of ~4400 images, convert to NumPy arrays and integrate into preprocessing scripts (e.g., saving as .npz with image and maps). Update dataset loaders to load directly from these files, bypassing image decoding during training for substantial performance boosts.


-----

## 1. Uncompressed NumPy Arrays (`.npy`)

This is the most direct and often the most effective solution for smaller datasets.

  * **What it is:** You decode the JPEG *once* and save the raw pixel data (the NumPy array) directly to disk as a binary `.npy` file.
  * **Why it's fast:** It completely **eliminates the JPEG decoding step** from your training loop. Loading a `.npy` file is a direct memory-map operation that is incredibly fast. The data is read from disk and is immediately ready as a NumPy array.
  * **Downside:** It uses a lot more disk space. A 100 KB JPEG might become a 1.2 MB `.npy` file (`640x640x3, uint8`), a 12x increase. For your dataset of ~4400 images, this is manageable (~5.3 GB).

-----

## 2. Lossless Compressed Formats (e.g., PNG)

This is a good middle-ground option if disk space is a concern.

  * **What it is:** Convert your JPEGs to a lossless format like PNG.
  * **Why it's fast:** PNG uses a simpler decompression algorithm (DEFLATE) that is generally **faster for the CPU to decode** than JPEG's more complex algorithm. You also avoid JPEG compression artifacts.
  * **Downside:** PNG files will be larger than your original JPEGs, but smaller than uncompressed `.npy` files. The performance gain is not as significant as with `.npy` because you still have a decoding step.

-----

## 3. Specialized ML Data Formats (e.g., WebDataset)

This is the architectural solution for large-scale projects.

  * **What it is:** Instead of thousands of small files, you pack your images (in any format—JPEG, PNG, or even `.npy`) and their labels into a few large archive files (e.g., `.tar`).
  * **Why it's fast:** This approach solves the **I/O bottleneck** of opening and closing thousands of individual files. It allows the data loader to perform fast, sequential reads from a single large file, which is highly efficient.
  * **Downside:** It requires a more significant change to your data loading pipeline, as you'd replace your `OCRDataset` with a `webdataset.WebDataset`.

-----

## Recommendation and Action Plan

For your dataset size (~4400 images), the trade-off is clear: the significant performance gain from eliminating JPEG decoding is worth the extra disk space.

**Your best option is to convert the images to NumPy arrays.**

The most efficient way to do this is to integrate it into the `preprocess_maps.py` script you're already building. Instead of just saving the maps, you can save the image array and the maps together in a single, compressed `.npz` file.

### **Action Plan**

Modify your `scripts/preprocess_maps.py` script to save the image along with the maps.

```python
# Inside your preprocess_single_item function in scripts/preprocess_maps.py

# ... after loading and normalizing the image (but BEFORE augmenting) ...
# 'image_np' should be your raw image as a NumPy array (H, W, C)
image_np = sample['image']
maps = collate_fn.make_prob_thresh_map(...)

# --- MODIFICATION HERE ---
# Instead of saving just the maps, save the image array too.
output_filename = output_dir / f"{Path(sample['image_filename']).stem}.npz"
np.savez_compressed(
    output_filename,
    image=image_np,         # <-- Add the image array
    prob_map=maps["prob_map"],
    thresh_map=maps["thresh_map"]
)
```

Then, update your `OCRDataset`'s `__getitem__` method to load the image directly from the `.npz` file instead of from the original JPEG file. This completely bypasses PIL/OpenCV for image loading during training, giving you a significant speed boost.
