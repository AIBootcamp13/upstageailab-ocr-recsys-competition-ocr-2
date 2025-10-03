## **Strategy: Pseudo-Labeling with an Open-Source Model**

This document outlines a cost-effective strategy to label your 3,000 receipt images using a locally-run, open-source OCR model. This process, known as pseudo-labeling, will generate initial bounding box annotations that you can then refine, saving significant time and money compared to manual labeling or using a paid API.

### **Why This Approach?**

* **Cost-Effective:** It's free. You leverage your existing high-end GPU instead of paying per-page API fees (saving you an estimated $30 for 3,000 images).
* **Fast:** Generating labels for 3,000 images will take minutes to a few hours on a good GPU, compared to many days of manual work.
* **High-Quality Start:** Modern open-source models are highly accurate. The generated labels will be a very strong starting point, often requiring only minor corrections.

### **Recommended Open-Source Tools**

While your own DBNet model could be used if well-trained, for generating initial labels it's best to use a model pre-trained on a massive, diverse dataset. Here are the top recommendations:

1. **doctr (Recommended for this task):** A user-friendly and powerful library by Mindee. It offers pre-trained models that are easy to download and run with just a few lines of code. We will use this in the example script.
2. **MMOCR:** A comprehensive OCR toolbox from the creators of PyTorch. It's more complex but contains a huge variety of state-of-the-art models (DBNet++, PANet, etc.).
3. **PaddleOCR:** An excellent and very popular OCR toolkit from Baidu with great performance, especially on multilingual documents.

### **The Pseudo-Labeling Workflow**

Here is the step-by-step process you will follow:

Step 1: Setup the Environment
Install the doctr library. Since you are using uv, you can add it to your requirements.txt or install it directly in your virtual environment.
\# Add this to your requirements.txt
\# tensorflow-cpu \# or tensorflow for GPU
\# python-doctr\[tf\]

\# Then sync with uv
uv sync

*Note: doctr can use either TensorFlow or PyTorch as a backend. The example script will use TensorFlow.*

Step 2: Run the Generation Script
Use the provided generate\_labels.py script. You will need to point it to the directory containing your 3,000 unlabeled receipt images. The script will:

1. Load a pre-trained text detection model from doctr.
2. Iterate through each image in your folder.
3. Run the model to get bounding box predictions.
4. Convert the predictions into the exact JSON format your OCRDataset class expects.
5. Save all the labels into a single train.json file.

Step 3: (Highly Recommended) Review and Refine
The generated labels will not be perfect. It's crucial to perform a quick review to fix errors. This is called a "human-in-the-loop" approach.

* **Tool:** Use a free annotation tool like **Label Studio** or **CVAT**.
* **Process:**
  1. Import your images and the generated train.json file into the tool.
  2. Quickly scan through each image.
  3. Correct any missed detections or inaccurate boxes.
* This review process is **much faster** than labeling from scratch. You are only correcting mistakes, not creating every single box.

Step 4: Train Your Model
Once you have the refined train.json, you can use it to train your own DBNet model just as you would with a manually labeled dataset. The larger, higher-quality dataset will lead to a much better-performing final model.
