# **Concept:**

**Objective:** Explain the core idea, the "why" behind a specific architectural choice, algorithm, or design pattern in this project.

### **1. What is**

?

*(Provide a high-level, one-paragraph summary of the concept. For example, "A Component Registry is a design pattern that allows us to dynamically discover and instantiate different parts of our model (like encoders or decoders) from a central catalog.")*

### **2. Why Do We Use It? (The Problem It Solves)**

*(Describe the problem this concept solves for our project. What would be difficult, inefficient, or impossible without it?)*

* **Problem 1:** *(e.g., "Hardcoding model architectures makes experimentation slow and error-prone.")*
* **Problem 2:** *(e.g., "Switching between a ResNet and a VGG backbone required significant code changes.")*
* **Our Solution:** *(e.g., "By using a registry, we can switch backbones by changing a single line in a configuration file, enabling rapid A/B testing.")*

### **3. How Does It Work in Our Codebase?**

*(Provide specific file paths and code snippets to show the concept in action.)*

* **Key File(s):**
  * ocr/models/core/registry.py (The definition)
  * ocr/models/architectures/dbnet.py (An example of registration)
  * ocr/models/architecture.py (Where it's used to build a model)
* **Example Code Snippet (Registration):**
  ``` # In ocr/models/architectures/dbnet.py
  registry.register_encoder("timm_backbone", TimmBackbone)

  ```

* **Example Code Snippet (Usage):**
  ``` # In ocr/models/architecture.py
  encoder_class = registry.get_encoder("timm_backbone")
  self.encoder = encoder_class(**config)
  ```

### **4. Trade-offs and Considerations**

*(Discuss any downsides or things to be aware of when working with this concept.)*

* **Pro:** *(e.g., "Extreme flexibility for experimentation.")*
* **Con:** *(e.g., "Can make it harder to trace code flow without understanding the configuration.")*
* **Consideration:** *(e.g., "When adding a new component, you MUST remember to register it, or the framework won't be able to find it.")*
