# **filename: docs/ai_handbook/03_references/coding/type_checking.md**
<!-- ai_cue:priority=high -->
<!-- ai_cue:use_when=type_checking,mypy,pylance,typing -->

# **Reference: Type Checking in the OCR Project**

This reference document provides comprehensive information about type checking practices, tools, and patterns used in the OCR project for quick lookup and detailed understanding.

## **Overview**

Type checking is a critical development practice in this project that helps prevent runtime errors by catching type inconsistencies at development time. The project uses both static type checking (mypy/Pylance) and gradual typing to balance code safety with development velocity.

## **Key Concepts**

### **Static Type Checking**
Using tools like mypy or Pylance to analyze code for type consistency without running it, identifying potential runtime errors before execution.

### **Gradual Typing**
A type system that allows mixing typed and untyped code, enabling incremental adoption of type annotations throughout the codebase.

### **Union Types**
Type annotations that allow a variable to be one of several possible types, such as `Union[PIL.Image, np.ndarray]` or `PIL.Image | np.ndarray`.

### **Type Guards**
Runtime checks (like `isinstance`) that narrow the type of a variable within a specific code block, improving type safety.

## **Detailed Information**

### **Type Checking Tools Configuration**
The project uses Pylance (integrated in VS Code) and mypy for static type analysis. Configuration is typically found in `pyproject.toml` or `.mypy.ini`.

### **Common Type Patterns**
- **Union Types**: Use for polymorphic variables that can be different types
- **Type Aliases**: Define reusable type definitions for complex types
- **Generic Types**: Use TypeVar for reusable type-safe utilities
- **Optional Types**: Use `Optional[T]` or `T | None` for values that might be None

### **Type Checking Benefits**
- **Early Bug Detection**: Catch type inconsistencies before runtime
- **Code Documentation**: Type hints serve as documentation for function interfaces
- **IDE Support**: Better autocomplete and refactoring support
- **Refactoring Safety**: Type checker prevents breaking changes during refactoring

## **Examples**

### **Basic Type Annotations**
```python
from typing import Union
from PIL import Image
import numpy as np

def get_image_shape(image: Union[Image.Image, np.ndarray]) -> tuple[int, int]:
    """Type-safe shape extraction with union type handling."""
    if isinstance(image, np.ndarray):
        height, width, *_ = image.shape
        return (width, height)
    else:  # PIL.Image
        return image.size
```

### **Type Guards for Safety**
```python
def process_image(image: Union[Image.Image, np.ndarray]) -> ProcessedImage:
    """Process image with type-safe handling."""
    if isinstance(image, np.ndarray):
        # Here, type checker knows image is np.ndarray
        return process_numpy_image(image)
    else:
        # Here, type checker knows image is PIL.Image
        return process_pil_image(image)
```

### **Generic Type Variables**
```python
from typing import TypeVar

T = TypeVar('T', bound=Image.Image)

def transform_image(image: T) -> T:
    """Transform that preserves the input type."""
    # Transform logic here
    return transformed_image
```

## **Configuration Options**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `python.analysis.typeCheckingMode` | str | "basic" | Type checking strictness level in Pylance |
| `mypy.strict` | bool | false | Enable all optional error checking in mypy |
| `mypy.disallow_untyped_defs` | bool | false | Require all function definitions to be typed |

## **Best Practices**

- **Use Union Types**: For polymorphic variables like `PIL.Image | np.ndarray`
- **Implement Type Guards**: Use `isinstance` checks to narrow types safely
- **Add Type Hints**: To critical pipeline functions to prevent runtime errors
- **Test Type Combinations**: Explicitly test different type scenarios
- **Enable Strict Checking**: Use strict mode in development for maximum safety
- **Document Type Contracts**: Between pipeline stages to ensure consistency

## **Troubleshooting**

### **Common Issues**
- **Too Many Type Errors**: Start with basic type checking and gradually increase strictness
- **Complex Type Signatures**: Break down complex functions or use type aliases
- **Third-party Library Types**: Use type stubs or `# type: ignore` comments when necessary
- **Performance Impact**: Type checking only occurs at development time, not runtime

### **Debug Information**
- Enable Pylance in VS Code for real-time type checking
- Run mypy regularly: `mypy src/`
- Check type errors before committing code
- Use `reveal_type()` in mypy to debug inferred types

## **Related References**

- `docs/ai_handbook/02_protocols/development/01_coding_standards.md` - Coding standards and type hinting practices
- `docs/ai_handbook/03_references/architecture/01_architecture.md` - System architecture with type interfaces
- `docs/ai_handbook/03_references/guides/performance_profiler_usage.md` - Performance considerations with type checking

---

*This document follows the references template. Last updated: October 13, 2025*
