# TurboQuant: Near-Optimal Vector Quantization

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![NumPy](https://img.shields.io/badge/dependency-NumPy-blueviolet.svg)](https://numpy.org/)

**TurboQuant** is a high-performance Python library for online vector quantization, based on the research paper: **"TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate"** ([arXiv:2504.19874](https://arxiv.org/pdf/2504.19874)).

It provides a fast, memory-efficient way to compress high-dimensional vectors (such as LLM embeddings) while maintaining near-optimal distortion rates and ensuring unbiased inner products.

---

##  Key Features

*   **Random Rotation (QR-based):** Employs orthogonal matrices to redistribute vector energy, making the data more amenable to scalar quantization.
*   **Two-Stage Quantization:** 
    1.  **Base Stage:** Fast scalar quantization with dynamic scaling.
    2.  **Residual Stage:** 1-bit bias correction (Sign-based) to minimize errors in downstream tasks like Similarity Search.
*   **Smart Bit-Rate Selection:** Automatically determines the minimum bits (2–12) required to meet a specific Mean Squared Error (MSE) target.
*   **Highly Optimized:** Pure NumPy implementation using direct arithmetic operations ($O(d)$ complexity) instead of slow distance-based codebook searches.

---

##  Installation

You can install TurboQuant directly from the source:

```bash
git clone https://github.com/FireMoonStudio/TurboQuant.py.git
```
## Quick Start


```
import numpy as np
from turboquant import TurboQuant

# 1. Initialize TurboQuant with residual correction enabled
tq = TurboQuant(use_residual=True)

# 2. Prepare high-dimensional data (e.g., 1024-d vector)
data = np.random.randn(1024)

# 3. Compress with a target MSE threshold
# Returns: Quantized base, 1-bit residuals, and the number of bits used
base_q, res_q, bits = tq.smart_compress(data, max_error=0.001)

# 4. Decompress to recover the vector
recovered = tq.decompress(base_q, res_q)

# 5. Evaluate results
mse = np.mean((data - recovered)**2)
print(f"Compressed using {bits} bits (+ 1-bit residual correction)")
print(f"Reconstruction MSE: {mse:.10f}")
```
## Scientific Background 

TurboQuant bridges the gap between simple scalar quantization and complex vector quantization. By rotating a vector x using a random orthogonal matrix Q:


$$
y = Qx
$$

The components of y follow a distribution that is easier to quantize optimally. To prevent bias during dot product operations (crucial for Transformer Attention mechanisms), TurboQuant stores the sign of the residual error:

$$
\hat{y}_{\text{final}} = \text{Quantized}(y) + \text{Sign}(y - \hat{y}) \cdot \sigma_{\text{error}}
$$


This ensures that the expected distortion is minimized and the inner product estimation remains unbiased.

## License
This project is licensed under the MIT License - see the LICENSE file for details.

```@article{turboquant2026,
  title={TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate},
  author={ FireMoonStudio},
  journal={arXiv preprint arXiv:2504.19874},
  year={2026}
}
```
