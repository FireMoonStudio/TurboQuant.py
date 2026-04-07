# TurboQuant: Near-Optimal Vector Quantization

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![NumPy](https://img.shields.io/badge/dependency-NumPy-blueviolet.svg)](https://numpy.org/)

**TurboQuant** is a high-performance Python library for vector quantization based on the research paper [arXiv:2504.19874](https://arxiv.org/pdf/2504.19874). It provides a fast, memory-efficient way to compress high-dimensional vectors (like LLM embeddings) while maintaining near-optimal distortion rates and unbiased inner products.

---

## 🚀 Key Features

*   **Random Rotation Transformation:** Uses orthogonal matrices (via QR decomposition) to redistribute vector energy, making the data more "quantization-friendly."
*   **Two-Stage Quantization:** 
    1.  **Base Stage:** Efficient scalar quantization with dynamic scaling.
    2.  **Residual Stage:** 1-bit bias correction (Sign-based) to ensure accuracy in downstream tasks.
*   **Smart Compression:** Automatically selects the minimum bit-rate (from 2 to 12 bits) required to meet your target Mean Squared Error (MSE).
*   **Zero Codebook Search:** Uses direct arithmetic operations ($O(1)$ per element) instead of slow distance-based searches.

---

## 🛠 Installation


# 1. Initialize TurboQuant
tq = TurboQuant(use_residual=True)

# 2. Prepare your data (e.g., a 1024-dimensional embedding)
data = np.random.randn(1024)

# 3. Compress with a target error threshold
base_q, res_q, bits = tq.smart_compress(data, max_error=0.001)

# 4. Decompress to recover the original vector
recovered = tq.decompress(base_q, res_q)

# 5. Evaluate
mse = np.mean((data - recovered)**2)
print(f"Compressed to {bits} bits (+ 1-bit residual)")
print(f"Mean Squared Error: {mse:.8f}")
🔬 Mathematical OverviewTurboQuant achieves near-optimal performance by rotating the input vector $x$ using a random orthogonal matrix $Q$:$$y = Qx$$In this rotated domain, the vector components tend to follow a predictable distribution, which allows simple scalar quantization to perform as well as complex vector quantization methods.To maintain an unbiased inner product, TurboQuant stores a 1-bit sign of the quantization error (residual):$$\hat{y}_{final} = \text{Quantized}(y) + \text{Sign}(y - \hat{y}) \cdot \mu_{\text{error}}$$📄 LicenseThis project is licensed under the MIT License.🔗 CitationIf you use this library in your research, please cite:مقتطف الرمز@article{turboquant2026,
  title={TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate},
  author={Your Name},
  journal={arXiv preprint arXiv:2504.19874},
  year={2026}
}
