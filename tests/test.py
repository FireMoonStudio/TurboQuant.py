import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from turboquant.core import TurboQuant

def test_quantization_logic():
    tq = TurboQuant(use_residual=True)
    
    data = np.random.randn(128) 
    
    
    
    base_q, res_q, bits = tq.smart_compress(data, max_error=0.001)
    
    recovered = tq.decompress(base_q, res_q)
    mse = np.mean((data - recovered)**2)
    
    assert mse < 0.005, f"Test failed: Error {mse} is too high!"
    
    print(f"Test passed!")
    print(f" MSE: {mse:.10f}")
    print(f" Bits: {bits}")

if __name__ == "__main__":
    test_quantization_logic()