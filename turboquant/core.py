import numpy as np

class TurboQuant:
    def __init__(self, bits=8, use_residual=True):
        self.bits = bits
        self.use_residual = use_residual 
        self.rotation_matrix = None
        self.stats = {} 

    def _get_rotation(self, d):
       
        if self.rotation_matrix is None or self.rotation_matrix.shape[0] != d:
            H = np.random.randn(d, d)
            q, _ = np.linalg.qr(H)
            self.rotation_matrix = q
        return self.rotation_matrix

    def compress(self, x):
        x = np.array(x)
        d = len(x)
        Q = self._get_rotation(d)
        
    
        x_rotated = x @ Q
        
       
        v_min, v_max = x_rotated.min(), x_rotated.max()
        diff = v_max - v_min if v_max > v_min else 1e-10
        
        
        x_norm = (x_rotated - v_min) / diff
        levels = (2**self.bits) - 1
        
        
        x_quant_norm = np.round(x_norm * levels) / levels
        

        x_quant_rot = (x_quant_norm * diff) + v_min
        
        residual_quant = None
        residual_scale = 0
        
        if self.use_residual:
        
            residual = x_rotated - x_quant_rot
            
            residual_scale = np.mean(np.abs(residual))
            residual_quant = np.sign(residual) 
            
        self.stats = {
            'min': v_min, 
            'diff': diff, 
            'res_scale': residual_scale
        }
        
        return x_quant_norm, residual_quant

    def decompress(self, x_quant_norm, residual_quant=None):
        v_min = self.stats['min']
        diff = self.stats['diff']
        res_scale = self.stats['res_scale']
        Q = self.rotation_matrix
        
       
        x_rot_approx = (x_quant_norm * diff) + v_min
        
       
        if self.use_residual and residual_quant is not None:
            x_rot_approx += (residual_quant * res_scale)
         

        return x_rot_approx @ Q.T

    def smart_compress(self, vector, max_error=0.01):
        best_base = None
        best_res = None
        chosen_bits = 2
        
        for b in range(2, 13): 
            self.bits = b
            base_q, res_q = self.compress(vector)
            recovered = self.decompress(base_q, res_q)
            
           
            error = np.mean((np.array(vector) - recovered)**2)
            
           
            if best_base is None or error <= max_error:
                best_base = base_q
                best_res = res_q
                chosen_bits = b

            if error <= max_error:
                print(f"Accuracy was achieved with {b} bits| MSE: {error:.6f}")
                break
        else:
            print(f"Maximum endurance (12-bit)| MSE: {error:.6f}")
                
        return best_base, best_res, chosen_bits
