import numpy as np
from scipy.stats import norm

def get_ei(pred, eta):
    def calculate_f():
        z = (eta - m) / s
        
        return (eta - m) * norm.cdf(z) + s * norm.pdf(z)
    
    pred = np.array(pred).transpose(1, 0)
    m = np.mean(pred, axis=1)
    s = np.std(pred, axis=1)
    if np.any(s == 0.0):
        s_copy = np.copy(s)
        s[s_copy == 0.0] = 1.0
        f = calculate_f()
        f[s_copy == 0.0] = 0.0
    else:
        f = calculate_f()

    return f


