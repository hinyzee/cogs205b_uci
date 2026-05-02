import scipy.integrate
from scipy.stats import binom

class BayesFactor:

    def __init__(self, n, k, spike_low, spike_high):
        if not isinstance(n, int) or isinstance(n, bool):
            raise TypeError(f"n must be an integer, got {type(n).__name__}")
        if not isinstance(k, int) or isinstance(k, bool):
            raise TypeError(f"k must be an integer, got {type(k).__name__}")
        if n < 0:
            raise ValueError(f"n must be non-negative, got {n}")
        if k < 0:
            raise ValueError(f"k must be non-negative, got {k}")
        if k > n:
            raise ValueError(f"k cannot exceed n, got k={k}, n={n}")
        
        if not (0 <= spike_low < spike_high <= 1):
            raise ValueError(
                f"spike_low and spike_high must satisfy 0 <= spike_low < spike_high <= 1,"
                f" got spike_low={spike_low}, spike_high={spike_high}"
                )   

        self.n = n
        self.k = k
        self.spike_low = spike_low
        self.spike_high = spike_high

    def likelihood(self, theta):
        if not 0 <= theta <= 1:
            raise ValueError(f"theta must be in [0, 1], got {theta}")
        return float(binom.pmf(self.k, self.n, theta))

    def evidence_slab(self):
        result, _ = scipy.integrate.quad(self.likelihood, 0, 1)        
        return float(result)

    def evidence_spike(self):
        width = self.spike_high - self.spike_low
        result, _ = scipy.integrate.quad(
            self.likelihood, self.spike_low, self.spike_high
        )
        return float(result / width)

    def bayes_factor(self):
        return self.evidence_spike() / self.evidence_slab()
    
