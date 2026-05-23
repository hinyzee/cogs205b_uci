import numpy as np
from scipy import integrate
from scipy.special import comb

class BayesFactor:
    def __init__(self, n, k):
        if not isinstance(n, int):
            raise TypeError("n must be an integer")
        if not isinstance(k, int):
            raise TypeError("k must be an integer")
        if n < 0:
            raise ValueError("n must be non-negative")
        if k < 0:
            raise ValueError("k must be non-negative")
        if k > n:
            raise ValueError("k cannot exceed n")
        
        self.n = n
        self.k = k

    def likelihood(self, theta):
        if not isinstance(theta, (int, float, np.number)):
            raise TypeError("theta must be numeric")
        if not (0 <= theta <= 1):
            raise ValueError("theta must be in [0, 1]")
        
        # Binomial likelihood: C(n, k) * theta^k * (1-theta)^(n-k)
        return float(comb(self.n, self.k) * (theta**self.k) * ((1 - theta)**(self.n - self.k)))

    def evidence_slab(self):
        # Slab prior is uniform over [0, 1].
        # Marginal likelihood = integral from 0 to 1 of likelihood(theta) * 1 d_theta.
        # The integral of C(n, k) * theta^k * (1-theta)^(n-k) from 0 to 1 is 1 / (n + 1).
        return 1.0 / (self.n + 1)

    def evidence_spike(self):
        # Spike prior is uniform over [0.47, 0.53].
        # Width = 0.06
        low = 0.47
        high = 0.53
        width = high - low
        
        # Marginal likelihood = (1/width) * integral from low to high of likelihood(theta) d_theta
        result, _ = integrate.quad(self.likelihood, low, high)
        return float(result / width)

    def bayes_factor(self):
        e_spike = self.evidence_spike()
        e_slab = self.evidence_slab()
        
        # BF = P(D | H_spike) / P(D | H_slab)
        return e_spike / e_slab
