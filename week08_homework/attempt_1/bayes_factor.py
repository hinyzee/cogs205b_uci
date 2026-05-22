import numpy as np
from scipy.special import comb
from scipy.integrate import quad

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
        """
        Marginal likelihood under slab prior theta ~ Uniform(0, 1).
        The integral of binomial(n, k, theta) from 0 to 1 is 1/(n+1).
        """
        return 1.0 / (self.n + 1)

    def evidence_spike(self):
        """
        Marginal likelihood under spike prior theta ~ Uniform(a, b).
        Prior density is 1/(b-a) over [a, b], 0 elsewhere.
        """
        a = 0.47
        b = 0.53
        # Integrate likelihood * density
        # Integral of (likelihood * (1/(b-a))) over [a, b]
        result, _ = quad(self.likelihood, a, b)
        return result / (b - a)

    def bayes_factor(self):
        """
        Ratio of evidence_spike / evidence_slab.
        """
        slab = self.evidence_slab()
        spike = self.evidence_spike()
        return spike / slab
