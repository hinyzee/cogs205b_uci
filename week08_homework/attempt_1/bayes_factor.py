import numpy as np
from scipy.integrate import quad
from scipy.special import comb

class BayesFactor:
    """
    A class for binomial spike-and-slab Bayes factor analysis.
    """
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
        """
        Calculates the binomial likelihood for k successes in n trials at success probability theta.
        """
        if not isinstance(theta, (int, float, np.number)):
            raise TypeError("theta must be numeric")
        if not (0 <= theta <= 1):
            raise ValueError("theta must be in [0, 1]")
        
        return float(comb(self.n, self.k) * (theta**self.k) * ((1 - theta)**(self.n - self.k)))

    def evidence_slab(self):
        """
        Calculates the marginal likelihood under a slab prior (uniform over [0, 1]).
        The integral of the binomial likelihood over [0, 1] is 1/(n+1).
        """
        # Marginal likelihood = Integral(Likelihood(theta) * Prior(theta) dtheta)
        # Prior(theta) = 1 for theta in [0, 1]
        # Integral of comb(n, k) * theta^k * (1-theta)^(n-k) from 0 to 1 is 1/(n+1).
        return 1.0 / (self.n + 1)

    def evidence_spike(self):
        """
        Calculates the marginal likelihood under a spike prior (uniform over [0.47, 0.53]).
        The width is 0.06.
        """
        low = 0.47
        high = 0.53
        # Prior(theta) = 1/width = 1/0.06 for theta in [0.47, 0.53], else 0.
        # Marginal likelihood = (1/0.06) * Integral(Likelihood(theta) dtheta from 0.47 to 0.53)
        res, _ = quad(self.likelihood, low, high)
        return float(res / 0.06)

    def bayes_factor(self):
        """
        Calculates the Bayes factor comparing the spike hypothesis to the slab hypothesis.
        """
        # BF = Evidence(Spike) / Evidence(Slab)
        ev_spike = self.evidence_spike()
        ev_slab = self.evidence_slab()
        
        # Handle edge case: n=0, k=0
        if self.n == 0:
            return 1.0
        
        return float(ev_spike / ev_slab)
