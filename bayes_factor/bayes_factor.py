import scipy.integrate
from math import comb


class BayesFactor:
    """
    Bayes factor for binomial data.

    Slab model:
        theta ~ U(0, 1)

    Spike model:
        theta = 0.5

    The spike is treated as a point-spike model rather than an arbitrary
    narrow interval such as [0.4999, 0.5001].
    """

    def __init__(self, n, k):
        if not isinstance(n, int) or isinstance(n, bool):
            raise TypeError("n must be an integer")
        if not isinstance(k, int) or isinstance(k, bool):
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
        if not isinstance(theta, (int, float)) or isinstance(theta, bool):
            raise TypeError("theta must be numeric")
        if theta < 0 or theta > 1:
            raise ValueError("theta must be in [0, 1]")

        return float(
            comb(self.n, self.k)
            * theta ** self.k
            * (1 - theta) ** (self.n - self.k)
        )

    def evidence_slab(self):
        result, _ = scipy.integrate.quad(self.likelihood, 0, 1)
        return float(result)

    def evidence_spike(self):
        return self.likelihood(0.5)

    def bayes_factor(self):
        return self.evidence_spike() / self.evidence_slab()