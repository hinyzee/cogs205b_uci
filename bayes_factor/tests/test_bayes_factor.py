import unittest
import bayes_factor


def make_bf(n, k, spike_low=0.4999, spike_high=0.5001):
    """Test helper. Defaults are convenience for tests only,
    not production behavior."""
    return bayes_factor.BayesFactor(n, k, spike_low, spike_high)


class TestBayesFactor(unittest.TestCase):
    # set up a BayesFactor instance for testing
    def setUp(self):
        self.bf = make_bf(n=10, k=3)

    # input validation tests

    def test_constructor_stores_n_and_k(self):
        self.assertEqual(self.bf.n, 10)
        self.assertEqual(self.bf.k, 3)

    def test_likelihood_returns_float(self):
        result = self.bf.likelihood(0.5)
        self.assertIsInstance(result, float)

    def test_likelihood_at_theta_zero_with_successes_is_zero(self):
        self.assertEqual(self.bf.likelihood(0), 0)

    def test_likelihood_at_theta_one_with_failures_is_zero(self):
        self.assertEqual(self.bf.likelihood(1), 0)

    def test_likelihood_known_value(self):
        # For n=2, k=1, theta=0.5: C(2,1) * 0.5^1 * 0.5^1 = 2 * 0.25 = 0.5
        bf = make_bf(n=2, k=1)
        self.assertAlmostEqual(bf.likelihood(0.5), 0.5)

    def test_likelihood_rejects_invalid_theta(self):
        with self.assertRaises(ValueError):
            self.bf.likelihood(-0.1)
        with self.assertRaises(ValueError):
            self.bf.likelihood(1.5)

    # constructor validation

    def test_constructor_rejects_k_greater_than_n(self):
        with self.assertRaises(ValueError):
            make_bf(n=5, k=10)

    def test_constructor_rejects_negative_n(self):
        with self.assertRaises(ValueError):
            make_bf(n=-1, k=0)

    def test_constructor_rejects_negative_k(self):
        with self.assertRaises(ValueError):
            make_bf(n=10, k=-1)

    def test_constructor_rejects_non_integer_n(self):
        with self.assertRaises(TypeError):
            make_bf(n=3.5, k=1)

    def test_constructor_rejects_non_integer_k(self):
        with self.assertRaises(TypeError):
            make_bf(n=10, k="three")

    def test_constructor_accepts_k_equal_to_n(self):
        bf = make_bf(n=5, k=5)
        self.assertEqual(bf.k, 5)

    def test_constructor_accepts_zero_k(self):
        bf = make_bf(n=5, k=0)
        self.assertEqual(bf.k, 0)

    def test_constructor_rejects_invalid_spike_bounds(self):
        with self.assertRaises(ValueError):
            bayes_factor.BayesFactor(n=10, k=3, spike_low=0.6, spike_high=0.4)

    # math checks

    def test_evidence_slab_returns_float(self):
        result = self.bf.evidence_slab()
        self.assertIsInstance(result, float)

    def test_evidence_slab_is_non_negative(self):
        self.assertGreaterEqual(self.bf.evidence_slab(), 0)

    def test_evidence_slab_is_at_most_one(self):
        self.assertLessEqual(self.bf.evidence_slab(), 1)

    def test_evidence_slab_known_value(self):
        # For n=10, evidence_slab should be 1/(n+1) = 1/11
        bf = make_bf(n=10, k=3)
        self.assertAlmostEqual(bf.evidence_slab(), 1 / 11, places=6)

    def test_evidence_slab_independent_of_k(self):
        bf_a = make_bf(n=10, k=3)
        bf_b = make_bf(n=10, k=7)
        self.assertAlmostEqual(
            bf_a.evidence_slab(), bf_b.evidence_slab(), places=6
        )

    def test_evidence_spike_returns_float(self):
        result = self.bf.evidence_spike()
        self.assertIsInstance(result, float)

    def test_evidence_spike_is_non_negative(self):
        self.assertGreaterEqual(self.bf.evidence_spike(), 0)

    def test_evidence_spike_approximates_likelihood_at_midpoint(self):
        # Spike interval is so narrow that the integral averages to ~likelihood(0.5)
        bf = make_bf(n=10, k=3)
        self.assertAlmostEqual(
            bf.evidence_spike(),
            bf.likelihood(0.5),
            places=4
        )

    def test_evidence_spike_equals_slab_when_bounds_are_full(self):
        # If spike prior covers [0,1] like the slab, evidences match
        bf = bayes_factor.BayesFactor(
            n=10, k=3, spike_low=0.0, spike_high=1.0
        )
        self.assertAlmostEqual(bf.evidence_spike(), bf.evidence_slab(), places=6)

    # bayes factor

    def test_bayes_factor_returns_float(self):
        result = self.bf.bayes_factor()
        self.assertIsInstance(result, float)

    def test_bayes_factor_is_non_negative(self):
        self.assertGreaterEqual(self.bf.bayes_factor(), 0)

    def test_bayes_factor_equals_one_when_priors_match(self):
        # If spike prior covers [0,1] like the slab, the BF must be 1
        bf = bayes_factor.BayesFactor(
            n=10, k=3, spike_low=0.0, spike_high=1.0
        )
        self.assertAlmostEqual(bf.bayes_factor(), 1.0, places=6)

    def test_bayes_factor_favors_spike_for_balanced_data(self):
        # k = n/2 is exactly what spike (theta ~ 0.5) predicts
        bf = make_bf(n=10, k=5)
        self.assertGreater(bf.bayes_factor(), 1.0)

    def test_bayes_factor_favors_slab_for_extreme_data(self):
        # k = 0 is terrible evidence for theta ~ 0.5
        bf = make_bf(n=10, k=0)
        self.assertLess(bf.bayes_factor(), 1.0)

    # intentional failure check 

    @unittest.expectedFailure
    def test_intentional_failure_placeholder(self):
        bf = make_bf(n=10, k=5)
        self.assertEqual(bf.bayes_factor(), 0.0)


if __name__ == '__main__':
    unittest.main()