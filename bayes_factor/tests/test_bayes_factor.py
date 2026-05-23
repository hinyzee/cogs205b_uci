import unittest
import bayes_factor


class TestBayesFactor(unittest.TestCase):
    # set up a BayesFactor instance for testing
    def setUp(self):
        self.bf = bayes_factor.BayesFactor(n=10, k=3)

    # constructor validation
    def test_constructor_stores_n_and_k(self):
        self.assertEqual(self.bf.n, 10)
        self.assertEqual(self.bf.k, 3)

    def test_constructor_accepts_required_api(self):
        bf = bayes_factor.BayesFactor(10, 3)
        self.assertEqual(bf.n, 10)
        self.assertEqual(bf.k, 3)

    def test_constructor_rejects_k_greater_than_n(self):
        with self.assertRaisesRegex(ValueError, "k cannot exceed n"):
            bayes_factor.BayesFactor(n=5, k=10)

    def test_constructor_rejects_negative_n(self):
        with self.assertRaisesRegex(ValueError, "n must be non-negative"):
            bayes_factor.BayesFactor(n=-1, k=0)

    def test_constructor_rejects_negative_k(self):
        with self.assertRaisesRegex(ValueError, "k must be non-negative"):
            bayes_factor.BayesFactor(n=10, k=-1)

    def test_constructor_rejects_non_integer_n(self):
        with self.assertRaisesRegex(TypeError, "n must be an integer"):
            bayes_factor.BayesFactor(n=3.5, k=1)

    def test_constructor_rejects_non_integer_k(self):
        with self.assertRaisesRegex(TypeError, "k must be an integer"):
            bayes_factor.BayesFactor(n=10, k="three")

    def test_constructor_accepts_k_equal_to_n(self):
        bf = bayes_factor.BayesFactor(n=5, k=5)
        self.assertEqual(bf.k, 5)

    def test_constructor_accepts_zero_k(self):
        bf = bayes_factor.BayesFactor(n=5, k=0)
        self.assertEqual(bf.k, 0)

    def test_constructor_accepts_zero_trials(self):
        bf = bayes_factor.BayesFactor(n=0, k=0)
        self.assertEqual(bf.n, 0)
        self.assertEqual(bf.k, 0)

    def test_likelihood_returns_float(self):
        result = self.bf.likelihood(0.5)
        self.assertIsInstance(result, float)

    def test_likelihood_at_theta_zero_with_successes_is_zero(self):
        self.assertEqual(self.bf.likelihood(0), 0)

    def test_likelihood_at_theta_one_with_failures_is_zero(self):
        self.assertEqual(self.bf.likelihood(1), 0)

    def test_likelihood_known_value(self):
        # For n=2, k=1, theta=0.5: C(2,1) * 0.5^1 * 0.5^1 = 2 * 0.25 = 0.5
        bf = bayes_factor.BayesFactor(n=2, k=1)
        self.assertAlmostEqual(bf.likelihood(0.5), 0.5)

    def test_likelihood_rejects_invalid_theta(self):
        with self.assertRaisesRegex(ValueError, r"theta must be in \[0, 1\]"):
            self.bf.likelihood(-0.1)
        with self.assertRaisesRegex(ValueError, r"theta must be in \[0, 1\]"):
            self.bf.likelihood(1.5)

    def test_likelihood_rejects_non_numeric_theta(self):
        with self.assertRaisesRegex(TypeError, "theta must be numeric"):
            self.bf.likelihood("0.5")


    def test_required_methods_exist_and_are_callable(self):
        for method_name in ["likelihood", "evidence_slab", "evidence_spike", "bayes_factor"]:
            self.assertTrue(hasattr(self.bf, method_name))
            self.assertTrue(callable(getattr(self.bf, method_name)))

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
        bf = bayes_factor.BayesFactor(n=10, k=3)
        self.assertAlmostEqual(bf.evidence_slab(), 1 / 11, places=6)

    def test_evidence_slab_independent_of_k(self):
        bf_a = bayes_factor.BayesFactor(n=10, k=3)
        bf_b = bayes_factor.BayesFactor(n=10, k=7)
        self.assertAlmostEqual(
            bf_a.evidence_slab(), bf_b.evidence_slab(), places=6
        )

    def test_evidence_spike_returns_float(self):
        result = self.bf.evidence_spike()
        self.assertIsInstance(result, float)

    def test_evidence_spike_is_non_negative(self):
        self.assertGreaterEqual(self.bf.evidence_spike(), 0)

    def test_evidence_spike_is_interval_average(self):
        bf = bayes_factor.BayesFactor(n=10, k=3)
        low = 0.4999
        high = 0.5001
        result, _ = scipy.integrate.quad(bf.likelihood, low, high)
        expected = result / (high - low)

        self.assertAlmostEqual(
            bf.evidence_spike(),
            expected,
            places=6
        )

    # bayes factor

    def test_bayes_factor_returns_float(self):
        result = self.bf.bayes_factor()
        self.assertIsInstance(result, float)

    def test_bayes_factor_is_non_negative(self):
        self.assertGreaterEqual(self.bf.bayes_factor(), 0)

    def test_bayes_factor_matches_evidence_ratio(self):
        expected = self.bf.evidence_spike() / self.bf.evidence_slab()
        self.assertAlmostEqual(self.bf.bayes_factor(), expected, places=6)

    def test_bayes_factor_favors_spike_for_balanced_data(self):
        # k = n/2 is exactly what the narrow spike around theta = 0.5 predicts
        bf = bayes_factor.BayesFactor(n=10, k=5)
        self.assertGreater(bf.bayes_factor(), 1.0)

    def test_bayes_factor_favors_slab_for_extreme_data(self):
        bf = bayes_factor.BayesFactor(n=10, k=0)
        self.assertLess(bf.bayes_factor(), 1.0)

    def test_bayes_factor_handles_zero_trials(self):
        bf = bayes_factor.BayesFactor(n=0, k=0)
        self.assertAlmostEqual(bf.bayes_factor(), 1.0)

    # intentional failure check 

    @unittest.expectedFailure
    def test_intentional_failure_placeholder(self):
        bf = bayes_factor.BayesFactor(n=10, k=5)
        self.assertEqual(bf.bayes_factor(), 0.0)


if __name__ == '__main__':
    unittest.main()