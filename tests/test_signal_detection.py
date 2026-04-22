import unittest
import math
from scripts.signal_detection import SignalDetection

class test_core_sdt_math(unittest.TestCase):
    def test_hit_rate_normal_case(self):
        sdt = SignalDetection(
            hits=8, misses=2, false_alarms=3, correct_rejections=7
            )

        self.assertEqual(sdt.hit_rate(), 0.8)

    def test_false_alarm_rate_normal_case(self):
        sdt = SignalDetection(
            hits=8, misses=2, false_alarms=3, correct_rejections=7
            )

        self.assertEqual(sdt.false_alarm_rate(), 0.3)

    def test_hit_rate_with_no_signal_trials_is_nan(self):
        sdt = SignalDetection(
            hits=0, misses=0, false_alarms=3, correct_rejections=7
            )

        self.assertTrue(math.isnan(sdt.hit_rate()))

    def test_false_alarm_rate_with_no_noise_trials_is_nan(self):
        sdt = SignalDetection(
            hits=8, misses=2, false_alarms=0, correct_rejections=0
            )

        self.assertTrue(math.isnan(sdt.false_alarm_rate()))

    def test_d_prime_matches_known_value(self):
        sdt = SignalDetection(
            hits=75, misses=25, false_alarms=25, correct_rejections=75
            )

        self.assertAlmostEqual(sdt.d_prime(), 1.3489795003921634, places=6)

    def test_criterion_matches_known_value(self):
        sdt = SignalDetection(
            hits=70, misses=30, false_alarms=20, correct_rejections=80
            )

        self.assertAlmostEqual(sdt.criterion(), 0.15861036043243665, places=6)




class test_input_object_validation(unittest.TestCase):

    def assert_counts(self, sdt, hits, misses, false_alarms, correct_rejections):
        self.assertEqual(sdt._SignalDetection__hits, hits)
        self.assertEqual(sdt._SignalDetection__misses, misses)
        self.assertEqual(sdt._SignalDetection__false_alarms, false_alarms)
        self.assertEqual(sdt._SignalDetection__correct_rejections, correct_rejections)

    def test_constructor_rejects_negative_counts(self):
        cases = [
            {"hits": -1, "misses": 2, "false_alarms": 3, "correct_rejections": 4},
            {"hits": 1, "misses": -2, "false_alarms": 3, "correct_rejections": 4},
            {"hits": 1, "misses": 2, "false_alarms": -3, "correct_rejections": 4},
            {"hits": 1, "misses": 2, "false_alarms": 3, "correct_rejections": -4},
        ]

        for kwargs in cases:
            with self.subTest(kwargs=kwargs):
                with self.assertRaisesRegex(ValueError, "non-negative integer"):
                    SignalDetection(**kwargs)

    def test_constructor_rejects_non_integer_types(self):
        with self.assertRaises(TypeError):
            SignalDetection(
                hits=1.5, misses=2, false_alarms=3, correct_rejections=4
                )
            
        with self.assertRaises(TypeError):
            SignalDetection(
                hits=1, misses="2", false_alarms=3, correct_rejections=4
            )

    def test_constructor_rejects_bool(self):
        with self.assertRaises(TypeError):
            SignalDetection(
                hits=1, misses=2, false_alarms=3, correct_rejections=True
                )

    def test_add_rejects_non_signal_detection(self):
        sdt = SignalDetection(1, 2, 3, 4)
        with self.assertRaises(TypeError):
            _ = sdt + 1

    def test_sub_rejects_non_signal_detection(self):
        sdt = SignalDetection(1, 2, 3, 4)
        with self.assertRaises(TypeError):
            _ = sdt - "something else"

    def test_mul_rejects_non_numeric_scalars(self):
        sdt = SignalDetection(1, 2, 3, 4)
        with self.assertRaises(TypeError):
            _ = sdt * "2"

    def test_mul_rejects_float_scalars_with_clear_message(self):
        sdt = SignalDetection(1, 2, 3, 4)
        with self.assertRaises(TypeError):
            _ = sdt * 2.5

    def test_addition_returns_new_object_without_mutating_operands(self):
        left = SignalDetection(8, 2, 3, 7)
        right = SignalDetection(2, 8, 7, 3)

        result = left + right

        self.assertIsInstance(result, SignalDetection)
        self.assert_counts(result, 10, 10, 10, 10)
        self.assert_counts(left, 8, 2, 3, 7)
        self.assert_counts(right, 2, 8, 7, 3)
    
    def test_failed_addition_is_clean_and_does_not_mutate_operands(self):
        sdt = SignalDetection(8, 2, 3, 7)

        with self.assertRaises(TypeError):
            _ = sdt + "not a signal detection object"

        self.assert_counts(sdt, 8, 2, 3, 7)

    def test_subtraction_returns_new_object_without_mutating_operands(self):
        left = SignalDetection(8, 2, 3, 7)
        right = SignalDetection(1, 1, 1, 1)

        result = left - right

        self.assertIsInstance(result, SignalDetection)
        self.assert_counts(result, 7, 1, 2, 6)
        self.assert_counts(left, 8, 2, 3, 7)
        self.assert_counts(right, 1, 1, 1, 1)

    def test_failed_subtraction_is_clean_and_does_not_mutate_operands(self):
        smaller = SignalDetection(1, 1, 1, 1)
        larger = SignalDetection(2, 2, 2, 2)

        with self.assertRaises(ValueError):
            _ = smaller - larger

        self.assert_counts(smaller, 1, 1, 1, 1)
        self.assert_counts(larger, 2, 2, 2, 2)


    def test_multiplication_is_clean_and_does_not_mutate_operands(self):
        sdt = SignalDetection(8, 2, 3, 7)

        result = sdt * 3

        self.assertIsInstance(result, SignalDetection)
        self.assert_counts(result, 24, 6, 9, 21)
        self.assert_counts(sdt, 8, 2, 3, 7)
    
    def test_failed_multiplication_is_clean_and_does_not_mutate_operands(self):
        sdt = SignalDetection(8, 2, 3, 7)

        with self.assertRaises(TypeError):
            _ = sdt * "not a number"
        self.assert_counts(sdt, 8, 2, 3, 7)

    # make a test that fails
    def test_this_should_fail(self):
        sdt = SignalDetection(8, 2, 3, 7)
        self.assertEqual(sdt, "Hi")

class test_operator_behavior:

    def test_element_addition(self):
        sdt1 = SignalDetection(8, 2, 3, 7)
        sdt2 = SignalDetection(2, 8, 7, 3)

        result = sdt1 + sdt2

        self.assertEqual(result, SignalDetection(10, 10, 10, 10))

    def test_element_subtraction(self):
        sdt1 = SignalDetection(8, 2, 3, 7)
        sdt2 = SignalDetection(1, 1, 1, 1)

        result = sdt1 - sdt2

        self.assertEqual(result, SignalDetection(7, 1, 2, 6))

    def test_scalar_multiplication(self):
        sdt = SignalDetection(8, 2, 3, 7)
        result = sdt * 3

        self.assertEqual(result, SignalDetection(24, 6, 9, 21))
    

class test_plotting(unittest.TestCase):
    # plot_sdt() returns matplotlib objects and labels key elements
    def test_plot_sdt_returns_matplotlib_objects(self):
        sdt = SignalDetection(8, 2, 3, 7)
        fig, ax = sdt.plot_sdt()

        self.assertIsNotNone(fig)
        self.assertIsNotNone(ax)
        self.assertEqual(ax.get_xlabel(), "Decision Variable")
        self.assertEqual(ax.get_ylabel(), "Density")
        self.assertEqual(ax.get_title(), "Signal Detection Theory Distributions")

    # plot_roc(sdt_list) handles a sequence of objects
    def test_plot_roc_handles_sequence_of_objects(self):
        sdt1 = SignalDetection(8, 2, 3, 7)
        sdt2 = SignalDetection(6, 4, 5, 5)
        sdt3 = SignalDetection(4, 6, 7, 3)

        fig, ax = SignalDetection.plot_roc([sdt1, sdt2, sdt3])

        self.assertIsNotNone(fig)
        self.assertIsNotNone(ax)
        self.assertEqual(ax.get_xlabel(), "False Alarm Rate")
        self.assertEqual(ax.get_ylabel(), "Hit Rate")
        self.assertEqual(ax.get_title(), "ROC Curve")

    # plot_roc includes (0,0) and (1,1)
    def test_plot_roc_includes_endpoints(self):
        sdt1 = SignalDetection(8, 2, 3, 7)
        sdt2 = SignalDetection(6, 4, 5, 5)
        sdt3 = SignalDetection(4, 6, 7, 3)

        fig, ax = SignalDetection.plot_roc([sdt1, sdt2, sdt3])

        lines = ax.get_lines()
        x_data = lines[0].get_xdata()
        y_data = lines[0].get_ydata()

        self.assertIn(0.0, x_data)
        self.assertIn(0.0, y_data)
        self.assertIn(1.0, x_data)
        self.assertIn(1.0, y_data)