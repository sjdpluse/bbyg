import unittest

import numpy as np

from truetrade.scalper.research_models import (
    RobustScaler,
    binary_metrics,
    fit_mlp,
    fit_predict_architecture,
    quadratic_expand,
)


class ScalperResearchModelTests(unittest.TestCase):
    def test_scaler_is_fit_on_training_only_and_finite(self):
        train = np.asarray([[0.0, 1.0], [1.0, 2.0], [2.0, 3.0], [1000.0, 4.0]])
        validation = np.asarray([[1e9, -1e9]])
        scaler = RobustScaler.fit(train)
        before_center = scaler.center.copy()
        transformed = scaler.transform(validation)
        self.assertTrue(np.isfinite(transformed).all())
        self.assertTrue((np.abs(transformed) <= 8.0).all())
        np.testing.assert_array_equal(before_center, scaler.center)

    def test_quadratic_expansion_has_expected_dimension(self):
        x = np.ones((3, 8), dtype=float)
        q = quadratic_expand(x)
        self.assertEqual(q.shape, (3, 44))
        self.assertTrue(np.isfinite(q).all())

    def test_deterministic_mlp_repeats_predictions(self):
        rng = np.random.default_rng(5)
        x = rng.normal(size=(300, 8))
        y = ((x[:, 0] * x[:, 1] + x[:, 2]) > 0).astype(int)
        model_a = fit_mlp(x, y, iterations=20, seed=123)
        model_b = fit_mlp(x, y, iterations=20, seed=123)
        np.testing.assert_allclose(model_a.probability(x[:20]), model_b.probability(x[:20]), atol=1e-12)

    def test_nonlinear_models_learn_xor_better_than_linear(self):
        rng = np.random.default_rng(9)
        x = rng.normal(size=(1200, 8))
        y = ((x[:, 0] * x[:, 1]) > 0).astype(int)
        train_x, validation_x = x[:1000], x[1000:]
        train_y, validation_y = y[:1000], y[1000:]
        linear = fit_predict_architecture("linear", train_x, train_y, validation_x,
                                          linear_iterations=120, mlp_iterations=80, seed=7)
        quadratic = fit_predict_architecture("quadratic", train_x, train_y, validation_x,
                                             linear_iterations=160, mlp_iterations=80, seed=7)
        mlp = fit_predict_architecture("mlp", train_x, train_y, validation_x,
                                       linear_iterations=120, mlp_iterations=100, seed=7)
        lm = binary_metrics(validation_y, linear)
        qm = binary_metrics(validation_y, quadratic)
        mm = binary_metrics(validation_y, mlp)
        self.assertGreater(qm["balanced_accuracy"], lm["balanced_accuracy"] + 0.15)
        self.assertGreater(mm["balanced_accuracy"], lm["balanced_accuracy"] + 0.05)

    def test_unknown_architecture_fails_closed(self):
        x = np.zeros((10, 8), dtype=float)
        y = np.asarray([0, 1] * 5)
        with self.assertRaises(ValueError):
            fit_predict_architecture("unknown", x, y, x)


if __name__ == "__main__":
    unittest.main()
