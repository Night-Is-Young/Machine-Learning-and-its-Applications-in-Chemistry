from ..src.data import normalize
import numpy as np

X_train = np.random.randn(100, 5)
X_test = np.random.randn(20, 5)

def test_normalize_train():
    X_train_normalized, _ = normalize(X_train, X_test)
    np.testing.assert_allclose(
        X_train_normalized.mean(axis=0), np.zeros(5), atol=1.0e-5,
        err_msg="Incorrect X_train normalization: non-zero mean."
    )
    np.testing.assert_allclose(
        X_train_normalized.std(axis=0), np.ones(5), rtol=1.0e-3,
        err_msg="Incorrect X_train normalization: non-unit std."
    )

def test_normalize_consistency():
    X_train_mean, X_train_std = X_train.mean(axis=0), X_train.std(axis=0)
    _, X_test_normalized = normalize(X_train, X_test)
    X_test_normalized_mean = (X_test.mean(axis=0) - X_train_mean) / X_train_std
    X_test_normalized_std = X_test.std(axis=0) / X_train_std
    np.testing.assert_allclose(
        X_test_normalized.mean(axis=0), X_test_normalized_mean, atol=1.0e-6,
        err_msg="inconsistent normalization: didn't use the training mean correctly."
    )
    np.testing.assert_allclose(
        X_test_normalized.std(axis=0), X_test_normalized_std, atol=1.0e-6,
        err_msg="inconsistent normalization: didn't use the training std correctly."
    )