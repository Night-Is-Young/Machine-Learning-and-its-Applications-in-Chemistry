from ..src.data import normalize_tpms
import numpy as np

X_train = np.random.randn(100, 5)
X_test = np.random.randn(20, 5)

def test_standardscale_train():
    X_train_normalized, _ = normalize_tpms(X_train, X_test, method='standard')
    np.testing.assert_allclose(
        X_train_normalized.mean(axis=0), np.zeros(5), atol=1.0e-5,
        err_msg="Incorrect X_train normalization: non-zero mean."
    )
    np.testing.assert_allclose(
        X_train_normalized.std(axis=0), np.ones(5), rtol=1.0e-3,
        err_msg="Incorrect X_train normalization: non-unit std."
    )

def test_standardscale_consistency():
    X_train_mean, X_train_std = X_train.mean(axis=0), X_train.std(axis=0)
    _, X_test_normalized = normalize_tpms(X_train, X_test, method='standard')
    X_test_normalized_mean = (X_test.mean(axis=0) - X_train_mean) / X_train_std
    X_test_normalized_std = X_test.std(axis=0) / X_train_std
    np.testing.assert_allclose(
        X_test_normalized.mean(axis=0), X_test_normalized_mean, atol=1.0e-5,
        err_msg="Inconsistent normalization: didn't use the training mean correctly."
    )
    np.testing.assert_allclose(
        X_test_normalized.std(axis=0), X_test_normalized_std, rtol=1.0e-3,
        err_msg="Inconsistent normalization: didn't use the training std correctly."
    )

def test_minmaxscale_train():
    X_train_normalized, _ = normalize_tpms(X_train, X_test, method='minmax')
    np.testing.assert_allclose(
        X_train_normalized.min(axis=0), np.zeros(5), atol=1.0e-5,
        err_msg="Invalid X_train normalization: min not equal to zero."
    )
    np.testing.assert_allclose(
        X_train_normalized.max(axis=0), np.ones(5), rtol=1.0e-3,
        err_msg="Invalid X_train normalization: max not equal to one."
    )

def test_minmaxscale_consistency():
    X_train_min, X_train_max = X_train.min(axis=0), X_train.max(axis=0)
    _, X_test_normalized = normalize_tpms(X_train, X_test, method='minmax')
    min_normalized = (X_test.min(axis=0) - X_train_min) / (X_train_max - X_train_min)
    max_normalized = (X_test.max(axis=0) - X_train_min) / (X_train_max - X_train_min)
    np.testing.assert_allclose(
        X_test_normalized.min(axis=0), min_normalized, atol=1.0e-5,
        err_msg="Inconsistent normalization: didn't use the training min correctly."
    )
    np.testing.assert_allclose(
        X_test_normalized.max(axis=0), max_normalized, rtol=1.0e-3,
        err_msg="Inconsistent normalization: didn't use the training max correctly."
    )