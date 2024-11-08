import numpy as np
import pytest

from bls_predict import ridge_regression


def test_ridge_regression_no_regularization():
    Z = np.array([[1, 1], [1, 2], [2, 3]])
    y = np.array([1, 2, 3])
    alpha = 0.0
    beta = ridge_regression(Z, y, alpha)
    expected_beta = np.linalg.inv(Z.T @ Z) @ (Z.T @ y)
    np.testing.assert_almost_equal(beta, expected_beta, decimal=6)


def test_ridge_regression_with_regularization():
    # Case with regularization
    Z = np.array([[1, 2], [2, 3], [3, 4]])
    y = np.array([2, 3, 4])
    alpha = 1.0
    beta = ridge_regression(Z, y, alpha)
    ZTWZ = Z.T @ Z + alpha * np.eye(Z.shape[1])
    expected_beta = np.linalg.inv(ZTWZ) @ (Z.T @ y)
    np.testing.assert_almost_equal(beta, expected_beta, decimal=6)


def test_ridge_regression_with_weights():
    # Case with sample weights
    Z = np.array([[1, 2], [3, 4], [5, 6]])
    y = np.array([1, 2, 3])
    alpha = 0.5
    weights = np.array([0.5, 1.0, 1.5])
    beta = ridge_regression(Z, y, alpha, weights=weights)
    w = weights * len(weights) / np.sum(weights)
    W = np.diag(w)
    ZTWZ = Z.T @ W @ Z + alpha * np.eye(Z.shape[1])
    ZTWy = Z.T @ W @ y
    expected_beta = np.linalg.inv(ZTWZ) @ ZTWy
    np.testing.assert_almost_equal(beta, expected_beta, decimal=6)


def test_ridge_regression_ill_conditioned_matrix():
    # Case where ZTWZ is ill-conditioned and ridge regression stabilizes it
    Z = np.array([[1, 1], [1, 1.00001], [1, 1.00002]])
    y = np.array([1, 2, 3])
    alpha = 1e-2  # Small regularization parameter to stabilize the solution
    beta = ridge_regression(Z, y, alpha)

    # Check if the solution does not have large deviations
    assert np.all(
        np.isfinite(beta)
    ), "Coefficients should be finite even for ill-conditioned matrices"
    assert np.linalg.cond(Z.T @ Z) > 1e10, "Matrix should be ill-conditioned"

    # Ensure the result is stable compared to no regularization
    beta_no_reg = np.linalg.pinv(Z.T @ Z) @ (Z.T @ y)
    assert not np.allclose(
        beta, beta_no_reg, atol=1e-2
    ), "Ridge regression should differ from the non-regularized solution"


def test_ridge_regression_invalid_inverse_method():
    # Check that an invalid inverse method raises a ValueError
    Z = np.array([[1, 2], [3, 4], [5, 6]])
    y = np.array([1, 2, 3])
    alpha = 0.1
    with pytest.raises(ValueError, match="Invalid value for inverse"):
        ridge_regression(Z, y, alpha, inverse="invalid")
