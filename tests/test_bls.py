import numpy as np
import pytest

from bls_predict import ridge_regression, bilinear_least_squares, BLSResult


def test_ridge_regression_no_regularization():
    Z = np.array([[1, 1], [1, 2], [2, 3]])
    y = np.array([1, 2, 3])
    alpha = 0.0
    beta, _ = ridge_regression(Z, y, alpha)
    expected_beta = np.linalg.inv(Z.T @ Z) @ (Z.T @ y)
    np.testing.assert_almost_equal(beta, expected_beta, decimal=6)


def test_ridge_regression_with_regularization():
    # Case with regularization
    Z = np.array([[1, 2], [2, 3], [3, 4]])
    y = np.array([2, 3, 4])
    alpha = 1.0
    beta, _ = ridge_regression(Z, y, alpha)
    ZTWZ = Z.T @ Z + alpha * np.eye(Z.shape[1])
    expected_beta = np.linalg.inv(ZTWZ) @ (Z.T @ y)
    np.testing.assert_almost_equal(beta, expected_beta, decimal=6)


def test_ridge_regression_with_weights():
    # Case with sample weights
    Z = np.array([[1, 2], [3, 4], [5, 6]])
    y = np.array([1, 2, 3])
    alpha = 0.5
    weights = np.array([0.5, 1.0, 1.5])
    beta, _ = ridge_regression(Z, y, alpha, weights=weights)
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
    beta, _ = ridge_regression(Z, y, alpha)

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


def test_basic_factorization():
    """Test basic factorization without regularization."""
    X = np.array([[1, 2], [3, 4], [5, 6]])
    result = bilinear_least_squares(X, rank=2)
    assert isinstance(result, BLSResult)
    assert result.A.shape == (3, 2)
    assert result.Y.shape == (2, 2)
    assert result.n_iter <= 500
    assert result.residual_error < 1e-4
    assert result.converged


def test_rank_one_factorization():
    """Test factorization with rank-1 approximation."""
    X = np.array([[1, 2], [1, 2], [1, 2]])
    result = bilinear_least_squares(X, rank=1)
    assert result.A.shape == (3, 1)
    assert result.Y.shape == (2, 1)
    assert result.residual_error < 1e-4
    assert result.converged


def test_regularization_effect():
    """Test the effect of regularization parameter alpha."""
    X = np.random.rand(10, 5)
    result_no_reg = bilinear_least_squares(X, rank=2, alpha=0.0)
    result_with_reg = bilinear_least_squares(X, rank=2, alpha=0.1)
    assert result_with_reg.residual_error >= result_no_reg.residual_error


def test_weights_functionality():
    """Test the functionality of sample weights."""
    X = np.random.rand(10, 5)
    weights = np.random.rand(10)
    result = bilinear_least_squares(X, rank=2, weights=weights)
    assert result.converged


def test_max_iterations():
    """Test behavior when maximum iterations are reached."""
    X = np.random.rand(10, 5)
    result = bilinear_least_squares(X, rank=2, max_iter=1, rtol=1e-12)
    assert result.n_iter == 1
    assert not result.converged


def test_invalid_rank():
    """Test handling of invalid rank parameter."""
    X = np.random.rand(5, 5)
    with pytest.raises(
        ValueError,
        match="Rank must be less than or equal to the minimum dimension of X.",
    ):
        bilinear_least_squares(X, rank=6)


def test_invalid_weights_length():
    """Test handling of invalid weights length."""
    X = np.random.rand(5, 5)
    weights = np.random.rand(4)  # Incorrect length
    with pytest.raises(
        ValueError, match="Number of weights must match the number of samples N."
    ):
        bilinear_least_squares(X, rank=2, weights=weights)


def test_alpha_scaling():
    """Test the scaling of alpha during iterations."""
    X = np.random.rand(10, 5)
    result = bilinear_least_squares(X, rank=2, alpha=0.1, alpha_scale_factor=0.5)
    assert result.converged
    assert result.alpha < 0.1  # Assuming alpha is returned in BLSResult


def test_alpha_threshold():
    """Test behavior when condition number exceeds alpha_threshold."""

    # A Hilbert matrix with a high condition number
    X = np.array(
        [
            [1.0, 0.5, 0.33333333, 0.25, 0.2],
            [0.5, 0.33333333, 0.25, 0.2, 0.16666667],
            [0.33333333, 0.25, 0.2, 0.16666667, 0.14285714],
            [0.25, 0.2, 0.16666667, 0.14285714, 0.125],
            [0.2, 0.16666667, 0.14285714, 0.125, 0.11111111],
        ]
    )

    assert np.linalg.cond(X) > 1e5

    result = bilinear_least_squares(
        X, rank=1, alpha=0.1, alpha_threshold=1e-2, alpha_scale_factor=0.5
    )
    assert result.converged

    # Alpha should not be rescaled because the system is ill-conditioned
    assert result.alpha == 0.1
