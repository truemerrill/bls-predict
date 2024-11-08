"""
The MIT License (MIT)
Copyright (C) 2024 True Merrill <true.merrill@gmail.com>

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""

from typing import Literal
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


__version__ = "0.1.0"


Array = NDArray[np.float64]


def ridge_regression(
    Z: Array,
    y: Array,
    alpha: float,
    weights: Array | None = None,
    inverse: Literal["inverse", "pseudoinverse"] = "inverse",
) -> tuple[Array, float]:
    """Perform ridge regression with optional weights.

    Note:
        In the following, N is the number of samples and F is the number of
        features (columns in the design matrix)

    Args:
        Z (Array): Design matrix of shape (N, F).
        y (Array): Response vector of shape (N,).
        alpha (float): Regularization parameter.  Setting alpha to zero turns
            off regularization.
        weights (Array, optional): Optional sample weights of shape
            (N,). The weights do not need to be normalized.  Defaults to None.
        inverse (Literal["inverse", "pseudoinverse"], optional): Method for
            matrix inversion, either "inverse" or "pseudoinverse". Defaults to
            "inverse".

    Raises:
        ValueError: if the matrix inversion method is invalid.

    Returns:
        tuple[Array, float]: Tuple containing:
            - coefficient vector of shape (F,)
            - the condition number of the ridge matrix
    """
    N, F = Z.shape

    if weights is not None:
        w = np.array(weights)
        w = w * len(w) / np.sum(w)
        W = np.diag(w)

        ZTWZ = Z.T @ W @ Z
        ZTWy = Z.T @ W @ y
    else:
        ZTWZ = Z.T @ Z
        ZTWy = Z.T @ y

    Id = np.eye(F)
    R = ZTWZ + alpha * Id
    condition_number = np.linalg.cond(R)

    if inverse == "inverse":
        beta = np.linalg.inv(R) @ ZTWy
    elif inverse == "pseudoinverse":
        beta = np.linalg.pinv(R) @ ZTWy
    else:
        raise ValueError(f"Invalid value for inverse: {inverse}")
    return beta, condition_number


@dataclass
class BLSResult:
    """Result of bilinear least-squares.

    Note:
        - N: Number of samples (rows in the observed data matrix X).
        - F: Number of features (columns in the observed data matrix X).
        - R: Desired rank for the factorization, representing the number of
          latent factors.

    Attributes:
        A (Array): Factor matrix of shape (N, R).
        Y (Array): Factor matrix of shape (F, R).
        n_iter (int): Number of iterations performed.
        residual_error (float): Final residual error (Frobenius norm of
            the difference between X and A @ Y.T).
        delta_error (float): Relative change in residual error during the
            last iteration.
        alpha (float): Final value of the regularization parameter.
        converged (bool): Whether the algorithm converged before reaching
            the maximum number of iteration
    """

    A: Array
    Y: Array
    n_iter: int
    residual_error: float
    delta_error: float
    alpha: float
    converged: bool


def bilinear_least_squares(
    X: Array,
    rank: int,
    alpha: float = 0.0,
    alpha_scale_factor: float = 1.0,
    alpha_threshold: float = 1e4,
    weights: Array | None = None,
    inverse: Literal["inverse", "pseudoinverse"] = "inverse",
    max_iter: int = 500,
    rtol: float = 1e-4,
) -> BLSResult:
    """Perform bilinear least squares factorization.

    This function approximates the observed data matrix X as the product of two
    lower-rank matrices, A and Y, by minimizing the Frobenius norm of the
    difference between X and A @ Y.T. Regularization and sample weights can be
    applied to improve the stability and accuracy of the factorization.

    Note:
        - N: Number of samples (rows in the observed data matrix X).
        - F: Number of features (columns in the observed data matrix X).
        - R: Desired rank for the factorization, representing the number of
          latent factors.

    Args:
        X (Array): Observed data matrix of shape (N, F).
        rank (int): Desired rank for the factorization.
        alpha (float, optional): Regularization parameter. Setting alpha to
            zero turns off regularization. Defaults to 0.0.
        alpha_scale_factor (float, optional): Factor used to rescale the
            regularization parameter after each iteration. A scale factor less
            than one will lower the regularization parameter. The parameter is
            not rescaled if the system is ill-conditioned. Defaults to 1.0.
        alpha_threshold (float, optional): Threshold for an ill-conditioned
            system.  If the condition number of the ridge matrix is larger than
            alpha_threshold, then the regularization parameter will not be
            scaled.  Defaults to 1e4.
        weights (Array, optional): Optional sample weights of shape (N,). The
            weights do not need to be normalized. Defaults to None.
        max_iter (int, optional): Maximum number of iterations for the
            alternating least squares algorithm. Defaults to 500.
        rtol (float, optional): Tolerance for the stopping condition. The
            algorithm stops when the relative change in the residual error
            is less than rtol.  The residual error is the Frobenius norm of the
            difference between X and A @ Y.T. Defaults to 1e-4.

    Raises:
        ValueError: If R is greater than the minimum dimension of X.
        ValueError: If the dimensions of weights do not match the number of
            samples N.

    Returns:
        BLSResult: The bilinear least-squares result.
    """
    N, F = X.shape
    R = rank

    if R > min(N, F):
        raise ValueError(
            "Rank must be less than or equal to the minimum dimension of X."
        )

    if weights is not None:
        if len(weights) != N:
            raise ValueError("Number of weights must match the number of samples N.")

    # SVD-based initialization
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    A = U[:, :R] @ np.diag(np.sqrt(S[:R]))
    Y = Vt[:R, :].T @ np.diag(np.sqrt(S[:R]))
    residual_error = np.inf

    for n_iter in range(1, max_iter + 1):
        residual_error_old = residual_error
        mean_condition_number = 0.0
        for f in range(F):
            Y[f, :], c = ridge_regression(
                A, X[:, f], alpha, weights=weights, inverse=inverse
            )
            mean_condition_number += c / F

        for n in range(N):
            # Note:
            #   Since this regression occurs over a constant sample slice, all
            #   of these points have the same weight.  So we can ignore the
            #   weighting during this step.

            A[n, :], c = ridge_regression(Y, X[n, :], alpha, inverse=inverse)
            mean_condition_number += c / N

        residual_error = float(np.linalg.norm(X - A @ Y.T, "fro"))
        delta_error = (
            abs(residual_error - residual_error_old) / residual_error_old
            if residual_error_old != 0
            else abs(residual_error - residual_error_old)
        )

        if delta_error < rtol:
            converged = True
            break

        if mean_condition_number < alpha_threshold:
            alpha = alpha_scale_factor * alpha
    else:
        converged = False

    return BLSResult(A, Y, n_iter, residual_error, delta_error, alpha, converged)
