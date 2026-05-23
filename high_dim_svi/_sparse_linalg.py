"""Numba-accelerated sparse linear-algebra kernels for high-dimensional SVI.

The sparse precision-parameterized variational distribution
(:mod:`high_dim_svi.sparse_inverse_variational`) and the SVI iterator
(:mod:`high_dim_svi.high_dim_rpvi`) repeatedly solve sparse triangular systems
with the variational precision Cholesky factor ``L_Q`` (and its transpose), and
form sparse mat-vec products. At a stochastic dimension of ``O(1e5-1e6)`` these
operations dominate the per-iteration cost, so they are implemented as
hand-written ``@njit`` kernels that operate directly on the CSR index arrays.

Why not ``scipy.sparse``? Two reasons:

* ``scipy.sparse.linalg.spsolve_triangular`` allocates and is comparatively slow
  inside the optimisation hot loop, and
* ``scipy``'s compiled CSR mat-vec is only instantiated for ``int32`` index
  arrays; calling it with the ``int64`` indices produced by large sparse factors
  can segfault rather than raise. The explicit kernels below sidestep both
  issues and give deterministic, allocation-light performance.

All solves assume the matrix is *triangular with an explicitly stored diagonal*
(true for a Cholesky factor). ``cache=True`` persists the compiled machine code
across runs, and :func:`warmup` triggers compilation on tiny inputs at import
time so the JIT does not run inside the Dask worker hot loop.
"""

import numpy as np
from numba import njit


@njit(cache=True)
def csr_lower_triangular_solve(indptr, indices, data, b, n):
    """Solve ``L @ x = b`` for a lower-triangular CSR matrix ``L``.

    Forward substitution: ``x[i] = (b[i] - sum_{j<i} L[i, j] x[j]) / L[i, i]``.

    Args:
        indptr (np.ndarray): CSR row-pointer array of ``L``.
        indices (np.ndarray): CSR column-index array of ``L``.
        data (np.ndarray): CSR value array of ``L``.
        b (np.ndarray): Right-hand side vector of length ``n``.
        n (int): Matrix dimension.

    Returns:
        np.ndarray: Solution vector ``x`` of length ``n``.
    """
    x = np.empty(n, dtype=np.float64)
    for i in range(n):
        s = b[i]
        diag_val = 1.0
        for p in range(indptr[i], indptr[i + 1]):
            j = indices[p]
            if j < i:
                s -= data[p] * x[j]
            elif j == i:
                diag_val = data[p]
        x[i] = s / diag_val
    return x


@njit(cache=True)
def csr_upper_triangular_solve(indptr, indices, data, b, n):
    """Solve ``U @ x = b`` for an upper-triangular CSR matrix ``U``.

    Backward substitution: ``x[i] = (b[i] - sum_{j>i} U[i, j] x[j]) / U[i, i]``.

    Args:
        indptr (np.ndarray): CSR row-pointer array of ``U``.
        indices (np.ndarray): CSR column-index array of ``U``.
        data (np.ndarray): CSR value array of ``U``.
        b (np.ndarray): Right-hand side vector of length ``n``.
        n (int): Matrix dimension.

    Returns:
        np.ndarray: Solution vector ``x`` of length ``n``.
    """
    x = np.empty(n, dtype=np.float64)
    for i in range(n - 1, -1, -1):
        s = b[i]
        diag_val = 1.0
        for p in range(indptr[i], indptr[i + 1]):
            j = indices[p]
            if j > i:
                s -= data[p] * x[j]
            elif j == i:
                diag_val = data[p]
        x[i] = s / diag_val
    return x


@njit(cache=True)
def csr_lower_triangular_solve_multi(indptr, indices, data, b_mat, n, m):
    """Solve ``L @ X = B`` for multiple right-hand-side columns.

    Args:
        indptr (np.ndarray): CSR row-pointer array of the lower-triangular ``L``.
        indices (np.ndarray): CSR column-index array of ``L``.
        data (np.ndarray): CSR value array of ``L``.
        b_mat (np.ndarray): Right-hand side matrix of shape ``(n, m)``.
        n (int): Matrix dimension.
        m (int): Number of right-hand sides.

    Returns:
        np.ndarray: Solution matrix ``X`` of shape ``(n, m)``.
    """
    x_mat = np.empty((n, m), dtype=np.float64)
    for k in range(m):
        for i in range(n):
            s = b_mat[i, k]
            diag_val = 1.0
            for p in range(indptr[i], indptr[i + 1]):
                j = indices[p]
                if j < i:
                    s -= data[p] * x_mat[j, k]
                elif j == i:
                    diag_val = data[p]
            x_mat[i, k] = s / diag_val
    return x_mat


@njit(cache=True)
def csr_upper_triangular_solve_multi(indptr, indices, data, b_mat, n, m):
    """Solve ``U @ X = B`` for multiple right-hand-side columns.

    Args:
        indptr (np.ndarray): CSR row-pointer array of the upper-triangular ``U``.
        indices (np.ndarray): CSR column-index array of ``U``.
        data (np.ndarray): CSR value array of ``U``.
        b_mat (np.ndarray): Right-hand side matrix of shape ``(n, m)``.
        n (int): Matrix dimension.
        m (int): Number of right-hand sides.

    Returns:
        np.ndarray: Solution matrix ``X`` of shape ``(n, m)``.
    """
    x_mat = np.empty((n, m), dtype=np.float64)
    for k in range(m):
        for i in range(n - 1, -1, -1):
            s = b_mat[i, k]
            diag_val = 1.0
            for p in range(indptr[i], indptr[i + 1]):
                j = indices[p]
                if j > i:
                    s -= data[p] * x_mat[j, k]
                elif j == i:
                    diag_val = data[p]
            x_mat[i, k] = s / diag_val
    return x_mat


@njit(cache=True)
def csr_matvec_multi(indptr, indices, data, b_mat, n, m):
    """Compute ``Y = A @ B`` for a CSR matrix ``A`` and dense ``B``.

    Args:
        indptr (np.ndarray): CSR row-pointer array of ``A``.
        indices (np.ndarray): CSR column-index array of ``A``.
        data (np.ndarray): CSR value array of ``A``.
        b_mat (np.ndarray): Dense matrix of shape ``(n, m)``.
        n (int): Number of rows of ``A`` (and rows of ``B``).
        m (int): Number of columns of ``B``.

    Returns:
        np.ndarray: Result matrix ``Y`` of shape ``(n, m)``.
    """
    y_mat = np.zeros((n, m), dtype=np.float64)
    for i in range(n):
        for p in range(indptr[i], indptr[i + 1]):
            j = indices[p]
            a_ij = data[p]
            for k in range(m):
                y_mat[i, k] += a_ij * b_mat[j, k]
    return y_mat


def warmup():
    """Trigger numba JIT compilation of all kernels on tiny dummy inputs.

    Call this once at import time (before the Dask cluster is started) so the
    one-off compilation cost is not paid inside the optimisation hot loop while
    background worker threads are active. With ``cache=True`` subsequent runs
    load the compiled code from disk instantly.
    """
    # 2x2 lower-triangular CSR identity: [[1, 0], [0, 1]]
    indptr = np.array([0, 1, 2], dtype=np.int32)
    indices = np.array([0, 1], dtype=np.int32)
    data = np.array([1.0, 1.0], dtype=np.float64)
    b_vec = np.ones(2, dtype=np.float64)
    b_mat = np.ones((2, 1), dtype=np.float64)

    csr_lower_triangular_solve(indptr, indices, data, b_vec, 2)
    csr_upper_triangular_solve(indptr, indices, data, b_vec, 2)
    csr_lower_triangular_solve_multi(indptr, indices, data, b_mat, 2, 1)
    csr_upper_triangular_solve_multi(indptr, indices, data, b_mat, 2, 1)
    csr_matvec_multi(indptr, indices, data, b_mat, 2, 1)
