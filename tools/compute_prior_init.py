#!/usr/bin/env python3
"""Generate the SVI initial variational parameters via a CHOLMOD-free IC(0).

This is a dependency-light replacement for the deal.II project's
``compute_prior_init.py`` (which needs ``scikit-sparse``/CHOLMOD). It computes the
**incomplete Cholesky of the prior precision restricted to the sparsity pattern**
(IC(0), the paper's Section 2.3 initialization) with the numba kernel
:func:`high_dim_svi._sparse_linalg.incomplete_cholesky_lower`, scales it to a target
marginal variance, and writes the parameters in the QUEENS
``SparseInverseNormal`` format ``[mu, lambda_{L_Q}]`` (diagonal entries inverse-softplus,
off-diagonal identity).

IMPORTANT: the inputs are DOF-ordering dependent and must come from the SAME MPI rank
count as the SVI run (see the demo's NUM_PROCS_PER_JOB note). Run ``export_sparsity`` at
that rank count first to produce ``rf_sparsity_{row,col}_idx.npy`` and
``rf_A_kappa_values.npy``.

Usage:
    python tools/compute_prior_init.py \
        --data-dir /path/to/darcy_flow_3d_donut \
        --target-variance 0.013523 --mean 0.1 \
        --output /path/to/darcy_flow_3d_donut/initial_variational_params_inverse.npy
"""

import argparse

import numpy as np
import scipy.sparse as sp

from high_dim_svi._sparse_linalg import incomplete_cholesky_lower


def _build_lower_csr(rows, cols, vals, n):
    """Build a lower-triangular CSR (columns sorted ascending, diagonal last per row)."""
    csr = sp.coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsr()
    csr.sort_indices()
    # IC(0) kernel requires the diagonal to be the last stored entry of each row.
    last_cols = csr.indices[csr.indptr[1:] - 1]
    if not np.array_equal(last_cols, np.arange(n)):
        raise ValueError("Each row must store its diagonal last (lower-triangular pattern).")
    return csr


def ic0_with_shift(csr, n, max_tries=8):
    """Run IC(0); on breakdown (non-positive pivot) retry with a growing diagonal shift.

    Returns the L values (aligned with ``csr.indices``) and the shift that succeeded.
    """
    diag_pos = csr.indptr[1:] - 1  # position of each diagonal entry
    shift = 0.0
    for attempt in range(max_tries):
        data = csr.data.copy()
        if shift > 0.0:
            data[diag_pos] *= 1.0 + shift
        ell = incomplete_cholesky_lower(csr.indptr, csr.indices, data, n)
        if np.all(np.isfinite(ell)) and np.all(ell[diag_pos] > 0.0):
            return ell, shift
        shift = 1e-10 if shift == 0.0 else shift * 10.0
        print(f"  IC(0) breakdown; retrying with diagonal shift {shift:.1e}")
    raise RuntimeError("IC(0) failed to factorize even with a diagonal shift.")


def self_test():
    """Validate the kernel: full pattern == exact Cholesky; sparse pattern == IC(0) property."""
    rng = np.random.default_rng(0)

    # (1) Dense/full lower-triangular pattern -> IC(0) must equal the exact Cholesky.
    n = 8
    m = rng.standard_normal((n, n))
    a = m @ m.T + n * np.eye(n)
    full = sp.csr_matrix(np.tril(a))
    full.sort_indices()
    ell = incomplete_cholesky_lower(full.indptr, full.indices, full.data, n)
    l_mat = sp.csr_matrix((ell, full.indices, full.indptr), shape=(n, n)).toarray()
    assert np.allclose(l_mat, np.linalg.cholesky(a)), "IC(0) != exact Cholesky on full pattern"

    # (2) Sparse pattern with genuine dropped fill-in: a 2-D 5-point Laplacian (its exact
    #     Cholesky has fill, so IC(0) must reproduce A only ON the stored pattern).
    g = 6
    tri = sp.diags([-1.0, 2.0, -1.0], [-1, 0, 1], shape=(g, g))
    a2d = (sp.kron(sp.identity(g), tri) + sp.kron(tri, sp.identity(g))).tocsr()
    a2d = a2d + 0.5 * sp.identity(g * g)  # shift to SPD / diagonally dominant
    low = sp.tril(a2d).tocsr()
    low.sort_indices()
    nn = g * g
    ell = incomplete_cholesky_lower(low.indptr, low.indices, low.data, nn)
    assert np.all(np.isfinite(ell)), "IC(0) broke down on the 2-D Laplacian test"
    l_mat = sp.csr_matrix((ell, low.indices, low.indptr), shape=(nn, nn))
    resid = (l_mat @ l_mat.T - a2d).tocsr()
    on_pattern = np.abs(resid.multiply(low.astype(bool))).max()
    # also confirm the test is meaningful: the exact Cholesky has fill beyond the pattern
    fill = (sp.csr_matrix(np.linalg.cholesky(a2d.toarray())).astype(bool) > low.astype(bool)).nnz
    assert on_pattern < 1e-10, f"IC(0) property violated: max|LL^T - A| on pattern = {on_pattern:.2e}"
    print(f"Self-test passed: exact Cholesky on full pattern; IC(0) property holds on the 2-D "
          f"Laplacian (max|LL^T - A| on pattern = {on_pattern:.1e}; exact Cholesky dropped "
          f"{fill} fill entries).")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True, help="Dir with rf_sparsity_*/rf_A_kappa_values.npy")
    parser.add_argument("--target-variance", type=float, default=0.013523)
    parser.add_argument("--mean", type=float, default=0.1)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    self_test()

    print("Loading sparsity pattern and prior-precision values...")
    rows = np.load(f"{args.data_dir}/rf_sparsity_row_idx.npy").ravel().astype(np.int64)
    cols = np.load(f"{args.data_dir}/rf_sparsity_col_idx.npy").ravel().astype(np.int64)
    vals = np.load(f"{args.data_dir}/rf_A_kappa_values.npy").ravel().astype(np.float64)
    n = int(max(rows.max(), cols.max())) + 1
    nnz = rows.size
    print(f"  dimension {n}, lower-triangular nnz {nnz}")

    csr = _build_lower_csr(rows, cols, vals, n)
    print("Computing IC(0) of the prior precision (numba)...")
    ell_csr, shift = ic0_with_shift(csr, n)
    if shift:
        print(f"  used diagonal shift {shift:.1e} to avoid breakdown")

    # Map L from CSR order back to the original (rows, cols) order expected by SparseInverseNormal.
    l_mat = sp.csr_matrix((ell_csr, csr.indices, csr.indptr), shape=(n, n))
    l_vals = np.asarray(l_mat[rows, cols]).ravel()

    # Scale L so the average marginal variance ~ target (var_i ~ 1 / sum_j L[i,j]^2).
    row_sq = np.zeros(n)
    np.add.at(row_sq, rows, l_vals**2)
    scale = 1.0 / np.sqrt(args.target_variance * np.mean(row_sq))
    l_vals *= scale
    row_sq *= scale**2
    print(f"  scale factor {scale:.4e}; avg marginal variance after scaling "
          f"{np.mean(1.0 / row_sq):.6f} (target {args.target_variance})")

    # QUEENS SparseInverseNormal format: diagonal via inverse-softplus, off-diagonal identity.
    is_diag = rows == cols
    if np.any(l_vals[is_diag] <= 0):
        raise ValueError("Non-positive diagonal in L_Q after scaling.")
    chol = l_vals.copy()
    d = chol[is_diag]
    chol[is_diag] = np.where(d > 20.0, d, np.log(np.expm1(d)))  # inverse softplus (stable)
    params = np.concatenate([np.full(n, args.mean), chol])

    np.save(args.output, params)
    n_off = nnz - int(is_diag.sum())
    print(f"Saved {params.size} params = {n} mean + {nnz} L_Q ({int(is_diag.sum())} diag "
          f"+ {n_off} off-diag) to {args.output}")


if __name__ == "__main__":
    main()
