"""Sparse precision-parameterized Gaussian variational distribution.

This is the variational family of the paper *"Scalable High-Dimensional Bayesian
Field Reconstruction with Finite Elements"*. The approximate posterior ``q(x)`` is a
multivariate Gaussian whose **precision** matrix ``Q = L_Q L_Qᵀ`` is stored through its
lower Cholesky factor ``L_Q``. The sparsity pattern of ``L_Q`` is inherited from the
finite-element Laplacian of the forward problem (loaded from a ``sparsity_pattern_file``),
so a dense ``O(n²)`` covariance is never formed even though ``Q⁻¹`` is dense (encoding
smooth, long-range spatial correlations).

Variational parameters ``λ = [μ, λ_{L_Q}]``:

* ``μ`` -- the posterior mean (first ``dimension`` entries),
* ``λ_{L_Q}`` -- the nonzero entries of ``L_Q`` in row-major (COO) order, with a
  ``softplus`` transform on the diagonal (keeps it positive with bounded gradients) and
  an identity transform on the off-diagonals.

Sampling uses ``x = μ + L_Q⁻ᵀ ε`` with ``ε ~ N(0, I)``. The ELBO gradient is computed
*directly* (sticking-the-landing estimator) by :meth:`compute_direct_elbo_gradient`,
avoiding the reparameterization Jacobian through the inverse Cholesky factor that limits
covariance-based parameterizations.
"""

import logging

import numpy as np
import scipy.special
from scipy.sparse import csr_array

from queens.utils.logger_settings import log_init_args
from queens.variational_distributions._variational_distribution import Variational

from ._sparse_linalg import (
    csr_lower_triangular_solve,
    csr_lower_triangular_solve_multi,
    csr_matvec_multi,
    csr_upper_triangular_solve,
    csr_upper_triangular_solve_multi,
    warmup,
)

_logger = logging.getLogger(__name__)

# Diagonal variational parameters above this value are treated as their own softplus output
# (softplus(λ) ≈ λ for large λ), avoiding overflow in ``exp``/``expm1``.
_SOFTPLUS_LINEAR_THRESHOLD = 20.0


class SparseInverseNormal(Variational):
    r"""Sparse precision (inverse-covariance) Cholesky Gaussian variational family.

    Parameterizes ``q(x) = N(x | μ, Q⁻¹)`` via the lower Cholesky factor ``L_Q`` of the
    precision ``Q = L_Q L_Qᵀ``.

    Attributes:
        row_idx_chol (np.ndarray): Row indices of the nonzero ``L_Q`` entries (lower triangle).
        col_idx_chol (np.ndarray): Column indices of the nonzero ``L_Q`` entries.
        half_off_diag_width (int | None): Band half-width when no sparsity file is given
            (``None`` when the pattern is loaded from file).
        nugget_l_diag (float): Constant added to the ``L_Q`` diagonal for stability.
    """

    @log_init_args
    def __init__(
        self,
        dimension: int,
        sparsity_pattern_file: str | None = None,
        half_off_diag_width: int | None = None,
        nugget_var_diag: float | None = None,
    ) -> None:
        """Initialize the sparse precision variational distribution.

        Args:
            dimension: Dimension of the random field ``x``.
            sparsity_pattern_file: Path *prefix* of the sparsity pattern. Loads
                ``<prefix>_row_idx.npy`` and ``<prefix>_col_idx.npy`` (lower-triangular COO
                indices). When given, the FE mesh connectivity is used and
                ``half_off_diag_width`` is ignored.
            half_off_diag_width: Half-width of a banded lower-triangular pattern, used only
                when ``sparsity_pattern_file`` is ``None``.
            nugget_var_diag: Variance nugget; ``sqrt(nugget_var_diag)`` is added to the
                ``L_Q`` diagonal to keep the factor well-conditioned.
        """
        self.row_idx_chol, self.col_idx_chol, self.half_off_diag_width = self._build_pattern(
            dimension, sparsity_pattern_file, half_off_diag_width
        )
        super().__init__(dimension, n_parameters=dimension + len(self.row_idx_chol))

        self.nugget_l_diag = float(np.sqrt(nugget_var_diag)) if nugget_var_diag else 0.0
        self._is_diag = self.row_idx_chol == self.col_idx_chol
        self._is_off_diag = ~self._is_diag

        self._lt_coo_order, self._lt_col_idx, self._lt_indptr = self._build_transpose_pattern()
        self._reset_caches()

        # Compile the numba kernels now (before the Dask cluster starts), not in the hot loop.
        warmup()

    # ----------------------------------------------------------------------------------
    # Construction helpers
    # ----------------------------------------------------------------------------------
    @staticmethod
    def _build_pattern(
        dimension: int, sparsity_pattern_file: str | None, half_off_diag_width: int | None
    ) -> tuple[np.ndarray, np.ndarray, int | None]:
        """Return the lower-triangular sparsity pattern (row idx, col idx, band half-width).

        Either loads the FE-connectivity pattern from file or builds a banded pattern.
        """
        if sparsity_pattern_file is not None:
            row_idx = np.load(f"{sparsity_pattern_file}_row_idx.npy").flatten().astype(np.int32)
            col_idx = np.load(f"{sparsity_pattern_file}_col_idx.npy").flatten().astype(np.int32)
            _logger.info(
                "Loaded sparsity pattern from %s: %d nonzeros, dimension %d",
                sparsity_pattern_file,
                len(row_idx),
                dimension,
            )
            return row_idx, col_idx, None

        if half_off_diag_width is None:
            raise ValueError("Provide either 'sparsity_pattern_file' or 'half_off_diag_width'.")
        half_off_diag_width = int(half_off_diag_width)
        rows, cols = [], []
        for row in range(dimension):
            for col in range(max(row - half_off_diag_width, 0), row + 1):
                rows.append(row)
                cols.append(col)
        return np.array(rows, dtype=np.int32), np.array(cols, dtype=np.int32), half_off_diag_width

    def _build_transpose_pattern(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Precompute the CSR structure of ``L_Qᵀ`` from the (fixed) ``L_Q`` pattern.

        ``L_Q`` is lower triangular at ``(row_idx_chol, col_idx_chol)``; entry ``L_Q[i, j]``
        becomes ``L_Qᵀ[j, i]``. Sorting the COO entries by ``(col, row)`` gives the CSR row
        order of ``L_Qᵀ``; the returned permutation maps cached COO values into that order.

        Returns:
            tuple: ``(coo_to_csr_order, lt_col_idx, lt_indptr)`` for building ``L_Qᵀ``.
        """
        coo_order = np.lexsort((self.row_idx_chol, self.col_idx_chol))
        lt_rows = self.col_idx_chol[coo_order]
        lt_col_idx = self.row_idx_chol[coo_order].astype(np.int32)
        lt_indptr = np.zeros(self.dimension + 1, dtype=np.int32)
        lt_indptr[1:] = np.cumsum(np.bincount(lt_rows, minlength=self.dimension))
        return coo_order, lt_col_idx, lt_indptr

    def _reset_caches(self) -> None:
        """Clear the per-iteration caches (set by :meth:`conduct_reparameterization`)."""
        self._cached_chol_vals_coo: np.ndarray | None = None
        self._cached_epsilon: np.ndarray | None = None
        self._cached_l_q: csr_array | None = None
        self._cached_l_q_t: csr_array | None = None
        self._cached_mean: np.ndarray | None = None
        self._cached_sigmoid_diag: np.ndarray | None = None

    @property
    def supports_direct_elbo_gradient(self) -> bool:
        """Whether the ELBO gradient is computed directly (no reparameterization Jacobian)."""
        return True

    # ----------------------------------------------------------------------------------
    # Parameter (de)construction
    # ----------------------------------------------------------------------------------
    def construct_variational_parameters(
        self, mean: np.ndarray, covariance: np.ndarray, off_diag_range: float = 0.0
    ) -> np.ndarray:  # pylint: disable=arguments-differ
        """Build variational parameters from a mean and a *diagonal* covariance.

        The precision Cholesky diagonal is set to ``1/sqrt(var_i)`` so that
        ``Q = diag(1/var)``, and the corresponding variational parameter is the inverse
        softplus of that value. Off-diagonal entries are set to ``off_diag_range``.

        Args:
            mean: Mean values, shape ``(d,)`` or ``(d, 1)``.
            covariance: Diagonal variance vector, shape ``(d,)`` or ``(d, 1)``.
            off_diag_range: Initial value for the off-diagonal parameters.

        Returns:
            np.ndarray: Variational parameters ``[μ, λ_{L_Q}]``.
        """
        if mean.size != covariance.size:
            raise ValueError(
                f"Mean size {mean.size} does not match covariance size {covariance.size}."
            )
        l_q_diag = np.sqrt(1.0 / covariance.flatten())
        # Inverse softplus: λ = log(exp(y) - 1), stable for large y.
        inv_softplus_diag = np.where(
            l_q_diag > _SOFTPLUS_LINEAR_THRESHOLD, l_q_diag, np.log(np.expm1(l_q_diag))
        )
        cholesky_params = np.where(
            self._is_diag, inv_softplus_diag[self.row_idx_chol], off_diag_range
        )
        return np.hstack((mean.flatten(), cholesky_params))

    def reconstruct_distribution_parameters(
        self, variational_parameters: np.ndarray, return_cholesky: bool = False
    ) -> tuple[np.ndarray, csr_array]:
        """Reconstruct the mean and the precision (or its Cholesky factor) from parameters.

        Also caches the transformed Cholesky COO values (used to assemble ``L_Qᵀ``).

        Args:
            variational_parameters: Variational parameters ``[μ, λ_{L_Q}]``.
            return_cholesky: If ``True`` return ``(mean, L_Q)``; otherwise ``(mean, Q)`` with
                ``Q = L_Q L_Qᵀ``.

        Returns:
            tuple: ``(mean, L_Q)`` or ``(mean, Q)`` depending on ``return_cholesky``.
        """
        mean = variational_parameters[: self.dimension].reshape(-1, 1)
        cholesky_array = variational_parameters[self.dimension :].copy()

        # Diagonal: softplus transform (positive, numerically stable); off-diagonal: identity.
        diag_raw = cholesky_array[self._is_diag]
        cholesky_array[self._is_diag] = np.where(
            diag_raw > _SOFTPLUS_LINEAR_THRESHOLD, diag_raw, np.log1p(np.exp(diag_raw))
        )
        if self.nugget_l_diag > 0:
            cholesky_array[self._is_diag] += self.nugget_l_diag
        self._cached_chol_vals_coo = cholesky_array

        l_q = csr_array(
            (cholesky_array, (self.row_idx_chol, self.col_idx_chol)),
            shape=(self.dimension, self.dimension),
        )
        if return_cholesky:
            return mean, l_q
        return mean, l_q.dot(csr_array(l_q.transpose()))

    def _grad_reconstruct_distribution_parameters(
        self, variational_parameters: np.ndarray
    ) -> np.ndarray:
        """Chain-rule factors of the parameter transforms.

        Diagonal: ``d softplus(λ)/dλ = sigmoid(λ)``; off-diagonal and mean: identity.

        Args:
            variational_parameters: Variational parameters.

        Returns:
            np.ndarray: Row vector ``(1, n_parameters)`` of chain-rule factors.
        """
        grad_mean = np.ones((1, self.dimension))
        grad_cholesky = np.empty((1, self.n_parameters - self.dimension))
        if self._cached_sigmoid_diag is not None:
            grad_cholesky[:, self._is_diag] = self._cached_sigmoid_diag
        else:
            chol_params = variational_parameters[self.dimension :]
            grad_cholesky[:, self._is_diag] = scipy.special.expit(chol_params[self._is_diag])
        grad_cholesky[:, self._is_off_diag] = 1.0
        return np.hstack((grad_mean, grad_cholesky))

    # ----------------------------------------------------------------------------------
    # Sampling, density and gradients
    # ----------------------------------------------------------------------------------
    def draw(self, variational_parameters: np.ndarray, n_draws: int = 1) -> np.ndarray:
        """Draw samples via ``x = μ + L_Q⁻ᵀ ε``.

        Args:
            variational_parameters: Variational parameters.
            n_draws: Number of samples to draw.

        Returns:
            np.ndarray: Samples, shape ``(n_draws, dimension)``.
        """
        mean, l_q = self.reconstruct_distribution_parameters(
            variational_parameters, return_cholesky=True
        )
        l_q_t = csr_array(l_q.transpose())
        epsilon = np.random.randn(self.dimension, n_draws)
        z = csr_upper_triangular_solve_multi(
            l_q_t.indptr, l_q_t.indices, l_q_t.data, epsilon, self.dimension, n_draws
        )
        return z.T + mean.reshape(1, -1)

    def logpdf(self, variational_parameters: np.ndarray, x: np.ndarray) -> np.ndarray:
        """Evaluate ``log q(x)`` at row-wise samples ``x``.

        ``log q = -0.5 d log(2π) + sum log(diag(L_Q)) - 0.5 (x-μ)ᵀ Q (x-μ)``.

        Args:
            variational_parameters: Variational parameters.
            x: Row-wise samples, shape ``(n_samples, dimension)``.

        Returns:
            np.ndarray: Log-pdf values, shape ``(n_samples,)``.
        """
        mean, l_q = self.reconstruct_distribution_parameters(
            variational_parameters, return_cholesky=True
        )
        precision = l_q.dot(l_q.transpose())
        log_det_q = 2.0 * np.sum(np.log(np.abs(l_q.diagonal())))
        const = -0.5 * self.dimension * np.log(2.0 * np.pi) + 0.5 * log_det_q

        x = np.atleast_2d(x)
        result = np.empty(x.shape[0])
        for i in range(x.shape[0]):
            diff = x[i, :].reshape(-1, 1) - mean
            result[i] = const - 0.5 * float(diff.T @ precision.dot(diff))
        return result

    def pdf(self, variational_parameters: np.ndarray, x: np.ndarray) -> np.ndarray:
        """Evaluate ``q(x)`` (see :meth:`logpdf`)."""
        return np.exp(self.logpdf(variational_parameters, x))

    def grad_params_logpdf(self, variational_parameters, x):
        """Score function -- not used (``score_function_bool`` is ``False``)."""
        raise NotImplementedError(
            "grad_params_logpdf (score function) is not implemented for SparseInverseNormal."
        )

    def fisher_information_matrix(self, variational_parameters):
        """Dense FIM is infeasible here; natural gradients use sparse solves instead."""
        raise NotImplementedError(
            "fisher_information_matrix is not implemented for SparseInverseNormal."
        )

    def initialize_variational_parameters(self, random: bool = False) -> np.ndarray:
        """Random/default initialization is not supported (see the iterator's init logic).

        The high-dimensional run is initialized from a file or from prior moments via
        :meth:`construct_variational_parameters`, handled by
        :class:`high_dim_svi.high_dim_rpvi.HighDimRPVI`.
        """
        raise NotImplementedError(
            "SparseInverseNormal does not support random initialization; initialize from a "
            "file or via construct_variational_parameters()."
        )

    def grad_logpdf_sample(
        self, sample_batch: np.ndarray, variational_parameters: np.ndarray
    ) -> np.ndarray:
        r"""Gradient of ``log q`` w.r.t. the sample ``x``: ``grad_x log q = -Q (x - μ)``.

        Exploits the cached reparameterization: since ``x = μ + L_Q⁻ᵀ ε`` we have
        ``Q (x - μ) = L_Q ε``, so the gradient is ``-L_Q ε`` -- one sparse mat-vec, no need
        to form ``Q``.

        Args:
            sample_batch: Row-wise samples, shape ``(n_samples, dimension)``.
            variational_parameters: Variational parameters.

        Returns:
            np.ndarray: Gradients, shape ``(n_samples, dimension, 1)``.
        """
        if self._cached_l_q is not None and self._cached_epsilon is not None:
            eps_t = np.ascontiguousarray(self._cached_epsilon.T)  # (d, M)
            result = -csr_matvec_multi(
                self._cached_l_q.indptr,
                self._cached_l_q.indices,
                self._cached_l_q.data,
                eps_t,
                self.dimension,
                self._cached_epsilon.shape[0],
            )  # (d, M)
            return result.T[:, :, np.newaxis]  # (M, d, 1)

        # Fallback (called outside the SVI loop): form Q and apply it explicitly.
        mean, precision = self.reconstruct_distribution_parameters(
            variational_parameters, return_cholesky=False
        )
        gradients = [(-precision.dot(s.reshape(-1, 1) - mean)).reshape(-1, 1) for s in sample_batch]
        return np.array(gradients)

    # ----------------------------------------------------------------------------------
    # Reparameterization, entropy and the direct ELBO gradient
    # ----------------------------------------------------------------------------------
    def conduct_reparameterization(
        self, variational_parameters: np.ndarray, n_samples: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Draw a reparameterized sample batch ``x = μ + L_Q⁻ᵀ ε``.

        Populates the per-iteration caches (``L_Q``, ``L_Qᵀ``, ``ε``, the mean and
        ``sigmoid`` of the diagonal parameters) reused by the score, entropy, ELBO gradient
        and :meth:`apply_inverse_precision`.

        Args:
            variational_parameters: Variational parameters.
            n_samples: Number of samples in the batch.

        Returns:
            tuple[np.ndarray, np.ndarray]: ``(samples, epsilon)``, both ``(n_samples, d)``.
        """
        mean, l_q = self.reconstruct_distribution_parameters(
            variational_parameters, return_cholesky=True
        )
        epsilon = np.random.normal(0.0, 1.0, size=(n_samples, self.dimension))

        self._cached_epsilon = epsilon
        self._cached_l_q = l_q
        self._cached_mean = mean
        self._cached_sigmoid_diag = scipy.special.expit(
            variational_parameters[self.dimension :][self._is_diag]
        )
        self._cached_l_q_t = self._assemble_l_q_transpose()

        # numba kernels segfault on inf/nan; guard before entering JIT code.
        if not np.all(np.isfinite(self._cached_l_q_t.data)):
            n_bad = int(np.sum(~np.isfinite(self._cached_l_q_t.data)))
            raise ValueError(
                f"L_Q contains {n_bad} non-finite entries. Variational parameters have "
                "diverged -- reduce the learning rate or enable gradient clipping."
            )

        z = csr_upper_triangular_solve_multi(
            self._cached_l_q_t.indptr,
            self._cached_l_q_t.indices,
            self._cached_l_q_t.data,
            np.ascontiguousarray(epsilon.T),
            self.dimension,
            n_samples,
        )
        return z.T + mean.reshape(1, -1), epsilon

    def _assemble_l_q_transpose(self) -> csr_array:
        """Assemble ``L_Qᵀ`` (CSR) from the cached COO values via the precomputed pattern."""
        lt_data = self._cached_chol_vals_coo[self._lt_coo_order].astype(np.float64)
        return csr_array(
            (lt_data, self._lt_col_idx.copy(), self._lt_indptr.copy()),
            shape=(self.dimension, self.dimension),
        )

    def entropy(self, variational_parameters: np.ndarray) -> float:
        """Differential entropy ``H(q) = 0.5 d log(2πe) - sum log(diag(L_Q))``.

        Args:
            variational_parameters: Variational parameters.

        Returns:
            float: Entropy of ``q``.
        """
        if self._cached_l_q is not None:
            diag_vals = self._cached_l_q.diagonal()
        else:
            _, l_q = self.reconstruct_distribution_parameters(
                variational_parameters, return_cholesky=True
            )
            diag_vals = l_q.diagonal()
        log_det_q = 2.0 * np.sum(np.log(np.abs(diag_vals)))
        return 0.5 * self.dimension * np.log(2.0 * np.pi * np.e) - 0.5 * log_det_q

    def apply_inverse_precision(self, vector: np.ndarray) -> np.ndarray:
        r"""Apply ``Q⁻¹`` to a vector using the cached precision Cholesky factors.

        Computes ``Q⁻¹ v = L_Q⁻ᵀ (L_Q⁻¹ v)`` with two sparse triangular solves. This is the
        mean-only natural-gradient preconditioner used by the iterator. Requires
        :meth:`conduct_reparameterization` to have been called first (it populates the caches).

        Args:
            vector: Right-hand side ``v`` of length ``dimension``.

        Returns:
            np.ndarray: ``Q⁻¹ v``.
        """
        if self._cached_l_q is None or self._cached_l_q_t is None:
            raise RuntimeError(
                "No cached Cholesky factor; call conduct_reparameterization() first."
            )
        if not np.all(np.isfinite(self._cached_l_q.data)):
            raise ValueError("Cached L_Q has non-finite entries; variational parameters diverged.")
        z = csr_lower_triangular_solve(
            self._cached_l_q.indptr,
            self._cached_l_q.indices,
            self._cached_l_q.data,
            vector,
            self.dimension,
        )
        return csr_upper_triangular_solve(
            self._cached_l_q_t.indptr,
            self._cached_l_q_t.indices,
            self._cached_l_q_t.data,
            z,
            self.dimension,
        )

    def compute_direct_elbo_gradient(
        self,
        sample_batch: np.ndarray,
        grad_log_likelihood_batch: np.ndarray,
        grad_log_variational_batch: np.ndarray,
        variational_parameters: np.ndarray,
    ) -> np.ndarray:
        r"""Direct ELBO gradient via the sticking-the-landing (STL) estimator.

        With the control-variate-reduced per-sample gradient
        ``g_i = grad_x log p(y|x_i) - grad_x log q(x_i)`` (the prior gradient is added inside
        the deal.II adjoint, so it is already part of ``grad log p``):

        * mean: ``dELBO/dμ = (1/M) Σ_i g_i``
        * precision Cholesky: ``dELBO/d(L_Q)_{kl} = -(1/M) Σ_i (x_i - μ)_k (z_i)_l`` with
          ``z_i = L_Q⁻¹ g_i``.

        The entropy gradient is captured implicitly through the ``grad log q`` term in
        ``g_i`` (no explicit entropy term). The parameter-transform chain rule is applied at
        the end.

        Args:
            sample_batch: Samples, shape ``(M, d)``.
            grad_log_likelihood_batch: ``grad_x log p`` per sample, shape ``(M, d)``.
            grad_log_variational_batch: ``grad_x log q`` per sample, shape ``(M, d)``.
            variational_parameters: Current variational parameters.

        Returns:
            np.ndarray: ELBO gradient, shape ``(n_parameters,)``.
        """
        n_samples = sample_batch.shape[0]
        if self._cached_l_q is not None and self._cached_mean is not None:
            l_q, mu = self._cached_l_q, self._cached_mean.flatten()
        else:
            mean, l_q = self.reconstruct_distribution_parameters(
                variational_parameters, return_cholesky=True
            )
            mu = mean.flatten()

        if not np.all(np.isfinite(l_q.data)):
            n_bad = int(np.sum(~np.isfinite(l_q.data)))
            raise ValueError(
                f"L_Q contains {n_bad} non-finite entries in compute_direct_elbo_gradient."
            )

        g_batch = grad_log_likelihood_batch - grad_log_variational_batch  # (M, d)
        grad_mu = np.mean(g_batch, axis=0)

        # z_i = L_Q⁻¹ g_i for all samples at once: solve L_Q @ Z = g_batchᵀ -> (d, M).
        z = csr_lower_triangular_solve_multi(
            l_q.indptr,
            l_q.indices,
            l_q.data,
            np.ascontiguousarray(g_batch.T),
            self.dimension,
            n_samples,
        )

        # Accumulate -(x_i - μ)_k (z_i)_l at the sparse (k, l) indices, averaged over samples.
        delta_batch = sample_batch - mu[np.newaxis, :]  # (M, d)
        outer_accum = np.sum(
            delta_batch[:, self.row_idx_chol] * z[self.col_idx_chol, :].T, axis=0
        )
        grad_l_q_entries = -outer_accum / n_samples

        # Chain rule for the parameter transforms (the mean block is identity).
        grad_chain = self._grad_reconstruct_distribution_parameters(variational_parameters)
        grad_l_q_params = grad_l_q_entries * grad_chain[0, self.dimension :]

        return np.concatenate([grad_mu, grad_l_q_params])

    def export_dict(self, variational_parameters: np.ndarray) -> dict:
        """Export the distribution as a dictionary for the results pickle.

        Args:
            variational_parameters: Variational parameters.

        Returns:
            dict: Mean, precision, precision Cholesky and the raw parameters.
        """
        mean, l_q = self.reconstruct_distribution_parameters(
            variational_parameters, return_cholesky=True
        )
        return {
            "type": "sparse_inverse_Normal",
            "mean": mean,
            "precision": l_q.dot(csr_array(l_q.transpose())),
            "cholesky_precision": l_q,
            "variational_parameters": variational_parameters,
        }
