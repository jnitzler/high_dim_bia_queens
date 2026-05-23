"""Reparameterization-based SVI iterator for high-dimensional field reconstruction.

This is the optimisation driver of the paper. It maximises the ELBO of the sparse
precision-parameterized Gaussian posterior
(:class:`high_dim_svi.sparse_inverse_variational.SparseInverseNormal`) with the
**sticking-the-landing** (STL) gradient estimator and a **mean-only natural
gradient** preconditioner, while the noise variance is updated by the VB-EM step
inside the likelihood (:class:`high_dim_svi.gaussian_vbem_likelihood.GaussianVBEM`).

It subclasses QUEENS' :class:`~queens.iterators._variational_inference.VariationalInference`
to reuse the stochastic-optimisation loop, NaN handling, result writing and the
``max_feval`` stopping criterion, and overrides the gradient/ELBO computation.

Differences from QUEENS' generic ``RPVI`` that motivate a dedicated iterator:

* the ELBO gradient is formed *directly* (no reparameterization Jacobian) via the
  variational family's :meth:`compute_direct_elbo_gradient`;
* the natural gradient uses two sparse triangular solves with the variational
  precision Cholesky factor instead of inverting a dense Fisher matrix; and
* the prior term of the ELBO is **0 on the Python side** -- the SPDE/GMRF
  log-prior gradient is assembled inside the deal.II adjoint, so the gradient
  returned by the model is already the full log-posterior gradient.
"""

import logging

import numpy as np

from queens.iterators._variational_inference import (
    VALID_EXPORT_FIELDS as _BASE_VALID_EXPORT_FIELDS,
)
from queens.iterators._variational_inference import VariationalInference
from queens.utils.collection import CollectionObject
from queens.utils.logger_settings import log_init_args
from queens.utils.valid_options import check_if_valid_options

_logger = logging.getLogger(__name__)

# Extend the base export fields with the cheap convergence diagnostics this iterator
# records (those needing no extra input files, i.e. no ground truth or prior precision).
VALID_EXPORT_FIELDS = _BASE_VALID_EXPORT_FIELDS + [
    "tau_tilde",
    "noise_variance",
    "grad_norm",
    "predictive_residual",
    "reconstruction_error",
]


class HighDimRPVI(VariationalInference):
    """High-dimensional sparse-precision SVI iterator with STL + natural gradients.

    Attributes:
        natural_gradient_mean_only (bool): If ``True``, precondition the mean gradient
            with the variational precision via two sparse triangular solves
            (``Q^{-1} grad_mu = L_Q^{-T} L_Q^{-1} grad_mu``).
        score_function_bool (bool): Always ``False`` (the STL estimator already includes
            the score term through the variational-gradient control variate).
    """

    @log_init_args
    def __init__(
        self,
        model,
        parameters,
        global_settings,
        result_description,
        variational_distribution,
        n_samples_per_iter,
        random_seed,
        max_feval,
        stochastic_optimizer,
        variational_parameter_initialization=None,
        ground_truth_file=None,
        natural_gradient_mean_only=True,
        variational_transformation=None,
        verbose_every_n_iter=10,
    ):
        """Initialize the iterator.

        Args:
            model (Model): Likelihood model (here :class:`GaussianVBEM`).
            parameters (Parameters): Parameters object (a flat-prior field of dimension d).
            global_settings (GlobalSettings): QUEENS experiment settings.
            result_description (dict): Result-storage settings; ``iterative_field_names``
                must be a subset of :data:`VALID_EXPORT_FIELDS`.
            variational_distribution (SparseInverseNormal): Sparse-precision variational family.
            n_samples_per_iter (int): Monte-Carlo batch size per iteration.
            random_seed (int): Seed for the reparameterization RNG (set in :meth:`pre_run`).
            max_feval (int): Maximum number of forward-model evaluations.
            stochastic_optimizer (obj): QUEENS stochastic optimizer (e.g. ``AdamAx``).
            variational_parameter_initialization: ``.npy`` path to load the initial
                parameters, or a dict ``{"mean", "variance", "off_diag_range"}`` to build
                them from a diagonal covariance.
            ground_truth_file (str | None): Optional ``.npy`` path to the ground-truth field
                ``x_gt``. When given, the reconstruction error (posterior-mean discrepancy)
                ``eps_pm = ||mu - x_gt|| / ||x_gt||`` is recorded each iteration.
            natural_gradient_mean_only (bool): Enable the mean-only natural gradient.
            variational_transformation (str | None): Optional transform (kept ``None``).
            verbose_every_n_iter (int): Logging / result-writing interval.
        """
        iterative_data_names = result_description.get("iterative_field_names", [])
        check_if_valid_options(VALID_EXPORT_FIELDS, iterative_data_names)
        iteration_data = CollectionObject(*iterative_data_names)

        super().__init__(
            model=model,
            parameters=parameters,
            global_settings=global_settings,
            result_description=result_description,
            variational_distribution=variational_distribution,
            variational_params_initialization=variational_parameter_initialization,
            n_samples_per_iter=n_samples_per_iter,
            variational_transformation=variational_transformation,
            random_seed=random_seed,
            max_feval=max_feval,
            natural_gradient=False,  # mean-only natural gradient is applied internally
            FIM_dampening=False,
            decay_start_iter=50,
            dampening_coefficient=1e-2,
            FIM_dampening_lower_bound=1e-8,
            stochastic_optimizer=stochastic_optimizer,
            iteration_data=iteration_data,
            verbose_every_n_iter=verbose_every_n_iter,
        )
        self.natural_gradient_mean_only = natural_gradient_mean_only
        self.score_function_bool = False
        self._last_grad_norm = np.nan
        self._last_entropy = None
        self._last_log_lik = None

        # Optional ground truth for the reconstruction error (the paper's posterior-mean
        # discrepancy eps_pm = ||mu - x_gt|| / ||x_gt||), recorded each iteration when set.
        self._x_gt = None
        self._x_gt_norm = None
        if ground_truth_file is not None:
            self._x_gt = np.load(ground_truth_file).flatten()
            self._x_gt_norm = float(np.linalg.norm(self._x_gt))
            _logger.info(
                "Loaded ground truth (%d DoFs) from %s", self._x_gt.size, ground_truth_file
            )

    def pre_run(self):
        """Seed the RNG, route package logging to QUEENS' handlers, and initialize params."""
        np.random.seed(self.random_seed)
        # QUEENS attaches its file/console handlers to the "queens" logger (not the root
        # logger), so this package's per-iteration diagnostics (recon_err, pred_res, ...) would
        # otherwise be dropped. Mirror those handlers onto the "high_dim_svi" logger once.
        package_logger = logging.getLogger("high_dim_svi")
        if not package_logger.handlers:
            queens_logger = logging.getLogger("queens")
            for handler in queens_logger.handlers:
                package_logger.addHandler(handler)
            package_logger.setLevel(queens_logger.level or logging.INFO)
            package_logger.propagate = False
        super().pre_run()

    def core_run(self):
        """Run the sparse-precision SVI optimisation loop."""
        _logger.info("Starting high-dimensional sparse-precision SVI...")
        super().core_run()

    def _initialize_variational_params(self):
        """Initialize variational parameters from a file, a dict, or the base options.

        * ``.npy`` path -> load the stored parameter vector (e.g. a prior-derived init);
        * dict ``{"mean", "variance", "off_diag_range"}`` -> build from a diagonal
          covariance via the variational family;
        * ``"random"`` / ``"prior"`` -> delegate to the base implementation.
        """
        init = self.variational_params_initialization_approach
        if isinstance(init, str) and init.endswith(".npy"):
            self.variational_params = np.load(init).flatten()
            n_expected = self.variational_distribution.n_parameters
            if self.variational_params.size != n_expected:
                raise ValueError(
                    f"Loaded variational parameters have size {self.variational_params.size}, "
                    f"but the distribution expects {n_expected}."
                )
            _logger.info("Loaded initial variational parameters from %s", init)
        elif isinstance(init, dict):
            dimension = self.variational_distribution.dimension
            mean = np.full(dimension, init["mean"], dtype=float)
            variance = np.full(dimension, init["variance"], dtype=float)
            self.variational_params = (
                self.variational_distribution.construct_variational_parameters(
                    mean, variance, off_diag_range=init.get("off_diag_range", 0.0)
                )
            )
            _logger.info("Initialized variational parameters from prior moments (dict).")
        else:
            super()._initialize_variational_params()

    def _calculate_elbo_gradient(self, variational_parameters):
        """Compute the ELBO gradient (STL estimator + mean-only natural gradient).

        Args:
            variational_parameters (np.ndarray): Current variational parameters.

        Returns:
            np.ndarray: ELBO gradient, shape ``(n_parameters,)``.
        """
        self.variational_params = variational_parameters.flatten()
        dist = self.variational_distribution

        sample_batch, _ = dist.conduct_reparameterization(
            self.variational_params, self.n_samples_per_iter
        )
        grad_variational_batch = dist.grad_logpdf_sample(
            sample_batch, self.variational_params
        ).reshape(self.n_samples_per_iter, -1)

        log_likelihood_batch, grad_log_likelihood_batch = self.evaluate_and_gradient(sample_batch)

        # Direct ELBO gradient for the precision parameterization (sticking-the-landing).
        grad_elbo = dist.compute_direct_elbo_gradient(
            sample_batch,
            grad_log_likelihood_batch,
            grad_variational_batch,
            self.variational_params,
        )

        # Mean-only natural gradient: nat_grad_mu = Q^{-1} grad_mu = L_Q^{-T} (L_Q^{-1} grad_mu).
        if self.natural_gradient_mean_only:
            grad_elbo = self._apply_mean_natural_gradient(grad_elbo)

        # ELBO = entropy + E[log p(y|x)]; the prior term is added inside the C++ adjoint
        # (gradient only), so the Python-side log-prior contribution to the ELBO is zero.
        log_unnormalized_posterior = float(np.nansum(log_likelihood_batch)) / self.n_samples_per_iter
        self._calculate_elbo(log_unnormalized_posterior, self.variational_params)

        self._last_grad_norm = float(np.linalg.norm(grad_elbo))
        self._record_convergence_metrics(log_likelihood_batch)
        return grad_elbo

    def _apply_mean_natural_gradient(self, grad_elbo):
        """Precondition the mean block of the gradient with the variational precision.

        ``nat_grad_mu = Q⁻¹ grad_mu`` via two sparse triangular solves, delegated to the
        variational family's :meth:`~high_dim_svi.sparse_inverse_variational.SparseInverseNormal.apply_inverse_precision`
        (which reuses the Cholesky factors cached during the reparameterization).

        Args:
            grad_elbo (np.ndarray): Raw ELBO gradient, shape ``(n_parameters,)``.

        Returns:
            np.ndarray: Gradient with a natural-gradient mean block.
        """
        dim = self.variational_distribution.dimension
        grad_elbo[:dim] = self.variational_distribution.apply_inverse_precision(grad_elbo[:dim])
        return grad_elbo

    def _calculate_elbo(self, log_unnormalized_posterior_mean, variational_parameters):
        """Set the current ELBO ``= entropy(q) + E_q[log p(y|x)]``.

        Args:
            log_unnormalized_posterior_mean (float): MC estimate of ``E_q[log p(y|x)]``.
            variational_parameters (np.ndarray): Current variational parameters.
        """
        entropy = self.variational_distribution.entropy(variational_parameters)
        elbo = entropy + log_unnormalized_posterior_mean
        self.iteration_data.add(elbo=elbo)
        self.elbo = elbo
        self._last_entropy = entropy
        self._last_log_lik = log_unnormalized_posterior_mean

    def evaluate_and_gradient(self, sample_batch):
        """Evaluate the log-likelihood and its gradient w.r.t. the latent field.

        Args:
            sample_batch (np.ndarray): Samples, shape ``(M, dimension)``.

        Returns:
            tuple[np.ndarray, np.ndarray]: Log-likelihood per sample ``(M,)`` and its
            gradient w.r.t. the samples ``(M, dimension)`` (the full log-posterior
            gradient, since the prior gradient is added by the adjoint).
        """
        log_likelihood, grad_log_likelihood = self.model.evaluate_and_gradient(
            sample_batch.reshape(-1, self.num_parameters)
        )
        self.iteration_data.add(n_sims=self.model.num_evaluations, samples=sample_batch)
        return log_likelihood, grad_log_likelihood

    def _record_convergence_metrics(self, log_likelihood_batch):
        """Record cheap per-iteration diagnostics into ``iteration_data``.

        Args:
            log_likelihood_batch (np.ndarray): Log-likelihood values per sample.
        """
        noise_variance = float(self.model.cov_factor)
        forward_output = self.model.response["forward_model_output"]
        y_mean = np.mean(forward_output, axis=0)
        predictive_residual = float(
            np.linalg.norm(y_mean - self.model.y_obs) / np.linalg.norm(self.model.y_obs)
        )

        # Reconstruction error eps_pm = ||mu - x_gt|| / ||x_gt|| (posterior-mean discrepancy);
        # mu is the variational mean = first `dimension` variational parameters.
        reconstruction_error = np.nan
        if self._x_gt is not None:
            mean = self.variational_params[: self.variational_distribution.dimension]
            reconstruction_error = float(np.linalg.norm(mean - self._x_gt) / self._x_gt_norm)

        self.iteration_data.add(
            tau_tilde=1.0 / noise_variance,
            noise_variance=noise_variance,
            grad_norm=self._last_grad_norm,
            predictive_residual=predictive_residual,
            reconstruction_error=reconstruction_error,
        )

    def _verbose_output(self):
        """Log ELBO components and convergence diagnostics."""
        super()._verbose_output()
        parts = []
        if self._last_entropy is not None:
            parts.append(f"entropy: {self._last_entropy:.4f}")
        if self._last_log_lik is not None:
            parts.append(f"E[log p]: {self._last_log_lik:.4f}")
        if np.isfinite(self._last_grad_norm):
            parts.append(f"|grad|: {self._last_grad_norm:.4e}")
        if getattr(self.iteration_data, "noise_variance", None):
            parts.append(f"sigma^2: {self.iteration_data.noise_variance[-1]:.4e}")
        if getattr(self.iteration_data, "predictive_residual", None):
            parts.append(f"pred_res: {self.iteration_data.predictive_residual[-1]:.6f}")
        if getattr(self.iteration_data, "reconstruction_error", None):
            recon_error = self.iteration_data.reconstruction_error[-1]
            if np.isfinite(recon_error):
                parts.append(f"recon_err (eps_pm): {recon_error:.6f}")
        if parts:
            _logger.info("         %s", " | ".join(parts))
