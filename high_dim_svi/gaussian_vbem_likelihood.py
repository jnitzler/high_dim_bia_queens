"""Isotropic Gaussian likelihood with a VB-EM noise-variance update.

The latest QUEENS :class:`queens.models.likelihoods.gaussian.Gaussian` builds a
*dense* ``eye(n_obs)`` covariance for the ``MAP_jeffrey_variance`` noise model,
which is infeasible for the thousands of observations in the 3D reconstruction
problem. This class is the high-dimensional-friendly variant used in the paper:
the observation noise is a single **scalar** variance ``sigma^2`` (isotropic), so
the log-density and its gradient are evaluated without ever forming a matrix.

The scalar noise variance is updated each iteration by the closed-form MAP
estimate under a Jeffreys prior (the "E/M-step" of the VB-EM scheme),
``sigma^2 = (b0 + 0.5 * sum r^2) / (a0 + 0.5 * n_r)`` with ``a0 = b0 = 1e-9`` and
``r`` the model-minus-observation residuals, optionally smoothed with exponential
iterative averaging.
"""

import logging

import numpy as np

from queens.models.likelihoods._likelihood import Likelihood
from queens.utils.logger_settings import log_init_args

_logger = logging.getLogger(__name__)

# Jeffreys-prior hyperparameters for the noise precision (weakly informative).
_A0 = 1.0e-9
_B0 = 1.0e-9


class GaussianVBEM(Likelihood):
    r"""Gaussian likelihood with a scalar (isotropic) noise variance and VB-EM updates.

    ``log p(y | x) = -0.5 N log(2 pi sigma^2) - 0.5 ||f(x) - y||^2 / sigma^2``,
    where ``N = y.size`` and ``sigma^2`` (``cov_factor``) is re-estimated each call
    for ``MAP_jeffrey_variance``.

    Attributes:
        cov_factor (float): Current scalar noise variance ``sigma^2``.
        noise_type (str): ``"MAP_jeffrey_variance"`` (adaptive) or ``"fixed_variance"``.
        nugget_noise_variance (float): Lower bound on ``cov_factor``.
        noise_var_iterative_averaging (obj | None): Optional averaging of ``cov_factor``.
    """

    @log_init_args
    def __init__(
        self,
        forward_model,
        y_obs,
        noise_type: str = "MAP_jeffrey_variance",
        noise_value: float | None = None,
        nugget_noise_variance: float = 1.0e-9,
        noise_var_iterative_averaging=None,
    ) -> None:
        """Initialize the likelihood.

        Args:
            forward_model: Forward model (here the adjoint-enabled deal.II model).
            y_obs (np.ndarray): Flattened observation vector. Its ordering must match the
                Fortran-order flattening of the forward-model output (see :meth:`_evaluate`).
            noise_type: ``"MAP_jeffrey_variance"`` for the adaptive VB-EM update or
                ``"fixed_variance"`` to keep ``noise_value`` constant.
            noise_value: Fixed scalar variance (required for ``"fixed_variance"``).
            nugget_noise_variance: Lower bound on the noise variance.
            noise_var_iterative_averaging: Optional ``IterativeAveraging`` object applied
                to ``cov_factor`` after each MAP update (e.g. ``ExponentialAveraging``).
        """
        super().__init__(forward_model, y_obs)

        if noise_type == "fixed_variance":
            if noise_value is None:
                raise ValueError("'noise_value' is required for noise_type='fixed_variance'.")
            self.cov_factor = float(noise_value)
        elif noise_type == "MAP_jeffrey_variance":
            self.cov_factor = 1.0  # overwritten by the first VB-EM update
        else:
            raise NotImplementedError(
                f"noise_type '{noise_type}' is not supported by GaussianVBEM."
            )

        self.noise_type = noise_type
        self.nugget_noise_variance = nugget_noise_variance
        self.noise_var_iterative_averaging = noise_var_iterative_averaging

    def _evaluate(self, samples: np.ndarray) -> dict:
        """Evaluate the log-likelihood for a batch of input samples.

        The forward-model output is flattened per sample in Fortran order to match the
        observation vector ``y_obs``; ``y_obs`` must be built with the same convention
        in the driving script.

        Args:
            samples (np.ndarray): Input samples, shape ``(M, dimension)``.

        Returns:
            dict: ``{"result": log_likelihood}`` with ``log_likelihood`` of shape ``(M,)``.
        """
        num_samples = samples.shape[0]
        output = self.forward_model.evaluate(samples)
        forward_model_output = output["result"][:num_samples].reshape(num_samples, -1, order="F")

        if self.noise_type == "MAP_jeffrey_variance":
            self._update_noise_variance(forward_model_output)

        log_likelihood = self._isotropic_logpdf(forward_model_output)

        # Cache the forward output for grad() and the iterator's convergence diagnostics.
        self.response = {
            "forward_model_output": forward_model_output,
            "log_likelihood": log_likelihood,
        }
        return {"result": log_likelihood}

    def grad(self, samples: np.ndarray, upstream_gradient: np.ndarray) -> np.ndarray:
        r"""Chain the likelihood gradient through the forward model.

        The gradient of the log-likelihood w.r.t. the model output is
        ``-(f(x) - y) / sigma^2``; this is handed to the adjoint forward model as its
        upstream gradient. The ``upstream_gradient`` argument (the ones-vector passed by
        ``Model.evaluate_and_gradient``) is intentionally not used: the likelihood is the
        top of the objective, so it defines the upstream gradient itself.

        Args:
            samples (np.ndarray): Input samples, shape ``(M, dimension)``.
            upstream_gradient (np.ndarray): Unused (see above).

        Returns:
            np.ndarray: Gradient of ``log p(y | x)`` w.r.t. the input samples.
        """
        forward_model_output = self.response["forward_model_output"]
        log_likelihood_grad = -(forward_model_output - self.y_obs[np.newaxis, :]) / self.cov_factor
        return self.forward_model.grad(samples, log_likelihood_grad)

    def _isotropic_logpdf(self, y_model: np.ndarray) -> np.ndarray:
        """Isotropic Gaussian log-density per sample.

        Args:
            y_model (np.ndarray): Forward-model output, shape ``(M, N)``.

        Returns:
            np.ndarray: Log-likelihood per sample, shape ``(M,)``.
        """
        n_obs = self.y_obs.size
        residual = y_model - self.y_obs[np.newaxis, :]
        return -0.5 * n_obs * np.log(2.0 * np.pi * self.cov_factor) - 0.5 * np.sum(
            residual**2, axis=1
        ) / self.cov_factor

    def _update_noise_variance(self, y_model: np.ndarray) -> None:
        """MAP/Jeffreys VB-EM update of the scalar noise variance ``cov_factor``.

        Args:
            y_model (np.ndarray): Forward-model output, shape ``(M, N)``.
        """
        residual = y_model - self.y_obs[np.newaxis, :]
        aa = _A0 + 0.5 * residual.size  # samples x observation points
        bb = _B0 + 0.5 * np.sum(residual**2)
        cov_factor = max(bb / aa, self.nugget_noise_variance)

        if self.noise_var_iterative_averaging is not None:
            cov_factor = self.noise_var_iterative_averaging.update_average(cov_factor)

        self.cov_factor = cov_factor
        _logger.debug("VB-EM noise variance update: sigma^2 = %.6e", self.cov_factor)
