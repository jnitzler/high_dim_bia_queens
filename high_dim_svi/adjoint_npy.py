"""Adjoint model that exchanges gradient data with deal.II through ``.npy`` files.

QUEENS' :class:`queens.models.adjoint.Adjoint` writes the upstream gradient as a
CSV file. The deal.II ``darcy_adjoint`` executable instead reads it from a NumPy
binary ``adjoint_data.npy`` located in the same job directory as the field input
``.npy`` (see the deal.II project's ``CLAUDE.md``). This subclass overrides
:meth:`grad` to write that file with ``np.save`` while keeping the rest of the
adjoint workflow (re-running the gradient driver in the forward job directories).

The deal.II adjoint returns the gradient of the *log-posterior* w.r.t. the field,
i.e. ``d/dx [log p(y|x) + log p(x)]``: the SPDE/GMRF prior gradient is assembled
in C++, so the Python side treats the returned gradient as the full log-posterior
gradient (and uses a flat prior for the QUEENS ``Parameters`` object).
"""

import logging

import numpy as np

from queens.models.adjoint import Adjoint
from queens.utils.config_directories import current_job_directory

_logger = logging.getLogger(__name__)


class AdjointNpy(Adjoint):
    """Adjoint model writing the upstream gradient to ``adjoint_data.npy``."""

    def grad(self, samples: np.ndarray, upstream_gradient: np.ndarray) -> np.ndarray:
        r"""Evaluate the model gradient via the adjoint solver.

        Writes the per-sample upstream gradient ``d g / d y`` (here the
        log-likelihood gradient w.r.t. the observation-point outputs) to
        ``adjoint_data.npy`` in each forward job directory, then re-runs the gradient
        driver there to obtain ``d g / d x`` (log-posterior gradient w.r.t. the field).

        Args:
            samples (np.ndarray): Input samples, shape ``(M, dimension)``.
            upstream_gradient (np.ndarray): Upstream gradient ``d g / d y``, shape
                ``(M, n_obs)``.

        Returns:
            np.ndarray: Gradient ``d g / d x``, shape ``(M, dimension)``.
        """
        num_samples = samples.shape[0]
        # The forward runs were the most recent jobs; reuse their directories.
        last_job_ids = [self.scheduler.next_job_id - num_samples + i for i in range(num_samples)]
        experiment_dir = self.scheduler.experiment_dir

        for job_id, grad_objective in zip(last_job_ids, upstream_gradient, strict=True):
            job_dir = current_job_directory(experiment_dir, job_id)
            adjoint_file_path = job_dir.joinpath(self.adjoint_file)
            np.save(adjoint_file_path, np.asarray(grad_objective).reshape(-1))

        gradient = self.create_result_dict_from_scheduler_output(
            self.scheduler.evaluate(samples, self.gradient_driver, job_ids=last_job_ids)
        )["result"]
        return gradient
