"""Script-based demo: high-dimensional Bayesian field reconstruction via sparse SVI.

Reference run for the paper *"Scalable High-Dimensional Bayesian Field Reconstruction
with Finite Elements: Application to 3D Porous Media Flow"*. It infers a
high-dimensional (``~4e5`` DoF) log-permeability random field of a 3D Darcy flow on
an eccentric hyper-shell ("donut") domain from noisy point-wise velocity
observations, using stochastic variational inference with a sparse
precision-parameterized Gaussian posterior.

The forward PDE solve and the adjoint (log-posterior gradient, *including* the
SPDE/GMRF prior gradient) are provided by two compiled deal.II executables
(``darcy_forward`` / ``darcy_adjoint``); QUEENS orchestrates their parallel
execution and this script drives the SVI optimisation in pure Python.

Configuration -- the paper *baseline* (Table 3, row (1) of Table 4)
-------------------------------------------------------------------
The constants below reproduce the **baseline configuration** of the paper, executed
**locally** (QUEENS ``Local`` scheduler + MPI) instead of on the SLURM cluster. The
values are taken from the inverse-problem setup in Table 3 and the cluster input file
``darcy_flow_3d_cluster.yml``:

    samples per batch        n_batch = 4
    forward/adjoint calls    2000 each  (= 500 SVI iterations x 4 samples)
    stochastic optimizer     Adam, learning rate 5e-2
    variational family       sparse precision-Cholesky Gaussian, prior-mean 0.1
    L_Q initialization       incomplete Cholesky of the prior precision (file)
    prior  p(x|delta)        GMRF, kappa^2 = 1e-4, delta ~ Gamma(1e-9, 1e-9)   <- in C++
    likelihood p(y|x,tau)    isotropic Gaussian, tau ~ Gamma(1e-9, 1e-9), VB-EM
    observation noise        i.i.d. Gaussian at SNR = 20 (Sec. 3.2)

**One deliberate deviation from the headline baseline.** The baseline of Algorithm 1
applies a natural gradient to *both* the mean and the precision-Cholesky parameters
(the latter via Takahashi selected inversion). This package implements only the
**mean-only natural gradient** -- which the ablation (Table 4) reports as
statistically equivalent to the full baseline (eps_pm = 34.6 % for mean-only, row (3),
versus 35.3 % for the full baseline, row (1); "the cheaper mean-only variant suffices
in practice"). The Takahashi precision natural gradient is intentionally out of scope
here (see ``CLAUDE.md``). The cluster file's ``freeze_covariance`` / ``_dirac`` setting
is the *Laplace/MAP comparison* (row (14) of Table 4), **not** the baseline, and is not
reproduced.

Prerequisites:
    * QUEENS installed and importable (``conda activate queens``);
    * the deal.II executables built under ``<DEALII_PROJECT_DIR>/build/release``; and
    * the referenced data files present (set the paths in the CONFIGURATION block).

Quick smoke test (a few iterations instead of the full 500):
    DARCY_SVI_MAX_FEVAL=12 DARCY_SVI_VERBOSE_EVERY=1 python darcy_svi_demo.py
"""

import os
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# --------------------------------------------------------------------------------------
# CONFIGURATION -- adjust these paths/constants to your environment
# --------------------------------------------------------------------------------------
# Directory of the deal.II Darcy project (executables, input template, FE-operator data).
DEALII_PROJECT_DIR = Path("/home/nitzler/workspace/Own_dealii_projects/darcy_flow_3d_donut")

# Compiled deal.II executables (the locally built release binaries).
EXECUTABLE_DIR = DEALII_PROJECT_DIR / "build" / "release"
FORWARD_EXECUTABLE = EXECUTABLE_DIR / "darcy_forward"
ADJOINT_EXECUTABLE = EXECUTABLE_DIR / "darcy_adjoint"

# deal.II JSON input template (placeholders: input_file_path, output_directory, output_prefix).
# Also fixes the in-C++ prior constants: kappa^2 ("nugget" = 1e-4) and the VB-EM delta update
# ("fixed prior precision": false). The Python side adds no prior gradient (see CLAUDE.md).
INPUT_TEMPLATE = DEALII_PROJECT_DIR / "parameters_template.json"

# Sparsity pattern PREFIX: loads <prefix>_row_idx.npy and <prefix>_col_idx.npy. The pattern
# is the FE Laplacian connectivity and defines the nonzero structure of the precision factor.
SPARSITY_PATTERN_PREFIX = str(DEALII_PROJECT_DIR / "rf_sparsity")

# Initial variational parameters [mu, lambda_{L_Q}]. This file is the paper's baseline
# initialization: the *incomplete Cholesky of the prior precision* with prior mean mu_x = 0.1
# (Table 3). Layout: 405570 mean + 405570 diagonal + 12478560 off-diagonal = 13289700 entries.
INITIAL_VARIATIONAL_PARAMS = str(DEALII_PROJECT_DIR / "initial_variational_params_inverse.npy")

# Ground-truth log-permeability field x_gt used to monitor the reconstruction error
# eps_pm = ||mu - x_gt|| / ||x_gt|| (the paper's posterior-mean discrepancy, Sec. 3.4 / Fig. 9a).
# This is a diagnostic only -- it does not enter the inference. (cluster YAML: ground_truth_file.)
GROUND_TRUTH_FILE = str(DEALII_PROJECT_DIR / "output" / "ground_truth" / "new_random_field.npy")

# Experimental observations (point-wise velocity components u_1, u_2, u_3), bundled in data/.
# These are the *clean* ground-truth velocities; SNR-20 noise is added in load_observations().
OBSERVATIONS_DIR = Path("./data")
OBSERVATIONS_FILE = "observations.csv"

# Output directory for results (pickle, plots, posterior fields).
OUTPUT_DIR = Path("./output")

# --- Problem / algorithm constants (paper baseline; Table 3 + darcy_flow_3d_cluster.yml) ---
DIMENSION = 405570            # number of random-field DoFs (must match sparsity/init files)
N_SAMPLES_PER_ITER = 4        # Monte-Carlo batch size per SVI iteration (Table 3: n_batch = 4)
LEARNING_RATE = 0.05          # Adam step size (Table 3: 5e-2)
RANDOM_SEED = 4               # seed for the reparameterization RNG (cluster YAML: random_seed 4)
NUGGET_VAR_DIAG = 1.0e-9      # diagonal nugget of the variational precision factor (cluster YAML)
NUGGET_NOISE_VARIANCE = 1.0e-9  # lower bound on the (VB-EM) noise variance (a0 = b0 = 1e-9)
NOISE_AVG_COEFFICIENT = 0.9   # exponential averaging coefficient for the noise variance

# Local MPI resources. This machine has only 6 physical cores, so -- unlike the cluster, which
# used 16 MPI ranks per solve (darcy_flow_3d_cluster.yml) -- we use 3 ranks per deal.II solve and
# run 2 solves concurrently: NUM_PARALLEL_JOBS x NUM_PROCS_PER_JOB = 2 x 3 = 6 cores, no
# oversubscription. The N_SAMPLES_PER_ITER = 4 samples of each batch then run in two waves of two.
# On a bigger machine, raise NUM_PARALLEL_JOBS (ideally to N_SAMPLES_PER_ITER) and/or NUM_PROCS_PER_JOB.
NUM_PROCS_PER_JOB = 3         # MPI ranks per forward/adjoint solve
NUM_PARALLEL_JOBS = 2         # concurrent solves (Local scheduler workers); 2 x 3 = 6 cores

# Run length. MAX_FEVAL counts forward-model evaluations; with N_SAMPLES_PER_ITER = 4 the paper
# baseline of 500 iterations corresponds to 2000 evaluations (Table 3: "2000 solver calls each").
# Override via the environment for a quick smoke test (see the module docstring).
MAX_FEVAL = int(os.environ.get("DARCY_SVI_MAX_FEVAL", 2000))
VERBOSE_EVERY_N_ITER = int(os.environ.get("DARCY_SVI_VERBOSE_EVERY", 10))

# Synthetic observation noise. The bundled observations.csv holds the *clean* ground-truth
# velocity field; the paper corrupts it with i.i.d. Gaussian noise at a signal-to-noise ratio
# SNR = 20 (Sec. 3.2), i.e. a noise standard deviation of (signal RMS) / SNR -- a relative noise
# amplitude of 1/SNR = 5 %, the "irreducible noise floor" the predictive residual approaches.
# A fixed seed makes the corruption reproducible. Set OBS_SNR = None to use the data as-is.
OBS_SNR = 20.0
OBS_NOISE_SEED = 42

# Posterior post-processing.
N_POSTERIOR_SAMPLES = 100     # samples used to estimate the marginal posterior std

# Validate the configuration early with informative messages.
for _path, _desc in [
    (FORWARD_EXECUTABLE, "forward executable (darcy_forward)"),
    (ADJOINT_EXECUTABLE, "adjoint executable (darcy_adjoint)"),
    (INPUT_TEMPLATE, "deal.II input template"),
    (Path(f"{SPARSITY_PATTERN_PREFIX}_row_idx.npy"), "sparsity pattern row indices"),
    (Path(f"{SPARSITY_PATTERN_PREFIX}_col_idx.npy"), "sparsity pattern column indices"),
    (Path(INITIAL_VARIATIONAL_PARAMS), "initial variational parameters"),
    (Path(GROUND_TRUTH_FILE), "ground-truth field (reconstruction-error diagnostic)"),
    (OBSERVATIONS_DIR / OBSERVATIONS_FILE, "observations CSV"),
    (OUTPUT_DIR, "output directory"),
]:
    assert _path.exists(), f"Missing {_desc}: {_path}"

# --------------------------------------------------------------------------------------
# QUEENS imports (reused as-is) and custom high_dim_svi modules
# --------------------------------------------------------------------------------------
from queens.data_processors.csv_file import CsvFile
from queens.data_processors.numpy_file import NumpyFile
from queens.distributions.free_variable import FreeVariable
from queens.global_settings import GlobalSettings
from queens.main import run_iterator
from queens.parameters import Parameters
from queens.schedulers.local import Local
from queens.stochastic_optimizers.adam import Adam
from queens.utils.iterative_averaging import ExponentialAveraging

from high_dim_svi import AdjointNpy, DealDriver, GaussianVBEM, HighDimRPVI, SparseInverseNormal


def load_observations() -> np.ndarray:
    """Load the velocity observations and add reproducible SNR-20 Gaussian noise.

    The deal.II solution is stored component-major (all ``u_1`` values, then all ``u_2``,
    then all ``u_3``); ``observations.csv`` mirrors this with one column per component. The
    observation vector is assembled in the same component-major order so it lines up with the
    forward model output read back from ``*_sol.npy``.

    The bundled CSV contains the *noise-free* ground-truth velocities, so the synthetic
    measurement noise of the paper (Sec. 3.2) is added here: i.i.d. Gaussian with standard
    deviation ``(signal RMS) / SNR`` (a relative amplitude of ``1/SNR``), under a fixed seed.

    Returns:
        np.ndarray: Flattened observation vector (component-major), with noise.
    """
    csv_reader = CsvFile(
        file_name_identifier=OBSERVATIONS_FILE,
        file_options_dict={
            "header_row": 0,
            "index_column": False,
            "returned_filter_format": "dict",
            "filter": {"type": "entire_file"},
        },
    )
    data = csv_reader.get_data_from_file(OBSERVATIONS_DIR.resolve())
    output_labels = ["u_1", "u_2", "u_3"]
    y_obs = np.array(
        [np.array(data[label]).reshape(-1) for label in output_labels]
    ).reshape(-1, order="C")

    if OBS_SNR is not None and OBS_SNR > 0.0:
        signal_mean_square = np.sum(y_obs**2) / y_obs.size
        noise_std = np.sqrt(signal_mean_square) / OBS_SNR  # sigma = RMS / SNR
        rng = np.random.default_rng(OBS_NOISE_SEED)
        y_obs = y_obs + rng.normal(0.0, noise_std, size=y_obs.shape)

    return y_obs


def post_process(variational_distribution: SparseInverseNormal, result_file: Path) -> None:
    """Plot ELBO convergence and export posterior mean / mean +/- std fields.

    The exported ``.npy`` fields can be fed back to the deal.II solver and visualised in
    ParaView (run the forward model once on each field to write the corresponding ``.pvtu``).

    Args:
        variational_distribution: The fitted variational family.
        result_file: Path to the results pickle written by the iterator.
    """
    with open(result_file, "rb") as stream:
        results = pickle.load(stream)

    iteration_data = results["iteration_data"]

    # ELBO convergence.
    fig, axis = plt.subplots()
    axis.plot(iteration_data["elbo"])
    axis.set_xlabel("iteration")
    axis.set_ylabel("ELBO")
    axis.set_title("ELBO convergence")
    fig.savefig(OUTPUT_DIR / "elbo_convergence.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Reconstruction error eps_pm and predictive residual eps_pred (paper Sec. 3.4 / Fig. 9):
    # the mean error eps_pm = ||mu - x_gt|| / ||x_gt|| is the headline convergence metric.
    fig, axis = plt.subplots()
    if "reconstruction_error" in iteration_data:
        axis.plot(
            np.asarray(iteration_data["reconstruction_error"]) * 100.0,
            marker=".", label=r"$\varepsilon_{\mathrm{pm}}$ (reconstruction error)",
        )
    if "predictive_residual" in iteration_data:
        axis.plot(
            np.asarray(iteration_data["predictive_residual"]) * 100.0,
            marker=".", label=r"$\varepsilon_{\mathrm{pred}}$ (predictive residual)",
        )
    axis.set_xlabel("iteration")
    axis.set_ylabel("relative error [%]")
    axis.set_title("Reconstruction error / predictive residual")
    axis.legend()
    fig.savefig(OUTPUT_DIR / "error_convergence.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    final_params = results["final_variational_parameters"]
    # Posterior mean is exact (first d variational parameters); estimate the marginal std
    # from a small sample (full marginal variance would need a selected sparse inverse).
    posterior_mean, _ = variational_distribution.reconstruct_distribution_parameters(
        final_params, return_cholesky=True
    )
    posterior_mean = posterior_mean.flatten()
    samples = variational_distribution.draw(final_params, n_draws=N_POSTERIOR_SAMPLES)
    posterior_std = np.std(samples, axis=0)

    np.save(OUTPUT_DIR / "posterior_mean_field.npy", posterior_mean)
    np.save(OUTPUT_DIR / "posterior_mean_plus_std_field.npy", posterior_mean + posterior_std)
    np.save(OUTPUT_DIR / "posterior_mean_minus_std_field.npy", posterior_mean - posterior_std)


if __name__ == "__main__":
    experiment_name = "darcy_svi_baseline_local"
    global_settings = GlobalSettings(experiment_name=experiment_name, output_dir=OUTPUT_DIR)

    # Latent field with a flat (improper) prior: the actual SPDE/GMRF prior is assembled in
    # the deal.II adjoint, so the QUEENS-side prior must not contribute a gradient.
    parameters = Parameters(field=FreeVariable(dimension=DIMENSION))

    # Data processors: forward velocity at observation points, and adjoint field gradient.
    forward_data_processor = NumpyFile(
        file_name_identifier="*_sol.npy", file_options_dict={"delete_field_data": False}
    )
    gradient_data_processor = NumpyFile(
        file_name_identifier="*_grad_solution.npy", file_options_dict={"delete_field_data": False}
    )

    # Forward and adjoint drivers share the deal.II output prefix so the adjoint finds the
    # forward's "..._solution_full.npy" in the (shared) job directory.
    forward_driver = DealDriver(
        parameters,
        INPUT_TEMPLATE,
        FORWARD_EXECUTABLE,
        data_processor=forward_data_processor,
        output_prefix="darcy_",
    )
    adjoint_driver = DealDriver(
        parameters,
        INPUT_TEMPLATE,
        ADJOINT_EXECUTABLE,
        data_processor=gradient_data_processor,
        output_prefix="darcy_",
    )

    y_obs = load_observations()

    with global_settings:
        # Local scheduler: run the per-iteration batch of forward solves in parallel.
        # overwrite_existing_experiment=True avoids an interactive prompt on re-runs.
        scheduler = Local(
            experiment_name,
            num_jobs=NUM_PARALLEL_JOBS,
            num_procs=NUM_PROCS_PER_JOB,
            restart_workers=False,
            verbose=True,
            overwrite_existing_experiment=True,
        )

        # Adjoint-enabled forward model (writes adjoint_data.npy, returns the log-posterior
        # gradient via the adjoint solve).
        forward_model = AdjointNpy(
            scheduler=scheduler,
            driver=forward_driver,
            gradient_driver=adjoint_driver,
            adjoint_file="adjoint_data.npy",
        )

        # Isotropic Gaussian likelihood with the VB-EM (MAP/Jeffreys) noise-variance update.
        likelihood = GaussianVBEM(
            forward_model,
            y_obs,
            noise_type="MAP_jeffrey_variance",
            nugget_noise_variance=NUGGET_NOISE_VARIANCE,
            noise_var_iterative_averaging=ExponentialAveraging(coefficient=NOISE_AVG_COEFFICIENT),
        )

        # Sparse precision-Cholesky variational family (sparsity from the FE Laplacian).
        variational_distribution = SparseInverseNormal(
            DIMENSION,
            sparsity_pattern_file=SPARSITY_PATTERN_PREFIX,
            nugget_var_diag=NUGGET_VAR_DIAG,
        )

        # Adam (the baseline optimizer; AdaMax is the slower ablation row (7) in Table 4).
        # rel_l1/l2_change_threshold = -1 disables early stopping (run to max_feval).
        optimizer = Adam(
            learning_rate=LEARNING_RATE,
            optimization_type="max",
            rel_l1_change_threshold=-1,
            rel_l2_change_threshold=-1,
        )

        result_description = {
            "iterative_field_names": [
                "elbo",
                "noise_variance",
                "grad_norm",
                "predictive_residual",
                "reconstruction_error",  # eps_pm = ||mu - x_gt|| / ||x_gt|| (the paper's mean error)
            ],
            "write_results": True,
        }

        iterator = HighDimRPVI(
            likelihood,
            parameters,
            global_settings,
            result_description,
            variational_distribution,
            n_samples_per_iter=N_SAMPLES_PER_ITER,
            random_seed=RANDOM_SEED,
            max_feval=MAX_FEVAL,
            stochastic_optimizer=optimizer,
            variational_parameter_initialization=INITIAL_VARIATIONAL_PARAMS,
            ground_truth_file=GROUND_TRUTH_FILE,
            natural_gradient_mean_only=True,
            verbose_every_n_iter=VERBOSE_EVERY_N_ITER,
        )

        run_iterator(iterator, global_settings=global_settings)

    post_process(variational_distribution, global_settings.result_file(".pickle"))
    print("Finished high-dimensional SVI reconstruction.")

# ======================================================================================
# CLUSTER ALTERNATIVE (SLURM) -- sketch based on darcy_flow_3d_cluster.yml
# --------------------------------------------------------------------------------------
# For large overnight runs, replace the `Local` scheduler above with QUEENS' SSH-based
# `Cluster` scheduler and a remote connection. The deal.II executables and data files must
# exist on the remote host; adapt the host/user/paths and the SLURM resources to your
# cluster. Sketch (verify argument names against your installed QUEENS version):
#
#   from queens.schedulers.cluster import Cluster
#
#   scheduler = Cluster(
#       experiment_name=experiment_name,
#       workload_manager="slurm",
#       cluster_address="<host>",            # e.g. 129.187.58.22
#       cluster_user="<user>",
#       cluster_python_path="/home/<user>/miniforge3/envs/queens/bin/python",
#       cluster_queens_repository="/home/<user>/workspace/queens",
#       num_jobs=4, num_procs=16, num_nodes=1,
#       walltime="20:00:00", queue="normal",
#   )
#
# Point FORWARD_EXECUTABLE / ADJOINT_EXECUTABLE / data paths at their REMOTE locations, and
# pass a `dask_jobscript_template` (e.g. job_queens.sh) to the drivers if required by your
# QUEENS version. Everything else (likelihood, variational family, iterator) is unchanged.
# ======================================================================================
