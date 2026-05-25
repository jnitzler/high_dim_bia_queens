# High-dimensional Bayesian field reconstruction with sparse SVI

This repository is the official code release for the inference algorithm of the paper:

> Nitzler, J., Bergbauer, M., Koutsourelakis, P.-S., & Wall, W. A. (2026). *Scalable
> High-Dimensional Bayesian Field Reconstruction with Finite Elements: Application to 3D
> Porous Media Flow* (preprint).

It reconstructs a high-dimensional (`~4 × 10⁵` DoF) log-permeability random field of a
3D Darcy flow on an eccentric hyper-shell ("donut") domain from noisy point-wise
velocity observations, using **stochastic variational inference (SVI)** with a
**sparse precision-parameterized Gaussian** posterior.

We use the open-source library [QUEENS](https://github.com/queens-py/queens) to handle
the simulation management and inference (see also the [QUEENS
paper](https://arxiv.org/abs/2508.16316)). The forward and adjoint porous-media-flow
models are efficiently implemented in [deal.II](https://dealii.org/) and are freely
available in this GitHub repository:
**[github.com/jnitzler/porous_media_flow_3d](https://github.com/jnitzler/porous_media_flow_3d)**.

This repository is a *thin extension layer* on top of QUEENS: it adds only the
components that are not part of upstream QUEENS (the sparse-precision variational
family, the SVI iterator, the isotropic VB-EM likelihood, and the deal.II driver/adjoint
glue) and reuses everything else (scheduler, optimizer, data processors, run loop)
directly.

## Method in one paragraph

The approximate posterior `q(x) = N(x | μ, Q⁻¹)` is stored through the lower Cholesky
factor `L_Q` of the **precision** `Q = L_Q L_Qᵀ`, whose sparsity pattern is inherited
from the finite-element Laplacian of the forward problem. Samples are drawn as
`x = μ + L_Q⁻ᵀ ε`, and the ELBO gradient is formed *directly* with the
sticking-the-landing estimator (no reparameterization Jacobian). The mean gradient is
preconditioned by a **mean-only natural gradient** (two sparse triangular solves), and
the observation-noise variance is updated by a closed-form **VB-EM** (MAP/Jeffreys)
step. The forward PDE solve and the adjoint **log-posterior** gradient — including the
SPDE/GMRF prior gradient — are computed by two compiled deal.II executables; the Python
side performs only the SVI updates and diagnostics.

## Repository layout

```
high_dim_svi/                     # the extension package
├── sparse_inverse_variational.py # sparse precision-Cholesky Gaussian family
├── high_dim_rpvi.py              # SVI iterator (direct/STL ELBO grad + natural grad)
├── gaussian_vbem_likelihood.py   # isotropic Gaussian likelihood with VB-EM noise update
├── deal_driver.py                # MPI driver for the deal.II executables
├── adjoint_npy.py                # adjoint model exchanging gradients via .npy files
└── _sparse_linalg.py             # numba kernels (triangular solves, mat-vec)
darcy_svi_demo.py                 # script-based entry point (local run)
pyproject.toml                    # packaging + extra dependencies (numba, matplotlib)
data/observations.csv             # bundled velocity observations
```

## Setup and installation

1. **Get the computational models.** Clone the deal.II forward/adjoint solvers from
   [github.com/jnitzler/porous_media_flow_3d](https://github.com/jnitzler/porous_media_flow_3d)
   and follow its setup and compilation instructions. You will need the built
   `darcy_forward` and `darcy_adjoint` executables and the FE-operator/data files they
   ship or generate: the variational sparsity pattern (`rf_sparsity_{row,col}_idx.npy`),
   the initial variational parameters (`initial_variational_params_inverse.npy`), the
   deal.II input template (`parameters_template.json`), and — for the optional
   reconstruction-error diagnostic — the ground-truth field (`new_random_field.npy`).

   > **⚠️ MPI-rank / DOF-ordering consistency.** deal.II numbers the distributed DOFs *per
   > parallel partition*, so the DOF-ordered `.npy` files — the sparsity pattern
   > (`rf_sparsity_*`), the ground-truth field, and the optional incomplete-Cholesky init — are
   > valid **only at the MPI rank count they were generated with**, and must match
   > `NUM_PROCS_PER_JOB` in `darcy_svi_demo.py`. The paper's data was generated at 16 ranks
   > (cluster); to run locally at a different rank count, regenerate them at *that* count
   > (`export_sparsity`; `darcy_forward` in ground-truth mode; `compute_prior_init.py`). A
   > mismatch silently scrambles the field and the reconstruction fails. `observations.csv` is
   > rank-independent (its output is coordinate-sorted), so it never needs regenerating.
2. **Install and set up QUEENS** by following the instructions in the
   [QUEENS GitHub repository](https://github.com/queens-py/queens), then activate its
   environment (e.g. `conda activate queens`). QUEENS provides numpy and scipy. This
   release was developed and tested against QUEENS at commit
   [`694e30c5`](https://github.com/queens-py/queens/commit/694e30c5df1772a57d6a18f2e62c1605cbf007c8)
   on the `main` branch.
3. **Install this package on top of QUEENS** (with the `queens` environment active):

   ```bash
   conda activate queens
   pip install -e .
   ```

   This reads `pyproject.toml` and pulls the only extra dependencies (`numba` for the
   sparse-linalg kernels, `matplotlib` for the convergence plots), and makes
   `import high_dim_svi` work from any directory. QUEENS is deliberately *not* a declared
   dependency — install it first, separately, as in step 2.

## Configuration and running

All paths and algorithm constants live in the `CONFIGURATION` block at the top of
`darcy_svi_demo.py`. The defaults reproduce the paper's **baseline configuration**
(Table 3) executed **locally** with the QUEENS `Local` scheduler and MPI: 4 samples per
iteration, the Adam optimizer at learning rate `5·10⁻²`, the sparse precision-Cholesky
Gaussian initialized from the incomplete Cholesky of the prior precision, SNR-20
observation noise, and VB-EM updates of the prior- and noise-precision hyperparameters.
Point the paths at your local copies of the executables and data, then:

```bash
conda activate queens
python darcy_svi_demo.py
```

The script validates every referenced path up front, runs the SVI optimization, and
writes results to `output/`. For a quick smoke test (a few iterations instead of the
full run), set the environment overrides:

```bash
DARCY_SVI_MAX_FEVAL=12 DARCY_SVI_VERBOSE_EVERY=1 python darcy_svi_demo.py
```

## Outputs

* `output/darcy_svi_baseline_local.pickle` — full results, including per-iteration
  diagnostics (`elbo`, `noise_variance`, `grad_norm`, `predictive_residual`, and the
  reconstruction error `reconstruction_error` = `‖μ − x_gt‖ / ‖x_gt‖`);
* `output/error_convergence.png` — reconstruction error `ε_pm` and predictive residual
  `ε_pred` vs. iteration (cf. Fig. 9 of the paper);
* `output/elbo_convergence.png` — ELBO trace;
* `output/posterior_mean_field.npy` and `posterior_mean_{plus,minus}_std_field.npy` —
  the posterior mean and ±1σ fields. Feed these back to `darcy_forward` once to produce
  `.pvtu` files for visualization in ParaView.

## Running on a cluster

For large overnight runs that reach full convergence (the paper uses ~500 iterations),
swap the `Local` scheduler for QUEENS' SSH-based `Cluster` scheduler (SLURM). See the
commented block at the bottom of `darcy_svi_demo.py`; the likelihood, variational family
and iterator are unchanged.

## Notes on the setup of this repository

This release favors reusing QUEENS over re-implementing it, so the custom code here is
deliberately small. A few aspects are worth pointing out:

- **The SPDE/GMRF prior lives in the deal.II adjoint.** `darcy_adjoint` returns the full
  log-posterior gradient `∇ₓ[log-likelihood + log-prior]`, assembling the prior gradient
  from the same FE Laplacian and mass matrices used by the forward solve. The Python-side
  prior is therefore flat (`FreeVariable`) and the ELBO computed in Python is
  `entropy(q) + E_q[log-likelihood]` only.
- **Custom `AdjointNpy` model and `DealDriver`.** The adjoint upstream gradient is
  exchanged with deal.II through binary NumPy files (faster than QUEENS' default CSV),
  and the high-dimensional field sample is written next to the rendered JSON input rather
  than injected into it. These are thin wrappers around QUEENS classes.
- **Isotropic likelihood on purpose.** The latest QUEENS `Gaussian` likelihood builds a
  dense `eye(n_obs)` covariance, which is infeasible for thousands of observations, so
  `GaussianVBEM` keeps a single scalar noise variance with a closed-form VB-EM update.
- **Scope.** This demo implements the baseline (full-covariance) method with the
  mean-only natural gradient, which the paper's ablation reports as statistically
  equivalent to the full baseline. The covariance/whitened variational families, the
  Takahashi precision natural gradient, the AMG coarse-to-fine continuation, and the
  freeze/Dirac (Laplace) variant are out of scope for this release.

In the future, the custom pieces should migrate into QUEENS so the setup becomes easier
to use and to adapt to other problems.
