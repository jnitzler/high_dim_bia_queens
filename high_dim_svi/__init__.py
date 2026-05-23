"""High-dimensional sparse-precision SVI on top of QUEENS.

Reference implementation for the paper *"Scalable High-Dimensional Bayesian Field
Reconstruction with Finite Elements: Application to 3D Porous Media Flow"*.

The package adds the components that are not (yet) part of the upstream QUEENS
framework and reuses everything else from QUEENS directly:

* :class:`~high_dim_svi.sparse_inverse_variational.SparseInverseNormal` -- the sparse
  precision-Cholesky Gaussian variational family;
* :class:`~high_dim_svi.high_dim_rpvi.HighDimRPVI` -- the SVI iterator (direct/STL ELBO
  gradient and mean-only natural gradient);
* :class:`~high_dim_svi.gaussian_vbem_likelihood.GaussianVBEM` -- an isotropic Gaussian
  likelihood with a VB-EM noise-variance update;
* :class:`~high_dim_svi.deal_driver.DealDriver` -- an MPI driver for the deal.II
  forward/adjoint executables; and
* :class:`~high_dim_svi.adjoint_npy.AdjointNpy` -- an adjoint model exchanging gradient
  data through ``.npy`` files.
"""

from .adjoint_npy import AdjointNpy
from .deal_driver import DealDriver
from .gaussian_vbem_likelihood import GaussianVBEM
from .high_dim_rpvi import HighDimRPVI
from .sparse_inverse_variational import SparseInverseNormal

__all__ = [
    "AdjointNpy",
    "DealDriver",
    "GaussianVBEM",
    "HighDimRPVI",
    "SparseInverseNormal",
]
