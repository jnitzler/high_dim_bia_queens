"""MPI driver for the deal.II Darcy forward/adjoint executables.

A thin wrapper around QUEENS' :class:`queens.drivers.jobscript.Jobscript` that runs
``mpirun -np <num_procs> <executable> <input.json>``. The only customization is in
:meth:`prepare_input_files`: the high-dimensional random-field sample (``O(1e5)``
entries) cannot be injected into the JSON template, so it is written to a NumPy
binary next to the rendered input file, and the JSON is filled with the paths the
deal.II solver expects.

The deal.II I/O convention (see the deal.II project's ``CLAUDE.md``):

* ``input npy file``   -> the field ``.npy`` (its *parent directory* is also where
  the adjoint expects ``adjoint_data.npy``);
* ``output directory`` + ``output prefix`` -> outputs are written as
  ``<output_directory>/<output_prefix>sol.npy``, ``...solution_full.npy``,
  ``...grad_solution.npy``.

The forward and adjoint drivers must therefore share the same ``output_prefix`` so
the adjoint finds the forward's ``solution_full.npy`` in the (shared) job directory.
"""

import logging

import numpy as np

from queens.drivers.jobscript import Jobscript
from queens.utils.injector import inject
from queens.utils.logger_settings import log_init_args

_logger = logging.getLogger(__name__)

# Jobscript template: a plain MPI launch of the deal.II executable on the rendered
# JSON input file. `num_procs` is supplied per-job by the scheduler.
_JOBSCRIPT_TEMPLATE = "{{ mpi_cmd }} -np {{ num_procs }} {{ executable }} {{ input_file }}"


class DealDriver(Jobscript):
    """Driver that runs a deal.II Darcy executable via MPI.

    Attributes:
        output_prefix (str): Filename prefix shared by the deal.II output files. Must
            match between the forward and adjoint driver so the adjoint can locate the
            forward solution.
    """

    @log_init_args
    def __init__(
        self,
        parameters,
        input_template,
        executable,
        data_processor=None,
        gradient_data_processor=None,
        output_prefix: str = "darcy_",
        mpi_cmd: str = "/usr/bin/mpirun --bind-to none",
        files_to_copy=None,
    ) -> None:
        """Initialize the deal.II MPI driver.

        Args:
            parameters (Parameters): Parameters object.
            input_template (str | Path): Path to the deal.II JSON input template.
            executable (str | Path): Path to the deal.II executable
                (``darcy_forward`` or ``darcy_adjoint``).
            data_processor (obj, optional): Processor reading the primal result
                (``*_sol.npy`` for the forward, ``*_grad_solution.npy`` for the adjoint).
            gradient_data_processor (obj, optional): Processor for gradient data
                (unused with the adjoint workflow).
            output_prefix (str): Shared deal.II output filename prefix.
            mpi_cmd (str): MPI launch command.
            files_to_copy (list, optional): Extra files/directories to copy to the
                experiment directory.
        """
        super().__init__(
            parameters=parameters,
            input_templates=input_template,
            jobscript_template=_JOBSCRIPT_TEMPLATE,
            executable=executable,
            files_to_copy=files_to_copy,
            data_processor=data_processor,
            gradient_data_processor=gradient_data_processor,
            extra_options={"mpi_cmd": mpi_cmd},
        )
        self.output_prefix = output_prefix

    def prepare_input_files(self, sample_dict, experiment_dir, input_files):
        """Render the JSON input and write the field sample as a NumPy binary.

        Args:
            sample_dict (dict): Combined job options and sample values (the field
                components are keyed by ``self.parameters.parameters_keys``).
            experiment_dir (Path): QUEENS experiment directory (holds the copied template).
            input_files (dict): Mapping of template name to the per-job input file path.
        """
        for template_name, template_path in self.input_templates.items():
            # Field .npy lives next to the rendered JSON, inside the job directory.
            field_npy_path = input_files[template_name].with_suffix(".npy")

            sample_dict["input_file_path"] = str(field_npy_path)
            sample_dict["output_directory"] = str(sample_dict["output_dir"])
            sample_dict["output_prefix"] = self.output_prefix

            inject(
                sample_dict,
                experiment_dir / template_path.name,
                input_files[template_name],
            )

            # Reconstruct the field vector in parameter-key order and store it.
            field = np.array(
                [sample_dict[key] for key in self.parameters.parameters_keys], dtype=float
            )
            np.save(field_npy_path, field)
            _logger.debug("Wrote field sample (%d entries) to %s", field.size, field_npy_path)
