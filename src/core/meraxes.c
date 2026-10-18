#define _MAIN
#include <fenv.h>
#include <string.h>
#include <sys/stat.h>

#include "init.h"
#include "interactive.h"
#include "meraxes.h"
#include "utils.h"

int main(int argc, char** argv)
{
  MPI_Init(&argc, &argv);
  MPI_Comm_dup(MPI_COMM_WORLD, &run_globals.mpi_comm);
  MPI_Comm_rank(MPI_COMM_WORLD, &run_globals.mpi_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &run_globals.mpi_size);

  // init mlog
  init_mlog(MPI_COMM_WORLD, stdout, stdout, stderr);

  struct stat filestatus;

  // deal with any input arguments
  const bool check_params = (argc == 3 && strcmp(argv[1], "--check-params") == 0);
  if (!check_params && (argc != 2 || strcmp(argv[1], "--check-params") == 0)) {
    mlog_error("usage: %s [--check-params] <parameterfile>", argv[0]);
    ABORT(EXIT_FAILURE);
  }
  char* parameter_file = argv[check_params ? 2 : 1];

  // set the rounding mode
  if (!fesetround(1)) // nearest number
  {
    mlog_error("Failed to set rounding mode!");
    ABORT(EXIT_FAILURE);
  }

  // read the input parameter file
  read_parameter_file(parameter_file, 0);

  // Validate the complete native parameter chain without creating output or
  // reading simulation grids, trees, or physical tables.
  if (check_params) {
    if (run_globals.mpi_rank == 0) {
      fprintf(stdout, "\nParameter check passed: %s\n", parameter_file);
      fflush(stdout);
    }
    MPI_Finalize();
    return EXIT_SUCCESS;
  }

  // Check to see if the output directory exists and if not, create it
  if (stat(run_globals.params.OutputDir, &filestatus) != 0)
    mkdir(run_globals.params.OutputDir, 02755);

  // initiate meraxes
  init_meraxes();
  init_storage();

  // Run the model!
  if ((!run_globals.params.FlagInteractive) & (!run_globals.params.FlagMCMC))
    dracarys();
  else
    while (run_globals.params.FlagInteractive) {
      dracarys();
      continue_prompt(argv[1]);
    }

  // cleanup
  cleanup();

  MPI_Finalize();
  return EXIT_SUCCESS;
}
