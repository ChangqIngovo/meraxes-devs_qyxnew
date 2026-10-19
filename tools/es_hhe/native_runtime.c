#include <omp.h>

/* Small ABI and runtime checks used before any scientific state is changed. */
int es_kernel_abi_version(void) { return 1; }
int es_kernel_openmp_version(void) { return _OPENMP; }
int es_kernel_team_size(int requested)
{
  int actual = 0;
  if (requested < 1) return 0;
  #pragma omp parallel num_threads(requested) reduction(max:actual)
  actual = omp_get_num_threads();
  return actual;
}
