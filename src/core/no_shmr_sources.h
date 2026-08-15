#ifndef NO_SHMR_SOURCES_H
#define NO_SHMR_SOURCES_H

#include <stdbool.h>

#ifdef __cplusplus
extern "C"
{
#endif

  void no_shmr_sources_restore(void);
  void no_shmr_sources_free(void);
  bool no_shmr_sources_is_applied(void);

  /* Compatibility entry point retained for the existing Meraxes startup. */
  void init_reion_source_tables(void);

#ifdef __cplusplus
}
#endif

#endif
