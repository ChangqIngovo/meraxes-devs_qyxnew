#include <complex.h>
#include <float.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include <fftw3-mpi.h>

#include "XRayHeatingFunctions.h"
#include "meraxes.h"
#include "photon_conservation_feedback.h"
#include "reionization.h"
#include "virial_properties.h"

static void pc_feedback_require(int local_failure, const char* message)
{
  int failure = 0;
  if (MPI_Allreduce(&local_failure, &failure, 1, MPI_INT, MPI_MAX, run_globals.mpi_comm) != MPI_SUCCESS) {
    mlog_error("Photon feedback: collective status check failed.");
    ABORT(EXIT_FAILURE);
  }
  if (failure) {
    mlog_error("Photon feedback: %s", message);
    ABORT(EXIT_FAILURE);
  }
}

/* Match the legacy ES field sanitization, after checking finiteness. These
 * fields estimate a background; this clipping never debits a photon ledger.
 */
static double pc_feedback_nonnegative(float value, int* invalid)
{
  if (!isfinite(value)) {
    *invalid = 1;
    return 0.0;
  }
  const float positive = fmaxf(value, 0.0f);
  return positive < ABS_TOL ? 0.0 : (double)positive;
}

static void pc_feedback_filter(fftwf_complex* output,
                               const fftwf_complex* input,
                               fftwf_plan inverse,
                               size_t ncomplex,
                               int local_start,
                               int local_nix,
                               int dim,
                               double radius)
{
  memcpy(output, input, ncomplex * sizeof(*output));
  if (radius > 0.0)
    filter(output, local_start, local_nix, dim, (float)radius, run_globals.params.ReionFilterType);
  fftwf_execute(inverse);
}

void photon_conservation_update_feedback(int snapshot,
                                         const double* radii_mpc_h,
                                         int nradii,
                                         const int* radius_slot,
                                         const float* old_zre,
                                         const float* old_j21)
{
  reion_grids_t* grids = &run_globals.reion_grids;
  run_units_t* units = &run_globals.units;
  const int dim = run_globals.params.ReionGridDim;
  const int local_nix = (int)grids->slab_nix[run_globals.mpi_rank];
  const int local_start = (int)grids->slab_ix_start[run_globals.mpi_rank];
  const size_t ncomplex = (size_t)grids->slab_n_complex[run_globals.mpi_rank];
  const int use_rec = run_globals.params.Flag_IncludeRecombinations;
  const int use_bh = run_globals.params.physics.Flag_BHFeedback;
  const int uvb = run_globals.params.ReionUVBFlag;
  const int use_xray = run_globals.params.Flag_IncludeSpinTemp;
  int minimum_n = 0, maximum_n = 0;
  MPI_Allreduce(&nradii, &minimum_n, 1, MPI_INT, MPI_MIN, run_globals.mpi_comm);
  MPI_Allreduce(&nradii, &maximum_n, 1, MPI_INT, MPI_MAX, run_globals.mpi_comm);
  pc_feedback_require(nradii <= 0 || minimum_n != maximum_n || radii_mpc_h == NULL || radius_slot == NULL ||
                        old_zre == NULL || (uvb && old_j21 == NULL) || uvb < 0 || uvb > 2 || snapshot < 0 ||
                        snapshot >= run_globals.params.SnaplistLength || dim <= 0 || local_nix < 0 ||
                        run_globals.params.ReionFilterType != 0,
                      "invalid input, inconsistent radius count, or unsupported ES filter.");

  /* A shared ladder and collective use flags give every MPI rank the same
   * sequence of distributed inverse FFTs, including ranks with no targets.
   */
  int invalid = radii_mpc_h[nradii - 1] != 0.0;
  unsigned long long hash = 1469598103934665603ULL;
  for (int k = 0; k < nradii; ++k) {
    uint64_t bits;
    memcpy(&bits, &radii_mpc_h[k], sizeof(bits));
    hash = (hash ^ (unsigned long long)bits) * 1099511628211ULL;
    invalid |=
      !isfinite(radii_mpc_h[k]) || radii_mpc_h[k] < 0.0 || (k + 1 < nradii && radii_mpc_h[k] <= radii_mpc_h[k + 1]);
  }
  unsigned long long minimum_hash = 0, maximum_hash = 0;
  MPI_Allreduce(&hash, &minimum_hash, 1, MPI_UNSIGNED_LONG_LONG, MPI_MIN, run_globals.mpi_comm);
  MPI_Allreduce(&hash, &maximum_hash, 1, MPI_UNSIGNED_LONG_LONG, MPI_MAX, run_globals.mpi_comm);
  pc_feedback_require(invalid || minimum_hash != maximum_hash,
                      "radius ladder is not shared, strictly descending, and terminated by the identity filter.");

  enum
  {
    NEED_GAMMA = 1,
    NEED_J = 2,
    NEED_RADIUS = 4
  };
  int* used = calloc((size_t)nradii, sizeof(*used));
  pc_feedback_require(used == NULL, "could not allocate radius-use flags.");

  const double redshift = run_globals.ZZ[snapshot];
  const double cell_width = run_globals.params.BoxSize / (double)dim;
  const double cell_volume = cell_width * cell_width * cell_width;
  const double self_radius = (cell_width < 1.0 ? 1.0 : L_FACTOR) * cell_width;
  const double sfr_timescale = run_globals.params.ReionSfrTimescale * hubble_time(snapshot);
  const double alpha = run_globals.params.physics.ReionAlphaUV;
  const double alpha_bh = run_globals.params.physics.ReionAlphaUVBH;
  const double nion = run_globals.params.physics.ReionNionPhotPerBary;
  pc_feedback_require(!isfinite(redshift) || redshift <= -1.0 || !isfinite(cell_width) || cell_width <= 0.0 ||
                        !isfinite(sfr_timescale) || sfr_timescale <= 0.0 || !isfinite(alpha) || alpha < 0.0 ||
                        !isfinite(alpha_bh) || alpha_bh < 0.0 || !isfinite(nion) || nion <= 0.0 ||
                        !isfinite(run_globals.params.physics.ReionGammaHaloBias) ||
                        run_globals.params.physics.ReionGammaHaloBias < 0.0 || !isfinite(units->UnitLength_in_cm) ||
                        units->UnitLength_in_cm <= 0.0 || !isfinite(units->UnitMass_in_g) ||
                        units->UnitMass_in_g <= 0.0 || !isfinite(units->UnitTime_in_s) || units->UnitTime_in_s <= 0.0,
                      "invalid physical background normalization.");
#if USE_MINI_HALOS
  const double nion_ratio = run_globals.params.physics.ReionNionPhotPerBaryIII / nion;
  pc_feedback_require(!isfinite(nion_ratio) || nion_ratio < 0.0, "invalid Pop. III photon-yield ratio.");
#endif

  float tk = 0.0f, ct_ad = 0.0f;
  if (!use_xray) {
    pc_feedback_require(T_RECFAST(100.0f, 1) < 0.0, "could not initialize RECFAST temperatures.");
    tk = (float)T_RECFAST((float)redshift, 0);
    pc_feedback_require(T_RECFAST(100.0f, 2) < 0.0, "could not release RECFAST temperatures.");
    ct_ad = cT_approx((float)redshift);
  }

  invalid = 0;
  for (int ix = 0; ix < local_nix; ++ix)
    for (int iy = 0; iy < dim; ++iy)
      for (int iz = 0; iz < dim; ++iz) {
        const size_t i = (size_t)iz + (size_t)dim * ((size_t)iy + (size_t)dim * (size_t)ix);
        const size_t p = (size_t)iz + (size_t)(2 * (dim / 2 + 1)) * ((size_t)iy + (size_t)dim * (size_t)ix);
        const float neutral = grids->xH[i];
        const float neutral_temperature = use_xray ? grids->Tk_box[i] : tk * (1.0f + ct_ad * grids->deltax[p]);
        invalid |= !isfinite(neutral) || neutral < 0.0f || neutral > 1.0f || !isfinite(old_zre[i]) ||
                   !isfinite(neutral_temperature) || neutral_temperature < 0.0f;
        if (uvb)
          invalid |= !isfinite(old_j21[i]) || old_j21[i] < 0.0f;
        grids->z_at_ionization[i] = old_zre[i];
        if (uvb)
          grids->J_21_at_ionization[i] = old_j21[i];
        if (use_rec)
          grids->Gamma12[i] = 0.0f;
        grids->r_bubble[i] = 0.0f;
        /* Re-evaluate the legacy phase mixture using the CORRECTED xH. The
         * caller then applies its existing full-cell evolution and floor.
         */
        grids->temp_kinetic_all_gas[i] = neutral <= 0.0f ? (float)T_RE
                                         : neutral >= 1.0f
                                           ? neutral_temperature
                                           : neutral_temperature * neutral + (float)T_RE * (1.0f - neutral);
        if (neutral < REL_TOL) {
          const int k = radius_slot[i];
          invalid |= k < 0 || k >= nradii;
          if (k >= 0 && k < nradii) {
            used[k] |= NEED_RADIUS;
            if (use_rec)
              used[k] |= NEED_GAMMA;
            if (uvb == 2 || (uvb == 1 && old_zre[i] < 0.0f))
              used[k] |= NEED_J;
          }
          if (old_zre[i] < 0.0f)
            grids->z_at_ionization[i] = (float)redshift;
        }
      }
  pc_feedback_require(invalid, "invalid corrected field, historical feedback, temperature, or selected radius.");
  pc_feedback_require(MPI_Allreduce(MPI_IN_PLACE, used, nradii, MPI_INT, MPI_BOR, run_globals.mpi_comm) != MPI_SUCCESS,
                      "could not combine radius-use flags.");

  for (int k = 0; k < nradii; ++k) {
    if (!used[k])
      continue;
    const double radius = radii_mpc_h[k];
    const double effective_radius = radius > 0.0 ? radius : self_radius;
    const int need_gamma = used[k] & NEED_GAMMA;
    const int need_j = used[k] & NEED_J;
#define PC_FILTER(field)                                                                                               \
  pc_feedback_filter(grids->field##_filtered,                                                                          \
                     grids->field##_unfiltered,                                                                        \
                     grids->field##_filtered_reverse_plan,                                                             \
                     ncomplex,                                                                                         \
                     local_start,                                                                                      \
                     local_nix,                                                                                        \
                     dim,                                                                                              \
                     radius)
    if (need_gamma) {
      PC_FILTER(weighted_sfr);
      if (use_bh)
        PC_FILTER(effective_bhar);
#if USE_MINI_HALOS
      PC_FILTER(weighted_sfrIII);
#endif
    }
    if (need_j) {
      PC_FILTER(stars);
      if (use_bh)
        PC_FILTER(effective_bhm);
#if USE_MINI_HALOS
      PC_FILTER(starsIII);
#endif
    }
#undef PC_FILTER

    /* Same geometric, spectral and unit prefactors as _find_HII_bubbles.
     * Gamma uses instantaneous escaped rates; J uses cumulative escaped
     * masses divided by the existing star-formation time-scale proxy.
     */
    double gamma_base = (1.0 + redshift) * (1.0 + redshift) * effective_radius * units->UnitLength_in_cm / cell_volume *
                        pow(units->UnitLength_in_cm, -3.0);
    gamma_base *= (units->UnitMass_in_g / units->UnitTime_in_s) / PROTONMASS * nion;
    const double gamma_weight = gamma_base * SIGMA_HI * alpha / (alpha + 2.75) * 1e12;
    const double gamma_weight_bh = gamma_base * SIGMA_HI * alpha_bh / (alpha_bh + 2.75) * 1e12;
    const double j_base =
      gamma_base * PLANCK * 1e21 / (4.0 * M_PI) * run_globals.params.physics.ReionGammaHaloBias / sfr_timescale;
    invalid = !isfinite(gamma_weight) || gamma_weight < 0.0 || !isfinite(gamma_weight_bh) || gamma_weight_bh < 0.0 ||
              !isfinite(j_base) || j_base < 0.0 || effective_radius > FLT_MAX;
    for (int ix = 0; ix < local_nix; ++ix)
      for (int iy = 0; iy < dim; ++iy)
        for (int iz = 0; iz < dim; ++iz) {
          const size_t i = (size_t)iz + (size_t)dim * ((size_t)iy + (size_t)dim * (size_t)ix);
          const size_t p = (size_t)iz + (size_t)(2 * (dim / 2 + 1)) * ((size_t)iy + (size_t)dim * (size_t)ix);
          if (grids->xH[i] >= REL_TOL || radius_slot[i] != k)
            continue;
          grids->r_bubble[i] = (float)effective_radius;
          if (use_rec) {
            double gamma = gamma_weight * pc_feedback_nonnegative(((float*)grids->weighted_sfr_filtered)[p], &invalid);
            if (use_bh)
              gamma += gamma_weight_bh * pc_feedback_nonnegative(((float*)grids->effective_bhar_filtered)[p], &invalid);
#if USE_MINI_HALOS
            gamma += gamma_weight * nion_ratio *
                     pc_feedback_nonnegative(((float*)grids->weighted_sfrIII_filtered)[p], &invalid);
#endif
            invalid |= !isfinite(gamma) || gamma < 0.0 || gamma > FLT_MAX;
            grids->Gamma12[i] = (float)gamma;
          }
          if (uvb == 2 || (uvb == 1 && old_zre[i] < 0.0f)) {
            double intensity = alpha * pc_feedback_nonnegative(((float*)grids->stars_filtered)[p], &invalid);
            if (use_bh)
              intensity += alpha_bh * pc_feedback_nonnegative(((float*)grids->effective_bhm_filtered)[p], &invalid);
#if USE_MINI_HALOS
            intensity += alpha * nion_ratio * pc_feedback_nonnegative(((float*)grids->starsIII_filtered)[p], &invalid);
#endif
            intensity *= j_base;
            invalid |= !isfinite(intensity) || intensity < 0.0 || intensity > FLT_MAX;
            grids->J_21_at_ionization[i] = (float)intensity;
          }
        }
    pc_feedback_require(invalid, "non-finite or unrepresentable reconstructed UV background.");
  }
  free(used);
}
