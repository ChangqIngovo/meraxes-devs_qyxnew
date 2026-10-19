#include "native_chemistry.h"

#include <float.h>
#include <math.h>

/* Small selected supports are faster without an OpenMP team launch. The
 * threshold changes execution only; every cell uses the same arithmetic.
 */
#define ES_CHEMISTRY_PARALLEL_ROWS 1024

static int upward_cell(const double *n, const double *rate, double t,
                       double *new_state, double *counts)
{
    for (int j = 0; j < 5; ++j)
        if (!isfinite(n[j]) || n[j] < 0.)
            return 2;
    for (int j = 0; j < 3; ++j)
        if (!isfinite(rate[j]) || rate[j] < 0.)
            return 2;
    if (!isfinite(t) || t < 0.)
        return 2;

    const double x_h = rate[0] * t;
    const double a = rate[1] * t;
    const double b = rate[2] * t;
    if (!isfinite(x_h) || !isfinite(a) || !isfinite(b))
        return 2;
    const double loss_h = -expm1(-x_h);
    const double loss_i = -expm1(-a);
    const double loss_ii = -expm1(-b);
    const double difference = fabs(a - b);
    const double smaller = fmin(a, b);
    const double exponential = exp(-smaller);
    const double ratio = difference > 0. ? -expm1(-difference) / difference : 1.;
    double to_ii = a * exponential * ratio;

    /* Evaluate the small probabilities directly. Subtracting order-unity
     * survival fractions would erase He II -> He III captures in thin cells.
     * These are the same convergent series as the Python reference.
     */
    double first = -expm1(-smaller) - smaller * exponential;
    if (smaller > 0. && smaller < .5) {
        double term = .5 * smaller * smaller;
        double total = term;
        for (int k = 3; k < 20; ++k) {
            term *= -smaller * (k - 1) / (k * (k - 2));
            total += term;
            if (fabs(term) <= DBL_EPSILON * fabs(total))
                break;
        }
        first = total;
    }
    double second = 1. - ratio;
    if (difference > 0. && difference < .5) {
        double term = .5 * difference;
        double total = term;
        for (int k = 2; k < 20; ++k) {
            term *= -difference / (k + 1);
            total += term;
            if (fabs(term) <= DBL_EPSILON * fabs(total))
                break;
        }
        second = total;
    }
    double to_iii = first + smaller * exponential * second;
    to_ii = fmin(fmax(to_ii, 0.), loss_i);
    to_iii = fmin(fmax(to_iii, 0.), loss_i);
    /* Form the larger branch by subtraction, preserving the smaller one. */
    if (to_iii <= to_ii)
        to_ii = loss_i - to_iii;
    else
        to_iii = loss_i - to_ii;

    counts[0] = n[0] * loss_h;
    counts[1] = n[2] * loss_i;
    counts[2] = n[2] * to_iii + n[3] * loss_ii;
    new_state[0] = n[0] * (1. - loss_h);
    new_state[1] = n[1] + counts[0];
    new_state[2] = n[2] * (1. - loss_i);
    new_state[3] = n[3] * (1. - loss_ii) + n[2] * to_ii;
    new_state[4] = n[4] + counts[2];
    for (int j = 0; j < 5; ++j)
        if (!isfinite(new_state[j]) || new_state[j] < 0.)
            return 3;
    for (int j = 0; j < 3; ++j)
        if (!isfinite(counts[j]) || counts[j] < 0.)
            return 3;
    return 0;
}

int es_upward_cells(size_t rows, const double *n, const double *rate,
                    const double *exposure, int vector_exposure,
                    double *new_state, double *counts, int threads)
{
    if (threads < 1 || (vector_exposure != 0 && vector_exposure != 1))
        return 1;
    if (!rows)
        return 0;
    if (!n || !rate || !exposure || !new_state || !counts)
        return 1;
    int status = 0;
    #pragma omp parallel for num_threads(threads) if(rows >= ES_CHEMISTRY_PARALLEL_ROWS) reduction(max:status)
    for (size_t i = 0; i < rows; ++i) {
        const int cell_status = upward_cell(n + 5 * i, rate + 3 * i,
            exposure[vector_exposure ? i : 0], new_state + 5 * i, counts + 3 * i);
        if (cell_status > status)
            status = cell_status;
    }
    return status;
}
