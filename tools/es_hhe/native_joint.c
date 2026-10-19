#include "native_joint.h"
#include "native_chemistry.h"
#include "native_geometry.h"

#include <float.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define ES_JOINT_PARALLEL_ROWS 1024

typedef struct {
    int64_t n, g, k, e;
    int threads;
    const double *cross;
    const int64_t *indices, *offsets;
    int64_t *owners, *scatter_offsets, *eligible;
    double *inverse, *volume, *residual, *cross_max, *threshold;
    double *cell_a, *cell_b, *sphere_a, *sphere_abs, *next_sphere;
    double *rate, *mid_rate, *light, *middle, *half_debit, *next_debit;
    double *next_state, *counts;
    int64_t evaluations;
} Joint;

static void release(Joint *j)
{
    free(j->owners); free(j->scatter_offsets); free(j->eligible);
    free(j->inverse); free(j->volume); free(j->residual); free(j->cross_max);
    free(j->threshold); free(j->cell_a); free(j->cell_b); free(j->sphere_a);
    free(j->sphere_abs); free(j->next_sphere); free(j->rate); free(j->mid_rate);
    free(j->light); free(j->middle); free(j->half_debit); free(j->next_debit);
    free(j->next_state); free(j->counts);
}

static int aggregate(Joint *j, const double *cells, double *spheres)
{
    return es_indexed_sum(cells, j->n, 3, j->indices, j->e, j->offsets,
                          j->k, spheres, j->threads) ? 7 : 0;
}

static int scatter(Joint *j, const double *spheres, double *cells)
{
    return es_indexed_sum(spheres, j->k, 3, j->owners, j->e,
                          j->scatter_offsets, j->n, cells, j->threads) ? 7 : 0;
}

static void absorbers(Joint *j, const double *state)
{
    #pragma omp parallel for num_threads(j->threads) if(j->n >= ES_JOINT_PARALLEL_ROWS)
    for (int64_t i = 0; i < j->n; ++i) {
        j->cell_a[3*i] = state[5*i];
        j->cell_a[3*i+1] = state[5*i+2];
        j->cell_a[3*i+2] = state[5*i+3];
    }
}

static int fields(Joint *j, const double *light, double *rate)
{
    #pragma omp parallel for num_threads(j->threads) if(j->n >= ES_JOINT_PARALLEL_ROWS)
    for (int64_t i = 0; i < j->n; ++i)
        for (int s = 0; s < 3; ++s) {
            double sum = 0.;
            for (int64_t g = 0; g < j->g; ++g)
                sum += (light[i*j->g+g] * j->inverse[i]) * j->cross[3*g+s];
            j->cell_a[3*i+s] = sum;
        }
    int status = aggregate(j, j->cell_a, j->sphere_a);
    if (status) return status;
    for (int64_t r = 0; r < j->k; ++r)
        for (int s = 0; s < 3; ++s) j->sphere_a[3*r+s] /= j->volume[r];
    return scatter(j, j->sphere_a, rate);
}

static int accounting(Joint *j, const double *light, const double *rate,
                      const double *counts, double *debit)
{
    #pragma omp parallel for num_threads(j->threads) if(j->n >= ES_JOINT_PARALLEL_ROWS)
    for (int64_t i = 0; i < j->n * 3; ++i)
        j->cell_a[i] = rate[i] > 0. ? counts[i] / rate[i] : 0.;
    int status = aggregate(j, j->cell_a, j->sphere_a);
    if (status) return status;
    for (int64_t r = 0; r < j->k; ++r)
        for (int s = 0; s < 3; ++s) j->sphere_a[3*r+s] /= j->volume[r];
    status = scatter(j, j->sphere_a, j->cell_b);
    if (status) return status;
    int invalid = 0;
    #pragma omp parallel for num_threads(j->threads) if(j->n >= ES_JOINT_PARALLEL_ROWS) reduction(max:invalid)
    for (int64_t i = 0; i < j->n; ++i)
        for (int64_t g = 0; g < j->g; ++g) {
            double response = 0.;
            for (int s = 0; s < 3; ++s)
                response += j->cell_b[3*i+s] * j->cross[3*g+s];
            double value = (light[i*j->g+g] * j->inverse[i]) * response;
            debit[i*j->g+g] = value;
            if (!isfinite(value) || value < 0.) invalid = 1;
        }
    return invalid ? 2 : 0;
}

static int capture(Joint *j, const double *state, const double *light,
                   const double *rate, double exposure, double *debit)
{
    ++j->evaluations;
    if (es_upward_cells((size_t)j->n, state, rate, &exposure, 0,
                        j->next_state, j->counts, j->threads)) return 7;
    return accounting(j, light, rate, j->counts, debit);
}

/* Feasibility is separate from errors: rejected trials leave accepted state
 * unchanged and are retried at half the common exposure. */
static int attempt(Joint *j, const double *state, const double *photons,
                   double fraction, double exposure, int *feasible)
{
    *feasible = 0;
    int status = capture(j, state, j->light, j->rate, .5 * exposure, j->half_debit);
    if (status) return status;
    int rejected = 0;
    #pragma omp parallel for num_threads(j->threads) if(j->n >= ES_JOINT_PARALLEL_ROWS) reduction(max:rejected)
    for (int64_t i = 0; i < j->n * j->g; ++i) {
        j->middle[i] = j->light[i] - j->half_debit[i];
        if (j->half_debit[i] > .5 * fraction * photons[i] || j->middle[i] < 0.)
            rejected = 1;
    }
    if (rejected) return 0;
    status = fields(j, j->middle, j->mid_rate);
    if (status) return status;
    status = capture(j, state, j->middle, j->mid_rate, exposure, j->next_debit);
    if (status) return status;
    #pragma omp parallel for num_threads(j->threads) if(j->n >= ES_JOINT_PARALLEL_ROWS) reduction(max:rejected)
    for (int64_t i = 0; i < j->n * j->g; ++i)
        if (j->next_debit[i] > fraction * photons[i]) rejected = 1;
    *feasible = !rejected;
    return 0;
}

static int crossing(Joint *j, int64_t *event_mask)
{
    int any = 0;
    for (int64_t r = 0; r < j->k; ++r) {
        const double *x = j->next_sphere + 3*r;
        const double remain[3] = {x[0], x[1], x[1] + x[2]};
        for (int s = 0; s < 3; ++s) {
            int hit = j->eligible[3*r+s] && remain[s] <= j->threshold[3*r+s];
            if (event_mask) event_mask[3*r+s] = hit;
            if (hit) any = 1;
        }
    }
    return any;
}

int es_joint_compact(int64_t n, int64_t g, int64_t k, int64_t e,
    const double *initial, const double *p0, const double *cross,
    const int64_t *indices, const int64_t *offsets, double epsilon,
    double fraction, double tolerance, double floor, int64_t max_iterations,
    int threads, double *state, double *photons, double *absorbed,
    double *debit, int64_t *event_mask, int64_t *info_i, double *info_f)
{
    if (k < 2) return 100;
    if (n < 1 || g < 1 || e < 1 || threads < 1 || threads > 1024 ||
        max_iterations < 1 || !isfinite(epsilon) || epsilon <= 0. || epsilon >= 1. ||
        !isfinite(fraction) || fraction <= 0. || fraction >= 1. ||
        !isfinite(tolerance) || tolerance <= 0. || tolerance >= 1. ||
        !isfinite(floor) || floor < 0. || !initial || !p0 || !cross || !indices ||
        !offsets || !state || !photons || !absorbed || !debit || !event_mask ||
        !info_i || !info_f || state == initial || photons == p0) return 1;
    if ((uint64_t)n > SIZE_MAX / (5 * sizeof(double)) ||
        (uint64_t)g > SIZE_MAX / (3 * sizeof(double)) ||
        (uint64_t)n > SIZE_MAX / sizeof(double) / (uint64_t)g ||
        (uint64_t)k > SIZE_MAX / (5 * sizeof(double)) ||
        (uint64_t)e > SIZE_MAX / sizeof(int64_t)) return 1;
    if (offsets[0] != 0 || offsets[k] != e) return 1;
    for (int64_t r = 0; r < k; ++r)
        if (offsets[r] < 0 || offsets[r] >= offsets[r+1]) return 1;
    for (int64_t i = 0; i < n*5; ++i)
        if (!isfinite(initial[i]) || initial[i] < 0.) return 2;
    for (int64_t i = 0; i < n*g; ++i)
        if (!isfinite(p0[i]) || p0[i] < 0.) return 2;
    for (int64_t i = 0; i < g*3; ++i)
        if (!isfinite(cross[i]) || cross[i] < 0.) return 2;

    Joint j = {0};
    j.n=n; j.g=g; j.k=k; j.e=e; j.threads=threads;
    j.cross=cross; j.indices=indices; j.offsets=offsets;
    int status = 0, stop = 0;
    int64_t steps = 0;
    double exposure = 0.;
    int64_t *marks = NULL, *cursor = NULL;
    double *sphere_initial = NULL;
    #define ALLOC(member,count,type) do { j.member=calloc((size_t)(count),sizeof(type)); if(!j.member){status=3;goto done;} } while(0)
    ALLOC(owners,e,int64_t); ALLOC(scatter_offsets,n+1,int64_t); ALLOC(eligible,k*3,int64_t);
    ALLOC(inverse,n,double); ALLOC(volume,k,double); ALLOC(residual,n*g,double);
    ALLOC(cross_max,g,double); ALLOC(threshold,k*3,double);
    ALLOC(cell_a,n*3,double); ALLOC(cell_b,n*3,double); ALLOC(sphere_a,k*3,double);
    ALLOC(sphere_abs,k*3,double); ALLOC(next_sphere,k*3,double);
    ALLOC(rate,n*3,double); ALLOC(mid_rate,n*3,double);
    ALLOC(light,n*g,double); ALLOC(middle,n*g,double);
    ALLOC(half_debit,n*g,double); ALLOC(next_debit,n*g,double);
    ALLOC(next_state,n*5,double); ALLOC(counts,n*3,double);
    #undef ALLOC
    marks = malloc((size_t)n * sizeof(int64_t));
    cursor = malloc((size_t)n * sizeof(int64_t));
    sphere_initial = malloc((size_t)k * 5 * sizeof(double));
    if (!marks || !cursor || !sphere_initial) {status=3;goto done;}
    for (int64_t i=0;i<n;++i) marks[i]=-1;
    for (int64_t r=0;r<k;++r) {
        j.volume[r]=(double)(offsets[r+1]-offsets[r]);
        for (int64_t q=offsets[r];q<offsets[r+1];++q) {
            int64_t i=indices[q];
            if (i<0 || i>=n || marks[i]==r) {status=1;goto done;}
            marks[i]=r; ++j.scatter_offsets[i+1];
        }
    }
    for (int64_t i=0;i<n;++i) {
        j.inverse[i]=j.scatter_offsets[i+1] ? 1./(double)j.scatter_offsets[i+1] : 0.;
        j.scatter_offsets[i+1]+=j.scatter_offsets[i];
        cursor[i]=j.scatter_offsets[i];
    }
    for (int64_t r=0;r<k;++r)
        for (int64_t q=offsets[r];q<offsets[r+1];++q) j.owners[cursor[indices[q]]++]=r;
    if (es_indexed_sum(initial,n,5,indices,e,offsets,k,sphere_initial,threads)) {status=7;goto done;}
    for (int64_t r=0;r<k;++r) {
        const double *a=sphere_initial+5*r;
        const double normal[3]={a[0]+a[1],(a[2]+a[3])+a[4],(a[2]+a[3])+a[4]};
        const double remaining[3]={a[0],a[2],a[2]+a[3]};
        for (int s=0;s<3;++s) {
            if (!isfinite(normal[s]) || !isfinite(remaining[s])) {status=2;goto done;}
            j.threshold[3*r+s]=epsilon*normal[s];
            j.eligible[3*r+s]=normal[s]>0. && remaining[s]>j.threshold[3*r+s]+64*DBL_EPSILON*normal[s];
        }
        j.sphere_abs[3*r]=a[0];j.sphere_abs[3*r+1]=a[2];j.sphere_abs[3*r+2]=a[3];
    }
    for (int64_t i=0;i<n;++i) {
        j.cell_a[3*i]=initial[5*i];j.cell_a[3*i+1]=initial[5*i+2];
        j.cell_a[3*i+2]=initial[5*i+2]+initial[5*i+3];
    }
    if ((status=aggregate(&j,j.cell_a,j.sphere_a)) || (status=scatter(&j,j.sphere_a,j.cell_b))) goto done;
    for (int64_t band=0;band<g;++band)
        j.cross_max[band]=fmax(cross[3*band],fmax(cross[3*band+1],cross[3*band+2]));
    for (int64_t i=0;i<n;++i)
        for (int64_t band=0;band<g;++band) {
            double capacity=0.;
            for (int s=0;s<3;++s) capacity+=j.cell_b[3*i+s]*(cross[3*band+s]>0.);
            capacity*=j.inverse[i];
            if (!isfinite(capacity)) {status=2;goto done;}
            j.residual[i*g+band]=fmax(floor,tolerance*fmin(p0[i*g+band],capacity));
        }
    memcpy(state,initial,(size_t)n*5*sizeof(double));
    memcpy(photons,p0,(size_t)n*g*sizeof(double));
    memset(absorbed,0,(size_t)n*3*sizeof(double));
    memset(debit,0,(size_t)n*g*sizeof(double));
    memset(event_mask,0,(size_t)k*3*sizeof(int64_t));
    for (int64_t iteration=0;iteration<max_iterations;++iteration) {
        for (int64_t r=0;r<k;++r)
            for (int s=0;s<3;++s) j.sphere_a[3*r+s]=j.sphere_abs[3*r+s]/j.volume[r];
        if ((status=scatter(&j,j.sphere_a,j.cell_b))) goto done;
        double max_hazard=0.;
        int any_hazard=0,any_active=0,invalid=0;
        #pragma omp parallel for num_threads(threads) if(n >= ES_JOINT_PARALLEL_ROWS) reduction(max:max_hazard,any_hazard,any_active,invalid)
        for (int64_t i=0;i<n;++i)
            for (int64_t band=0;band<g;++band) {
                double hazard=0.;
                for (int s=0;s<3;++s) hazard+=j.cell_b[3*i+s]*cross[3*band+s];
                hazard*=j.inverse[i];
                if (!isfinite(hazard)) invalid=1;
                if (hazard>0.) any_hazard=1;
                double p=photons[i*g+band];
                int active=p>j.residual[i*g+band] && hazard>0. &&
                    .5*fraction*p>=nextafter(0.,1.) && p*j.cross_max[band]>0.;
                j.light[i*g+band]=active?p:0.;
                if (active) {any_active=1;if(hazard>max_hazard)max_hazard=hazard;}
            }
        if (invalid) {status=2;goto done;}
        if (!any_active) {
            int any_absorption=0;
            for(int64_t i=0;i<n*3;++i) if(absorbed[i]>0.) {any_absorption=1;break;}
            stop=any_absorption ? (any_hazard?1:2) : 0;
            goto success;
        }
        if ((status=fields(&j,j.light,j.rate))) goto done;
        double step=fraction/max_hazard;
        if (!isfinite(step) || step<=0.) {stop=3;goto success;}
        int feasible=0;
        for (int backtrack=0;backtrack<100;++backtrack) {
            if ((status=attempt(&j,state,photons,fraction,step,&feasible))) goto done;
            if(feasible)break;
            step*=.5;
        }
        if (!feasible) {status=5;goto done;}
        int changed=0;
        for (int64_t i=0;i<n*g;++i) if(j.next_debit[i]>0.) {changed=1;break;}
        if(!changed) for(int64_t i=0;i<n*5;++i) if(j.next_state[i]!=state[i]) {changed=1;break;}
        if(!changed) {stop=3;goto success;}
        absorbers(&j,j.next_state);
        if ((status=aggregate(&j,j.cell_a,j.next_sphere))) goto done;
        int hit=crossing(&j,NULL);
        if(hit) {
            double low=0.,high=step;
            for(int bisect=0;bisect<64;++bisect) {
                double middle=low+.5*(high-low);
                if(middle==low || middle==high)break;
                if((status=attempt(&j,state,photons,fraction,middle,&feasible)))goto done;
                if(!feasible){status=6;goto done;}
                absorbers(&j,j.next_state);
                if((status=aggregate(&j,j.cell_a,j.next_sphere)))goto done;
                if(crossing(&j,NULL))high=middle;else low=middle;
            }
            step=high;
            if((status=attempt(&j,state,photons,fraction,step,&feasible)))goto done;
            if(!feasible){status=6;goto done;}
            absorbers(&j,j.next_state);
            if((status=aggregate(&j,j.cell_a,j.next_sphere)))goto done;
            hit=crossing(&j,event_mask);
        }
        memcpy(state,j.next_state,(size_t)n*5*sizeof(double));
        #pragma omp parallel for num_threads(threads) if(n >= ES_JOINT_PARALLEL_ROWS)
        for(int64_t i=0;i<n*g;++i) {photons[i]-=j.next_debit[i];debit[i]+=j.next_debit[i];}
        #pragma omp parallel for num_threads(threads) if(n >= ES_JOINT_PARALLEL_ROWS)
        for(int64_t i=0;i<n*3;++i)absorbed[i]+=j.counts[i];
        exposure+=step;++steps;
        memcpy(j.sphere_abs,j.next_sphere,(size_t)k*3*sizeof(double));
        if(hit){stop=4;goto success;}
    }
    status=4;
    goto done;
success:
    info_i[0]=stop;info_i[1]=j.evaluations;info_i[2]=steps;info_f[0]=exposure;
done:
    free(marks);free(cursor);free(sphere_initial);release(&j);
    return status;
}
