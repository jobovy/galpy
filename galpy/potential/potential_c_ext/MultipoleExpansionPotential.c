// MultipoleExpansionPotential: C implementation of multipole expansion potential
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <gsl/gsl_spline.h>
#include <gsl/gsl_sf_legendre.h>
#include <galpy_potentials.h>

#ifndef GSL_MAJOR_VERSION
#define GSL_MAJOR_VERSION 1
#endif
// Cache tags for force/deriv caching
#define MEP_CACHE_FORCE 1
#define MEP_CACHE_DERIV 2

// PPoly order (quintic from BPoly.from_derivatives with 3 constraints)
#define PPOLY_K 6


// ============================================================================
// Pre-computed multipole data
// ============================================================================

// Cubic polynomial order for time interpolation
#define TIME_PPOLY_K 4

struct multipole_data {
    int L, M, isNonAxi;
    int Nr;                 // number of radial grid points
    double *rgrid;          // radial grid (owned, Nr points)
    double amp;             // overall amplitude
    double rmin, rmax;      // radial grid bounds
    // Rho spline offsets: index into spline1d for rho_cos (rho_sin = +1)
    int *rho_spline_offset; // size L*M
    // PPoly data for I_inner/I_outer
    double *ppoly_data;     // all PPoly coefficients (owned)
    int *ppoly_offset;      // per (l,m): offset into ppoly_data (size L*M)
    // Legendre array size
    int Psize;
    // PPoly interval search cache
    int last_interval;
    // Caching for forces (per potentialArgs copy, thread-safe)
    double cached_R, cached_Z, cached_phi;
    double cached_F[3];     // dPhi/dr, dPhi/dtheta, dPhi/dphi
    double cached_D[3];     // d2Phi/dr2, d2Phi/dphi2, d2Phi/drdphi
    int cache_valid;
    // Time-dependent fields (all NULL/0 for static)
    int Nt;                      // 0 = static
    double *tgrid;               // owned, Nt points
    double tmin, tmax;
    int last_t_interval;
    double cached_t;             // NAN = not yet initialized

    double *time_ppoly_data;     // all time-PPoly coefficients for I_inner/I_outer (owned)
    int *time_ppoly_offset;      // per (l,m): offset into time_ppoly_data (size L*M)

    double *time_rho_ppoly_data; // time-PPoly for rho values (owned)
    int *time_rho_ppoly_offset;  // per (l,m): offset (size L*M)

    double *rho_scratch;         // Nr doubles, scratch for rho reconstruction
};

static void freeMultipoleData(void *data)
{
    struct multipole_data *d = (struct multipole_data *)data;
    if (d->rgrid) free(d->rgrid);
    if (d->rho_spline_offset) free(d->rho_spline_offset);
    if (d->ppoly_data) free(d->ppoly_data);
    if (d->ppoly_offset) free(d->ppoly_offset);
    if (d->tgrid) free(d->tgrid);
    if (d->time_ppoly_data) free(d->time_ppoly_data);
    if (d->time_ppoly_offset) free(d->time_ppoly_offset);
    if (d->time_rho_ppoly_data) free(d->time_rho_ppoly_data);
    if (d->time_rho_ppoly_offset) free(d->time_rho_ppoly_offset);
    if (d->rho_scratch) free(d->rho_scratch);
    free(d);
}

// ============================================================================
// PPoly evaluation helpers (Horner's method)
// ============================================================================

// Binary search for interval: find i such that rgrid[i] <= r < rgrid[i+1]
// Uses cached last_interval for O(1) amortized lookup during orbit integration.
static inline int ppoly_find_interval(const double *rgrid, int Nr, double r, int *last)
{
    int lo, hi, mid;
    // Check cached interval first
    int i = *last;
    if (i >= 0 && i < Nr - 1 && rgrid[i] <= r && r < rgrid[i + 1])
        return i;
    // Check neighbor intervals (common during orbit integration)
    if (i > 0 && rgrid[i - 1] <= r && r < rgrid[i])
        { *last = i - 1; return i - 1; }
    if (i < Nr - 2 && rgrid[i + 1] <= r && r < rgrid[i + 2])
        { *last = i + 1; return i + 1; }
    // Full binary search
    lo = 0;
    hi = Nr - 2;
    while (lo < hi) {
        mid = (lo + hi + 1) >> 1;
        if (rgrid[mid] <= r)
            lo = mid;
        else
            hi = mid - 1;
    }
    // Clamp to last interval for r == rmax
    if (lo >= Nr - 1) lo = Nr - 2;
    *last = lo;
    return lo;
}

// Evaluate PPoly value at r. coeffs points to 6*(Nr-1) doubles, interval-major.
static inline double ppoly_eval(const double *coeffs, const double *rgrid,
                                 int Nr, double r, int *last)
{
    int i = ppoly_find_interval(rgrid, Nr, r, last);
    double dx = r - rgrid[i];
    const double *c = coeffs + i * PPOLY_K;
    return ((((c[0]*dx + c[1])*dx + c[2])*dx + c[3])*dx + c[4])*dx + c[5];
}

// Evaluate PPoly 1st derivative at r.
static inline double ppoly_eval_deriv(const double *coeffs, const double *rgrid,
                                       int Nr, double r, int *last)
{
    int i = ppoly_find_interval(rgrid, Nr, r, last);
    double dx = r - rgrid[i];
    const double *c = coeffs + i * PPOLY_K;
    return (((5*c[0]*dx + 4*c[1])*dx + 3*c[2])*dx + 2*c[3])*dx + c[4];
}

// Combined: evaluate value, 1st and 2nd derivatives at r (one interval lookup).
static inline void ppoly_eval_all(const double *coeffs, const double *rgrid,
                                   int Nr, double r, int *last,
                                   double *val, double *d1, double *d2)
{
    int i = ppoly_find_interval(rgrid, Nr, r, last);
    double dx = r - rgrid[i];
    const double *c = coeffs + i * PPOLY_K;
    *val = ((((c[0]*dx + c[1])*dx + c[2])*dx + c[3])*dx + c[4])*dx + c[5];
    *d1 = (((5*c[0]*dx + 4*c[1])*dx + 3*c[2])*dx + 2*c[3])*dx + c[4];
    *d2 = ((20*c[0]*dx + 12*c[1])*dx + 6*c[2])*dx + 2*c[3];
}

// ============================================================================
// Initialization: consume pot_args, build PPoly + GSL rho splines
// ============================================================================

void initMultipoleExpansionPotentialArgs(struct potentialArg *potentialArgs,
                                         double **pot_args)
{
    struct multipole_data *d = (struct multipole_data *)malloc(sizeof(struct multipole_data));

    // Read header
    int Nr = (int) *(*pot_args)++;
    d->Nr = Nr;
    d->L = (int) *(*pot_args)++;
    d->M = (int) *(*pot_args)++;
    d->isNonAxi = (int) *(*pot_args)++;

    int L = d->L;
    int M = d->M;

    // Copy radial grid (owned by this struct)
    d->rgrid = (double *)malloc(Nr * sizeof(double));
    memcpy(d->rgrid, *pot_args, Nr * sizeof(double));
    double *rgrid = d->rgrid;
    *pot_args += Nr;
    d->rmin = rgrid[0];
    d->rmax = rgrid[Nr - 1];

    // Read amplitude
    d->amp = *(*pot_args)++;

    // Read Nt (0 = static, >0 = time-dependent)
    int Nt = (int) *(*pot_args)++;
    d->Nt = Nt;

    // Count PPoly and rho spline requirements per (l,m)
    int ppoly_coeffs_per_spline = PPOLY_K * (Nr - 1);
    d->rho_spline_offset = (int *)calloc(L * M, sizeof(int));
    d->ppoly_offset = (int *)calloc(L * M, sizeof(int));

    int nrho_splines = 0;
    int total_ppoly_doubles = 0;
    for (int l = 0; l < L; l++) {
        int mmax = l + 1 < M ? l + 1 : M;
        for (int m = 0; m < mmax; m++) {
            d->rho_spline_offset[l * M + m] = nrho_splines;
            d->ppoly_offset[l * M + m] = total_ppoly_doubles;
            // cos: I_inner + I_outer PPoly, 1 rho spline
            nrho_splines += 1;
            total_ppoly_doubles += 2 * ppoly_coeffs_per_spline;
            if (m > 0) {
                // sin: same
                nrho_splines += 1;
                total_ppoly_doubles += 2 * ppoly_coeffs_per_spline;
            }
        }
    }

    // Allocate GSL splines for rho only
    potentialArgs->nspline1d = nrho_splines;
    potentialArgs->spline1d = (gsl_spline **)malloc(nrho_splines * sizeof(gsl_spline *));
    potentialArgs->acc1d = (gsl_interp_accel **)malloc(nrho_splines * sizeof(gsl_interp_accel *));
    for (int i = 0; i < nrho_splines; i++) {
        potentialArgs->acc1d[i] = gsl_interp_accel_alloc();
        potentialArgs->spline1d[i] = gsl_spline_alloc(gsl_interp_cspline, Nr);
    }

    // Allocate PPoly storage
    d->ppoly_data = (double *)malloc(total_ppoly_doubles * sizeof(double));

    if (Nt > 0) {
        // ================================================================
        // Time-dependent path
        // ================================================================
        // Read tgrid
        d->tgrid = (double *)malloc(Nt * sizeof(double));
        memcpy(d->tgrid, *pot_args, Nt * sizeof(double));
        *pot_args += Nt;
        d->tmin = d->tgrid[0];
        d->tmax = d->tgrid[Nt - 1];
        d->last_t_interval = 0;
        d->cached_t = NAN;

        // Compute offsets for time_ppoly_data and time_rho_ppoly_data
        d->time_ppoly_offset = (int *)calloc(L * M, sizeof(int));
        d->time_rho_ppoly_offset = (int *)calloc(L * M, sizeof(int));
        int total_time_ppoly = 0;
        int total_time_rho_ppoly = 0;
        int n_r = ppoly_coeffs_per_spline;  // 6*(Nr-1)
        for (int l = 0; l < L; l++) {
            int mmax = l + 1 < M ? l + 1 : M;
            for (int m = 0; m < mmax; m++) {
                d->time_ppoly_offset[l * M + m] = total_time_ppoly;
                d->time_rho_ppoly_offset[l * M + m] = total_time_rho_ppoly;
                // cos: I_inner + I_outer time-PPoly, rho time-PPoly
                total_time_ppoly += 2 * TIME_PPOLY_K * (Nt - 1) * n_r;
                total_time_rho_ppoly += TIME_PPOLY_K * (Nt - 1) * Nr;
                if (m > 0) {
                    // sin: same
                    total_time_ppoly += 2 * TIME_PPOLY_K * (Nt - 1) * n_r;
                    total_time_rho_ppoly += TIME_PPOLY_K * (Nt - 1) * Nr;
                }
            }
        }

        // Allocate and read time-PPoly data
        d->time_ppoly_data = (double *)malloc(total_time_ppoly * sizeof(double));
        d->time_rho_ppoly_data = (double *)malloc(total_time_rho_ppoly * sizeof(double));
        d->rho_scratch = (double *)malloc(Nr * sizeof(double));

        // Read data per (l,m) from serialized args
        for (int l = 0; l < L; l++) {
            int mmax = l + 1 < M ? l + 1 : M;
            for (int m = 0; m < mmax; m++) {
                int n_trig = (m > 0) ? 2 : 1;
                int tp_base = d->time_ppoly_offset[l * M + m];
                int rho_tp_base = d->time_rho_ppoly_offset[l * M + m];

                for (int trig = 0; trig < n_trig; trig++) {
                    int tp_offset = tp_base + trig * 2 * TIME_PPOLY_K * (Nt - 1) * n_r;
                    int rho_tp_offset = rho_tp_base + trig * TIME_PPOLY_K * (Nt - 1) * Nr;

                    // I_inner time-PPoly
                    int inner_size = TIME_PPOLY_K * (Nt - 1) * n_r;
                    memcpy(d->time_ppoly_data + tp_offset, *pot_args,
                           inner_size * sizeof(double));
                    *pot_args += inner_size;

                    // I_outer time-PPoly
                    int outer_size = TIME_PPOLY_K * (Nt - 1) * n_r;
                    memcpy(d->time_ppoly_data + tp_offset + inner_size, *pot_args,
                           outer_size * sizeof(double));
                    *pot_args += outer_size;

                    // rho time-PPoly
                    int rho_size = TIME_PPOLY_K * (Nt - 1) * Nr;
                    memcpy(d->time_rho_ppoly_data + rho_tp_offset, *pot_args,
                           rho_size * sizeof(double));
                    *pot_args += rho_size;
                }
            }
        }

        // Initialize ppoly_data and rho splines with zeros (will be filled by ensure_time)
        memset(d->ppoly_data, 0, total_ppoly_doubles * sizeof(double));
        for (int i = 0; i < nrho_splines; i++) {
            // Initialize with zeros; ensure_time will reinit before first use
            double *zeros = (double *)calloc(Nr, sizeof(double));
            gsl_spline_init(potentialArgs->spline1d[i], rgrid, zeros, Nr);
            free(zeros);
        }
    } else {
        // ================================================================
        // Static path (Nt=0): existing code
        // ================================================================
        d->tgrid = NULL;
        d->tmin = d->tmax = 0.0;
        d->last_t_interval = 0;
        d->cached_t = 0.0;
        d->time_ppoly_data = NULL;
        d->time_ppoly_offset = NULL;
        d->time_rho_ppoly_data = NULL;
        d->time_rho_ppoly_offset = NULL;
        d->rho_scratch = NULL;

        // Read data per (l,m)
        // Layout: I_inner_cos PPoly (6*(Nr-1)), I_outer_cos PPoly (6*(Nr-1)),
        //         rho_cos (Nr) [, I_inner_sin, I_outer_sin, rho_sin for m>0]
        for (int l = 0; l < L; l++) {
            int mmax = l + 1 < M ? l + 1 : M;
            for (int m = 0; m < mmax; m++) {
                int pp_base = d->ppoly_offset[l * M + m];
                int rho_base = d->rho_spline_offset[l * M + m];

                // I_inner_cos PPoly coefficients
                memcpy(d->ppoly_data + pp_base, *pot_args, ppoly_coeffs_per_spline * sizeof(double));
                *pot_args += ppoly_coeffs_per_spline;
                // I_outer_cos PPoly coefficients
                memcpy(d->ppoly_data + pp_base + ppoly_coeffs_per_spline,
                       *pot_args, ppoly_coeffs_per_spline * sizeof(double));
                *pot_args += ppoly_coeffs_per_spline;
                // rho_cos values -> GSL spline
                gsl_spline_init(potentialArgs->spline1d[rho_base], rgrid, *pot_args, Nr);
                *pot_args += Nr;

                if (m > 0) {
                    int pp_sin = pp_base + 2 * ppoly_coeffs_per_spline;
                    // I_inner_sin PPoly
                    memcpy(d->ppoly_data + pp_sin, *pot_args, ppoly_coeffs_per_spline * sizeof(double));
                    *pot_args += ppoly_coeffs_per_spline;
                    // I_outer_sin PPoly
                    memcpy(d->ppoly_data + pp_sin + ppoly_coeffs_per_spline,
                           *pot_args, ppoly_coeffs_per_spline * sizeof(double));
                    *pot_args += ppoly_coeffs_per_spline;
                    // rho_sin -> GSL spline
                    gsl_spline_init(potentialArgs->spline1d[rho_base + 1], rgrid, *pot_args, Nr);
                    *pot_args += Nr;
                }
            }
        }
    }

    // Legendre array size
    if (d->isNonAxi)
        d->Psize = L * (L + 1) / 2;
    else
        d->Psize = L;

    // Initialize caches
    d->cache_valid = 0;
    d->last_interval = 0;

    // Store in potentialArgs
    potentialArgs->pot_data = d;
    potentialArgs->free_pot_data = &freeMultipoleData;
}

// ============================================================================
// Helpers: R_lm, dR_lm, d²R_lm from I_inner/I_outer PPoly splines
// ============================================================================

// Compute extended integrals for r < rmin assuming rho = const = rho(rmin).
// P_rho0 = pref*rho0 extracted from the I_inner PPoly derivative at rmin.
static inline void below_grid_integrals(const double *I_inner_pp, const double *I_outer_pp,
                                         const double *rgrid, int Nr, int *last,
                                         double r, int l, double rmin,
                                         double *I_inner_ext, double *I_outer_ext, double *P_rho0)
{
    *P_rho0 = ppoly_eval_deriv(I_inner_pp, rgrid, Nr, rmin, last) / pow(rmin, l + 2);
    *I_inner_ext = *P_rho0 / (l + 3) * pow(r, l + 3);
    double I_outer_rmin = ppoly_eval(I_outer_pp, rgrid, Nr, rmin, last);
    double extra;
    if (l == 2) {
        extra = *P_rho0 * log(rmin / r);
    } else {
        extra = *P_rho0 / (2 - l) * (pow(rmin, 2 - l) - pow(r, 2 - l));
    }
    *I_outer_ext = I_outer_rmin + extra;
}

// Evaluation mode flags for eval_radial_lm
#define EVAL_VALUE  0  // value only
#define EVAL_FORCE  1  // value + 1st derivative
#define EVAL_DERIV2 2  // value + 1st + 2nd derivative

// Compute R_lm(r) and optionally dR_lm/dr and d²R_lm/dr² using PPoly splines.
// mode: EVAL_VALUE (value only), EVAL_FORCE (+ 1st deriv), EVAL_DERIV2 (+ both derivs).
// dR_out and d2R_out may be NULL when not needed.
static inline double eval_radial_lm(const double *I_inner_pp, const double *I_outer_pp,
                                     const double *rgrid, int Nr, int *last,
                                     double r, int l, double rmin, double rmax,
                                     int mode, double *dR_out, double *d2R_out)
{
    if (r < rmin) {
        double I_inner_ext, I_outer_ext, P_rho0;
        below_grid_integrals(I_inner_pp, I_outer_pp, rgrid, Nr, last,
                             r, l, rmin, &I_inner_ext, &I_outer_ext, &P_rho0);
        double r_neg_lp1 = pow(r, -(l + 1));
        double r_l = pow(r, l);
        double R = r_neg_lp1 * I_inner_ext + r_l * I_outer_ext;
        if (mode >= EVAL_FORCE)
            *dR_out = -(l + 1) * r_neg_lp1 / r * I_inner_ext
                      + l * r_l / r * I_outer_ext;
        if (mode >= EVAL_DERIV2)
            *d2R_out = (l + 1) * (l + 2) * r_neg_lp1 / (r * r) * I_inner_ext
                       + l * (l - 1) * r_l / (r * r) * I_outer_ext
                       - (2 * l + 1) * P_rho0;
        return R;
    }
    if (r <= rmax) {
        double r_neg_lp1 = pow(r, -(l + 1));
        double r_l = pow(r, l);
        double I_inner, I_outer;
        if (mode >= EVAL_DERIV2) {
            double dI_inner, d2I_inner, dI_outer, d2I_outer;
            ppoly_eval_all(I_inner_pp, rgrid, Nr, r, last, &I_inner, &dI_inner, &d2I_inner);
            ppoly_eval_all(I_outer_pp, rgrid, Nr, r, last, &I_outer, &dI_outer, &d2I_outer);
            *dR_out = -(l + 1) * r_neg_lp1 / r * I_inner
                      + r_neg_lp1 * dI_inner
                      + l * r_l / r * I_outer
                      + r_l * dI_outer;
            *d2R_out = (l + 1) * (l + 2) * r_neg_lp1 / (r * r) * I_inner
                       - 2 * (l + 1) * r_neg_lp1 / r * dI_inner
                       + r_neg_lp1 * d2I_inner
                       + l * (l - 1) * r_l / (r * r) * I_outer
                       + 2 * l * r_l / r * dI_outer
                       + r_l * d2I_outer;
        } else if (mode >= EVAL_FORCE) {
            double dI_inner = ppoly_eval_deriv(I_inner_pp, rgrid, Nr, r, last);
            double dI_outer = ppoly_eval_deriv(I_outer_pp, rgrid, Nr, r, last);
            I_inner = ppoly_eval(I_inner_pp, rgrid, Nr, r, last);
            I_outer = ppoly_eval(I_outer_pp, rgrid, Nr, r, last);
            *dR_out = -(l + 1) * r_neg_lp1 / r * I_inner
                      + r_neg_lp1 * dI_inner
                      + l * r_l / r * I_outer
                      + r_l * dI_outer;
        } else {
            I_inner = ppoly_eval(I_inner_pp, rgrid, Nr, r, last);
            I_outer = ppoly_eval(I_outer_pp, rgrid, Nr, r, last);
        }
        return r_neg_lp1 * I_inner + r_l * I_outer;
    }
    // r > rmax: only I_inner contributes
    double I_inner_rmax = ppoly_eval(I_inner_pp, rgrid, Nr, rmax, last);
    double r_neg_lp1 = pow(r, -(l + 1));
    if (mode >= EVAL_FORCE)
        *dR_out = (-(l + 1)) * I_inner_rmax * r_neg_lp1 / r;
    if (mode >= EVAL_DERIV2)
        *d2R_out = (l + 1) * (l + 2) * I_inner_rmax * r_neg_lp1 / (r * r);
    return I_inner_rmax * r_neg_lp1;
}

// ============================================================================
// Helper: get PPoly coefficient pointers for a given (l,m) pair
// ============================================================================

// Returns pointers to I_inner and I_outer PPoly coefficient arrays for
// cosine (trig=0) or sine (trig=1) component.
static inline void get_ppoly_ptrs(const struct multipole_data *d,
                                   int l, int m, int trig,
                                   const double **I_inner_pp, const double **I_outer_pp)
{
    int pp_base = d->ppoly_offset[l * d->M + m];
    int ppoly_coeffs_per_spline = PPOLY_K * (d->Nr - 1);
    int offset = pp_base + trig * 2 * ppoly_coeffs_per_spline;
    *I_inner_pp = d->ppoly_data + offset;
    *I_outer_pp = d->ppoly_data + offset + ppoly_coeffs_per_spline;
}

// ============================================================================
// Time reconstruction: ensure PPoly/rho data matches current time
// ============================================================================

static void ensure_time(struct multipole_data *d,
                        struct potentialArg *potentialArgs,
                        double t)
{
    if (d->Nt == 0) return;
    if (d->cached_t == t) return;

    int i_t = ppoly_find_interval(d->tgrid, d->Nt, t, &d->last_t_interval);
    double dt = t - d->tgrid[i_t];
    int n_r = PPOLY_K * (d->Nr - 1);  // 6*(Nr-1) PPoly coefficients per spline
    int L = d->L, M = d->M;
    int Nt = d->Nt;
    int Nr = d->Nr;

    for (int l = 0; l < L; l++) {
        int mmax = l + 1 < M ? l + 1 : M;
        for (int m = 0; m < mmax; m++) {
            int n_trig = (m > 0) ? 2 : 1;
            for (int trig = 0; trig < n_trig; trig++) {
                // Reconstruct I_inner and I_outer PPoly-in-r coefficients
                int pp_dst = d->ppoly_offset[l * M + m] + trig * 2 * n_r;
                int tp_src = d->time_ppoly_offset[l * M + m]
                             + trig * 2 * TIME_PPOLY_K * (Nt - 1) * n_r;

                for (int s = 0; s < 2; s++) {  // inner, outer
                    double *dst = d->ppoly_data + pp_dst + s * n_r;
                    const double *src = d->time_ppoly_data + tp_src
                                        + s * TIME_PPOLY_K * (Nt - 1) * n_r
                                        + i_t * TIME_PPOLY_K * n_r;
                    // Horner: dst[j] = ((src[0*n+j]*dt + src[1*n+j])*dt + src[2*n+j])*dt + src[3*n+j]
                    memcpy(dst, src, n_r * sizeof(double));
                    for (int k = 1; k < TIME_PPOLY_K; k++) {
                        const double *ck = src + k * n_r;
                        for (int j = 0; j < n_r; j++)
                            dst[j] = dst[j] * dt + ck[j];
                    }
                }

                // Reconstruct rho values and reinit GSL spline
                int rho_idx = d->rho_spline_offset[l * M + m] + trig;
                int rho_tp = d->time_rho_ppoly_offset[l * M + m]
                             + trig * TIME_PPOLY_K * (Nt - 1) * Nr;
                const double *rho_src = d->time_rho_ppoly_data + rho_tp
                                        + i_t * TIME_PPOLY_K * Nr;
                double *rho_dst = d->rho_scratch;
                memcpy(rho_dst, rho_src, Nr * sizeof(double));
                for (int k = 1; k < TIME_PPOLY_K; k++) {
                    const double *ck = rho_src + k * Nr;
                    for (int j = 0; j < Nr; j++)
                        rho_dst[j] = rho_dst[j] * dt + ck[j];
                }
                gsl_interp_accel_reset(potentialArgs->acc1d[rho_idx]);
                gsl_spline_init(potentialArgs->spline1d[rho_idx],
                                d->rgrid, rho_dst, Nr);
            }
        }
    }
    d->cache_valid = 0;
    d->cached_t = t;
}

// ============================================================================
// Potential evaluation
// ============================================================================

double MultipoleExpansionPotentialEval(double R, double Z, double phi, double t,
                                       struct potentialArg *potentialArgs)
{
    struct multipole_data *d = (struct multipole_data *)potentialArgs->pot_data;
    ensure_time(d, potentialArgs, t);
    int L = d->L, M = d->M;

    double r, theta;
    cyl_to_spher(R, Z, &r, &theta);

    // LCOV_EXCL_START
    if (r == 0.0 || !isfinite(r))
        return 0.0;
    // LCOV_EXCL_STOP

    double costheta = cos(theta);

    double *P = (double *)malloc(d->Psize * sizeof(double));
    compute_legendre(costheta, L, d->isNonAxi ? M : 1, P);

    double result = 0.0;
    for (int l = 0; l < L; l++) {
        int mmax = l + 1 < M ? l + 1 : M;
        for (int m = 0; m < mmax; m++) {
            int pi = d->isNonAxi ? legendre_index(l, m, L) : l;
            const double *I_inner_pp, *I_outer_pp;

            // Cosine component
            get_ppoly_ptrs(d, l, m, 0, &I_inner_pp, &I_outer_pp);
            double radial = eval_radial_lm(I_inner_pp, I_outer_pp, d->rgrid, d->Nr,
                                            &d->last_interval, r, l, d->rmin, d->rmax,
                                            EVAL_VALUE, NULL, NULL)
                            * cos(m * phi);

            if (m > 0) {
                get_ppoly_ptrs(d, l, m, 1, &I_inner_pp, &I_outer_pp);
                radial += eval_radial_lm(I_inner_pp, I_outer_pp, d->rgrid, d->Nr,
                                          &d->last_interval, r, l, d->rmin, d->rmax,
                                          EVAL_VALUE, NULL, NULL)
                          * sin(m * phi);
            }
            result += P[pi] * radial;
        }
    }

    free(P);
    return d->amp * result;
}

// ============================================================================
// Spherical force computation with caching
// ============================================================================

static void compute_multipole_spher_forces(struct multipole_data *d,
                                           struct potentialArg *potentialArgs,
                                           double R, double Z, double phi,
                                           double t,
                                           double *F)
{
    ensure_time(d, potentialArgs, t);
    // Check cache
    if (d->cache_valid == MEP_CACHE_FORCE
        && d->cached_R == R
        && d->cached_Z == Z
        && d->cached_phi == phi) {
        F[0] = d->cached_F[0];
        F[1] = d->cached_F[1];
        F[2] = d->cached_F[2];
        return;
    }

    int L = d->L, M = d->M;

    double r, theta;
    cyl_to_spher(R, Z, &r, &theta);

    // LCOV_EXCL_START
    if (r == 0.0 || !isfinite(r)) {
        F[0] = F[1] = F[2] = 0.0;
        d->cache_valid = MEP_CACHE_FORCE;
        d->cached_R = R;
        d->cached_Z = Z;
        d->cached_phi = phi;
        d->cached_F[0] = d->cached_F[1] = d->cached_F[2] = 0.0;
        return;
    }
    // LCOV_EXCL_STOP

    double costheta = cos(theta);
    double sintheta = sin(theta);

    double *PdP = (double *)malloc(2 * d->Psize * sizeof(double));
    double *P = PdP;
    double *dP = P + d->Psize;

    if (d->isNonAxi)
        compute_legendre_deriv(costheta, L, M, P, dP);
    else
        compute_legendre_deriv(costheta, L, 1, P, dP);

    double dPhi_dr = 0.0, dPhi_dtheta = 0.0, dPhi_dphi = 0.0;

    for (int l = 0; l < L; l++) {
        int mmax = l + 1 < M ? l + 1 : M;
        for (int m = 0; m < mmax; m++) {
            int pi = d->isNonAxi ? legendre_index(l, m, L) : l;
            const double *I_inner_pp, *I_outer_pp;

            // Cosine component
            get_ppoly_ptrs(d, l, m, 0, &I_inner_pp, &I_outer_pp);
            double dradial_cos;
            double radial_cos = eval_radial_lm(I_inner_pp, I_outer_pp, d->rgrid, d->Nr,
                                                &d->last_interval, r, l, d->rmin, d->rmax,
                                                EVAL_FORCE, &dradial_cos, NULL);
            double cos_mphi = cos(m * phi);
            double sin_mphi = sin(m * phi);

            dPhi_dr += P[pi] * cos_mphi * dradial_cos;
            dPhi_dtheta += dP[pi] * (-sintheta) * cos_mphi * radial_cos;
            dPhi_dphi += P[pi] * (-m * sin_mphi) * radial_cos;

            if (m > 0) {
                get_ppoly_ptrs(d, l, m, 1, &I_inner_pp, &I_outer_pp);
                double dradial_sin;
                double radial_sin = eval_radial_lm(I_inner_pp, I_outer_pp, d->rgrid, d->Nr,
                                                    &d->last_interval, r, l, d->rmin, d->rmax,
                                                    EVAL_FORCE, &dradial_sin, NULL);
                dPhi_dr += P[pi] * sin_mphi * dradial_sin;
                dPhi_dtheta += dP[pi] * (-sintheta) * sin_mphi * radial_sin;
                dPhi_dphi += P[pi] * (m * cos_mphi) * radial_sin;
            }
        }
    }

    // Return negative gradient (force = -grad Phi), with amplitude
    F[0] = -d->amp * dPhi_dr;
    F[1] = -d->amp * dPhi_dtheta;
    F[2] = -d->amp * dPhi_dphi;

    free(PdP);

    // Update cache
    d->cache_valid = MEP_CACHE_FORCE;
    d->cached_R = R;
    d->cached_Z = Z;
    d->cached_phi = phi;
    d->cached_F[0] = F[0];
    d->cached_F[1] = F[1];
    d->cached_F[2] = F[2];
}

// ============================================================================
// Cylindrical force components
// ============================================================================

double MultipoleExpansionPotentialRforce(double R, double Z, double phi, double t,
                                         struct potentialArg *potentialArgs)
{
    double r, theta;
    cyl_to_spher(R, Z, &r, &theta);
    if (r == 0.0 || !isfinite(r)) return 0.0;
    double F[3];
    compute_multipole_spher_forces(
        (struct multipole_data *)potentialArgs->pot_data,
        potentialArgs, R, Z, phi, t, F);
    return F[0] * (R / r) + F[1] * (Z / (r * r));
}

double MultipoleExpansionPotentialzforce(double R, double Z, double phi, double t,
                                          struct potentialArg *potentialArgs)
{
    double r, theta;
    cyl_to_spher(R, Z, &r, &theta);
    if (r == 0.0 || !isfinite(r)) return 0.0;
    double F[3];
    compute_multipole_spher_forces(
        (struct multipole_data *)potentialArgs->pot_data,
        potentialArgs, R, Z, phi, t, F);
    return F[0] * (Z / r) + F[1] * (-R / (r * r));
}

double MultipoleExpansionPotentialphitorque(double R, double Z, double phi, double t,
                                            struct potentialArg *potentialArgs)
{
    double F[3];
    compute_multipole_spher_forces(
        (struct multipole_data *)potentialArgs->pot_data,
        potentialArgs, R, Z, phi, t, F);
    return F[2];
}

double MultipoleExpansionPotentialPlanarRforce(double R, double phi, double t,
                                               struct potentialArg *potentialArgs)
{
    return MultipoleExpansionPotentialRforce(R, 0.0, phi, t, potentialArgs);
}

double MultipoleExpansionPotentialPlanarphitorque(double R, double phi, double t,
                                                   struct potentialArg *potentialArgs)
{
    return MultipoleExpansionPotentialphitorque(R, 0.0, phi, t, potentialArgs);
}

// ============================================================================
// Density evaluation
// ============================================================================

double MultipoleExpansionPotentialDens(double R, double Z, double phi, double t,
                                       struct potentialArg *potentialArgs)
{
    struct multipole_data *d = (struct multipole_data *)potentialArgs->pot_data;
    ensure_time(d, potentialArgs, t);
    int L = d->L, M = d->M;

    double r, theta;
    cyl_to_spher(R, Z, &r, &theta);

    if (r == 0.0 || !isfinite(r) || r > d->rmax)
        return 0.0;
    if (r < d->rmin)
        r = d->rmin;

    double costheta = cos(theta);

    double *P = (double *)malloc(d->Psize * sizeof(double));
    if (d->isNonAxi)
        compute_legendre(costheta, L, M, P);
    else
        compute_legendre(costheta, L, 1, P);

    double result = 0.0;
    for (int l = 0; l < L; l++) {
        int mmax = l + 1 < M ? l + 1 : M;
        for (int m = 0; m < mmax; m++) {
            int rho_base = d->rho_spline_offset[l * M + m];
            int pi = d->isNonAxi ? legendre_index(l, m, L) : l;
            double rho_cos = gsl_spline_eval(potentialArgs->spline1d[rho_base],
                                             r, potentialArgs->acc1d[rho_base]);
            double contrib = P[pi] * cos(m * phi) * rho_cos;
            if (m > 0) {
                double rho_sin = gsl_spline_eval(potentialArgs->spline1d[rho_base + 1],
                                                 r, potentialArgs->acc1d[rho_base + 1]);
                contrib += P[pi] * sin(m * phi) * rho_sin;
            }
            result += contrib;
        }
    }

    free(P);
    return d->amp * result;
}

// ============================================================================
// Spherical second derivative computation with caching
// ============================================================================

static void compute_multipole_spher_2nd_derivs(struct multipole_data *d,
                                               struct potentialArg *potentialArgs,
                                               double R, double Z, double phi,
                                               double t,
                                               double *F)
{
    ensure_time(d, potentialArgs, t);
    // Check cache
    if (d->cache_valid == MEP_CACHE_DERIV
        && d->cached_R == R
        && d->cached_Z == Z
        && d->cached_phi == phi) {
        F[0] = d->cached_D[0];
        F[1] = d->cached_D[1];
        F[2] = d->cached_D[2];
        return;
    }

    int L = d->L, M = d->M;

    double r, theta;
    cyl_to_spher(R, Z, &r, &theta);

    // LCOV_EXCL_START
    if (r == 0.0 || !isfinite(r)) {
        F[0] = F[1] = F[2] = 0.0;
        d->cache_valid = MEP_CACHE_DERIV;
        d->cached_R = R;
        d->cached_Z = Z;
        d->cached_phi = phi;
        d->cached_D[0] = d->cached_D[1] = d->cached_D[2] = 0.0;
        return;
    }
    // LCOV_EXCL_STOP

    double costheta = cos(theta);

    double *P = (double *)malloc(d->Psize * sizeof(double));
    if (d->isNonAxi)
        compute_legendre(costheta, L, M, P);
    else
        compute_legendre(costheta, L, 1, P);

    double d2Phi_dr2 = 0.0, d2Phi_dphi2 = 0.0, d2Phi_drdphi = 0.0;

    for (int l = 0; l < L; l++) {
        int mmax = l + 1 < M ? l + 1 : M;
        for (int m = 0; m < mmax; m++) {
            int pi = d->isNonAxi ? legendre_index(l, m, L) : l;
            const double *I_inner_pp, *I_outer_pp;

            // Cosine component
            get_ppoly_ptrs(d, l, m, 0, &I_inner_pp, &I_outer_pp);
            double dradial_cos, d2radial_cos;
            double radial_cos = eval_radial_lm(I_inner_pp, I_outer_pp, d->rgrid, d->Nr,
                                                &d->last_interval, r, l, d->rmin, d->rmax,
                                                EVAL_DERIV2, &dradial_cos, &d2radial_cos);
            double cos_mphi = cos(m * phi);
            double sin_mphi = sin(m * phi);

            d2Phi_dr2 += P[pi] * cos_mphi * d2radial_cos;
            d2Phi_dphi2 += P[pi] * (-m * m * cos_mphi) * radial_cos;
            d2Phi_drdphi += P[pi] * (-m * sin_mphi) * dradial_cos;

            if (m > 0) {
                get_ppoly_ptrs(d, l, m, 1, &I_inner_pp, &I_outer_pp);
                double dradial_sin, d2radial_sin;
                double radial_sin = eval_radial_lm(I_inner_pp, I_outer_pp, d->rgrid, d->Nr,
                                                    &d->last_interval, r, l, d->rmin, d->rmax,
                                                    EVAL_DERIV2, &dradial_sin, &d2radial_sin);
                d2Phi_dr2 += P[pi] * sin_mphi * d2radial_sin;
                d2Phi_dphi2 += P[pi] * (-m * m * sin_mphi) * radial_sin;
                d2Phi_drdphi += P[pi] * (m * cos_mphi) * dradial_sin;
            }
        }
    }

    F[0] = d->amp * d2Phi_dr2;
    F[1] = d->amp * d2Phi_dphi2;
    F[2] = d->amp * d2Phi_drdphi;

    free(P);

    // Update cache
    d->cache_valid = MEP_CACHE_DERIV;
    d->cached_R = R;
    d->cached_Z = Z;
    d->cached_phi = phi;
    d->cached_D[0] = F[0];
    d->cached_D[1] = F[1];
    d->cached_D[2] = F[2];
}

// ============================================================================
// Second derivative components (planar)
// ============================================================================

double MultipoleExpansionPotentialPlanarR2deriv(double R, double phi, double t,
                                                struct potentialArg *potentialArgs)
{
    double F[3];
    compute_multipole_spher_2nd_derivs(
        (struct multipole_data *)potentialArgs->pot_data,
        potentialArgs, R, 0.0, phi, t, F);
    return F[0];
}

double MultipoleExpansionPotentialPlanarphi2deriv(double R, double phi, double t,
                                                   struct potentialArg *potentialArgs)
{
    double F[3];
    compute_multipole_spher_2nd_derivs(
        (struct multipole_data *)potentialArgs->pot_data,
        potentialArgs, R, 0.0, phi, t, F);
    return F[1];
}

double MultipoleExpansionPotentialPlanarRphideriv(double R, double phi, double t,
                                                    struct potentialArg *potentialArgs)
{
    double F[3];
    compute_multipole_spher_2nd_derivs(
        (struct multipole_data *)potentialArgs->pot_data,
        potentialArgs, R, 0.0, phi, t, F);
    return F[2];
}
