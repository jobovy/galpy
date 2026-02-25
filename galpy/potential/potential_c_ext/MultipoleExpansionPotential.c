// MultipoleExpansionPotential: C implementation of multipole expansion potential
#include <math.h>
#include <stdlib.h>
#include <gsl/gsl_spline.h>
#include <gsl/gsl_sf_legendre.h>
#include <galpy_potentials.h>

#ifndef GSL_MAJOR_VERSION
#define GSL_MAJOR_VERSION 1
#endif
// Cache tags for force/deriv caching
#define MEP_CACHE_FORCE 1
#define MEP_CACHE_DERIV 2

// ============================================================================
// Pre-computed multipole data
// ============================================================================

struct multipole_data {
    int L, M, isNonAxi;
    int *spline_offset;     // offset into spline1d array for each (l,m), size L*M
    double amp;             // overall amplitude
    int total_splines;      // total number of splines
    double rmin, rmax;      // radial grid bounds for clamping
    // Legendre array size
    int Psize;
    // Caching for forces (per potentialArgs copy, thread-safe)
    double cached_R, cached_Z, cached_phi;
    double cached_F[3];     // dPhi/dr, dPhi/dtheta, dPhi/dphi
    double cached_D[3];     // d2Phi/dr2, d2Phi/dphi2, d2Phi/drdphi
    int cache_valid;
};

static void freeMultipoleData(void *data)
{
    struct multipole_data *d = (struct multipole_data *)data;
    if (d->spline_offset) free(d->spline_offset);
    free(d);
}

// ============================================================================
// Initialization: consume pot_args, build GSL splines
// ============================================================================

void initMultipoleExpansionPotentialArgs(struct potentialArg *potentialArgs,
                                         double **pot_args)
{
    struct multipole_data *d = (struct multipole_data *)malloc(sizeof(struct multipole_data));

    // Read header
    int Nr = (int) *(*pot_args)++;
    d->L = (int) *(*pot_args)++;
    d->M = (int) *(*pot_args)++;
    d->isNonAxi = (int) *(*pot_args)++;

    int L = d->L;
    int M = d->M;

    // Read shared radial grid
    double *rgrid = *pot_args;
    *pot_args += Nr;
    d->rmin = rgrid[0];
    d->rmax = rgrid[Nr - 1];

    // Read amplitude
    d->amp = *(*pot_args)++;

    // Count total splines and build offset table
    // Per (l,m): I_inner_cos, I_outer_cos, rho_cos (+ sin variants for m>0)
    d->spline_offset = (int *)calloc(L * M, sizeof(int));
    int nsplines = 0;
    for (int l = 0; l < L; l++) {
        int mmax = l + 1 < M ? l + 1 : M;
        for (int m = 0; m < mmax; m++) {
            d->spline_offset[l * M + m] = nsplines;
            nsplines += (m > 0) ? 6 : 3;  // cos: I_inner+I_outer+rho; sin: same
        }
    }
    d->total_splines = nsplines;

    // Allocate spline arrays
    potentialArgs->nspline1d = nsplines;
    potentialArgs->spline1d = (gsl_spline **)malloc(nsplines * sizeof(gsl_spline *));
    potentialArgs->acc1d = (gsl_interp_accel **)malloc(nsplines * sizeof(gsl_interp_accel *));

    for (int i = 0; i < nsplines; i++) {
        potentialArgs->acc1d[i] = gsl_interp_accel_alloc();
        potentialArgs->spline1d[i] = gsl_spline_alloc(gsl_interp_cspline, Nr);
    }

    // Read spline y-values and initialize splines
    // Layout per (l,m): I_inner_cos, I_outer_cos, rho_cos [, I_inner_sin, I_outer_sin, rho_sin]
    for (int l = 0; l < L; l++) {
        int mmax = l + 1 < M ? l + 1 : M;
        for (int m = 0; m < mmax; m++) {
            int base = d->spline_offset[l * M + m];
            // I_inner_cos
            gsl_spline_init(potentialArgs->spline1d[base + 0], rgrid, *pot_args, Nr);
            *pot_args += Nr;
            // I_outer_cos
            gsl_spline_init(potentialArgs->spline1d[base + 1], rgrid, *pot_args, Nr);
            *pot_args += Nr;
            // rho_cos
            gsl_spline_init(potentialArgs->spline1d[base + 2], rgrid, *pot_args, Nr);
            *pot_args += Nr;
            if (m > 0) {
                // I_inner_sin
                gsl_spline_init(potentialArgs->spline1d[base + 3], rgrid, *pot_args, Nr);
                *pot_args += Nr;
                // I_outer_sin
                gsl_spline_init(potentialArgs->spline1d[base + 4], rgrid, *pot_args, Nr);
                *pot_args += Nr;
                // rho_sin
                gsl_spline_init(potentialArgs->spline1d[base + 5], rgrid, *pot_args, Nr);
                *pot_args += Nr;
            }
        }
    }

    // Legendre array size
    if (d->isNonAxi)
        d->Psize = L * (L + 1) / 2;
    else
        d->Psize = L;

    // Initialize cache as invalid
    d->cache_valid = 0;

    // Store in potentialArgs
    potentialArgs->pot_data = d;
    potentialArgs->free_pot_data = &freeMultipoleData;
}

// ============================================================================
// Helpers: analytical R_lm, dR_lm, d²R_lm from I_inner/I_outer spline values
// ============================================================================

// Compute extended integrals for r < rmin assuming rho = const = rho(rmin).
// P_rho0 = pref*rho0 extracted from the I_inner spline derivative at rmin.
static inline void below_grid_integrals(gsl_spline *I_inner_sp, gsl_interp_accel *I_inner_acc,
                                         gsl_spline *I_outer_sp, gsl_interp_accel *I_outer_acc,
                                         double r, int l, double rmin,
                                         double *I_inner_ext, double *I_outer_ext, double *P_rho0)
{
    *P_rho0 = gsl_spline_eval_deriv(I_inner_sp, rmin, I_inner_acc) / pow(rmin, l + 2);
    *I_inner_ext = *P_rho0 / (l + 3) * pow(r, l + 3);
    double I_outer_rmin = gsl_spline_eval(I_outer_sp, rmin, I_outer_acc);
    double extra;
    if (l == 2) {
        extra = *P_rho0 * log(rmin / r);
    } else {
        extra = *P_rho0 / (2 - l) * (pow(rmin, 2 - l) - pow(r, 2 - l));
    }
    *I_outer_ext = I_outer_rmin + extra;
}

// Compute R_lm(r) = r^{-(l+1)} * I_inner + r^l * I_outer
// Extrapolation:
// r < rmin: constant density rho(rmin) assumed below grid
// r > rmax: point-mass (rho=0 above grid, I_outer(rmax)=0)
static inline double eval_R_lm(gsl_spline *I_inner_sp, gsl_interp_accel *I_inner_acc,
                                gsl_spline *I_outer_sp, gsl_interp_accel *I_outer_acc,
                                double r, int l, double rmin, double rmax)
{
    if (r < rmin) {
        double I_inner_ext, I_outer_ext, P_rho0;
        below_grid_integrals(I_inner_sp, I_inner_acc, I_outer_sp, I_outer_acc,
                             r, l, rmin, &I_inner_ext, &I_outer_ext, &P_rho0);
        return pow(r, -(l + 1)) * I_inner_ext + pow(r, l) * I_outer_ext;
    }
    if (r <= rmax) {
        double I_inner = gsl_spline_eval(I_inner_sp, r, I_inner_acc);
        double I_outer = gsl_spline_eval(I_outer_sp, r, I_outer_acc);
        return pow(r, -(l + 1)) * I_inner + pow(r, l) * I_outer;
    }
    double I_inner_rmax = gsl_spline_eval(I_inner_sp, rmax, I_inner_acc);
    return I_inner_rmax * pow(r, -(l + 1));
}

// Compute dR_lm/dr via product rule using spline values and derivatives.
// For r < rmin with constant density, the dI/dr terms cancel:
// dR/dr = -(l+1) r^{-(l+2)} I_inner_ext + l r^{l-1} I_outer_ext
static inline double eval_dR_lm(gsl_spline *I_inner_sp, gsl_interp_accel *I_inner_acc,
                                 gsl_spline *I_outer_sp, gsl_interp_accel *I_outer_acc,
                                 double r, int l, double rmin, double rmax)
{
    if (r < rmin) {
        double I_inner_ext, I_outer_ext, P_rho0;
        below_grid_integrals(I_inner_sp, I_inner_acc, I_outer_sp, I_outer_acc,
                             r, l, rmin, &I_inner_ext, &I_outer_ext, &P_rho0);
        return -(l + 1) * pow(r, -(l + 2)) * I_inner_ext
               + l * pow(r, l - 1) * I_outer_ext;
    }
    if (r <= rmax) {
        double I_inner = gsl_spline_eval(I_inner_sp, r, I_inner_acc);
        double I_outer = gsl_spline_eval(I_outer_sp, r, I_outer_acc);
        double dI_inner = gsl_spline_eval_deriv(I_inner_sp, r, I_inner_acc);
        double dI_outer = gsl_spline_eval_deriv(I_outer_sp, r, I_outer_acc);
        return -(l + 1) * pow(r, -(l + 2)) * I_inner
               + pow(r, -(l + 1)) * dI_inner
               + l * pow(r, l - 1) * I_outer
               + pow(r, l) * dI_outer;
    }
    double I_inner_rmax = gsl_spline_eval(I_inner_sp, rmax, I_inner_acc);
    return (-(l + 1)) * I_inner_rmax * pow(r, -(l + 2));
}

// Compute d²R_lm/dr² via product rule.
// For r < rmin with constant density:
// d²R/dr² = (l+1)(l+2) r^{-(l+3)} I_inner_ext + l(l-1) r^{l-2} I_outer_ext
//           - (2l+1) * P_rho0
static inline double eval_d2R_lm(gsl_spline *I_inner_sp, gsl_interp_accel *I_inner_acc,
                                  gsl_spline *I_outer_sp, gsl_interp_accel *I_outer_acc,
                                  gsl_spline *rho_sp, gsl_interp_accel *rho_acc,
                                  double r, int l, double rmin, double rmax)
{
    (void)rho_sp;
    (void)rho_acc;
    if (r < rmin) {
        double I_inner_ext, I_outer_ext, P_rho0;
        below_grid_integrals(I_inner_sp, I_inner_acc, I_outer_sp, I_outer_acc,
                             r, l, rmin, &I_inner_ext, &I_outer_ext, &P_rho0);
        return (l + 1) * (l + 2) * pow(r, -(l + 3)) * I_inner_ext
               + l * (l - 1) * pow(r, l - 2) * I_outer_ext
               - (2 * l + 1) * P_rho0;
    }
    if (r <= rmax) {
        double I_inner = gsl_spline_eval(I_inner_sp, r, I_inner_acc);
        double I_outer = gsl_spline_eval(I_outer_sp, r, I_outer_acc);
        double dI_inner = gsl_spline_eval_deriv(I_inner_sp, r, I_inner_acc);
        double dI_outer = gsl_spline_eval_deriv(I_outer_sp, r, I_outer_acc);
        double d2I_inner = gsl_spline_eval_deriv2(I_inner_sp, r, I_inner_acc);
        double d2I_outer = gsl_spline_eval_deriv2(I_outer_sp, r, I_outer_acc);
        return (l + 1) * (l + 2) * pow(r, -(l + 3)) * I_inner
               - 2 * (l + 1) * pow(r, -(l + 2)) * dI_inner
               + pow(r, -(l + 1)) * d2I_inner
               + l * (l - 1) * pow(r, l - 2) * I_outer
               + 2 * l * pow(r, l - 1) * dI_outer
               + pow(r, l) * d2I_outer;
    }
    double I_inner_rmax = gsl_spline_eval(I_inner_sp, rmax, I_inner_acc);
    return (l + 1) * (l + 2) * I_inner_rmax * pow(r, -(l + 3));
}

// ============================================================================
// Potential evaluation
// ============================================================================

double MultipoleExpansionPotentialEval(double R, double Z, double phi, double t,
                                       struct potentialArg *potentialArgs)
{
    struct multipole_data *d = (struct multipole_data *)potentialArgs->pot_data;
    int L = d->L, M = d->M;

    double r, theta;
    cyl_to_spher(R, Z, &r, &theta);

    // LCOV_EXCL_START
    if (r == 0.0 || !isfinite(r))
        return 0.0;
    // LCOV_EXCL_STOP

    double costheta = cos(theta);

    // Allocate Legendre workspace
    double *P = (double *)malloc(d->Psize * sizeof(double));
    compute_legendre(costheta, L, d->isNonAxi ? M : 1, P);

    double result = 0.0;
    for (int l = 0; l < L; l++) {
        int mmax = l + 1 < M ? l + 1 : M;
        for (int m = 0; m < mmax; m++) {
            int base = d->spline_offset[l * M + m];
            int pi = d->isNonAxi ? legendre_index(l, m, L) : l;
            double radial = eval_R_lm(potentialArgs->spline1d[base + 0],
                                       potentialArgs->acc1d[base + 0],
                                       potentialArgs->spline1d[base + 1],
                                       potentialArgs->acc1d[base + 1],
                                       r, l, d->rmin, d->rmax) * cos(m * phi);
            radial += ( m > 0 ) ? eval_R_lm(potentialArgs->spline1d[base + 3],potentialArgs->acc1d[base + 3],potentialArgs->spline1d[base + 4],potentialArgs->acc1d[base + 4],r, l, d->rmin, d->rmax) * sin(m * phi) : 0.0;
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
                                           double *F)
{
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

    // Allocate Legendre workspace (P and dP)
    double *ws = (double *)malloc(2 * d->Psize * sizeof(double));
    double *P = ws;
    double *dP = ws + d->Psize;

    if (d->isNonAxi)
        compute_legendre_deriv(costheta, L, M, P, dP);
    else
        compute_legendre_deriv(costheta, L, 1, P, dP);

    double dPhi_dr = 0.0, dPhi_dtheta = 0.0, dPhi_dphi = 0.0;

    for (int l = 0; l < L; l++) {
        int mmax = l + 1 < M ? l + 1 : M;
        for (int m = 0; m < mmax; m++) {
            int base = d->spline_offset[l * M + m];
            int pi = d->isNonAxi ? legendre_index(l, m, L) : l;

            double radial_cos = eval_R_lm(potentialArgs->spline1d[base + 0],
                                            potentialArgs->acc1d[base + 0],
                                            potentialArgs->spline1d[base + 1],
                                            potentialArgs->acc1d[base + 1],
                                            r, l, d->rmin, d->rmax);
            double dradial_cos = eval_dR_lm(potentialArgs->spline1d[base + 0],
                                              potentialArgs->acc1d[base + 0],
                                              potentialArgs->spline1d[base + 1],
                                              potentialArgs->acc1d[base + 1],
                                              r, l, d->rmin, d->rmax);
            double cos_mphi = cos(m * phi);
            double sin_mphi = sin(m * phi);

            // dPhi/dr
            dPhi_dr += P[pi] * cos_mphi * dradial_cos;
            // dPhi/dtheta: dP_l^m/dtheta = dP_l^m/d(costheta) * (-sin(theta))
            dPhi_dtheta += dP[pi] * (-sintheta) * cos_mphi * radial_cos;
            // dPhi/dphi
            dPhi_dphi += P[pi] * (-m * sin_mphi) * radial_cos;

            if (m > 0) {
                double radial_sin = eval_R_lm(potentialArgs->spline1d[base + 3],
                                                potentialArgs->acc1d[base + 3],
                                                potentialArgs->spline1d[base + 4],
                                                potentialArgs->acc1d[base + 4],
                                                r, l, d->rmin, d->rmax);
                double dradial_sin = eval_dR_lm(potentialArgs->spline1d[base + 3],
                                                  potentialArgs->acc1d[base + 3],
                                                  potentialArgs->spline1d[base + 4],
                                                  potentialArgs->acc1d[base + 4],
                                                  r, l, d->rmin, d->rmax);
                dPhi_dr += P[pi] * sin_mphi * dradial_sin;
                dPhi_dtheta += dP[pi] * (-sintheta) * sin_mphi * radial_sin;
                dPhi_dphi += P[pi] * (m * cos_mphi) * radial_sin;
            }
        }
    }

    free(ws);

    // Return negative gradient (force = -grad Phi), with amplitude
    F[0] = -d->amp * dPhi_dr;
    F[1] = -d->amp * dPhi_dtheta;
    F[2] = -d->amp * dPhi_dphi;

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
        potentialArgs, R, Z, phi, F);
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
        potentialArgs, R, Z, phi, F);
    return F[0] * (Z / r) + F[1] * (-R / (r * r));
}

double MultipoleExpansionPotentialphitorque(double R, double Z, double phi, double t,
                                            struct potentialArg *potentialArgs)
{
    double F[3];
    compute_multipole_spher_forces(
        (struct multipole_data *)potentialArgs->pot_data,
        potentialArgs, R, Z, phi, F);
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
    int L = d->L, M = d->M;

    double r, theta;
    cyl_to_spher(R, Z, &r, &theta);

    if (r == 0.0 || !isfinite(r) || r > d->rmax)
        return 0.0;
    if (r < d->rmin)
        r = d->rmin;

    double costheta = cos(theta);

    // Allocate Legendre workspace
    double *P = (double *)malloc(d->Psize * sizeof(double));
    if (d->isNonAxi)
        compute_legendre(costheta, L, M, P);
    else
        compute_legendre(costheta, L, 1, P);

    double result = 0.0;
    for (int l = 0; l < L; l++) {
        int mmax = l + 1 < M ? l + 1 : M;
        for (int m = 0; m < mmax; m++) {
            int base = d->spline_offset[l * M + m];
            int pi = d->isNonAxi ? legendre_index(l, m, L) : l;
            // rho_cos is at offset +2
            double rho_cos = gsl_spline_eval(potentialArgs->spline1d[base + 2],
                                             r, potentialArgs->acc1d[base + 2]);
            double contrib = P[pi] * cos(m * phi) * rho_cos;
            if (m > 0) {
                // rho_sin is at offset +5
                double rho_sin = gsl_spline_eval(potentialArgs->spline1d[base + 5],
                                                 r, potentialArgs->acc1d[base + 5]);
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

// Compute spherical second derivative components at z=0 (theta=pi/2):
// F[0] = d²Phi/dr², F[1] = d²Phi/dphi², F[2] = d²Phi/drdphi
// At z=0, cylindrical and spherical derivatives coincide so no theta
// derivatives of the Legendre polynomials are needed.
static void compute_multipole_spher_2nd_derivs(struct multipole_data *d,
                                               struct potentialArg *potentialArgs,
                                               double R, double Z, double phi,
                                               double *F)
{
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

    // Only P values needed (no derivatives)
    double *P = (double *)malloc(d->Psize * sizeof(double));
    if (d->isNonAxi)
        compute_legendre(costheta, L, M, P);
    else
        compute_legendre(costheta, L, 1, P);

    double d2Phi_dr2 = 0.0, d2Phi_dphi2 = 0.0, d2Phi_drdphi = 0.0;

    for (int l = 0; l < L; l++) {
        int mmax = l + 1 < M ? l + 1 : M;
        for (int m = 0; m < mmax; m++) {
            int base = d->spline_offset[l * M + m];
            int pi = d->isNonAxi ? legendre_index(l, m, L) : l;

            double radial_cos = eval_R_lm(potentialArgs->spline1d[base + 0],
                                            potentialArgs->acc1d[base + 0],
                                            potentialArgs->spline1d[base + 1],
                                            potentialArgs->acc1d[base + 1],
                                            r, l, d->rmin, d->rmax);
            double dradial_cos = eval_dR_lm(potentialArgs->spline1d[base + 0],
                                              potentialArgs->acc1d[base + 0],
                                              potentialArgs->spline1d[base + 1],
                                              potentialArgs->acc1d[base + 1],
                                              r, l, d->rmin, d->rmax);
            double d2radial_cos = eval_d2R_lm(potentialArgs->spline1d[base + 0],
                                                potentialArgs->acc1d[base + 0],
                                                potentialArgs->spline1d[base + 1],
                                                potentialArgs->acc1d[base + 1],
                                                potentialArgs->spline1d[base + 2],
                                                potentialArgs->acc1d[base + 2],
                                                r, l, d->rmin, d->rmax);
            double cos_mphi = cos(m * phi);
            double sin_mphi = sin(m * phi);

            // d²Phi/dr²
            d2Phi_dr2 += P[pi] * cos_mphi * d2radial_cos;
            // d²Phi/dphi² : d²cos(mφ)/dφ² = -m²cos(mφ)
            d2Phi_dphi2 += P[pi] * (-m * m * cos_mphi) * radial_cos;
            // d²Phi/drdφ : dcos(mφ)/dφ = -m sin(mφ)
            d2Phi_drdphi += P[pi] * (-m * sin_mphi) * dradial_cos;

            if (m > 0) {
                double radial_sin = eval_R_lm(potentialArgs->spline1d[base + 3],
                                                potentialArgs->acc1d[base + 3],
                                                potentialArgs->spline1d[base + 4],
                                                potentialArgs->acc1d[base + 4],
                                                r, l, d->rmin, d->rmax);
                double dradial_sin = eval_dR_lm(potentialArgs->spline1d[base + 3],
                                                  potentialArgs->acc1d[base + 3],
                                                  potentialArgs->spline1d[base + 4],
                                                  potentialArgs->acc1d[base + 4],
                                                  r, l, d->rmin, d->rmax);
                double d2radial_sin = eval_d2R_lm(potentialArgs->spline1d[base + 3],
                                                    potentialArgs->acc1d[base + 3],
                                                    potentialArgs->spline1d[base + 4],
                                                    potentialArgs->acc1d[base + 4],
                                                    potentialArgs->spline1d[base + 5],
                                                    potentialArgs->acc1d[base + 5],
                                                    r, l, d->rmin, d->rmax);
                d2Phi_dr2 += P[pi] * sin_mphi * d2radial_sin;
                d2Phi_dphi2 += P[pi] * (-m * m * sin_mphi) * radial_sin;
                d2Phi_drdphi += P[pi] * (m * cos_mphi) * dradial_sin;
            }
        }
    }

    free(P);

    F[0] = d->amp * d2Phi_dr2;
    F[1] = d->amp * d2Phi_dphi2;
    F[2] = d->amp * d2Phi_drdphi;

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
        potentialArgs, R, 0.0, phi, F);
    return F[0];
}

double MultipoleExpansionPotentialPlanarphi2deriv(double R, double phi, double t,
                                                   struct potentialArg *potentialArgs)
{
    double F[3];
    compute_multipole_spher_2nd_derivs(
        (struct multipole_data *)potentialArgs->pot_data,
        potentialArgs, R, 0.0, phi, F);
    return F[1];
}

double MultipoleExpansionPotentialPlanarRphideriv(double R, double phi, double t,
                                                    struct potentialArg *potentialArgs)
{
    double F[3];
    compute_multipole_spher_2nd_derivs(
        (struct multipole_data *)potentialArgs->pot_data,
        potentialArgs, R, 0.0, phi, F);
    return F[2];
}
