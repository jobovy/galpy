// MultipoleExpansionPotential: C implementation of multipole expansion potential
// Uses GSL splines for radial functions (Radial_cos/sin, rho_cos/sin)
// and shared Legendre utilities from SCFPotential.c
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

    // Count total splines and build offset table
    d->spline_offset = (int *)calloc(L * M, sizeof(int));
    int nsplines = 0;
    for (int l = 0; l < L; l++) {
        int mmax = l + 1 < M ? l + 1 : M;
        for (int m = 0; m < mmax; m++) {
            d->spline_offset[l * M + m] = nsplines;
            nsplines += (m > 0) ? 4 : 2;  // cos: Radial+rho; sin: Radial+rho
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
    for (int l = 0; l < L; l++) {
        int mmax = l + 1 < M ? l + 1 : M;
        for (int m = 0; m < mmax; m++) {
            int base = d->spline_offset[l * M + m];
            // Radial_cos
            gsl_spline_init(potentialArgs->spline1d[base + 0], rgrid, *pot_args, Nr);
            *pot_args += Nr;
            // rho_cos
            gsl_spline_init(potentialArgs->spline1d[base + 1], rgrid, *pot_args, Nr);
            *pot_args += Nr;
            if (m > 0) {
                // Radial_sin
                gsl_spline_init(potentialArgs->spline1d[base + 2], rgrid, *pot_args, Nr);
                *pot_args += Nr;
                // rho_sin
                gsl_spline_init(potentialArgs->spline1d[base + 3], rgrid, *pot_args, Nr);
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
// Helpers
// ============================================================================

// Evaluate radial spline value, with point-mass extrapolation for r > rmax.
// For r > rmax, R_lm(r) = R_lm(rmax) * (rmax/r)^{l+1} (all mass inside rmax).
// For r < rmin, clamp to rmin.
static inline double eval_radial(gsl_spline *spline, gsl_interp_accel *acc,
                                 double r, int l, double rmin, double rmax)
{
    if (r <= rmax) {
        double rc = r < rmin ? rmin : r;
        return gsl_spline_eval(spline, rc, acc);
    }
    double val_at_rmax = gsl_spline_eval(spline, rmax, acc);
    return val_at_rmax * pow(rmax / r, l + 1);
}

// Evaluate radial spline derivative, with point-mass extrapolation for r > rmax.
// For r > rmax, dR_lm/dr = -(l+1)/r * R_lm(r).
// For r < rmin, clamp to rmin.
static inline double eval_radial_deriv(gsl_spline *spline, gsl_interp_accel *acc,
                                       double r, int l, double rmin, double rmax)
{
    if (r <= rmax) {
        double rc = r < rmin ? rmin : r;
        return gsl_spline_eval_deriv(spline, rc, acc);
    }
    double val_at_rmax = gsl_spline_eval(spline, rmax, acc);
    double radial = val_at_rmax * pow(rmax / r, l + 1);
    return -(l + 1) / r * radial;
}

// Evaluate radial spline second derivative, with point-mass extrapolation for r > rmax.
// For r > rmax, d²R_lm/dr² = (l+1)(l+2)/r² * R_lm(r).
// For r < rmin, clamp to rmin.
static inline double eval_radial_deriv2(gsl_spline *spline, gsl_interp_accel *acc,
                                        double r, int l, double rmin, double rmax)
{
    if (r <= rmax) {
        double rc = r < rmin ? rmin : r;
        return gsl_spline_eval_deriv2(spline, rc, acc);
    }
    double val_at_rmax = gsl_spline_eval(spline, rmax, acc);
    double radial = val_at_rmax * pow(rmax / r, l + 1);
    return (l + 1) * (l + 2) / (r * r) * radial;
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
            double radial = eval_radial(potentialArgs->spline1d[base + 0],
                                 potentialArgs->acc1d[base + 0],
                                 r, l, d->rmin, d->rmax) * cos(m * phi);
            radial += (m > 0) ? eval_radial(potentialArgs->spline1d[base + 2],potentialArgs->acc1d[base + 2],r, l, d->rmin, d->rmax) * sin(m * phi) : 0.0;
            result += P[pi] * radial;
        }
    }

    free(P);
    return result;
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

            double radial_cos = eval_radial(potentialArgs->spline1d[base + 0],
                                            potentialArgs->acc1d[base + 0],
                                            r, l, d->rmin, d->rmax);
            double dradial_cos = eval_radial_deriv(potentialArgs->spline1d[base + 0],
                                                   potentialArgs->acc1d[base + 0],
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
                double radial_sin = eval_radial(potentialArgs->spline1d[base + 2],
                                                potentialArgs->acc1d[base + 2],
                                                r, l, d->rmin, d->rmax);
                double dradial_sin = eval_radial_deriv(potentialArgs->spline1d[base + 2],
                                                       potentialArgs->acc1d[base + 2],
                                                       r, l, d->rmin, d->rmax);
                dPhi_dr += P[pi] * sin_mphi * dradial_sin;
                dPhi_dtheta += dP[pi] * (-sintheta) * sin_mphi * radial_sin;
                dPhi_dphi += P[pi] * (m * cos_mphi) * radial_sin;
            }
        }
    }

    free(ws);

    // Return negative gradient (force = -grad Phi)
    F[0] = -dPhi_dr;
    F[1] = -dPhi_dtheta;
    F[2] = -dPhi_dphi;

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
            double rho_cos = gsl_spline_eval(potentialArgs->spline1d[base + 1],
                                             r, potentialArgs->acc1d[base + 1]);
            double contrib = P[pi] * cos(m * phi) * rho_cos;
            if (m > 0) {
                double rho_sin = gsl_spline_eval(potentialArgs->spline1d[base + 3],
                                                 r, potentialArgs->acc1d[base + 3]);
                contrib += P[pi] * sin(m * phi) * rho_sin;
            }
            result += contrib;
        }
    }

    free(P);
    return result;
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

            double radial_cos = eval_radial(potentialArgs->spline1d[base + 0],
                                            potentialArgs->acc1d[base + 0],
                                            r, l, d->rmin, d->rmax);
            double dradial_cos = eval_radial_deriv(potentialArgs->spline1d[base + 0],
                                                   potentialArgs->acc1d[base + 0],
                                                   r, l, d->rmin, d->rmax);
            double d2radial_cos = eval_radial_deriv2(potentialArgs->spline1d[base + 0],
                                                     potentialArgs->acc1d[base + 0],
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
                double radial_sin = eval_radial(potentialArgs->spline1d[base + 2],
                                                potentialArgs->acc1d[base + 2],
                                                r, l, d->rmin, d->rmax);
                double dradial_sin = eval_radial_deriv(potentialArgs->spline1d[base + 2],
                                                       potentialArgs->acc1d[base + 2],
                                                       r, l, d->rmin, d->rmax);
                double d2radial_sin = eval_radial_deriv2(potentialArgs->spline1d[base + 2],
                                                         potentialArgs->acc1d[base + 2],
                                                         r, l, d->rmin, d->rmax);
                d2Phi_dr2 += P[pi] * sin_mphi * d2radial_sin;
                d2Phi_dphi2 += P[pi] * (-m * m * sin_mphi) * radial_sin;
                d2Phi_drdphi += P[pi] * (m * cos_mphi) * dradial_sin;
            }
        }
    }

    free(P);

    F[0] = d2Phi_dr2;
    F[1] = d2Phi_dphi2;
    F[2] = d2Phi_drdphi;

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
