// Self-Consistent Field (SCF) potential implementation
// Based on Hernquist & Ostriker (1992) basis-function expansion:
//   Phi(r,theta,phi) = sum_{nlm} A_nlm * phiTilde_nl(r) * P_l^m(cos theta) * [cos/sin](m*phi)
//   rho(r,theta,phi) = sum_{nlm} A_nlm * rhoTilde_nl(r) * P_l^m(cos theta) * [cos/sin](m*phi)
// The radial basis functions (phiTilde, rhoTilde) are Hernquist-Ostriker specific.
// The angular basis functions (Legendre polynomials, cos/sin) are general spherical harmonics.
#include <math.h>
#include <stdlib.h>
#include <galpy_potentials.h>
#include <gsl/gsl_sf_gegenbauer.h>
#include <gsl/gsl_sf_legendre.h>

#ifndef GSL_MAJOR_VERSION
#define GSL_MAJOR_VERSION 1
#endif
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Cache tags for force/derivative caching
#define CACHE_FORCE 1
#define CACHE_DERIV 2

// ============================================================================
// Pre-computed SCF data: parsed parameters (immutable after init).
// Initialized once via initSCFPotentialArgs, reused on every evaluation.
// Workspace arrays are allocated per-call for thread safety, since some
// callers (e.g., actionAngle) share potentialArgs across OpenMP threads.
// ============================================================================

struct scf_data {
    // Parsed parameters (pointers into potentialArgs->args)
    double a;
    int isNonAxi, N, L, M;
    double *Acos, *Asin;
    double *cached_type, *cached_coords, *cached_values;
    // Derived constants
    int M_eff, Psize;
};

static void freeSCFData(void *data)
{
    free(data);
}

void initSCFPotentialArgs(struct potentialArg *potentialArgs)
{
    struct scf_data *d = (struct scf_data *)malloc(sizeof(struct scf_data));

    // Parse parameters from args
    double *args = potentialArgs->args;
    d->a = *args++;
    d->isNonAxi = (int)*args++;
    d->N = (int)*args++;
    d->L = (int)*args++;
    d->M = (int)*args++;
    d->Acos = args;
    d->Asin = d->isNonAxi ? args + d->N * d->L * d->M : NULL;
    double *cache = args + (d->isNonAxi + 1) * d->N * d->L * d->M;
    d->cached_type = cache;
    d->cached_coords = cache + 1;
    d->cached_values = cache + 4;

    // Derived constants
    d->M_eff = d->isNonAxi ? d->M : 1;
    d->Psize = d->isNonAxi ? d->L * (d->L + 1) / 2 : d->L;

    // Store in potentialArgs
    potentialArgs->pot_data = d;
    potentialArgs->free_pot_data = &freeSCFData;
}

// ============================================================================
// Utility functions
// ============================================================================

static inline void cyl_to_spher(double R, double Z, double *r, double *theta)
{
    *r = sqrt(R * R + Z * Z);
    *theta = atan2(R, Z);
}

static inline double calculateXi(double r, double a)
{
    return (r - a) / (r + a);
}

// Index into the Legendre polynomial array for P_l^m(cos theta).
// The storage layout depends on the GSL version.
static inline int legendre_index(int l, int m, int L)
{
#if GSL_MAJOR_VERSION == 2
    // GSL 2: triangle layout P(0,0), P(1,0), P(1,1), P(2,0), ...
    return l * (l + 1) / 2 + m;
#else
    // GSL 1: block-per-m layout from gsl_sf_legendre_Plm_array
    return m * L - m * (m - 1) / 2 + (l - m);
#endif
}

// ============================================================================
// Gegenbauer polynomials (radial basis, Hernquist-Ostriker specific)
// ============================================================================

// C_n^{2l+3/2}(xi) for 0 <= n < N, 0 <= l < L
static void compute_C(double xi, int N, int L, double *C)
{
    for (int l = 0; l < L; l++)
        gsl_sf_gegenpoly_array(N - 1, 1.5 + 2 * l, xi, C + l * N);
}

// dC_n^{2l+3/2}/dxi for 0 <= n < N, 0 <= l < L
// Uses: dC_n^a/dx = 2a * C_{n-1}^{a+1}(x)
static void compute_dC(double xi, int N, int L, double *dC)
{
    for (int l = 0; l < L; l++) {
        dC[l * N] = 0.0;
        if (N > 1)
            gsl_sf_gegenpoly_array(N - 2, 2.5 + 2 * l, xi, dC + l * N + 1);
        for (int n = 0; n < N; n++)
            dC[l * N + n] *= 2.0 * (2 * l + 1.5);
    }
}

// d2C_n^{2l+3/2}/dxi2 for 0 <= n < N, 0 <= l < L
// Uses: d2C_n^a/dx2 = 4a(a+1) * C_{n-2}^{a+2}(x)
static void compute_d2C(double xi, int N, int L, double *d2C)
{
    for (int l = 0; l < L; l++) {
        d2C[l * N] = 0.0;
        if (N > 1)
            d2C[l * N + 1] = 0.0;
        if (N > 2)
            gsl_sf_gegenpoly_array(N - 3, 3.5 + 2 * l, xi, d2C + l * N + 2);
        for (int n = 0; n < N; n++)
            d2C[l * N + n] *= 4.0 * (2 * l + 1.5) * (2 * l + 2.5);
    }
}

// ============================================================================
// Radial basis functions (Hernquist-Ostriker specific)
// Storage: radial[l * N + n] for degree l, order n
// ============================================================================

// rhoTilde_nl(r) = K_nl / sqrt(pi) * (ar)^l / ((r/a)(a+r)^{2l+3}) * C_n^{2l+3/2}(xi)
static void compute_rhoTilde(double r, double a, int N, int L,
                             double *C, double *rhoTilde)
{
    double rterms = a * pow(r + a, -3.0) / r;
    for (int l = 0; l < L; l++) {
        if (l > 0)
            rterms *= r * a / ((a + r) * (a + r));
        for (int n = 0; n < N; n++) {
            double K = 0.5 * n * (n + 4.0 * l + 3.0) + (l + 1.0) * (2.0 * l + 1.0);
            rhoTilde[l * N + n] = K * rterms * C[l * N + n];
        }
    }
}

// phiTilde_nl(r) = -(ar)^l / (a+r)^{2l+1} * C_n^{2l+3/2}(xi)
static void compute_phiTilde(double r, double a, int N, int L,
                             double *C, double *phiTilde)
{
    double rterms = -1.0 / (r + a);
    for (int l = 0; l < L; l++) {
        if (l > 0)
            rterms *= r * a / ((a + r) * (a + r));
        for (int n = 0; n < N; n++)
            phiTilde[l * N + n] = rterms * C[l * N + n];
    }
}

// d(phiTilde_nl)/dr
static void compute_dphiTilde(double r, double a, int N, int L,
                              double *C, double *dC, double *dphiTilde)
{
    double ar = a + r;
    double rterm = 1.0 / (r * ar * ar * ar);
    for (int l = 0; l < L; l++) {
        if (l > 0)
            rterm *= a * r / (ar * ar);
        for (int n = 0; n < N; n++) {
            int i = l * N + n;
            dphiTilde[i] = rterm * (((2 * l + 1) * r * ar - l * ar * ar) * C[i]
                                    - 2.0 * a * r * dC[i]);
        }
    }
}

// d2(phiTilde_nl)/dr2
static void compute_d2phiTilde(double r, double a, int N, int L,
                               double *C, double *dC, double *d2C,
                               double *d2phiTilde)
{
    double ar = a + r;
    double rterm = 1.0 / (r * r * ar * ar * ar * ar * ar);
    for (int l = 0; l < L; l++) {
        if (l > 0)
            rterm *= a * r / (ar * ar);
        for (int n = 0; n < N; n++) {
            int i = l * N + n;
            double ar2 = ar * ar;
            double ar3 = ar2 * ar;
            double ar4 = ar3 * ar;
            d2phiTilde[i] = rterm * (
                C[i] * (l * (1 - l) * ar4
                        - (4.0 * l * l + 6.0 * l + 2.0) * r * r * ar2
                        + l * (4 * l + 2) * r * ar3)
                + a * r * ((4.0 * r * r + 4.0 * a * r
                            + (8 * l + 4) * r * ar
                            - 4 * l * ar2) * dC[i]
                           - 4.0 * a * r * d2C[i]));
        }
    }
}

// ============================================================================
// Angular functions (associated Legendre polynomials - reusable)
// ============================================================================

// Compute P_l^m(x) for 0 <= l < L, 0 <= m < M
static void compute_legendre(double x, int L, int M, double *P)
{
    if (M == 1) {
        gsl_sf_legendre_Pl_array(L - 1, x, P);
    } else {
#if GSL_MAJOR_VERSION == 2
        gsl_sf_legendre_array_e(GSL_SF_LEGENDRE_NONE, L - 1, x, -1, P);
#else
        double *ptr = P;
        for (int m = 0; m < M; m++) {
            gsl_sf_legendre_Plm_array(L - 1, m, x, ptr);
            ptr += L - m;
        }
#endif
    }
}

// Compute P_l^m(x) and dP_l^m/dx for 0 <= l < L, 0 <= m < M
static void compute_legendre_deriv(double x, int L, int M,
                                   double *P, double *dP)
{
    if (M == 1) {
        gsl_sf_legendre_Pl_deriv_array(L - 1, x, P, dP);
    } else {
#if GSL_MAJOR_VERSION == 2
        gsl_sf_legendre_deriv_array_e(GSL_SF_LEGENDRE_NONE, L - 1, x, -1, P, dP);
#else
        double *pptr = P;
        double *dpptr = dP;
        for (int m = 0; m < M; m++) {
            gsl_sf_legendre_Plm_deriv_array(L - 1, m, x, pptr, dpptr);
            pptr += L - m;
            dpptr += L - m;
        }
#endif
    }
}

// ============================================================================
// Summation functions: combine radial * angular * coefficients
// These perform sum_{nlm} A_nlm * radial_nl * angular_lm
// ============================================================================

// Sum the basis-function expansion (used for both potential and density).
// radial[l*N+n] contains either phiTilde_nl or rhoTilde_nl.
// P[...] contains P_l^m(cos theta) in GSL layout.
static double sum_expansion(int N, int L, int M, int isNonAxi,
                            double *Acos, double *Asin,
                            double *radial, double *P, double phi)
{
    double result = 0.0;
    if (isNonAxi) { // LCOV_EXCL_START
        for (int l = 0; l < L; l++) {
            for (int m = 0; m <= l; m++) {
                double mcos = cos(m * phi);
                double msin = sin(m * phi);
                int pi = legendre_index(l, m, L);
                for (int n = 0; n < N; n++) {
                    int ci = n * L * M + l * M + m;
                    result += (Acos[ci] * mcos + Asin[ci] * msin)
                              * P[pi] * radial[l * N + n];
                }
            }
        }
    } else { // LCOV_EXCL_STOP
        for (int l = 0; l < L; l++) {
            for (int n = 0; n < N; n++) {
                result += Acos[n * L * M + l * M]
                          * P[l] * radial[l * N + n];
            }
        }
    }
    return result * sqrt(4.0 * M_PI);
}

// Sum the spherical force components: dPhi/dr, dPhi/dtheta, dPhi/dphi.
// F[0] = dPhi/dr, F[1] = dPhi/dtheta, F[2] = dPhi/dphi.
static void sum_spher_forces(int N, int L, int M, int isNonAxi,
                             double *Acos, double *Asin,
                             double *phiTilde, double *dphiTilde,
                             double *P, double *dP,
                             double phi, double sintheta, double *F)
{
    F[0] = F[1] = F[2] = 0.0;
    if (isNonAxi) {
        for (int l = 0; l < L; l++) {
            for (int m = 0; m <= l; m++) {
                double mcos = cos(m * phi);
                double msin = sin(m * phi);
                int pi = legendre_index(l, m, L);
                for (int n = 0; n < N; n++) {
                    int ci = n * L * M + l * M + m;
                    int ri = l * N + n;
                    double cos_sum = Acos[ci] * mcos + Asin[ci] * msin;
                    double sin_diff = Acos[ci] * msin - Asin[ci] * mcos;
                    F[0] -= cos_sum * P[pi] * dphiTilde[ri];
                    F[1] -= cos_sum * dP[pi] * phiTilde[ri];
                    F[2] += m * sin_diff * P[pi] * phiTilde[ri];
                }
            }
        }
    } else {
        for (int l = 0; l < L; l++) {
            for (int n = 0; n < N; n++) {
                int ci = n * L * M + l * M;
                int ri = l * N + n;
                F[0] -= Acos[ci] * P[l] * dphiTilde[ri];
                F[1] -= Acos[ci] * dP[l] * phiTilde[ri];
            }
        }
    }
    double sqrt4pi = sqrt(4.0 * M_PI);
    F[0] *= sqrt4pi;
    F[1] *= sqrt4pi * (-sintheta);
    F[2] *= sqrt4pi;
}

// Sum the spherical 2nd-derivative components.
// F[0] = d2Phi/dr2, F[1] = d2Phi/dphi2, F[2] = d2Phi/drdphi.
static void sum_spher_2nd_derivs(int N, int L, int M, int isNonAxi,
                                 double *Acos, double *Asin,
                                 double *phiTilde, double *dphiTilde,
                                 double *d2phiTilde, double *P,
                                 double phi, double *F)
{
    F[0] = F[1] = F[2] = 0.0;
    if (isNonAxi) {
        for (int l = 0; l < L; l++) {
            for (int m = 0; m <= l; m++) {
                double mcos = cos(m * phi);
                double msin = sin(m * phi);
                int pi = legendre_index(l, m, L);
                for (int n = 0; n < N; n++) {
                    int ci = n * L * M + l * M + m;
                    int ri = l * N + n;
                    double cos_sum = Acos[ci] * mcos + Asin[ci] * msin;
                    double sin_diff = Acos[ci] * msin - Asin[ci] * mcos;
                    F[0] -= cos_sum * P[pi] * d2phiTilde[ri];
                    F[1] += m * m * cos_sum * P[pi] * phiTilde[ri];
                    F[2] += m * sin_diff * P[pi] * dphiTilde[ri];
                }
            }
        }
    } else {
        for (int l = 0; l < L; l++) {
            for (int n = 0; n < N; n++) {
                int ci = n * L * M + l * M;
                F[0] -= Acos[ci] * P[l] * d2phiTilde[l * N + n];
            }
        }
    }
    double sqrt4pi = sqrt(4.0 * M_PI);
    F[0] *= sqrt4pi;
    F[1] *= sqrt4pi;
    F[2] *= sqrt4pi;
}

// ============================================================================
// Internal computation: spherical forces and 2nd derivatives with caching
// ============================================================================

// Compute spherical force components (dPhi/dr, dPhi/dtheta, dPhi/dphi)
// with caching to avoid redundant work when Rforce/zforce are called
// at the same point.
static void compute_spher_forces(struct scf_data *d,
                                 double R, double Z, double phi, double *F)
{
    // Check cache
    if ((int)*d->cached_type == CACHE_FORCE
        && d->cached_coords[0] == R
        && d->cached_coords[1] == Z
        && d->cached_coords[2] == phi) {
        F[0] = d->cached_values[0];
        F[1] = d->cached_values[1];
        F[2] = d->cached_values[2];
        return;
    }

    double r, theta;
    cyl_to_spher(R, Z, &r, &theta);
    double xi = calculateXi(r, d->a);

    int NL = d->N * d->L;
    // Allocate workspace per call (thread-safe)
    double *ws = (double *)malloc((4 * NL + 2 * d->Psize) * sizeof(double));
    double *C       = ws;
    double *dCArr    = ws + NL;
    double *phiT    = ws + 2 * NL;
    double *dphiT   = ws + 3 * NL;
    double *P       = ws + 4 * NL;
    double *dP      = ws + 4 * NL + d->Psize;

    // Radial part
    compute_C(xi, d->N, d->L, C);
    compute_dC(xi, d->N, d->L, dCArr);
    compute_phiTilde(r, d->a, d->N, d->L, C, phiT);
    compute_dphiTilde(r, d->a, d->N, d->L, C, dCArr, dphiT);

    // Angular part
    compute_legendre_deriv(cos(theta), d->L, d->M_eff, P, dP);

    // Sum
    sum_spher_forces(d->N, d->L, d->M, d->isNonAxi,
                     d->Acos, d->Asin,
                     phiT, dphiT, P, dP,
                     phi, sin(theta), F);

    free(ws);

    // Update cache
    *d->cached_type = (double)CACHE_FORCE;
    d->cached_coords[0] = R;
    d->cached_coords[1] = Z;
    d->cached_coords[2] = phi;
    d->cached_values[0] = F[0];
    d->cached_values[1] = F[1];
    d->cached_values[2] = F[2];
}

// Compute spherical 2nd-derivative components (d2Phi/dr2, d2Phi/dphi2, d2Phi/drdphi)
// with caching.
static void compute_spher_2nd_derivs(struct scf_data *d,
                                     double R, double Z, double phi, double *F)
{
    // Check cache
    if ((int)*d->cached_type == CACHE_DERIV
        && d->cached_coords[0] == R
        && d->cached_coords[1] == Z
        && d->cached_coords[2] == phi) {
        F[0] = d->cached_values[0];
        F[1] = d->cached_values[1];
        F[2] = d->cached_values[2];
        return;
    }

    double r, theta;
    cyl_to_spher(R, Z, &r, &theta);
    double xi = calculateXi(r, d->a);

    int NL = d->N * d->L;
    // Allocate workspace per call (thread-safe)
    double *ws = (double *)malloc((6 * NL + d->Psize) * sizeof(double));
    double *C       = ws;
    double *dCArr    = ws + NL;
    double *d2CArr   = ws + 2 * NL;
    double *phiT    = ws + 3 * NL;
    double *dphiT   = ws + 4 * NL;
    double *d2phiT  = ws + 5 * NL;
    double *P       = ws + 6 * NL;

    // Radial part
    compute_C(xi, d->N, d->L, C);
    compute_dC(xi, d->N, d->L, dCArr);
    compute_d2C(xi, d->N, d->L, d2CArr);
    compute_phiTilde(r, d->a, d->N, d->L, C, phiT);
    compute_dphiTilde(r, d->a, d->N, d->L, C, dCArr, dphiT);
    compute_d2phiTilde(r, d->a, d->N, d->L, C, dCArr, d2CArr, d2phiT);

    // Angular part (no derivatives needed for 2nd derivs)
    compute_legendre(cos(theta), d->L, d->M_eff, P);

    // Sum
    sum_spher_2nd_derivs(d->N, d->L, d->M, d->isNonAxi,
                         d->Acos, d->Asin,
                         phiT, dphiT, d2phiT, P,
                         phi, F);

    free(ws);

    // Update cache
    *d->cached_type = (double)CACHE_DERIV;
    d->cached_coords[0] = R;
    d->cached_coords[1] = Z;
    d->cached_coords[2] = phi;
    d->cached_values[0] = F[0];
    d->cached_values[1] = F[1];
    d->cached_values[2] = F[2];
}

// ============================================================================
// Public API: potential, forces, derivatives, density
// ============================================================================

double SCFPotentialEval(double R, double Z, double phi, double t,
                        struct potentialArg *potentialArgs)
{
    struct scf_data *d = (struct scf_data *)potentialArgs->pot_data;

    double r, theta;
    cyl_to_spher(R, Z, &r, &theta);
    double xi = calculateXi(r, d->a);

    int NL = d->N * d->L;
    double *ws = (double *)malloc((2 * NL + d->Psize) * sizeof(double));
    double *C      = ws;
    double *radial = ws + NL;
    double *P      = ws + 2 * NL;

    // Radial part
    compute_C(xi, d->N, d->L, C);
    compute_phiTilde(r, d->a, d->N, d->L, C, radial);

    // Angular part
    compute_legendre(cos(theta), d->L, d->M_eff, P);

    // Sum
    double result = sum_expansion(d->N, d->L, d->M, d->isNonAxi,
                                  d->Acos, d->Asin,
                                  radial, P, phi);
    free(ws);
    return result;
}

double SCFPotentialRforce(double R, double Z, double phi, double t,
                          struct potentialArg *potentialArgs)
{
    double r, theta;
    cyl_to_spher(R, Z, &r, &theta);
    double F[3];
    compute_spher_forces((struct scf_data *)potentialArgs->pot_data,
                         R, Z, phi, F);
    return F[0] * (R / r) + F[1] * (Z / (r * r));
}

double SCFPotentialzforce(double R, double Z, double phi, double t,
                          struct potentialArg *potentialArgs)
{
    double r, theta;
    cyl_to_spher(R, Z, &r, &theta);
    double F[3];
    compute_spher_forces((struct scf_data *)potentialArgs->pot_data,
                         R, Z, phi, F);
    return F[0] * (Z / r) + F[1] * (-R / (r * r));
}

double SCFPotentialphitorque(double R, double Z, double phi, double t,
                             struct potentialArg *potentialArgs)
{
    double F[3];
    compute_spher_forces((struct scf_data *)potentialArgs->pot_data,
                         R, Z, phi, F);
    return F[2];
}

double SCFPotentialPlanarRforce(double R, double phi, double t,
                                struct potentialArg *potentialArgs)
{
    return SCFPotentialRforce(R, 0.0, phi, t, potentialArgs);
}

double SCFPotentialPlanarphitorque(double R, double phi, double t,
                                   struct potentialArg *potentialArgs)
{
    return SCFPotentialphitorque(R, 0.0, phi, t, potentialArgs);
}

double SCFPotentialPlanarR2deriv(double R, double phi, double t,
                                 struct potentialArg *potentialArgs)
{
    double F[3];
    compute_spher_2nd_derivs((struct scf_data *)potentialArgs->pot_data,
                             R, 0.0, phi, F);
    return F[0];
}

double SCFPotentialPlanarphi2deriv(double R, double phi, double t,
                                   struct potentialArg *potentialArgs)
{
    double F[3];
    compute_spher_2nd_derivs((struct scf_data *)potentialArgs->pot_data,
                             R, 0.0, phi, F);
    return F[1];
}

double SCFPotentialPlanarRphideriv(double R, double phi, double t,
                                   struct potentialArg *potentialArgs)
{
    double F[3];
    compute_spher_2nd_derivs((struct scf_data *)potentialArgs->pot_data,
                             R, 0.0, phi, F);
    return F[2];
}

double SCFPotentialDens(double R, double Z, double phi, double t,
                        struct potentialArg *potentialArgs)
{
    struct scf_data *d = (struct scf_data *)potentialArgs->pot_data;

    double r, theta;
    cyl_to_spher(R, Z, &r, &theta);
    double xi = calculateXi(r, d->a);

    int NL = d->N * d->L;
    double *ws = (double *)malloc((2 * NL + d->Psize) * sizeof(double));
    double *C      = ws;
    double *radial = ws + NL;
    double *P      = ws + 2 * NL;

    // Radial part (rhoTilde instead of phiTilde)
    compute_C(xi, d->N, d->L, C);
    compute_rhoTilde(r, d->a, d->N, d->L, C, radial);

    // Angular part
    compute_legendre(cos(theta), d->L, d->M_eff, P);

    // Sum
    double result = sum_expansion(d->N, d->L, d->M, d->isNonAxi,
                                  d->Acos, d->Asin,
                                  radial, P, phi) / (2.0 * M_PI);
    free(ws);
    return result;
}
