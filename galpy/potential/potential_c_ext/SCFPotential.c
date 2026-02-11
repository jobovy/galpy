#include <math.h>
#include <galpy_potentials.h>
#include <stdio.h>
#include <gsl/gsl_sf_gegenbauer.h>
#include <gsl/gsl_sf_legendre.h>

#ifndef GSL_MAJOR_VERSION
#define GSL_MAJOR_VERSION 1
#endif
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/*
 * SCFPotential: Self-Consistent Field expansion potential
 * 
 * Implements the Hernquist & Ostriker (1992) basis-function expansion method.
 * The potential is expanded in:
 *   - Radial basis: Gegenbauer polynomials (C_n^alpha)
 *   - Angular basis: Associated Legendre polynomials (P_l^m)
 *   - Azimuthal basis: cos(m*phi) and sin(m*phi)
 * 
 * Arguments: amp, Acos, Asin, a (scale length)
 */

// Cache types for coordinate evaluations
const int FORCE = 1;
const int DERIV = 2;

/*=============================================================================
 * COORDINATE TRANSFORMATION UTILITIES
 *===========================================================================*/

/**
 * Convert cylindrical (R, Z) to spherical (r, theta) coordinates
 * 
 * @param R Cylindrical radius
 * @param Z Height above plane
 * @param r Output: spherical radius sqrt(R^2 + Z^2)
 * @param theta Output: polar angle atan2(R, Z)
 */
static inline void cyl_to_spher(double R, double Z, double *r, double *theta)
{
    *r = sqrt(R*R + Z*Z);
    *theta = atan2(R, Z);
}

/**
 * Calculate the coordinate transformation xi = (r - a) / (r + a)
 * Used in the Hernquist-Ostriker basis functions
 * 
 * @param r Spherical radius
 * @param a Scale length
 * @param xi Output: transformed coordinate in [-1, 1]
 */
static inline void calculateXi(double r, double a, double *xi)
{
    *xi = (r - a) / (r + a);
}

/**
 * Compute integer power x^i efficiently using iterative multiplication
 * For small integer powers, this is faster than pow()
 * 
 * @param x Base value
 * @param i Integer exponent (must be >= 0)
 * @return x^i
 */
static inline double power(double x, int i)
{
    double result = 1.0;
    while (i > 0) {
        if (i & 1) result *= x;  // If i is odd, multiply by x
        x *= x;                   // Square x
        i >>= 1;                  // Divide i by 2
    }
    return result;
}


/*=============================================================================
 * SPHERICAL EXPANSION TERM EVALUATION FUNCTIONS
 * 
 * These functions evaluate individual terms in the spherical harmonic expansion.
 * Each function combines:
 *   - Expansion coefficients (Acos, Asin)
 *   - Azimuthal factors (cos(m*phi), sin(m*phi))
 *   - Angular factors (Legendre polynomials P and their derivatives)
 *   - Radial factors (phiTilde and its derivatives)
 * 
 * Functions come in pairs:
 *   - Non-axisymmetric version (computeXXX): full 3D case with both cos and sin terms
 *   - Axisymmetric version (computeAxiXXX): simplified case with only m=0 terms
 *===========================================================================*/

/**
 * Potential term (non-axisymmetric)
 * Evaluates: (Acos*cos(m*phi) + Asin*sin(m*phi)) * P_l^m * phiTilde
 */
// LCOV_EXCL_START
double computePhi(double Acos_val, double Asin_val, double mCos, double mSin, 
                  double P, double phiTilde, int m)
{
    return (Acos_val*mCos + Asin_val*mSin)*P*phiTilde;
}
// LCOV_EXCL_STOP

/**
 * Potential term (axisymmetric, m=0 only)
 */
double computeAxiPhi(double Acos_val, double P, double phiTilde)
{
    return Acos_val*P*phiTilde;
}

/**
 * Radial force component: F_r = -dPhi/dr
 * Uses derivative of radial function (dphiTilde)
 */
double computeF_r(double Acos_val, double Asin_val, double mCos, double mSin, 
                  double P, double dphiTilde, int m)
{
    return -(Acos_val*mCos + Asin_val*mSin)*P*dphiTilde;
}

double computeAxiF_r(double Acos_val, double P, double dphiTilde)
{
    return -Acos_val*P*dphiTilde;
}

/**
 * Theta force component: F_theta = -dPhi/dtheta
 * Uses derivative of angular function (dP)
 */
double computeF_theta(double Acos_val, double Asin_val, double mCos, double mSin, 
                      double dP, double phiTilde, int m)
{
    return -(Acos_val*mCos + Asin_val*mSin)*dP*phiTilde;
}

double computeAxiF_theta(double Acos_val, double dP, double phiTilde)
{
    return -Acos_val*dP*phiTilde;
}

/**
 * Phi force component: F_phi = -dPhi/dphi
 * Note the coefficient swap (Acos*sin - Asin*cos) from chain rule
 */
double computeF_phi(double Acos_val, double Asin_val, double mCos, double mSin, 
                    double P, double phiTilde, int m)
{
    return m*(Acos_val*mSin - Asin_val*mCos)*P*phiTilde;
}

double computeAxiF_phi(double Acos_val, double P, double phiTilde)
{
    return 0.;  // No phi dependence in axisymmetric case
}

/**
 * Second derivative d^2Phi/dr^2
 */
double computeF_rr(double Acos_val, double Asin_val, double mCos, double mSin, 
                   double P, double d2phiTilde, int m)
{
    return -(Acos_val*mCos + Asin_val*mSin)*P*d2phiTilde;
}

double computeAxiF_rr(double Acos_val, double P, double d2phiTilde)
{
    return -Acos_val*P*d2phiTilde;
}

/**
 * Mixed derivative d^2Phi/dr/dphi
 */
double computeF_rphi(double Acos_val, double Asin_val, double mCos, double mSin, 
                     double P, double dphiTilde, int m)
{
    return m*(Acos_val*mSin - Asin_val*mCos)*P*dphiTilde;
}

double computeAxiF_rphi(double Acos_val, double P, double dphiTilde)
{
    return 0.;  // No phi dependence in axisymmetric case
}

/**
 * Second derivative d^2Phi/dphi^2
 */
double computeF_phiphi(double Acos_val, double Asin_val, double mCos, double mSin, 
                       double P, double phiTilde, int m)
{
    return m*m*(Acos_val*mCos + Asin_val*mSin)*P*phiTilde;
}

double computeAxiF_phiphi(double Acos_val, double P, double phiTilde)
{
    return 0.;  // No phi dependence in axisymmetric case
}

/*=============================================================================
 * GEGENBAUER POLYNOMIAL EVALUATION (RADIAL BASIS FUNCTIONS)
 * 
 * Gegenbauer (ultraspherical) polynomials C_n^(alpha)(xi) form the radial basis.
 * For SCF expansion: alpha = 3/2 + 2*l for each angular momentum l
 * 
 * These functions use GSL's gsl_sf_gegenpoly_array for efficient array evaluation.
 * Derivatives are computed via recursion relations:
 *   dC_n^(alpha)/dxi = 2*alpha * C_(n-1)^(alpha+1)(xi)
 *   d^2C_n^(alpha)/dxi^2 = 4*alpha*(alpha+1) * C_(n-2)^(alpha+2)(xi)
 *===========================================================================*/

/**
 * Compute Gegenbauer polynomials C_n^(3/2+2l)(xi) for all n, l
 * 
 * @param xi Transformed radial coordinate in [-1, 1]
 * @param N Number of radial terms (0 <= n < N)
 * @param L Number of angular momentum terms (0 <= l < L)
 * @param C_array Output array of size N*L, indexed as C_array[l*N + n]
 */
void compute_C(double xi, int N, int L, double *C_array)
{
    for (int l = 0; l < L; l++) {
        double alpha = 3.0/2.0 + 2*l;
        gsl_sf_gegenpoly_array(N - 1, alpha, xi, C_array + l*N);
    }
}

/**
 * Compute first derivative of Gegenbauer polynomials: dC_n/dxi
 * 
 * Uses recursion: dC_n^(alpha)/dxi = 2*alpha * C_(n-1)^(alpha+1)
 * Note: dC_0/dxi = 0 (constant term has zero derivative)
 * 
 * @param xi Transformed radial coordinate
 * @param N Number of radial terms
 * @param L Number of angular momentum terms
 * @param dC_array Output array of size N*L
 */
void compute_dC(double xi, int N, int L, double *dC_array)
{
    for (int l = 0; l < L; l++) {
        double alpha = 3.0/2.0 + 2*l;
        
        // First element is always zero (derivative of constant)
        dC_array[l*N] = 0.0;
        
        // Compute shifted Gegenbauer polynomials for n >= 1
        if (N > 1) {
            gsl_sf_gegenpoly_array(N - 2, alpha + 1.0, xi, dC_array + l*N + 1);
            
            // Apply scaling factor from recursion relation
            double scale = 2.0 * alpha;
            for (int n = 1; n < N; n++) {
                dC_array[l*N + n] *= scale;
            }
        }
    }
}

/**
 * Compute second derivative of Gegenbauer polynomials: d^2C_n/dxi^2
 * 
 * Uses recursion: d^2C_n^(alpha)/dxi^2 = 4*alpha*(alpha+1) * C_(n-2)^(alpha+2)
 * Note: d^2C_0/dxi^2 = d^2C_1/dxi^2 = 0
 * 
 * @param xi Transformed radial coordinate
 * @param N Number of radial terms
 * @param L Number of angular momentum terms
 * @param d2C_array Output array of size N*L
 */
void compute_d2C(double xi, int N, int L, double *d2C_array)
{
    for (int l = 0; l < L; l++) {
        double alpha = 3.0/2.0 + 2*l;
        
        // First two elements are always zero
        d2C_array[l*N] = 0.0;
        if (N > 1) {
            d2C_array[l*N + 1] = 0.0;
        }
        
        // Compute double-shifted Gegenbauer polynomials for n >= 2
        if (N > 2) {
            gsl_sf_gegenpoly_array(N - 3, alpha + 2.0, xi, d2C_array + l*N + 2);
            
            // Apply scaling factor from recursion relation
            double scale = 4.0 * alpha * (alpha + 1.0);
            for (int n = 2; n < N; n++) {
                d2C_array[l*N + n] *= scale;
            }
        }
    }
}

/*=============================================================================
 * RADIAL BASIS FUNCTIONS (HERNQUIST-OSTRIKER BASIS)
 * 
 * These functions combine Gegenbauer polynomials with radial scaling factors
 * to create the Hernquist-Ostriker basis functions used in SCF expansion.
 * 
 * The basis has the form:
 *   phiTilde_nl(r) = -(ar)^l / (a+r)^(2l+1) * C_n^(3/2+2l)(xi)
 *   rhoTilde_nl(r) = K_nl * (ar)^l / [r(a+r)^(2l+3)] * C_n^(3/2+2l)(xi)
 * 
 * where K_nl = 0.5*n*(n+4l+3) + (l+1)*(2l+1) is a normalization constant
 *===========================================================================*/

/**
 * Compute density basis functions rhoTilde_nl(r)
 * 
 * These are the building blocks for density evaluation in the SCF expansion.
 * Note: Input C array should already be computed at the appropriate xi.
 * 
 * @param r Spherical radius
 * @param a Scale length
 * @param N Number of radial terms
 * @param L Number of angular momentum terms
 * @param C Input: Gegenbauer polynomials C_n^(3/2+2l)(xi)
 * @param rhoTilde Output: density basis functions, size N*L
 */
void compute_rhoTilde(double r, double a, int N, int L, double *C, double *rhoTilde)
{
    // Initial radial scaling: a / [r * (r+a)^3]
    double rterms = a * pow(r + a, -3.0) / r;
    
    for (int l = 0; l < L; l++) {
        // Update radial scaling for l > 0: multiply by (ar) / (a+r)^2
        if (l != 0) {
            rterms *= r * a / ((a + r) * (a + r));
        }
        
        // Normalization constant K_nl
        for (int n = 0; n < N; n++) {
            double K_nl = 0.5 * n * (n + 4.0 * l + 3.0) + (l + 1.0) * (2.0 * l + 1.0);
            rhoTilde[l*N + n] = K_nl * rterms * C[n + l*N];
        }
    }
}

/**
 * Compute potential basis functions phiTilde_nl(r)
 * 
 * These are the radial parts of the potential expansion.
 * The form is: phiTilde = -(ar)^l / (a+r)^(2l+1) * C_n^(alpha)(xi)
 * 
 * @param r Spherical radius
 * @param a Scale length
 * @param N Number of radial terms
 * @param L Number of angular momentum terms
 * @param C Input: Gegenbauer polynomials
 * @param phiTilde Output: potential basis functions, size N*L
 */
void compute_phiTilde(double r, double a, int N, int L, double *C, double *phiTilde)
{
    // Initial radial scaling: -1 / (r+a)
    double rterms = -1.0 / (r + a);
    
    for (int l = 0; l < L; l++) {
        // Update radial scaling for l > 0: multiply by (ar) / (a+r)^2
        if (l != 0) {
            rterms *= (r * a) / ((a + r) * (a + r));
        }
        
        // Multiply Gegenbauer polynomials by radial factor
        for (int n = 0; n < N; n++) {
            phiTilde[l*N + n] = rterms * C[n + l*N];
        }
    }
}

/**
 * Compute first derivative of phiTilde with respect to r
 * 
 * Uses chain rule: d(phiTilde)/dr = d(rterms)/dr * C + rterms * d(C)/dxi * dxi/dr
 * where dxi/dr = 2a / (r+a)^2
 * 
 * @param r Spherical radius
 * @param a Scale length
 * @param N Number of radial terms
 * @param L Number of angular momentum terms
 * @param C Input: Gegenbauer polynomials
 * @param dC Input: derivatives of Gegenbauer polynomials w.r.t. xi
 * @param dphiTilde Output: derivative of potential basis, size N*L
 */
void compute_dphiTilde(double r, double a, int N, int L, double *C, double *dC, double *dphiTilde)
{
    // Base radial term: 1 / [r * (r+a)^3]
    double rterm = 1.0 / (r * power(a + r, 3));
    
    for (int l = 0; l < L; l++) {
        // Update radial scaling for l > 0
        if (l != 0) {
            rterm *= (a * r) / power(a + r, 2);
        }
        
        for (int n = 0; n < N; n++) {
            // Chain rule application
            double C_val = C[l*N + n];
            double dC_val = dC[l*N + n];
            
            // d(phiTilde)/dr = rterm * [angular_deriv_term * C - xi_deriv_term * dC]
            double angular_term = (2*l + 1) * r * (a + r) - l * power(a + r, 2);
            double xi_deriv_term = 2.0 * a * r;  // dxi/dr * (r+a)^2 factor
            
            dphiTilde[l*N + n] = rterm * (angular_term * C_val - xi_deriv_term * dC_val);
        }
    }
}

/**
 * Compute second derivative of phiTilde with respect to r
 * 
 * This requires both first and second derivatives of Gegenbauer polynomials.
 * 
 * @param r Spherical radius
 * @param a Scale length
 * @param N Number of radial terms
 * @param L Number of angular momentum terms
 * @param C Input: Gegenbauer polynomials
 * @param dC Input: first derivatives w.r.t. xi
 * @param d2C Input: second derivatives w.r.t. xi
 * @param d2phiTilde Output: second derivative of potential basis, size N*L
 */
void compute_d2phiTilde(double r, double a, int N, int L, double *C, double *dC, 
                        double *d2C, double *d2phiTilde)
{
    // Base radial term: 1 / [r^2 * (r+a)^5]
    double rterm = 1.0 / (r * r * power(a + r, 5));
    
    for (int l = 0; l < L; l++) {
        // Update radial scaling for l > 0
        if (l != 0) {
            rterm *= (a * r) / power(a + r, 2);
        }
        
        for (int n = 0; n < N; n++) {
            double C_val = C[l*N + n];
            double dC_val = dC[l*N + n];
            double d2C_val = d2C[l*N + n];
            
            // Complex expression from double chain rule application
            // Terms involving C (from second derivative of radial part)
            double C_term = C_val * (
                l * (1 - l) * power(a + r, 4) 
                - (4*l*l + 6*l + 2.0) * r*r * power(a + r, 2) 
                + l * (4*l + 2) * r * power(a + r, 3)
            );
            
            // Terms involving dC and d2C (from chain rule with xi derivatives)
            double dC_term = a * r * (
                4*r*r + 4*a*r + (8*l + 4) * r * (a + r) 
                - 4*l * power(a + r, 2)
            ) * dC_val;
            
            double d2C_term = -4.0 * a * a * r * r * d2C_val;
            
            d2phiTilde[l*N + n] = rterm * (C_term + dC_term + d2C_term);
        }
    }
}


/*=============================================================================
 * ASSOCIATED LEGENDRE POLYNOMIAL EVALUATION (ANGULAR BASIS FUNCTIONS)
 * 
 * Associated Legendre polynomials P_l^m(cos(theta)) form the angular basis.
 * These functions wrap GSL's Legendre polynomial routines with version
 * compatibility handling for GSL 1.x and 2.x.
 * 
 * Storage format:
 *   - For axisymmetric case (M=1): Simple array P_l for l=0..L-1
 *   - For general case (M>1): Packed array with all (l,m) where 0 <= m <= l
 *     GSL 1.x: Grouped by m, then l (m=0: P_0..P_(L-1), m=1: P_1..P_(L-1), ...)
 *     GSL 2.x: Single call returns all (l,m) pairs in standard triangular order
 *===========================================================================*/

/**
 * Compute associated Legendre polynomials P_l^m(x)
 * 
 * @param x Argument (typically cos(theta))
 * @param L Maximum degree + 1 (compute for 0 <= l < L)
 * @param M Maximum order + 1 (compute for 0 <= m < M)
 * @param P_array Output array (size depends on L, M and GSL version)
 */
void compute_P(double x, int L, int M, double *P_array)
{
    if (M == 1) {
        // Axisymmetric case: only m=0 terms (standard Legendre polynomials)
        gsl_sf_legendre_Pl_array(L - 1, x, P_array);
    } else {
        // Non-axisymmetric case: full P_l^m array
        #if GSL_MAJOR_VERSION == 2
            // GSL 2.x: Single function returns all (l,m) at once
            gsl_sf_legendre_array_e(GSL_SF_LEGENDRE_NONE, L - 1, x, -1, P_array);
        #else
            // GSL 1.x: Loop over m, computing P_l^m for each
            for (int m = 0; m < M; m++) {
                gsl_sf_legendre_Plm_array(L - 1, m, x, P_array);
                P_array += L - m;  // Advance pointer for next m block
            }
        #endif
    }
}

/**
 * Compute associated Legendre polynomials P_l^m(x) and their derivatives dP/dx
 * 
 * @param x Argument (typically cos(theta))
 * @param L Maximum degree + 1
 * @param M Maximum order + 1
 * @param P_array Output: Legendre polynomials
 * @param dP_array Output: derivatives with respect to x
 */
void compute_P_dP(double x, int L, int M, double *P_array, double *dP_array)
{
    if (M == 1) {
        // Axisymmetric case
        gsl_sf_legendre_Pl_deriv_array(L - 1, x, P_array, dP_array);
    } else {
        // Non-axisymmetric case
        #if GSL_MAJOR_VERSION == 2
            // GSL 2.x: Single function returns both P and dP
            gsl_sf_legendre_deriv_array_e(GSL_SF_LEGENDRE_NONE, L - 1, x, -1, 
                                          P_array, dP_array);
        #else
            // GSL 1.x: Loop over m
            for (int m = 0; m < M; m++) {
                gsl_sf_legendre_Plm_deriv_array(L - 1, m, x, P_array, dP_array);
                P_array += L - m;
                dP_array += L - m;
            }
        #endif
    }
}





typedef struct equations equations;
struct equations
{
    double ((**Eq)(double, double, double, double, double, double, int));
    double *(*phiTilde);
    double *(*P);
    double *Constant;
};

typedef struct axi_equations axi_equations;
struct axi_equations
{
    double ((**Eq)(double, double, double));
    double *(*phiTilde);
    double *(*P);
    double *Constant;
};

//Compute axi symmetric potentials.
void compute(double a, int N, int L, int M,
	     double r, double theta, double phi,
	     double *Acos, int eq_size,
	     axi_equations e,
	     double *F)
{
    int i,n,l;
    for (i = 0; i < eq_size; i++)
    {
        *(F + i) =0; //Initialize each F
    }

    for (l = 0; l < L; l++)
    {

        for (n = 0; n < N; n++)
        {



            double Acos_val = *(Acos + M*l + M*L*n);
            for (i = 0; i < eq_size; i++)
            {
                double (*Eq)(double, double, double) = *(e.Eq + i);
                double *P = *(e.P + i);
                double *phiTilde = *(e.phiTilde + i);
                *(F + i) += (*Eq)(Acos_val, P[l], phiTilde[l*N + n]);
            }


        }
    }

    //Multiply F by constants
    for (i = 0; i < eq_size; i++)
    {
        double constant = *(e.Constant + i);
        *(F + i) *= constant*sqrt(4*M_PI);
    }

}

//Compute Non Axi symmetric Potentials
void computeNonAxi(double a, int N, int L, int M,
		   double r, double theta, double phi,
		   double *Acos, double *Asin, int eq_size,
		   equations e,
		   double *F)
{
    int i,n,l,m;
    for (i = 0; i < eq_size; i++)
    {
        *(F + i) =0; //Initialize each F
    }

    int v1counter=0;
    int v2counter = 0;
    int Pindex = 0;
    for (l = 0; l < L; l++)
    {
        v1counter = 0;
        for (m = 0; m<=l; m++)
        {
            #if GSL_MAJOR_VERSION == 2
                Pindex = v2counter;
            #else
                Pindex = v1counter + l;
            #endif

            double mCos = cos(m*phi);
            double mSin = sin(m*phi);
            for (n = 0; n < N; n++)
            {
                double Acos_val = *(Acos +m + M*l + M*L*n);
                double Asin_val = *(Asin +m + M*l + M*L*n);
                for (i = 0; i < eq_size; i++)
                {
                    double (*Eq)(double, double, double, double, double, double, int) = *(e.Eq + i);
                    double *P = *(e.P + i);
                    double *phiTilde = *(e.phiTilde + i);
                    *(F + i) += (*Eq)(Acos_val, Asin_val, mCos, mSin, P[Pindex], phiTilde[l*N + n], m);
                }


            }



                v2counter++;
                v1counter+=M-m - 1;

        }




    }
    //Multiply F by constants
    for (i = 0; i < eq_size; i++)
    {
        double constant = *(e.Constant + i);
        *(F + i) *= constant*sqrt(4*M_PI);
    }

}



//Compute the Forces
void computeForce(double R,double Z, double phi,
		  double t,
		  struct potentialArg * potentialArgs, double * F)
{
    double * args= potentialArgs->args;
    //Get args
    double a = *args++;
    int isNonAxi = (int)*args++;
    int N = (int)*args++;
    int L = (int)*args++;
    int M = (int)*args++;

    double* Acos = args;

    double* caching_i = (args + (isNonAxi + 1)*N*L*M);
    double *Asin;
    if (isNonAxi == 1)
    {
        Asin = args + N*L*M;
    }
    double *cached_type = caching_i;
    double * cached_coords = (caching_i+ 1);
    double * cached_values = (caching_i + 4);
    if ((int)*cached_type==FORCE)
    {
        if (*cached_coords == R && *(cached_coords + 1) == Z && *(cached_coords + 2) == phi)
        {
            *F = *cached_values;
            *(F + 1) = *(cached_values + 1);
            *(F + 2) = *(cached_values + 2);
            return;
        }
    }
    double r;
    double theta;
    cyl_to_spher(R, Z, &r, &theta);

    double xi;
    calculateXi(r, a, &xi);

//Compute the gegenbauer polynomials and its derivative.
    double *C= (double *) malloc ( N*L * sizeof(double) );
    double *dC= (double *) malloc ( N*L * sizeof(double) );
    double *phiTilde= (double *) malloc ( N*L * sizeof(double) );
    double *dphiTilde= (double *) malloc ( N*L * sizeof(double) );

    compute_C(xi, N, L, C);
    compute_dC(xi, N, L, dC);

//Compute phiTilde and its derivative
    compute_phiTilde(r, a, N, L, C, phiTilde);

    compute_dphiTilde(r, a, N, L, C, dC, dphiTilde);

//Compute Associated Legendre Polynomials
    int M_eff = M;
    int size = 0;

    if (isNonAxi==0)
    {
    M_eff = 1;
    size = L;
    } else{
    size = L*L - L*(L-1)/2;
    }

    double *P= (double *) malloc ( size * sizeof(double) );
    double *dP= (double *) malloc ( size * sizeof(double) );
    compute_P_dP(cos(theta), L, M_eff, P, dP);

    double (*PhiTilde_Pointer[3]) = {dphiTilde,phiTilde,phiTilde};
    double (*P_Pointer[3]) = {P, dP, P};

    double Constant[3] = {1., -sin(theta), 1.};

    if (isNonAxi == 1)
    {
        double (*Eq[3])(double, double, double, double, double, double, int) = {&computeF_r, &computeF_theta, &computeF_phi};
        equations e = {Eq,&PhiTilde_Pointer[0], &P_Pointer[0], &Constant[0]};
        computeNonAxi(a, N, L, M,r, theta, phi, Acos, Asin, 3, e, F);
    }
    else
    {
        double (*Eq[3])(double, double, double) = {&computeAxiF_r, &computeAxiF_theta, &computeAxiF_phi};
        axi_equations e = {Eq,&PhiTilde_Pointer[0], &P_Pointer[0], &Constant[0]};
        compute(a, N, L, M,r, theta, phi, Acos, 3, e, F);
    }



    //Caching

    *cached_type = (double)FORCE;

    * cached_coords = R;
    * (cached_coords + 1) = Z;
    * (cached_coords + 2) = phi;
    * (cached_values) = *F;
    * (cached_values + 1) = *(F + 1);
    * (cached_values + 2) = *(F + 2);

    // Free memory
    free(C);
    free(dC);
    free(phiTilde);
    free(dphiTilde);
    free(P);
    free(dP);

}

//Compute the Derivatives
void computeDeriv(double R,double Z, double phi,
		  double t,
		  struct potentialArg * potentialArgs, double * F)
{
    double * args= potentialArgs->args;
    //Get args
    double a = *args++;
    int isNonAxi = (int)*args++;
    int N = (int) *args++;
    int L = (int) *args++;
    int M = (int) *args++;
    double* Acos = args;

    double * caching_i = (args + (isNonAxi + 1)*N*L*M);
    double *Asin;
    if (isNonAxi == 1)
    {
        Asin = args + N*L*M;
    }

    double *cached_type = caching_i;
    double * cached_coords = (caching_i+ 1);
    double * cached_values = (caching_i + 4);
    if ((int)*cached_type==DERIV)
    {
        if (*cached_coords == R && *(cached_coords + 1) == Z && *(cached_coords + 2) == phi)
        {
            *F = *cached_values;
            *(F + 1) = *(cached_values + 1);
            *(F + 2) = *(cached_values + 2);
            return;
        }
    }

    double r;
    double theta;
    cyl_to_spher(R, Z, &r, &theta);

    double xi;
    calculateXi(r, a, &xi);

//Compute the gegenbauer polynomials and its derivative.
    double *C= (double *) malloc ( N*L * sizeof(double) );
    double *dC= (double *) malloc ( N*L * sizeof(double) );
    double *d2C= (double *) malloc ( N*L * sizeof(double) );
    double *phiTilde= (double *) malloc ( N*L * sizeof(double) );
    double *dphiTilde= (double *) malloc ( N*L * sizeof(double) );
    double *d2phiTilde= (double *) malloc ( N*L * sizeof(double) );

    compute_C(xi, N, L, C);
    compute_dC(xi, N, L, dC);
    compute_d2C(xi, N, L, d2C);

//Compute phiTilde and its derivative
    compute_phiTilde(r, a, N, L, C, phiTilde);
    compute_dphiTilde(r, a, N, L, C, dC, dphiTilde);
    compute_d2phiTilde(r, a, N, L, C, dC, d2C, d2phiTilde);


//Compute Associated Legendre Polynomials
    int M_eff = M;
    int size = 0;

    if (isNonAxi==0)
    {
    M_eff = 1;
    size = L;
    } else{
    size = L*L - L*(L-1)/2;
    }
    double *P= (double *) malloc ( size * sizeof(double) );

    compute_P(cos(theta), L,M_eff, P);

    double (*PhiTilde_Pointer[3])= {d2phiTilde,phiTilde,dphiTilde};
    double (*P_Pointer[3]) = {P, P, P};

    double Constant[3] = {1., 1., 1.};

    if (isNonAxi==1)
    {
        double (*Eq[3])(double, double, double, double, double, double, int) = {&computeF_rr, &computeF_phiphi, &computeF_rphi};
        equations e = {Eq,&PhiTilde_Pointer[0],&P_Pointer[0],&Constant[0]};
        computeNonAxi(a, N, L, M,r, theta, phi, Acos, Asin, 3, e, F);
    }
    else
    {
        double (*Eq[3])(double, double, double) = {&computeAxiF_rr, &computeAxiF_phiphi, &computeAxiF_rphi};
        axi_equations e = {Eq,&PhiTilde_Pointer[0],&P_Pointer[0],&Constant[0]};
        compute(a, N, L, M,r, theta, phi, Acos, 3, e, F);
    }


    //Caching

    *cached_type = (double)DERIV;

    * cached_coords = R;
    * (cached_coords + 1) = Z;
    * (cached_coords + 2) = phi;
    * (cached_values) = *F;
    * (cached_values + 1) = *(F + 1);
    * (cached_values + 2) = *(F + 2);

    //Free memory
    free(C);
    free(dC);
    free(d2C);
    free(phiTilde);
    free(dphiTilde);
    free(d2phiTilde);
    free(P);
}

//Compute the Potential
double SCFPotentialEval(double R,double Z, double phi,
                        double t,
                        struct potentialArg * potentialArgs)
{
    double * args= potentialArgs->args;
    //Get args
    double a = *args++;
    int isNonAxi = (int)*args++;
    int N = (int) *args++;
    int L = (int) *args++;
    int M = (int) *args++;
    double* Acos = args;
    double* Asin;
    if (isNonAxi==1) // LCOV_EXCL_START
    {
        Asin = args + N*L*M;
    } // LCOV_EXCL_STOP
    //convert R,Z to r, theta
    double r;
    double theta;
    cyl_to_spher(R, Z,&r, &theta);
    double xi;
    calculateXi(r, a, &xi);

    //Compute the gegenbauer polynomials and its derivative.
    double *C= (double *) malloc ( N*L * sizeof(double) );
    double *phiTilde= (double *) malloc ( N*L * sizeof(double) );

    compute_C(xi, N, L, C);

    //Compute phiTilde and its derivative
    compute_phiTilde(r, a, N, L, C, phiTilde);
    //Compute Associated Legendre Polynomials

    int M_eff = M;
    int size = 0;

    if (isNonAxi==0)
    {
    M_eff = 1;
    size = L;
    } else{ // LCOV_EXCL_START
    size = L*L - L*(L-1)/2;
    } // LCOV_EXCL_STOP

    double *P= (double *) malloc ( size * sizeof(double) );

    compute_P(cos(theta), L,M_eff, P);

    double potential;

    double (*PhiTilde_Pointer[1]) = {phiTilde};
    double (*P_Pointer[1]) = {P};

    double Constant[1] = {1.};

    if (isNonAxi==1) // LCOV_EXCL_START
    {
        double (*Eq[1])(double, double, double, double, double, double, int) = {&computePhi};
        equations e = {Eq,&PhiTilde_Pointer[0],&P_Pointer[0],&Constant[0]};
        computeNonAxi(a, N, L, M,r, theta, phi, Acos, Asin, 1, e, &potential);
    } // LCOV_EXCL_STOP
    else
    {
        double (*Eq[1])(double, double, double) = {&computeAxiPhi};
        axi_equations e = {Eq,&PhiTilde_Pointer[0],&P_Pointer[0],&Constant[0]};
        compute(a, N, L, M,r, theta, phi, Acos, 1, e, &potential);
    }

    //Free memory
    free(C);
    free(phiTilde);
    free(P);

    return potential;

}

//Compute the force in the R direction
double SCFPotentialRforce(double R,double Z, double phi,
                          double t,
                          struct potentialArg * potentialArgs)
{
    double r;
    double theta;
    cyl_to_spher(R, Z, &r, &theta);
    // The derivatives
    double dr_dR = R/r;
    double dtheta_dR = Z/(r*r);
    double dphi_dR = 0;


    double F[3];
    computeForce(R, Z, phi, t,potentialArgs, &F[0]) ;

    return *(F + 0)*dr_dR + *(F + 1)*dtheta_dR + *(F + 2)*dphi_dR;


}

//Compute the force in the z direction
double SCFPotentialzforce(double R,double Z, double phi,
                          double t,
                          struct potentialArg * potentialArgs)
{
    double r;
    double theta;
    cyl_to_spher(R, Z,&r, &theta);

    double dr_dz = Z/r;
    double dtheta_dz = -R/(r*r);
    double dphi_dz = 0;

    double F[3];
    computeForce(R, Z, phi, t,potentialArgs, &F[0]) ;
    return *(F + 0)*dr_dz + *(F + 1)*dtheta_dz + *(F + 2)*dphi_dz;
}

//Compute the force in the phi direction
double SCFPotentialphitorque(double R,double Z, double phi,
                            double t,
                            struct potentialArg * potentialArgs)
{

    double r;
    double theta;
    cyl_to_spher(R, Z, &r, &theta);

    double dr_dphi = 0;
    double dtheta_dphi = 0;
    double dphi_dphi = 1;

    double F[3];
    computeForce(R, Z, phi, t,potentialArgs, &F[0]) ;

    return *(F + 0)*dr_dphi + *(F + 1)*dtheta_dphi + *(F + 2)*dphi_dphi;
}

//Compute the planar force in the R direction
double SCFPotentialPlanarRforce(double R,double phi,
                                double t,
                                struct potentialArg * potentialArgs)
{
    return SCFPotentialRforce(R,0., phi,t,potentialArgs);

}

//Compute the planar force in the phi direction
double SCFPotentialPlanarphitorque(double R,double phi,
                                  double t,
                                  struct potentialArg * potentialArgs)
{
    return SCFPotentialphitorque(R,0., phi,t,potentialArgs);
}


//Compute the planar double derivative of the potential with respect to R
double SCFPotentialPlanarR2deriv(double R, double phi,
                                 double t,
                                 struct potentialArg * potentialArgs)
{
    double Farray[3];
    computeDeriv(R, 0, phi, t,potentialArgs, &Farray[0]) ;
    return *Farray;
}

//Compute the planar double derivative of the potential with respect to phi
double SCFPotentialPlanarphi2deriv(double R, double phi,
                                   double t,
                                   struct potentialArg * potentialArgs)
{
    double Farray[3];
    computeDeriv(R, 0, phi, t,potentialArgs, &Farray[0]) ;
    return *(Farray + 1);
}

//Compute the planar double derivative of the potential with respect to R, Phi
double SCFPotentialPlanarRphideriv(double R, double phi,
                                   double t,
                                   struct potentialArg * potentialArgs)
{
    double Farray[3];
    computeDeriv(R, 0, phi, t,potentialArgs, &Farray[0]) ;
    return *(Farray + 2);
}
//Compute the density
double SCFPotentialDens(double R,double Z, double phi,
			double t,
			struct potentialArg * potentialArgs)
{
    double * args= potentialArgs->args;
    //Get args
    double a = *args++;
    int isNonAxi = (int)*args++;
    int N = (int) *args++;
    int L = (int) *args++;
    int M = (int) *args++;
    double* Acos = args;
    double* Asin;
    if (isNonAxi==1)
    {
        Asin = args + N*L*M;
    }
    //convert R,Z to r, theta
    double r;
    double theta;
    cyl_to_spher(R, Z,&r, &theta);
    double xi;
    calculateXi(r, a, &xi);

    //Compute the gegenbauer polynomials and its derivative.
    double *C= (double *) malloc ( N*L * sizeof(double) );
    double *rhoTilde= (double *) malloc ( N*L * sizeof(double) );

    compute_C(xi, N, L, C);

    //Compute rhoTilde and its derivative
    compute_rhoTilde(r, a, N, L, C, rhoTilde);
    //Compute Associated Legendre Polynomials
    int M_eff = M;
    int size = 0;

    if (isNonAxi==0) {
      M_eff = 1;
      size = L;
    } else
      size = L*L - L*(L-1)/2;

    double *P= (double *) malloc ( size * sizeof(double) );

    compute_P(cos(theta), L,M_eff, P);

    double density;

    double (*RhoTilde_Pointer[1]) = {rhoTilde};
    double (*P_Pointer[1]) = {P};

    double Constant[1] = {1.};

    if (isNonAxi==1)
    {
        double (*Eq[1])(double, double, double, double, double, double, int) = {&computePhi};
        equations e = {Eq,&RhoTilde_Pointer[0],&P_Pointer[0],&Constant[0]};
        computeNonAxi(a, N, L, M,r, theta, phi, Acos, Asin, 1, e, &density);
    }
    else
    {
        double (*Eq[1])(double, double, double) = {&computeAxiPhi};
        axi_equations e = {Eq,&RhoTilde_Pointer[0],&P_Pointer[0],&Constant[0]};
        compute(a, N, L, M,r, theta, phi, Acos, 1, e, &density);
    }

    //Free memory
    free(C);
    free(rhoTilde);
    free(P);

    return density / 2. / M_PI;

}
