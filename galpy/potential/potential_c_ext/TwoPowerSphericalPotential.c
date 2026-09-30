#include <math.h>
#include <gsl/gsl_sf_gamma.h>
#include "galpy_potentials.h"
#include "incomplete_beta.h"

// Define M_PI if not already defined (needed for Windows)
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

//TwoPowerSphericalPotential
//4 arguments: amp, a, alpha, beta
// The potential through two incomplete beta integrals (w = x/(1+x), x = r/a):
//   Phi = -(amp/a) [M(x)/x + O(x)],  M = B_w(3-alpha, beta-3),
//   O = B_{1-w}(beta-2, 2-alpha),  B_z(p,q) = int_0^z u^(p-1) (1-u)^(q-1) du,
// with no cancellation as beta -> 3 or alpha -> 2 and no Gamma overflow at large
// beta (galpy/util/incomplete_beta.c; the python implementation is
// galpy.util.special.incomplete_beta).
double TwoPowerSphericalPotentialEval(double R,double Z, double phi,
                                       double t,
                                       struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  double amp= *args++;
  double a= *args++;
  double alpha= *args++;
  double beta= *args;
  //Calculate potential
  double r= sqrt(R*R+Z*Z);
  if ( r == 0. ) // Phi(0) = -B(2-alpha, beta-2)/a, finite for alpha < 2 only
    return alpha < 2. ? -amp * exp(gsl_sf_lnbeta(2.-alpha, beta-2.)) / a : -INFINITY;
  if ( isinf(r) ) return 0.;
  double x= r / a;
  double w= x / (1. + x), s= 1. / (1. + x);
  return -amp * ( galpy_incomplete_beta(3.-alpha, beta-3., w, s) / x
                  + galpy_incomplete_beta(beta-2., 2.-alpha, s, w) ) / a;
}

// Forces and second derivatives from the same M (amp = 1; 4 pi rho a^3 = D =
// w^-alpha s^beta): dPhi/dr / r = M/(x a)^3 and, by Poisson,
//   Phi'' = 4 pi rho - 2 dPhi/dr / r,  Phi'' - dPhi/dr / r = 4 pi rho - 3 dPhi/dr / r.
// Below the split c of galpy_incomplete_beta, M = w^p s^q / p (1 + G) with
// G = 2F1(1, p+q; p+1; w) - 1 = (p+q)/(p+1) w 2F1(1, p+q+1; p+2; w), so with
// E = D/p these are E (1 + G), E (1 - alpha - 2 G) and -E (alpha + 3 G): the
// x^-alpha terms of 4 pi rho and k M/x^3 that cancel at alpha = 1 (k = 2) and
// alpha = 0 (k = 3) are subtracted in closed form. Above c (where G >~ 1/2)
// D - k M/x^3 directly. No hyp2f1(..., -r/a): that was 3e-3 off at
// beta = 3 +- 1e-12 and NaN at large beta and r.
// f[0] = dPhi/dr / r; if hess, also f[1] = Phi'', f[2] = Phi'' - dPhi/dr / r
static void tp_radial(double r, double a, double alpha, double beta, int hess,
                      double * f){
  double x= r / a;
  double w= x / (1. + x), s= 1. / (1. + x);
  double p= 3. - alpha, q= beta - 3.;
  double c= galpy_incomplete_beta_split(p, q);
  double a3= a * a * a;
  if ( w <= c ) {
    double E= pow(w, -alpha) * pow(s, beta) / p / a3;
    double G= (p + q) / (p + 1.) * w * galpy_hyp2f1_1(p + q + 1., p + 2., w);
    f[0]= E * (1. + G);
    if ( hess ) {
      f[1]= E * (1. - alpha - 2. * G);
      f[2]= -E * (alpha + 3. * G);
    }
    return;
  }
  double m= galpy_incomplete_beta(p, q, w, s) * pow(s / w, 3.) / a3;
  f[0]= m;
  if ( hess ) {
    double D= pow(w, -alpha) * pow(s, beta) / a3;
    f[1]= D - 2. * m;
    f[2]= D - 3. * m;
  }
}
double TwoPowerSphericalPotentialRforce(double R,double Z, double phi,
                                        double t,
                                        struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  double amp= *args++;
  double a= *args++;
  double alpha= *args++;
  double beta= *args;
  double f[1];
  tp_radial(sqrt(R*R+Z*Z), a, alpha, beta, 0, f);
  return -amp * R * f[0];
}

double TwoPowerSphericalPotentialPlanarRforce(double R,double phi,
                                              double t,
                                              struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  double amp= *args++;
  double a= *args++;
  double alpha= *args++;
  double beta= *args;
  double f[1];
  tp_radial(R, a, alpha, beta, 0, f);
  return -amp * R * f[0];
}

double TwoPowerSphericalPotentialzforce(double R,double Z,double phi,
                                        double t,
                                        struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  double amp= *args++;
  double a= *args++;
  double alpha= *args++;
  double beta= *args;
  double f[1];
  tp_radial(sqrt(R*R+Z*Z), a, alpha, beta, 0, f);
  return -amp * Z * f[0];
}

double TwoPowerSphericalPotentialPlanarR2deriv(double R,double phi,
                                               double t,
                                               struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  double amp= *args++;
  double a= *args++;
  double alpha= *args++;
  double beta= *args;
  double f[3];
  tp_radial(R, a, alpha, beta, 1, f);
  return amp * f[1];
}

// Spherical: R2deriv = (R^2 Phi'' + z^2 Phi'/r)/r^2, z2deriv the same with
// R <-> z, Rzderiv = R z (Phi'' - Phi'/r)/r^2
double TwoPowerSphericalPotentialR2deriv(double R,double Z, double phi,
                                         double t,
                                         struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  double amp= *args++;
  double a= *args++;
  double alpha= *args++;
  double beta= *args;
  double r2= R * R + Z * Z;
  double f[3];
  tp_radial(sqrt(r2), a, alpha, beta, 1, f);
  return amp * (R * R * f[1] + Z * Z * f[0]) / r2;
}
double TwoPowerSphericalPotentialz2deriv(double R,double Z, double phi,
                                         double t,
                                         struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  double amp= *args++;
  double a= *args++;
  double alpha= *args++;
  double beta= *args;
  double r2= R * R + Z * Z;
  double f[3];
  tp_radial(sqrt(r2), a, alpha, beta, 1, f);
  return amp * (Z * Z * f[1] + R * R * f[0]) / r2;
}
double TwoPowerSphericalPotentialRzderiv(double R,double Z, double phi,
                                         double t,
                                         struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  double amp= *args++;
  double a= *args++;
  double alpha= *args++;
  double beta= *args;
  double r2= R * R + Z * Z;
  double f[3];
  tp_radial(sqrt(r2), a, alpha, beta, 1, f);
  return amp * R * Z * f[2] / r2;
}
double TwoPowerSphericalPotentialDens(double R,double Z, double phi,
                                      double t,
                                      struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  double amp= *args++;
  double a= *args++;
  double alpha= *args++;
  double beta= *args;
  //Calculate density
  double r= sqrt(R*R+Z*Z);
  return amp * pow(a / r, alpha) * pow(1. + r / a, alpha -beta)
             / 4. / M_PI * pow(a, -3.);
}
