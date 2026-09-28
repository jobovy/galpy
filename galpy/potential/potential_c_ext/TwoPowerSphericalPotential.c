#include <math.h>
#include <gsl/gsl_sf_gamma.h>
#include "galpy_potentials.h"
#include "wrap_xsf.h"

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
// beta (the python implementation, TwoPowerSphericalPotential.py, is the same).
#define TP_QSMALL 0.05
// 2F1(1, b; c; z) = sum_k (b)_k/(c)_k z^k for b > -1, c > 0, 0 <= z < 1.
// Terms are negative when -1 < b < 0 (alpha > beta); galpy's general hyp2f1
// loses up to ~1e-3 here at large positive b.
static double tp_2f1_1(double b, double c, double z){
  double t= 1., out= 1.;
  int k;
  for (k=0; k < 10000000; k++){
    t*= (b + k) / (c + k) * z;
    out+= t;
    if ( k > 5 && fabs(t) <= 1e-17 * fabs(out) ) break;
  }
  return out;
}
// K(s) = ((1-s)^p 2F1(1, p+q; q+1; s) - 1)/q (or its q = 0 limit), summed so that
// the O(q) difference from 1 is never formed by subtraction
static double tp_k_series(double p, double q, double s){
  double t= 1., L= 0., out= 0., term;
  int k;
  for (k=1; k < 10000000; k++){
    t*= (p + k - 1.) / k * s;
    if ( q != 0. ) {
      L+= log1p(q / (p + k - 1.)) - log1p(q / k);
      term= t * expm1(L) / q;
    }
    else {
      L+= 1. / (p + k - 1.) - 1. / k;
      term= t * L;
    }
    out+= term;
    if ( k > 5 && (p + k) / (k + 1.) * s < 1. && fabs(term) <= 1e-17 * fabs(out) )
      break;
  }
  return pow(1. - s, p) * out;
}
// B_z(p, q) for p > 0, q > -1, 0 <= z < 1, given s = 1 - z (exact); split at the
// integrand's mass centre c = (p+1)/(p+q+2) (at most 0.9): below it the direct
// positive series, above it B_c plus the reflected int_{1-z}^{1-c} v^(q-1)
// (1-v)^(p-1) dv, which holds the mass (through K(s) when |q| is small)
static double tp_ibeta(double p, double q, double z, double s){
  double c= (p + 1.) / (p + q + 2.);
  if ( c > 0.9 ) c= 0.9;
  if ( z <= c )
    return pow(z, p) * pow(s, q) / p * tp_2f1_1(p + q, p + 1., z);
  double s2= 1. - c;
  double ibc= pow(c, p) * pow(s2, q) / p * tp_2f1_1(p + q, p + 1., c);
  if ( fabs(q) >= TP_QSMALL )
    return ibc + pow(s2, q) * pow(1. - s2, p) / q * tp_2f1_1(p + q, q + 1., s2)
      - pow(s, q) * pow(1. - s, p) / q * tp_2f1_1(p + q, q + 1., s);
  double K2= tp_k_series(p, q, s2);
  double lg= log(s2 / s);
  double first;
  if ( q == 0. ) first= lg;
  else if ( fabs(q * lg) < 1. ) first= pow(s, q) * expm1(q * lg) / q;
  else first= (pow(s2, q) - pow(s, q)) / q;
  return ibc + first * (1. + q * K2) + pow(s, q) * (K2 - tp_k_series(p, q, s));
}
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
  return -amp * ( tp_ibeta(3.-alpha, beta-3., w, s) / x
                  + tp_ibeta(beta-2., 2.-alpha, s, w) ) / a;
}

double TwoPowerSphericalPotentialRforce(double R,double Z, double phi,
                                        double t,
                                        struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  double amp= *args++;
  double a= *args++;
  double alpha= *args++;
  double beta= *args;
  //Calculate Rforce
  double r= sqrt(R*R+Z*Z);
  return -amp * R * pow(r, -alpha) * pow(a, alpha - 3.) / (3. - alpha)
              * hyp2f1(3. - alpha, beta - alpha, 4. - alpha, -r/a);
}

double TwoPowerSphericalPotentialPlanarRforce(double R,double phi,
                                              double t,
                                              struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  double amp= *args++;
  double a= *args++;
  double alpha= *args++;
  double beta= *args;
  //Calculate Rforce
  return -amp * pow(R, 1.0 - alpha) * pow(a, alpha - 3.) / (3. - alpha)
              * hyp2f1(3. - alpha, beta - alpha, 4. - alpha, -R/a);
}

double TwoPowerSphericalPotentialzforce(double R,double Z,double phi,
                                        double t,
                                        struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  double amp= *args++;
  double a= *args++;
  double alpha= *args++;
  double beta= *args;
  //Calculate zforce
  double r= sqrt(R*R+Z*Z);
  return -amp * Z * pow(r, -alpha) * pow(a, alpha - 3.) / (3. - alpha)
              * hyp2f1(3. - alpha, beta - alpha, 4. - alpha, -r/a);
}

double TwoPowerSphericalPotentialPlanarR2deriv(double R,double phi,
                                               double t,
                                               struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  double amp= *args++;
  double a= *args++;
  double alpha= *args++;
  double beta= *args;
  //Calculate R2deriv using analytical derivative
  double A = pow(a, alpha - 3.) / (3. - alpha);
  double hyper = hyp2f1(3. - alpha, beta - alpha, 4. - alpha, -R/a);
  double hyper_deriv = (3. - alpha) * (beta - alpha) / (4. - alpha)
                       * hyp2f1(4. - alpha, 1. + beta - alpha, 5. - alpha, -R/a);

  double term1 = A * pow(R, -alpha) * hyper;
  double term2 = -alpha * A * pow(R, -alpha) * hyper;
  double term3 = -A * pow(R, 1. - alpha) * pow(a, -1.) * hyper_deriv;
  return amp * (term1 + term2 + term3);
}

double TwoPowerSphericalPotentialR2deriv(double R,double Z, double phi,
                                         double t,
                                         struct potentialArg * potentialArgs){
  //Spherical: Phi''(r)=PlanarR2deriv(r), Phi'(r)=-PlanarRforce(r) (incl. amp)
  double r2= R * R + Z * Z;
  double r= sqrt( r2 );
  double Phipp= TwoPowerSphericalPotentialPlanarR2deriv(r,phi,t,potentialArgs);
  double Phip= -TwoPowerSphericalPotentialPlanarRforce(r,phi,t,potentialArgs);
  double ir2= 1. / r2;
  double ir3= ir2 / r;
  //R2deriv = Phi''*R^2/r^2 + Phi'*z^2/r^3
  return Phipp * R * R * ir2 + Phip * Z * Z * ir3;
}
double TwoPowerSphericalPotentialz2deriv(double R,double Z, double phi,
                                         double t,
                                         struct potentialArg * potentialArgs){
  double r2= R * R + Z * Z;
  double r= sqrt( r2 );
  double Phipp= TwoPowerSphericalPotentialPlanarR2deriv(r,phi,t,potentialArgs);
  double Phip= -TwoPowerSphericalPotentialPlanarRforce(r,phi,t,potentialArgs);
  double ir2= 1. / r2;
  double ir3= ir2 / r;
  //z2deriv = Phi''*z^2/r^2 + Phi'*R^2/r^3
  return Phipp * Z * Z * ir2 + Phip * R * R * ir3;
}
double TwoPowerSphericalPotentialRzderiv(double R,double Z, double phi,
                                         double t,
                                         struct potentialArg * potentialArgs){
  double r2= R * R + Z * Z;
  double r= sqrt( r2 );
  double Phipp= TwoPowerSphericalPotentialPlanarR2deriv(r,phi,t,potentialArgs);
  double Phip= -TwoPowerSphericalPotentialPlanarRforce(r,phi,t,potentialArgs);
  double ir2= 1. / r2;
  double ir3= ir2 / r;
  //Rzderiv = R*z*(Phi''/r^2 - Phi'/r^3)
  return R * Z * ( Phipp * ir2 - Phip * ir3 );
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
