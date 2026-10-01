// Incomplete beta B_z(p, q) = int_0^z u^(p-1) (1-u)^(q-1) du for p > 0, q > -1,
// accurate to round-off including q -> 0, q -> -1 and large p + q (the python
// version is galpy.util.special.incomplete_beta). Used by
// TwoPowerSphericalPotential.
#include <math.h>
#include "incomplete_beta.h"
// the integrand's mass centre (p+1)/(p+q+2), at most 0.9 (and 0.9 for
// p+q+2 <= 0, where the mass piles up at 1; TwoPower's alpha >= beta + 2)
double galpy_incomplete_beta_split(double p, double q){
  if ( p + q + 2. <= 0. ) return 0.9;
  double c= (p + 1.) / (p + q + 2.);
  return c > 0.9 ? 0.9 : c;
}
#define IBETA_QSMALL 0.05
// b < 0 < c - 1: Euler's 2F1(1, b; c; z) = (1-z)^(c-1-b) 2F1(c-1, c-b; c; z),
// whose terms are all positive (the direct ones alternate and cancel for
// b << 0: TwoPower's beta << 0); the sum is rescaled so it cannot overflow
static double hyp2f1_1_euler(double b, double c, double z){
  double t= 1., out= 1., lsc= 0.;
  int k;
  for (k=0; k < 10000000; k++){
    t*= (c - 1. + k) * (c - b + k) / ((c + k) * (k + 1.)) * z;
    out+= t;
    if ( out > 1e200 ) {
      out*= 1e-200;
      t*= 1e-200;
      lsc+= 460.51701859880914; // log(1e200)
    }
    if ( k > 5 && t <= 1e-17 * out ) break;
    if ( ! isfinite(out) ) break; // a NaN argument never converges
  }
  return exp(lsc + (c - 1. - b) * log1p(-z)) * out;
}
// 2F1(1, b; c; z) = sum_k (b)_k/(c)_k z^k for c > 0, 0 <= z < 1 (Euler's
// transformation above for b < 0 < c - 1). galpy's general hyp2f1 loses up to
// ~1e-3 here at large positive b.
double galpy_hyp2f1_1(double b, double c, double z){
  if ( b < 0. && c > 1. ) return hyp2f1_1_euler(b, c, z);
  double t= 1., out= 1.;
  int k;
  for (k=0; k < 10000000; k++){
    t*= (b + k) / (c + k) * z;
    out+= t;
    if ( k > 5 && fabs(t) <= 1e-17 * fabs(out) ) break;
    if ( ! isfinite(out) ) break; // a NaN argument never converges
  }
  return out;
}
// K(s) = ((1-s)^p 2F1(1, p+q; q+1; s) - 1)/q (or its q = 0 limit), summed so that
// the O(q) difference from 1 is never formed by subtraction
static double ibeta_k_series(double p, double q, double s){
  double t= 1., D= 0., out= 0., term;
  int k;
  for (k=1; k < 10000000; k++){
    t*= (p + k - 1.) / k * s;
    // D_k = (R_k - 1)/q, R_k the Pochhammer ratio (p+q)_k k!/((p)_k (1+q)_k):
    // no O(q) difference is formed, and no log (p + q < 0 when alpha > beta)
    D= D * (1. + q / (p + k - 1.)) / (1. + q / k) + (1. - p) / ((p + k - 1.) * (k + q));
    term= t * D;
    out+= term;
    if ( k > 5 && (p + k) / (k + 1.) * s < 1. && fabs(term) <= 1e-17 * fabs(out) )
      break;
    if ( ! isfinite(out) ) break; // a NaN argument never converges
  }
  return pow(1. - s, p) * out;
}
// base + int_{s1}^{s2} v^(q-1) (1-v)^(p-1) dv for |q| < IBETA_QSMALL (through K)
static double ibeta_reflected_smallq(double p, double q, double s1, double s2,
                                  double base){
  double K2= ibeta_k_series(p, q, s2);
  double lg= log(s2 / s1);
  double first;
  if ( q == 0. ) first= lg;
  else if ( fabs(q * lg) < 1. ) first= pow(s1, q) * expm1(q * lg) / q;
  else first= (pow(s2, q) - pow(s1, q)) / q;
  return base + first * (1. + q * K2) + pow(s1, q) * (K2 - ibeta_k_series(p, q, s1));
}
// int_{s1}^{s2} v^(q-1) (1-v)^(p-1) dv: through K(s) when |q| is small; near a
// negative integer q (beta -> 2, 1, ...), where the antiderivative's
// 2F1(1, p+q; q+1; v) has a pole, integrated by parts to q + 1 first
static double ibeta_reflected(double p, double q, double s1, double s2){
  if ( fabs(q) < IBETA_QSMALL )
    return ibeta_reflected_smallq(p, q, s1, s2, 0.);
  double n= round(q);
  if ( n <= -1. && fabs(q - n) < IBETA_QSMALL )
    return pow(s2, q) * pow(1. - s2, p) / q - pow(s1, q) * pow(1. - s1, p) / q
      + (p + q) / q * ibeta_reflected(p, q + 1., s1, s2);
  return pow(s2, q) * pow(1. - s2, p) / q * galpy_hyp2f1_1(p + q, q + 1., s2)
    - pow(s1, q) * pow(1. - s1, p) / q * galpy_hyp2f1_1(p + q, q + 1., s1);
}
// B_z(p, q) for p > 0, q > -1, 0 <= z < 1, given s = 1 - z (exact); split at the
// integrand's mass centre c = (p+1)/(p+q+2) (at most 0.9): below it the direct
// positive series, above it B_c plus the reflected int_{1-z}^{1-c} v^(q-1)
// (1-v)^(p-1) dv, which holds the mass
double galpy_incomplete_beta(double p, double q, double z, double s){
  double c= galpy_incomplete_beta_split(p, q);
  if ( z <= c )
    return pow(z, p) * pow(s, q) / p * galpy_hyp2f1_1(p + q, p + 1., z);
  double s2= 1. - c;
  double ibc= pow(c, p) * pow(s2, q) / p * galpy_hyp2f1_1(p + q, p + 1., c);
  return ibc + ibeta_reflected(p, q, s, s2);
}
