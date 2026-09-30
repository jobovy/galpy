// Incomplete beta B_z(p, q) = int_0^z u^(p-1) (1-u)^(q-1) du for p > 0, q > -1,
// accurate to round-off including q -> 0, q -> -1 and large p + q (the python
// version is galpy.util.special.incomplete_beta). Used by
// TwoPowerSphericalPotential.
#include <math.h>
#include "incomplete_beta.h"
double galpy_incomplete_beta_split(double p, double q){
  double c= (p + 1.) / (p + q + 2.);
  return c > 0.9 ? 0.9 : c;
}
#define IBETA_QSMALL 0.05
// 2F1(1, b; c; z) = sum_k (b)_k/(c)_k z^k for b > -1, c > 0, 0 <= z < 1.
// Terms are negative when -1 < b < 0 (alpha > beta); galpy's general hyp2f1
// loses up to ~1e-3 here at large positive b.
double galpy_hyp2f1_1(double b, double c, double z){
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
// B_z(p, q) for p > 0, q > -1, 0 <= z < 1, given s = 1 - z (exact); split at the
// integrand's mass centre c = (p+1)/(p+q+2) (at most 0.9): below it the direct
// positive series, above it B_c plus the reflected int_{1-z}^{1-c} v^(q-1)
// (1-v)^(p-1) dv, which holds the mass (through K(s) when |q| is small; as
// q -> -1 (beta -> 2) integrated by parts to q + 1 first)
double galpy_incomplete_beta(double p, double q, double z, double s){
  double c= galpy_incomplete_beta_split(p, q);
  if ( z <= c )
    return pow(z, p) * pow(s, q) / p * galpy_hyp2f1_1(p + q, p + 1., z);
  double s2= 1. - c;
  double ibc= pow(c, p) * pow(s2, q) / p * galpy_hyp2f1_1(p + q, p + 1., c);
  if ( fabs(q) < IBETA_QSMALL )
    return ibeta_reflected_smallq(p, q, s, s2, ibc);
  if ( fabs(q + 1.) < IBETA_QSMALL )
    return ibc + pow(s2, q) * pow(1. - s2, p) / q - pow(s, q) * pow(1. - s, p) / q
      + (p + q) / q * ibeta_reflected_smallq(p, q + 1., s, s2, 0.);
  return ibc + pow(s2, q) * pow(1. - s2, p) / q * galpy_hyp2f1_1(p + q, q + 1., s2)
    - pow(s, q) * pow(1. - s, p) / q * galpy_hyp2f1_1(p + q, q + 1., s);
}
