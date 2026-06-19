#include <math.h>
#include <gsl/gsl_sf_expint.h>
#include <galpy_potentials.h>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
//ExpTruncNFWPotential
// 3 arguments: amp, a, rc
// Mirrors the branching in the Python _F (closed form vs. small-r series) and
// the closed-form _G; see ExpTruncNFWPotential.py for the derivation.
static double ExpTruncNFW_F(double r, double a, double rc) {
  double small_r_thresh = 1e-3 * (a < rc ? a : rc);
  if (r < small_r_thresh) {
    double c2 = 0.5;
    double c3 = -(1.0 / rc + 2.0 / a) / 3.0;
    double c4 = (0.5 / (rc * rc) + 2.0 / (rc * a) + 3.0 / (a * a)) / 4.0;
    double c5 = -(1.0 / (6.0 * rc * rc * rc) + 1.0 / (rc * rc * a) +
                  3.0 / (rc * a * a) + 4.0 / (a * a * a)) /
                5.0;
    return (r * r / (a * a)) * (c2 + r * (c3 + r * (c4 + r * c5)));
  } else {
    double alpha = a / rc;
    double beta = (a + r) / rc;
    return (exp(alpha) * (1.0 + alpha) *
                (gsl_sf_expint_E1(alpha) - gsl_sf_expint_E1(beta)) -
            1.0 + a * exp(-r / rc) / (a + r));
  }
}
static double ExpTruncNFW_G(double r, double a, double rc) {
  double alpha = a / rc;
  double beta = (a + r) / rc;
  return exp(-r / rc) / (a + r) - exp(alpha) * gsl_sf_expint_E1(beta) / rc;
}
double ExpTruncNFWPotentialrdens(double r, double t,
                                  struct potentialArg *potentialArgs) {
  double *args = potentialArgs->args;
  //Get args
  double a = *(args + 1);
  double rc = *(args + 2);
  return exp(-r / rc) / (4.0 * M_PI * a * a * r * pow(1.0 + r / a, 2));
}
double ExpTruncNFWPotentialrevaluate(double r, double t,
                                      struct potentialArg *potentialArgs) {
  double *args = potentialArgs->args;
  //Get args
  double a = *(args + 1);
  double rc = *(args + 2);
  if (r == 0.)
    return -ExpTruncNFW_G(0., a, rc);
  return -(ExpTruncNFW_F(r, a, rc) / r + ExpTruncNFW_G(r, a, rc));
}
double ExpTruncNFWPotentialrforce(double r, double t,
                                   struct potentialArg *potentialArgs) {
  double *args = potentialArgs->args;
  //Get args
  double a = *(args + 1);
  double rc = *(args + 2);
  if (r == 0.)
    return -0.5 / (a * a);
  return -ExpTruncNFW_F(r, a, rc) / (r * r);
}
double ExpTruncNFWPotentialr2deriv(double r, double t,
                                    struct potentialArg *potentialArgs) {
  double *args = potentialArgs->args;
  //Get args
  double a = *(args + 1);
  double rc = *(args + 2);
  return 4.0 * M_PI * ExpTruncNFWPotentialrdens(r, t, potentialArgs) -
         2.0 * ExpTruncNFW_F(r, a, rc) / (r * r * r);
}
