#include <math.h>
#include <galpy_potentials.h>
// Special case of EllipsoidalPotential, only need to define three functions
double TriaxialGaussianPotentialpsi(double m,double * args){
  double minustwosigma2= *args;
  return minustwosigma2 * exp ( m * m / minustwosigma2 );
}
double TriaxialGaussianPotentialmdens(double m,double * args){
  double minustwosigma2= *args;
  return exp ( m * m / minustwosigma2 );
}
// LCOV_EXCL_START
double TriaxialGaussianPotentialmdensDeriv(double m,double * args){
  double minustwosigma2= *args;
  return 2. * m / minustwosigma2 * exp ( m * m / minustwosigma2 );
}
// LCOV_EXCL_STOP
