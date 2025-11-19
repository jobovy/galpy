#include <stdbool.h>
#include <math.h>
#include <gsl/gsl_sf_gamma.h>
#include <galpy_potentials.h>
#include "wrap_xsf.h"
//TriaxialHernquistPotential
// Special case of EllipsoidalPotential, only need to define three functions
double TriaxialHernquistPotentialpsi(double m,double * args){
  double a= *args++;
  double a4= *args;
  double mpa= m+a;
  return - a4 / mpa / mpa;
}
double TriaxialHernquistPotentialmdens(double m,double * args){
  double a= *args++;
  double a4= *args;
  double mpa= m+a;
  return a4 * pow ( mpa , -3 ) / m;
}
// LCOV_EXCL_START
double TriaxialHernquistPotentialmdensDeriv(double m,double * args){
  double a= *args++;
  double a4= *args;
  double mpa= m+a;
  return - a4 * ( a + 4. * m ) / m / m  * pow ( mpa , -4 ) ;
}
// LCOV_EXCL_STOP
//TriaxialJaffePotential
// Special case of EllipsoidalPotential, only need to define three functions
double TriaxialJaffePotentialpsi(double m,double * args){
  double a= *args++;
  double a2= *args;
  double mpa= m+a;
  return 2. * a2 * ( a / mpa + log ( m / mpa ) );
}
double TriaxialJaffePotentialmdens(double m,double * args){
  double a= *args++;
  double a2= *args;
  double mpa= m+a;
  return a2 * a2 * pow ( m * mpa , -2 );
}
// LCOV_EXCL_START
double TriaxialJaffePotentialmdensDeriv(double m,double * args){
  double a= *args++;
  double a2= *args;
  double mpa= m+a;
  return - 2. * a2 * a2 * ( a + 2. * m ) * pow ( m * mpa , -3 );
}
// LCOV_EXCL_STOP
//TriaxialNFWPotential
// Special case of EllipsoidalPotential, only need to define three functions
double TriaxialNFWPotentialpsi(double m,double * args){
  double a= *args++;
  double a3= *args;
  double mpa= m+a;
  return - 2. * a3 / mpa;
}
double TriaxialNFWPotentialmdens(double m,double * args){
  double a= *args++;
  double a3= *args;
  double mpa= m+a;
  return a3 / m / mpa / mpa;
}
// LCOV_EXCL_START
double TriaxialNFWPotentialmdensDeriv(double m,double * args){
  double a= *args++;
  double a3= *args;
  double mpa= m+a;
  return - a3 * ( a + 3. * m ) / m / m  * pow ( mpa , -3 ) ;
}
// LCOV_EXCL_STOP
//TwoPowerTriaxialPotential
// Special case of EllipsoidalPotential, only need to define three functions
double TwoPowerTriaxialPotentialpsi(double m,double * args){
  // args: a, alpha, beta, betaminusalpha, twominusalpha, threeminusalpha, psi_inf
  double a= *args++;
  double alpha= *args++;
  double beta= *args++;
  double betaminusalpha= *args++;
  double twominusalpha= *args++;
  double threeminusalpha= *args++;
  double psi_inf= *args;

  double a2= a * a;
  double moa, aom;

  if ( fabs(twominusalpha) < 1e-10 ) {
    // Special case: twominusalpha == 0
    aom= a / m;
    return -2. * a2 * pow(aom, betaminusalpha) / betaminusalpha
           * hyp2f1(betaminusalpha, betaminusalpha, betaminusalpha + 1., -aom);
  } else {
    // General case
    moa= m / a;
    return -2. * a2 * ( psi_inf
           - pow(moa, twominusalpha) / twominusalpha
           * hyp2f1(twominusalpha, betaminusalpha, threeminusalpha, -moa) );
  }
}
double TwoPowerTriaxialPotentialmdens(double m,double * args){
  // args: a, alpha, beta, betaminusalpha, twominusalpha, threeminusalpha, psi_inf
  double a= *args++;
  double alpha= *args++;
  double beta= *args++;
  double betaminusalpha= *args;  // We don't need the rest

  double mpa= m / a + 1.0;
  return pow(a / m, alpha) / pow(mpa , betaminusalpha);
}
// LCOV_EXCL_START
double TwoPowerTriaxialPotentialmdensDeriv(double m,double * args){
  // args: a, alpha, beta, betaminusalpha, twominusalpha, threeminusalpha, psi_inf
  double a= *args++;
  double alpha= *args++;
  double beta= *args++;
  double betaminusalpha= *args;

  double mpa= m + a;
  double mdens= pow(a / m, alpha) / pow(mpa / a, betaminusalpha);
  return -mdens * (a * alpha + beta * m) / m / mpa;
}
// LCOV_EXCL_STOP
