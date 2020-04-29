#include <stdbool.h>
#include <math.h>
#include <galpy_potentials.h>
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
