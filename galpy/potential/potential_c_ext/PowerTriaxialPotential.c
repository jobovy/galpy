#include <math.h>
#include <galpy_potentials.h>
// Special case of EllipsoidalPotential, only need to define three functions
double PowerTriaxialPotentialpsi(double m,double * args){
  double alpha= *args;
  double m2= m*m;
  return 2. / ( 2. - alpha ) * pow ( m2 , 1. - alpha / 2. );
}
double PowerTriaxialPotentialmdens(double m,double * args){
  double alpha= *args;
  double m2= m*m;
  return pow ( m2 , - alpha / 2. );
}
// LCOV_EXCL_START
double PowerTriaxialPotentialmdensDeriv(double m,double * args){
  double alpha= *args;
  double m2= m*m;
  return - alpha * pow ( m2 , ( -alpha - 1. ) / 2. );
}
// LCOV_EXCL_STOP
