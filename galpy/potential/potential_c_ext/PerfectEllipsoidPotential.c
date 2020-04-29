#include <math.h>
#include <galpy_potentials.h>
// Special case of EllipsoidalPotential, only need to define three functions
double PerfectEllipsoidPotentialpsi(double m,double * args){
  double a2= *args;
  double m2= m*m;
  return -1. / ( a2 + m2 );
}
double PerfectEllipsoidPotentialmdens(double m,double * args){
  double a2= *args;
  double m2= m*m;
  return 1. / ( a2 + m2 ) / ( a2 + m2 );
}
// LCOV_EXCL_START
double PerfectEllipsoidPotentialmdensDeriv(double m,double * args){
  double a2= *args;
  double m2= m*m;
  return -4. * m * pow ( a2 + m2 ,-3 );
}
// LCOV_EXCL_STOP
