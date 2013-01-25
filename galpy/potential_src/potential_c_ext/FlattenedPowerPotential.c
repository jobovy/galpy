#include <math.h>
#include <galpy_potentials.h>
//FlattenedPowerPotential
//4 arguments: amp, alpha, q^2, and core^2
double FlattenedPowerPotentialEval(double R,double Z, double phi,
				   double t,
				   struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  //Get args
  double amp= *args;
  double alpha= *(args+1);
  double q2= *(args+2);
  double core2= *(args+3);
  double m2;
  //Calculate potential
  if ( alpha == 0. ) 
    return 0.5 * amp * log(R*R+Z*Z/q2+core2);
  else {
    m2= core2+R*R+Z*Z/q2;
    return - amp * pow(m2,-0.5 * alpha) / alpha;
  }
}
double FlattenedPowerPotentialRforce(double R,double Z, double phi,
				     double t,
				     struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  //Get args
  double amp= *args;
  double alpha= *(args+1);
  double q2= *(args+2);
  double core2= *(args+3);
  double m2;
  //Calculate potential
  if ( alpha == 0. ) 
    return - amp * R/(R*R+Z*Z/q2+core2);
  else {
    m2= core2+R*R+Z*Z/q2;
    return - amp * pow(m2,-0.5 * alpha-1.) * R;
  }
}
double FlattenedPowerPotentialPlanarRforce(double R,double phi,
					   double t,
					   struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  //Get args
  double amp= *args;
  double alpha= *(args+1);
  double core2= *(args+2);
  double m2;
  //Calculate potential
  if ( alpha == 0. ) 
    return - amp * R/(R*R+core2);
  else {
    m2= core2+R*R;
    return - amp * pow(m2,-0.5 * alpha - 1.) * R;
  }
}
double FlattenedPowerPotentialzforce(double R,double Z, double phi,
				     double t,
				     struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  //Get args
  double amp= *args;
  double alpha= *(args+1);
  double q2= *(args+2);
  double core2= *(args+3);
  double m2;
  //Calculate potential
  if ( alpha == 0. ) 
    return -amp * Z/q2/(R*R+Z*Z/q2+core2);
  else {
    m2= core2+R*R+Z*Z/q2;
    return - amp * pow(m2,-0.5 * alpha - 1.) * Z / q2;
  }
}
double FlattenedPowerPotentialPlanarR2deriv(double R,double phi,
					    double t,
					    struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  //Get args
  double amp= *args;
  double alpha= *(args+1);
  double core2= *(args+2);
  double m2;
  //Calculate potential
  if ( alpha == 0. ) 
    return amp * (1.- 2.*R*R/(R*R+core2))/(R*R+core2);
  else {
    m2= core2+R*R;
    return - amp * pow(m2,-0.5 * alpha - 1.) * ( (alpha + 1.) * R*R/m2 -1.);
  }
}
