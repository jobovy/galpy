#include <math.h>
#include <galpy_potentials.h>
//JaffePotential
//2 arguments: amp, a
double JaffePotentialEval(double R,double Z, double phi,
			  double t,
			  struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  //Get args
  double amp= *args++;
  double a= *args;
  //Calculate Rforce
  double sqrtRz= pow(R*R+Z*Z,0.5);
  return - amp * log ( 1. + a / sqrtRz ) / a;
}
double JaffePotentialRforce(double R,double Z, double phi,
			    double t,
			    struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  //Get args
  double amp= *args++;
  double a= *args;
  //Calculate Rforce
  double sqrtRz= pow(R*R+Z*Z,0.5);
  return - amp * R * pow( sqrtRz , -3. ) / ( 1. + a / sqrtRz );
}
double JaffePotentialPlanarRforce(double R,double phi,
				  double t,
				  struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  //Get args
  double amp= *args++;
  double a= *args;
  //Calculate Rforce
  return - amp * pow(R,-2.) / ( 1. + a / R );
}
double JaffePotentialzforce(double R,double Z,double phi,
			    double t,
			    struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  //Get args
  double amp= *args++;
  double a= *args;
  //Calculate Rforce
  double sqrtRz= pow(R*R+Z*Z,0.5);
  return - amp * Z * pow( sqrtRz , -3. ) / ( 1. + a / sqrtRz );
}
double JaffePotentialPlanarR2deriv(double R,double phi,
				   double t,
				   struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  //Get args
  double amp= *args++;
  double a= *args;
  //Calculate R2deriv
  return - amp * (a + 2. * R) * pow(R,-4.) * pow(1.+a/R,-2.);
}
