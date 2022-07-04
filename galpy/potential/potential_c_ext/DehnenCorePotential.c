#include <math.h>
#include "galpy_potentials.h"
//CoreDehnenPotential
//2 arguments: amp, a
double DehnenCoreSphericalPotentialEval(double R,double Z, double phi,
			      double t,
			      struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  double amp= *args++;
  double a= *args++;
  //Calculate Rforce
  double sqrtRz= pow(R*R+Z*Z,0.5);
  return -amp * (1. - pow(sqrtRz/(sqrtRz+a), 2.)) / (6. * a);
}

double DehnenCoreSphericalPotentialRforce(double R,double Z, double phi,
				double t,
				struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  double amp= *args++;
  double a= *args++;
  //Calculate Rforce
  double sqrtRz= pow(R*R+Z*Z,0.5);
  return - amp * (R / pow(a+sqrtRz,3.) / 3.);
}

double DehnenCoreSphericalPotentialPlanarRforce(double R,double phi,
					    double t,
				      struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  //Get args
  double amp= *args++;
  double a= *args++;
  //Calculate Rforce
  return - amp * (R / pow(a+R, 3.) / 3.);
}

double DehnenCoreSphericalPotentialzforce(double R,double Z,double phi,
				double t,
				struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  //Get args
  double amp= *args++;
  double a= *args++;
  //Calculate Rforce
  double sqrtRz= pow(R*R+Z*Z,0.5);
  return - amp * (Z / pow(a+sqrtRz, 3.) / 3.);
}

double DehnenCoreSphericalPotentialPlanarR2deriv(double R,double phi,
					     double t,
				       struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  //Get args
  double amp= *args++;
  double a= *args++;
  // return
  return -amp * (pow(a+R, -4.) * (2.*R - a)) / 3.;
}
double DehnenCoreSphericalPotentialDens(double R,double Z, double phi,
					double t,
					struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  double amp= *args++;
  double a= *args++;
  //Calculate Rforce
  double r= sqrt ( R * R + Z * Z );
  return amp * M_1_PI / 4. * pow ( 1. + r / a, -4.) * pow (a, - 3.);
}
