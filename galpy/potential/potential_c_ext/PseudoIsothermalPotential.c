#include <math.h>
#include <galpy_potentials.h>
//PseudoIsothermalPotential
//2 arguments: amp, a
double PseudoIsothermalPotentialEval(double R,double Z, double phi,
				    double t,
				    struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  //Get args
  double amp= *args;
  double a= *(args+1);
  double a2= a*a;
  //Calculate potential
  double r2= R*R+Z*Z;
  double r= sqrt(r2);
  return amp * (0.5 * log(1 + r2 / a2) + a / r * atan(r / a)) / a;
}
double PseudoIsothermalPotentialRforce(double R,double Z, double phi,
				      double t,
				      struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  //Get args
  double amp= *args;
  double a= *(args+1);
  //Calculate potential
  double r2= R*R+Z*Z;
  double r= sqrt(r2);
  return - amp * (1. / r - a / r2 * atan(r / a)) / a * R / r;
}
double PseudoIsothermalPotentialPlanarRforce(double R,double phi,
					    double t,
					    struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  //Get args
  double amp= *args;
  double a= *(args+1);
  //Calculate potential
  return - amp * (1. / R - a / R / R * atan(R / a)) / a;
}
double PseudoIsothermalPotentialzforce(double R,double z,double phi,
				      double t,
				      struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  //Get args
  double amp= *args;
  double a= *(args+1);
  //Calculate potential
  double r2= R*R+z*z;
  double r= sqrt(r2);
  return - amp * (1. / r - a / r2 * atan(r / a)) / a * z / r;
}
double PseudoIsothermalPotentialPlanarR2deriv(double R,double phi,
					     double t,
					     struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  //Get args
  double amp= *args;
  double a= *(args+1);
  double a2= a*a;
  //Calculate potential
  double R2= R*R;
  return amp / R2 * (2. * a / R * atan(R / a) - ( 2. * a2 +  R2)/(a2 + R2) )
    / a;
}
