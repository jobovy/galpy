#include <math.h>
#include <galpy_potentials.h>
//IsochronePotential
//2  arguments: amp, b
double IsochronePotentialEval(double R,double Z, double phi,
			      double t,
			      struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  //Get args
  double amp= *args;
  double b= *(args+1);
  //Calculate potential
  double r2= R*R+Z*Z;
  return -amp / ( b + sqrt(r2 + b * b) );
}
double IsochronePotentialRforce(double R,double Z, double phi,
				      double t,
				      struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  //Get args
  double amp= *args;
  double b= *(args+1);
  //Calculate Rforce
  double r2= R*R+Z*Z;
  double rb= sqrt(r2 + b * b);
  return - amp * R / rb * pow(b + rb,-2.);
}
double IsochronePotentialPlanarRforce(double R,double phi,
					    double t,
					    struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  //Get args
  double amp= *args;
  double b= *(args+1);
  //Calculate Rforce
  double r2= R*R;
  double rb= sqrt(r2 + b * b);
  return - amp * R / rb * pow(b + rb,-2.);
}
double IsochronePotentialzforce(double R,double Z,double phi,
				double t,
				struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  //Get args
  double amp= *args;
  double b= *(args+1);
  //Calculate zforce
  double r2= R*R+Z*Z;
  double rb= sqrt(r2 + b * b);
  return - amp * Z / rb * pow(b + rb,-2.);
}
double IsochronePotentialPlanarR2deriv(double R,double phi,
					     double t,
					     struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  //Get args
  double amp= *args;
  double b= *(args+1);
  //Calculate Rforce
  double r2= R*R;
  double rb= sqrt(r2 + b * b);
  return - amp * ( -pow(b,3.) - b * b * rb + 2. * r2 * rb ) * pow(rb * ( b + rb ),-3.);
}
