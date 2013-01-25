#include <math.h>
#include <galpy_potentials.h>
//LogarithmicHaloPotential
//3 (2)  arguments: amp, c2, (and q)
double LogarithmicHaloPotentialEval(double R,double Z, double phi,
				    double t,
				    struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  //Get args
  double amp= *args;
  double q= *(args+1);
  double c= *(args+2);
  //Calculate potential
  double zq= Z/q;
  return 0.5 * amp * log(R*R+zq*zq+c);
}
double LogarithmicHaloPotentialRforce(double R,double Z, double phi,
				      double t,
				      struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  //Get args
  double amp= *args++;
  double q= *args++;
  double c= *args--;
  //Calculate Rforce
  double zq= Z/q;
  return - amp * R/(R*R+zq*zq+c);
}
double LogarithmicHaloPotentialPlanarRforce(double R,double phi,
					    double t,
					    struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  //Get args
  double amp= *args++;
  double c= *args;
  //Calculate Rforce
  return -amp * R/(R*R+c);
}
double LogarithmicHaloPotentialzforce(double R,double z,double phi,
				      double t,
				      struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  //Get args
  double amp= *args++;
  double q= *args++;
  double c= *args--;
  //Calculate zforce
  double zq= z/q;
  return -amp * z/q/q/(R*R+zq*zq+c);
}
double LogarithmicHaloPotentialPlanarR2deriv(double R,double phi,
					     double t,
					     struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  //Get args
  double amp= *args++;
  double c= *args;
  //Calculate Rforce
  return amp * (1.- 2.*R*R/(R*R+c))/(R*R+c);
}
