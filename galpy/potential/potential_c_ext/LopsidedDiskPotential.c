#include <math.h>
#include <galpy_potentials.h>
//LopsidedDiskPotential
double LopsidedDiskPotentialRforce(double R,double phi,double t,
				   struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  //Get args
  double amp= *args++;
  double phio= *args++;
  double p= *args++;
  double phib= *args;
  //Calculate Rforce
  return -amp * p * phio * pow(R,p-1.) * cos( phi - phib);
}
double LopsidedDiskPotentialphitorque(double R,double phi,double t,
				     struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  //Get args
  double amp= *args++;
  double phio= *args++;
  double p= *args++;
  double phib= *args;
  //Calculate phitorque
  return amp * phio * pow(R,p) * sin( phi-phib);
}
double LopsidedDiskPotentialR2deriv(double R,double phi,double t,
				    struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  //Get args
  double amp= *args++;
  double phio= *args++;
  double p= *args++;
  double phib= *args;
  //Calculate Rforce
  return amp * p * ( p - 1) * phio * pow(R,p-2.) * cos(phi - phib);
}
double LopsidedDiskPotentialphi2deriv(double R,double phi,double t,
				      struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  //Get args
  double amp= *args++;
  double phio= *args++;
  double p= *args++;
  double phib= *args;
  //Calculate Rforce
  return - amp * phio * pow(R,p) * cos( phi - phib );
}
double LopsidedDiskPotentialRphideriv(double R,double phi,double t,
				      struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  //Get args
  double amp= *args++;
  double phio= *args++;
  double p= *args++;
  double phib= *args;
  //Calculate Rforce
  return - amp * p * phio * pow(R,p-1.) * sin( phi - phib );
}
