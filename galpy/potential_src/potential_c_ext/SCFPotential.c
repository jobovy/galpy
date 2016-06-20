#include <math.h>
#include <galpy_potentials.h>
//SCF Disk potential
//4 arguments: amp, Acos, Asin, a


double SCFPotentialRforce(double R,double Z, double phi,
				double t,
				struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  //Get args
  double amp= *args++;
  double Acos= *args++;
  double Asin= *args++;
  double a= *args;
  //Calculate Rforce
 return 1.;
}

double SCFPotentialzforce(double R,double Z, double phi,
				double t,
				struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  //Get args
  double amp= *args++;
  double Acos= *args++;
  double Asin= *args++;
  double a= *args;
  //Calculate zforce
  return 1.;
}

double SCFPotentialphiforce(double R,double Z, double phi,
				double t,
				struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  //Get args
  double amp= *args++;
  double Acos= *args++;
  double Asin= *args++;
  double a= *args;
  //Calculate phiforce
  
  return 1.;
}
