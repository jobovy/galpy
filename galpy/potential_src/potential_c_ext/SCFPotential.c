#include <math.h>
#include <galpy_potentials.h>
#include <stdio.h>

//SCF Disk potential
//4 arguments: amp, Acos, Asin, a


inline void cyl_to_spher(double R, double Z, double phi,double *r, double *theta){
  *r = sqrt(R*R + Z*Z);
  *theta = atan2(R, Z);
}


double SCFPotentialRforce(double R,double Z, double phi,
				double t,
				struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  //Get args
  double a = *args++;
  int N = *args++;
  int L = *args++;
  int M = *args++;
  double* Acos = args;
  double* Asin = args + N*L*M;

 return 1.;
}

double SCFPotentialzforce(double R,double Z, double phi,
				double t,
				struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  //Get args
  double a = *args++;
  int N = *args++;
  int L = *args++;
  int M = *args++;
  double* Acos = args;
  double* Asin = args + N*L*M;

  
  //Calculate zforce
  return 1.;
}

double SCFPotentialphiforce(double R,double Z, double phi,
				double t,
				struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  //Get args
  double a = *args++;
  int N = *args++;
  int L = *args++;
  int M = *args++;
  double* Acos = args;
  double* Asin = args + N*L*M;

  //Calculate phiforce
  
  return 1.;
}
