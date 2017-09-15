#include <math.h>
#include <galpy_potentials.h>
//CosmphiDiskPotential
double CosmphiDiskPotentialRforce(double R,double phi,double t,
				   struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  //Get args
  double amp= *args++;
  double mphio= *args++;
  double p= *args++;
  double mphib= *args++;
  int m= (int) *args;
  return -amp * p * mphio / m * pow(R,p-1.) * cos( m * phi - mphib);
}
double CosmphiDiskPotentialphiforce(double R,double phi,double t,
				     struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  //Get args
  double amp= *args++;
  double mphio= *args++;
  double p= *args++;
  double mphib= *args++;
  int m= (int) *args;
  return amp * mphio * pow(R,p) * sin( m * phi-mphib);
}
double CosmphiDiskPotentialR2deriv(double R,double phi,double t,
				    struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  //Get args
  double amp= *args++;
  double mphio= *args++;
  double p= *args++;
  double mphib= *args++;
  int m= (int) *args;
  return amp * p * ( p - 1) * mphio / m * pow(R,p-2.) * cos(m * phi - mphib);
} 
double CosmphiDiskPotentialphi2deriv(double R,double phi,double t,
				      struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  //Get args
  double amp= *args++;
  double mphio= *args++;
  double p= *args++;
  double mphib= *args++;
  int m= (int) *args;
  return - amp * m * mphio * pow(R,p) * cos( m * phi - mphib );
} 
double CosmphiDiskPotentialRphideriv(double R,double phi,double t,
				      struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  //Get args
  double amp= *args++;
  double mphio= *args++;
  double p= *args++;
  double mphib= *args++;
  int m= (int) *args;
  return - amp * p * mphio * pow(R,p-1.) * sin( m * phi - mphib );
} 
