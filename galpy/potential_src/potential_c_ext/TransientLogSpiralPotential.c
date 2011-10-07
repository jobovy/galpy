#include <math.h>
#include <galpy_potentials.h>
//TransientLogSpiralPotential
//
double TransientLogSpiralPotentialRforce(double R,double phi,double t,
					 int nargs, double *args){
  //Get args
  double amp= *args++;
  double A= *args++;
  double to= *args++;
  double sigma2= *args++;
  double alpha= *args++;
  double m= *args++;
  double omegas= *args++;
  double gamma= *args++;
  //Calculate Rforce
  return amp * A * exp(-pow(t-to,2.)/2./sigma2) / R 
    * sin(alpha*log(R)+m*(phi-omegas*t-gamma));
}
double TransientLogSpiralPotentialphiforce(double R,double phi,double t,
					   int nargs, double *args){
  //Get args
  double amp= *args++;
  double A= *args++;
  double to= *args++;
  double sigma2= *args++;
  double alpha= *args++;
  double m= *args++;
  double omegas= *args++;
  double gamma= *args++;
  //Calculate phiforce
  return amp * A * exp(-pow(t-to,2.)/2./sigma2) / alpha * m 
    * sin(alpha*log(R)+m*(phi-omegas*t-gamma));
}
