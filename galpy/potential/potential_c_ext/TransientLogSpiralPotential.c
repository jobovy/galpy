#include <math.h>
#include <galpy_potentials.h>
//TransientLogSpiralPotential
//
double TransientLogSpiralPotentialRforce(double R,double phi,double t,
					 struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
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
    * sin(alpha*log(R)-m*(phi-omegas*t-gamma));
}
double TransientLogSpiralPotentialphitorque(double R,double phi,double t,
					   struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  //Get args
  double amp= *args++;
  double A= *args++;
  double to= *args++;
  double sigma2= *args++;
  double alpha= *args++;
  double m= *args++;
  double omegas= *args++;
  double gamma= *args++;
  //Calculate phitorque
  return -amp * A * exp(-pow(t-to,2.)/2./sigma2) / alpha * m
    * sin(alpha*log(R)-m*(phi-omegas*t-gamma));
}
double TransientLogSpiralPotentialR2deriv(double R,double phi,double t,
					  struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  //declare
  double chi;
  //Get args
  double amp= *args++;
  double A= *args++;
  double to= *args++;
  double sigma2= *args++;
  double alpha= *args++;
  double m= *args++;
  double omegas= *args++;
  double gamma= *args++;
  //Calculate R2deriv
  chi= alpha*log(R)-m*(phi-omegas*t-gamma);
  return amp * A * exp(-pow(t-to,2.)/2./sigma2) / R / R
    * ( sin(chi) - alpha * cos(chi) );
}
double TransientLogSpiralPotentialphi2deriv(double R,double phi,double t,
					    struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  //Get args
  double amp= *args++;
  double A= *args++;
  double to= *args++;
  double sigma2= *args++;
  double alpha= *args++;
  double m= *args++;
  double omegas= *args++;
  double gamma= *args++;
  //Calculate phi2deriv
  return -amp * A * exp(-pow(t-to,2.)/2./sigma2) / alpha * m * m
    * cos(alpha*log(R)-m*(phi-omegas*t-gamma));
}
double TransientLogSpiralPotentialRphideriv(double R,double phi,double t,
					    struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  //Get args
  double amp= *args++;
  double A= *args++;
  double to= *args++;
  double sigma2= *args++;
  double alpha= *args++;
  double m= *args++;
  double omegas= *args++;
  double gamma= *args++;
  //Calculate Rphideriv
  return amp * A * exp(-pow(t-to,2.)/2./sigma2) * m / R
    * cos(alpha*log(R)-m*(phi-omegas*t-gamma));
}
