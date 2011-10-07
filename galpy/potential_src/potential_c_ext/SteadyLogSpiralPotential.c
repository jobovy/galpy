#include <math.h>
#include <galpy_potentials.h>
//SteadyLogSpiralPotential
//
double SteadyLogSpiralPotentialRforce(double R,double phi,double t,
				      int nargs, double *args){
  //declare
  double smooth;
  //Get args
  double amp= *args++;
  double tform= *args++;
  double tsteady= *args++;
  double A= *args++;
  double alpha= *args++;
  double m= *args++;
  double omegas= *args++;
  double gamma= *args++;
  //Calculate Rforce
  smooth= dehnenSmooth(t,tform,tsteady);
  return amp * smooth * A / R * sin(alpha * log(R) + m * (phi-omegas*t-gamma));
}
double SteadyLogSpiralPotentialphiforce(double R,double phi,double t,
				      int nargs, double *args){
  //declare
  double smooth;
  //Get args
  double amp= *args++;
  double tform= *args++;
  double tsteady= *args++;
  double A= *args++;
  double alpha= *args++;
  double m= *args++;
  double omegas= *args++;
  double gamma= *args++;
  //Calculate Rforce
  smooth= dehnenSmooth(t,tform,tsteady);
  return amp * smooth * A / alpha * m * 
    sin(alpha * log(R) + m * (phi-omegas*t-gamma));
}
