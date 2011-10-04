#include <galpy_potentials.h>
//DehnenBarPotential
//
double DehnenBarPotentialRforce(double R,double phi,double t,
				      int nargs, double *args){
  //Get args
  double amp= *args++;
  //Calculate Rforce
  return 0.;
}
double DehnenBarPotentialphiforce(double R,double phi,double t,
				      int nargs, double *args){
  //Get args
  double amp= *args++;
  //Calculate phiforce
  return 0.;
}
