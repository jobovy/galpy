#include <galpy_potentials.h>
//LogarithmicHaloPotential
//2 arguments: q and c2
double LogarithmicHaloPotentialRforce(double R,double Z, double phi,
				      int nargs, double *args){
  //Get args
  double amp= *args++;
  double q= *args++;
  double c= *args--;
  //Calculate Rforce
  double zq= Z/q;
  return - amp * R/(R*R+zq*zq+c);
}
double LogarithmicHaloPotentialPlanarRforce(double R,double phi,
					    int nargs, double *args){
  //Get args
  double amp= *args++;
  double c= *args;
  //Calculate Rforce
  return -amp * R/(R*R+c);
}
double LogarithmicHaloPotentialzforce(double R,double z,double phi,
				      int nargs, double *args){
  //Get args
  double amp= *args++;
  double q= *args++;
  double c= *args--;
  //Calculate zforce
  double zq= z/q;
  return -amp * z/q/q/(R*R+zq*zq+c);
}
