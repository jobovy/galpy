#include <galpy_potentials.h>
//Dummy zero forces
double ZeroPlanarForce(double R, double phi,
		       int nargs, double *args){
  return 0.;
}
double ZeroForce(double R, double Z, double phi,
		 int nargs, double *args){
  return 0.;
}
