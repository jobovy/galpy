#include <galpy_potentials.h>
//Dummy zero forces
double ZeroPlanarForce(double R, double phi,double t,
		       struct potentialArg * potentialArgs){
  return 0.;
}
double ZeroForce(double R, double Z, double phi,double t,
		 struct potentialArg * potentialArgs){
  return 0.;
}
