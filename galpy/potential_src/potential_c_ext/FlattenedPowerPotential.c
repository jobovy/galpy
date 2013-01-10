#include <math.h>
#include <galpy_potentials.h>
//FlattenedPowerPotential
//4 arguments: amp, alpha, q^2, and core^2
double FlattenedPowerPotentialEval(double R,double Z, double phi,
				   double t,
				   int nargs, double *args){
  //Get args
  double amp= *args;
  double alpha= *(args+1);
  double q2= *(args+2);
  double core2= *(args+3);
  double m2;
  //Calculate potential
  if ( alpha == 0. ) 
    return 0.5 * amp * log(R*R+Z*Z/q2+core2);
  else {
    m2= core2+R*R+Z*Z/q2;
    return - amp * pow(m2,-0.5 * alpha) / alpha;
  }
}
