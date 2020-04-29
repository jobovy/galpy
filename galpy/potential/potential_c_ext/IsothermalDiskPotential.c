#include <math.h>
#include <galpy_potentials.h>
//IsothermalDiskPotential: 2 parameters: amp = (real amp) * sigma2 / H, 2H
double IsothermalDiskPotentialLinearForce(double x, double t,
					  struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  return - *args * tanh ( x / *(args+1) );
}
