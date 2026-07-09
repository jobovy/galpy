#include <math.h>
#include <galpy_potentials.h>
//IsothermalDiskPotential: 2 parameters: amp = (real amp) * sigma2 / H, 2H
double IsothermalDiskPotentialLinearForce(double x, double t,
					  struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  return - *args * tanh ( x / *(args+1) );
}
// d^2 Phi / dx^2 = amp / (2H) * sech^2(x/2H) = amp/args[1] * (1-tanh^2(x/args[1]))
// (args[0]=amp=sigma^2/H, args[1]=2H); differentiating -Force=amp tanh(x/2H).
double IsothermalDiskPotentialLinear2deriv(double x, double t,
					   struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  double tx= tanh ( x / *(args+1) );
  return *args / *(args+1) * ( 1. - tx * tx );
}
