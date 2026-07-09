#include <math.h>
#include <galpy_potentials.h>
//KGPotential: 4 parameters: amp, K, D^2, 2F
//Routines to evaluate forces etc. for a verticalPotential, i.e., a 1D
//potential derived from a 3D potential as the z potential at a given (R,phi)
double KGPotentialLinearForce(double x, double t,
			      struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  //double amp= *args;
  //double K= *(args+1);
  //double D2= *(args+2);
  //double F= *(args+3);
  return - *args * x * ( *(args+1) / sqrt ( x * x + *(args+2) ) + *(args+3) );
}
// d^2 Phi / dx^2 = amp * ( K D^2 / (x^2+D^2)^(3/2) + 2F ) (args+3 holds 2F);
// differentiating -Force = amp x (K/sqrt(x^2+D^2)+2F).
double KGPotentialLinear2deriv(double x, double t,
			       struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  return *args * ( *(args+1) * *(args+2) / pow( x * x + *(args+2), 1.5 )
		   + *(args+3) );
}
