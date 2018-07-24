#include <math.h>
#include <galpy_potentials.h>
//HenonHeilesPotential
double HenonHeilesPotentialRforce(double R,double phi,double t,
				     struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  return - *args * R * (1. + R * sin( 3. * phi ) );
}
double HenonHeilesPotentialphiforce(double R,double phi,double t,
				    struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  //Calculate phiforce
  return - *args * pow(R,3.) * cos( 3. * phi );
}
double HenonHeilesPotentialR2deriv(double R,double phi,double t,
				      struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  return *args * ( 1. + 2. * R * sin( 3. * phi ) );
} 
double HenonHeilesPotentialphi2deriv(double R,double phi,double t,
					struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  return - 3. * *args * pow(R,3.) * sin( 3. * phi );
} 
double HenonHeilesPotentialRphideriv(double R,double phi,double t,
					struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  return 3. * *args * R * R * cos( 3. * phi );
} 
