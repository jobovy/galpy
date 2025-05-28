#include <math.h>
#include <galpy_potentials.h>
//CosmphiDiskPotential
double CosmphiDiskPotentialRforce(double R,double phi,double t,
				   struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  //Get args
  double amp= *args++;
  double mphio= *args++;
  double p= *args++;
  double mphib= *args++;
  int m= (int) *args++;
  double rb= *args++;
  double rb2p= *(args+1);
  if ( R <= rb )
    return -amp * p * mphio / m * rb2p / pow(R,p+1.) * cos( m * phi - mphib);
  else
    return -amp * p * mphio / m * pow(R,p-1.) * cos( m * phi - mphib);
}
double CosmphiDiskPotentialphitorque(double R,double phi,double t,
				     struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  //Get args
  double amp= *args++;
  double mphio= *args++;
  double p= *args++;
  double mphib= *args++;
  int m= (int) *args++;
  double rb= *args++;
  double rbp= *args++;
  double r1p= *(args+1);
  if ( R <= rb )
    return amp * mphio * rbp * ( 2. * r1p - rbp / pow(R,p) )\
      * sin( m * phi-mphib);
  else
    return amp * mphio * pow(R,p) * sin( m * phi-mphib);
}
double CosmphiDiskPotentialR2deriv(double R,double phi,double t,
				    struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  //Get args
  double amp= *args++;
  double mphio= *args++;
  double p= *args++;
  double mphib= *args++;
  int m= (int) *args++;
  double rb= *args++;
  double rb2p= *(args+1);
  if ( R <= rb )
    return -amp * p * ( p + 1 ) * mphio / m * rb2p / pow(R,p+2.) \
      * cos( m * phi - mphib);
  else
    return amp * p * ( p - 1) * mphio / m * pow(R,p-2.) * cos(m * phi - mphib);
}
double CosmphiDiskPotentialphi2deriv(double R,double phi,double t,
				      struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  //Get args
  double amp= *args++;
  double mphio= *args++;
  double p= *args++;
  double mphib= *args++;
  int m= (int) *args++;
  double rb= *args++;
  double rbp= *args++;
  double r1p= *(args+1);
  if ( R <= rb )
    return - amp * m * mphio * rbp * ( 2. * r1p - rbp / pow(R,p) )\
      * cos( m * phi-mphib);
  else
    return - amp * m * mphio * pow(R,p) * cos( m * phi - mphib );
}
double CosmphiDiskPotentialRphideriv(double R,double phi,double t,
				      struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  //Get args
  double amp= *args++;
  double mphio= *args++;
  double p= *args++;
  double mphib= *args++;
  int m= (int) *args++;
  double rb= *args++;
  double rb2p= *(args+1);
  if ( R <= rb )
    return - amp * p * mphio / m * rb2p / pow(R,p+1) * sin( m * phi-mphib);
  else
    return - amp * p * mphio * pow(R,p-1.) * sin( m * phi - mphib );
}
