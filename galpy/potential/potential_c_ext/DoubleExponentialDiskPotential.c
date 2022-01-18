#include <math.h>
#include <galpy_potentials.h>
//Double exponential disk potential
double DoubleExponentialDiskPotentialEval(double R,double z, double phi,
					  double t,
					  struct potentialArg * potentialArgs){
  double x;
  double * args= potentialArgs->args;
  //Get args
  double amp= *(args+1);
  double alpha= *(args+2);
  double beta= *(args+3);
  int de_n= (int) *(args+4);
  double * de_j0_xs= args + 5;
  double * de_j0_ws= args + 5 + 2 * de_n;
  double alpha2= alpha * alpha;
  double beta2= beta * beta;
  double fz= fabs(z);
  double ebetafz= exp( - beta * fz );
  double out= 0;
  double prev_term= 1;
  int ii= 0;
  while ( fabs(prev_term) > 1e-15 && ii < de_n ) {
    x= *(de_j0_xs+ii) / R;
    prev_term= *(de_j0_ws+ii) * pow( alpha2 + x * x , -1.5 )	\
      * ( beta * exp( -x * fz ) - x * ebetafz ) \
      / ( beta2 - x * x );
    out+= prev_term;
    prev_term/= out;
    ii+= 1;
  }
  return amp * out / R;
}
double DoubleExponentialDiskPotentialRforce(double R,double z, double phi,
					    double t,
					    struct potentialArg * potentialArgs){
  double x;
  double * args= potentialArgs->args;
  //Get args
  double amp= *(args+1);
  double alpha= *(args+2);
  double beta= *(args+3);
  int de_n= (int) *(args+4);
  double * de_j1_xs= args + 5 +     de_n;
  double * de_j1_ws= args + 5 + 3 * de_n;
  double alpha2= alpha * alpha;
  double beta2= beta * beta;
  double fz= fabs(z);
  double ebetafz= exp( - beta * fz );
  double out= 0;
  double prev_term= 1;
  int ii= 0;
  while ( fabs(prev_term) > 1e-15 && ii < de_n ) {
    x= *(de_j1_xs+ii) / R;
    prev_term= *(de_j1_ws+ii) * x * pow( alpha2 + x * x , -1.5) \
      * ( beta * exp(-x * fz )-x * ebetafz )\
      / ( beta2 - x * x);
    out+= prev_term;
    prev_term/= out;
    ii+= 1;
  }
  return amp * out / R;
}
double DoubleExponentialDiskPotentialPlanarRforce(double R,double phi,
						  double t,
						  struct potentialArg * potentialArgs){
  double x;
  double * args= potentialArgs->args;
  //Get args
  double amp= *(args+1);
  double alpha= *(args+2);
  double beta= *(args+3);
  int de_n= (int) *(args+4);
  double * de_j1_xs= args + 5 +     de_n;
  double * de_j1_ws= args + 5 + 3 * de_n;
  double alpha2= alpha * alpha;
  double out= 0;
  double prev_term= 1;
  int ii= 0;
  while ( fabs(prev_term) > 1e-15 && ii < de_n ) {
    x= *(de_j1_xs+ii) / R;
    prev_term= *(de_j1_ws+ii) * x * pow( alpha2 + x * x , -1.5) / ( beta + x );
    out+= prev_term;
    prev_term/= out;
    ii+= 1;
  }
  return amp * out / R;
}
double DoubleExponentialDiskPotentialzforce(double R,double z,double phi,
					    double t,
					    struct potentialArg * potentialArgs){
  double x;
  double * args= potentialArgs->args;
  //Get args
  double amp= *(args+1);
  double alpha= *(args+2);
  double beta= *(args+3);
  int de_n= (int) *(args+4);
  double * de_j0_xs= args + 5;
  double * de_j0_ws= args + 5 + 2 * de_n;
  double alpha2= alpha * alpha;
  double beta2= beta * beta;
  double fz= fabs(z);
  double ebetafz= exp(-beta * fabs(z) );
  double out= 0;
  double prev_term= 1;
  int ii= 0;
  while ( fabs(prev_term) > 1e-15 && ii < de_n ) {
    x= *(de_j0_xs+ii) / R;
    prev_term= *(de_j0_ws+ii) * pow( alpha2 + x * x , -1.5) \
      * x * ( exp(-x * fz ) - ebetafz )\
      /( beta2 - x * x );
    out+= prev_term;
    prev_term/= out;
    ii+= 1;
  }
  if ( z > 0. )
    return amp * out * beta / R;
  else
    return -amp * out * beta / R;
}
double DoubleExponentialDiskPotentialDens(double R,double z, double phi,
					  double t,
					  struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  //Get args
  double amp= *args;
  double alpha= *(args+2);
  double beta= *(args+3);
  // calculate density
  return amp * exp ( - alpha * R - beta * fabs ( z ) );
}
