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
double DoubleExponentialDiskPotentialR2deriv(double R,double z, double phi,
					     double t,
					     struct potentialArg * potentialArgs){
  // Port of DoubleExponentialDiskPotential._R2deriv: Ogata/Hankel quadrature
  // using BOTH the J0 and J1 nodes/weights (same nodes as the C forces).
  double x;
  double * args= potentialArgs->args;
  //Get args
  double amp= *args; // true amp (args[1] already folds in -4*pi*alpha for the forces)
  double alpha= *(args+2);
  double beta= *(args+3);
  int de_n= (int) *(args+4);
  double * de_j0_xs= args + 5;
  double * de_j1_xs= args + 5 +     de_n;
  double * de_j0_ws= args + 5 + 2 * de_n;
  double * de_j1_ws= args + 5 + 3 * de_n;
  double alpha2= alpha * alpha;
  double beta2= beta * beta;
  double fz= fabs(z);
  double ebetafz= exp( - beta * fz );
  double out= 0;
  double prev_term= 1;
  int ii= 0;
  // J0 sum: + xn^2 * (alpha^2+x^2)^-1.5 * (beta*exp(-x*|z|)-x*exp(-beta*|z|))/(beta^2-x^2)
  while ( fabs(prev_term) > 1e-15 && ii < de_n ) {
    x= *(de_j0_xs+ii) / R;
    prev_term= *(de_j0_ws+ii) * *(de_j0_xs+ii) * *(de_j0_xs+ii)	\
      * pow( alpha2 + x * x , -1.5 )				\
      * ( beta * exp( -x * fz ) - x * ebetafz )			\
      / ( beta2 - x * x );
    out+= prev_term;
    prev_term/= out;
    ii+= 1;
  }
  // J1 sum: - xn * (alpha^2+x^2)^-1.5 * (beta*exp(-x*|z|)-x*exp(-beta*|z|))/(beta^2-x^2)
  prev_term= 1;
  ii= 0;
  while ( fabs(prev_term) > 1e-15 && ii < de_n ) {
    x= *(de_j1_xs+ii) / R;
    prev_term= - *(de_j1_ws+ii) * *(de_j1_xs+ii)	\
      * pow( alpha2 + x * x , -1.5 )			\
      * ( beta * exp( -x * fz ) - x * ebetafz )		\
      / ( beta2 - x * x );
    out+= prev_term;
    prev_term/= out;
    ii+= 1;
  }
  return 4. * M_PI * amp * alpha / R / R / R * out;
}
double DoubleExponentialDiskPotentialPlanarR2deriv(double R,double phi,
						   double t,
						   struct potentialArg * potentialArgs){
  // Planar (in-plane, z=0) d^2Phi/dR^2: the full 3D R2deriv evaluated at z=0.
  return DoubleExponentialDiskPotentialR2deriv(R,0.,phi,t,potentialArgs);
}
double DoubleExponentialDiskPotentialz2deriv(double R,double z, double phi,
					     double t,
					     struct potentialArg * potentialArgs){
  // Port of DoubleExponentialDiskPotential._z2deriv: Ogata/Hankel quadrature
  // over the J0 nodes/weights.
  double x;
  double * args= potentialArgs->args;
  //Get args
  double amp= *args; // true amp (args[1] already folds in -4*pi*alpha for the forces)
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
    prev_term= *(de_j0_ws+ii) * pow( alpha2 + x * x , -1.5 ) * x	\
      * ( x * exp( -x * fz ) - beta * ebetafz )			\
      / ( beta2 - x * x );
    out+= prev_term;
    prev_term/= out;
    ii+= 1;
  }
  return -4. * M_PI * amp * alpha * beta / R * out;
}
double DoubleExponentialDiskPotentialRzderiv(double R,double z, double phi,
					     double t,
					     struct potentialArg * potentialArgs){
  // Port of DoubleExponentialDiskPotential._Rzderiv: Ogata/Hankel quadrature
  // over the J1 nodes/weights, with the z>0 / z<=0 sign flip.
  double x;
  double * args= potentialArgs->args;
  //Get args
  double amp= *args; // true amp (args[1] already folds in -4*pi*alpha for the forces)
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
    prev_term= *(de_j1_ws+ii) * pow( alpha2 + x * x , -1.5 ) * x * x	\
      * ( exp( -x * fz ) - ebetafz )					\
      / ( beta2 - x * x );
    out+= prev_term;
    prev_term/= out;
    ii+= 1;
  }
  if ( z > 0. )
    return -4. * M_PI * amp * alpha * beta / R * out;
  else
    return  4. * M_PI * amp * alpha * beta / R * out;
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
