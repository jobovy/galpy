///////////////////////////////////////////////////////////////////////////////
//
//   DiskSCFPotential.c:
//
//      The C implementation of DiskSCFPotential uses the fact that
//      (a) the SCF part of the potential can be gotten directly from the SCF
//          implementation
//      (b) the approximation part can be written as a sum over pairs
//          [Sigma_i(R),h_i(z)]
//      Thus, this file only implements the potential and forces coming from
//      a single of the approximation pairs; the entire potential is obtained
//      by summing all of these (by treating them as separate instances of the
//      potential) and adding the SCF; this is all handled through the parsing
//      of the potential in the Python code.
//
//      Surface-density profile is passed by type:
//
//         0= exponential: amp x exp(-R/h)
//         1= exponential w/ hole: amp x exp(-Rhole/R-R/h)
//
//      Vertical profile is passed by type:
//
//         0= exponential: exp(-|z|/h)/[2h]
//         1= sech2: sech^2(z/[2h])/[4h]
//
///////////////////////////////////////////////////////////////////////////////
#include <math.h>
#include <galpy_potentials.h>
#ifndef M_LN2
#define M_LN2 0.693147180559945309417232121
#endif
//DiskSCFPotential
//Only the part coming from a single approximation pair
// Arguments: nsigma_args,sigma_type,sigma_amp,sigma_h[,sigma_rhole],
//            hz_type,hz_h
double Sigma(double R,double * Sigma_args){
  int Sigma_type= (int) *Sigma_args;
  switch ( Sigma_type ) {
  case 0: // Pure exponential
    return *(Sigma_args+1) * exp(-R / *(Sigma_args+2) );
  case 1: // Exponential with central hole
    return *(Sigma_args+1) * exp(- *(Sigma_args+3) / R - R / *(Sigma_args+2) );
  }
  return -1; // LCOV_EXCL_LINE
}
double dSigmadR(double R,double * Sigma_args){
  int Sigma_type= (int) *Sigma_args;
  switch ( Sigma_type ) {
  case 0: // Pure exponential
    return - *(Sigma_args+1) * exp(-R / *(Sigma_args+2) ) / *(Sigma_args+2);
  case 1: // Exponential with central hole
    return *(Sigma_args+1) * ( *(Sigma_args+3) / R / R - 1. / *(Sigma_args+2))\
      * exp(- *(Sigma_args+3) / R - R / *(Sigma_args+2) );
  }
  return -1; // LCOV_EXCL_LINE
}
double d2SigmadR2(double R,double * Sigma_args){
  int Sigma_type= (int) *Sigma_args;
  switch ( Sigma_type ) {
  case 0: // Pure exponential
    return *(Sigma_args+1) * exp(-R / *(Sigma_args+2) ) \
      / *(Sigma_args+2) / *(Sigma_args+2);
  case 1: // Exponential with central hole
    return *(Sigma_args+1) * ( pow( *(Sigma_args+3) / R / R	\
				    - 1. / *(Sigma_args+2) , 2 ) \
			       -2. * *(Sigma_args+3) / R / R / R )\
      * exp(- *(Sigma_args+3) / R - R / *(Sigma_args+2) );
  }
  return -1; // LCOV_EXCL_LINE
}
double hz(double z,double * hz_args){
  int hz_type= (int) *hz_args;
  double fz;
  switch ( hz_type ) {
  case 0: // exponential
    fz= fabs(z);
    return 0.5 * exp ( - fz /  *(hz_args+1) ) / *(hz_args+1);
  case 1: // sech2
    return 0.25 * pow ( cosh ( 0.5 * z / *(hz_args+1) ) , -2 ) \
      / *(hz_args+1);
  }
  return -1; // LCOV_EXCL_LINE
}
double Hz(double z,double * hz_args){
  int hz_type= (int) *hz_args;
  double fz= fabs(z);
  switch ( hz_type ) {
  case 0: // exponential
    return 0.5 * ( exp ( - fz / *(hz_args+1) ) - 1. + fz / *(hz_args+1) ) \
      * *(hz_args+1);
  case 1: // sech2
    return *(hz_args+1) * ( log ( 1. + exp ( - fz / *(hz_args+1) ) )	\
			    + 0.5 * fz / *(hz_args+1)  - M_LN2 );
  }
  return -1; // LCOV_EXCL_LINE
}
double dHzdz(double z,double * hz_args){
  int hz_type= (int) *hz_args;
  double fz;
  switch ( hz_type ) {
  case 0: // exponential
    fz= fabs(z);
    return 0.5 * copysign ( 1. - exp ( - fz / *(hz_args+1) ) , z);
  case 1: // sech2
    return 0.5 * tanh ( 0.5 * z / *(hz_args+1) );
  }
  return -1; // LCOV_EXCL_LINE
}
double DiskSCFPotentialEval(double R,double Z, double phi,
			    double t,
			    struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  //Get args
  int nsigma_args= (int) *args;
  double * Sigma_args= args+1;
  double * hz_args= args+1+nsigma_args;
  //Calculate Rforce
  double r= sqrt( R * R + Z * Z );
  return Sigma(r,Sigma_args) * Hz(Z,hz_args);
}
double DiskSCFPotentialRforce(double R,double Z, double phi,
			      double t,
			      struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  //Get args
  int nsigma_args= (int) *args;
  double * Sigma_args= args+1;
  double * hz_args= args+1+nsigma_args;
  //Calculate Rforce
  double r= sqrt( R * R + Z * Z );
  return -dSigmadR(r,Sigma_args) * Hz(Z,hz_args) * R / r;
}
double DiskSCFPotentialPlanarRforce(double R,double phi,
				    double t,
				    struct potentialArg * potentialArgs){
  //Supposed to be zero (bc H(0) supposed to be zero), but just to make sure
  double * args= potentialArgs->args;
  //Get args
  int nsigma_args= (int) *args;
  double * Sigma_args= args+1;
  double * hz_args= args+1+nsigma_args;
  //Calculate Rforce
  return -dSigmadR(R,Sigma_args) * Hz(0.,hz_args);
}
double DiskSCFPotentialzforce(double R,double Z, double phi,
			      double t,
			      struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  //Get args
  int nsigma_args= (int) *args;
  double * Sigma_args= args+1;
  double * hz_args= args+1+nsigma_args;
  //Calculate Rforce
  double r= sqrt( R * R + Z * Z );
  return -dSigmadR(r,Sigma_args) * Hz(Z,hz_args) * Z / r \
    -Sigma(r,Sigma_args) * dHzdz(Z,hz_args);
}
double DiskSCFPotentialDens(double R,double Z, double phi,
			    double t,
			    struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  //Get args
  int nsigma_args= (int) *args;
  double * Sigma_args= args+1;
  double * hz_args= args+1+nsigma_args;
  //Calculate Rforce
  double r= sqrt( R * R + Z * Z );
  return M_1_PI / 4. * (Sigma(r,Sigma_args) * hz(Z,hz_args)
			+ d2SigmadR2(r,Sigma_args) * Hz(Z,hz_args)
			+ 2. / r * dSigmadR(r,Sigma_args)	\
			    * ( Hz(Z,hz_args) + Z * dHzdz(Z,hz_args) ) );
  }
