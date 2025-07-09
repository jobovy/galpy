#include <math.h>
// Constants not defined in MSVC's math.h
#ifndef M_SQRT1_2
#define M_SQRT1_2 0.70710678118654746172
#endif
#ifndef M_2_SQRTPI
#define M_2_SQRTPI 1.12837916709551255856
#endif
#include <gsl/gsl_spline.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_sf_expint.h>
#include <gsl/gsl_spline.h>
#include <galpy_potentials.h>
// FDMDynamicalFrictionForceFDMvsCDM: FDM force / CDM force
double FDMDynamicalFrictionForceFDMvsCDM(double R,double z,
						    double phi,double t,
						    double r2,
						    struct potentialArg * potentialArgs,
						    double vR,double vT,
						    double vz){
  double FDMfactor, CDMfactor, C_fdm, C_fdm_disp;
  double kr, sr, d_ind, X, Xfactor, GMvs, M_sigma, mu;
  double * args= potentialArgs->args;
  //Get args
  double amp= *args;
  double ms= *(args+9);
  double rhm= *(args+10);
  double gamma2= *(args+11);
  double lnLambda= *(args+12);
  double ro= *(args+14);
  double rf= *(args+15);
  double mhbar= *(args+16);
  double const_FDMfactor= *(args+17);
  double r= sqrt( r2 );
  double v2=  vR * vR + vT * vT + vz * vz;
  double v= sqrt( v2 );
  // Constant or variable Lambda
  if ( const_FDMfactor < 0 ) {
    GMvs= ms/v/v;
    if ( lnLambda < 0 ) {
      // lnLambda is constant
      if ( GMvs < rhm )
        lnLambda= 0.5 * log ( 1. + r2 / gamma2 / rhm / rhm );
      else
        lnLambda= 0.5 * log ( 1. + r2 / gamma2 / GMvs / GMvs );
    }

    // FDMfactor
    kr = 2.0 * mhbar * v * r;

    // CDMfactor
    d_ind= (r-ro)/(rf-ro);
    d_ind= d_ind <  0 ? 0. : ( d_ind > 1 ? 1. : d_ind);
    sr= gsl_spline_eval(*potentialArgs->spline1d,d_ind,*potentialArgs->acc1d);
    X= M_SQRT1_2 * v / sr;
    Xfactor= erf ( X ) - M_2_SQRTPI * X * exp ( - X * X );
    CDMfactor = lnLambda * Xfactor;

    M_sigma = v/sr; // Mach number

    if(kr<M_sigma)
    { // Zero-dispersion regime
      FDMfactor = M_EULER + log ( kr ) - gsl_sf_Ci(kr) + sin (kr) / (kr) - 1.0;
    }
    else if(kr>4* M_sigma)
    {
      // Dispersion regime
      FDMfactor = log(kr/M_sigma)*Xfactor;
    } else {
      // Intermediate regime between zero-velocity and dispersion regimes
      // Re-evaluating FDM zero-velocity regime at kr = M_sigma/2 and FDM dispersion regime at kr = 2 * M_sigma
      C_fdm = M_EULER + log ( M_sigma ) - gsl_sf_Ci( M_sigma ) + sin ( M_sigma ) / ( M_sigma ) - 1.0;
      C_fdm_disp = log( 4.0 ) * Xfactor;
      // Interpolating between zero-velocity and dispersion regimes
      mu = (2*M_sigma - kr/2)/(2*M_sigma-M_sigma/2);
      FDMfactor = mu * C_fdm + (1-mu) * C_fdm_disp;
    }

  } else {
    FDMfactor = const_FDMfactor;
  }
  return FDMfactor / CDMfactor;
}
double FDMDynamicalFrictionForceAmplitude(double R,double z,
						    double phi,double t,
						    double r2,
						    struct potentialArg * potentialArgs,
						    double vR,double vT,
						    double vz){
  double forceAmplitude, FDMvsCDMamplitude;
  // Compute the force amplitude for Chandrasekhar (CDM) dynamical friction
  double * args= potentialArgs->args;
  forceAmplitude = ChandrasekharDynamicalFrictionForceAmplitude(R,z,phi,t,r2,
                            potentialArgs->wrappedPotentialArg,vR,vT,vz);
  // Now compute the relative CDM vs FDM force amplitude
  FDMvsCDMamplitude = FDMDynamicalFrictionForceFDMvsCDM(R,z,phi,t,r2,
         potentialArgs,vR,vT,vz);
  if ( FDMvsCDMamplitude < 1.)
    forceAmplitude *= FDMvsCDMamplitude;
  // Caching
  *(args + 1)= R;
  *(args + 2)= z;
  *(args + 3)= phi;
  *(args + 4)= t;
  *(args + 5)= vR;
  *(args + 6)= vT;
  *(args + 7)= vz;
  *(args + 8)= forceAmplitude;
  return forceAmplitude;
}
double FDMDynamicalFrictionForceRforce(double R,double z, double phi,
						 double t,
						 struct potentialArg * potentialArgs,
						 double vR,double vT,
						 double vz){
  double forceAmplitude;
  double * args= potentialArgs->args;
  double r2=  R * R + z * z;
  if ( r2 < *(args+13) )  // r < minr, don't bother caching
    return 0.;
  //Get args
  double cached_R= *(args + 1);
  double cached_z= *(args + 2);
  double cached_phi= *(args + 3);
  double cached_t= *(args + 4);
  double cached_vR= *(args + 5);
  double cached_vT= *(args + 6);
  double cached_vz= *(args + 7);
  if ( R != cached_R || phi != cached_phi || z != cached_z || t != cached_t \
       || vR != cached_vR || vT != cached_vT || vz != cached_vz )
    forceAmplitude= FDMDynamicalFrictionForceAmplitude(R,z,phi,t,r2,
								 potentialArgs,
								 vR,vT,vz);
  else
    forceAmplitude= *(args + 8);
  return forceAmplitude * vR;
}
double FDMDynamicalFrictionForcezforce(double R,double z, double phi,
						 double t,
						 struct potentialArg * potentialArgs,
						 double vR,double vT,
						 double vz){
  double forceAmplitude;
  double * args= potentialArgs->args;
  double r2=  R * R + z * z;
  if ( r2 < *(args+13) )  // r < minr, don't bother caching
    return 0.;
  //Get args
  double cached_R= *(args + 1);
  double cached_z= *(args + 2);
  double cached_phi= *(args + 3);
  double cached_t= *(args + 4);
  double cached_vR= *(args + 5);
  double cached_vT= *(args + 6);
  double cached_vz= *(args + 7);
  if ( R != cached_R || phi != cached_phi || z != cached_z || t != cached_t \
       || vR != cached_vR || vT != cached_vT || vz != cached_vz )
    // LCOV_EXCL_START
    forceAmplitude= FDMDynamicalFrictionForceAmplitude(R,z,phi,t,r2,
								 potentialArgs,
								 vR,vT,vz);
    // LCOV_EXCL_STOP
  else
    forceAmplitude= *(args + 8);
  return forceAmplitude * vz;
}
double FDMDynamicalFrictionForcephitorque(double R,double z,
						   double phi,double t,
						   struct potentialArg * potentialArgs,
						   double vR,double vT,
						   double vz){
  double forceAmplitude;
  double * args= potentialArgs->args;
  double r2=  R * R + z * z;
  if ( r2 < *(args+13) )  // r < minr, don't bother caching
    return 0.;
  //Get args
  double cached_R= *(args + 1);
  double cached_z= *(args + 2);
  double cached_phi= *(args + 3);
  double cached_t= *(args + 4);
  double cached_vR= *(args + 5);
  double cached_vT= *(args + 6);
  double cached_vz= *(args + 7);
  if ( R != cached_R || phi != cached_phi || z != cached_z || t != cached_t \
       || vR != cached_vR || vT != cached_vT || vz != cached_vz )
    // LCOV_EXCL_START
    forceAmplitude= FDMDynamicalFrictionForceAmplitude(R,z,phi,t,r2,
								 potentialArgs,
								 vR,vT,vz);
    // LCOV_EXCL_STOP
  else
    forceAmplitude= *(args + 8);
  return forceAmplitude * vT * R;
}
