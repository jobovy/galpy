#include <math.h>
// Constants not defined in MSVC's math.h
#ifndef M_SQRT1_2
#define M_SQRT1_2 0.70710678118654746172
#endif
#ifndef M_2_SQRTPI
#define M_2_SQRTPI 1.12837916709551255856
#endif
#include <gsl/gsl_spline.h>
#include <galpy_potentials.h>
// ChandrasekharDynamicalFrictionForce: 8 arguments: amp,ms,rhm,gamma^2,
// lnLambda, minr^2, ro, rf
double ChandrasekharDynamicalFrictionForceAmplitude(double R,double z,
						    double phi,double t,
						    double r2,
						    struct potentialArg * potentialArgs,
						    double vR,double vT,
						    double vz){
  double sr,X,Xfactor,d_ind,forceAmplitude;
  double * args= potentialArgs->args;
  //Get args
  double amp= *args;
  double ms= *(args+9);
  double rhm= *(args+10);
  double gamma2= *(args+11);
  double lnLambda= *(args+12);
  double ro= *(args+14);
  double rf= *(args+15);
  double GMvs;
  double r= sqrt( r2 );
  double v2=  vR * vR + vT * vT + vz * vz;
  double v= sqrt( v2 );
  // Constant or variable Lambda
  if ( lnLambda < 0 ) {
    GMvs= ms/v/v;
    if ( GMvs < rhm )
      lnLambda= 0.5 * log ( 1. + r2 / gamma2 / rhm / rhm );
    else
      lnLambda= 0.5 * log ( 1. + r2 / gamma2 / GMvs / GMvs );
  }
  d_ind= (r-ro)/(rf-ro);
  d_ind= d_ind <  0 ? 0. : ( d_ind > 1 ? 1. : d_ind);
  sr= gsl_spline_eval(*potentialArgs->spline1d,d_ind,*potentialArgs->acc1d);
  X= M_SQRT1_2 * v / sr;
  Xfactor= erf ( X ) - M_2_SQRTPI * X * exp ( - X * X );
  forceAmplitude= - amp * Xfactor * lnLambda / v2 / v \
    * calcDensity(R,z,phi,t,potentialArgs->nwrapped,
		  potentialArgs->wrappedPotentialArg);
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
double ChandrasekharDynamicalFrictionForceRforce(double R,double z, double phi,
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
    forceAmplitude= ChandrasekharDynamicalFrictionForceAmplitude(R,z,phi,t,r2,
								 potentialArgs,
								 vR,vT,vz);
  else
    forceAmplitude= *(args + 8);
  return forceAmplitude * vR;
}
double ChandrasekharDynamicalFrictionForcezforce(double R,double z, double phi,
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
    forceAmplitude= ChandrasekharDynamicalFrictionForceAmplitude(R,z,phi,t,r2,
								 potentialArgs,
								 vR,vT,vz);
    // LCOV_EXCL_STOP
  else
    forceAmplitude= *(args + 8);
  return forceAmplitude * vz;
}
double ChandrasekharDynamicalFrictionForcephitorque(double R,double z,
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
    forceAmplitude= ChandrasekharDynamicalFrictionForceAmplitude(R,z,phi,t,r2,
								 potentialArgs,
								 vR,vT,vz);
    // LCOV_EXCL_STOP
  else
    forceAmplitude= *(args + 8);
  return forceAmplitude * vT * R;
}
