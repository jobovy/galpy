#include <math.h>
#include <galpy_potentials.h>
//HomogeneousSpherePotential
//3 arguments: amp, R2, R3
double HomogeneousSpherePotentialEval(double R,double Z, double phi,
				      double t,
				      struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  //Get args
  double amp= *args;
  double R2= *(args+1);
  double R3= *(args+2);
  //Calculate potential
  double r2= R * R + Z * Z;
  if ( r2 < R2 )
    return amp * ( r2 - 3. * R2 );
  else
    return -2. * amp * R3 / sqrt( r2 );
}
double HomogeneousSpherePotentialRforce(double R,double Z, double phi,
					double t,
					struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  //Get args
  double amp= *args;
  double R2= *(args+1);
  double R3= *(args+2);
  //Calculate Rforce
  double r2= R * R + Z * Z;
  if ( r2 < R2 )
    return -2. * amp * R;
  else
    return -2. * amp * R3 * R / pow( r2 , 1.5 );
}
double HomogeneousSpherePotentialPlanarRforce(double R,double phi,
					      double t,
					      struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  //Get args
  double amp= *args;
  double R2= *(args+1);
  double R3= *(args+2);
  //Calculate Rforce
  double r2= R * R;
  if ( r2 < R2 )
    return -2. * amp * R;
  else
    return -2. * amp * R3 / r2;
}
double HomogeneousSpherePotentialzforce(double R,double z,double phi,
					double t,
					struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  //Get args
  double amp= *args;
  double R2= *(args+1);
  double R3= *(args+2);
  //Calculate zforce
  double r2= R * R + z * z;
  if ( r2 < R2 )
    return -2. * amp * z;
  else
    return -2. * amp * R3 * z / pow ( r2 , 1.5 );
}
double HomogeneousSpherePotentialPlanarR2deriv(double R,double phi,
					       double t,
					       struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  //Get args
  double amp= *args;
  double R2= *(args+1);
  double R3= *(args+2);
  //Calculate R2deriv
  double r2= R * R;
  if ( r2 < R2 )
    return 2. * amp;
  else
    return -4. * amp * R3 / pow ( r2 , 1.5 );
}
double HomogeneousSpherePotentialR2deriv(double R,double Z, double phi,
					 double t,
					 struct potentialArg * potentialArgs){
  //Spherical: Phi''(r)=PlanarR2deriv(r), Phi'(r)=-PlanarRforce(r) (incl. amp)
  double r2= R * R + Z * Z;
  double r= sqrt( r2 );
  double Phipp= HomogeneousSpherePotentialPlanarR2deriv(r,phi,t,potentialArgs);
  double Phip= -HomogeneousSpherePotentialPlanarRforce(r,phi,t,potentialArgs);
  double ir2= 1. / r2;
  double ir3= ir2 / r;
  //R2deriv = Phi''*R^2/r^2 + Phi'*z^2/r^3
  return Phipp * R * R * ir2 + Phip * Z * Z * ir3;
}
double HomogeneousSpherePotentialz2deriv(double R,double Z, double phi,
					 double t,
					 struct potentialArg * potentialArgs){
  double r2= R * R + Z * Z;
  double r= sqrt( r2 );
  double Phipp= HomogeneousSpherePotentialPlanarR2deriv(r,phi,t,potentialArgs);
  double Phip= -HomogeneousSpherePotentialPlanarRforce(r,phi,t,potentialArgs);
  double ir2= 1. / r2;
  double ir3= ir2 / r;
  //z2deriv = Phi''*z^2/r^2 + Phi'*R^2/r^3
  return Phipp * Z * Z * ir2 + Phip * R * R * ir3;
}
double HomogeneousSpherePotentialRzderiv(double R,double Z, double phi,
					 double t,
					 struct potentialArg * potentialArgs){
  double r2= R * R + Z * Z;
  double r= sqrt( r2 );
  double Phipp= HomogeneousSpherePotentialPlanarR2deriv(r,phi,t,potentialArgs);
  double Phip= -HomogeneousSpherePotentialPlanarRforce(r,phi,t,potentialArgs);
  double ir2= 1. / r2;
  double ir3= ir2 / r;
  //Rzderiv = R*z*(Phi''/r^2 - Phi'/r^3)
  return R * Z * ( Phipp * ir2 - Phip * ir3 );
}
double HomogeneousSpherePotentialDens(double R,double Z, double phi,
				      double t,
				      struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  //Get args
  double amp= *args;
  double R2= *(args+1);
  //Calculate potential
  double r2= R * R + Z * Z;
  if ( r2 < R2 )
    return 1.5 * amp * M_1_PI;
  else
    return 0.;
}
