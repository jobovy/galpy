#include <math.h>
#include <galpy_potentials.h>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
//BurkertPotential
// 2 arguments: amp, a
double BurkertPotentialEval(double R,double Z, double phi,
			    double t,
			    struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  //Get args
  double amp= *args;
  double a= *(args+1);
  //Calculate potential
  double x= sqrt( R*R + Z*Z) / a;
  return -amp * a * a * M_PI / x * ( -M_PI + 2. * ( 1. + x ) * atan( 1 / x ) + 2. * ( 1. + x ) * log ( 1. + x ) + ( 1. - x ) * log ( 1. + x * x ));
}
double BurkertPotentialRforce(double R,double Z, double phi,
				      double t,
				      struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  //Get args
  double amp= *args;
  double a= *(args+1);
  //Calculate Rforce
  double r= sqrt( R*R + Z*Z);
  double x= r / a;
  return amp * a * M_PI / x / x * ( M_PI - 2. * atan ( 1. / x ) - 2. * log ( 1. + x ) - log ( 1. + x * x)) * R / r;
}
double BurkertPotentialPlanarRforce(double R,double phi,
					    double t,
					    struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  //Get args
  double amp= *args;
  double a= *(args+1);
  //Calculate Rforce
  double x= R / a;
  return amp * a * M_PI / x / x * ( M_PI - 2. * atan ( 1. / x ) - 2. * log ( 1. + x ) - log ( 1. + x * x));
}
double BurkertPotentialzforce(double R,double z,double phi,
				      double t,
				      struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  //Get args
  double amp= *args;
  double a= *(args+1);
  //Calculate zforce
  double r= sqrt( R*R + z*z);
  double x= r / a;
  return amp * a * M_PI / x / x * ( M_PI - 2. * atan ( 1. / x ) - 2. * log ( 1. + x ) - log ( 1. + x * x)) * z / r;
}
double BurkertPotentialPlanarR2deriv(double R,double phi,
					     double t,
					     struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  //Get args
  double amp= *args;
  double a= *(args+1);
  //Calculate R2deriv
  double x= R / a;
  double R5= pow(R,5);
  return -amp * M_PI * pow(a,3) / R5 * (-4. * R5 / ( a * a + R * R ) / ( a + R ) - 2. * R * R * ( M_PI - 2. * atan ( 1. / x ) - 2. * log( 1. + x ) - log( 1. + x * x )));
}
double BurkertPotentialR2deriv(double R,double Z, double phi,
			       double t,
			       struct potentialArg * potentialArgs){
  //Spherical: Phi''(r)=PlanarR2deriv(r), Phi'(r)=-PlanarRforce(r) (incl. amp)
  double r2= R * R + Z * Z;
  double r= sqrt( r2 );
  double Phipp= BurkertPotentialPlanarR2deriv(r,phi,t,potentialArgs);
  double Phip= -BurkertPotentialPlanarRforce(r,phi,t,potentialArgs);
  double ir2= 1. / r2;
  double ir3= ir2 / r;
  //R2deriv = Phi''*R^2/r^2 + Phi'*z^2/r^3
  return Phipp * R * R * ir2 + Phip * Z * Z * ir3;
}
double BurkertPotentialz2deriv(double R,double Z, double phi,
			       double t,
			       struct potentialArg * potentialArgs){
  double r2= R * R + Z * Z;
  double r= sqrt( r2 );
  double Phipp= BurkertPotentialPlanarR2deriv(r,phi,t,potentialArgs);
  double Phip= -BurkertPotentialPlanarRforce(r,phi,t,potentialArgs);
  double ir2= 1. / r2;
  double ir3= ir2 / r;
  //z2deriv = Phi''*z^2/r^2 + Phi'*R^2/r^3
  return Phipp * Z * Z * ir2 + Phip * R * R * ir3;
}
double BurkertPotentialRzderiv(double R,double Z, double phi,
			       double t,
			       struct potentialArg * potentialArgs){
  double r2= R * R + Z * Z;
  double r= sqrt( r2 );
  double Phipp= BurkertPotentialPlanarR2deriv(r,phi,t,potentialArgs);
  double Phip= -BurkertPotentialPlanarRforce(r,phi,t,potentialArgs);
  double ir2= 1. / r2;
  double ir3= ir2 / r;
  //Rzderiv = R*z*(Phi''/r^2 - Phi'/r^3)
  return R * Z * ( Phipp * ir2 - Phip * ir3 );
}
double BurkertPotentialDens(double R,double Z, double phi,
			    double t,
			    struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  //Get args
  double amp= *args;
  double a= *(args+1);
  //Calculate potential
  double x= sqrt( R*R + Z*Z) / a;
  return amp / ( 1. + x ) / ( 1. + x * x );
}
