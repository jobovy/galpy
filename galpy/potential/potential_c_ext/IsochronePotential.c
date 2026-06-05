#include <math.h>
#include <galpy_potentials.h>
//IsochronePotential
//2  arguments: amp, b
double IsochronePotentialEval(double R,double Z, double phi,
			      double t,
			      struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  //Get args
  double amp= *args;
  double b= *(args+1);
  //Calculate potential
  double r2= R*R+Z*Z;
  return -amp / ( b + sqrt(r2 + b * b) );
}
double IsochronePotentialRforce(double R,double Z, double phi,
				      double t,
				      struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  //Get args
  double amp= *args;
  double b= *(args+1);
  //Calculate Rforce
  double r2= R*R+Z*Z;
  double rb= sqrt(r2 + b * b);
  return - amp * R / rb * pow(b + rb,-2.);
}
double IsochronePotentialPlanarRforce(double R,double phi,
					    double t,
					    struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  //Get args
  double amp= *args;
  double b= *(args+1);
  //Calculate Rforce
  double r2= R*R;
  double rb= sqrt(r2 + b * b);
  return - amp * R / rb * pow(b + rb,-2.);
}
double IsochronePotentialzforce(double R,double Z,double phi,
				double t,
				struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  //Get args
  double amp= *args;
  double b= *(args+1);
  //Calculate zforce
  double r2= R*R+Z*Z;
  double rb= sqrt(r2 + b * b);
  return - amp * Z / rb * pow(b + rb,-2.);
}
double IsochronePotentialPlanarR2deriv(double R,double phi,
					     double t,
					     struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  //Get args
  double amp= *args;
  double b= *(args+1);
  //Calculate Rforce
  double r2= R*R;
  double rb= sqrt(r2 + b * b);
  return - amp * ( -pow(b,3.) - b * b * rb + 2. * r2 * rb ) * pow(rb * ( b + rb ),-3.);
}
double IsochronePotentialR2deriv(double R,double Z, double phi,
				 double t,
				 struct potentialArg * potentialArgs){
  //Spherical: Phi''(r)=PlanarR2deriv(r), Phi'(r)=-PlanarRforce(r) (incl. amp)
  double r= sqrt( R * R + Z * Z );
  double Phipp= IsochronePotentialPlanarR2deriv(r,phi,t,potentialArgs);
  double Phip= -IsochronePotentialPlanarRforce(r,phi,t,potentialArgs);
  //R2deriv = Phi''*R^2/r^2 + Phi'*z^2/r^3
  return Phipp * R * R / r / r + Phip * Z * Z / r / r / r;
}
double IsochronePotentialz2deriv(double R,double Z, double phi,
				 double t,
				 struct potentialArg * potentialArgs){
  double r= sqrt( R * R + Z * Z );
  double Phipp= IsochronePotentialPlanarR2deriv(r,phi,t,potentialArgs);
  double Phip= -IsochronePotentialPlanarRforce(r,phi,t,potentialArgs);
  //z2deriv = Phi''*z^2/r^2 + Phi'*R^2/r^3
  return Phipp * Z * Z / r / r + Phip * R * R / r / r / r;
}
double IsochronePotentialRzderiv(double R,double Z, double phi,
				 double t,
				 struct potentialArg * potentialArgs){
  double r= sqrt( R * R + Z * Z );
  double Phipp= IsochronePotentialPlanarR2deriv(r,phi,t,potentialArgs);
  double Phip= -IsochronePotentialPlanarRforce(r,phi,t,potentialArgs);
  //Rzderiv = R*z*(Phi''/r^2 - Phi'/r^3)
  return R * Z * ( Phipp / r / r - Phip / r / r / r );
}
double IsochronePotentialDens(double R,double Z, double phi,
			      double t,
			      struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  //Get args
  double amp= *args;
  double b= *(args+1);
  //Calculate potential
  double r2= R * R + Z * Z;
  double rb= sqrt ( r2 + b * b );
  double brbrb= ( b + rb ) * rb;
  return amp * M_1_PI / 4. * ( 3. * brbrb * rb \
			       - r2 * ( b + 3. * rb ) )	\
    * pow ( brbrb , -3.);
}
