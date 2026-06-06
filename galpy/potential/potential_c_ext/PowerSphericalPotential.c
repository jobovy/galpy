#include <math.h>
#include <galpy_potentials.h>
//PowerSphericalPotential
//2  arguments: amp, alpha
double PowerSphericalPotentialEval(double R,double Z, double phi,
				   double t,
				   struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  //Get args
  double amp= *args++;
  double alpha= *args;
  //Calculate Rforce
  if ( alpha == 2. )
    return 0.5 * amp * log ( R*R+Z*Z);
  else
    return - amp * pow(R*R+Z*Z,1.-0.5*alpha) / (alpha - 2.);
}
double PowerSphericalPotentialRforce(double R,double Z, double phi,
				      double t,
				     struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  //Get args
  double amp= *args++;
  double alpha= *args;
  //Calculate Rforce
  return - amp * R * pow(R*R+Z*Z,-0.5*alpha);
}
double PowerSphericalPotentialPlanarRforce(double R,double phi,
					   double t,
					   struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  //Get args
  double amp= *args++;
  double alpha= *args;
  //Calculate Rforce
  return - amp * pow(R,-alpha + 1.);
}
double PowerSphericalPotentialzforce(double R,double Z,double phi,
				     double t,
				     struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  //Get args
  double amp= *args++;
  double alpha= *args;
  //Calculate zforce
  return - amp * Z * pow(R*R+Z*Z,-0.5*alpha);
}
double PowerSphericalPotentialPlanarR2deriv(double R,double phi,
					     double t,
					    struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  //Get args
  double amp= *args++;
  double alpha= *args;
  //Calculate R2deriv
  return amp * (1. - alpha ) * pow(R,-alpha);
}
double PowerSphericalPotentialR2deriv(double R,double Z, double phi,
				      double t,
				      struct potentialArg * potentialArgs){
  //Spherical: Phi''(r)=PlanarR2deriv(r), Phi'(r)=-PlanarRforce(r) (incl. amp)
  double r2= R * R + Z * Z;
  double r= sqrt( r2 );
  double Phipp= PowerSphericalPotentialPlanarR2deriv(r,phi,t,potentialArgs);
  double Phip= -PowerSphericalPotentialPlanarRforce(r,phi,t,potentialArgs);
  double ir2= 1. / r2;
  double ir3= ir2 / r;
  //R2deriv = Phi''*R^2/r^2 + Phi'*z^2/r^3
  return Phipp * R * R * ir2 + Phip * Z * Z * ir3;
}
double PowerSphericalPotentialz2deriv(double R,double Z, double phi,
				      double t,
				      struct potentialArg * potentialArgs){
  double r2= R * R + Z * Z;
  double r= sqrt( r2 );
  double Phipp= PowerSphericalPotentialPlanarR2deriv(r,phi,t,potentialArgs);
  double Phip= -PowerSphericalPotentialPlanarRforce(r,phi,t,potentialArgs);
  double ir2= 1. / r2;
  double ir3= ir2 / r;
  //z2deriv = Phi''*z^2/r^2 + Phi'*R^2/r^3
  return Phipp * Z * Z * ir2 + Phip * R * R * ir3;
}
double PowerSphericalPotentialRzderiv(double R,double Z, double phi,
				      double t,
				      struct potentialArg * potentialArgs){
  double r2= R * R + Z * Z;
  double r= sqrt( r2 );
  double Phipp= PowerSphericalPotentialPlanarR2deriv(r,phi,t,potentialArgs);
  double Phip= -PowerSphericalPotentialPlanarRforce(r,phi,t,potentialArgs);
  double ir2= 1. / r2;
  double ir3= ir2 / r;
  //Rzderiv = R*z*(Phi''/r^2 - Phi'/r^3)
  return R * Z * ( Phipp * ir2 - Phip * ir3 );
}
double PowerSphericalPotentialDens(double R,double Z, double phi,
				   double t,
				   struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  //Get args
  double amp= *args++;
  double alpha= *args;
  //Calculate density
  return amp * M_1_PI / 4. * ( 3. - alpha ) * pow (R*R + Z*Z, -0.5 * alpha);
}
