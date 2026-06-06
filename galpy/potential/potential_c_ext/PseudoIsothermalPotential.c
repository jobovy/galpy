#include <math.h>
#include <galpy_potentials.h>
//PseudoIsothermalPotential
//2 arguments: amp, a
double PseudoIsothermalPotentialEval(double R,double Z, double phi,
				    double t,
				    struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  //Get args
  double amp= *args;
  double a= *(args+1);
  double a2= a*a;
  //Calculate potential
  double r2= R*R+Z*Z;
  double r= sqrt(r2);
  return amp * (0.5 * log(1 + r2 / a2) + a / r * atan(r / a)) / a;
}
double PseudoIsothermalPotentialRforce(double R,double Z, double phi,
				      double t,
				      struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  //Get args
  double amp= *args;
  double a= *(args+1);
  //Calculate potential
  double r2= R*R+Z*Z;
  double r= sqrt(r2);
  return - amp * (1. / r - a / r2 * atan(r / a)) / a * R / r;
}
double PseudoIsothermalPotentialPlanarRforce(double R,double phi,
					    double t,
					    struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  //Get args
  double amp= *args;
  double a= *(args+1);
  //Calculate potential
  return - amp * (1. / R - a / R / R * atan(R / a)) / a;
}
double PseudoIsothermalPotentialzforce(double R,double z,double phi,
				      double t,
				      struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  //Get args
  double amp= *args;
  double a= *(args+1);
  //Calculate potential
  double r2= R*R+z*z;
  double r= sqrt(r2);
  return - amp * (1. / r - a / r2 * atan(r / a)) / a * z / r;
}
double PseudoIsothermalPotentialPlanarR2deriv(double R,double phi,
					     double t,
					     struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  //Get args
  double amp= *args;
  double a= *(args+1);
  double a2= a*a;
  //Calculate potential
  double R2= R*R;
  return amp / R2 * (2. * a / R * atan(R / a) - ( 2. * a2 +  R2)/(a2 + R2) )
    / a;
}
double PseudoIsothermalPotentialR2deriv(double R,double Z, double phi,
					double t,
					struct potentialArg * potentialArgs){
  //Spherical: Phi''(r)=PlanarR2deriv(r), Phi'(r)=-PlanarRforce(r) (incl. amp)
  double r2= R * R + Z * Z;
  double r= sqrt( r2 );
  double Phipp= PseudoIsothermalPotentialPlanarR2deriv(r,phi,t,potentialArgs);
  double Phip= -PseudoIsothermalPotentialPlanarRforce(r,phi,t,potentialArgs);
  double ir2= 1. / r2;
  double ir3= ir2 / r;
  //R2deriv = Phi''*R^2/r^2 + Phi'*z^2/r^3
  return Phipp * R * R * ir2 + Phip * Z * Z * ir3;
}
double PseudoIsothermalPotentialz2deriv(double R,double Z, double phi,
					double t,
					struct potentialArg * potentialArgs){
  double r2= R * R + Z * Z;
  double r= sqrt( r2 );
  double Phipp= PseudoIsothermalPotentialPlanarR2deriv(r,phi,t,potentialArgs);
  double Phip= -PseudoIsothermalPotentialPlanarRforce(r,phi,t,potentialArgs);
  double ir2= 1. / r2;
  double ir3= ir2 / r;
  //z2deriv = Phi''*z^2/r^2 + Phi'*R^2/r^3
  return Phipp * Z * Z * ir2 + Phip * R * R * ir3;
}
double PseudoIsothermalPotentialRzderiv(double R,double Z, double phi,
					double t,
					struct potentialArg * potentialArgs){
  double r2= R * R + Z * Z;
  double r= sqrt( r2 );
  double Phipp= PseudoIsothermalPotentialPlanarR2deriv(r,phi,t,potentialArgs);
  double Phip= -PseudoIsothermalPotentialPlanarRforce(r,phi,t,potentialArgs);
  double ir2= 1. / r2;
  double ir3= ir2 / r;
  //Rzderiv = R*z*(Phi''/r^2 - Phi'/r^3)
  return R * Z * ( Phipp * ir2 - Phip * ir3 );
}
double PseudoIsothermalPotentialDens(double R,double Z, double phi,
				     double t,
				     struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  //Get args
  double amp= *args;
  double a= *(args+1);
  double a2= a*a;
  //Calculate potential
  double r2= R*R+Z*Z;
  return amp * M_1_PI / 4. / ( 1. + r2 / a2 ) / a2 / a;
}
