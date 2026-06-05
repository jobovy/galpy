#include <math.h>
#include <galpy_potentials.h>
//PlummerPotential
//2  arguments: amp, b
double PlummerPotentialEval(double R,double Z, double phi,
				    double t,
				    struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  //Get args
  double amp= *args;
  double b2= *(args+1) * *(args+1);
  //Calculate potential
  return - amp / sqrt( R * R + Z * Z + b2 );
}
double PlummerPotentialRforce(double R,double Z, double phi,
				      double t,
				      struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  //Get args
  double amp= *args;
  double b2= *(args+1) * *(args+1);
  //Calculate Rforce
  return - amp * R * pow(R*R+Z*Z+b2,-1.5);
}
double PlummerPotentialPlanarRforce(double R,double phi,
					    double t,
					    struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  //Get args
  double amp= *args;
  double b2= *(args+1) * *(args+1);
  //Calculate Rforce
  return - amp * R * pow(R*R+b2,-1.5);
}
double PlummerPotentialzforce(double R,double Z,double phi,
				      double t,
				      struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  //Get args
  double amp= *args;
  double b2= *(args+1) * *(args+1);
  //Calculate zforce
  return - amp * Z * pow(R*R+Z*Z+b2,-1.5);
}
double PlummerPotentialPlanarR2deriv(double R,double phi,
					     double t,
					     struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  //Get args
  double amp= *args;
  double b2= *(args+1) * *(args+1);
  //Calculate Rforce
  return amp * (b2 - 2.*R*R)*pow(R*R+b2,-2.5);
}
double PlummerPotentialR2deriv(double R,double Z, double phi,
			       double t,
			       struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  //Get args
  double amp= *args;
  double b2= *(args+1) * *(args+1);
  //Calculate R2deriv (d^2Phi/dR^2)
  return amp * (b2 - 2.*R*R + Z*Z) * pow(R*R+Z*Z+b2,-2.5);
}
double PlummerPotentialz2deriv(double R,double Z, double phi,
			       double t,
			       struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  //Get args
  double amp= *args;
  double b2= *(args+1) * *(args+1);
  //Calculate z2deriv (d^2Phi/dz^2)
  return amp * (b2 + R*R - 2.*Z*Z) * pow(R*R+Z*Z+b2,-2.5);
}
double PlummerPotentialRzderiv(double R,double Z, double phi,
			       double t,
			       struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  //Get args
  double amp= *args;
  double b2= *(args+1) * *(args+1);
  //Calculate Rzderiv (d^2Phi/dR/dz)
  return amp * ( -3. * R * Z * pow(R*R+Z*Z+b2,-2.5) );
}
double PlummerPotentialDens(double R,double Z, double phi,
			    double t,
			    struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  //Get args
  double amp= *args;
  double b2= *(args+1) * *(args+1);
  //Calculate density
  return 3. * amp *M_1_PI / 4. * b2 * pow ( R * R + Z * Z + b2 , -2.5 );
}
