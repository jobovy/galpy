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
