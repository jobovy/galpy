#include <math.h>
#include <galpy_potentials.h>
//Miyamoto-Nagai potential
//3 arguments: amp, a, b
double MiyamotoNagaiPotentialEval(double R,double z, double phi,
				  double t,
				  struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  //Get args
  double amp= *args++;
  double a= *args++;
  double b= *args;
  //Calculate Rforce
  return - amp * pow(R*R+pow(a+pow(z*z+b*b,0.5),2.),-0.5);
}
double MiyamotoNagaiPotentialRforce(double R,double z, double phi,
				    double t,
				    struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  //Get args
  double amp= *args++;
  double a= *args++;
  double b= *args;
  //Calculate Rforce
  return - amp * R * pow(R*R+pow(a+pow(z*z+b*b,0.5),2.),-1.5);
}
double MiyamotoNagaiPotentialPlanarRforce(double R,double phi,
					  double t,
					  struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  //Get args
  double amp= *args++;
  double a= *args++;
  double b= *args;
  //Calculate Rforce
  return - amp * R * pow(R*R+pow(a+b,2.),-1.5);
}
double MiyamotoNagaiPotentialzforce(double R,double z,double phi,
				    double t,
				    struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  //Get args
  double amp= *args++;
  double a= *args++;
  double b= *args;
  //Calculate zforce
  double sqrtbz= pow(b*b+z*z,.5);
  double asqrtbz= a+sqrtbz;
  if ( a == 0. )
    return amp * ( -z * pow(R*R+asqrtbz*asqrtbz,-1.5) );
  else
    return amp * ( -z * asqrtbz / sqrtbz * pow(R*R+asqrtbz*asqrtbz,-1.5) );
}
double MiyamotoNagaiPotentialPlanarR2deriv(double R,double phi,
					   double t,
					   struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  //Get args
  double amp= *args++;
  double a= *args++;
  double b= *args;
  //calculate R2deriv
  double denom= R*R+pow(a+b,2.);
  return amp * (pow(denom,-1.5) - 3. * R * R * pow(denom,-2.5));
}
double MiyamotoNagaiPotentialDens(double R,double z, double phi,
				  double t,
				  struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  //Get args
  double amp= *args++;
  double a= *args++;
  double b= *args;
  //Calculate density
  double b2= b*b;
  double sqrtbz= sqrt ( b2 + z * z );
  double asqrtbz= a+sqrtbz;
  asqrtbz*= asqrtbz;
  if ( a == 0. )
    return 3. * amp * M_1_PI / 4. * b2 * pow ( R * R + sqrtbz * sqrtbz, -2.5);
  else
    return amp * M_1_PI / 4. * b2 \
      * ( a * R * R + ( a + 3. * sqrtbz ) * asqrtbz )	\
      * pow ( R * R + asqrtbz,-2.5 ) * pow ( sqrtbz,-3.);
}
