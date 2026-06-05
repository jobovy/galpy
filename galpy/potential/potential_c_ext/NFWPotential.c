#include <math.h>
#include <galpy_potentials.h>
//NFWPotential
//2 arguments: amp, a
double NFWPotentialEval(double R,double Z, double phi,
			  double t,
			struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  //Get args
  double amp= *args++;
  double a= *args;
  //Calculate Rforce
  double sqrtRz= pow(R*R+Z*Z,0.5);
  return - amp * log ( 1. + sqrtRz / a ) / sqrtRz;
}
double NFWPotentialRforce(double R,double Z, double phi,
			  double t,
			  struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  //Get args
  double amp= *args++;
  double a= *args;
  //Calculate Rforce
  double Rz= R*R+Z*Z;
  double sqrtRz= pow(Rz,0.5);
  return amp * R * (1. / Rz / (a + sqrtRz)-log(1.+sqrtRz / a)/sqrtRz/Rz);
}
double NFWPotentialPlanarRforce(double R,double phi,
					    double t,
				struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  //Get args
  double amp= *args++;
  double a= *args;
  //Calculate Rforce
  return amp / R * (1. / (a + R)-log(1.+ R / a)/ R);
}
double NFWPotentialzforce(double R,double Z,double phi,
			  double t,
			  struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  //Get args
  double amp= *args++;
  double a= *args;
  //Calculate Rforce
  double Rz= R*R+Z*Z;
  double sqrtRz= pow(Rz,0.5);
  return amp * Z * (1. / Rz / (a + sqrtRz)-log(1.+sqrtRz / a)/sqrtRz/Rz);
}
double NFWPotentialPlanarR2deriv(double R,double phi,
				 double t,
				 struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  //Get args
  double amp= *args++;
  double a= *args;
  //Calculate R2deriv
  double aR= a+R;
  double aR2= aR*aR;
  return amp * (((R*(2.*a+3.*R))-2.*aR2*log(1.+R/a))/R/R/R/aR2);
}
double NFWPotentialR2deriv(double R,double Z, double phi,
			   double t,
			   struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  //Get args
  double amp= *args++;
  double a= *args;
  //Spherical: r, Phi'(r), Phi''(r)
  double r= sqrt( R * R + Z * Z );
  double dphi= amp * (log(1.+r/a)/r/r - 1./r/(a+r)); // Phi'
  double ar= a+r;
  double d2phi= amp * ((r*(2.*a+3.*r)-2.*ar*ar*log(1.+r/a))/r/r/r/ar/ar); // Phi''
  //R2deriv = Phi''*R^2/r^2 + Phi'*z^2/r^3
  return d2phi * R * R / r / r + dphi * Z * Z / r / r / r;
}
double NFWPotentialz2deriv(double R,double Z, double phi,
			   double t,
			   struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  //Get args
  double amp= *args++;
  double a= *args;
  //Spherical: r, Phi'(r), Phi''(r)
  double r= sqrt( R * R + Z * Z );
  double dphi= amp * (log(1.+r/a)/r/r - 1./r/(a+r)); // Phi'
  double ar= a+r;
  double d2phi= amp * ((r*(2.*a+3.*r)-2.*ar*ar*log(1.+r/a))/r/r/r/ar/ar); // Phi''
  //z2deriv = Phi''*z^2/r^2 + Phi'*R^2/r^3
  return d2phi * Z * Z / r / r + dphi * R * R / r / r / r;
}
double NFWPotentialRzderiv(double R,double Z, double phi,
			   double t,
			   struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  //Get args
  double amp= *args++;
  double a= *args;
  //Spherical: r, Phi'(r), Phi''(r)
  double r= sqrt( R * R + Z * Z );
  double dphi= amp * (log(1.+r/a)/r/r - 1./r/(a+r)); // Phi'
  double ar= a+r;
  double d2phi= amp * ((r*(2.*a+3.*r)-2.*ar*ar*log(1.+r/a))/r/r/r/ar/ar); // Phi''
  //Rzderiv = R*z*(Phi''/r^2 - Phi'/r^3)
  return R * Z * ( d2phi / r / r - dphi / r / r / r );
}
double NFWPotentialDens(double R,double Z, double phi,
			double t,
			struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  //Get args
  double amp= *args++;
  double a= *args;
  //Calculate density
  double sqrtRz= sqrt ( R * R + Z * Z );
  return amp * M_1_PI / 4. / a / a \
    / ( 1. + sqrtRz / a ) / ( 1. + sqrtRz / a ) / sqrtRz;
}
