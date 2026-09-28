#include <math.h>
#include <galpy_potentials.h>
//NFWPotential
//2 arguments: amp, a
// Below r/a = NFW_SMALL_X the closed forms below cancel terms of order 1/r^2
// (R2deriv 3e2 off at r/a = 1e-6); there, with t = x/(2+x) and
// S = sum_{m>=1} t^(2m+1)/(2m+1) (log1p(x) = 2 atanh(t), all terms positive)
//   h(x) = log1p(x) - x/(1+x) = 2 t^2/(1+t) + 2 S             [Phi'/amp = h/r^2]
//   k(x) = x^2/(1+x)^2 - 2 h(x) = -4 t^3/(1+t)^2 - 4 S        [Phi''/amp = k/r^3]
// (8 terms: < 1e-17 truncation at x = 0.25), matching the python implementation.
#define NFW_SMALL_X 0.25
static double nfw_S(double t){
  double t2= t * t;
  double out= 1. / 17.;
  int m;
  for (m=7; m >= 1; m--)
    out= out * t2 + 1. / ( 2. * m + 1. );
  return out * t2 * t;
}
static double nfw_h(double x){
  double t= x / ( 2. + x );
  return 2. * ( t * t / ( 1. + t ) + nfw_S(t) );
}
static void nfw_hk(double x, double * h, double * k){
  double t= x / ( 2. + x );
  double S= nfw_S(t);
  double u= t / ( 1. + t );
  *h= 2. * ( t * u + S );
  *k= -4. * ( t * u * u + S );
}
double NFWPotentialEval(double R,double Z, double phi,
			  double t,
			struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  //Get args
  double amp= *args++;
  double a= *args;
  //Calculate Rforce
  double sqrtRz= pow(R*R+Z*Z,0.5);
  if ( sqrtRz < NFW_SMALL_X * a )
    return - amp * log1p ( sqrtRz / a ) / sqrtRz;
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
  if ( sqrtRz < NFW_SMALL_X * a )
    return - amp * R * nfw_h(sqrtRz / a) / sqrtRz / sqrtRz / sqrtRz;
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
  if ( R < NFW_SMALL_X * a )
    return - amp * nfw_h(R / a) / R / R;
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
  if ( sqrtRz < NFW_SMALL_X * a )
    return - amp * Z * nfw_h(sqrtRz / a) / sqrtRz / sqrtRz / sqrtRz;
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
  if ( R < NFW_SMALL_X * a )
  {
    double h, k;
    nfw_hk(R / a, &h, &k);
    return amp * k / R / R / R;
  }
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
  double dphi, d2phi, ar;
  if ( r < NFW_SMALL_X * a ) {
    nfw_hk(r / a, &dphi, &d2phi);
    dphi/= r * r; // Phi'/amp
    d2phi/= r * r * r; // Phi''/amp
  }
  else {
    dphi= log(1.+r/a)/r/r - 1./r/(a+r); // Phi'/amp
    ar= a+r;
    d2phi= (r*(2.*a+3.*r)-2.*ar*ar*log(1.+r/a))/r/r/r/ar/ar; // Phi''/amp
  }
  //R2deriv = Phi''*R^2/r^2 + Phi'*z^2/r^3
  return amp * ( d2phi * R * R / r / r + dphi * Z * Z / r / r / r );
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
  double dphi, d2phi, ar;
  if ( r < NFW_SMALL_X * a ) {
    nfw_hk(r / a, &dphi, &d2phi);
    dphi/= r * r; // Phi'/amp
    d2phi/= r * r * r; // Phi''/amp
  }
  else {
    dphi= log(1.+r/a)/r/r - 1./r/(a+r); // Phi'/amp
    ar= a+r;
    d2phi= (r*(2.*a+3.*r)-2.*ar*ar*log(1.+r/a))/r/r/r/ar/ar; // Phi''/amp
  }
  //z2deriv = Phi''*z^2/r^2 + Phi'*R^2/r^3
  return amp * ( d2phi * Z * Z / r / r + dphi * R * R / r / r / r );
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
  double dphi, d2phi, ar;
  if ( r < NFW_SMALL_X * a ) {
    nfw_hk(r / a, &dphi, &d2phi);
    dphi/= r * r; // Phi'/amp
    d2phi/= r * r * r; // Phi''/amp
  }
  else {
    dphi= log(1.+r/a)/r/r - 1./r/(a+r); // Phi'/amp
    ar= a+r;
    d2phi= (r*(2.*a+3.*r)-2.*ar*ar*log(1.+r/a))/r/r/r/ar/ar; // Phi''/amp
  }
  //Rzderiv = R*z*(Phi''/r^2 - Phi'/r^3)
  return amp * R * Z * ( d2phi / r / r - dphi / r / r / r );
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
