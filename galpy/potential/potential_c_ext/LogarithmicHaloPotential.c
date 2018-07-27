#include <math.h>
#include <galpy_potentials.h>
//LogarithmicHaloPotential
//4 (3)  arguments: amp, c2, (and q), now also 1-1/b^2 for triaxial!
double LogarithmicHaloPotentialEval(double R,double Z, double phi,
				    double t,
				    struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  //Get args
  double amp= *args;
  double q= *(args+1);
  double c= *(args+2);
  double onem1overb2= *(args+3);
  //Calculate potential
  double zq= Z/q;
  if ( onem1overb2 < 1 )
    return 0.5 * amp * log(R*R * (1. - onem1overb2 * pow(sin(phi),2))+zq*zq+c);
  else
    return 0.5 * amp * log(R*R+zq*zq+c);
}
double LogarithmicHaloPotentialRforce(double R,double Z, double phi,
				      double t,
				      struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  //Get args
  double amp= *args;
  double q= *(args+1);
  double c= *(args+2);
  double onem1overb2= *(args+3);
  //Calculate Rforce
  double zq= Z/q;
  double Rt2;
  if ( onem1overb2 < 1 ) {
    Rt2= R*R * (1. - onem1overb2 * pow(sin(phi),2));
    return - amp * Rt2/R/(Rt2+zq*zq+c);
  } else
    return - amp * R/(R*R+zq*zq+c);
}
double LogarithmicHaloPotentialPlanarRforce(double R,double phi,
					    double t,
					    struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  //Get args
  double amp= *args;
  double c= *(args+2); // skip q
  double onem1overb2= *(args+3);
  //Calculate Rforce
  double Rt2;
  if ( onem1overb2 < 1 ) {
    Rt2= R*R * (1. - onem1overb2 * pow(sin(phi),2));
    return -amp * Rt2/R/(Rt2+c);
  } else
    return -amp * R/(R*R+c);
}
double LogarithmicHaloPotentialzforce(double R,double z,double phi,
				      double t,
				      struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  //Get args
  double amp= *args;
  double q= *(args+1);
  double c= *(args+2);
  double onem1overb2= *(args+3);
  //Calculate zforce
  double zq= z/q;
  double Rt2;
  if ( onem1overb2 < 1 ) {
    Rt2= R*R * (1. - onem1overb2 * pow(sin(phi),2));
    return -amp * zq/q/(Rt2+zq*zq+c);
  } else
    return -amp * zq/q/(R*R+zq*zq+c);
}
double LogarithmicHaloPotentialphiforce(double R,double z,double phi,
					double t,
					struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  //Get args
  double amp= *args;
  double q= *(args+1);
  double c= *(args+2);
  double onem1overb2= *(args+3);
  //Calculate phiforce
  double zq;
  double Rt2;
  if ( onem1overb2 < 1 ) {
    zq= z/q;
    Rt2= R*R * (1. - onem1overb2 * pow(sin(phi),2));
    return amp * R*R / (Rt2+zq*zq+c) * sin(2*phi) * onem1overb2 / 2.;
  } else
    return 0.;
}
double LogarithmicHaloPotentialPlanarphiforce(double R,double phi,
					      double t,
					      struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  //Get args
  double amp= *args;
  double c= *(args+2); // skip q
  double onem1overb2= *(args+3);
  //Calculate phiforce
  double Rt2;
  if ( onem1overb2 < 1 ) {
    Rt2= R*R * (1. - onem1overb2 * pow(sin(phi),2));
    return amp * R*R / (Rt2+c) * sin(2*phi) * onem1overb2 / 2.;
  } else
    return 0.;
}
double LogarithmicHaloPotentialPlanarR2deriv(double R,double phi,
					     double t,
					     struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  //Get args
  double amp= *args;
  double c= *(args+2); // skip q
  double onem1overb2= *(args+3);
  //Calculate Rforce
  double Rt2;
  if ( onem1overb2 < 1 ) {
    Rt2= R*R * (1. - onem1overb2 * pow(sin(phi),2));
    return amp * (1.- 2.*Rt2/(Rt2+c))/(Rt2+c)*Rt2/R/R;
  } else
    return amp * (1.- 2.*R*R/(R*R+c))/(R*R+c);
}
double LogarithmicHaloPotentialPlanarphi2deriv(double R,double phi,
					       double t,
					       struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  //Get args
  double amp= *args;
  double c= *(args+2); // skip q
  double onem1overb2= *(args+3);
  //Calculate Rforce
  double Rt2;
  if ( onem1overb2 < 1 ) {
    Rt2= R*R * (1. - onem1overb2 * pow(sin(phi),2));
    return - amp * onem1overb2 * (0.5 * pow(R*R*sin(2.*phi),2.) * onem1overb2 \
				  /(Rt2+c)/(Rt2+c)+R*R/(Rt2+c)*cos(2.*phi));
  } else
    return 0.;
}
double LogarithmicHaloPotentialPlanarRphideriv(double R,double phi,
					     double t,
					     struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  //Get args
  double amp= *args;
  double c= *(args+2); // skip q
  double onem1overb2= *(args+3);
  //Calculate Rforce
  double Rt2;
  if ( onem1overb2 < 1 ) {
    Rt2= R*R * (1. - onem1overb2 * pow(sin(phi),2));
    return - amp * c / (Rt2+c) / (Rt2+c) * R * sin(2.*phi) * onem1overb2;
  } else
    return 0.;
}
