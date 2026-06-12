#include <math.h>
#include <galpy_potentials.h>
//SteadyLogSpiralPotential
//
double dehnenSpiralSmooth(double t,double tform, double tsteady){
  double smooth, xi,deltat;
  //Slightly different smoothing
  if ( ! isnan(tform) )
    if ( t < tform )
      smooth= 0.;
    else if ( t < tsteady ) {
      deltat= t-tform;
      xi= 2.*deltat/(tsteady-tform)-1.;
      smooth= (3./16.*pow(xi,5.)-5./8.*pow(xi,3.)+15./16.*xi+.5);
    }
    else
      smooth= 1.;
  else
    smooth=1.;
  return smooth;
}
double SteadyLogSpiralPotentialRforce(double R,double phi,double t,
				      struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  //declare
  double smooth;
  //Get args
  double amp= *args++;
  double tform= *args++;
  double tsteady= *args++;
  double A= *args++;
  double alpha= *args++;
  double m= *args++;
  double omegas= *args++;
  double gamma= *args++;
  //Calculate Rforce
  smooth= dehnenSpiralSmooth(t,tform,tsteady);
  return amp * smooth * A / R * sin(alpha * log(R) - m * (phi-omegas*t-gamma));
}
double SteadyLogSpiralPotentialphitorque(double R,double phi,double t,
					struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  //declare
  double smooth;
  //Get args
  double amp= *args++;
  double tform= *args++;
  double tsteady= *args++;
  double A= *args++;
  double alpha= *args++;
  double m= *args++;
  double omegas= *args++;
  double gamma= *args++;
  //Calculate Rforce
  smooth= dehnenSpiralSmooth(t,tform,tsteady);
  return -amp * smooth * A / alpha * m *
    sin(alpha * log(R) - m * (phi-omegas*t-gamma));
}
double SteadyLogSpiralPotentialR2deriv(double R,double phi,double t,
				       struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  //declare
  double smooth, chi;
  //Get args
  double amp= *args++;
  double tform= *args++;
  double tsteady= *args++;
  double A= *args++;
  double alpha= *args++;
  double m= *args++;
  double omegas= *args++;
  double gamma= *args++;
  //Calculate R2deriv
  smooth= dehnenSpiralSmooth(t,tform,tsteady);
  chi= alpha * log(R) - m * (phi-omegas*t-gamma);
  return amp * smooth * A / R / R * ( sin(chi) - alpha * cos(chi) );
}
double SteadyLogSpiralPotentialphi2deriv(double R,double phi,double t,
					 struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  //declare
  double smooth;
  //Get args
  double amp= *args++;
  double tform= *args++;
  double tsteady= *args++;
  double A= *args++;
  double alpha= *args++;
  double m= *args++;
  double omegas= *args++;
  double gamma= *args++;
  //Calculate phi2deriv
  smooth= dehnenSpiralSmooth(t,tform,tsteady);
  return -amp * smooth * A / alpha * m * m *
    cos(alpha * log(R) - m * (phi-omegas*t-gamma));
}
double SteadyLogSpiralPotentialRphideriv(double R,double phi,double t,
					 struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  //declare
  double smooth;
  //Get args
  double amp= *args++;
  double tform= *args++;
  double tsteady= *args++;
  double A= *args++;
  double alpha= *args++;
  double m= *args++;
  double omegas= *args++;
  double gamma= *args++;
  //Calculate Rphideriv
  smooth= dehnenSpiralSmooth(t,tform,tsteady);
  return amp * smooth * A * m / R *
    cos(alpha * log(R) - m * (phi-omegas*t-gamma));
}
