#include <math.h>
#include <galpy_potentials.h>
//DehnenBarPotential
//
double dehnenSmooth(double t,double tform, double tsteady){
  double smooth, xi,deltat;
  if ( t < tform )
    smooth= 0.;
  else if ( t < tsteady ) {
    deltat= t-tform;
    xi= 2.*deltat/(tsteady-tform)-1.;
    smooth= (3./16.*pow(xi,5.)-5./8.*pow(xi,3.)+15./16.*xi+.5);
  }
  else
    smooth= 1.;
  return smooth;
}
double DehnenBarPotentialRforce(double R,double phi,double t,
				      int nargs, double *args){
  //declare
  double smooth;
  //Get args
  double amp= *args++;
  double tform= *args++;
  double tsteady= *args++;
  double rb= *args++;
  double af= *args++;
  double omegab= *args++;
  double barphi= *args++;
  //Calculate Rforce
  smooth= dehnenSmooth(t,tform,tsteady);
  if (R <= rb )
    return -3.*amp*af*smooth*cos(2.*(phi-omegab*t-barphi))*pow(R/rb,3.)/R;
  else
    return -3.*amp*af*smooth*cos(2.*(phi-omegab*t-barphi))*pow(rb/R,3.)/R;   
}
double DehnenBarPotentialphiforce(double R,double phi,double t,
				      int nargs, double *args){
  //declare
  double smooth;
  //Get args
  double amp= *args++;
  double tform= *args++;
  double tsteady= *args++;
  double rb= *args++;
  double af= *args++;
  double omegab= *args++;
  double barphi= *args++;
  //Calculate phiforce
  smooth= dehnenSmooth(t,tform,tsteady);
  if ( R <= rb )
    return 2.*amp*af*smooth*sin(2.*(phi-omegab*t-barphi))*(pow(R/rb,3.)-2.);
  else
    return -2.*amp*af*smooth*sin(2.*(phi-omegab*t-barphi))*pow(rb/R,3.);
}
