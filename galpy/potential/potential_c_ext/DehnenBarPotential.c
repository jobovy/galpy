#include <math.h>
#include <galpy_potentials.h>
//DehnenBarPotential
//
double dehnenBarSmooth(double t,double tform, double tsteady){
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
double DehnenBarPotentialRforce(double R,double z,double phi,double t,
				struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  //declare
  double smooth;
  double r;
  //Get args
  double amp= *args++;
  double tform= *args++;
  double tsteady= *args++;
  double rb= *args++;
  double omegab= *args++;
  double barphi= *args++;
  //Calculate Rforce
  smooth= dehnenBarSmooth(t,tform,tsteady);
  r= sqrt( R * R + z * z );
  if (r <= rb )
    return -amp*smooth*cos(2.*(phi-omegab*t-barphi))*\
      (pow(r/rb,3.)*R*(3.*R*R+2.*z*z)-4.*R*z*z)/pow(r,4.);
  else
    return -amp*smooth*cos(2.*(phi-omegab*t-barphi))\
      *pow(rb/r,3.)*R/pow(r,4)*(3.*R*R-2.*z*z);
}
double DehnenBarPotentialPlanarRforce(double R,double phi,double t,
				      struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  //declare
  double smooth;
  //Get args
  double amp= *args++;
  double tform= *args++;
  double tsteady= *args++;
  double rb= *args++;
  double omegab= *args++;
  double barphi= *args++;
  //Calculate Rforce
  smooth= dehnenBarSmooth(t,tform,tsteady);
  if (R <= rb )
    return -3.*amp*smooth*cos(2.*(phi-omegab*t-barphi))*pow(R/rb,3.)/R;
  else
    return -3.*amp*smooth*cos(2.*(phi-omegab*t-barphi))*pow(rb/R,3.)/R;
}
double DehnenBarPotentialphitorque(double R,double z,double phi,double t,
				  struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  //declare
  double smooth;
  double r, r2;
  //Get args
  double amp= *args++;
  double tform= *args++;
  double tsteady= *args++;
  double rb= *args++;
  double omegab= *args++;
  double barphi= *args++;
  //Calculate phitorque
  smooth= dehnenBarSmooth(t,tform,tsteady);
  r2= R * R + z * z;
  r= sqrt( r2 );
  if ( R <= rb )
    return 2.*amp*smooth*sin(2.*(phi-omegab*t-barphi))*(pow(r/rb,3.)-2.)\
      *R*R/r2;
  else
    return -2.*amp*smooth*sin(2.*(phi-omegab*t-barphi))*pow(rb/r,3.)*R*R/r2;
}
double DehnenBarPotentialPlanarphitorque(double R,double phi,double t,
					struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  //declare
  double smooth;
  //Get args
  double amp= *args++;
  double tform= *args++;
  double tsteady= *args++;
  double rb= *args++;
  double omegab= *args++;
  double barphi= *args++;
  //Calculate phitorque
  smooth= dehnenBarSmooth(t,tform,tsteady);
  if ( R <= rb )
    return 2.*amp*smooth*sin(2.*(phi-omegab*t-barphi))*(pow(R/rb,3.)-2.);
  else
    return -2.*amp*smooth*sin(2.*(phi-omegab*t-barphi))*pow(rb/R,3.);
}
double DehnenBarPotentialzforce(double R,double z,double phi,double t,
				struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  //declare
  double smooth;
  double r;
  //Get args
  double amp= *args++;
  double tform= *args++;
  double tsteady= *args++;
  double rb= *args++;
  double omegab= *args++;
  double barphi= *args++;
  //Calculate Rforce
  smooth= dehnenBarSmooth(t,tform,tsteady);
  r= sqrt( R * R + z * z );
  if (r <= rb )
    return -amp*smooth*cos(2.*(phi-omegab*t-barphi))*\
      (pow(r/rb,3.)+4.)*R*R*z/pow(r,4.);
  else
    return -5.*amp*smooth*cos(2.*(phi-omegab*t-barphi))\
      *pow(rb/r,3.)*R*R*z/pow(r,4);
}
double DehnenBarPotentialPlanarR2deriv(double R,double phi,double t,
				       struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  //declare
  double smooth;
  //Get args
  double amp= *args++;
  double tform= *args++;
  double tsteady= *args++;
  double rb= *args++;
  double omegab= *args++;
  double barphi= *args++;
  smooth= dehnenBarSmooth(t,tform,tsteady);
  if (R <= rb )
    return 6.*amp*smooth*cos(2.*(phi-omegab*t-barphi))*pow(R/rb,3.)/R/R;
  else
    return -12.*amp*smooth*cos(2.*(phi-omegab*t-barphi))*pow(rb/R,3.)/R/R;
}
double DehnenBarPotentialPlanarphi2deriv(double R,double phi,double t,
					 struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  //declare
  double smooth;
  //Get args
  double amp= *args++;
  double tform= *args++;
  double tsteady= *args++;
  double rb= *args++;
  double omegab= *args++;
  double barphi= *args++;
  smooth= dehnenBarSmooth(t,tform,tsteady);
  if (R <= rb )
    return -4.*amp*smooth*cos(2.*(phi-omegab*t-barphi))*(pow(R/rb,3.)-2.);
  else
    return 4.*amp*smooth*cos(2.*(phi-omegab*t-barphi))*pow(rb/R,3.);
}
double DehnenBarPotentialPlanarRphideriv(double R,double phi,double t,
					 struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  //declare
  double smooth;
  //Get args
  double amp= *args++;
  double tform= *args++;
  double tsteady= *args++;
  double rb= *args++;
  double omegab= *args++;
  double barphi= *args++;
  smooth= dehnenBarSmooth(t,tform,tsteady);
  if (R <= rb )
    return -6.*amp*smooth*sin(2.*(phi-omegab*t-barphi))*pow(R/rb,3.)/R;
  else
    return -6.*amp*smooth*sin(2.*(phi-omegab*t-barphi))*pow(rb/R,3.)/R;
}
