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
  if ( r <= rb )
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
// Full-3D Hessian for the 3D variational equations (integrate_dxdv).
double DehnenBarPotentialR2deriv(double R,double z,double phi,double t,
				 struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  double smooth, r2, r4, r6, R2, z2, prefac;
  //Get args
  double amp= *args++;
  double tform= *args++;
  double tsteady= *args++;
  double rb= *args++;
  double omegab= *args++;
  double barphi= *args++;
  smooth= dehnenBarSmooth(t,tform,tsteady);
  R2= R * R;
  z2= z * z;
  r2= R2 + z2;
  r4= r2 * r2;
  r6= r4 * r2;
  prefac= amp * smooth * cos(2.*(phi-omegab*t-barphi));
  if ( r2 <= rb * rb )
    return prefac * ( pow(r2,1.5)/(rb*rb*rb)
		      * ( (9.*R2+2.*z2)/r4 - R2/r6*(3.*R2+2.*z2) )
		      + 4.*z2/r6*(4.*R2-r2) );
  else
    return prefac * pow(rb,3.)*pow(r2,-1.5)/r6
      * ( (r2-7.*R2)*(3.*R2-2.*z2) + 6.*R2*r2 );
}
double DehnenBarPotentialphi2deriv(double R,double z,double phi,double t,
				   struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  double smooth, r2, R2, prefac;
  //Get args
  double amp= *args++;
  double tform= *args++;
  double tsteady= *args++;
  double rb= *args++;
  double omegab= *args++;
  double barphi= *args++;
  smooth= dehnenBarSmooth(t,tform,tsteady);
  R2= R * R;
  r2= R2 + z * z;
  prefac= 4. * amp * smooth * cos(2.*(phi-omegab*t-barphi));
  if ( r2 <= rb * rb )
    return -prefac * ( pow(r2,1.5)/(rb*rb*rb) - 2. ) * R2/r2;
  else
    return prefac * pow(rb,3.)*pow(r2,-1.5) * R2/r2;
}
double DehnenBarPotentialRphideriv(double R,double z,double phi,double t,
				   struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  double smooth, r2, r4, R2, z2, prefac;
  //Get args
  double amp= *args++;
  double tform= *args++;
  double tsteady= *args++;
  double rb= *args++;
  double omegab= *args++;
  double barphi= *args++;
  smooth= dehnenBarSmooth(t,tform,tsteady);
  R2= R * R;
  z2= z * z;
  r2= R2 + z2;
  r4= r2 * r2;
  prefac= -2. * amp * smooth * sin(2.*(phi-omegab*t-barphi));
  if ( r2 <= rb * rb )
    return prefac * ( pow(r2,1.5)/(rb*rb*rb)*R*(3.*R2+2.*z2)
		      - 4.*R*z2 ) / r4;
  else
    return prefac * pow(rb,3.)*pow(r2,-1.5) * R/r4 * (3.*R2-2.*z2);
}
double DehnenBarPotentialz2deriv(double R,double z,double phi,double t,
				 struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  double smooth, r2, r6, R2, z2, prefac;
  //Get args
  double amp= *args++;
  double tform= *args++;
  double tsteady= *args++;
  double rb= *args++;
  double omegab= *args++;
  double barphi= *args++;
  smooth= dehnenBarSmooth(t,tform,tsteady);
  R2= R * R;
  z2= z * z;
  r2= R2 + z2;
  r6= r2 * r2 * r2;
  prefac= amp * smooth * cos(2.*(phi-omegab*t-barphi));
  if ( r2 <= rb * rb )
    return prefac * R2/r6
      * ( pow(r2,1.5)/(rb*rb*rb)*(r2-z2) + 4.*(r2-4.*z2) );
  else
    return prefac * 5. * pow(rb,3.)*pow(r2,-1.5) * R2/r6 * (r2-7.*z2);
}
double DehnenBarPotentialRzderiv(double R,double z,double phi,double t,
				 struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  double smooth, r2, r6, R2, prefac;
  //Get args
  double amp= *args++;
  double tform= *args++;
  double tsteady= *args++;
  double rb= *args++;
  double omegab= *args++;
  double barphi= *args++;
  smooth= dehnenBarSmooth(t,tform,tsteady);
  R2= R * R;
  r2= R2 + z * z;
  r6= r2 * r2 * r2;
  prefac= amp * smooth * cos(2.*(phi-omegab*t-barphi));
  if ( r2 <= rb * rb )
    return prefac * R*z/r6
      * ( pow(r2,1.5)/(rb*rb*rb)*(2.*r2-R2) + 8.*(r2-2.*R2) );
  else
    return prefac * 5. * pow(rb,3.)*pow(r2,-1.5) * R*z/r6 * (2.*r2-7.*R2);
}
double DehnenBarPotentialzphideriv(double R,double z,double phi,double t,
				   struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  double smooth, r2, r4, R2, prefac;
  //Get args
  double amp= *args++;
  double tform= *args++;
  double tsteady= *args++;
  double rb= *args++;
  double omegab= *args++;
  double barphi= *args++;
  smooth= dehnenBarSmooth(t,tform,tsteady);
  R2= R * R;
  r2= R2 + z * z;
  r4= r2 * r2;
  prefac= -2. * amp * smooth * sin(2.*(phi-omegab*t-barphi));
  if ( r2 <= rb * rb )
    return prefac * ( pow(r2,1.5)/(rb*rb*rb) + 4. ) * R2*z/r4;
  else
    return prefac * 5. * pow(rb,3.)*pow(r2,-1.5) * R2*z/r4;
}
