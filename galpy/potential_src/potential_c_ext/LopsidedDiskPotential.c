#include <math.h>
#include <galpy_potentials.h>
//LopsidedDiskPotential
//
inline double LopsidedDiskSmooth(double t,double tform, double tsteady){
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
double LopsidedDiskPotentialRforce(double R,double phi,double t,
				   struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  //declare
  double smooth;
  //Get args
  double amp= *args++;
  double tform= *args++;
  double tsteady= *args++;
  double phio= *args++;
  double p= *args++;
  double phib= *args;
  //Calculate Rforce
  smooth= LopsidedDiskSmooth(t,tform,tsteady);
  return -amp * smooth * p * phio * pow(R,p-1.) 
    * cos( phi - phib);
}
double LopsidedDiskPotentialphiforce(double R,double phi,double t,
				     struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  //declare
  double smooth;
  //Get args
  double amp= *args++;
  double tform= *args++;
  double tsteady= *args++;
  double phio= *args++;
  double p= *args++;
  double phib= *args;
  //Calculate phiforce
  smooth= LopsidedDiskSmooth(t,tform,tsteady);
  return amp * smooth * phio * pow(R,p) 
    * sin( phi-phib);
}
double LopsidedDiskPotentialR2deriv(double R,double phi,double t,
				    struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  //declare
  double smooth;
  //Get args
  double amp= *args++;
  double tform= *args++;
  double tsteady= *args++;
  double phio= *args++;
  double p= *args++;
  double phib= *args;
  //Calculate Rforce
  smooth= LopsidedDiskSmooth(t,tform,tsteady);
  return amp * smooth * p * ( p - 1) * phio * pow(R,p-2.)
    * cos(phi - phib);
} 
double LopsidedDiskPotentialphi2deriv(double R,double phi,double t,
				      struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  //declare
  double smooth;
  //Get args
  double amp= *args++;
  double tform= *args++;
  double tsteady= *args++;
  double phio= *args++;
  double p= *args++;
  double phib= *args;
  //Calculate Rforce
  smooth= LopsidedDiskSmooth(t,tform,tsteady);
  return - amp * smooth * phio * pow(R,p)
    * cos( phi - phib );
} 
double LopsidedDiskPotentialRphideriv(double R,double phi,double t,
				      struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  //declare
  double smooth;
  //Get args
  double amp= *args++;
  double tform= *args++;
  double tsteady= *args++;
  double phio= *args++;
  double p= *args++;
  double phib= *args;
  //Calculate Rforce
  smooth= LopsidedDiskSmooth(t,tform,tsteady);
  return - amp * smooth * p * phio * pow(R,p-1.)
    * sin( phi - phib );
} 
