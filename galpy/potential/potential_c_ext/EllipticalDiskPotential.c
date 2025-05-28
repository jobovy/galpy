#include <math.h>
#include <galpy_potentials.h>
//EllipticalDiskPotential
//
double EllipticalDiskSmooth(double t,double tform, double tsteady){
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
double EllipticalDiskPotentialRforce(double R,double phi,double t,
				     struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  //declare
  double smooth;
  //Get args
  double amp= *args++;
  double tform= *args++;
  double tsteady= *args++;
  double twophio= *args++;
  double p= *args++;
  double phib= *args;
  //Calculate Rforce
  smooth= EllipticalDiskSmooth(t,tform,tsteady);
  return -amp * smooth * p * twophio / 2. * pow(R,p-1.)
    * cos( 2. * (phi - phib));
}
double EllipticalDiskPotentialphitorque(double R,double phi,double t,
				       struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  //declare
  double smooth;
  //Get args
  double amp= *args++;
  double tform= *args++;
  double tsteady= *args++;
  double twophio= *args++;
  double p= *args++;
  double phib= *args;
  //Calculate phitorque
  smooth= EllipticalDiskSmooth(t,tform,tsteady);
  return amp * smooth * twophio * pow(R,p)
    * sin( 2. * (phi-phib));
}
double EllipticalDiskPotentialR2deriv(double R,double phi,double t,
				      struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  //declare
  double smooth;
  //Get args
  double amp= *args++;
  double tform= *args++;
  double tsteady= *args++;
  double twophio= *args++;
  double p= *args++;
  double phib= *args;
  //Calculate Rforce
  smooth= EllipticalDiskSmooth(t,tform,tsteady);
  return amp * smooth * p * ( p - 1) * twophio / 2. * pow(R,p-2.)
    * cos( 2. * ( phi - phib ) );
}
double EllipticalDiskPotentialphi2deriv(double R,double phi,double t,
					struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  //declare
  double smooth;
  //Get args
  double amp= *args++;
  double tform= *args++;
  double tsteady= *args++;
  double twophio= *args++;
  double p= *args++;
  double phib= *args;
  //Calculate Rforce
  smooth= EllipticalDiskSmooth(t,tform,tsteady);
  return - 2. * amp * smooth * twophio * pow(R,p)
    * cos( 2. * ( phi - phib ) );
}
double EllipticalDiskPotentialRphideriv(double R,double phi,double t,
					struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  //declare
  double smooth;
  //Get args
  double amp= *args++;
  double tform= *args++;
  double tsteady= *args++;
  double twophio= *args++;
  double p= *args++;
  double phib= *args;
  //Calculate Rforce
  smooth= EllipticalDiskSmooth(t,tform,tsteady);
  return - amp * smooth * p * twophio * pow(R,p-1.)
    * sin( 2. * ( phi - phib ) );
}
