#include <stdio.h>
#include <math.h>
#include <galpy_potentials.h>
//DehnenSmoothWrapperPotential
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
double DehnenSmoothWrapperPotentialEval(double R,double z,double phi,
					double t,
					struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  //Calculate potential, only used in actionAngle, so phi=0, t=0
  return *args * dehnenSmooth(t,*(args+1),*(args+2))	\
    * evaluatePotentials(R,z,
			 potentialArgs->nwrapped,
			 potentialArgs->wrappedPotentialArg);
}
double DehnenSmoothWrapperPotentialRforce(double R,double z,double phi,
					  double t,
					  struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  //Calculate Rforce
  return *args * dehnenSmooth(t,*(args+1),*(args+2))	\
    * calcRforce(R,z,phi,t,
		 potentialArgs->nwrapped,potentialArgs->wrappedPotentialArg);
}
double DehnenSmoothWrapperPotentialphiforce(double R,double z,double phi,
					    double t,
					    struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  //Calculate phiforce
  return *args * dehnenSmooth(t,*(args+1),*(args+2))	\
    * calcPhiforce(R,z,phi,t,
		   potentialArgs->nwrapped,potentialArgs->wrappedPotentialArg);
}
double DehnenSmoothWrapperPotentialzforce(double R,double z,double phi,
					  double t,
					  struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  //Calculate zforce
  return *args * dehnenSmooth(t,*(args+1),*(args+2))	\
    * calczforce(R,z,phi,t,
		 potentialArgs->nwrapped,potentialArgs->wrappedPotentialArg);
}
double DehnenSmoothWrapperPotentialPlanarRforce(double R,double phi,double t,
						struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  //Calculate Rforce
  return *args * dehnenSmooth(t,*(args+1),*(args+2))	\
    * calcPlanarRforce(R,phi,t,
		       potentialArgs->nwrapped,
		       potentialArgs->wrappedPotentialArg);
}
double DehnenSmoothWrapperPotentialPlanarphiforce(double R,double phi,double t,
						  struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  //Calculate phiforce
  return *args * dehnenSmooth(t,*(args+1),*(args+2))	\
    * calcPlanarphiforce(R,phi,t,
			 potentialArgs->nwrapped,
			 potentialArgs->wrappedPotentialArg);
}
double DehnenSmoothWrapperPotentialPlanarR2deriv(double R,double phi,double t,
						 struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  //Calculate R2deriv
  return *args * dehnenSmooth(t,*(args+1),*(args+2))	\
    * calcPlanarR2deriv(R,phi,t,
			potentialArgs->nwrapped,
			potentialArgs->wrappedPotentialArg);
}
double DehnenSmoothWrapperPotentialPlanarphi2deriv(double R,double phi,
						   double t,
						   struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  //Calculate phi2deriv
  return *args * dehnenSmooth(t,*(args+1),*(args+2))	\
    * calcPlanarphi2deriv(R,phi,t,
			  potentialArgs->nwrapped,
			  potentialArgs->wrappedPotentialArg);
}
double DehnenSmoothWrapperPotentialPlanarRphideriv(double R,double phi,
						   double t,
						   struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  //Calculate Rphideriv
  return *args * dehnenSmooth(t,*(args+1),*(args+2))	\
    * calcPlanarRphideriv(R,phi,t,
			  potentialArgs->nwrapped,
			  potentialArgs->wrappedPotentialArg);
}
