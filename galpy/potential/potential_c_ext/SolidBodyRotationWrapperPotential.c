#include <stdio.h>
#include <math.h>
#include <galpy_potentials.h>
//SolidBodyRotationWrapperPotential
double SolidBodyRotationWrapperPotentialRforce(double R,double z,double phi,
					  double t,
					  struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  //Calculate Rforce
  return *args * calcRforce(R,z,phi - *(args+1) * t - *(args+2),t,
		 potentialArgs->nwrapped,potentialArgs->wrappedPotentialArg);
}
double SolidBodyRotationWrapperPotentialphiforce(double R,double z,double phi,
					    double t,
					    struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  //Calculate phiforce
  return *args * calcPhiforce(R,z,phi - *(args+1) * t - *(args+2),t,
		   potentialArgs->nwrapped,potentialArgs->wrappedPotentialArg);
}
double SolidBodyRotationWrapperPotentialzforce(double R,double z,double phi,
					  double t,
					  struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  //Calculate zforce
  return *args * calczforce(R,z,phi - *(args+1) * t - *(args+2),t,
		 potentialArgs->nwrapped,potentialArgs->wrappedPotentialArg);
}
double SolidBodyRotationWrapperPotentialPlanarRforce(double R,double phi,double t,
						struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  //Calculate Rforce
  return *args * calcPlanarRforce(R,phi - *(args+1) * t - *(args+2),t,
		       potentialArgs->nwrapped,
		       potentialArgs->wrappedPotentialArg);
}
double SolidBodyRotationWrapperPotentialPlanarphiforce(double R,double phi,double t,
						  struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  //Calculate phiforce
  return *args * calcPlanarphiforce(R,phi - *(args+1) * t - *(args+2),t,
			 potentialArgs->nwrapped,
			 potentialArgs->wrappedPotentialArg);
}
double SolidBodyRotationWrapperPotentialPlanarR2deriv(double R,double phi,double t,
						 struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  //Calculate R2deriv
  return *args * calcPlanarR2deriv(R,phi - *(args+1) * t - *(args+2),t,
			potentialArgs->nwrapped,
			potentialArgs->wrappedPotentialArg);
}
double SolidBodyRotationWrapperPotentialPlanarphi2deriv(double R,double phi,
						   double t,
						   struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  //Calculate phi2deriv
  return *args * calcPlanarphi2deriv(R,phi - *(args+1) * t - *(args+2),t,
			  potentialArgs->nwrapped,
			  potentialArgs->wrappedPotentialArg);
}
double SolidBodyRotationWrapperPotentialPlanarRphideriv(double R,double phi,
						   double t,
						   struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  //Calculate Rphideriv
  return *args * calcPlanarRphideriv(R,phi - *(args+1) * t - *(args+2),t,
			  potentialArgs->nwrapped,
			  potentialArgs->wrappedPotentialArg);
}
