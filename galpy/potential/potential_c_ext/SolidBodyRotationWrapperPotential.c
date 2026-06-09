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
double SolidBodyRotationWrapperPotentialphitorque(double R,double z,double phi,
					    double t,
					    struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  //Calculate phitorque
  return *args * calcphitorque(R,z,phi - *(args+1) * t - *(args+2),t,
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
double SolidBodyRotationWrapperPotentialPlanarphitorque(double R,double phi,double t,
						  struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  //Calculate phitorque
  return *args * calcPlanarphitorque(R,phi - *(args+1) * t - *(args+2),t,
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

// --- 3D Hessian for the variational equations: modulation x wrapped Hessian ---
double SolidBodyRotationWrapperPotentialR2deriv(double R,double z,double phi,double t,
				 struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  return *args \
    * calcR2deriv(R,z,phi - *(args+1) * t - *(args+2),t,
		 potentialArgs->nwrapped,potentialArgs->wrappedPotentialArg);
}
double SolidBodyRotationWrapperPotentialz2deriv(double R,double z,double phi,double t,
				 struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  return *args \
    * calcz2deriv(R,z,phi - *(args+1) * t - *(args+2),t,
		 potentialArgs->nwrapped,potentialArgs->wrappedPotentialArg);
}
double SolidBodyRotationWrapperPotentialRzderiv(double R,double z,double phi,double t,
				 struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  return *args \
    * calcRzderiv(R,z,phi - *(args+1) * t - *(args+2),t,
		 potentialArgs->nwrapped,potentialArgs->wrappedPotentialArg);
}
double SolidBodyRotationWrapperPotentialphi2deriv(double R,double z,double phi,double t,
				 struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  return *args \
    * calcphi2deriv(R,z,phi - *(args+1) * t - *(args+2),t,
		 potentialArgs->nwrapped,potentialArgs->wrappedPotentialArg);
}
double SolidBodyRotationWrapperPotentialRphideriv(double R,double z,double phi,double t,
				 struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  return *args \
    * calcRphideriv(R,z,phi - *(args+1) * t - *(args+2),t,
		 potentialArgs->nwrapped,potentialArgs->wrappedPotentialArg);
}
double SolidBodyRotationWrapperPotentialzphideriv(double R,double z,double phi,double t,
				 struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  return *args \
    * calczphideriv(R,z,phi - *(args+1) * t - *(args+2),t,
		 potentialArgs->nwrapped,potentialArgs->wrappedPotentialArg);
}
