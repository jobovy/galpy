#include <math.h>
#include <galpy_potentials.h>
//CorotatingRotationWrapperPotential
double CorotatingRotationWrapperPotentialRforce(double R,double z,double phi,
					  double t,
					  struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  //Calculate Rforce
  return *args * calcRforce(R,z,phi - *(args+1) / R * ( t - *(args+3) ) \
			    - *(args+2),t,
		 potentialArgs->nwrapped,potentialArgs->wrappedPotentialArg);
}
double CorotatingRotationWrapperPotentialphiforce(double R,double z,double phi,
					    double t,
					    struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  //Calculate phiforce
  return *args * calcPhiforce(R,z,phi - *(args+1) / R * ( t - *(args+3) ) \
			      - *(args+2),t,
		   potentialArgs->nwrapped,potentialArgs->wrappedPotentialArg);
}
double CorotatingRotationWrapperPotentialzforce(double R,double z,double phi,
					  double t,
					  struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  //Calculate zforce
  return *args * calczforce(R,z,phi - *(args+1) / R * ( t - *(args+3) ) \
			    - *(args+2),t,
		 potentialArgs->nwrapped,potentialArgs->wrappedPotentialArg);
}
double CorotatingRotationWrapperPotentialPlanarRforce(double R,double phi,double t,
						struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  //Calculate Rforce
  return *args * calcPlanarRforce(R,phi - *(args+1) / R * ( t - *(args+3) ) \
				  - *(args+2),t,
				  potentialArgs->nwrapped,
				  potentialArgs->wrappedPotentialArg);
}
double CorotatingRotationWrapperPotentialPlanarphiforce(double R,double phi,double t,
						  struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  //Calculate phiforce
  return *args * calcPlanarphiforce(R,phi - *(args+1) / R * ( t - *(args+3) ) \
				    - *(args+2),t,
				    potentialArgs->nwrapped,
				    potentialArgs->wrappedPotentialArg);
}
double CorotatingRotationWrapperPotentialPlanarR2deriv(double R,double phi,double t,
						 struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  //Calculate R2deriv
  return *args * calcPlanarR2deriv(R,phi - *(args+1) / R * ( t - *(args+3) ) \
				   - *(args+2),t,
				   potentialArgs->nwrapped,
				   potentialArgs->wrappedPotentialArg);
}
double CorotatingRotationWrapperPotentialPlanarphi2deriv(double R,double phi,
						   double t,
						   struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  //Calculate phi2deriv
  return *args * calcPlanarphi2deriv(R,phi - *(args+1) / R * ( t - *(args+3) )\
				     - *(args+2),t,
				     potentialArgs->nwrapped,
				     potentialArgs->wrappedPotentialArg);
}
double CorotatingRotationWrapperPotentialPlanarRphideriv(double R,double phi,
						   double t,
						   struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  //Calculate Rphideriv
  return *args * calcPlanarRphideriv(R,phi - *(args+1) / R * ( t - *(args+3) )\
				     - *(args+2),t,
				     potentialArgs->nwrapped,
				     potentialArgs->wrappedPotentialArg);
}
