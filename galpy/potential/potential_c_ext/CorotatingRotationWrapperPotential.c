#include <math.h>
#include <galpy_potentials.h>
//CorotatingRotationWrapperPotential
// 5 arguments: amp, vpo, beta, pa, to
double CorotatingRotationWrapperPotentialRforce(double R,double z,double phi,
					  double t,
					  struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  //Calculate Rforce
  double phi_new= phi-*(args+1) * pow(R,*(args+2)-1) * (t-*(args+4))\
    - *(args+3);
  return *args * ( calcRforce(R,z,phi_new,t,potentialArgs->nwrapped,
			      potentialArgs->wrappedPotentialArg)	\
		   - calcphitorque(R,z,phi_new,t,
		 potentialArgs->nwrapped,potentialArgs->wrappedPotentialArg)\
		   * *(args+1) * ( *(args+2) - 1 ) * pow(R,*(args+2)-2) * (t-*(args+4)));
}
double CorotatingRotationWrapperPotentialphitorque(double R,double z,double phi,
					    double t,
					    struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  //Calculate phitorque
  return *args * calcphitorque(R,z,
			      phi-*(args+1) * pow(R,*(args+2)-1) * (t-*(args+4)) \
			      - *(args+3),t,
		   potentialArgs->nwrapped,potentialArgs->wrappedPotentialArg);
}
double CorotatingRotationWrapperPotentialzforce(double R,double z,double phi,
					  double t,
					  struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  //Calculate zforce
  return *args * calczforce(R,z,
			    phi-*(args+1) * pow(R,*(args+2)-1) * (t-*(args+4)) \
			    - *(args+3),t,
		 potentialArgs->nwrapped,potentialArgs->wrappedPotentialArg);
}
double CorotatingRotationWrapperPotentialPlanarRforce(double R,double phi,double t,
						struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  //Calculate Rforce
  double phi_new= phi-*(args+1) * pow(R,*(args+2)-1) * (t-*(args+4))\
    - *(args+3);
  return *args * ( calcPlanarRforce(R,phi_new,t,potentialArgs->nwrapped,
				    potentialArgs->wrappedPotentialArg)	\
		   - calcPlanarphitorque(R,phi_new,t,potentialArgs->nwrapped,
				      potentialArgs->wrappedPotentialArg) \
		   * *(args+1) * ( *(args+2) - 1 ) * pow(R,*(args+2)-2) * (t-*(args+4)));
}
double CorotatingRotationWrapperPotentialPlanarphitorque(double R,double phi,double t,
						  struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  //Calculate phitorque
  return *args * calcPlanarphitorque(R,
				    phi-*(args+1) * pow(R,*(args+2)-1) * (t-*(args+4)) \
				    - *(args+3),t,
				    potentialArgs->nwrapped,
				    potentialArgs->wrappedPotentialArg);
}
double CorotatingRotationWrapperPotentialPlanarR2deriv(double R,double phi,double t,
						 struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  //Calculate R2deriv
  double phi_new= phi-*(args+1) * pow(R,*(args+2)-1) * (t-*(args+4))\
    - *(args+3);
  double phiRderiv= -*(args+1) * (*(args+2)-1) * pow(R,*(args+2)-2) \
    * (t-*(args+4));
  return *args * (calcPlanarR2deriv(R,phi_new,t,
				    potentialArgs->nwrapped,
				    potentialArgs->wrappedPotentialArg)
		  + 2. * phiRderiv * calcPlanarRphideriv(R,phi_new,t,
				    potentialArgs->nwrapped,
				    potentialArgs->wrappedPotentialArg)
		  + phiRderiv * phiRderiv * calcPlanarphi2deriv(R,phi_new,t,
				    potentialArgs->nwrapped,
				    potentialArgs->wrappedPotentialArg)
		  + calcPlanarphitorque(R,phi_new,t,
				    potentialArgs->nwrapped,
				    potentialArgs->wrappedPotentialArg)
		  * *(args+1) * (*(args+2)-1) * (*(args+2)-2)
		  * pow(R,*(args+2)-3) * (t-*(args+4)));
}
double CorotatingRotationWrapperPotentialPlanarphi2deriv(double R,double phi,
						   double t,
						   struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  //Calculate phi2deriv
  return *args * calcPlanarphi2deriv(R,
				     phi-*(args+1) * pow(R,*(args+2)-1) * (t-*(args+4))	\
				     - *(args+3),t,
				     potentialArgs->nwrapped,
				     potentialArgs->wrappedPotentialArg);
}
double CorotatingRotationWrapperPotentialPlanarRphideriv(double R,double phi,
						   double t,
						   struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  //Calculate Rphideriv
  double phi_new= phi-*(args+1) * pow(R,*(args+2)-1) * (t-*(args+4))\
    - *(args+3);
  return *args * ( calcPlanarRphideriv(R,phi_new,t,potentialArgs->nwrapped,
				       potentialArgs->wrappedPotentialArg)
		   - calcPlanarphi2deriv(R,phi_new,t,potentialArgs->nwrapped,
				       potentialArgs->wrappedPotentialArg)
		   * *(args+1) * (*(args+2)-1) * pow(R,*(args+2)-2)	\
		   * (t-*(args+4)));
}
