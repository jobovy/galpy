#include <math.h>
#include <galpy_potentials.h>
//GaussianAmplitudeWrapperPotential
//3 parameters: amp, to, sigma^2
double gaussSmooth(double t,double to, double sigma2){
  return exp(-0.5*(t-to)*(t-to)/sigma2);
}
double GaussianAmplitudeWrapperPotentialEval(double R,double z,double phi,
					double t,
					struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  //Calculate potential, only used in actionAngle, so phi=0, t=0
  return *args * gaussSmooth(t,*(args+1),*(args+2))	\
    * evaluatePotentials(R,z,
			 potentialArgs->nwrapped,
			 potentialArgs->wrappedPotentialArg);
}
double GaussianAmplitudeWrapperPotentialRforce(double R,double z,double phi,
					  double t,
					  struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  //Calculate Rforce
  return *args * gaussSmooth(t,*(args+1),*(args+2))	\
    * calcRforce(R,z,phi,t,
		 potentialArgs->nwrapped,potentialArgs->wrappedPotentialArg);
}
double GaussianAmplitudeWrapperPotentialphiforce(double R,double z,double phi,
					    double t,
					    struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  //Calculate phiforce
  return *args * gaussSmooth(t,*(args+1),*(args+2))	\
    * calcPhiforce(R,z,phi,t,
		   potentialArgs->nwrapped,potentialArgs->wrappedPotentialArg);
}
double GaussianAmplitudeWrapperPotentialzforce(double R,double z,double phi,
					  double t,
					  struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  //Calculate zforce
  return *args * gaussSmooth(t,*(args+1),*(args+2))	\
    * calczforce(R,z,phi,t,
		 potentialArgs->nwrapped,potentialArgs->wrappedPotentialArg);
}
double GaussianAmplitudeWrapperPotentialPlanarRforce(double R,double phi,double t,
						struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  //Calculate Rforce
  return *args * gaussSmooth(t,*(args+1),*(args+2))	\
    * calcPlanarRforce(R,phi,t,
		       potentialArgs->nwrapped,
		       potentialArgs->wrappedPotentialArg);
}
double GaussianAmplitudeWrapperPotentialPlanarphiforce(double R,double phi,double t,
						  struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  //Calculate phiforce
  return *args * gaussSmooth(t,*(args+1),*(args+2))	\
    * calcPlanarphiforce(R,phi,t,
			 potentialArgs->nwrapped,
			 potentialArgs->wrappedPotentialArg);
}
double GaussianAmplitudeWrapperPotentialPlanarR2deriv(double R,double phi,double t,
						 struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  //Calculate R2deriv
  return *args * gaussSmooth(t,*(args+1),*(args+2))	\
    * calcPlanarR2deriv(R,phi,t,
			potentialArgs->nwrapped,
			potentialArgs->wrappedPotentialArg);
}
double GaussianAmplitudeWrapperPotentialPlanarphi2deriv(double R,double phi,
						   double t,
						   struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  //Calculate phi2deriv
  return *args * gaussSmooth(t,*(args+1),*(args+2))	\
    * calcPlanarphi2deriv(R,phi,t,
			  potentialArgs->nwrapped,
			  potentialArgs->wrappedPotentialArg);
}
double GaussianAmplitudeWrapperPotentialPlanarRphideriv(double R,double phi,
						   double t,
						   struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  //Calculate Rphideriv
  return *args * gaussSmooth(t,*(args+1),*(args+2))	\
    * calcPlanarRphideriv(R,phi,t,
			  potentialArgs->nwrapped,
			  potentialArgs->wrappedPotentialArg);
}
