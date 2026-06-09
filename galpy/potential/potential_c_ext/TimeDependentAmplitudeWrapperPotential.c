#include <galpy_potentials.h>
//TimeDependentAmplitudeWrapperPotential: 1 argument, 1 tfunc
double TimeDependentAmplitudeWrapperPotentialEval(double R,double z,double phi,
					double t,
					struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  //Calculate potential, only used in actionAngle, so phi=0, t=0
  return *args * (*(*(potentialArgs->tfuncs)))(t)	\
              * evaluatePotentials(R,z,potentialArgs->nwrapped,
			                             potentialArgs->wrappedPotentialArg);
}
double TimeDependentAmplitudeWrapperPotentialRforce(double R,double z,double phi,
					  double t,
					  struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  //Calculate Rforce
  return *args * (*(*(potentialArgs->tfuncs)))(t)	\
    * calcRforce(R,z,phi,t,potentialArgs->nwrapped,
                 potentialArgs->wrappedPotentialArg);
}
double TimeDependentAmplitudeWrapperPotentialphitorque(double R,double z,double phi,
					    double t,
					    struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  //Calculate phitorque
  return *args * (*(*(potentialArgs->tfuncs)))(t)	\
    * calcphitorque(R,z,phi,t,potentialArgs->nwrapped,
                   potentialArgs->wrappedPotentialArg);
}
double TimeDependentAmplitudeWrapperPotentialzforce(double R,double z,double phi,
					  double t,
					  struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  //Calculate zforce
  return *args * (*(*(potentialArgs->tfuncs)))(t)	\
    * calczforce(R,z,phi,t,potentialArgs->nwrapped,
                 potentialArgs->wrappedPotentialArg);
}
double TimeDependentAmplitudeWrapperPotentialPlanarRforce(double R,double phi,double t,
						struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  //Calculate Rforce
  return *args * (*(*(potentialArgs->tfuncs)))(t)	\
    * calcPlanarRforce(R,phi,t,potentialArgs->nwrapped,
		                   potentialArgs->wrappedPotentialArg);
}
double TimeDependentAmplitudeWrapperPotentialPlanarphitorque(double R,double phi,double t,
						  struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  //Calculate phitorque
  return *args * (*(*(potentialArgs->tfuncs)))(t)	\
    * calcPlanarphitorque(R,phi,t,potentialArgs->nwrapped,
			                   potentialArgs->wrappedPotentialArg);
}
double TimeDependentAmplitudeWrapperPotentialPlanarR2deriv(double R,double phi,double t,
						 struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  //Calculate R2deriv
  return *args * (*(*(potentialArgs->tfuncs)))(t)	\
    * calcPlanarR2deriv(R,phi,t,potentialArgs->nwrapped,
			                  potentialArgs->wrappedPotentialArg);
}
double TimeDependentAmplitudeWrapperPotentialPlanarphi2deriv(double R,double phi,
						   double t,
						   struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  //Calculate phi2deriv
  return *args * (*(*(potentialArgs->tfuncs)))(t)	\
    * calcPlanarphi2deriv(R,phi,t,potentialArgs->nwrapped,
			                    potentialArgs->wrappedPotentialArg);
}
double TimeDependentAmplitudeWrapperPotentialPlanarRphideriv(double R,double phi,
						   double t,
						   struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  //Calculate Rphideriv
  return *args * (*(*(potentialArgs->tfuncs)))(t)	\
    * calcPlanarRphideriv(R,phi,t,potentialArgs->nwrapped,
			                    potentialArgs->wrappedPotentialArg);
}

// --- 3D Hessian for the variational equations: modulation x wrapped Hessian ---
double TimeDependentAmplitudeWrapperPotentialR2deriv(double R,double z,double phi,double t,
				 struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  return *args * (*(*(potentialArgs->tfuncs)))(t) \
    * calcR2deriv(R,z,phi,t,
		 potentialArgs->nwrapped,potentialArgs->wrappedPotentialArg);
}
double TimeDependentAmplitudeWrapperPotentialz2deriv(double R,double z,double phi,double t,
				 struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  return *args * (*(*(potentialArgs->tfuncs)))(t) \
    * calcz2deriv(R,z,phi,t,
		 potentialArgs->nwrapped,potentialArgs->wrappedPotentialArg);
}
double TimeDependentAmplitudeWrapperPotentialRzderiv(double R,double z,double phi,double t,
				 struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  return *args * (*(*(potentialArgs->tfuncs)))(t) \
    * calcRzderiv(R,z,phi,t,
		 potentialArgs->nwrapped,potentialArgs->wrappedPotentialArg);
}
double TimeDependentAmplitudeWrapperPotentialphi2deriv(double R,double z,double phi,double t,
				 struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  return *args * (*(*(potentialArgs->tfuncs)))(t) \
    * calcphi2deriv(R,z,phi,t,
		 potentialArgs->nwrapped,potentialArgs->wrappedPotentialArg);
}
double TimeDependentAmplitudeWrapperPotentialRphideriv(double R,double z,double phi,double t,
				 struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  return *args * (*(*(potentialArgs->tfuncs)))(t) \
    * calcRphideriv(R,z,phi,t,
		 potentialArgs->nwrapped,potentialArgs->wrappedPotentialArg);
}
double TimeDependentAmplitudeWrapperPotentialzphideriv(double R,double z,double phi,double t,
				 struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  return *args * (*(*(potentialArgs->tfuncs)))(t) \
    * calczphideriv(R,z,phi,t,
		 potentialArgs->nwrapped,potentialArgs->wrappedPotentialArg);
}
