#include <galpy_potentials.h>
// CylindricallySeparablePotentialWrapper: amp, Rp, refpot
double CylindricallySeparablePotentialWrapperPotentialEval(double R,double z,double phi,
					  double t,
					  struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  return *args * (evaluatePotentials(R,0.0,potentialArgs->nwrapped,
			                         potentialArgs->wrappedPotentialArg)
                  + evaluatePotentials(*(args+1),z,potentialArgs->nwrapped,
                                      potentialArgs->wrappedPotentialArg)
                  - *(args+2) );
}
double CylindricallySeparablePotentialWrapperPotentialRforce(double R,double z,double phi,
					    double t,
					    struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
    return *args * calcRforce(R,0.0,0.0,0.0,
                    potentialArgs->nwrapped,
                    potentialArgs->wrappedPotentialArg);
}
double CylindricallySeparablePotentialWrapperPotentialzforce(double R,double z,double phi,
                        double t,
                        struct potentialArg * potentialArgs){
    double * args= potentialArgs->args;
    return *args * calczforce(*(args+1),z,0.0,0.0,
                    potentialArgs->nwrapped,
                    potentialArgs->wrappedPotentialArg);
}
double CylindricallySeparablePotentialWrapperPotentialPlanarRforce(double R,double phi,
                                                double t,
                                                struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
    return *args * calcPlanarRforce(R,0.0,0.0,
                    potentialArgs->nwrapped,
                    potentialArgs->wrappedPotentialArg);
}
