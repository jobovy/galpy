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
// --- Second derivatives for the variational equations ---
// Separable: Phi(R,z) = Phi_w(R,0) + Phi_w(Rp,z) - Phi_w(Rp,0), so the
// Hessian splits into the wrapped potential's own second derivatives along
// the two reference curves: Phi_RR(R,z) = Phi_w,RR(R,0) and
// Phi_zz(R,z) = Phi_w,zz(Rp,z), while Phi_Rz = 0 identically (left NULL in
// the parser; the NULL-safe aggregators return 0 for it), as are the
// (axisymmetric) phi-derivatives. Direct transcriptions of the Python
// _R2deriv/_z2deriv.
double CylindricallySeparablePotentialWrapperPotentialR2deriv(double R,double z,double phi,
                        double t,
                        struct potentialArg * potentialArgs){
    double * args= potentialArgs->args;
    return *args * calcR2deriv(R,0.0,0.0,0.0,
                    potentialArgs->nwrapped,
                    potentialArgs->wrappedPotentialArg);
}
double CylindricallySeparablePotentialWrapperPotentialz2deriv(double R,double z,double phi,
                        double t,
                        struct potentialArg * potentialArgs){
    double * args= potentialArgs->args;
    return *args * calcz2deriv(*(args+1),z,0.0,0.0,
                    potentialArgs->nwrapped,
                    potentialArgs->wrappedPotentialArg);
}
double CylindricallySeparablePotentialWrapperPotentialPlanarR2deriv(double R,double phi,
                                                double t,
                                                struct potentialArg * potentialArgs){
    double * args= potentialArgs->args;
    return *args * calcPlanarR2deriv(R,0.0,0.0,
                    potentialArgs->nwrapped,
                    potentialArgs->wrappedPotentialArg);
}
