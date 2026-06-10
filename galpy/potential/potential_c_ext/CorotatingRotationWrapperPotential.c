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
// --- 3D Hessian for the variational equations ---
// The wrapper evaluates the wrapped potential at the shifted azimuth
//   phi' = phi - s(R,t),  s(R,t) = vpo * R^(beta-1) * (t-to) + pa
// (the pattern ANGULAR speed at radius R is Vp(R)/R = vpo * R^(beta-1), so the
// accumulated rotation angle s depends on R unless beta==1). Because s depends
// on R, the radial derivative acting on any function of (R,z,phi',t) obeys the
// chain rule  d/dR -> d/dR - s_R * d/dphi', with
//   s_R  = dsdR   = vpo * (beta-1) * R^(beta-2) * (t-to)
//   s_RR = d2sdR2 = vpo * (beta-1) * (beta-2) * R^(beta-3) * (t-to)
// while d/dz and d/dphi leave phi' untouched (dphi'/dphi = 1, dphi'/dz = 0).
// Writing Phi(R,z,phi,t) = amp * Phi_w(R,z,phi',t) and applying the chain rule
// twice gives the six cylindrical second derivatives:
//   Phi_RR     = amp * ( Phi_w,RR - 2 s_R Phi_w,Rphi' + s_R^2 Phi_w,phi'phi'
//                        - s_RR Phi_w,phi' )
//   Phi_zz     = amp *   Phi_w,zz
//   Phi_Rz     = amp * ( Phi_w,Rz - s_R Phi_w,zphi' )
//   Phi_phiphi = amp *   Phi_w,phi'phi'
//   Phi_Rphi   = amp * ( Phi_w,Rphi' - s_R Phi_w,phi'phi' )
//   Phi_zphi   = amp *   Phi_w,zphi'
// Below, phiRderiv = dphi'/dR = -s_R, and the s_RR term enters through the
// wrapped phitorque ( = -Phi_w,phi' ), exactly as in the Planar versions above.
double CorotatingRotationWrapperPotentialR2deriv(double R,double z,double phi,
						 double t,
						 struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  //Calculate R2deriv
  double phi_new= phi-*(args+1) * pow(R,*(args+2)-1) * (t-*(args+4))\
    - *(args+3);
  double phiRderiv= -*(args+1) * (*(args+2)-1) * pow(R,*(args+2)-2) \
    * (t-*(args+4));
  return *args * (calcR2deriv(R,z,phi_new,t,
			      potentialArgs->nwrapped,
			      potentialArgs->wrappedPotentialArg)
		  + 2. * phiRderiv * calcRphideriv(R,z,phi_new,t,
			      potentialArgs->nwrapped,
			      potentialArgs->wrappedPotentialArg)
		  + phiRderiv * phiRderiv * calcphi2deriv(R,z,phi_new,t,
			      potentialArgs->nwrapped,
			      potentialArgs->wrappedPotentialArg)
		  + calcphitorque(R,z,phi_new,t,
			      potentialArgs->nwrapped,
			      potentialArgs->wrappedPotentialArg)
		  * *(args+1) * (*(args+2)-1) * (*(args+2)-2)
		  * pow(R,*(args+2)-3) * (t-*(args+4)));
}
double CorotatingRotationWrapperPotentialz2deriv(double R,double z,double phi,
						 double t,
						 struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  //Calculate z2deriv: s is z-independent, so this is just the wrapped z2deriv
  //at the shifted azimuth
  return *args * calcz2deriv(R,z,
			     phi-*(args+1) * pow(R,*(args+2)-1) * (t-*(args+4)) \
			     - *(args+3),t,
			     potentialArgs->nwrapped,
			     potentialArgs->wrappedPotentialArg);
}
double CorotatingRotationWrapperPotentialRzderiv(double R,double z,double phi,
						 double t,
						 struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  //Calculate Rzderiv = Phi_w,Rz - s_R * Phi_w,zphi' (chain rule on d/dR)
  double phi_new= phi-*(args+1) * pow(R,*(args+2)-1) * (t-*(args+4))\
    - *(args+3);
  double phiRderiv= -*(args+1) * (*(args+2)-1) * pow(R,*(args+2)-2) \
    * (t-*(args+4));
  return *args * ( calcRzderiv(R,z,phi_new,t,potentialArgs->nwrapped,
			       potentialArgs->wrappedPotentialArg)
		   + phiRderiv * calczphideriv(R,z,phi_new,t,
			       potentialArgs->nwrapped,
			       potentialArgs->wrappedPotentialArg));
}
double CorotatingRotationWrapperPotentialphi2deriv(double R,double z,double phi,
						   double t,
						   struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  //Calculate phi2deriv: dphi'/dphi = 1, so this is just the wrapped phi2deriv
  //at the shifted azimuth
  return *args * calcphi2deriv(R,z,
			       phi-*(args+1) * pow(R,*(args+2)-1) * (t-*(args+4)) \
			       - *(args+3),t,
			       potentialArgs->nwrapped,
			       potentialArgs->wrappedPotentialArg);
}
double CorotatingRotationWrapperPotentialRphideriv(double R,double z,double phi,
						   double t,
						   struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  //Calculate Rphideriv = Phi_w,Rphi' - s_R * Phi_w,phi'phi' (chain rule on d/dR)
  double phi_new= phi-*(args+1) * pow(R,*(args+2)-1) * (t-*(args+4))\
    - *(args+3);
  double phiRderiv= -*(args+1) * (*(args+2)-1) * pow(R,*(args+2)-2) \
    * (t-*(args+4));
  return *args * ( calcRphideriv(R,z,phi_new,t,potentialArgs->nwrapped,
				 potentialArgs->wrappedPotentialArg)
		   + phiRderiv * calcphi2deriv(R,z,phi_new,t,
				 potentialArgs->nwrapped,
				 potentialArgs->wrappedPotentialArg));
}
double CorotatingRotationWrapperPotentialzphideriv(double R,double z,double phi,
						   double t,
						   struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  //Calculate zphideriv: s is z-independent and dphi'/dphi = 1, so this is just
  //the wrapped zphideriv at the shifted azimuth
  return *args * calczphideriv(R,z,
			       phi-*(args+1) * pow(R,*(args+2)-1) * (t-*(args+4)) \
			       - *(args+3),t,
			       potentialArgs->nwrapped,
			       potentialArgs->wrappedPotentialArg);
}
