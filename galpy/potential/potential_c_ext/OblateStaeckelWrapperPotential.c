#include <math.h>
#include <galpy_potentials.h>
//OblateStaeckelWrapperPotential: amp, delta, u0, v0, refpot
void Rz_to_uv(double R,double z,double * u, double * v,double delta){
  double d12, d22, coshu, cosv;
  d12= pow(z+delta,2) + pow(R,2);
  d22= pow(z-delta,2) + pow(R,2);
  coshu= 0.5 / delta * ( sqrt(d12) + sqrt(d22) );
  cosv=  0.5 / delta * ( sqrt(d12) - sqrt(d22) );
  *u= acosh(coshu);
  *v= acos(cosv);
}
void uv_to_Rz(double u,double v,double * R, double * z,double delta){
  *R= delta * sinh(u) * sin(v);
  *z= delta * cosh(u) * cos(v);
}
double staeckel_prefactor(double u,double v){
  return pow(sinh(u),2)+pow(sin(v),2);
}
void dstaeckel_prefactordudv(double u,double v,
			       double * dprefacdu, double * dprefacdv){
  *dprefacdu= 2 * sinh(u) * cosh(u);
  *dprefacdv= 2 * sin(v) * cos(v);
}
void dstaeckel_prefactord2ud2v(double u,double v,
			       double * d2prefacdu2, double * d2prefacdv2){
  // mirrors _dstaeckel_prefactord2ud2v in Python
  *d2prefacdu2= 2 * cosh(2 * u);
  *d2prefacdv2= 2 * cos(2 * v);
}
double U(double u,double v0,double delta,struct potentialArg * potentialArgs){
  double R,z0;
  uv_to_Rz(u,v0,&R,&z0,delta);
  return pow(cosh(u),2) \
    * evaluatePotentials(R,z0,potentialArgs->nwrapped,
			 potentialArgs->wrappedPotentialArg);
}
double dUdu(double u,double v0,double delta,
	    struct potentialArg * potentialArgs){
  double R,z0;
  uv_to_Rz(u,v0,&R,&z0,delta);
  // 1e-12 bc force should win the 0/0 battle
  return 2 * cosh(u) * sinh(u)				\
    * evaluatePotentials(R,z0,potentialArgs->nwrapped,
			 potentialArgs->wrappedPotentialArg) \
    - pow(cosh(u),2) \
    * ( calcRforce(R,z0,0.,0.,potentialArgs->nwrapped,
		  potentialArgs->wrappedPotentialArg) * R / ( tanh(u) + 1e-12)
	+ calczforce(R,z0,0.,0.,potentialArgs->nwrapped,
		   potentialArgs->wrappedPotentialArg) * z0 * tanh(u));
}
double d2Udu2(double u,double v0,double delta,
	      struct potentialArg * potentialArgs){
  // mirrors OblateStaeckelWrapperPotential._d2Udu2 in Python
  double R,z0;
  double tRforce, tzforce;
  uv_to_Rz(u,v0,&R,&z0,delta);
  tRforce= calcRforce(R,z0,0.,0.,potentialArgs->nwrapped,
		      potentialArgs->wrappedPotentialArg);
  tzforce= calczforce(R,z0,0.,0.,potentialArgs->nwrapped,
		      potentialArgs->wrappedPotentialArg);
  // 1e-12 bc force should win the 0/0 battle (as in dUdu)
  return 2 * cosh(2 * u)						\
    * evaluatePotentials(R,z0,potentialArgs->nwrapped,
			 potentialArgs->wrappedPotentialArg)		\
    - 4 * cosh(u) * sinh(u)						\
    * ( tRforce * R / ( tanh(u) + 1e-12 )
	+ tzforce * z0 * tanh(u) )					\
    - pow(cosh(u),2)							\
    * ( - calcR2deriv(R,z0,0.,0.,potentialArgs->nwrapped,
		      potentialArgs->wrappedPotentialArg)
	* R * R / pow( tanh(u) + 1e-12 , 2 )
	- 2. * calcRzderiv(R,z0,0.,0.,potentialArgs->nwrapped,
			   potentialArgs->wrappedPotentialArg) * R * z0
	+ tRforce * R
	- calcz2deriv(R,z0,0.,0.,potentialArgs->nwrapped,
		      potentialArgs->wrappedPotentialArg)
	* z0 * z0 * pow( tanh(u) , 2 )
	+ tzforce * z0 );
}
double planardUdu(double u,double v0,double delta,
		  struct potentialArg * potentialArgs){
  double R,z0;
  uv_to_Rz(u,v0,&R,&z0,delta);
  // 1e-12 bc force should win the 0/0 battle
  return 2 * cosh(u) * sinh(u)				\
    * evaluatePotentials(R,z0,potentialArgs->nwrapped,
			 potentialArgs->wrappedPotentialArg) \
    - pow(cosh(u),2) \
    *  calcPlanarRforce(R,0.,0.,potentialArgs->nwrapped,
			potentialArgs->wrappedPotentialArg) * R / ( tanh(u) + 1e-12);
}
double planard2Udu2(double u,double v0,double delta,
		    struct potentialArg * potentialArgs){
  // planar counterpart of d2Udu2: at v0 = pi/2 the U reference curve lies in
  // the z=0 plane (z0 = delta cosh u cos(pi/2) = O(1e-16)), so every
  // z0-suppressed term (zforce, Rzderiv, z2deriv) drops and only the wrapped
  // potential's in-plane Rforce/R2deriv survive (as in planardUdu); the
  // planar parser only wires the wrapped potential's planar functions.
  double R,z0;
  double tRforce;
  uv_to_Rz(u,v0,&R,&z0,delta);
  tRforce= calcPlanarRforce(R,0.,0.,potentialArgs->nwrapped,
			    potentialArgs->wrappedPotentialArg);
  // 1e-12 bc force should win the 0/0 battle (as in dUdu)
  return 2 * cosh(2 * u)						\
    * evaluatePotentials(R,z0,potentialArgs->nwrapped,
			 potentialArgs->wrappedPotentialArg)		\
    - 4 * cosh(u) * sinh(u) * tRforce * R / ( tanh(u) + 1e-12 )	\
    - pow(cosh(u),2)							\
    * ( - calcPlanarR2deriv(R,0.,0.,potentialArgs->nwrapped,
			    potentialArgs->wrappedPotentialArg)
	* R * R / pow( tanh(u) + 1e-12 , 2 )
	+ tRforce * R );
}
double V(double v,double u0,double delta,double refpot,
	 struct potentialArg * potentialArgs){
  double R0, z;
  uv_to_Rz(u0,v,&R0,&z,delta);
  return refpot - staeckel_prefactor(u0,v)	\
    * evaluatePotentials(R0,z,potentialArgs->nwrapped,
			 potentialArgs->wrappedPotentialArg);
}
double dVdv(double v,double u0,double delta,double refpot,
	    struct potentialArg * potentialArgs){
  double R0, z;
  uv_to_Rz(u0,v,&R0,&z,delta);
  return -2 * sin(v) * cos(v)				\
    *evaluatePotentials(R0,z,potentialArgs->nwrapped,
			potentialArgs->wrappedPotentialArg) \
    + staeckel_prefactor(u0,v)					\
    * ( calcRforce(R0,z,0.,0.,potentialArgs->nwrapped,
                         potentialArgs->wrappedPotentialArg) * R0 / tan(v)
	- calczforce(R0,z,0.,0.,potentialArgs->nwrapped,
		     potentialArgs->wrappedPotentialArg) * z * tan(v));
}
double d2Vdv2(double v,double u0,double delta,
	      struct potentialArg * potentialArgs){
  // mirrors OblateStaeckelWrapperPotential._d2Vdv2 in Python
  double R0, z;
  double tRforce, tzforce;
  uv_to_Rz(u0,v,&R0,&z,delta);
  tRforce= calcRforce(R0,z,0.,0.,potentialArgs->nwrapped,
		      potentialArgs->wrappedPotentialArg);
  tzforce= calczforce(R0,z,0.,0.,potentialArgs->nwrapped,
		      potentialArgs->wrappedPotentialArg);
  return -2. * cos(2. * v)						\
    * evaluatePotentials(R0,z,potentialArgs->nwrapped,
			 potentialArgs->wrappedPotentialArg)		\
    + 2. * sin(2. * v)							\
    * ( tRforce * R0 / tan(v) - tzforce * z * tan(v) )			\
    + staeckel_prefactor(u0,v)						\
    * ( - calcR2deriv(R0,z,0.,0.,potentialArgs->nwrapped,
		      potentialArgs->wrappedPotentialArg)
	* R0 * R0 / pow( tan(v) , 2 )
	+ 2. * calcRzderiv(R0,z,0.,0.,potentialArgs->nwrapped,
			   potentialArgs->wrappedPotentialArg) * R0 * z
	- tRforce * R0
	- calcz2deriv(R0,z,0.,0.,potentialArgs->nwrapped,
		      potentialArgs->wrappedPotentialArg)
	* z * z * pow( tan(v) , 2 )
	- tzforce * z );
}
double OblateStaeckelWrapperPotentialEval(double R,double z,double phi,
					  double t,
					  struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  //Calculate potential
  double u,v;
  Rz_to_uv(R,z,&u,&v,*(args+1));
  return *args * ( U(u,*(args+3),*(args+1),potentialArgs)
		   - V(v,*(args+2),*(args+1),*(args+4),potentialArgs) ) \
    / staeckel_prefactor(u,v);
}
double OblateStaeckelWrapperPotentialRforce(double R,double z,double phi,
					    double t,
					    struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  //Calculate Rforce
  double u,v;
  double prefac, dprefacdu, dprefacdv;
  Rz_to_uv(R,z,&u,&v,*(args+1));
  prefac= staeckel_prefactor(u,v);
  dstaeckel_prefactordudv(u,v,&dprefacdu,&dprefacdv);
  return *args * ( ( -dUdu(u,*(args+3),*(args+1),potentialArgs)
		     * *(args+1) * sin(v) * cosh(u)
		     + dVdv(v,*(args+2),*(args+1),*(args+4),potentialArgs)
		     * tanh(u) * z
		     + ( U(u,*(args+3),*(args+1),potentialArgs)
			 - V(v,*(args+2),*(args+1),*(args+4),potentialArgs))
		     * ( dprefacdu * *(args+1) * sin(v) * cosh(u)
			 + dprefacdv * tanh(u) * z ) / prefac )
		   / pow(*(args+1) * prefac,2));
}
double OblateStaeckelWrapperPotentialzforce(double R,double z,double phi,
					    double t,
					    struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  //Calculate zforce
  double u,v;
  double prefac, dprefacdu, dprefacdv;
  Rz_to_uv(R,z,&u,&v,*(args+1));
  prefac= staeckel_prefactor(u,v);
  dstaeckel_prefactordudv(u,v,&dprefacdu,&dprefacdv);
  return *args * (( -dUdu(u,*(args+3),*(args+1),potentialArgs) * R / tan(v)
		    - dVdv(v,*(args+2),*(args+1),*(args+4),potentialArgs)
		    * *(args+1) * sin(v) * cosh(u)
		    + ( U(u,*(args+3),*(args+1),potentialArgs)
			- V(v,*(args+2),*(args+1),*(args+4),potentialArgs))
		    * ( dprefacdu / tan(v) * R
			- dprefacdv * *(args+1) * sin(v) * cosh(u))
		    /prefac)
		  / pow(*(args+1) * prefac,2));
}
double OblateStaeckelWrapperPotentialPlanarRforce(double R,double phi,
						  double t,
						  struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  //Calculate Rforce
  double u,v;
  double prefac, dprefacdu, dprefacdv;
  Rz_to_uv(R,0.,&u,&v,*(args+1));
  prefac= staeckel_prefactor(u,v);
  dstaeckel_prefactordudv(u,v,&dprefacdu,&dprefacdv);
  return *args * ( ( -planardUdu(u,*(args+3),*(args+1),potentialArgs)
		     * *(args+1) * sin(v) * cosh(u)
		     + U(u,*(args+3),*(args+1),potentialArgs)
		     * dprefacdu * *(args+1) * sin(v) * cosh(u) / prefac )
		   / pow(*(args+1) * prefac,2));
}
// --- Full 3D Hessian for the variational equations ---
// Direct transcriptions of the Python _R2deriv/_z2deriv/_Rzderiv: chain rule
// of Phi(u,v) = (U(u)-V(v))/(sinh^2 u + sin^2 v) through the prolate
// spheroidal (R,z) -> (u,v) transform with focal length delta, with
// U''(u)/V''(v) built from the wrapped potential's forces and second
// derivatives along the v=pi/2 and u=u0 reference curves (d2Udu2/d2Vdv2
// above). The wrapper output is axisymmetric by construction, so
// phi2deriv/Rphideriv/zphideriv vanish identically -> left NULL in the
// parser (the NULL-safe aggregators return 0 for them). NB: the trailing
// force terms use the wrapper's own C Rforce/zforce, which already include
// amp, so only the leading bracket is multiplied by amp here (in Python the
// caller applies amp to the whole _R2deriv).
double OblateStaeckelWrapperPotentialR2deriv(double R,double z,double phi,
					     double t,
					     struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  double amp= *args;
  double delta= *(args+1);
  double u0= *(args+2);
  double v0= *(args+3);
  double refpot= *(args+4);
  double u,v;
  double prefac, dprefacdu, dprefacdv, d2prefacdu2, d2prefacdv2;
  double umvfac, tU, tdUdu, td2Udu2, tV, tdVdv, td2Vdv2;
  Rz_to_uv(R,z,&u,&v,delta);
  prefac= staeckel_prefactor(u,v);
  dstaeckel_prefactordudv(u,v,&dprefacdu,&dprefacdv);
  dstaeckel_prefactord2ud2v(u,v,&d2prefacdu2,&d2prefacdv2);
  // x (U-V) in Rforce (as in the Python _R2deriv)
  umvfac= ( dprefacdu * delta * sin(v) * cosh(u)
	    + dprefacdv * tanh(u) * z ) / prefac;
  tU= U(u,v0,delta,potentialArgs);
  tdUdu= dUdu(u,v0,delta,potentialArgs);
  td2Udu2= d2Udu2(u,v0,delta,potentialArgs);
  tV= V(v,u0,delta,refpot,potentialArgs);
  tdVdv= dVdv(v,u0,delta,refpot,potentialArgs);
  td2Vdv2= d2Vdv2(v,u0,delta,potentialArgs);
  return amp * (
      td2Udu2 * pow(sin(v),2) * pow(cosh(u),2)
    + tdUdu * sinh(u) * cosh(u)
    - td2Vdv2 * pow(sinh(u),2) * pow(cos(v),2)
    - tdVdv * sin(v) * cos(v)
    + ( ( -tdUdu * cosh(u) * sin(v) + tdVdv * sinh(u) * cos(v) )
	/ delta * umvfac
	+ ( tU - tV )
	* ( -d2prefacdu2 * pow(cosh(u),2) * pow(sin(v),2)
	    - dprefacdu * sinh(u) * cosh(u)
	    - d2prefacdv2 * pow(sinh(u),2) * pow(cos(v),2)
	    - dprefacdv * sin(v) * cos(v) ) / prefac
	+ ( tU - tV ) * umvfac / prefac / delta
	* ( dprefacdu * cosh(u) * sin(v)
	    + dprefacdv * sinh(u) * cos(v) ) ) )
    / pow(delta,2) / pow(prefac,3)
    + 2. * OblateStaeckelWrapperPotentialRforce(R,z,phi,t,potentialArgs)
    / pow(prefac,2)
    * ( dprefacdu * cosh(u) * sin(v) + dprefacdv * sinh(u) * cos(v) )
    / delta;
}
double OblateStaeckelWrapperPotentialz2deriv(double R,double z,double phi,
					     double t,
					     struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  double amp= *args;
  double delta= *(args+1);
  double u0= *(args+2);
  double v0= *(args+3);
  double refpot= *(args+4);
  double u,v;
  double prefac, dprefacdu, dprefacdv, d2prefacdu2, d2prefacdv2;
  double umvfac, tU, tdUdu, td2Udu2, tV, tdVdv, td2Vdv2;
  Rz_to_uv(R,z,&u,&v,delta);
  prefac= staeckel_prefactor(u,v);
  dstaeckel_prefactordudv(u,v,&dprefacdu,&dprefacdv);
  dstaeckel_prefactord2ud2v(u,v,&d2prefacdu2,&d2prefacdv2);
  // x (U-V) in zforce (as in the Python _z2deriv)
  umvfac= ( dprefacdu / tan(v) * R
	    - dprefacdv * delta * sin(v) * cosh(u) ) / prefac;
  tU= U(u,v0,delta,potentialArgs);
  tdUdu= dUdu(u,v0,delta,potentialArgs);
  td2Udu2= d2Udu2(u,v0,delta,potentialArgs);
  tV= V(v,u0,delta,refpot,potentialArgs);
  tdVdv= dVdv(v,u0,delta,refpot,potentialArgs);
  td2Vdv2= d2Vdv2(v,u0,delta,potentialArgs);
  return amp * (
      td2Udu2 * pow(sinh(u),2) * pow(cos(v),2)
    + tdUdu * cosh(u) * sinh(u)
    - td2Vdv2 * pow(sin(v),2) * pow(cosh(u),2)
    - tdVdv * cos(v) * sin(v)
    + ( ( -tdUdu * sinh(u) * cos(v) - tdVdv * cosh(u) * sin(v) )
	/ delta * umvfac
	+ ( tU - tV )
	* ( -d2prefacdu2 * pow(sinh(u),2) * pow(cos(v),2)
	    - dprefacdu * sinh(u) * cosh(u)
	    - d2prefacdv2 * pow(sin(v),2) * pow(cosh(u),2)
	    - dprefacdv * cos(v) * sin(v) ) / prefac
	- ( tU - tV ) * umvfac / prefac / delta
	* ( -dprefacdu * sinh(u) * cos(v)
	    + dprefacdv * cosh(u) * sin(v) ) ) )
    / pow(delta,2) / pow(prefac,3)
    - 2. * OblateStaeckelWrapperPotentialzforce(R,z,phi,t,potentialArgs)
    / pow(prefac,2)
    * ( -dprefacdu * sinh(u) * cos(v) + dprefacdv * cosh(u) * sin(v) )
    / delta;
}
double OblateStaeckelWrapperPotentialRzderiv(double R,double z,double phi,
					     double t,
					     struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  double amp= *args;
  double delta= *(args+1);
  double u0= *(args+2);
  double v0= *(args+3);
  double refpot= *(args+4);
  double u,v;
  double prefac, dprefacdu, dprefacdv, d2prefacdu2, d2prefacdv2;
  double umvfac, tU, tdUdu, td2Udu2, tV, tdVdv, td2Vdv2;
  Rz_to_uv(R,z,&u,&v,delta);
  prefac= staeckel_prefactor(u,v);
  dstaeckel_prefactordudv(u,v,&dprefacdu,&dprefacdv);
  dstaeckel_prefactord2ud2v(u,v,&d2prefacdu2,&d2prefacdv2);
  // x (U-V) in zforce (as in the Python _Rzderiv)
  umvfac= ( dprefacdu / tan(v) * R
	    - dprefacdv * delta * sin(v) * cosh(u) ) / prefac;
  tU= U(u,v0,delta,potentialArgs);
  tdUdu= dUdu(u,v0,delta,potentialArgs);
  td2Udu2= d2Udu2(u,v0,delta,potentialArgs);
  tV= V(v,u0,delta,refpot,potentialArgs);
  tdVdv= dVdv(v,u0,delta,refpot,potentialArgs);
  td2Vdv2= d2Vdv2(v,u0,delta,potentialArgs);
  return amp * (
      ( td2Udu2 + td2Vdv2 ) * cosh(u) * sin(v) * cos(v) * sinh(u)
    + tdUdu * sin(v) * cos(v)
    + tdVdv * sinh(u) * cosh(u)
    + ( ( -tdUdu * cosh(u) * sin(v) + tdVdv * sinh(u) * cos(v) )
	/ delta * umvfac
	+ ( tU - tV )
	* ( ( -d2prefacdu2 + d2prefacdv2 )
	    * sin(v) * cosh(u) * sinh(u) * cos(v)
	    - dprefacdu * sin(v) * cos(v)
	    + dprefacdv * cosh(u) * sinh(u) ) / prefac
	+ ( tU - tV ) * umvfac / prefac / delta
	* ( dprefacdu * cosh(u) * sin(v)
	    + dprefacdv * sinh(u) * cos(v) ) ) )
    / pow(delta,2) / pow(prefac,3)
    + 2. * OblateStaeckelWrapperPotentialzforce(R,z,phi,t,potentialArgs)
    / pow(prefac,2)
    * ( dprefacdu * cosh(u) * sin(v) + dprefacdv * sinh(u) * cos(v) )
    / delta;
}
double OblateStaeckelWrapperPotentialPlanarR2deriv(double R,double phi,
						   double t,
						   struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  double amp= *args;
  double delta= *(args+1);
  double v0= *(args+3);
  double u,v;
  double prefac, dprefacdu, dprefacdv, d2prefacdu2, d2prefacdv2;
  double umvfac, tU, tdUdu, td2Udu2;
  Rz_to_uv(R,0.,&u,&v,delta);
  prefac= staeckel_prefactor(u,v);
  dstaeckel_prefactordudv(u,v,&dprefacdu,&dprefacdv);
  dstaeckel_prefactord2ud2v(u,v,&d2prefacdu2,&d2prefacdv2);
  // In the z=0 plane v = pi/2 exactly: V(pi/2) = 0 (the U/V split is
  // anchored there) and dVdv(pi/2) = 0, so every V term and every
  // cos(v)/dprefacdv-suppressed term of the full R2deriv vanishes
  // identically and umvfac keeps only its dprefacdu piece (z=0 kills the
  // dprefacdv piece) -- the same simplification as in
  // OblateStaeckelWrapperPotentialPlanarRforce. Only the wrapped potential's
  // in-plane Phi/planarRforce/planarR2deriv are needed (the planar parser
  // does not wire the wrapped potential's 3D functions).
  umvfac= dprefacdu * delta * sin(v) * cosh(u) / prefac;
  tU= U(u,v0,delta,potentialArgs);
  tdUdu= planardUdu(u,v0,delta,potentialArgs);
  td2Udu2= planard2Udu2(u,v0,delta,potentialArgs);
  return amp * (
      td2Udu2 * pow(sin(v),2) * pow(cosh(u),2)
    + tdUdu * sinh(u) * cosh(u)
    + ( -tdUdu * cosh(u) * sin(v) / delta * umvfac
	+ tU * ( -d2prefacdu2 * pow(cosh(u),2) * pow(sin(v),2)
		 - dprefacdu * sinh(u) * cosh(u) ) / prefac
	+ tU * umvfac / prefac / delta * dprefacdu * cosh(u) * sin(v) ) )
    / pow(delta,2) / pow(prefac,3)
    + 2. * OblateStaeckelWrapperPotentialPlanarRforce(R,phi,t,potentialArgs)
    / pow(prefac,2) * dprefacdu * cosh(u) * sin(v) / delta;
}
