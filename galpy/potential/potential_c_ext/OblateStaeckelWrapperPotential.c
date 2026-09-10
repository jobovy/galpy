#include <math.h>
#include <galpy_potentials.h>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
#ifndef M_PI_2
#define M_PI_2 1.57079632679489661923
#endif
//Cache-safety contract: every C caller must give each OpenMP thread its own
//parsed potentialArg copy (the orbit integrators always did; actionAngle_c
//does since the parse-per-thread change this stacks on), so the exact-mode
//cache below can live in plain per-instance scratch with no thread indexing.
static inline double ostw_sq(double x){return x*x;}
static inline double ostw_cb(double x){return x*x*x;}
void Rz_to_uv(double R,double z,double * u, double * v,double delta){
  double d12, d22, coshu, cosv;
  d12= ostw_sq(z+delta) + ostw_sq(R);
  d22= ostw_sq(z-delta) + ostw_sq(R);
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
  return ostw_sq(sinh(u))+ostw_sq(sin(v));
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
//Tabulated mode (nargs > 5): args= [amp,delta,u0,v0,refpot,ntab,umax,
//  then six (values, natural-cubic 2nd derivs) table pairs of ntab each:
//  U, dU/du, d2U/du2 on uniform u in [0,umax]; V, dV/dv, d2V/dv2 on
//  uniform v in [0,pi/2]]
//parsed as plain type 47 (no wrapped potential in C); exact mode (type -3,
//nargs=5) keeps the wrapped-potential path below.
static inline double ostw_spl(double x,double h,int n,double *y,double *M){
  int i; double a,b;
  if ( x <= 0. ) x= 0.;
  if ( x >= (n-1)*h ) x= (n-1)*h;
  i= (int) (x/h); if (i > n-2) i= n-2;
  b= (x - i*h)/h; a= 1.-b;
  return a*y[i]+b*y[i+1]+((a*a*a-a)*M[i]+(b*b*b-b)*M[i+1])*h*h/6.;
}
//Exact-mode cache (type -3, nargs = 20): args[5] = 0 flag, args[6..19] are
//per-instance scratch (each thread owns its parsed copy, see contract above):
//              [last_u_phi, Phi_u, last_u_F, FR_u, Fz_u,
//               last_v_phi, Phi_v, last_v_F, FR_v, Fz_v,
//               last_R, last_z, FR_out, Fz_out] (the last four: a
//point-level cache computing both forces in one pass -- exact mode only).
//The wrapped Phi and forces along the reference curves depend on u (or v)
//alone, and successive Eval/Rforce/zforce calls hit the same point, so a
//one-slot exact cache removes the dominant redundancy. Lazily filled per
//quantity group: Phi separately from the forces, because planar-parsed
//instances only wire the wrapped potential's planar functions (planardUdu
//stores the planar Rforce in the FR slot; a planar-parsed instance never
//calls the 3D primitives, so the slots never mix semantics).
static inline double ostw_phiu(double u,double v0,double delta,
                               struct potentialArg * potentialArgs){
  double * c= ostw_scratch(potentialArgs);
  double R,z0;
  if ( u != *c ) {
    uv_to_Rz(u,v0,&R,&z0,delta);
    *(c+1)= evaluatePotentials(R,z0,potentialArgs->nwrapped,
                               potentialArgs->wrappedPotentialArg);
    *c= u;
  }
  return *(c+1);
}
static inline void ostw_Fu(double u,double v0,double delta,
                           struct potentialArg * potentialArgs,int planar,
                           double * FR,double * Fz){
  double * c= ostw_scratch(potentialArgs) + 2;
  double R,z0;
  if ( u != *c ) {
    uv_to_Rz(u,v0,&R,&z0,delta);
    if ( planar ) {
      *(c+1)= calcPlanarRforce(R,0.,0.,potentialArgs->nwrapped,
                               potentialArgs->wrappedPotentialArg);
      *(c+2)= 0.;
    }
    else {
      *(c+1)= calcRforce(R,z0,0.,0.,potentialArgs->nwrapped,
                         potentialArgs->wrappedPotentialArg);
      *(c+2)= calczforce(R,z0,0.,0.,potentialArgs->nwrapped,
                         potentialArgs->wrappedPotentialArg);
    }
    *c= u;
  }
  *FR= *(c+1);
  *Fz= *(c+2);
}
static inline double ostw_phiv(double v,double u0,double delta,
                               struct potentialArg * potentialArgs){
  double * c= ostw_scratch(potentialArgs) + 5;
  double R0,z;
  if ( v != *c ) {
    uv_to_Rz(u0,v,&R0,&z,delta);
    *(c+1)= evaluatePotentials(R0,z,potentialArgs->nwrapped,
                               potentialArgs->wrappedPotentialArg);
    *c= v;
  }
  return *(c+1);
}
static inline void ostw_Fv(double v,double u0,double delta,
                           struct potentialArg * potentialArgs,
                           double * FR,double * Fz){
  double * c= ostw_scratch(potentialArgs) + 7;
  double R0,z;
  if ( v != *c ) {
    uv_to_Rz(u0,v,&R0,&z,delta);
    *(c+1)= calcRforce(R0,z,0.,0.,potentialArgs->nwrapped,
                       potentialArgs->wrappedPotentialArg);
    *(c+2)= calczforce(R0,z,0.,0.,potentialArgs->nwrapped,
                       potentialArgs->wrappedPotentialArg);
    *c= v;
  }
  *FR= *(c+1);
  *Fz= *(c+2);
}
static inline double ostw_utab(double u,int k,struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  int n= (int) args[5];
  return ostw_spl(u,args[6]/(n-1),n,args+7+2*k*n,args+7+(2*k+1)*n);
}
static inline double ostw_vtab(double v,int k,struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  int n= (int) args[5];
  double s= 1.;
  if ( v > M_PI_2 ) { v= M_PI - v; if ( k == 1 ) s= -1.; }
  return s * ostw_spl(v,M_PI_2/(n-1),n,args+7+(6+2*k)*n,args+7+(7+2*k)*n);
}
double U(double u,double v0,double delta,struct potentialArg * potentialArgs){
  if ( potentialArgs->nargs > 5 && potentialArgs->args[5] > 0 )
    return ostw_utab(u,0,potentialArgs);
  return ostw_sq(cosh(u)) * ostw_phiu(u,v0,delta,potentialArgs);
}
double dUdu(double u,double v0,double delta,
	    struct potentialArg * potentialArgs){
  if ( potentialArgs->nargs > 5 && potentialArgs->args[5] > 0 )
    return ostw_utab(u,1,potentialArgs);
  double R,z0,FR,Fz;
  uv_to_Rz(u,v0,&R,&z0,delta);
  ostw_Fu(u,v0,delta,potentialArgs,0,&FR,&Fz);
  // 1e-12 bc force should win the 0/0 battle
  return 2 * cosh(u) * sinh(u) * ostw_phiu(u,v0,delta,potentialArgs)	\
    - ostw_sq(cosh(u)) \
    * ( FR * R / ( tanh(u) + 1e-12) + Fz * z0 * tanh(u));
}
double d2Udu2(double u,double v0,double delta,
	      struct potentialArg * potentialArgs){
  if ( potentialArgs->nargs > 5 && potentialArgs->args[5] > 0 )
    return ostw_utab(u,2,potentialArgs);
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
    - ostw_sq(cosh(u))							\
    * ( - calcR2deriv(R,z0,0.,0.,potentialArgs->nwrapped,
		      potentialArgs->wrappedPotentialArg)
	* R * R / ostw_sq(tanh(u) + 1e-12)
	- 2. * calcRzderiv(R,z0,0.,0.,potentialArgs->nwrapped,
			   potentialArgs->wrappedPotentialArg) * R * z0
	+ tRforce * R
	- calcz2deriv(R,z0,0.,0.,potentialArgs->nwrapped,
		      potentialArgs->wrappedPotentialArg)
	* z0 * z0 * ostw_sq(tanh(u))
	+ tzforce * z0 );
}
double planardUdu(double u,double v0,double delta,
		  struct potentialArg * potentialArgs){
  if ( potentialArgs->nargs > 5 && potentialArgs->args[5] > 0 )
    return ostw_utab(u,1,potentialArgs);
  double R,z0,FR,Fz;
  uv_to_Rz(u,v0,&R,&z0,delta);
  ostw_Fu(u,v0,delta,potentialArgs,1,&FR,&Fz);
  // 1e-12 bc force should win the 0/0 battle
  return 2 * cosh(u) * sinh(u) * ostw_phiu(u,v0,delta,potentialArgs)	\
    - ostw_sq(cosh(u)) * FR * R / ( tanh(u) + 1e-12);
}
double planard2Udu2(double u,double v0,double delta,
		    struct potentialArg * potentialArgs){
  if ( potentialArgs->nargs > 5 && potentialArgs->args[5] > 0 )
    return ostw_utab(u,2,potentialArgs);
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
    - ostw_sq(cosh(u))							\
    * ( - calcPlanarR2deriv(R,0.,0.,potentialArgs->nwrapped,
			    potentialArgs->wrappedPotentialArg)
	* R * R / ostw_sq(tanh(u) + 1e-12)
	+ tRforce * R );
}
double V(double v,double u0,double delta,double refpot,
	 struct potentialArg * potentialArgs){
  if ( potentialArgs->nargs > 5 && potentialArgs->args[5] > 0 )
    return ostw_vtab(v,0,potentialArgs);
  return refpot - staeckel_prefactor(u0,v)	\
    * ostw_phiv(v,u0,delta,potentialArgs);
}
double dVdv(double v,double u0,double delta,double refpot,
	    struct potentialArg * potentialArgs){
  if ( potentialArgs->nargs > 5 && potentialArgs->args[5] > 0 )
    return ostw_vtab(v,1,potentialArgs);
  double R0,z,FR,Fz;
  uv_to_Rz(u0,v,&R0,&z,delta);
  ostw_Fv(v,u0,delta,potentialArgs,&FR,&Fz);
  return -2 * sin(v) * cos(v) * ostw_phiv(v,u0,delta,potentialArgs)	\
    + staeckel_prefactor(u0,v)					\
    * ( FR * R0 / tan(v) - Fz * z * tan(v));
}
double d2Vdv2(double v,double u0,double delta,
	      struct potentialArg * potentialArgs){
  if ( potentialArgs->nargs > 5 && potentialArgs->args[5] > 0 )
    return ostw_vtab(v,2,potentialArgs);
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
	* R0 * R0 / ostw_sq(tan(v))
	+ 2. * calcRzderiv(R0,z,0.,0.,potentialArgs->nwrapped,
			   potentialArgs->wrappedPotentialArg) * R0 * z
	- tRforce * R0
	- calcz2deriv(R0,z,0.,0.,potentialArgs->nwrapped,
		      potentialArgs->wrappedPotentialArg)
	* z * z * ostw_sq(tan(v))
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
static void ostw_forces(double R,double z,
                        struct potentialArg * potentialArgs,
                        double * FRout,double * Fzout){
  double * args= potentialArgs->args;
  // the point cache lives in exact-mode scratch only: in tabulated mode
  // (args[5] = ntab > 0) args+16 is table data and must not be written
  int exact= ( (int) *(args+5) ) == 0;
  double * c= ostw_scratch(potentialArgs) + 10;
  if ( exact && R == *c && z == *(c+1) ) { *FRout= *(c+2); *Fzout= *(c+3); return; }
  double amp= *args, delta= *(args+1), u0= *(args+2), v0= *(args+3),
    refpot= *(args+4);
  double u,v;
  Rz_to_uv(R,z,&u,&v,delta);
  double shu= sinh(u), chu= cosh(u), snv= sin(v), csv= cos(v);
  double thu= shu/chu;
  double prefac= shu*shu + snv*snv;
  double dprefacdu= 2.*shu*chu, dprefacdv= 2.*snv*csv;
  double tU= U(u,v0,delta,potentialArgs);
  double tdU= dUdu(u,v0,delta,potentialArgs);
  double tV= V(v,u0,delta,refpot,potentialArgs);
  double tdV= dVdv(v,u0,delta,refpot,potentialArgs);
  double denom= amp / ostw_sq( delta * prefac );
  double umv= (tU - tV) / prefac;
  double dsc= delta * snv * chu;
  *FRout= denom * ( -tdU * dsc + tdV * thu * z
                  + umv * ( dprefacdu * dsc + dprefacdv * thu * z ) );
  *Fzout= denom * ( -tdU * R * csv / snv - tdV * dsc
                  + umv * ( dprefacdu * R * csv / snv - dprefacdv * dsc ) );
  if ( exact ) { *c= R; *(c+1)= z; *(c+2)= *FRout; *(c+3)= *Fzout; }
}
double OblateStaeckelWrapperPotentialRforce(double R,double z,double phi,
					    double t,
					    struct potentialArg * potentialArgs){
  double FR,Fz;
  ostw_forces(R,z,potentialArgs,&FR,&Fz);
  return FR;
}
double OblateStaeckelWrapperPotentialzforce(double R,double z,double phi,
					    double t,
					    struct potentialArg * potentialArgs){
  double FR,Fz;
  ostw_forces(R,z,potentialArgs,&FR,&Fz);
  return Fz;
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
		   / ostw_sq(*(args+1) * prefac));
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
      td2Udu2 * ostw_sq(sin(v)) * ostw_sq(cosh(u))
    + tdUdu * sinh(u) * cosh(u)
    - td2Vdv2 * ostw_sq(sinh(u)) * ostw_sq(cos(v))
    - tdVdv * sin(v) * cos(v)
    + ( ( -tdUdu * cosh(u) * sin(v) + tdVdv * sinh(u) * cos(v) )
	/ delta * umvfac
	+ ( tU - tV )
	* ( -d2prefacdu2 * ostw_sq(cosh(u)) * ostw_sq(sin(v))
	    - dprefacdu * sinh(u) * cosh(u)
	    - d2prefacdv2 * ostw_sq(sinh(u)) * ostw_sq(cos(v))
	    - dprefacdv * sin(v) * cos(v) ) / prefac
	+ ( tU - tV ) * umvfac / prefac / delta
	* ( dprefacdu * cosh(u) * sin(v)
	    + dprefacdv * sinh(u) * cos(v) ) ) )
    / ostw_sq(delta) / ostw_cb(prefac)
    + 2. * OblateStaeckelWrapperPotentialRforce(R,z,phi,t,potentialArgs)
    / ostw_sq(prefac)
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
      td2Udu2 * ostw_sq(sinh(u)) * ostw_sq(cos(v))
    + tdUdu * cosh(u) * sinh(u)
    - td2Vdv2 * ostw_sq(sin(v)) * ostw_sq(cosh(u))
    - tdVdv * cos(v) * sin(v)
    + ( ( -tdUdu * sinh(u) * cos(v) - tdVdv * cosh(u) * sin(v) )
	/ delta * umvfac
	+ ( tU - tV )
	* ( -d2prefacdu2 * ostw_sq(sinh(u)) * ostw_sq(cos(v))
	    - dprefacdu * sinh(u) * cosh(u)
	    - d2prefacdv2 * ostw_sq(sin(v)) * ostw_sq(cosh(u))
	    - dprefacdv * cos(v) * sin(v) ) / prefac
	- ( tU - tV ) * umvfac / prefac / delta
	* ( -dprefacdu * sinh(u) * cos(v)
	    + dprefacdv * cosh(u) * sin(v) ) ) )
    / ostw_sq(delta) / ostw_cb(prefac)
    - 2. * OblateStaeckelWrapperPotentialzforce(R,z,phi,t,potentialArgs)
    / ostw_sq(prefac)
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
    / ostw_sq(delta) / ostw_cb(prefac)
    + 2. * OblateStaeckelWrapperPotentialzforce(R,z,phi,t,potentialArgs)
    / ostw_sq(prefac)
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
      td2Udu2 * ostw_sq(sin(v)) * ostw_sq(cosh(u))
    + tdUdu * sinh(u) * cosh(u)
    + ( -tdUdu * cosh(u) * sin(v) / delta * umvfac
	+ tU * ( -d2prefacdu2 * ostw_sq(cosh(u)) * ostw_sq(sin(v))
		 - dprefacdu * sinh(u) * cosh(u) ) / prefac
	+ tU * umvfac / prefac / delta * dprefacdu * cosh(u) * sin(v) ) )
    / ostw_sq(delta) / ostw_cb(prefac)
    + 2. * OblateStaeckelWrapperPotentialPlanarRforce(R,phi,t,potentialArgs)
    / ostw_sq(prefac) * dprefacdu * cosh(u) * sin(v) / delta;
}
