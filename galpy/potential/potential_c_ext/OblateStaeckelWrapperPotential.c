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
