#include <math.h>
#include <stdio.h>
#include <galpy_potentials.h>
// Helper functions
double KuzminLikeWrapperPotential_xi(double R,double z,double a,double b2){
  double asqrtbz= a + sqrt ( z * z + b2 );
  return sqrt ( R * R + asqrtbz * asqrtbz );
}
double KuzminLikeWrapperPotential_dxidR(double R,double z,double a,double b2){
  double asqrtbz= a + sqrt ( z * z + b2 );
  return R / sqrt ( R * R + asqrtbz * asqrtbz );
}
double KuzminLikeWrapperPotential_dxidz(double R,double z,double a,double b2){
  double sqrtbz= sqrt ( z * z + b2 );
  double asqrtbz= a + sqrtbz;
  return asqrtbz * z / sqrt ( R * R + asqrtbz * asqrtbz ) / sqrtbz;
}
double KuzminLikeWrapperPotential_d2xidR2(double R,double z,double a,double b2){
  // = asqrtbz^2 / xi^3 with xi^2 = R^2 + asqrtbz^2 (the exponent is 1.5, NOT
  // 3.0: pow(R^2+asqrtbz^2,3.0) would be xi^6, a long-standing bug in the
  // planar dxdv path fixed together with the 3D Hessian)
  double asqrtbz= a + sqrt ( z * z + b2 );
  return asqrtbz * asqrtbz / pow ( R * R + asqrtbz * asqrtbz , 1.5 );
}
double KuzminLikeWrapperPotential_d2xidz2(double R,double z,double a,double b2){
  // mirrors KuzminLikeWrapperPotential._d2xidz2 in Python
  double sqrtbz= sqrt ( z * z + b2 );
  double xi= KuzminLikeWrapperPotential_xi(R,z,a,b2);
  return ( a * a * a * b2
	   + 3. * a * a * b2 * sqrtbz
	   + a * b2 * ( 3. * b2 + R * R + 3. * z * z )
	   + ( b2 + R * R ) * pow ( sqrtbz , 3.0 ) )
    / pow ( sqrtbz , 3.0 ) / pow ( xi , 3.0 );
}
double KuzminLikeWrapperPotential_d2xidRdz(double R,double z,double a,double b2){
  // mirrors KuzminLikeWrapperPotential._d2xidRdz in Python
  double sqrtbz= sqrt ( z * z + b2 );
  double asqrtbz= a + sqrtbz;
  return - R * z * asqrtbz / sqrtbz
    / pow ( R * R + asqrtbz * asqrtbz , 1.5 );
}

//KuzminLikeWrapperPotential: 3 arguments: amp, a, b**2
double KuzminLikeWrapperPotentialEval(double R,double z,double phi,
					double t,
					struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  double amp= *args;
  double a= *(args+1);
  double b2= *(args+2);
  //Calculate potential, only used in actionAngle, so phi=0, t=0
  return amp * evaluatePotentials(
    KuzminLikeWrapperPotential_xi(R,z,a,b2),
    0.0,
    potentialArgs->nwrapped,
		potentialArgs->wrappedPotentialArg
  );
}
double KuzminLikeWrapperPotentialRforce(double R,double z,double phi,
					  double t,
					  struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  double amp= *args;
  double a= *(args+1);
  double b2= *(args+2);
  //Calculate Rforce
  return amp * calcRforce(
    KuzminLikeWrapperPotential_xi(R,z,a,b2),
    0.0,
    0.0,
    t,
    potentialArgs->nwrapped,
    potentialArgs->wrappedPotentialArg
  ) * KuzminLikeWrapperPotential_dxidR(R,z,a,b2);
}
double KuzminLikeWrapperPotentialzforce(double R,double z,double phi,
					  double t,
					  struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  double amp= *args;
  double a= *(args+1);
  double b2= *(args+2);
  //Calculate zforce
  return amp * calcRforce(
    KuzminLikeWrapperPotential_xi(R,z,a,b2),
    0.0,
    0.0,
    t,
    potentialArgs->nwrapped,
    potentialArgs->wrappedPotentialArg
  ) * KuzminLikeWrapperPotential_dxidz(R,z,a,b2);
}
double KuzminLikeWrapperPotentialPlanarRforce(double R,double phi,double t,
						struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  double amp= *args;
  double a= *(args+1);
  double b2= *(args+2);
  //Calculate planarRforce
  return amp * calcPlanarRforce(
    KuzminLikeWrapperPotential_xi(R,0.0,a,b2),
    0.0,
    t,
    potentialArgs->nwrapped,
    potentialArgs->wrappedPotentialArg
  ) * KuzminLikeWrapperPotential_dxidR(R,0.0,a,b2);
}
double KuzminLikeWrapperPotentialPlanarR2deriv(double R,double phi,double t,
						struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  double amp= *args;
  double a= *(args+1);
  double b2= *(args+2);
  //Calculate planarRforce
  return amp * (
    calcPlanarR2deriv(
      KuzminLikeWrapperPotential_xi(R,0.0,a,b2),
      0.0,
      t,
      potentialArgs->nwrapped,
      potentialArgs->wrappedPotentialArg
    ) * KuzminLikeWrapperPotential_dxidR(R,0.0,a,b2) * KuzminLikeWrapperPotential_dxidR(R,0.0,a,b2)
    - calcPlanarRforce(
      KuzminLikeWrapperPotential_xi(R,0.0,a,b2),
      0.0,
      t,
      potentialArgs->nwrapped,
      potentialArgs->wrappedPotentialArg
    ) * KuzminLikeWrapperPotential_d2xidR2(R,0.0,a,b2)
  );
}
// --- Full 3D Hessian for the variational equations ---
// Phi(R,z) = amp * f(xi(R,z)) with f(xi) = Phi_wrapped(xi, z=0), so by the
// chain rule (calcR2deriv = f'', calcRforce = -f', both evaluated at (xi,0)):
//   Phi_RR = amp * ( f'' xi_R^2 + f' xi_RR )
//   Phi_zz = amp * ( f'' xi_z^2 + f' xi_zz )
//   Phi_Rz = amp * ( f'' xi_R xi_z + f' xi_Rz )
// The wrapper is axisymmetric by construction (the wrapped potential is
// required to be axisymmetric and only enters through xi), so
// phi2deriv/Rphideriv/zphideriv vanish identically -> left NULL in the parser
// (the NULL-safe aggregators return 0 for them), as for MiyamotoNagai.
double KuzminLikeWrapperPotentialR2deriv(double R,double z,double phi,
					 double t,
					 struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  double amp= *args;
  double a= *(args+1);
  double b2= *(args+2);
  double xi= KuzminLikeWrapperPotential_xi(R,z,a,b2);
  double dxidR= KuzminLikeWrapperPotential_dxidR(R,z,a,b2);
  return amp * (
    calcR2deriv(xi,0.0,0.0,t,
		potentialArgs->nwrapped,potentialArgs->wrappedPotentialArg)
      * dxidR * dxidR
    - calcRforce(xi,0.0,0.0,t,
		 potentialArgs->nwrapped,potentialArgs->wrappedPotentialArg)
      * KuzminLikeWrapperPotential_d2xidR2(R,z,a,b2)
  );
}
double KuzminLikeWrapperPotentialz2deriv(double R,double z,double phi,
					 double t,
					 struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  double amp= *args;
  double a= *(args+1);
  double b2= *(args+2);
  double xi= KuzminLikeWrapperPotential_xi(R,z,a,b2);
  double dxidz= KuzminLikeWrapperPotential_dxidz(R,z,a,b2);
  return amp * (
    calcR2deriv(xi,0.0,0.0,t,
		potentialArgs->nwrapped,potentialArgs->wrappedPotentialArg)
      * dxidz * dxidz
    - calcRforce(xi,0.0,0.0,t,
		 potentialArgs->nwrapped,potentialArgs->wrappedPotentialArg)
      * KuzminLikeWrapperPotential_d2xidz2(R,z,a,b2)
  );
}
double KuzminLikeWrapperPotentialRzderiv(double R,double z,double phi,
					 double t,
					 struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  double amp= *args;
  double a= *(args+1);
  double b2= *(args+2);
  double xi= KuzminLikeWrapperPotential_xi(R,z,a,b2);
  return amp * (
    calcR2deriv(xi,0.0,0.0,t,
		potentialArgs->nwrapped,potentialArgs->wrappedPotentialArg)
      * KuzminLikeWrapperPotential_dxidR(R,z,a,b2)
      * KuzminLikeWrapperPotential_dxidz(R,z,a,b2)
    - calcRforce(xi,0.0,0.0,t,
		 potentialArgs->nwrapped,potentialArgs->wrappedPotentialArg)
      * KuzminLikeWrapperPotential_d2xidRdz(R,z,a,b2)
  );
}
