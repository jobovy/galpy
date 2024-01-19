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
  double asqrtbz= a + sqrt ( z * z + b2 );
  return asqrtbz * asqrtbz / pow ( R * R + asqrtbz * asqrtbz , 3.0 );
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
