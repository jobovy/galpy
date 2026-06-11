#include <math.h>
// Constants not defined in MSVC's math.h
#ifndef M_SQRT1_2
#define M_SQRT1_2 0.70710678118654746172
#endif
#ifndef M_2_SQRTPI
#define M_2_SQRTPI 1.12837916709551255856
#endif
#include <gsl/gsl_spline.h>
#include <galpy_potentials.h>
// ChandrasekharDynamicalFrictionForce: 8 arguments: amp,ms,rhm,gamma^2,
// lnLambda, minr^2, ro, rf
double ChandrasekharDynamicalFrictionForceAmplitude(double R,double z,
						    double phi,double t,
						    double r2,
						    struct potentialArg * potentialArgs,
						    double vR,double vT,
						    double vz){
  double sr,X,Xfactor,d_ind,forceAmplitude;
  double * args= potentialArgs->args;
  //Get args
  double amp= *args;
  double ms= *(args+9);
  double rhm= *(args+10);
  double gamma2= *(args+11);
  double lnLambda= *(args+12);
  double ro= *(args+14);
  double rf= *(args+15);
  double GMvs;
  double r= sqrt( r2 );
  double v2=  vR * vR + vT * vT + vz * vz;
  double v= sqrt( v2 );
  // Constant or variable Lambda
  if ( lnLambda < 0 ) {
    GMvs= ms/v/v;
    if ( GMvs < rhm )
      lnLambda= 0.5 * log ( 1. + r2 / gamma2 / rhm / rhm );
    else
      lnLambda= 0.5 * log ( 1. + r2 / gamma2 / GMvs / GMvs );
  }
  d_ind= (r-ro)/(rf-ro);
  d_ind= d_ind <  0 ? 0. : ( d_ind > 1 ? 1. : d_ind);
  sr= gsl_spline_eval(*potentialArgs->spline1d,d_ind,*potentialArgs->acc1d);
  X= M_SQRT1_2 * v / sr;
  Xfactor= erf ( X ) - M_2_SQRTPI * X * exp ( - X * X );
  forceAmplitude= - amp * Xfactor * lnLambda / v2 / v \
    * calcDensity(R,z,phi,t,potentialArgs->nwrapped,
		  potentialArgs->wrappedPotentialArg);
  // Caching
  *(args + 1)= R;
  *(args + 2)= z;
  *(args + 3)= phi;
  *(args + 4)= t;
  *(args + 5)= vR;
  *(args + 6)= vT;
  *(args + 7)= vz;
  *(args + 8)= forceAmplitude;
  return forceAmplitude;
}
double ChandrasekharDynamicalFrictionForceRforce(double R,double z, double phi,
						 double t,
						 struct potentialArg * potentialArgs,
						 double vR,double vT,
						 double vz){
  double forceAmplitude;
  double * args= potentialArgs->args;
  double r2=  R * R + z * z;
  if ( r2 < *(args+13) )  // r < minr, don't bother caching
    return 0.;
  //Get args
  double cached_R= *(args + 1);
  double cached_z= *(args + 2);
  double cached_phi= *(args + 3);
  double cached_t= *(args + 4);
  double cached_vR= *(args + 5);
  double cached_vT= *(args + 6);
  double cached_vz= *(args + 7);
  if ( R != cached_R || phi != cached_phi || z != cached_z || t != cached_t \
       || vR != cached_vR || vT != cached_vT || vz != cached_vz )
    forceAmplitude= ChandrasekharDynamicalFrictionForceAmplitude(R,z,phi,t,r2,
								 potentialArgs,
								 vR,vT,vz);
  else
    forceAmplitude= *(args + 8);
  return forceAmplitude * vR;
}
double ChandrasekharDynamicalFrictionForcezforce(double R,double z, double phi,
						 double t,
						 struct potentialArg * potentialArgs,
						 double vR,double vT,
						 double vz){
  double forceAmplitude;
  double * args= potentialArgs->args;
  double r2=  R * R + z * z;
  if ( r2 < *(args+13) )  // r < minr, don't bother caching
    return 0.;
  //Get args
  double cached_R= *(args + 1);
  double cached_z= *(args + 2);
  double cached_phi= *(args + 3);
  double cached_t= *(args + 4);
  double cached_vR= *(args + 5);
  double cached_vT= *(args + 6);
  double cached_vz= *(args + 7);
  if ( R != cached_R || phi != cached_phi || z != cached_z || t != cached_t \
       || vR != cached_vR || vT != cached_vT || vz != cached_vz )
    // LCOV_EXCL_START
    forceAmplitude= ChandrasekharDynamicalFrictionForceAmplitude(R,z,phi,t,r2,
								 potentialArgs,
								 vR,vT,vz);
    // LCOV_EXCL_STOP
  else
    forceAmplitude= *(args + 8);
  return forceAmplitude * vz;
}
double ChandrasekharDynamicalFrictionForcephitorque(double R,double z,
						   double phi,double t,
						   struct potentialArg * potentialArgs,
						   double vR,double vT,
						   double vz){
  double forceAmplitude;
  double * args= potentialArgs->args;
  double r2=  R * R + z * z;
  if ( r2 < *(args+13) )  // r < minr, don't bother caching
    return 0.;
  //Get args
  double cached_R= *(args + 1);
  double cached_z= *(args + 2);
  double cached_phi= *(args + 3);
  double cached_t= *(args + 4);
  double cached_vR= *(args + 5);
  double cached_vT= *(args + 6);
  double cached_vz= *(args + 7);
  if ( R != cached_R || phi != cached_phi || z != cached_z || t != cached_t \
       || vR != cached_vR || vT != cached_vT || vz != cached_vz )
    // LCOV_EXCL_START
    forceAmplitude= ChandrasekharDynamicalFrictionForceAmplitude(R,z,phi,t,r2,
								 potentialArgs,
								 vR,vT,vz);
    // LCOV_EXCL_STOP
  else
    forceAmplitude= *(args + 8);
  return forceAmplitude * vT * R;
}
/*
  Rectangular force Jacobian (dF/dx and dF/dv) for the 3D variational
  equations (integrate_dxdv with a dissipative force).

  In rectangular coordinates the Chandrasekhar friction force is exactly

    F_i = A(x,|v|) v_i ,   i=x,y,z ,

  with the scalar amplitude (matching ...ForceAmplitude above; G=1)

    A = -amp * lnLambda * Xf(X) * rho(R,z,phi,t) / v^3 ,

  where v=|v|, r^2=x^2+y^2+z^2,
    X  = v / (sqrt(2) sigma_r(r))         [sigma_r: normalized GSL spline,
                                           argument clamped to [ro,rf]]
    Xf = erf(X) - (2/sqrt(pi)) X e^{-X^2}
    lnLambda = constant                       (input lnLambda >= 0), or
             = 0.5 ln(1 + r^2/(gamma^2 D^2)) ,
               D = rhm        if GMvs = ms/v^2 <  rhm    (r-only branch)
               D = GMvs       otherwise                  (r- and v-branch)

  Hence (delta_ij the Kronecker delta)

    dF_i/dx_j = v_i dA/dx_j
    dF_i/dv_j = A delta_ij + v_i (v_j/v) dA/dv

  with, writing C = -amp/v^3:

    dA/dx_j = C [ (dlnL/dx_j) Xf rho + lnL Xf'(X) (dX/dx_j) rho
                  + lnL Xf (drho/dx_j) ]
    dA/dv   = C rho [ (dlnL/dv) Xf + lnL Xf'(X)/(sqrt(2) sigma_r)
                      - 3 lnL Xf / v ]

  and the individual pieces:

    Xf'(X)     = 2 (2/sqrt(pi)) X^2 e^{-X^2}
                 [erf' and the -2X e^{-X^2}/sqrt(pi) term partially cancel]
    dX/dx_j    = -X (sigma_r'(r)/sigma_r) (x_j/r)
                 [sigma_r' = 0 where the spline argument is clamped, exactly
                  matching the clamped force the Jacobian differentiates]
    dlnL/dx_j  = x_j / (gamma^2 D^2 + r^2)   [0 for constant lnLambda;
                  in the v-branch D=GMvs depends on v but not on x]
    dlnL/dv    = 2 r^2 v^3 / (gamma^2 ms^2 + r^2 v^4)
                 [v-branch only: lnL = 0.5 ln(1 + r^2 v^4/(gamma^2 ms^2));
                  0 in the r-only branch and for constant lnLambda. The
                  GMvs<rhm branch switch makes lnLambda only C^0 in v at
                  GMvs=rhm: the Jacobian uses the branch the force is on.]

  drho/dx_j: the C potentialArg interface provides each component's density
  (calcDensity over the wrapped potentialArgs) but NOT its gradient, so this
  single term is computed by CENTRAL FINITE DIFFERENCES of calcDensity with
  step h = 1e-5 sqrt(1+r^2) (relative accuracy ~1e-8 for the density
  gradients of smooth galactic densities; all other terms are analytic).
  This is the only non-analytic ingredient of the Jacobian.

  For r < minr the force is identically zero in a neighborhood -> zero
  Jacobian (same gate as the force; the force is only C^0 across r=minr).
  v=0 is singular for the force itself and is not handled specially.
*/
void ChandrasekharDynamicalFrictionForceRectDissipativeForceJacobian(
    double t, double *q,
    double *jac_x,
    double *jac_v,
    struct potentialArg * potentialArgs){
  int ii,jj;
  double * args= potentialArgs->args;
  //Get args (same layout as ...ForceAmplitude above)
  double amp= *args;
  double ms= *(args+9);
  double rhm= *(args+10);
  double gamma2= *(args+11);
  double lnLambda= *(args+12);
  double minr2= *(args+13);
  double ro= *(args+14);
  double rf= *(args+15);
  double x= *q;
  double y= *(q+1);
  double z= *(q+2);
  double xvec[3]= {x,y,z};
  double vvec[3]= {*(q+3),*(q+4),*(q+5)};
  double r2= x*x + y*y + z*z;
  if ( r2 < minr2 ) { // force identically 0 for r < minr -> zero Jacobian
    for (jj=0; jj < 9; jj++) {
      *(jac_x+jj)= 0.;
      *(jac_v+jj)= 0.;
    }
    return;
  }
  double r= sqrt( r2 );
  double R= sqrt( x*x + y*y );
  double phi= atan2( y , x );
  double v2= vvec[0]*vvec[0] + vvec[1]*vvec[1] + vvec[2]*vvec[2];
  double v= sqrt( v2 );
  // sigma_r and its r-derivative from the same normalized, clamped spline as
  // the force; the derivative of the clamped spline argument is 0 outside
  // [ro,rf], so sigma_r'=0 there (consistent with the constant clamped force)
  double d_ind= (r-ro)/(rf-ro);
  double dsigmadr;
  if ( d_ind < 0. || d_ind > 1. ) {
    d_ind= d_ind < 0. ? 0. : 1.;
    dsigmadr= 0.;
  }
  else
    dsigmadr= gsl_spline_eval_deriv(*potentialArgs->spline1d,d_ind,
				    *potentialArgs->acc1d) / (rf-ro);
  double sr= gsl_spline_eval(*potentialArgs->spline1d,d_ind,
			     *potentialArgs->acc1d);
  double X= M_SQRT1_2 * v / sr;
  double expmX2= exp( -X * X );
  double Xfactor= erf ( X ) - M_2_SQRTPI * X * expmX2;
  double dXfactordX= 2. * M_2_SQRTPI * X * X * expmX2;
  // Coulomb logarithm and its derivatives (see derivation above)
  double dlnLambdadr_overr; // (1/r) dlnL/dr, so dlnL/dx_j = x_j * this
  double dlnLambdadv;
  if ( lnLambda < 0 ) {
    double GMvs= ms/v2;
    if ( GMvs < rhm ) {
      lnLambda= 0.5 * log ( 1. + r2 / gamma2 / rhm / rhm );
      dlnLambdadr_overr= 1. / ( gamma2 * rhm * rhm + r2 );
      dlnLambdadv= 0.;
    }
    else {
      lnLambda= 0.5 * log ( 1. + r2 / gamma2 / GMvs / GMvs );
      dlnLambdadr_overr= 1. / ( gamma2 * GMvs * GMvs + r2 );
      dlnLambdadv= 2. * r2 * v2 * v / ( gamma2 * ms * ms + r2 * v2 * v2 );
    }
  }
  else {
    dlnLambdadr_overr= 0.;
    dlnLambdadv= 0.;
  }
  // Background density and its Cartesian gradient; the gradient is the one
  // genuinely non-analytic term (no density gradients in the C interface):
  // central finite differences of calcDensity, documented above.
  double dens= calcDensity(R,z,phi,t,potentialArgs->nwrapped,
			   potentialArgs->wrappedPotentialArg);
  double ddensdx[3];
  double dh= 1.0e-5 * sqrt( 1. + r2 );
  double qp[3],Rp,phip,densp,densm;
  for (jj=0; jj < 3; jj++) {
    qp[0]= x; qp[1]= y; qp[2]= z;
    qp[jj]+= dh;
    Rp= sqrt( qp[0]*qp[0] + qp[1]*qp[1] );
    phip= atan2( qp[1] , qp[0] );
    densp= calcDensity(Rp,qp[2],phip,t,potentialArgs->nwrapped,
		       potentialArgs->wrappedPotentialArg);
    qp[jj]-= 2. * dh;
    Rp= sqrt( qp[0]*qp[0] + qp[1]*qp[1] );
    phip= atan2( qp[1] , qp[0] );
    densm= calcDensity(Rp,qp[2],phip,t,potentialArgs->nwrapped,
		       potentialArgs->wrappedPotentialArg);
    ddensdx[jj]= 0.5 * ( densp - densm ) / dh;
  }
  // Assemble: A, dA/dv (scalar speed derivative), dA/dx_j
  double Av3= -amp / v2 / v; // C = -amp/v^3
  double A= Av3 * lnLambda * Xfactor * dens;
  double dAdv= Av3 * dens * ( dlnLambdadv * Xfactor
			      + lnLambda * dXfactordX * M_SQRT1_2 / sr
			      - 3. * lnLambda * Xfactor / v );
  double dXdxj,dAdxj;
  for (jj=0; jj < 3; jj++) {
    dXdxj= -X * dsigmadr / sr * xvec[jj] / r;
    dAdxj= Av3 * ( dlnLambdadr_overr * xvec[jj] * Xfactor * dens
		   + lnLambda * dXfactordX * dXdxj * dens
		   + lnLambda * Xfactor * ddensdx[jj] );
    for (ii=0; ii < 3; ii++) {
      *(jac_x+3*ii+jj)= vvec[ii] * dAdxj;
      *(jac_v+3*ii+jj)= vvec[ii] * vvec[jj] / v * dAdv
	+ ( ii == jj ? A : 0. );
    }
  }
}
