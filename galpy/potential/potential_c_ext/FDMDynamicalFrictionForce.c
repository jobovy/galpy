#include <math.h>
// Constants not defined in MSVC's math.h
#ifndef M_SQRT1_2
#define M_SQRT1_2 0.70710678118654746172
#endif
#ifndef M_2_SQRTPI
#define M_2_SQRTPI 1.12837916709551255856
#endif
#include <gsl/gsl_spline.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_sf_expint.h>
#include <gsl/gsl_spline.h>
#include <galpy_potentials.h>
// FDMDynamicalFrictionForceFDMvsCDM: FDM force / CDM force
double FDMDynamicalFrictionForceFDMvsCDM(double R,double z,
						    double phi,double t,
						    double r2,
						    struct potentialArg * potentialArgs,
						    double vR,double vT,
						    double vz){
  double FDMfactor, CDMfactor, C_fdm, C_fdm_disp;
  double kr, sr, d_ind, X, Xfactor, GMvs, M_sigma, mu;
  double * args= potentialArgs->args;
  //Get args
  double amp= *args;
  double ms= *(args+9);
  double rhm= *(args+10);
  double gamma2= *(args+11);
  double lnLambda= *(args+12);
  double ro= *(args+14);
  double rf= *(args+15);
  double mhbar= *(args+16);
  double const_FDMfactor= *(args+17);
  double r= sqrt( r2 );
  double v2=  vR * vR + vT * vT + vz * vz;
  double v= sqrt( v2 );
  // Constant or variable Lambda
  GMvs= ms/v/v;
  if ( lnLambda < 0 ) {
    // lnLambda is constant
    if ( GMvs < rhm )
      lnLambda= 0.5 * log ( 1. + r2 / gamma2 / rhm / rhm );
    else
      lnLambda= 0.5 * log ( 1. + r2 / gamma2 / GMvs / GMvs );
  }
  // CDMfactor
  d_ind= (r-ro)/(rf-ro);
  d_ind= d_ind <  0 ? 0. : ( d_ind > 1 ? 1. : d_ind);
  sr= gsl_spline_eval(*potentialArgs->spline1d,d_ind,*potentialArgs->acc1d);
  X= M_SQRT1_2 * v / sr;
  Xfactor= erf ( X ) - M_2_SQRTPI * X * exp ( - X * X );
  CDMfactor = lnLambda * Xfactor;

  // Constant or variable FDMfactor
  if ( const_FDMfactor < 0 ) {
    kr = 2.0 * mhbar * v * r;
    M_sigma = v/sr; // Mach number

    if(kr<M_sigma)
    { // Zero-dispersion regime
      FDMfactor = M_EULER + log ( kr ) - gsl_sf_Ci(kr) + sin (kr) / (kr) - 1.0;
    }
    else if(kr>4* M_sigma)
    {
      // Dispersion regime
      FDMfactor = log(kr/M_sigma)*Xfactor;
    } else {
      // Intermediate regime between zero-velocity and dispersion regimes
      // Re-evaluating FDM zero-velocity regime at kr = M_sigma/2 and FDM dispersion regime at kr = 2 * M_sigma
      C_fdm = M_EULER + log ( M_sigma ) - gsl_sf_Ci( M_sigma ) + sin ( M_sigma ) / ( M_sigma ) - 1.0;
      C_fdm_disp = log( 4.0 ) * Xfactor;
      // Interpolating between zero-velocity and dispersion regimes
      mu = (2*M_sigma - kr/2)/(2*M_sigma-M_sigma/2);
      FDMfactor = mu * C_fdm + (1-mu) * C_fdm_disp;
    }

  } else {
    FDMfactor = const_FDMfactor;
  }
  return FDMfactor / CDMfactor;
}
double FDMDynamicalFrictionForceAmplitude(double R,double z,
						    double phi,double t,
						    double r2,
						    struct potentialArg * potentialArgs,
						    double vR,double vT,
						    double vz){
  double forceAmplitude, FDMvsCDMamplitude;
  // Compute the force amplitude for Chandrasekhar (CDM) dynamical friction
  double * args= potentialArgs->args;
  forceAmplitude = ChandrasekharDynamicalFrictionForceAmplitude(R,z,phi,t,r2,
                            potentialArgs->wrappedPotentialArg,vR,vT,vz);
  // Now compute the relative CDM vs FDM force amplitude
  double const_FDMfactor= *(args+17);
  FDMvsCDMamplitude = FDMDynamicalFrictionForceFDMvsCDM(R,z,phi,t,r2,
         potentialArgs,vR,vT,vz);
  fflush(stdout);
  if ( FDMvsCDMamplitude < 1. || const_FDMfactor >= 0 ) // Always apply FDM factor if constant
    forceAmplitude *= FDMvsCDMamplitude;
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
double FDMDynamicalFrictionForceRforce(double R,double z, double phi,
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
    forceAmplitude= FDMDynamicalFrictionForceAmplitude(R,z,phi,t,r2,
								 potentialArgs,
								 vR,vT,vz);
  else
    forceAmplitude= *(args + 8);
  return forceAmplitude * vR;
}
double FDMDynamicalFrictionForcezforce(double R,double z, double phi,
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
    forceAmplitude= FDMDynamicalFrictionForceAmplitude(R,z,phi,t,r2,
								 potentialArgs,
								 vR,vT,vz);
    // LCOV_EXCL_STOP
  else
    forceAmplitude= *(args + 8);
  return forceAmplitude * vz;
}
double FDMDynamicalFrictionForcephitorque(double R,double z,
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
    forceAmplitude= FDMDynamicalFrictionForceAmplitude(R,z,phi,t,r2,
								 potentialArgs,
								 vR,vT,vz);
    // LCOV_EXCL_STOP
  else
    forceAmplitude= *(args + 8);
  return forceAmplitude * vT * R;
}
/*
  Rectangular force Jacobian (dF/dx and dF/dv) for the 3D variational
  equations (integrate_dxdv with this dissipative force).

  Like Chandrasekhar friction (see the derivation in
  ChandrasekharDynamicalFrictionForce.c, which this mirrors), the FDM
  friction force is exactly

    F_i = A(x,|v|) v_i ,   i=x,y,z ,

  but with the FDM quantum-pressure-suppressed amplitude (matching
  FDMDynamicalFrictionForceAmplitude/...FDMvsCDM above; G=1)

    A = -amp * Ceff(r,v) * rho(R,z,phi,t) / v^3 ,

  where v=|v|, r^2=x^2+y^2+z^2 and the effective friction coefficient is

    Ceff = const_FDMfactor                  (input const_FDMfactor >= 0), or
         = Cfdm   if Cfdm/Ccdm < 1          (FDM suppression active), or
         = Ccdm   otherwise                 (classical cutoff)

  with Ccdm = lnLambda * Xf(X) the classical (Chandrasekhar) coefficient
  [X = v/(sqrt(2) sigma_r(r)), Xf = erf(X) - (2/sqrt(pi)) X e^{-X^2},
   lnLambda with the same constant / r-only / r-and-v branches as
   Chandrasekhar] and, in terms of kr = 2 mhbar v r and the Mach number
   Msig = v/sigma_r(r),

    Cfdm = gamma_E + ln(kr) - Ci(kr) + sin(kr)/kr - 1
                                          (zero-velocity regime, kr < Msig)
         = ln(kr/Msig) * Xf               (dispersion regime, kr > 4 Msig)
         = mu * Czv + (1-mu) * ln(4) Xf   (intermediate regime, otherwise)
           with Czv = gamma_E + ln(Msig) - Ci(Msig) + sin(Msig)/Msig - 1
                      [the zero-velocity form re-evaluated at kr -> Msig]
           and  mu  = (2 Msig - kr/2) / (2 Msig - Msig/2) .

  Hence (delta_ij the Kronecker delta), exactly as for Chandrasekhar,

    dF_i/dx_j = v_i dA/dx_j ,
    dF_i/dv_j = A delta_ij + v_i (v_j/v) dA/dv ,

  with Cf = -amp/v^3 (Ceff depends on x only through r):

    dA/dx_j = Cf [ (dCeff/dr)(x_j/r) rho + Ceff (drho/dx_j) ]
    dA/dv   = Cf rho [ dCeff/dv - 3 Ceff / v ]

  and the per-branch coefficient derivatives (NOTHING dropped; using
  Ci'(z) = cos(z)/z so that d/dz [gamma_E + ln z - Ci(z) + sin(z)/z - 1]
  = 1/z - cos(z)/z + (z cos(z) - sin(z))/z^2 = (1 - sin(z)/z)/z, and the
  chain-rule pieces dkr/dr = kr/r, dkr/dv = kr/v, dMsig/dr =
  -Msig sigma_r'/sigma_r, dMsig/dv = 1/sigma_r, dX/dr = -X sigma_r'/sigma_r,
  dX/dv = 1/(sqrt(2) sigma_r), Xf'(X) = 2 (2/sqrt(pi)) X^2 e^{-X^2}):

    constant factor:  dCeff/dr = dCeff/dv = 0

    zero-velocity:    dCfdm/dr = [(1 - sin(kr)/kr)/kr] (kr/r)
                      dCfdm/dv = [(1 - sin(kr)/kr)/kr] (kr/v)

    dispersion:       dCfdm/dr = [(1/kr)(dkr/dr) - (1/Msig)(dMsig/dr)] Xf
                                 + ln(kr/Msig) Xf'(X) dX/dr
                      dCfdm/dv = [(1/kr)(dkr/dv) - (1/Msig)(dMsig/dv)] Xf
                                 + ln(kr/Msig) Xf'(X) dX/dv
                      [the bracket in dCfdm/dv vanishes analytically:
                       kr/Msig = 2 mhbar r sigma_r(r) is v-independent;
                       kept in chain-rule form for auditability]

    intermediate:     dCfdm/d* = (dmu/d*) (Czv - ln(4) Xf)
                                 + mu (dCzv/dMsig)(dMsig/d*)
                                 + (1-mu) ln(4) Xf'(X) dX/d* ,  * = r,v
                      with dCzv/dMsig = (1 - sin(Msig)/Msig)/Msig and
                      dmu/d* = [-(dkr/d*)/2 + 2 (dMsig/d*) kr/(4 Msig)]
                               / (1.5 Msig)
                      [from mu = 4/3 - kr/(3 Msig): dmu/dkr = -1/(3 Msig),
                       dmu/dMsig = kr/(3 Msig^2); dmu/dv = 0 analytically
                       since kr/Msig is v-independent]

    classical cutoff: dCcdm/dr = (dlnL/dr) Xf + lnL Xf'(X) dX/dr
                      dCcdm/dv = (dlnL/dv) Xf + lnL Xf'(X) dX/dv
                      with dlnL/dr, dlnL/dv exactly as in the Chandrasekhar
                      Jacobian (constant / r-only / r-and-v branches).

  sigma_r and sigma_r' come from the same normalized, clamped GSL spline as
  the force (sigma_r' = 0 where the spline argument is clamped outside
  [ro,rf], consistently with the constant clamped force).

  The branch selection exactly mirrors the force (same kr vs Msig regime
  tests; the FDM factor is applied when Cfdm/Ccdm < 1, always when
  const_FDMfactor >= 0): at the regime boundaries kr = Msig, kr = 4 Msig and
  at the Cfdm = Ccdm cutoff the force is only C^0 and the Jacobian uses the
  branch the force is on (same convention as the Chandrasekhar GMvs = rhm
  branch switch).

  drho/dx_j: as for Chandrasekhar, the background-density gradient is the
  single non-analytic term (no density gradients in the C potentialArg
  interface) and is computed by CENTRAL FINITE DIFFERENCES of calcDensity
  with step h = 1e-5 sqrt(1+r^2); the density potentialArgs are nested one
  level deeper here (FDM wraps a Chandrasekhar potentialArg, which wraps the
  density: see _parse_pot / FDMDynamicalFrictionForceAmplitude).

  For r < minr the force is identically zero in a neighborhood -> zero
  Jacobian (same gate as the force). v=0 is singular for the force itself
  and is not handled specially.
*/
void FDMDynamicalFrictionForceRectDissipativeForceJacobian(
    double t, double *q,
    double *jac_x,
    double *jac_v,
    struct potentialArg * potentialArgs){
  int ii,jj;
  double * args= potentialArgs->args;
  //Get args (same layout as ...ForceAmplitude/...FDMvsCDM above)
  double amp= *args;
  double ms= *(args+9);
  double rhm= *(args+10);
  double gamma2= *(args+11);
  double lnLambda= *(args+12);
  double minr2= *(args+13);
  double ro= *(args+14);
  double rf= *(args+15);
  double mhbar= *(args+16);
  double const_FDMfactor= *(args+17);
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
  double dXdr= -X * dsigmadr / sr; // dX/dx_j = dXdr * x_j/r
  double dXdv= M_SQRT1_2 / sr;
  // Coulomb logarithm and its derivatives: same branch logic and expressions
  // as the force / the Chandrasekhar Jacobian (see the derivation there)
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
  double CDMfactor= lnLambda * Xfactor;
  // Effective friction coefficient Ceff and its r- and v-derivatives,
  // mirroring the force's regime/branch selection exactly (see above)
  double Ceff,dCeffdr,dCeffdv;
  if ( const_FDMfactor < 0 ) {
    double kr= 2. * mhbar * v * r;
    double M_sigma= v / sr;
    double dkrdr= kr / r;
    double dkrdv= kr / v;
    double dM_sigmadr= -M_sigma * dsigmadr / sr;
    double dM_sigmadv= 1. / sr;
    double FDMfactor,dFDMfactordr,dFDMfactordv;
    if ( kr < M_sigma ) { // zero-velocity regime
      double sinckr= sin ( kr ) / kr;
      FDMfactor= M_EULER + log ( kr ) - gsl_sf_Ci ( kr ) + sinckr - 1.;
      double dFDMfactordkr= ( 1. - sinckr ) / kr;
      dFDMfactordr= dFDMfactordkr * dkrdr;
      dFDMfactordv= dFDMfactordkr * dkrdv;
    }
    else if ( kr > 4. * M_sigma ) { // dispersion regime
      double logkrM= log ( kr / M_sigma );
      FDMfactor= logkrM * Xfactor;
      dFDMfactordr= ( dkrdr / kr - dM_sigmadr / M_sigma ) * Xfactor
	+ logkrM * dXfactordX * dXdr;
      dFDMfactordv= ( dkrdv / kr - dM_sigmadv / M_sigma ) * Xfactor
	+ logkrM * dXfactordX * dXdv; // first term 0 analytically (see above)
    }
    else { // intermediate regime
      double sincM= sin ( M_sigma ) / M_sigma;
      double C_fdm= M_EULER + log ( M_sigma ) - gsl_sf_Ci ( M_sigma )
	+ sincM - 1.;
      double dC_fdmdM= ( 1. - sincM ) / M_sigma;
      double C_fdm_disp= log ( 4.0 ) * Xfactor;
      double mu= ( 2. * M_sigma - 0.5 * kr ) / ( 1.5 * M_sigma );
      double dmudr= ( -dkrdr / ( 3. * M_sigma )
		      + kr * dM_sigmadr / ( 3. * M_sigma * M_sigma ) );
      double dmudv= ( -dkrdv / ( 3. * M_sigma )
		      + kr * dM_sigmadv / ( 3. * M_sigma * M_sigma ) ); // = 0 analytically
      FDMfactor= mu * C_fdm + ( 1. - mu ) * C_fdm_disp;
      dFDMfactordr= dmudr * ( C_fdm - C_fdm_disp )
	+ mu * dC_fdmdM * dM_sigmadr
	+ ( 1. - mu ) * log ( 4.0 ) * dXfactordX * dXdr;
      dFDMfactordv= dmudv * ( C_fdm - C_fdm_disp )
	+ mu * dC_fdmdM * dM_sigmadv
	+ ( 1. - mu ) * log ( 4.0 ) * dXfactordX * dXdv;
    }
    if ( FDMfactor / CDMfactor < 1. ) { // FDM suppression active (same test
					// as the force: FDMvsCDM < 1)
      Ceff= FDMfactor;
      dCeffdr= dFDMfactordr;
      dCeffdv= dFDMfactordv;
    }
    else { // classical cutoff: pure Chandrasekhar coefficient
      Ceff= CDMfactor;
      dCeffdr= dlnLambdadr_overr * r * Xfactor + lnLambda * dXfactordX * dXdr;
      dCeffdv= dlnLambdadv * Xfactor + lnLambda * dXfactordX * dXdv;
    }
  }
  else { // constant FDM factor: always applied by the force, no x,v dependence
    Ceff= const_FDMfactor;
    dCeffdr= 0.;
    dCeffdv= 0.;
  }
  // Background density and its Cartesian gradient; the gradient is the one
  // genuinely non-analytic term (no density gradients in the C interface):
  // central finite differences of calcDensity, documented above. The density
  // potentialArgs are nested one level deeper than for Chandrasekhar: FDM
  // wraps a Chandrasekhar potentialArg, which wraps the density.
  struct potentialArg * densArgsHolder= potentialArgs->wrappedPotentialArg;
  double dens= calcDensity(R,z,phi,t,densArgsHolder->nwrapped,
			   densArgsHolder->wrappedPotentialArg);
  double ddensdx[3];
  double dh= 1.0e-5 * sqrt( 1. + r2 );
  double qp[3],Rp,phip,densp,densm;
  for (jj=0; jj < 3; jj++) {
    qp[0]= x; qp[1]= y; qp[2]= z;
    qp[jj]+= dh;
    Rp= sqrt( qp[0]*qp[0] + qp[1]*qp[1] );
    phip= atan2( qp[1] , qp[0] );
    densp= calcDensity(Rp,qp[2],phip,t,densArgsHolder->nwrapped,
		       densArgsHolder->wrappedPotentialArg);
    qp[jj]-= 2. * dh;
    Rp= sqrt( qp[0]*qp[0] + qp[1]*qp[1] );
    phip= atan2( qp[1] , qp[0] );
    densm= calcDensity(Rp,qp[2],phip,t,densArgsHolder->nwrapped,
		       densArgsHolder->wrappedPotentialArg);
    ddensdx[jj]= 0.5 * ( densp - densm ) / dh;
  }
  // Assemble: A, dA/dv (scalar speed derivative), dA/dx_j
  double Cf= -amp / v2 / v; // Cf = -amp/v^3
  double A= Cf * Ceff * dens;
  double dAdv= Cf * dens * ( dCeffdv - 3. * Ceff / v );
  double dAdxj;
  for (jj=0; jj < 3; jj++) {
    dAdxj= Cf * ( dCeffdr * xvec[jj] / r * dens + Ceff * ddensdx[jj] );
    for (ii=0; ii < 3; ii++) {
      *(jac_x+3*ii+jj)= vvec[ii] * dAdxj;
      *(jac_v+3*ii+jj)= vvec[ii] * vvec[jj] / v * dAdv
	+ ( ii == jj ? A : 0. );
    }
  }
}
