#include <galpy_potentials.h>
#include <bovy_coords.h>
//NonInertialFrameForce
// The time-dependent inputs are evaluated either by calling the supplied
// Python/numba functions (tfuncs; pot_type 39) or, when cinterp=True
// (pot_type 45), from GSL splines precomputed over the integration time range
// by initNonInertialFrameForceSplines. For the primitives a0/x0/v0/Omega the
// spline index equals the tfunc index; Omegadot is the spline derivative of
// Omega. These helpers hide that branch (spline1d != NULL => spline mode); tq
// is the time clamped to the spline's [tmin,tmax] (unused in tfunc mode).
static inline double NonInertialFrameForce_tdep(struct potentialArg * potentialArgs,
						int idx,double t,double tq){
  return potentialArgs->spline1d					\
    ? gsl_spline_eval(*(potentialArgs->spline1d + idx),tq,		\
		      *(potentialArgs->acc1d + idx))			\
    : (*(*(potentialArgs->tfuncs + idx)))(t);
}
// Omegadot: spline mode -> d/dt of the Omega spline at sidx; tfunc mode -> tfuncs[tidx].
static inline double NonInertialFrameForce_tdep_deriv(struct potentialArg * potentialArgs,
						      int sidx,int tidx,double t,double tq){
  return potentialArgs->spline1d					\
    ? gsl_spline_eval_deriv(*(potentialArgs->spline1d + sidx),tq,	\
			    *(potentialArgs->acc1d + sidx))		\
    : (*(*(potentialArgs->tfuncs + tidx)))(t);
}
//arguments: amp
void NonInertialFrameForcexyzforces_xyz(double R,double z,double phi,double t,
                                        double vR,double vT,double vz,
				                                double * Fx, double * Fy, double * Fz,
                                        struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  double x, y, vx, vy, tq;
  bool rot_acc, lin_acc, omegaz_only, const_freq, Omega_as_func;
  bool interp;
  double Omegax, Omegay, Omegaz;
  double Omega2, Omegatimesvecx;
  double Omegadotx, Omegadoty, Omegadotz;
  double x0x, x0y, x0z, v0x, v0y, v0z;
  // cinterp mode (pot_type 45) iff the splines were built; then clamp time
  // queries to [tmin,tmax]=args[23,24], as GSL splines error outside their range.
  interp= potentialArgs->spline1d != NULL;
  tq= interp \
    ? ( t < *(args + 23) ? *(args + 23) : ( t > *(args + 24) ? *(args + 24) : t ) ) \
    : t;
  cyl_to_rect(R,phi,&x,&y);
  cyl_to_rect_vec(vR,vT,phi,&vx,&vy);
  //Setup caching
  *(args + 1)= R;
  *(args + 2)= z;
  *(args + 3)= phi;
  *(args + 4)= t;
  *(args + 5)= vR;
  *(args + 6)= vT;
  *(args + 7)= vz;
  // Evaluate force
  *Fx= 0.;
  *Fy= 0.;
  *Fz= 0.;
  // Rotational acceleration part
  rot_acc= (bool) *(args + 11);
  lin_acc= (bool) *(args + 12);
  Omega_as_func= (bool) *(args + 15);
  if ( rot_acc ) {
    omegaz_only= (bool) *(args + 13);
    const_freq= (bool) *(args + 14);
    if ( omegaz_only ) {
      if ( Omega_as_func ) {
        Omegaz= NonInertialFrameForce_tdep(potentialArgs,9*lin_acc,t,tq);
        Omega2= Omegaz * Omegaz;
      } else {
        Omegaz= *(args + 18);
        if ( !const_freq ) {
          Omegaz+= *(args + 22) * t;
          Omega2= Omegaz * Omegaz;
        } else {
          Omega2= *(args + 19);
        }
      }
      *Fx+=  2. * Omegaz * vy + Omega2 * x;
      *Fy+= -2. * Omegaz * vx + Omega2 * y;
      if ( !const_freq ) {
        if ( Omega_as_func ) {
          Omegadotz= NonInertialFrameForce_tdep_deriv(potentialArgs,9*lin_acc,9*lin_acc+1,t,tq);
        } else {
          Omegadotz= *(args + 22);
        }
        *Fx+= Omegadotz * y;
        *Fy-= Omegadotz * x;
      }
      if ( lin_acc ) {
        x0x= NonInertialFrameForce_tdep(potentialArgs,3,t,tq);
        x0y= NonInertialFrameForce_tdep(potentialArgs,4,t,tq);
        v0x= NonInertialFrameForce_tdep(potentialArgs,6,t,tq);
        v0y= NonInertialFrameForce_tdep(potentialArgs,7,t,tq);
        *Fx+=  2. * Omegaz * v0y + Omega2 * x0x;
        *Fy+= -2. * Omegaz * v0x + Omega2 * x0y;
        if ( !const_freq ) {
          *Fx+= Omegadotz * x0y;
          *Fy-= Omegadotz * x0x;
        }
      }
    } else {
      if ( Omega_as_func ) {
        Omegax= NonInertialFrameForce_tdep(potentialArgs,9*lin_acc  ,t,tq);
        Omegay= NonInertialFrameForce_tdep(potentialArgs,9*lin_acc+1,t,tq);
        Omegaz= NonInertialFrameForce_tdep(potentialArgs,9*lin_acc+2,t,tq);
        Omega2= Omegax * Omegax + Omegay * Omegay + Omegaz * Omegaz;
      } else {
        Omegax= *(args + 16);
        Omegay= *(args + 17);
        Omegaz= *(args + 18);
        if ( !const_freq ) {
          Omegax+= *(args + 20) * t;
          Omegay+= *(args + 21) * t;
          Omegaz+= *(args + 22) * t;
          Omega2= Omegax * Omegax + Omegay * Omegay + Omegaz * Omegaz;
        } else {
          Omega2= *(args + 19);
        }
      }
      Omegatimesvecx= Omegax * x + Omegay * y + Omegaz * z;
      *Fx+=  2. * ( Omegaz * vy - Omegay * vz ) + Omega2 * x - Omegax * Omegatimesvecx;
      *Fy+= -2. * ( Omegaz * vx - Omegax * vz ) + Omega2 * y - Omegay * Omegatimesvecx;
      *Fz+=  2. * ( Omegay * vx - Omegax * vy ) + Omega2 * z - Omegaz * Omegatimesvecx;
      if ( !const_freq ) {
        if ( Omega_as_func ) {
          Omegadotx= NonInertialFrameForce_tdep_deriv(potentialArgs,9*lin_acc  ,9*lin_acc+3,t,tq);
          Omegadoty= NonInertialFrameForce_tdep_deriv(potentialArgs,9*lin_acc+1,9*lin_acc+4,t,tq);
          Omegadotz= NonInertialFrameForce_tdep_deriv(potentialArgs,9*lin_acc+2,9*lin_acc+5,t,tq);
        } else {
          Omegadotx= *(args + 20);
          Omegadoty= *(args + 21);
          Omegadotz= *(args + 22);
        }
        *Fx-= -Omegadotz * y + Omegadoty * z;
        *Fy-=  Omegadotz * x - Omegadotx * z;
        *Fz-= -Omegadoty * x + Omegadotx * y;
      }
      if ( lin_acc ) {
        x0x= NonInertialFrameForce_tdep(potentialArgs,3,t,tq);
        x0y= NonInertialFrameForce_tdep(potentialArgs,4,t,tq);
        x0z= NonInertialFrameForce_tdep(potentialArgs,5,t,tq);
        v0x= NonInertialFrameForce_tdep(potentialArgs,6,t,tq);
        v0y= NonInertialFrameForce_tdep(potentialArgs,7,t,tq);
        v0z= NonInertialFrameForce_tdep(potentialArgs,8,t,tq);
        // Reuse variable
        Omegatimesvecx= Omegax * x0x + Omegay * x0y + Omegaz * x0z;
        *Fx+=  2. * ( Omegaz * v0y - Omegay * v0z ) + Omega2 * x0x - Omegax * Omegatimesvecx;
        *Fy+= -2. * ( Omegaz * v0x - Omegax * v0z ) + Omega2 * x0y - Omegay * Omegatimesvecx;
        *Fz+=  2. * ( Omegay * v0x - Omegax * v0y ) + Omega2 * x0z - Omegaz * Omegatimesvecx;
        if ( !const_freq ) {
          *Fx-= -Omegadotz * x0y + Omegadoty * x0z;
          *Fy-=  Omegadotz * x0x - Omegadotx * x0z;
          *Fz-= -Omegadoty * x0x + Omegadotx * x0y;
        }
      }
    }
  }
  // Linear acceleration part
  if ( lin_acc ) {
    *Fx-= NonInertialFrameForce_tdep(potentialArgs,0,t,tq);
    *Fy-= NonInertialFrameForce_tdep(potentialArgs,1,t,tq);
    *Fz-= NonInertialFrameForce_tdep(potentialArgs,2,t,tq);
  }
  // Caching
  *(args +  8)= *Fx;
  *(args +  9)= *Fy;
  *(args + 10)= *Fz;
}
double NonInertialFrameForceRforce(double R,double z,double phi,double t,
				                           struct potentialArg * potentialArgs,
                                   double vR,double vT,double vz){
  double * args= potentialArgs->args;
  double amp= *args;
  // Get caching args: amp = 0, R,z,phi,t,vR,vT,vz,Fx,Fy,Fz
  double cached_R= *(args + 1);
  double cached_z= *(args + 2);
  double cached_phi= *(args + 3);
  double cached_t= *(args + 4);
  double cached_vR= *(args + 5);
  double cached_vT= *(args + 6);
  double cached_vz= *(args + 7);
  //Calculate potential
  double Fx, Fy, Fz;
  if ( R != cached_R || phi != cached_phi || z != cached_z || t != cached_t \
       || vR != cached_vR || vT != cached_vT || vz != cached_vz ) // LCOV_EXCL_LINE
    NonInertialFrameForcexyzforces_xyz(R,z,phi,t,vR,vT,vz,
                                       &Fx,&Fy,&Fz,potentialArgs);
  else {
    // LCOV_EXCL_START
    Fx= *(args +  8);
    Fy= *(args +  9);
    Fz= *(args + 10);
    // LCOV_EXCL_STOP
  }
  return amp * ( cos ( phi ) * Fx + sin( phi ) * Fy );
}
double NonInertialFrameForcePlanarRforce(double R,double phi,double t,
				                         struct potentialArg * potentialArgs,
                                         double vR,double vT){
  return NonInertialFrameForceRforce(R,0.,phi,t,potentialArgs,vR,vT,0.);
}
double NonInertialFrameForcephitorque(double R,double z,double phi,double t,
				                             struct potentialArg * potentialArgs,
                                     double vR,double vT,double vz){
  double * args= potentialArgs->args;
  double amp= *args;
  // Get caching args: amp = 0, R,z,phi,t,vR,vT,vz,Fx,Fy,Fz
  double cached_R= *(args + 1);
  double cached_z= *(args + 2);
  double cached_phi= *(args + 3);
  double cached_t= *(args + 4);
  double cached_vR= *(args + 5);
  double cached_vT= *(args + 6);
  double cached_vz= *(args + 7);
  //Calculate potential
  double Fx, Fy, Fz;
  if ( R != cached_R || phi != cached_phi || z != cached_z || t != cached_t \
       || vR != cached_vR || vT != cached_vT || vz != cached_vz )
    // LCOV_EXCL_START
    NonInertialFrameForcexyzforces_xyz(R,z,phi,t,vR,vT,vz,
                                       &Fx,&Fy,&Fz,potentialArgs);
    // LCOV_EXCL_STOP
  else {
    Fx= *(args +  8);
    Fy= *(args +  9);
    Fz= *(args + 10);
  }
  return amp * R * ( -sin ( phi ) * Fx + cos( phi ) * Fy );
}
double NonInertialFrameForcePlanarphitorque(double R,double phi,double t,
				                            struct potentialArg * potentialArgs,
                                            double vR,double vT){
  return NonInertialFrameForcephitorque(R,0.,phi,t,potentialArgs,vR,vT,0.);
}
double NonInertialFrameForcezforce(double R,double z,double phi,double t,
				                           struct potentialArg * potentialArgs,
                                   double vR,double vT,double vz){
  double * args= potentialArgs->args;
  double amp= *args;
  // Get caching args: amp = 0, R,z,phi,t,vR,vT,vz,Fx,Fy,Fz
  double cached_R= *(args + 1);
  double cached_z= *(args + 2);
  double cached_phi= *(args + 3);
  double cached_t= *(args + 4);
  double cached_vR= *(args + 5);
  double cached_vT= *(args + 6);
  double cached_vz= *(args + 7);
  //Calculate potential
  double Fx, Fy, Fz;
  if ( R != cached_R || phi != cached_phi || z != cached_z || t != cached_t \
       || vR != cached_vR || vT != cached_vT || vz != cached_vz )
    // LCOV_EXCL_START
    NonInertialFrameForcexyzforces_xyz(R,z,phi,t,vR,vT,vz,
                                       &Fx,&Fy,&Fz,potentialArgs);
    // LCOV_EXCL_STOP
  else {
    Fx= *(args +  8);
    Fy= *(args +  9);
    Fz= *(args + 10);
  }
  return amp * Fz;
}
/* Rectangular force Jacobian (dF/dx, dF/dv) for the 3D variational equations
   (Orbit.integrate_dxdv, phasedim 6). The fictitious force

     F = -2 Omega x (v + v0) - Omega x (Omega x [r + x0])
         - Omegadot x [r + x0] - a0(t)

   is LINEAR in the position r and the velocity v with purely time-dependent
   coefficients, so the Jacobian blocks are EXACT closed forms:

     dF/dv = -2 [Omega(t)]_x
     dF/dx = |Omega(t)|^2 I - Omega(t) Omega(t)^T - [Omegadot(t)]_x

   with [w]_x the cross-product (antisymmetric) matrix; the translation terms
   (x0, v0, a0) depend only on t and contribute nothing. Because dF/dv is
   antisymmetric, tr(dF/dv) = 0: a (even arbitrarily time-dependent) frame
   force preserves phase-space volume, det M(t) = 1 -- unlike friction.

   Omega(t) and Omegadot(t) are evaluated exactly as in the force above for
   every configuration the C force supports: constant scalar Omega about z or
   constant vector Omega (omegaz_only/const_freq, args 16-19), constant
   Omegadot (Omega(t) = Omega + Omegadot t, args 20-22), and Omega as
   function(s) of time via the tfunc callbacks (pot_type 39) or the cinterp
   GSL splines (pot_type 45; Omegadot is then the spline derivative of Omega
   and time queries are clamped to [tmin,tmax] = args[23,24], as in the
   force). For omegaz_only, Omega = (0,0,Omega_z) in the formulas above. */
void NonInertialFrameForceRectDissipativeForceJacobian(
    double t,double *q,
    double * jac_x,double * jac_v,
    struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  double amp= *args;
  bool rot_acc, lin_acc, omegaz_only, const_freq, Omega_as_func, interp;
  double tq;
  double Omegax= 0., Omegay= 0., Omegaz= 0., Omega2;
  double Omegadotx= 0., Omegadoty= 0., Omegadotz= 0.;
  int jj;
  for (jj=0; jj < 9; jj++) {
    *(jac_x+jj)= 0.;
    *(jac_v+jj)= 0.;
  }
  rot_acc= (bool) *(args + 11);
  if ( !rot_acc ) // translation-only frame (a0(t) only): F=F(t) -> Jacobian 0
    return;
  lin_acc= (bool) *(args + 12);
  omegaz_only= (bool) *(args + 13);
  const_freq= (bool) *(args + 14);
  Omega_as_func= (bool) *(args + 15);
  // cinterp mode (pot_type 45) iff the splines were built; then clamp time
  // queries to [tmin,tmax]=args[23,24] as the force does (GSL splines error
  // outside their range)
  interp= potentialArgs->spline1d != NULL;
  tq= interp \
    ? ( t < *(args + 23) ? *(args + 23) : ( t > *(args + 24) ? *(args + 24) : t ) ) \
    : t;
  // Omega(t) and Omegadot(t), exactly as in the force evaluation above
  if ( omegaz_only ) {
    if ( Omega_as_func )
      Omegaz= NonInertialFrameForce_tdep(potentialArgs,9*lin_acc,t,tq);
    else {
      Omegaz= *(args + 18);
      if ( !const_freq )
        Omegaz+= *(args + 22) * t;
    }
    if ( !const_freq )
      Omegadotz= Omega_as_func					\
        ? NonInertialFrameForce_tdep_deriv(potentialArgs,9*lin_acc,
                                           9*lin_acc+1,t,tq)	\
        : *(args + 22);
  } else {
    if ( Omega_as_func ) {
      Omegax= NonInertialFrameForce_tdep(potentialArgs,9*lin_acc  ,t,tq);
      Omegay= NonInertialFrameForce_tdep(potentialArgs,9*lin_acc+1,t,tq);
      Omegaz= NonInertialFrameForce_tdep(potentialArgs,9*lin_acc+2,t,tq);
    } else {
      Omegax= *(args + 16);
      Omegay= *(args + 17);
      Omegaz= *(args + 18);
      if ( !const_freq ) {
        Omegax+= *(args + 20) * t;
        Omegay+= *(args + 21) * t;
        Omegaz+= *(args + 22) * t;
      }
    }
    if ( !const_freq ) {
      if ( Omega_as_func ) {
        Omegadotx= NonInertialFrameForce_tdep_deriv(potentialArgs,9*lin_acc  ,
                                                    9*lin_acc+3,t,tq);
        Omegadoty= NonInertialFrameForce_tdep_deriv(potentialArgs,9*lin_acc+1,
                                                    9*lin_acc+4,t,tq);
        Omegadotz= NonInertialFrameForce_tdep_deriv(potentialArgs,9*lin_acc+2,
                                                    9*lin_acc+5,t,tq);
      } else {
        Omegadotx= *(args + 20);
        Omegadoty= *(args + 21);
        Omegadotz= *(args + 22);
      }
    }
  }
  // |Omega|^2: use the precomputed args[19] in the constant-frequency
  // non-function case, exactly as the force does
  Omega2= ( const_freq && !Omega_as_func )	\
    ? *(args + 19)				\
    : Omegax * Omegax + Omegay * Omegay + Omegaz * Omegaz;
  // dF/dx = |Omega|^2 I - Omega Omega^T - [Omegadot]_x (row-major dF_i/dx_j)
  *(jac_x  )= amp * ( Omega2 - Omegax * Omegax );
  *(jac_x+1)= amp * ( -Omegax * Omegay + Omegadotz );
  *(jac_x+2)= amp * ( -Omegax * Omegaz - Omegadoty );
  *(jac_x+3)= amp * ( -Omegay * Omegax - Omegadotz );
  *(jac_x+4)= amp * ( Omega2 - Omegay * Omegay );
  *(jac_x+5)= amp * ( -Omegay * Omegaz + Omegadotx );
  *(jac_x+6)= amp * ( -Omegaz * Omegax + Omegadoty );
  *(jac_x+7)= amp * ( -Omegaz * Omegay - Omegadotx );
  *(jac_x+8)= amp * ( Omega2 - Omegaz * Omegaz );
  // dF/dv = -2 [Omega]_x (antisymmetric -> traceless; diagonal stays 0)
  *(jac_v+1)= amp *  2. * Omegaz;
  *(jac_v+2)= amp * -2. * Omegay;
  *(jac_v+3)= amp * -2. * Omegaz;
  *(jac_v+5)= amp *  2. * Omegax;
  *(jac_v+6)= amp *  2. * Omegay;
  *(jac_v+7)= amp * -2. * Omegax;
}
