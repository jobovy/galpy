#include <stdbool.h>
#include <math.h>
#include <gsl/gsl_spline.h>
#include <bovy_coords.h>
#include <galpy_potentials.h>

//RotateAndTiltWrapperPotential
//
//Phi(x,t) = pot( R(t) . (x - c(t)) + offset, t )
//
//arguments layout (args):
//  0      : amp
//  1- 4   : cached (x,y,z,t) force-cache key (t included because the wrapped
//           potential may be explicitly time-dependent)
//  5- 7   : cached rotated-back Cartesian forces (Fx,Fy,Fz), w/o amp
//  8-16   : constant rotation matrix rot (row-major); x_aligned = rot . x
//  17     : rotSet (constant rotation)
//  18     : offsetSet
//  19-21  : offset (applied in the aligned frame, after the rotation)
//  22-25  : cached (x,y,z,t) Hessian-cache key (NaN when empty)
//  26-31  : cached rotated-back Cartesian Hessian (Hxx,Hxy,Hxz,Hyy,Hyz,Hzz),
//           w/o amp
//  32     : centerMode: 0 none, 1 constant (33-35), 2 splines of (x,y,z)(t)
//  33-35  : constant center
//  36     : orientSet: time-dependent rotation from splines of the unit
//           quaternion (w,x,y,z)(t), normalized after interpolation
//  37-38  : time range of the center splines (the time is clamped to it)
//  39-40  : time range of the orientation splines (the time is clamped to it)
//spline1d: the center's x,y,z splines (when centerMode == 2), followed by the
//orientation's w,x,y,z splines (when orientSet), set up by
//initRotateAndTiltWrapperPotentialSplines
static double RotateAndTiltWrapperPotentialclamp(double t, double t0,
                                                 double t1){
  return t < t0 ? t0 : ( t > t1 ? t1 : t );
}
//The rotation matrix of the unit quaternion q = (w,x,y,z) (the same formula
//as _quaternion_to_matrix in RotateAndTiltWrapperPotential.py)
static void RotateAndTiltWrapperPotentialquat2rot(double * q, double * rot){
  double w= *q, x= *(q+1), y= *(q+2), z= *(q+3);
  *(rot  )= 1. - 2. * ( y * y + z * z );
  *(rot+1)= 2. * ( x * y - w * z );
  *(rot+2)= 2. * ( x * z + w * y );
  *(rot+3)= 2. * ( x * y + w * z );
  *(rot+4)= 1. - 2. * ( x * x + z * z );
  *(rot+5)= 2. * ( y * z - w * x );
  *(rot+6)= 2. * ( x * z - w * y );
  *(rot+7)= 2. * ( y * z + w * x );
  *(rot+8)= 1. - 2. * ( x * x + y * y );
}
//Transform the point (x,y,z) at time t to the aligned frame of the wrapped
//potential: x_aligned = rot . (x - center) + offset; returns the rotation
//matrix used in rot (when *rotSet)
static void RotateAndTiltWrapperPotentialtobody(double * x, double * y,
                double * z, double t, double * rot, bool * rotSet,
                struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  int centerMode= (int) *(args+32);
  bool orientSet= (bool) *(args+36);
  bool offsetSet= (bool) *(args+18);
  int ii, ispline= 0;
  double tt, q[4], qnorm;
  if ( centerMode == 1 ) {
    *x -= *(args+33);
    *y -= *(args+34);
    *z -= *(args+35);
  }
  else if ( centerMode == 2 ) {
    tt= RotateAndTiltWrapperPotentialclamp(t, *(args+37), *(args+38));
    *x -= gsl_spline_eval(*potentialArgs->spline1d, tt, *potentialArgs->acc1d);
    *y -= gsl_spline_eval(*(potentialArgs->spline1d+1), tt,
                          *(potentialArgs->acc1d+1));
    *z -= gsl_spline_eval(*(potentialArgs->spline1d+2), tt,
                          *(potentialArgs->acc1d+2));
    ispline= 3;
  }
  if ( orientSet ) {
    tt= RotateAndTiltWrapperPotentialclamp(t, *(args+39), *(args+40));
    qnorm= 0.;
    for (ii=0; ii < 4; ii++) {
      q[ii]= gsl_spline_eval(*(potentialArgs->spline1d+ispline+ii), tt,
                             *(potentialArgs->acc1d+ispline+ii));
      qnorm+= q[ii] * q[ii];
    }
    qnorm= sqrt(qnorm);
    for (ii=0; ii < 4; ii++)
      q[ii]/= qnorm;
    RotateAndTiltWrapperPotentialquat2rot(q, rot);
    *rotSet= true;
  }
  else {
    for (ii=0; ii < 9; ii++)
      *(rot+ii)= *(args+8+ii);
    *rotSet= (bool) *(args+17);
  }
  if ( *rotSet )
    rotate(x, y, z, rot);
  if ( offsetSet ) {
    *x += *(args+19);
    *y += *(args+20);
    *z += *(args+21);
  }
}
void RotateAndTiltWrapperPotentialxyzforces(double R, double z, double phi,
                 double t, double * Fx, double * Fy, double * Fz,
                 struct potentialArg * potentialArgs){
    double * args= potentialArgs->args;
    double rot[9];
    bool rotSet;
    double x, y;
    double Rforce, phitorque;
    cyl_to_rect(R, phi, &x, &y);
    //caching
    *(args + 1)= x;
    *(args + 2)= y;
    *(args + 3)= z;
    *(args + 4)= t;
    //now get the forces in R, phi, z in the aligned frame
    RotateAndTiltWrapperPotentialtobody(&x, &y, &z, t, rot, &rotSet,
                                        potentialArgs);
    rect_to_cyl(x,y,&R,&phi);
    Rforce= calcRforce(R, z, phi, t, potentialArgs->nwrapped,
		       potentialArgs->wrappedPotentialArg);
    phitorque= calcphitorque(R, z, phi, t, potentialArgs->nwrapped,
			   potentialArgs->wrappedPotentialArg);
    *Fz= calczforce(R, z, phi, t, potentialArgs->nwrapped,
		    potentialArgs->wrappedPotentialArg);
    //back to rectangular
    *Fx= cos( phi )*Rforce - sin( phi )*phitorque / R;
    *Fy= sin( phi )*Rforce + cos( phi )*phitorque / R;
    //rotate back
    if (rotSet) {
      rotate_force(Fx,Fy,Fz,rot);
    }
    //cache
    *(args + 5)= *Fx;
    *(args + 6)= *Fy;
    *(args + 7)= *Fz;
}
//Cached getter for the rotated-back Cartesian forces: return the cached values
//when the (x,y,z,t) key matches, otherwise compute (and cache) them
static void RotateAndTiltWrapperPotentialgetforces(double R, double z,
                double phi, double t, double * Fx, double * Fy, double * Fz,
                struct potentialArg * potentialArgs){
    double * args= potentialArgs->args;
    double x, y;
    cyl_to_rect(R, phi, &x, &y);
    if ( x == *(args + 1) && y == *(args + 2) && z == *(args + 3)
         && t == *(args + 4) ){
      *Fx= *(args + 5);
      *Fy= *(args + 6);
      *Fz= *(args + 7);
    }
    else
      RotateAndTiltWrapperPotentialxyzforces(R, z, phi, t, Fx, Fy, Fz,
                                             potentialArgs);
}
double RotateAndTiltWrapperPotentialRforce(double R, double z, double phi,
        double t,
        struct potentialArg * potentialArgs){
   double * args = potentialArgs->args;
   double Fx, Fy, Fz;
   RotateAndTiltWrapperPotentialgetforces(R, z, phi, t, &Fx, &Fy, &Fz,
                                          potentialArgs);
   return *args * ( cos ( phi ) * Fx + sin ( phi ) * Fy );
}
double RotateAndTiltWrapperPotentialphitorque(double R, double z, double phi,
        double t,
        struct potentialArg * potentialArgs){
    double * args = potentialArgs->args;
    double Fx, Fy, Fz;
    RotateAndTiltWrapperPotentialgetforces(R, z, phi, t, &Fx, &Fy, &Fz,
                                           potentialArgs);
    return *args * R * ( -sin ( phi ) * Fx + cos ( phi ) * Fy );
}
double RotateAndTiltWrapperPotentialzforce(double R, double z, double phi,
        double t,
        struct potentialArg * potentialArgs){
    double * args = potentialArgs->args;
    double Fx, Fy, Fz;
    RotateAndTiltWrapperPotentialgetforces(R, z, phi, t, &Fx, &Fy, &Fz,
                                           potentialArgs);
    return *args * Fz;
}
// --- Full 3D Hessian for the variational equations ---
//Compute the Cartesian Hessian H_ij = d2Phi/dx_i/dx_j in the original
//(unprimed) frame: evaluate the wrapped potential's cylindrical Hessian at the
//transformed (aligned-frame) point x' = rot . x + offset, assemble the
//Cartesian Hessian H' there, and rotate back by conjugation
//H = rot^T . H' . rot (the constant offset does not affect derivatives).
//Mirrors the pure-Python RotateAndTiltWrapperPotential._2ndderiv_xyz.
//H holds the 6 unique components (Hxx,Hxy,Hxz,Hyy,Hyz,Hzz), w/o amp.
static void RotateAndTiltWrapperPotentialxyzhessian(double R, double z,
                double phi, double t, double * H,
                struct potentialArg * potentialArgs){
    double * args= potentialArgs->args;
    double rot[9];
    bool rotSet;
    double x, y;
    double Rforcep, phitorquep;
    double R2derivp, phi2derivp, z2derivp, Rzderivp, Rphiderivp, zphiderivp;
    double cp, sp, R2;
    double Hxx, Hxy, Hxz, Hyy, Hyz, Hzz;
    cyl_to_rect(R, phi, &x, &y);
    //caching
    *(args + 22)= x;
    *(args + 23)= y;
    *(args + 24)= z;
    *(args + 25)= t;
    //transform to the aligned frame (the center and offset do not affect
    //the derivatives)
    RotateAndTiltWrapperPotentialtobody(&x, &y, &z, t, rot, &rotSet,
                                        potentialArgs);
    rect_to_cyl(x,y,&R,&phi);
    //wrapped forces and second derivatives at the transformed point
    Rforcep= calcRforce(R, z, phi, t, potentialArgs->nwrapped,
		        potentialArgs->wrappedPotentialArg);
    phitorquep= calcphitorque(R, z, phi, t, potentialArgs->nwrapped,
		        potentialArgs->wrappedPotentialArg);
    R2derivp= calcR2deriv(R, z, phi, t, potentialArgs->nwrapped,
		        potentialArgs->wrappedPotentialArg);
    phi2derivp= calcphi2deriv(R, z, phi, t, potentialArgs->nwrapped,
		        potentialArgs->wrappedPotentialArg);
    z2derivp= calcz2deriv(R, z, phi, t, potentialArgs->nwrapped,
		        potentialArgs->wrappedPotentialArg);
    Rzderivp= calcRzderiv(R, z, phi, t, potentialArgs->nwrapped,
		        potentialArgs->wrappedPotentialArg);
    Rphiderivp= calcRphideriv(R, z, phi, t, potentialArgs->nwrapped,
		        potentialArgs->wrappedPotentialArg);
    zphiderivp= calczphideriv(R, z, phi, t, potentialArgs->nwrapped,
		        potentialArgs->wrappedPotentialArg);
    //assemble the Cartesian Hessian in the aligned frame
    cp= cos ( phi );
    sp= sin ( phi );
    R2= R * R;
    Hxx= R2derivp * cp * cp - 2. * Rphiderivp * cp * sp / R
      + phi2derivp * sp * sp / R2
      - Rforcep * sp * sp / R - 2. * phitorquep * cp * sp / R2;
    Hyy= R2derivp * sp * sp + 2. * Rphiderivp * cp * sp / R
      + phi2derivp * cp * cp / R2
      - Rforcep * cp * cp / R + 2. * phitorquep * cp * sp / R2;
    Hxy= R2derivp * cp * sp + Rphiderivp * ( cp * cp - sp * sp ) / R
      - phi2derivp * cp * sp / R2
      + Rforcep * cp * sp / R + phitorquep * ( cp * cp - sp * sp ) / R2;
    Hxz= Rzderivp * cp - zphiderivp * sp / R;
    Hyz= Rzderivp * sp + zphiderivp * cp / R;
    Hzz= z2derivp;
    //rotate back: H = rot^T . H' . rot
    if (rotSet) {
      double Hp[3][3]= {{Hxx,Hxy,Hxz},{Hxy,Hyy,Hyz},{Hxz,Hyz,Hzz}};
      double T[3][3]; // T = H' . rot
      int ii, jj;
      for (ii=0; ii < 3; ii++)
        for (jj=0; jj < 3; jj++)
          T[ii][jj]= Hp[ii][0] * *(rot+jj) + Hp[ii][1] * *(rot+3+jj)
            + Hp[ii][2] * *(rot+6+jj);
      // H_ij = sum_k rot_ki T_kj
      Hxx= *(rot)   * T[0][0] + *(rot+3) * T[1][0] + *(rot+6) * T[2][0];
      Hxy= *(rot)   * T[0][1] + *(rot+3) * T[1][1] + *(rot+6) * T[2][1];
      Hxz= *(rot)   * T[0][2] + *(rot+3) * T[1][2] + *(rot+6) * T[2][2];
      Hyy= *(rot+1) * T[0][1] + *(rot+4) * T[1][1] + *(rot+7) * T[2][1];
      Hyz= *(rot+1) * T[0][2] + *(rot+4) * T[1][2] + *(rot+7) * T[2][2];
      Hzz= *(rot+2) * T[0][2] + *(rot+5) * T[1][2] + *(rot+8) * T[2][2];
    }
    //cache
    *(args + 26)= Hxx;
    *(args + 27)= Hxy;
    *(args + 28)= Hxz;
    *(args + 29)= Hyy;
    *(args + 30)= Hyz;
    *(args + 31)= Hzz;
    H[0]= Hxx;
    H[1]= Hxy;
    H[2]= Hxz;
    H[3]= Hyy;
    H[4]= Hyz;
    H[5]= Hzz;
}
//Cached getter for the rotated-back Cartesian Hessian: return the cached
//values when the (x,y,z,t) key matches, otherwise compute (and cache) them
static void RotateAndTiltWrapperPotentialgethessian(double R, double z,
                double phi, double t, double * H,
                struct potentialArg * potentialArgs){
    double * args= potentialArgs->args;
    double x, y;
    cyl_to_rect(R, phi, &x, &y);
    if ( x == *(args + 22) && y == *(args + 23) && z == *(args + 24)
         && t == *(args + 25) ){
      H[0]= *(args + 26);
      H[1]= *(args + 27);
      H[2]= *(args + 28);
      H[3]= *(args + 29);
      H[4]= *(args + 30);
      H[5]= *(args + 31);
    }
    else
      RotateAndTiltWrapperPotentialxyzhessian(R, z, phi, t, H, potentialArgs);
}
//Cylindrical second derivatives consumed by the 3D variational equations:
//standard cylindrical components of the Cartesian Hessian H (the phi
//derivatives also pick up first-derivative (force) terms). These mirror the
//pure-Python _R2deriv/.../_phizderiv of RotateAndTiltWrapperPotential.
double RotateAndTiltWrapperPotentialR2deriv(double R, double z, double phi,
        double t,
        struct potentialArg * potentialArgs){
    double * args = potentialArgs->args;
    double H[6];
    double cp= cos ( phi );
    double sp= sin ( phi );
    RotateAndTiltWrapperPotentialgethessian(R, z, phi, t, H, potentialArgs);
    return *args * ( cp * cp * H[0] + sp * sp * H[3] + 2. * cp * sp * H[1] );
}
double RotateAndTiltWrapperPotentialz2deriv(double R, double z, double phi,
        double t,
        struct potentialArg * potentialArgs){
    double * args = potentialArgs->args;
    double H[6];
    RotateAndTiltWrapperPotentialgethessian(R, z, phi, t, H, potentialArgs);
    return *args * H[5];
}
double RotateAndTiltWrapperPotentialRzderiv(double R, double z, double phi,
        double t,
        struct potentialArg * potentialArgs){
    double * args = potentialArgs->args;
    double H[6];
    double cp= cos ( phi );
    double sp= sin ( phi );
    RotateAndTiltWrapperPotentialgethessian(R, z, phi, t, H, potentialArgs);
    return *args * ( cp * H[2] + sp * H[4] );
}
double RotateAndTiltWrapperPotentialphi2deriv(double R, double z, double phi,
        double t,
        struct potentialArg * potentialArgs){
    double * args = potentialArgs->args;
    double H[6];
    double Fx, Fy, Fz;
    double cp= cos ( phi );
    double sp= sin ( phi );
    RotateAndTiltWrapperPotentialgethessian(R, z, phi, t, H, potentialArgs);
    RotateAndTiltWrapperPotentialgetforces(R, z, phi, t, &Fx, &Fy, &Fz,
                                           potentialArgs);
    return *args * ( R * R * ( sp * sp * H[0] + cp * cp * H[3]
                               - 2. * cp * sp * H[1] )
                     + R * ( cp * Fx + sp * Fy ) );
}
double RotateAndTiltWrapperPotentialRphideriv(double R, double z, double phi,
        double t,
        struct potentialArg * potentialArgs){
    double * args = potentialArgs->args;
    double H[6];
    double Fx, Fy, Fz;
    double cp= cos ( phi );
    double sp= sin ( phi );
    RotateAndTiltWrapperPotentialgethessian(R, z, phi, t, H, potentialArgs);
    RotateAndTiltWrapperPotentialgetforces(R, z, phi, t, &Fx, &Fy, &Fz,
                                           potentialArgs);
    return *args * ( R * cp * sp * ( H[3] - H[0] )
                     + R * cos ( 2. * phi ) * H[1]
                     + sp * Fx - cp * Fy );
}
double RotateAndTiltWrapperPotentialzphideriv(double R, double z, double phi,
        double t,
        struct potentialArg * potentialArgs){
    double * args = potentialArgs->args;
    double H[6];
    double cp= cos ( phi );
    double sp= sin ( phi );
    RotateAndTiltWrapperPotentialgethessian(R, z, phi, t, H, potentialArgs);
    return *args * R * ( cp * H[4] - sp * H[2] );
}
