#include <stdbool.h>
#include <math.h>
#include <bovy_coords.h>
#include <galpy_potentials.h>

//RotateAndTiltWrapperPotential
//
//arguments layout (args):
//  0      : amp
//  1- 3   : cached (x,y,z) force-cache key
//  4- 6   : cached rotated-back Cartesian forces (Fx,Fy,Fz), w/o amp
//  7-15   : rotation matrix rot (row-major); x_aligned = rot . x
//  16     : rotSet
//  17     : offsetSet
//  18-20  : offset (applied in the aligned frame, after the rotation)
//  21-24  : cached (x,y,z,t) Hessian-cache key (NaN when empty; t included
//           because the wrapped potential may be explicitly time-dependent)
//  25-30  : cached rotated-back Cartesian Hessian (Hxx,Hxy,Hxz,Hyy,Hyz,Hzz),
//           w/o amp
void RotateAndTiltWrapperPotentialxyzforces(double R, double z, double phi,
                 double t, double * Fx, double * Fy, double * Fz,
                 struct potentialArg * potentialArgs){
    double * args= potentialArgs->args;
    double * rot= args+7;
    bool rotSet= (bool) *(args+16);
    bool offsetSet= (bool) *(args+17);
    double * offset= args+18;
    double x, y;
    double Rforce, phitorque;
    cyl_to_rect(R, phi, &x, &y);
    //caching
    *(args + 1)= x;
    *(args + 2)= y;
    *(args + 3)= z;
    //now get the forces in R, phi, z in the aligned frame
    if (rotSet) {
      rotate(&x,&y,&z,rot);
    }
    if (offsetSet) {
      x += *(offset);
      y += *(offset+1);
      z += *(offset+2);
    }
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
    *(args + 4)= *Fx;
    *(args + 5)= *Fy;
    *(args + 6)= *Fz;
}
//Cached getter for the rotated-back Cartesian forces: return the cached values
//when the (x,y,z) key matches, otherwise compute (and cache) them
static void RotateAndTiltWrapperPotentialgetforces(double R, double z,
                double phi, double t, double * Fx, double * Fy, double * Fz,
                struct potentialArg * potentialArgs){
    double * args= potentialArgs->args;
    double x, y;
    cyl_to_rect(R, phi, &x, &y);
    if ( x == *(args + 1) && y == *(args + 2) && z == *(args + 3) ){
      *Fx= *(args + 4);
      *Fy= *(args + 5);
      *Fz= *(args + 6);
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
    double * rot= args+7;
    bool rotSet= (bool) *(args+16);
    bool offsetSet= (bool) *(args+17);
    double * offset= args+18;
    double x, y;
    double Rforcep, phitorquep;
    double R2derivp, phi2derivp, z2derivp, Rzderivp, Rphiderivp, zphiderivp;
    double cp, sp, R2;
    double Hxx, Hxy, Hxz, Hyy, Hyz, Hzz;
    cyl_to_rect(R, phi, &x, &y);
    //caching
    *(args + 21)= x;
    *(args + 22)= y;
    *(args + 23)= z;
    *(args + 24)= t;
    //transform to the aligned frame
    if (rotSet) {
      rotate(&x,&y,&z,rot);
    }
    if (offsetSet) {
      x += *(offset);
      y += *(offset+1);
      z += *(offset+2);
    }
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
    *(args + 25)= Hxx;
    *(args + 26)= Hxy;
    *(args + 27)= Hxz;
    *(args + 28)= Hyy;
    *(args + 29)= Hyz;
    *(args + 30)= Hzz;
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
    if ( x == *(args + 21) && y == *(args + 22) && z == *(args + 23)
         && t == *(args + 24) ){
      H[0]= *(args + 25);
      H[1]= *(args + 26);
      H[2]= *(args + 27);
      H[3]= *(args + 28);
      H[4]= *(args + 29);
      H[5]= *(args + 30);
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
