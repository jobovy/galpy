#include <galpy_potentials.h>
#include <bovy_coords.h>
//NonInertialFrameForce
//arguments: amp
void NonInertialFrameForcexyzforces_xyz(double R,double z,double phi,double t,
                                        double vR,double vT,double vz,
				                                double * Fx, double * Fy, double * Fz,
                                        struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  double x, y, vx, vy;
  bool rot_acc, lin_acc, omegaz_only, const_freq;
  double Omegax, Omegay, Omegaz;
  double Omega2, Omegatimesvecx;
  double x0x, x0y, x0z, v0x, v0y, v0z;
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
  if ( rot_acc ) {
    omegaz_only= (bool) *(args + 13);
    const_freq= (bool) *(args + 14);
    if ( omegaz_only ) {
      Omegaz= *(args + 17);
      if ( !const_freq ) {
        Omegaz+= *(args + 21) * t;
        Omega2= Omegaz * Omegaz;
      } else {
        Omega2= *(args + 18);
      }
      *Fx+=  2. * Omegaz * vy + Omega2 * x;
      *Fy+= -2. * Omegaz * vx + Omega2 * y;
      if ( !const_freq ) {
        *Fx+= *(args + 21) * y;
        *Fy-= *(args + 21) * x;
      }
      if ( lin_acc ) {
        x0x= (*(*(potentialArgs->tfuncs+3)))(t);
        x0y= (*(*(potentialArgs->tfuncs+4)))(t);
        v0x= (*(*(potentialArgs->tfuncs+6)))(t);
        v0y= (*(*(potentialArgs->tfuncs+7)))(t);
        *Fx+=  2. * Omegaz * v0y + Omega2 * x0x;
        *Fy+= -2. * Omegaz * v0x + Omega2 * x0y;
        if ( !const_freq ) {
         *Fx+= *(args + 21) * x0y;
          *Fy-= *(args + 21) * x0x;
        }
      }
    } else {
      Omegax= *(args + 15);
      Omegay= *(args + 16);
      Omegaz= *(args + 17);
      if ( !const_freq ) {
        Omegax+= *(args + 19) * t;
        Omegay+= *(args + 20) * t;
        Omegaz+= *(args + 21) * t;
        Omega2= Omegax * Omegax + Omegay * Omegay + Omegaz * Omegaz;
      } else {
        Omega2= *(args + 18);
      }
      Omegatimesvecx= Omegax * x + Omegay * y + Omegaz * z;
      *Fx+=  2. * ( Omegaz * vy - Omegay * vz ) + Omega2 * x - Omegax * Omegatimesvecx;
      *Fy+= -2. * ( Omegaz * vx - Omegax * vz ) + Omega2 * y - Omegay * Omegatimesvecx;
      *Fz+=  2. * ( Omegay * vx - Omegax * vy ) + Omega2 * z - Omegaz * Omegatimesvecx;
      if ( !const_freq ) {
        *Fx-= -*(args + 21) * y + *(args + 20) * z;
        *Fy-=  *(args + 21) * x - *(args + 19) * z;
        *Fz-= -*(args + 20) * x + *(args + 19) * y;
      }
      if ( lin_acc ) {
        x0x= (*(*(potentialArgs->tfuncs+3)))(t);
        x0y= (*(*(potentialArgs->tfuncs+4)))(t);
        x0z= (*(*(potentialArgs->tfuncs+5)))(t);
        v0x= (*(*(potentialArgs->tfuncs+6)))(t);
        v0y= (*(*(potentialArgs->tfuncs+7)))(t);
        v0z= (*(*(potentialArgs->tfuncs+8)))(t);
        // Re-use variable
        Omegatimesvecx= Omegax * x0x + Omegay * x0y + Omegaz * x0z;
        *Fx+=  2. * ( Omegaz * v0y - Omegay * v0z ) + Omega2 * x0x - Omegax * Omegatimesvecx;
        *Fy+= -2. * ( Omegaz * v0x - Omegax * v0z ) + Omega2 * x0y - Omegay * Omegatimesvecx;
        *Fz+=  2. * ( Omegay * v0x - Omegax * v0y ) + Omega2 * x0z - Omegaz * Omegatimesvecx;
        if ( !const_freq ) {
          *Fx-= -*(args + 21) * x0y + *(args + 20) * x0z;
          *Fy-=  *(args + 21) * x0x - *(args + 19) * x0z;
          *Fz-= -*(args + 20) * x0x + *(args + 19) * x0y;
        }
      }
    }
  }
  // Linear acceleration part
  if ( lin_acc ) {
    *Fx-= (*(*(potentialArgs->tfuncs  )))(t);
    *Fy-= (*(*(potentialArgs->tfuncs+1)))(t);
    *Fz-= (*(*(potentialArgs->tfuncs+2)))(t);
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
double NonInertialFrameForcephiforce(double R,double z,double phi,double t,
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