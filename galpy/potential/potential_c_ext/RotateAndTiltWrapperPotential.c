#include <stdbool.h>
#include <math.h>
#include <bovy_coords.h>
#include <galpy_potentials.h>

//RotateAndTiltWrapperPotential
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
double RotateAndTiltWrapperPotentialRforce(double R, double z, double phi,
        double t,
        struct potentialArg * potentialArgs){
   double * args = potentialArgs->args;
   //get cached xyz
   double cached_x = *(args + 1);
   double cached_y = *(args + 2);
   double cached_z = *(args + 3);
   // change the zvector, calculate Rforce
   double Fx, Fy, Fz;
   double x, y;
   cyl_to_rect(R, phi, &x, &y);
   if ( x == cached_x && y == cached_y && z == cached_z ){
    Fx = *(args + 4);
    Fy = *(args + 5);
    Fz = *(args + 6);
   }
   else
    RotateAndTiltWrapperPotentialxyzforces(R, z, phi, t, &Fx, &Fy, &Fz,
                                           potentialArgs);
   return *args * ( cos ( phi ) * Fx + sin ( phi ) * Fy );
}
double RotateAndTiltWrapperPotentialphitorque(double R, double z, double phi,
        double t,
        struct potentialArg * potentialArgs){
    double * args = potentialArgs->args;
    //get cached xyz
    double cached_x = *(args + 1);
    double cached_y = *(args + 2);
    double cached_z = *(args + 3);
    // change the zvector, calculate Rforce
    double Fx, Fy, Fz;
    double x, y;
    cyl_to_rect(R, phi, &x, &y);
    if ( x == cached_x && y == cached_y && z == cached_z ){
     Fx = *(args + 4);
     Fy = *(args + 5);
     Fz = *(args + 6);
    }
    else
    // LCOV_EXCL_START
     RotateAndTiltWrapperPotentialxyzforces(R, z, phi, t, &Fx, &Fy, &Fz,
                                            potentialArgs);
    // LCOV_EXCL_STOP
    return *args * R * ( -sin ( phi ) * Fx + cos ( phi ) * Fy );
}
double RotateAndTiltWrapperPotentialzforce(double R, double z, double phi,
        double t,
        struct potentialArg * potentialArgs){
    double * args = potentialArgs->args;
    //get cached xyz
    double cached_x = *(args + 1);
    double cached_y = *(args + 2);
    double cached_z = *(args + 3);
    // change the zvector, calculate Rforce
    double Fx, Fy, Fz;
    double x, y;
    cyl_to_rect(R, phi, &x, &y);
    if ( x == cached_x && y == cached_y && z == cached_z ){
     Fx = *(args + 4);
     Fy = *(args + 5);
     Fz = *(args + 6);
    }
    else
    // LCOV_EXCL_START
     RotateAndTiltWrapperPotentialxyzforces(R, z, phi, t, &Fx, &Fy, &Fz,
                                            potentialArgs);
    // LCOV_EXCL_STOP
    return *args * Fz;
}
