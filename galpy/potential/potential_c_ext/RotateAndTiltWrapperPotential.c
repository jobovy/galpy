#include <stdbool.h>
#include <math.h>
#include <bovy_coords.h>
#include <galpy_potentials.h>
//RotateAndTiltWrapperPotential
static inline void rotate(double *x, double *y, double *z, double *rot){
  double xp,yp,zp;
  xp= *(rot)   * *x + *(rot+1) * *y + *(rot+2) * *z;
  yp= *(rot+3) * *x + *(rot+4) * *y + *(rot+5) * *z;
  zp= *(rot+6) * *x + *(rot+7) * *y + *(rot+8) * *z;
  *x= xp;
  *y= yp;
  *z= zp;
}
static inline void rotate_force(double *Fx, double *Fy, double *Fz,
				double *rot){
  double Fxp,Fyp,Fzp;
  Fxp= *(rot)   * *Fx + *(rot+3) * *Fy + *(rot+6) * *Fz;
  Fyp= *(rot+1) * *Fx + *(rot+4) * *Fy + *(rot+7) * *Fz;
  Fzp= *(rot+2) * *Fx + *(rot+5) * *Fy + *(rot+8) * *Fz;
  *Fx= Fxp;
  *Fy= Fyp;
  *Fz= Fzp;
}
void RotateAndTiltWrapperPotentialxyzforces(double R, double z, double phi,
                 double t, double * Fx, double * Fy, double * Fz,
                 struct potentialArg * potentialArgs){
    double * args = potentialArgs->args;
    double * rot = args+1;
    double x, y, t_z;
    double t_R, t_phi;
    cyl_to_rect(R, phi, &x, &y);
    rotate(&x,&y,&t_z,rot);
    rect_to_cyl(x,y,&t_R,&t_phi);
    //now get the forces in R, phi, z
    double Rforce, phiforce;
    Rforce = calcRforce(t_R, t_z, t_phi, t, potentialArgs->nwrapped,
                        potentialArgs->wrappedPotentialArg);
    phiforce = calcPhiforce(t_R, t_z, t_phi, t, potentialArgs->nwrapped,
                        potentialArgs->wrappedPotentialArg);
    *Fz = calczforce(t_R, t_z, t_phi, t, potentialArgs->nwrapped,
                        potentialArgs->wrappedPotentialArg);
    //back to rectangular
    *Fx = cos( t_phi )*Rforce - sin( t_phi )*phiforce;
    *Fy = sin( t_phi )*Rforce + cos( t_phi )*phiforce;
    //rotate back
    rotate_force(Fx,Fy,Fz,rot);
}
double RotateAndTiltWrapperPotentialRforce(double R, double z, double phi,
        double t,
        struct potentialArg * potentialArgs){
   double * args = potentialArgs->args;
   // change the zvector, calculate Rforce
   double Fx, Fy, Fz;
   RotateAndTiltWrapperPotentialxyzforces(R, z, phi, t, &Fx, &Fy, &Fz,
                                          potentialArgs);
   return *args * ( cos ( phi ) * Fx + sin ( phi ) * Fy );
}
double RotateAndTiltWrapperPotentialphiforce(double R, double z, double phi,
        double t,
        struct potentialArg * potentialArgs){
    double * args = potentialArgs->args;
    // change the zvector, calculate Rforce
    double Fx, Fy, Fz;
    RotateAndTiltWrapperPotentialxyzforces(R, z, phi, t, &Fx, &Fy, &Fz,
                                           potentialArgs);
    return *args * ( -sin ( phi ) * Fx + cos ( phi ) * Fy );
}
double RotateAndTiltWrapperPotentialzforce(double R, double z, double phi,
        double t,
        struct potentialArg * potentialArgs){
    double * args = potentialArgs->args;
    // change the zvector, calculate Rforce
    double Fx, Fy, Fz;
    RotateAndTiltWrapperPotentialxyzforces(R, z, phi, t, &Fx, &Fy, &Fz,
                                           potentialArgs);
    return *args * Fz;
}
double RotateAndTiltWrapperPotentialPlanarRforce(double R, double phi,
        double t,
        struct potentialArg * potentialArgs){
    return EllipsoidalPotentialRforce(R, 0., phi, t, potentialArgs);
}
double RotateAndTiltWrapperPotentialPlanarphiforce(double R, double phi,
        double t,
        struct potentialArg * potentialArgs){
    return EllipsoidalPotentialphiforce(R, 0., phi, t, potentialArgs);
}
