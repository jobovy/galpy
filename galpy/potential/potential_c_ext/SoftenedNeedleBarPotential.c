#include <math.h>
#include <bovy_coords.h>
#include <galpy_potentials.h>
//SoftenedNeedleBarPotentials
static inline void compute_TpTm(double x, double y, double z,
				double *Tp, double *Tm,
				double a, double b, double c2){
  double secondpart= y * y + pow( b + sqrt ( z * z + c2 ) , 2);
  *Tp= sqrt ( pow ( a + x , 2) + secondpart );
  *Tm= sqrt ( pow ( a - x , 2) + secondpart );
}
double SoftenedNeedleBarPotentialEval(double R,double z, double phi,
				      double t,
				      struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  //Get args: amp, a, b, c2, pa, omegab
  double amp= *args++;
  double a= *args++;
  double b= *args++;
  double c2= *args++;
  double pa= *args++;
  double omegab= *args++;
  double x,y;
  double Tp,Tm;
  //Calculate potential
  cyl_to_rect(R,phi-pa-omegab*t,&x,&y);
  compute_TpTm(x,y,z,&Tp,&Tm,a,b,c2);
  return 0.5 * amp * log( ( x - a + Tm ) / ( x + a + Tp ) ) / a;
}
void SoftenedNeedleBarPotentialxyzforces_xyz(double R,double z, double phi,
					     double t,double * args,
					     double a,double b, double c2,
					     double pa, double omegab,
					     double cached_R, double cached_z,
					     double cached_phi,
					     double cached_t){
  double x,y;
  double Tp,Tm;
  double Fx, Fy, Fz;
  double zc;
  double cp, sp;
  if ( R != cached_R || phi != cached_phi || z != cached_z || t != cached_t){
    // Set up cache
    *args= R;
    *(args + 1)= z;
    *(args + 2)= phi;
    *(args + 3)= t;
    // Compute forces in rectangular, aligned frame
    cyl_to_rect(R,phi-pa-omegab*t,&x,&y);
    compute_TpTm(x,y,z,&Tp,&Tm,a,b,c2);
    zc= sqrt ( z * z + c2 );
    Fx= -2. * x / Tp / Tm / (Tp+Tm);
    Fy= -y / 2. / Tp /Tm * ( Tp + Tm -4. * x * x / (Tp+Tm) )	\
      / ( y * y + pow( b + zc, 2));
    Fz= -z / 2. / Tp /Tm * ( Tp + Tm -4. * x * x / (Tp+Tm) )	\
      / ( y * y + pow( b + zc, 2)) * ( b + zc ) / zc;
    cp= cos ( pa + omegab * t );
    sp= sin ( pa + omegab * t );
    // Rotate to rectangular, correct frame
    *(args + 4)= cp * Fx - sp * Fy;
    *(args + 5)= sp * Fx + cp * Fy;
    *(args + 6)= Fz;
  }
}
double SoftenedNeedleBarPotentialRforce(double R,double z, double phi,
					double t,
					struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  //Get args: amp, a, b, c2, pa, omegab
  double amp= *args++;
  double a= *args++;
  double b= *args++;
  double c2= *args++;
  double pa= *args++;
  double omegab= *args++;
  double cached_R= *args;
  double cached_z= *(args + 1);
  double cached_phi= *(args + 2);
  double cached_t= *(args + 3);
  //Calculate potential
  SoftenedNeedleBarPotentialxyzforces_xyz(R,z,phi,t,args,a,b,c2,pa,omegab,
					  cached_R,cached_z,
					  cached_phi,cached_t);
  return amp * ( cos ( phi ) * *(args + 4) + sin( phi ) * *(args + 5) );
}
double SoftenedNeedleBarPotentialPlanarRforce(double R,double phi,double t,
					      struct potentialArg * potentialArgs){
  return SoftenedNeedleBarPotentialRforce(R,0.,phi,t,potentialArgs);
}
double SoftenedNeedleBarPotentialphitorque(double R,double z, double phi,
					 double t,
					 struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  //Get args: amp, a, b, c2, pa, omegab
  double amp= *args++;
  double a= *args++;
  double b= *args++;
  double c2= *args++;
  double pa= *args++;
  double omegab= *args++;
  double cached_R= *args;
  double cached_z= *(args + 1);
  double cached_phi= *(args + 2);
  double cached_t= *(args + 3);
  //Calculate potential
  SoftenedNeedleBarPotentialxyzforces_xyz(R,z,phi,t,args,a,b,c2,pa,omegab,
					  cached_R,cached_z,
					  cached_phi,cached_t);
  return amp * R * ( -sin ( phi ) * *(args + 4) + cos( phi ) * *(args + 5) );
}
double SoftenedNeedleBarPotentialPlanarphitorque(double R, double phi,double t,
					  struct potentialArg * potentialArgs){
  return SoftenedNeedleBarPotentialphitorque(R,0.,phi,t,potentialArgs);
}
double SoftenedNeedleBarPotentialzforce(double R,double z, double phi,
				       double t,
				       struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  //Get args: amp, a, b, c2, pa, omegab
  double amp= *args++;
  double a= *args++;
  double b= *args++;
  double c2= *args++;
  double pa= *args++;
  double omegab= *args++;
  double cached_R= *args;
  double cached_z= *(args + 1);
  double cached_phi= *(args + 2);
  double cached_t= *(args + 3);
  //Calculate potential
  SoftenedNeedleBarPotentialxyzforces_xyz(R,z,phi,t,args,a,b,c2,pa,omegab,
					  cached_R,cached_z,
					  cached_phi,cached_t);
  return amp * *(args + 6);
}
// Full-3D Hessian for the variational equations (integrate_dxdv).
// In the bar-aligned frame (phid = phi - pa - omegab t, x = R cos phid,
// y = R sin phid) the Cartesian Hessian of Phi = ln[(x-a+Tm)/(x+a+Tp)]/(2a)
// is closed-form; the cylindrical Hessian follows from the standard polar
// transformation evaluated at phid (the rigid bar rotation only shifts the
// azimuth, so d/dphi = d/dphid). hess[0..5]= RR, zz, phiphi, Rz, Rphi, zphi
// (all WITHOUT the amplitude).
static inline void SoftenedNeedleBarPotentialCylHess(double R,double z,
						     double phi,double t,
						     double a,double b,
						     double c2,double pa,
						     double omegab,
						     double * hess){
  double cd, sd, x, y, zc, u, s2, Tp, Tm, Tp3, Tm3;
  double G, H, K, w, px, py, pxx, pxy, pxz, pyy, pyz, pzz;
  cd= cos ( phi - pa - omegab * t );
  sd= sin ( phi - pa - omegab * t );
  x= R * cd;
  y= R * sd;
  zc= sqrt ( z * z + c2 );
  u= b + zc;
  s2= y * y + u * u;
  compute_TpTm(x,y,z,&Tp,&Tm,a,b,c2);
  Tp3= Tp * Tp * Tp;
  Tm3= Tm * Tm * Tm;
  // Building blocks: G/H weight the two bar ends, K their difference,
  // w = dzc-chain factor z(b+zc)/zc
  G= ( a - x ) / Tm + ( a + x ) / Tp;
  H= ( a - x ) / Tm3 + ( a + x ) / Tp3;
  K= 1. / Tm3 - 1. / Tp3;
  w= z * u / zc;
  // Aligned-frame gradient (= -force)
  px= ( 1. / Tm - 1. / Tp ) / 2. / a;
  py= y * G / 2. / a / s2;
  // Aligned-frame Cartesian Hessian
  pxx= H / 2. / a;
  pxy= -y * K / 2. / a;
  pxz= -w * K / 2. / a;
  pyy= ( G * ( s2 - 2. * y * y ) / s2 / s2 - y * y * H / s2 ) / 2. / a;
  pyz= -y * w * ( 2. * G / s2 / s2 + H / s2 ) / 2. / a;
  pzz= ( ( 1. + b * c2 / zc / zc / zc ) * G / s2
	 - w * w * ( H / s2 + 2. * G / s2 / s2 ) ) / 2. / a;
  // Cylindrical Hessian at phid
  *(hess + 0)= cd * cd * pxx + 2. * cd * sd * pxy + sd * sd * pyy;
  *(hess + 1)= pzz;
  *(hess + 2)= R * R * ( sd * sd * pxx - 2. * sd * cd * pxy + cd * cd * pyy )
    - R * ( cd * px + sd * py );
  *(hess + 3)= cd * pxz + sd * pyz;
  *(hess + 4)= R * ( -sd * cd * pxx + ( cd * cd - sd * sd ) * pxy
		     + sd * cd * pyy ) - sd * px + cd * py;
  *(hess + 5)= R * ( -sd * pxz + cd * pyz );
}
double SoftenedNeedleBarPotentialR2deriv(double R,double z,double phi,
					 double t,
					 struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  double hess[6];
  //Get args: amp, a, b, c2, pa, omegab
  double amp= *args++;
  double a= *args++;
  double b= *args++;
  double c2= *args++;
  double pa= *args++;
  double omegab= *args++;
  SoftenedNeedleBarPotentialCylHess(R,z,phi,t,a,b,c2,pa,omegab,hess);
  return amp * hess[0];
}
double SoftenedNeedleBarPotentialz2deriv(double R,double z,double phi,
					 double t,
					 struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  double hess[6];
  //Get args: amp, a, b, c2, pa, omegab
  double amp= *args++;
  double a= *args++;
  double b= *args++;
  double c2= *args++;
  double pa= *args++;
  double omegab= *args++;
  SoftenedNeedleBarPotentialCylHess(R,z,phi,t,a,b,c2,pa,omegab,hess);
  return amp * hess[1];
}
double SoftenedNeedleBarPotentialphi2deriv(double R,double z,double phi,
					   double t,
					   struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  double hess[6];
  //Get args: amp, a, b, c2, pa, omegab
  double amp= *args++;
  double a= *args++;
  double b= *args++;
  double c2= *args++;
  double pa= *args++;
  double omegab= *args++;
  SoftenedNeedleBarPotentialCylHess(R,z,phi,t,a,b,c2,pa,omegab,hess);
  return amp * hess[2];
}
double SoftenedNeedleBarPotentialRzderiv(double R,double z,double phi,
					 double t,
					 struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  double hess[6];
  //Get args: amp, a, b, c2, pa, omegab
  double amp= *args++;
  double a= *args++;
  double b= *args++;
  double c2= *args++;
  double pa= *args++;
  double omegab= *args++;
  SoftenedNeedleBarPotentialCylHess(R,z,phi,t,a,b,c2,pa,omegab,hess);
  return amp * hess[3];
}
double SoftenedNeedleBarPotentialRphideriv(double R,double z,double phi,
					   double t,
					   struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  double hess[6];
  //Get args: amp, a, b, c2, pa, omegab
  double amp= *args++;
  double a= *args++;
  double b= *args++;
  double c2= *args++;
  double pa= *args++;
  double omegab= *args++;
  SoftenedNeedleBarPotentialCylHess(R,z,phi,t,a,b,c2,pa,omegab,hess);
  return amp * hess[4];
}
double SoftenedNeedleBarPotentialzphideriv(double R,double z,double phi,
					   double t,
					   struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  double hess[6];
  //Get args: amp, a, b, c2, pa, omegab
  double amp= *args++;
  double a= *args++;
  double b= *args++;
  double c2= *args++;
  double pa= *args++;
  double omegab= *args++;
  SoftenedNeedleBarPotentialCylHess(R,z,phi,t,a,b,c2,pa,omegab,hess);
  return amp * hess[5];
}
double SoftenedNeedleBarPotentialPlanarR2deriv(double R,double phi,double t,
					       struct potentialArg * potentialArgs){
  return SoftenedNeedleBarPotentialR2deriv(R,0.,phi,t,potentialArgs);
}
double SoftenedNeedleBarPotentialPlanarphi2deriv(double R,double phi,double t,
						 struct potentialArg * potentialArgs){
  return SoftenedNeedleBarPotentialphi2deriv(R,0.,phi,t,potentialArgs);
}
double SoftenedNeedleBarPotentialPlanarRphideriv(double R,double phi,double t,
						 struct potentialArg * potentialArgs){
  return SoftenedNeedleBarPotentialRphideriv(R,0.,phi,t,potentialArgs);
}
