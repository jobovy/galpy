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
