#include <stdbool.h>
#include <math.h>
#include <galpy_potentials.h>
//General routines for TwoPowerTriaxialPotentials, specific potentials below
//TriaxialNFWPotential
inline void cyl_to_rect(double R, double phi,double *x, double *y){
  *x= R * cos ( phi );
  *y= R * sin ( phi );
}
inline void rotate(double *x, double *y, double *z, double *rot){
  double xp,yp,zp;
  xp= *(rot)   * *x + *(rot+1) * *y + *(rot+2) * *z;
  yp= *(rot+3) * *x + *(rot+4) * *y + *(rot+5) * *z;
  zp= *(rot+6) * *x + *(rot+7) * *y + *(rot+8) * *z;
  *x= xp;
  *y= yp;
  *z= zp;
}
inline void rotate_force(double *Fx, double *Fy, double *Fz, double *rot){
  double Fxp,Fyp,Fzp;
  Fxp= *(rot)   * *Fx + *(rot+3) * *Fy + *(rot+6) * *Fz;
  Fyp= *(rot+1) * *Fx + *(rot+4) * *Fy + *(rot+7) * *Fz;
  Fzp= *(rot+2) * *Fx + *(rot+5) * *Fy + *(rot+8) * *Fz;
  *Fx= Fxp;
  *Fy= Fyp;
  *Fz= Fzp;
}
inline double dens(double m, double a, double alpha, double beta){
  return pow ( a / m, alpha) * pow ( 1. + m / a, alpha - beta);
}
double TwoPowerTriaxialPotentialxforce_xyz_integrand(double s,
						     double x,double y,
						     double z,double a,
						     double alpha, double beta,
						     double b2,double c2){
  double t= 1. / s / s - 1.;
  return dens( sqrt ( x * x / ( 1. + t ) \
		      + y * y / ( b2 + t ) \
		      + z * z / ( c2 + t ) ),a,alpha,beta)\
    * x / ( 1. + t )							\
    / sqrt ( ( 1. + ( b2 - 1. ) * s * s ) * ( 1. + ( c2 - 1. ) * s * s ));
}
double TwoPowerTriaxialPotentialyforce_xyz_integrand(double s,
						     double x,double y,
						     double z,double a,
						     double alpha, double beta,
						     double b2,double c2){
  double t= 1. / s / s - 1.;
  return dens( sqrt ( x * x / ( 1. + t ) \
		      + y * y / ( b2 + t ) \
		      + z * z / ( c2 + t ) ),a,alpha,beta)\
    * y / ( b2 + t )							\
    / sqrt ( ( 1. + ( b2 - 1. ) * s * s ) * ( 1. + ( c2 - 1. ) * s * s ));
}
double TwoPowerTriaxialPotentialzforce_xyz_integrand(double s,
						     double x,double y,
						     double z,double a,
						     double alpha, double beta,
						     double b2,double c2){
  double t= 1. / s / s - 1.;
  return dens( sqrt ( x * x / ( 1. + t ) \
		      + y * y / ( b2 + t ) \
		      + z * z / ( c2 + t ) ),a,alpha,beta)\
    * z / ( c2 + t )							\
    / sqrt ( ( 1. + ( b2 - 1. ) * s * s ) * ( 1. + ( c2 - 1. ) * s * s ));
}
void TwoPowerTriaxialPotentialxyzforces_xyz(double x,double y, double z,
					    double * Fx, double * Fy, 
					    double * Fz,double *args,
					    double a,
					    double alpha, double beta,
					    double b, double c, 
					    double b2, double c2,
					    bool aligned, double * rot, 
					    int glorder,
					    double * glx, double * glw){
  int ii;
  *(args + 2 * glorder + 1)= x;
  *(args + 2 * glorder + 2)= y;
  *(args + 2 * glorder + 3)= z;
  if ( !aligned ) 
    rotate(&x,&y,&z,rot);
  *Fx= 0.;
  *Fy= 0.;
  *Fz= 0.;
  for (ii=0; ii < glorder; ii++) {
    *Fx+= *(glw+ii)							\
      * TwoPowerTriaxialPotentialxforce_xyz_integrand(*(glx+ii),x,y,z,
						      a,alpha,beta,b2,c2);
    *Fy+= *(glw+ii)							\
      * TwoPowerTriaxialPotentialyforce_xyz_integrand(*(glx+ii),x,y,z,
						      a,alpha,beta,b2,c2);
    *Fz+= *(glw+ii)							\
      * TwoPowerTriaxialPotentialzforce_xyz_integrand(*(glx+ii),x,y,z,
						      a,alpha,beta,b2,c2);
  }
  *Fx*= -b * c / a / a / a;
  *Fy*= -b * c / a / a / a;
  *Fz*= -b * c / a / a / a;
  *(args + 2 * glorder + 4)= *Fx;
  *(args + 2 * glorder + 5)= *Fy;
  *(args + 2 * glorder + 6)= *Fz;
}
double TwoPowerTriaxialPotentialRforce(double R,double z, double phi,
				       double t,
				       double alpha, double beta,
				       struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  //Get args
  double amp= *args++;
  double a= *args++;
  double b= *args++;
  double b2= *args++;
  double c= *args++;
  double c2= *args++;
  bool aligned= (bool) *args++;
  double * rot= args;
  args+= 9;
  int glorder= (int) *args++;
  double * glx= args;
  double * glw= args + glorder;
  double cached_x= *(args + 2 * glorder + 1);
  double cached_y= *(args + 2 * glorder + 2);
  double cached_z= *(args + 2 * glorder + 3);
  //Calculate potential
  double x, y;
  double Fx, Fy, Fz;
  cyl_to_rect(R,phi,&x,&y);
  if ( x == cached_x && y == cached_y && z == cached_z ){
    //LCOV_EXCL_START
    Fx= *(args + 2 * glorder + 4);
    Fy= *(args + 2 * glorder + 5);
    Fz= *(args + 2 * glorder + 6);
    //LCOV_EXCL_STOP
  }
  else 
    TwoPowerTriaxialPotentialxyzforces_xyz(x,y,z,&Fx,&Fy,&Fz,args,
					   a,alpha,beta,b,c,b2,c2,
					   aligned,rot,glorder,glx,glw);
  if ( !aligned )
    rotate_force(&Fx,&Fy,&Fz,rot);
  return amp * ( cos ( phi ) * Fx + sin( phi ) * Fy );
}
double TwoPowerTriaxialPotentialphiforce(double R,double z, double phi,
					 double t,
					 double alpha, double beta,
					 struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  //Get args
  double amp= *args++;
  double a= *args++;
  double b= *args++;
  double b2= *args++;
  double c= *args++;
  double c2= *args++;
  bool aligned= (bool) *args++;
  double * rot= args;
  args+= 9;
  int glorder= (int) *args++;
  double * glx= args;
  double * glw= args + glorder;
  double cached_x= *(args + 2 * glorder + 1);
  double cached_y= *(args + 2 * glorder + 2);
  double cached_z= *(args + 2 * glorder + 3);
  //Calculate potential
  double x, y;
  double Fx, Fy, Fz;
  cyl_to_rect(R,phi,&x,&y);
  if ( x == cached_x && y == cached_y && z == cached_z ){
    Fx= *(args + 2 * glorder + 4);
    Fy= *(args + 2 * glorder + 5);
    Fz= *(args + 2 * glorder + 6);
  }
  else 
    //LCOV_EXCL_START
    TwoPowerTriaxialPotentialxyzforces_xyz(x,y,z,&Fx,&Fy,&Fz,args,
					   a,alpha,beta,b,c,b2,c2,
					   aligned,rot,glorder,glx,glw);
    //LCOV_EXCL_STOP
  if ( !aligned )
    rotate_force(&Fx,&Fy,&Fz,rot);
  return amp * R * ( -sin ( phi ) * Fx + cos( phi ) * Fy );
}
double TwoPowerTriaxialPotentialzforce(double R,double z, double phi,
				       double t,
				       double alpha, double beta,
				       struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  //Get args
  double amp= *args++;
  double a= *args++;
  double b= *args++;
  double b2= *args++;
  double c= *args++;
  double c2= *args++;
  bool aligned= (bool) *args++;
  double * rot= args;
  args+= 9;
  int glorder= (int) *args++;
  double * glx= args;
  double * glw= args + glorder;
  double cached_x= *(args + 2 * glorder + 1);
  double cached_y= *(args + 2 * glorder + 2);
  double cached_z= *(args + 2 * glorder + 3);
  //Calculate potential
  double x, y;
  double Fx, Fy, Fz;
  cyl_to_rect(R,phi,&x,&y);
  if ( x == cached_x && y == cached_y && z == cached_z ){
    Fx= *(args + 2 * glorder + 4);
    Fy= *(args + 2 * glorder + 5);
    Fz= *(args + 2 * glorder + 6);
  }
  else 
    //LCOV_EXCL_START
    TwoPowerTriaxialPotentialxyzforces_xyz(x,y,z,&Fx,&Fy,&Fz,args,
					   a,alpha,beta,b,c,b2,c2,
					   aligned,rot,glorder,glx,glw);
    //LCOV_EXCL_STOP
  if ( !aligned )
    rotate_force(&Fx,&Fy,&Fz,rot);
  return amp * Fz;
}
//NFW
double TriaxialNFWPotentialRforce(double R,double z, double phi,
				  double t,
				  struct potentialArg * potentialArgs){
  return TwoPowerTriaxialPotentialRforce(R,z,phi,t,1,3,potentialArgs);
}
double TriaxialNFWPotentialPlanarRforce(double R,double phi,double t,
					struct potentialArg * potentialArgs){
  return TwoPowerTriaxialPotentialRforce(R,0.,phi,t,1,3,potentialArgs);
}
double TriaxialNFWPotentialphiforce(double R,double z, double phi,
				    double t,
				    struct potentialArg * potentialArgs){
  return TwoPowerTriaxialPotentialphiforce(R,z,phi,t,1,3,potentialArgs);
}
double TriaxialNFWPotentialPlanarphiforce(double R,double phi,double t,
					  struct potentialArg * potentialArgs){
  return TwoPowerTriaxialPotentialphiforce(R,0.,phi,t,1,3,potentialArgs);
}
double TriaxialNFWPotentialzforce(double R,double z, double phi,
				  double t,
				  struct potentialArg * potentialArgs){
  return TwoPowerTriaxialPotentialzforce(R,z,phi,t,1,3,potentialArgs);
}
//Hernquist
double TriaxialHernquistPotentialRforce(double R,double z, double phi,
				  double t,
				  struct potentialArg * potentialArgs){
  return TwoPowerTriaxialPotentialRforce(R,z,phi,t,1,4,potentialArgs);
}
double TriaxialHernquistPotentialPlanarRforce(double R,double phi,double t,
					struct potentialArg * potentialArgs){
  return TwoPowerTriaxialPotentialRforce(R,0.,phi,t,1,4,potentialArgs);
}
double TriaxialHernquistPotentialphiforce(double R,double z, double phi,
				    double t,
				    struct potentialArg * potentialArgs){
  return TwoPowerTriaxialPotentialphiforce(R,z,phi,t,1,4,potentialArgs);
}
double TriaxialHernquistPotentialPlanarphiforce(double R,double phi,double t,
					  struct potentialArg * potentialArgs){
  return TwoPowerTriaxialPotentialphiforce(R,0.,phi,t,1,4,potentialArgs);
}
double TriaxialHernquistPotentialzforce(double R,double z, double phi,
				  double t,
				  struct potentialArg * potentialArgs){
  return TwoPowerTriaxialPotentialzforce(R,z,phi,t,1,4,potentialArgs);
}
//Jaffe
double TriaxialJaffePotentialRforce(double R,double z, double phi,
				  double t,
				  struct potentialArg * potentialArgs){
  return TwoPowerTriaxialPotentialRforce(R,z,phi,t,2,4,potentialArgs);
}
double TriaxialJaffePotentialPlanarRforce(double R,double phi,double t,
					struct potentialArg * potentialArgs){
  return TwoPowerTriaxialPotentialRforce(R,0.,phi,t,2,4,potentialArgs);
}
double TriaxialJaffePotentialphiforce(double R,double z, double phi,
				    double t,
				    struct potentialArg * potentialArgs){
  return TwoPowerTriaxialPotentialphiforce(R,z,phi,t,2,4,potentialArgs);
}
double TriaxialJaffePotentialPlanarphiforce(double R,double phi,double t,
					  struct potentialArg * potentialArgs){
  return TwoPowerTriaxialPotentialphiforce(R,0.,phi,t,2,4,potentialArgs);
}
double TriaxialJaffePotentialzforce(double R,double z, double phi,
				  double t,
				  struct potentialArg * potentialArgs){
  return TwoPowerTriaxialPotentialzforce(R,z,phi,t,2,4,potentialArgs);
}

