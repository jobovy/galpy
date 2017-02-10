#include <stdbool.h>
#include <math.h>
#include <galpy_potentials.h>
//General routines for TwoPowerTriaxialPotentials, specific potentials below
//TriaxialNFWPotential
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
static inline double dens(double m, double alpha, double beta){
  if ( alpha == 1 && beta == 3) // NFW case
    return 1. / m / ( 1. + m ) / ( 1. + m );
  else if ( alpha == 1 && beta == 4) // Hernquist case
    return 1. / m / ( 1. + m ) / ( 1. + m ) / ( 1. + m );
  else if ( alpha == 2 && beta == 4) // Jaffe case
    return 1. / m / m / ( 1. + m ) / ( 1. + m );
  else // not currently used
    //LCOV_EXCL_START
    return pow ( m, -alpha) * pow ( 1. + m , alpha - beta);
    //LCOV_EXCL_STOP
}
void TwoPowerTriaxialPotentialxyzforces_xyz(double x,double y, double z,
					    double * Fx, double * Fy, 
					    double * Fz,double *args,
					    double a,
					    double alpha, double beta,
					    double b2, double c2,
					    bool aligned, double * rot, 
					    int glorder,
					    double * glx, double * glw){
  int ii;
  double t;
  double td;
  *args= x;
  *(args + 1)= y;
  *(args + 2)= z;
  if ( !aligned ) 
    rotate(&x,&y,&z,rot);
  *Fx= 0.;
  *Fy= 0.;
  *Fz= 0.;
  for (ii=0; ii < glorder; ii++) {
    t= 1. / *(glx+ii) / *(glx+ii) - 1.;
    td= *(glw+ii) * dens( sqrt ( x * x / ( 1. + t )	+ y * y / ( b2 + t ) \
				 + z * z / ( c2 + t ) ) / a,alpha,beta);
    *Fx+= td * x / ( 1. + t );
    *Fy+= td * y / ( b2 + t );
    *Fz+= td * z / ( c2 + t );
  }
  *(args + 3)= *Fx;
  *(args + 4)= *Fy;
  *(args + 5)= *Fz;
}
double TwoPowerTriaxialPotentialRforce(double R,double z, double phi,
				       double t,
				       double alpha, double beta,
				       struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  //Get args
  double amp= *args++;
  double a= *args++;
  double b2= *args++;
  double c2= *args++;
  bool aligned= (bool) *args++;
  double * rot= args;
  args+= 9;
  int glorder= (int) *args++;
  double * glx= args;
  double * glw= args + glorder;
  args+= 2 * glorder;
  double cached_x= *args;
  double cached_y= *(args + 1);
  double cached_z= *(args + 2);
  //Calculate potential
  double x, y;
  double Fx, Fy, Fz;
  cyl_to_rect(R,phi,&x,&y);
  if ( x == cached_x && y == cached_y && z == cached_z ){
    //LCOV_EXCL_START
    Fx= *(args + 3);
    Fy= *(args + 4);
    Fz= *(args + 5);
    //LCOV_EXCL_STOP
  }
  else 
    TwoPowerTriaxialPotentialxyzforces_xyz(x,y,z,&Fx,&Fy,&Fz,args,
					   a,alpha,beta,b2,c2,
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
  double b2= *args++;
  double c2= *args++;
  bool aligned= (bool) *args++;
  double * rot= args;
  args+= 9;
  int glorder= (int) *args++;
  double * glx= args;
  double * glw= args + glorder;
  args+= 2 * glorder;
  double cached_x= *args;
  double cached_y= *(args + 1);
  double cached_z= *(args + 2);
  //Calculate potential
  double x, y;
  double Fx, Fy, Fz;
  cyl_to_rect(R,phi,&x,&y);
  if ( x == cached_x && y == cached_y && z == cached_z ){
    Fx= *(args + 3);
    Fy= *(args + 4);
    Fz= *(args + 5);
  }
  else 
    //LCOV_EXCL_START
    TwoPowerTriaxialPotentialxyzforces_xyz(x,y,z,&Fx,&Fy,&Fz,args,
					   a,alpha,beta,b2,c2,
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
  double b2= *args++;
  double c2= *args++;
  bool aligned= (bool) *args++;
  double * rot= args;
  args+= 9;
  int glorder= (int) *args++;
  double * glx= args;
  double * glw= args + glorder;
  args+= 2 * glorder;
  double cached_x= *args;
  double cached_y= *(args + 1);
  double cached_z= *(args + 2);
  //Calculate potential
  double x, y;
  double Fx, Fy, Fz;
  cyl_to_rect(R,phi,&x,&y);
  if ( x == cached_x && y == cached_y && z == cached_z ){
    Fx= *(args + + 3);
    Fy= *(args + + 4);
    Fz= *(args + + 5);
  }
  else 
    //LCOV_EXCL_START
    TwoPowerTriaxialPotentialxyzforces_xyz(x,y,z,&Fx,&Fy,&Fz,args,
					   a,alpha,beta,b2,c2,
					   aligned,rot,glorder,glx,glw);
    //LCOV_EXCL_STOP
  if ( !aligned )
    rotate_force(&Fx,&Fy,&Fz,rot);
  return amp * Fz;
}
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
// Implement the potentials separately
//NFW
static inline double TriaxialNFWPotential_psi(double x){
  return 1. / ( 1. + x );
}
double TriaxialNFWPotentialpotential_xyz_integrand(double s,
						   double x,double y,
						   double z,double a,
						   double b2,double c2){
  double t= 1. / s / s - 1.;
  return TriaxialNFWPotential_psi( sqrt ( x * x / ( 1. + t ) \
					  + y * y / ( b2 + t )	\
					  + z * z / ( c2 + t ) ) / a );
}
double TriaxialNFWPotentialEval(double R,double z, double phi,
				double t,
				struct potentialArg * potentialArgs){
  int ii;
  double * args= potentialArgs->args;
  //Get args
  double amp= *args++;
  double a= *args++;
  double b2= *args++;
  double c2= *args++;
  bool aligned= (bool) *args++;
  double * rot= args;
  args+= 9;
  int glorder= (int) *args++;
  double * glx= args;
  double * glw= args + glorder;
  //Calculate potential
  double x, y;
  double out= 0.;
  cyl_to_rect(R,phi,&x,&y);
  if ( !aligned ) 
    rotate(&x,&y,&z,rot);
  for (ii=0; ii < glorder; ii++)
    out+= *(glw+ii) * a * a						\
      * TriaxialNFWPotentialpotential_xyz_integrand(*(glx+ii),x,y,z,
						    a,b2,c2);
  return amp * out;
}
//Hernquist
static inline double TriaxialHernquistPotential_psi(double x){
  return 0.5 / ( 1. + x ) / ( 1. + x );
}
double TriaxialHernquistPotentialpotential_xyz_integrand(double s,
						   double x,double y,
						   double z,double a,
						   double b2,double c2){
  double t= 1. / s / s - 1.;
  return TriaxialHernquistPotential_psi( sqrt ( x * x / ( 1. + t ) \
						+ y * y / ( b2 + t )	\
						+ z * z / ( c2 + t ) ) / a );
}
double TriaxialHernquistPotentialEval(double R,double z, double phi,
				double t,
				struct potentialArg * potentialArgs){
  int ii;
  double * args= potentialArgs->args;
  //Get args
  double amp= *args++;
  double a= *args++;
  double b2= *args++;
  double c2= *args++;
  bool aligned= (bool) *args++;
  double * rot= args;
  args+= 9;
  int glorder= (int) *args++;
  double * glx= args;
  double * glw= args + glorder;
  //Calculate potential
  double x, y;
  double out= 0.;
  cyl_to_rect(R,phi,&x,&y);
  if ( !aligned ) 
    rotate(&x,&y,&z,rot);
  for (ii=0; ii < glorder; ii++)
    out+= *(glw+ii) * a * a						\
      * TriaxialHernquistPotentialpotential_xyz_integrand(*(glx+ii),x,y,z,
							  a,b2,c2);
  return amp * out;
}
//Jaffe
static inline double TriaxialJaffePotential_psi(double x){
  return - 1. / ( 1. + x ) - log ( x / ( 1. + x ) ) ;
}
double TriaxialJaffePotentialpotential_xyz_integrand(double s,
						   double x,double y,
						   double z,double a,
						   double b2,double c2){
  double t= 1. / s / s - 1.;
  return TriaxialJaffePotential_psi( sqrt ( x * x / ( 1. + t )		\
					    + y * y / ( b2 + t )	\
					    + z * z / ( c2 + t ) ) / a );
}
double TriaxialJaffePotentialEval(double R,double z, double phi,
				double t,
				struct potentialArg * potentialArgs){
  int ii;
  double * args= potentialArgs->args;
  //Get args
  double amp= *args++;
  double a= *args++;
  double b2= *args++;
  double c2= *args++;
  bool aligned= (bool) *args++;
  double * rot= args;
  args+= 9;
  int glorder= (int) *args++;
  double * glx= args;
  double * glw= args + glorder;
  //Calculate potential
  double x, y;
  double out= 0.;
  cyl_to_rect(R,phi,&x,&y);
  if ( !aligned ) 
    rotate(&x,&y,&z,rot);
  for (ii=0; ii < glorder; ii++)
    out+= *(glw+ii) * a * a						\
      * TriaxialJaffePotentialpotential_xyz_integrand(*(glx+ii),x,y,z,
						      a,b2,c2);
  return amp * out;
}

