#include <stdbool.h>
#include <math.h>
#include <bovy_coords.h>
#include <galpy_potentials.h>
//General routines for EllipsoidalPotentials
double EllipsoidalPotentialEval(double R,double z, double phi,
				double t,
				struct potentialArg * potentialArgs){
  int ii;
  double s;
  //Get args
  double * args= potentialArgs->args;
  double amp= *args;
  double * ellipargs= args + 14 + (int) *(args+13); // *(args+13) = num. arguments psi
  double b2= *ellipargs++;
  double c2= *ellipargs++;
  bool aligned= (bool) *ellipargs++;
  double * rot= ellipargs;
  ellipargs+= 9;
  int glorder= (int) *ellipargs++;
  double * glx= ellipargs;
  double * glw= ellipargs + glorder;
  //Calculate potential
  double x, y;
  double out= 0.;
  cyl_to_rect(R,phi,&x,&y);
  if ( !aligned )
    rotate(&x,&y,&z,rot);
  for (ii=0; ii < glorder; ii++) {
    s= 1. / *(glx+ii) / *(glx+ii) - 1.;
    out+= *(glw+ii) * potentialArgs->psi ( sqrt (  x * x / ( 1. + s )
	 					 + y * y / ( b2 + s )
		 				 + z * z / ( c2 + s ) ),
					   args+14);
  }
  return -0.5 * amp * out;
}
void EllipsoidalPotentialxyzforces_xyz(double (*dens)(double m,
						      double * args),
				       double x,double y, double z,
				       double * Fx, double * Fy,
				       double * Fz,double * args){
  int ii;
  double t;
  double td;
  //Get args
  double * ellipargs= args + 14 + (int) *(args+13); // *(args+13) = num. arguments dens
  double b2= *ellipargs++;
  double c2= *ellipargs++;
  bool aligned= (bool) *ellipargs++;
  double * rot= ellipargs;
  ellipargs+= 9;
  int glorder= (int) *ellipargs++;
  double * glx= ellipargs;
  double * glw= ellipargs + glorder;
  //Setup caching
  *(args + 1)= x;
  *(args + 2)= y;
  *(args + 3)= z;
  if ( !aligned )
    rotate(&x,&y,&z,rot);
  *Fx= 0.;
  *Fy= 0.;
  *Fz= 0.;
  for (ii=0; ii < glorder; ii++) {
    t= 1. / *(glx+ii) / *(glx+ii) - 1.;
    td= *(glw+ii) * dens( sqrt ( x * x / ( 1. + t )	+ y * y / ( b2 + t ) \
				 + z * z / ( c2 + t ) ),args+14);
    *Fx+= td * x / ( 1. + t );
    *Fy+= td * y / ( b2 + t );
    *Fz+= td * z / ( c2 + t );
  }
  if ( !aligned )
    rotate_force(Fx,Fy,Fz,rot);
  *(args + 4)= *Fx;
  *(args + 5)= *Fy;
  *(args + 6)= *Fz;
}
double EllipsoidalPotentialRforce(double R,double z, double phi,
				  double t,
				  struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  double amp= *args;
  // Get caching args: amp = 0, x,y,z,Fx,Fy,Fz
  double cached_x= *(args + 1);
  double cached_y= *(args + 2);
  double cached_z= *(args + 3);
  //Calculate potential
  double x, y;
  double Fx, Fy, Fz;
  cyl_to_rect(R,phi,&x,&y);
  if ( x == cached_x && y == cached_y && z == cached_z ){
    // LCOV_EXCL_START
    Fx= *(args + 4);
    Fy= *(args + 5);
    Fz= *(args + 6);
    // LCOV_EXCL_STOP
  }
  else
    EllipsoidalPotentialxyzforces_xyz(potentialArgs->mdens,
				      x,y,z,&Fx,&Fy,&Fz,args);
  return amp * ( cos ( phi ) * Fx + sin( phi ) * Fy );
}
double EllipsoidalPotentialphitorque(double R,double z, double phi,
				    double t,
				    struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  double amp= *args;
  // Get caching args: amp = 0, x,y,z,Fx,Fy,Fz
  double cached_x= *(args + 1);
  double cached_y= *(args + 2);
  double cached_z= *(args + 3);
  //Calculate potential
  double x, y;
  double Fx, Fy, Fz;
  cyl_to_rect(R,phi,&x,&y);
  if ( x == cached_x && y == cached_y && z == cached_z ){
    Fx= *(args + 4);
    Fy= *(args + 5);
    Fz= *(args + 6);
  }
  else
    // LCOV_EXCL_START
    EllipsoidalPotentialxyzforces_xyz(potentialArgs->mdens,
				      x,y,z,&Fx,&Fy,&Fz,args);
    // LCOV_EXCL_STOP
  return amp * R * ( -sin ( phi ) * Fx + cos( phi ) * Fy );
}
double EllipsoidalPotentialzforce(double R,double z, double phi,
				  double t,
				  struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  double amp= *args;
  // Get caching args: amp = 0, x,y,z,Fx,Fy,Fz
  double cached_x= *(args + 1);
  double cached_y= *(args + 2);
  double cached_z= *(args + 3);
  //Calculate potential
  double x, y;
  double Fx, Fy, Fz;
  cyl_to_rect(R,phi,&x,&y);
  if ( x == cached_x && y == cached_y && z == cached_z ){
    Fx= *(args + + 4);
    Fy= *(args + + 5);
    Fz= *(args + + 6);
  }
  else
    // LCOV_EXCL_START
    EllipsoidalPotentialxyzforces_xyz(potentialArgs->mdens,
				      x,y,z,&Fx,&Fy,&Fz,args);
    // LCOV_EXCL_STOP
  return amp * Fz;
}

double EllipsoidalPotentialPlanarRforce(double R,double phi,double t,
					struct potentialArg * potentialArgs){
  return EllipsoidalPotentialRforce(R,0.,phi,t,potentialArgs);
}
double EllipsoidalPotentialPlanarphitorque(double R,double phi,double t,
					  struct potentialArg * potentialArgs){
  return EllipsoidalPotentialphitorque(R,0.,phi,t,potentialArgs);
}
double EllipsoidalPotentialDens(double R,double z, double phi,
				double t,
				struct potentialArg * potentialArgs){
  //Get args
  double * args= potentialArgs->args;
  double amp= *args;
  double * ellipargs= args + 14 + (int) *(args+13); // *(args+13) = num. arguments psi
  double b2= *ellipargs++;
  double c2= *ellipargs++;
  bool aligned= (bool) *ellipargs++;
  double * rot= ellipargs;
  ellipargs+= 9;
  //Calculate density
  double x, y;
  cyl_to_rect(R,phi,&x,&y);
  if ( !aligned )
    rotate(&x,&y,&z,rot);
  return amp * potentialArgs->mdens ( sqrt (x * x + y * y / b2 + z * z / c2 ),
				     args+14);
}

// Helper function to compute all second derivatives in xyz coordinates and cache them
void EllipsoidalPotential_2ndderiv_xyz_all(double (*dens)(double m, double * args),
					   double (*densDeriv)(double m, double * args),
					   double x, double y, double z,
					   double b2, double c2,
					   int glorder, double * glx, double * glw,
					   double * args,
					   double * phixx, double * phixy, double * phixz,
					   double * phiyy, double * phiyz, double * phizz) {
  int ii;
  double s, t, m;
  double integrand_xx, integrand_xy, integrand_xz;
  double integrand_yy, integrand_yz, integrand_zz;
  double result_xx = 0., result_xy = 0., result_xz = 0.;
  double result_yy = 0., result_yz = 0., result_zz = 0.;

  // Pre-compute values outside the loop
  double x_val = x;
  double y_val = y;
  double z_val = z;

  for (ii = 0; ii < glorder; ii++) {
    s = *(glx + ii);
    t = 1. / s / s - 1.;

    // Calculate m
    m = sqrt(x_val * x_val / (1. + t) + y_val * y_val / (b2 + t) + z_val * z_val / (c2 + t));

    // Pre-compute denominators
    double t1 = 1. + t;
    double t2 = b2 + t;
    double t3 = c2 + t;

    // Calculate all 6 unique second derivatives
    // Note: glw already includes the -4*pi*b*c / sqrt(...) factor from Python glue code
    // The negative sign is for forces; for second derivatives we need positive, so we negate

    // d^2phi/dx^2
    integrand_xx = densDeriv(m, args+14) * (x_val / t1) * (x_val / t1) / m + dens(m, args+14) / t1;

    // d^2phi/dxdy
    integrand_xy = densDeriv(m, args+14) * (x_val / t1) * (y_val / t2) / m;

    // d^2phi/dxdz
    integrand_xz = densDeriv(m, args+14) * (x_val / t1) * (z_val / t3) / m;

    // d^2phi/dy^2
    integrand_yy = densDeriv(m, args+14) * (y_val / t2) * (y_val / t2) / m + dens(m, args+14) / t2;

    // d^2phi/dydz
    integrand_yz = densDeriv(m, args+14) * (y_val / t2) * (z_val / t3) / m;

    // d^2phi/dz^2
    integrand_zz = densDeriv(m, args+14) * (z_val / t3) * (z_val / t3) / m + dens(m, args+14) / t3;

    result_xx += *(glw + ii) * integrand_xx;
    result_xy += *(glw + ii) * integrand_xy;
    result_xz += *(glw + ii) * integrand_xz;
    result_yy += *(glw + ii) * integrand_yy;
    result_yz += *(glw + ii) * integrand_yz;
    result_zz += *(glw + ii) * integrand_zz;
  }

  // Negate because glw has negative sign for forces, but we need positive for second derivatives
  *phixx = -result_xx;
  *phixy = -result_xy;
  *phixz = -result_xz;
  *phiyy = -result_yy;
  *phiyz = -result_yz;
  *phizz = -result_zz;

  // Cache the position and results
  *(args + 1) = x;
  *(args + 2) = y;
  *(args + 3) = z;
  *(args + 7) = *phixx;
  *(args + 8) = *phixy;
  *(args + 9) = *phixz;
  *(args + 10) = *phiyy;
  *(args + 11) = *phiyz;
  *(args + 12) = *phizz;
}


double EllipsoidalPotentialPlanarR2deriv(double R, double phi, double t,
					 struct potentialArg * potentialArgs) {
  double * args = potentialArgs->args;
  double amp = *args;
  // Get caching args: amp = 0, x,y,z,Fx,Fy,Fz,phixx,phixy,phixz,phiyy,phiyz,phizz
  double cached_x = *(args + 1);
  double cached_y = *(args + 2);
  double cached_z = *(args + 3);
  double * ellipargs = args + 14 + (int) *(args+13);
  double b2 = *ellipargs++;
  double c2 = *ellipargs++;
  bool aligned = (bool) *ellipargs++;
  double * rot = ellipargs;
  ellipargs += 9;
  int glorder = (int) *ellipargs++;
  double * glx = ellipargs;
  double * glw = ellipargs + glorder;

  // Convert to Cartesian (z=0 for planar)
  double x, y, z = 0.;
  cyl_to_rect(R, phi, &x, &y);

  // Get second derivatives in xyz coordinates (with caching)
  double phixx, phixy, phixz, phiyy, phiyz, phizz;
  if (x == cached_x && y == cached_y && z == cached_z) {
    // LCOV_EXCL_START
    phixx = *(args + 7);
    phixy = *(args + 8);
    phixz = *(args + 9);
    phiyy = *(args + 10);
    phiyz = *(args + 11);
    phizz = *(args + 12);
    // LCOV_EXCL_STOP
  } else {
    EllipsoidalPotential_2ndderiv_xyz_all(potentialArgs->mdens,
					  potentialArgs->mdensDeriv,
					  x, y, z, b2, c2,
					  glorder, glx, glw, args,
					  &phixx, &phixy, &phixz,
					  &phiyy, &phiyz, &phizz);
  }

  // Transform to cylindrical: d^2phi/dR^2
  double cosphi = cos(phi);
  double sinphi = sin(phi);

  return amp * (cosphi * cosphi * phixx + sinphi * sinphi * phiyy +
		2. * cosphi * sinphi * phixy);
}

double EllipsoidalPotentialPlanarphi2deriv(double R, double phi, double t,
					   struct potentialArg * potentialArgs) {
  double * args = potentialArgs->args;
  double amp = *args;
  // Get caching args: amp = 0, x,y,z,Fx,Fy,Fz,phixx,phixy,phixz,phiyy,phiyz,phizz
  double cached_x = *(args + 1);
  double cached_y = *(args + 2);
  double cached_z = *(args + 3);
  double * ellipargs = args + 14 + (int) *(args+13);
  double b2 = *ellipargs++;
  double c2 = *ellipargs++;
  bool aligned = (bool) *ellipargs++;
  double * rot = ellipargs;
  ellipargs += 9;
  int glorder = (int) *ellipargs++;
  double * glx = ellipargs;
  double * glw = ellipargs + glorder;

  // Convert to Cartesian (z=0 for planar)
  double x, y, z = 0.;
  cyl_to_rect(R, phi, &x, &y);

  // Get forces in xyz coordinates (with caching)
  double Fx, Fy, Fz;
  if (x == cached_x && y == cached_y && z == cached_z) {
    Fx = *(args + 4);
    Fy = *(args + 5);
    Fz = *(args + 6);
  } else {
    // LCOV_EXCL_START
    EllipsoidalPotentialxyzforces_xyz(potentialArgs->mdens, x, y, z, &Fx, &Fy, &Fz, args);
    // LCOV_EXCL_STOP
  }

  // Get second derivatives in xyz coordinates (with caching)
  double phixx, phixy, phixz, phiyy, phiyz, phizz;
  if (x == cached_x && y == cached_y && z == cached_z) {
    phixx = *(args + 7);
    phixy = *(args + 8);
    phixz = *(args + 9);
    phiyy = *(args + 10);
    phiyz = *(args + 11);
    phizz = *(args + 12);
  } else {
    // LCOV_EXCL_START
    EllipsoidalPotential_2ndderiv_xyz_all(potentialArgs->mdens,
					  potentialArgs->mdensDeriv,
					  x, y, z, b2, c2,
					  glorder, glx, glw, args,
					  &phixx, &phixy, &phixz,
					  &phiyy, &phiyz, &phizz);
    // LCOV_EXCL_STOP
  }

  // Transform to cylindrical: d^2phi/dphi^2
  double cosphi = cos(phi);
  double sinphi = sin(phi);

  return amp * (R * R * (sinphi * sinphi * phixx + cosphi * cosphi * phiyy -
			 2. * cosphi * sinphi * phixy) +
		R * (cosphi * Fx + sinphi * Fy));
}

double EllipsoidalPotentialPlanarRphideriv(double R, double phi, double t,
					   struct potentialArg * potentialArgs) {
  double * args = potentialArgs->args;
  double amp = *args;
  // Get caching args: amp = 0, x,y,z,Fx,Fy,Fz,phixx,phixy,phixz,phiyy,phiyz,phizz
  double cached_x = *(args + 1);
  double cached_y = *(args + 2);
  double cached_z = *(args + 3);
  double * ellipargs = args + 14 + (int) *(args+13);
  double b2 = *ellipargs++;
  double c2 = *ellipargs++;
  bool aligned = (bool) *ellipargs++;
  double * rot = ellipargs;
  ellipargs += 9;
  int glorder = (int) *ellipargs++;
  double * glx = ellipargs;
  double * glw = ellipargs + glorder;

  // Convert to Cartesian (z=0 for planar)
  double x, y, z = 0.;
  cyl_to_rect(R, phi, &x, &y);

  // Get forces in xyz coordinates (with caching)
  double Fx, Fy, Fz;
  if (x == cached_x && y == cached_y && z == cached_z) {
    Fx = *(args + 4);
    Fy = *(args + 5);
    Fz = *(args + 6);
  } else {
    // LCOV_EXCL_START
    EllipsoidalPotentialxyzforces_xyz(potentialArgs->mdens, x, y, z, &Fx, &Fy, &Fz, args);
    // LCOV_EXCL_STOP
  }

  // Get second derivatives in xyz coordinates (with caching)
  double phixx, phixy, phixz, phiyy, phiyz, phizz;
  if (x == cached_x && y == cached_y && z == cached_z) {
    phixx = *(args + 7);
    phixy = *(args + 8);
    phixz = *(args + 9);
    phiyy = *(args + 10);
    phiyz = *(args + 11);
    phizz = *(args + 12);
  } else {
    // LCOV_EXCL_START
    EllipsoidalPotential_2ndderiv_xyz_all(potentialArgs->mdens,
					  potentialArgs->mdensDeriv,
					  x, y, z, b2, c2,
					  glorder, glx, glw, args,
					  &phixx, &phixy, &phixz,
					  &phiyy, &phiyz, &phizz);
    // LCOV_EXCL_STOP
  }

  // Transform to cylindrical: d^2phi/dRdphi
  double cosphi = cos(phi);
  double sinphi = sin(phi);
  double cos2phi = cos(2. * phi);

  return amp * (R * (cosphi * sinphi * (phiyy - phixx) + cos2phi * phixy) +
		sinphi * Fx - cosphi * Fy);
}
