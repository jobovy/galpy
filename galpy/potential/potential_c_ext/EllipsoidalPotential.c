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
  double * ellipargs= args + 8 + (int) *(args+7); // *(args+7) = num. arguments psi
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
					   args+8);
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
  double * ellipargs= args + 8 + (int) *(args+7); // *(args+7) = num. arguments dens
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
				 + z * z / ( c2 + t ) ),args+8);
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
  double * ellipargs= args + 8 + (int) *(args+7); // *(args+7) = num. arguments psi
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
				     args+8);
}

// Helper function to compute second derivatives in xyz coordinates
double EllipsoidalPotential_2ndderiv_xyz(double (*dens)(double m, double * args),
					 double (*densDeriv)(double m, double * args),
					 double x, double y, double z,
					 int i, int j,
					 double b2, double c2,
					 int glorder, double * glx, double * glw,
					 double * args) {
  int ii;
  double s, t, m;
  double integrand;
  double result = 0.;
  double xi, xj;  // Components corresponding to i and j
  double ti, tj;  // Denominators (1+t, b2+t, or c2+t) for i and j

  for (ii = 0; ii < glorder; ii++) {
    s = *(glx + ii);
    t = 1. / s / s - 1.;

    // Calculate m
    m = sqrt(x * x / (1. + t) + y * y / (b2 + t) + z * z / (c2 + t));

    // Determine xi and ti based on index i
    if (i == 0) {
      xi = x;
      ti = 1. + t;
    } else if (i == 1) {
      xi = y;
      ti = b2 + t;
    } else {  // i == 2
      xi = z;
      ti = c2 + t;
    }

    // Determine xj and tj based on index j
    if (j == 0) {
      xj = x;
      tj = 1. + t;
    } else if (j == 1) {
      xj = y;
      tj = b2 + t;
    } else {  // j == 2
      xj = z;
      tj = c2 + t;
    }

    // Calculate the integrand
    // Note: glw already includes the -4*pi*b*c / sqrt(...) factor from Python glue code
    // The negative sign is for forces; for second derivatives we need positive, so we negate
    // We just compute the core integral: densDeriv(m) * xi/ti * xj/tj / m + dens(m) * delta_ij / ti
    if (m > 0.) {
      integrand = densDeriv(m, args) * (xi / ti) * (xj / tj) / m;
    } else {
      integrand = 0.;
    }
    if (i == j) {
      integrand += dens(m, args) / ti;
    }

    result += *(glw + ii) * integrand;
  }

  // Negate because glw has negative sign for forces, but we need positive for second derivatives
  return -result;
}

double EllipsoidalPotentialPlanarR2deriv(double R, double phi, double t,
					 struct potentialArg * potentialArgs) {
  double * args = potentialArgs->args;
  double amp = *args;
  double * ellipargs = args + 8 + (int) *(args+7);
  double b2 = *ellipargs++;
  double c2 = *ellipargs++;
  bool aligned = (bool) *ellipargs++;
  double * rot = ellipargs;
  ellipargs += 9;
  int glorder = (int) *ellipargs++;
  double * glx = ellipargs;
  double * glw = ellipargs + glorder;

  // Only support aligned potentials
  if (!aligned) {
    return 0.;
  }

  // Convert to Cartesian (z=0 for planar)
  double x, y, z = 0.;
  cyl_to_rect(R, phi, &x, &y);

  // Get second derivatives in xyz coordinates
  double phixx = EllipsoidalPotential_2ndderiv_xyz(potentialArgs->mdens,
						   potentialArgs->mdensDeriv,
						   x, y, z, 0, 0, b2, c2,
						   glorder, glx, glw, args+8);
  double phixy = EllipsoidalPotential_2ndderiv_xyz(potentialArgs->mdens,
						   potentialArgs->mdensDeriv,
						   x, y, z, 0, 1, b2, c2,
						   glorder, glx, glw, args+8);
  double phiyy = EllipsoidalPotential_2ndderiv_xyz(potentialArgs->mdens,
						   potentialArgs->mdensDeriv,
						   x, y, z, 1, 1, b2, c2,
						   glorder, glx, glw, args+8);

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
  double * ellipargs = args + 8 + (int) *(args+7);
  double b2 = *ellipargs++;
  double c2 = *ellipargs++;
  bool aligned = (bool) *ellipargs++;
  double * rot = ellipargs;
  ellipargs += 9;
  int glorder = (int) *ellipargs++;
  double * glx = ellipargs;
  double * glw = ellipargs + glorder;

  // Only support aligned potentials
  if (!aligned) {
    return 0.;
  }

  // Convert to Cartesian (z=0 for planar)
  double x, y, z = 0.;
  cyl_to_rect(R, phi, &x, &y);

  // Get forces in xyz coordinates (without amp or 4*pi*b*c factor)
  double Fx, Fy, Fz;
  EllipsoidalPotentialxyzforces_xyz(potentialArgs->mdens, x, y, z, &Fx, &Fy, &Fz, args);

  // Get second derivatives in xyz coordinates
  double phixx = EllipsoidalPotential_2ndderiv_xyz(potentialArgs->mdens,
						   potentialArgs->mdensDeriv,
						   x, y, z, 0, 0, b2, c2,
						   glorder, glx, glw, args+8);
  double phixy = EllipsoidalPotential_2ndderiv_xyz(potentialArgs->mdens,
						   potentialArgs->mdensDeriv,
						   x, y, z, 0, 1, b2, c2,
						   glorder, glx, glw, args+8);
  double phiyy = EllipsoidalPotential_2ndderiv_xyz(potentialArgs->mdens,
						   potentialArgs->mdensDeriv,
						   x, y, z, 1, 1, b2, c2,
						   glorder, glx, glw, args+8);

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
  double * ellipargs = args + 8 + (int) *(args+7);
  double b2 = *ellipargs++;
  double c2 = *ellipargs++;
  bool aligned = (bool) *ellipargs++;
  double * rot = ellipargs;
  ellipargs += 9;
  int glorder = (int) *ellipargs++;
  double * glx = ellipargs;
  double * glw = ellipargs + glorder;

  // Only support aligned potentials
  if (!aligned) {
    return 0.;
  }

  // Convert to Cartesian (z=0 for planar)
  double x, y, z = 0.;
  cyl_to_rect(R, phi, &x, &y);

  // Get forces in xyz coordinates (without amp or 4*pi*b*c factor)
  double Fx, Fy, Fz;
  EllipsoidalPotentialxyzforces_xyz(potentialArgs->mdens, x, y, z, &Fx, &Fy, &Fz, args);

  // Get second derivatives in xyz coordinates
  double phixx = EllipsoidalPotential_2ndderiv_xyz(potentialArgs->mdens,
						   potentialArgs->mdensDeriv,
						   x, y, z, 0, 0, b2, c2,
						   glorder, glx, glw, args+8);
  double phixy = EllipsoidalPotential_2ndderiv_xyz(potentialArgs->mdens,
						   potentialArgs->mdensDeriv,
						   x, y, z, 0, 1, b2, c2,
						   glorder, glx, glw, args+8);
  double phiyy = EllipsoidalPotential_2ndderiv_xyz(potentialArgs->mdens,
						   potentialArgs->mdensDeriv,
						   x, y, z, 1, 1, b2, c2,
						   glorder, glx, glw, args+8);

  // Transform to cylindrical: d^2phi/dRdphi
  double cosphi = cos(phi);
  double sinphi = sin(phi);
  double cos2phi = cos(2. * phi);

  return amp * (R * (cosphi * sinphi * (phiyy - phixx) + cos2phi * phixy) +
		sinphi * Fx - cosphi * Fy);
}
