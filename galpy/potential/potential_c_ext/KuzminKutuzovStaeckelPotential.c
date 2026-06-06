#include <math.h>
#include <galpy_potentials.h>
//KuzminKutuzovStaeckelPotential
//3  arguments: amp, ac, Delta
double KuzminKutuzovStaeckelPotentialEval(double R,double z, double phi,
				    double t,
				    struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  //Get args
  double amp   = *args;
  double ac    = *(args+1);
  double Delta = *(args+2);
  //Coordinate transformation
  double gamma = Delta*Delta / (1.-ac*ac);
  double alpha = gamma - Delta*Delta;
  double term  =     R*R + z*z - alpha - gamma;
  double discr = pow(R*R + z*z - Delta*Delta, 2.) + (4. * Delta*Delta * R*R);
  double l     = 0.5 * (term + sqrt(discr));
  double n     = 0.5 * (term - sqrt(discr));
  n= ((n > 0.) ? n: 0.);
  //Calculate potential
  return -amp /(sqrt(l) + sqrt(n));
}
double KuzminKutuzovStaeckelPotentialRforce(double R,double z, double phi,
				      double t,
				      struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  //Get args
  double amp   = *args;
  double ac    = *(args+1);
  double Delta = *(args+2);
  //Coordinate transformation
  double gamma = Delta*Delta / (1.-ac*ac);
  double alpha = gamma - Delta*Delta;
  double term  =     R*R + z*z - alpha - gamma;
  double discr = pow(R*R + z*z - Delta*Delta, 2.) + (4. * Delta*Delta * R*R);
  double l     = 0.5 * (term + sqrt(discr));
  double n     = 0.5 * (term - sqrt(discr));
  double dldR  = R * (1. + (R*R + z*z + Delta*Delta) / sqrt(discr));
  double dndR  = R * (1. - (R*R + z*z + Delta*Delta) / sqrt(discr));
  //Calculate Rforce
  double dVdl = 0.5/sqrt(l)/pow(sqrt(l)+sqrt(n),2.);
  double dVdn = 0.5/sqrt(n)/pow(sqrt(l)+sqrt(n),2.);
  return -amp * (dldR * dVdl + dndR * dVdn);
}
double KuzminKutuzovStaeckelPotentialPlanarRforce(double R,double phi,
					    double t,
					    struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  //Get args
  double amp   = *args;
  double ac    = *(args+1);
  double Delta = *(args+2);
  //Coordinate transformation (for z=0)
  double gamma = Delta*Delta / (1.-ac*ac);
  double alpha = gamma - Delta*Delta;
  double l     = R*R - alpha;
  double n     = -gamma;
  double dldR  = 2.*R;
  //Calculate Rforce
  double dVdl = 0.5/sqrt(l)/pow(sqrt(l)+sqrt(n),2.);
  return -amp * dldR * dVdl;
}
double KuzminKutuzovStaeckelPotentialzforce(double R,double z,double phi,
				      double t,
				      struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  //Get args
  double amp   = *args;
  double ac    = *(args+1);
  double Delta = *(args+2);
  //Coordinate transformation
  double gamma = Delta*Delta / (1.-ac*ac);
  double alpha = gamma - Delta*Delta;
  double term  =     R*R + z*z - alpha - gamma;
  double discr = pow(R*R + z*z - Delta*Delta, 2.) + (4. * Delta*Delta * R*R);
  double l     = 0.5 * (term + sqrt(discr));
  double n     = 0.5 * (term - sqrt(discr));
  double dldz  = z * (1. + (R*R + z*z - Delta*Delta) / sqrt(discr));
  double dndz  = z * (1. - (R*R + z*z - Delta*Delta) / sqrt(discr));
  //Calculate zforce
  double dVdl = 0.5/sqrt(l)/pow(sqrt(l)+sqrt(n),2.);
  double dVdn = 0.5/sqrt(n)/pow(sqrt(l)+sqrt(n),2.);
  return -amp * (dldz * dVdl + dndz * dVdn);
}
double KuzminKutuzovStaeckelPotentialPlanarR2deriv(double R,double phi,
					     double t,
					     struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  //Get args
  double amp   = *args;
  double ac    = *(args+1);
  double Delta = *(args+2);
  //Coordinate transformation (for z=0)
  double gamma  = Delta*Delta / (1.-ac*ac);
  double alpha  = gamma - Delta*Delta;
  double l      = R*R - alpha;
  double n      = -gamma;
  double dldR   = 2.*R;
  double d2ldR2 = 2.;
  //Calculate R2deriv
  double dVdl   = 0.5/sqrt(l)/pow(sqrt(l)+sqrt(n),2.);
  double d2Vdl2 = (-3.*sqrt(l)-sqrt(n)) / (4. * pow(l,1.5) * pow(sqrt(l)+sqrt(n),3.));
  return amp * (d2ldR2 * dVdl + dldR*dldR*d2Vdl2);
}
// Full-3D Hessian helpers for integrate_dxdv: port of the Python chain-rule
// expressions in KuzminKutuzovStaeckelPotential._R2deriv/_z2deriv/_Rzderiv.
double KuzminKutuzovStaeckelPotentialR2deriv(double R,double z, double phi,
					     double t,
					     struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  //Get args
  double amp   = *args;
  double ac    = *(args+1);
  double Delta = *(args+2);
  double D2    = Delta*Delta;
  //Coordinate transformation
  double gamma = D2 / (1.-ac*ac);
  double alpha = gamma - D2;
  double term  =     R*R + z*z - alpha - gamma;
  double discr = pow(R*R + z*z - D2, 2.) + (4. * D2 * R*R);
  double sdiscr= sqrt(discr);
  double l     = 0.5 * (term + sdiscr);
  double n     = 0.5 * (term - sdiscr);
  //Jacobian (d(l,n)/dR)
  double dldR  = R * (1. + (R*R + z*z + D2) / sdiscr);
  double dndR  = R * (1. - (R*R + z*z + D2) / sdiscr);
  //Hessian (d^2(l,n)/dR^2)
  double d2ldR2= 1. + (3.*R*R + z*z + D2) / sdiscr
                    - (2.*R*R * pow(R*R + z*z + D2,2.)) / pow(discr,1.5);
  double d2ndR2= 1. - (3.*R*R + z*z + D2) / sdiscr
                    + (2.*R*R * pow(R*R + z*z + D2,2.)) / pow(discr,1.5);
  //Potential derivatives w.r.t. (l,n)
  double srl   = sqrt(l);
  double srn   = sqrt(n);
  double sln   = srl + srn;
  double dVdl  = 0.5/srl/pow(sln,2.);
  double dVdn  = 0.5/srn/pow(sln,2.);
  double d2Vdl2= (-3.*srl-srn) / (4. * pow(l,1.5) * pow(sln,3.));
  double d2Vdn2= (-srl-3.*srn) / (4. * pow(n,1.5) * pow(sln,3.));
  double d2Vdln= -0.5 / (srl * srn * pow(sln,3.));
  return amp * ( d2ldR2 * dVdl + d2ndR2 * dVdn
		 + dldR*dldR * d2Vdl2 + dndR*dndR * d2Vdn2
		 + 2. * dldR * dndR * d2Vdln );
}
double KuzminKutuzovStaeckelPotentialz2deriv(double R,double z, double phi,
					     double t,
					     struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  //Get args
  double amp   = *args;
  double ac    = *(args+1);
  double Delta = *(args+2);
  double D2    = Delta*Delta;
  //Coordinate transformation
  double gamma = D2 / (1.-ac*ac);
  double alpha = gamma - D2;
  double term  =     R*R + z*z - alpha - gamma;
  double discr = pow(R*R + z*z - D2, 2.) + (4. * D2 * R*R);
  double sdiscr= sqrt(discr);
  double l     = 0.5 * (term + sdiscr);
  double n     = 0.5 * (term - sdiscr);
  //Jacobian (d(l,n)/dz)
  double dldz  = z * (1. + (R*R + z*z - D2) / sdiscr);
  double dndz  = z * (1. - (R*R + z*z - D2) / sdiscr);
  //Hessian (d^2(l,n)/dz^2)
  double d2ldz2= 1. + (R*R + 3.*z*z - D2) / sdiscr
                    - (2.*z*z * pow(R*R + z*z - D2,2.)) / pow(discr,1.5);
  double d2ndz2= 1. - (R*R + 3.*z*z - D2) / sdiscr
                    + (2.*z*z * pow(R*R + z*z - D2,2.)) / pow(discr,1.5);
  //Potential derivatives w.r.t. (l,n)
  double srl   = sqrt(l);
  double srn   = sqrt(n);
  double sln   = srl + srn;
  double dVdl  = 0.5/srl/pow(sln,2.);
  double dVdn  = 0.5/srn/pow(sln,2.);
  double d2Vdl2= (-3.*srl-srn) / (4. * pow(l,1.5) * pow(sln,3.));
  double d2Vdn2= (-srl-3.*srn) / (4. * pow(n,1.5) * pow(sln,3.));
  double d2Vdln= -0.5 / (srl * srn * pow(sln,3.));
  return amp * ( d2ldz2 * dVdl + d2ndz2 * dVdn
		 + dldz*dldz * d2Vdl2 + dndz*dndz * d2Vdn2
		 + 2. * dldz * dndz * d2Vdln );
}
double KuzminKutuzovStaeckelPotentialRzderiv(double R,double z, double phi,
					     double t,
					     struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  //Get args
  double amp   = *args;
  double ac    = *(args+1);
  double Delta = *(args+2);
  double D2    = Delta*Delta;
  double D4    = D2*D2;
  //Coordinate transformation
  double gamma = D2 / (1.-ac*ac);
  double alpha = gamma - D2;
  double term  =     R*R + z*z - alpha - gamma;
  double discr = pow(R*R + z*z - D2, 2.) + (4. * D2 * R*R);
  double sdiscr= sqrt(discr);
  double l     = 0.5 * (term + sdiscr);
  double n     = 0.5 * (term - sdiscr);
  //Jacobian (d(l,n)/dR and d(l,n)/dz)
  double dldR  = R * (1. + (R*R + z*z + D2) / sdiscr);
  double dndR  = R * (1. - (R*R + z*z + D2) / sdiscr);
  double dldz  = z * (1. + (R*R + z*z - D2) / sdiscr);
  double dndz  = z * (1. - (R*R + z*z - D2) / sdiscr);
  //Mixed Hessian (d^2(l,n)/dR/dz)
  double rz2   = (R*R + z*z);
  double d2ldRdz= 2.*R*z / sdiscr * (1. - (rz2*rz2 - D4) / discr);
  double d2ndRdz= 2.*R*z / sdiscr * (-1. + (rz2*rz2 - D4) / discr);
  //Potential derivatives w.r.t. (l,n)
  double srl   = sqrt(l);
  double srn   = sqrt(n);
  double sln   = srl + srn;
  double dVdl  = 0.5/srl/pow(sln,2.);
  double dVdn  = 0.5/srn/pow(sln,2.);
  double d2Vdl2= (-3.*srl-srn) / (4. * pow(l,1.5) * pow(sln,3.));
  double d2Vdn2= (-srl-3.*srn) / (4. * pow(n,1.5) * pow(sln,3.));
  double d2Vdln= -0.5 / (srl * srn * pow(sln,3.));
  return amp * ( d2ldRdz * dVdl + d2ndRdz * dVdn
		 + dldR * dldz * d2Vdl2 + dndR * dndz * d2Vdn2
		 + (dldR * dndz + dldz * dndR) * d2Vdln );
}
