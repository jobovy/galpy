/*
  Wrappers around the C integration code for Full Orbits
*/
#ifdef _WIN32
#include <Python.h>
#endif
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include <bovy_symplecticode.h>
#include <bovy_rk.h>
//Potentials
#include <galpy_potentials.h>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
//Macros to export functions in DLL on different OS
#if defined(_WIN32)
#define EXPORT __declspec(dllexport)
#elif defined(__GNUC__)
#define EXPORT __attribute__((visibility("default")))
#else
// Just do nothing?
#define EXPORT
#endif
#ifdef _WIN32
// On Windows, *need* to define this function to allow the package to be imported
#if PY_MAJOR_VERSION >= 3
PyMODINIT_FUNC PyInit_galpy_integrate_c(void) { // Python 3
  return NULL;
}
#else
PyMODINIT_FUNC initgalpy_integrate_c(void) {} // Python 2
#endif
#endif
/*
  Function Declarations
*/
void evalRectForce(double, double *, double *,
		   int, struct potentialArg *);
void evalRectDeriv(double, double *, double *,
			 int, struct potentialArg *);
void evalRectDeriv_dxdv(double,double *, double *,
			      int, struct potentialArg *);
/*
  Actual functions
*/
void parse_leapFuncArgs_Full(int npot,
			     struct potentialArg * potentialArgs,
			     int ** pot_type,
			     double ** pot_args){
  int ii,jj,kk;
  int nR, nz;
  double * Rgrid, * zgrid, * potGrid_splinecoeffs;
  init_potentialArgs(npot,potentialArgs);
  for (ii=0; ii < npot; ii++){
    switch ( *(*pot_type)++ ) {
    case 0: //LogarithmicHaloPotential, 4 arguments
      potentialArgs->potentialEval= &LogarithmicHaloPotentialEval;
      potentialArgs->Rforce= &LogarithmicHaloPotentialRforce;
      potentialArgs->zforce= &LogarithmicHaloPotentialzforce;
      potentialArgs->phiforce= &LogarithmicHaloPotentialphiforce;
      //potentialArgs->R2deriv= &LogarithmicHaloPotentialR2deriv;
      //potentialArgs->planarphi2deriv= &ZeroForce;
      //potentialArgs->planarRphideriv= &ZeroForce;
      potentialArgs->nargs= 4;
      break;
    case 1: //DehnenBarPotential, 6 arguments
      potentialArgs->Rforce= &DehnenBarPotentialRforce;
      potentialArgs->phiforce= &DehnenBarPotentialphiforce;
      potentialArgs->zforce= &DehnenBarPotentialzforce;
      potentialArgs->nargs= 6;
      break;
    case 5: //MiyamotoNagaiPotential, 3 arguments
      potentialArgs->potentialEval= &MiyamotoNagaiPotentialEval;
      potentialArgs->Rforce= &MiyamotoNagaiPotentialRforce;
      potentialArgs->zforce= &MiyamotoNagaiPotentialzforce;
      potentialArgs->phiforce= &ZeroForce;
      //potentialArgs->R2deriv= &MiyamotoNagaiPotentialR2deriv;
      //potentialArgs->planarphi2deriv= &ZeroForce;
      //potentialArgs->planarRphideriv= &ZeroForce;
      potentialArgs->nargs= 3;
      break;
    case 7: //PowerSphericalPotential, 2 arguments
      potentialArgs->potentialEval= &PowerSphericalPotentialEval;
      potentialArgs->Rforce= &PowerSphericalPotentialRforce;
      potentialArgs->zforce= &PowerSphericalPotentialzforce;
      potentialArgs->phiforce= &ZeroForce;
      //potentialArgs->R2deriv= &PowerSphericalPotentialR2deriv;
      //potentialArgs->planarphi2deriv= &ZeroForce;
      //potentialArgs->planarRphideriv= &ZeroForce;
      potentialArgs->nargs= 2;
      break;
    case 8: //HernquistPotential, 2 arguments
      potentialArgs->potentialEval= &HernquistPotentialEval;
      potentialArgs->Rforce= &HernquistPotentialRforce;
      potentialArgs->zforce= &HernquistPotentialzforce;
      potentialArgs->phiforce= &ZeroForce;
      //potentialArgs->R2deriv= &HernquistPotentialR2deriv;
      //potentialArgs->planarphi2deriv= &ZeroForce;
      //potentialArgs->planarRphideriv= &ZeroForce;
      potentialArgs->nargs= 2;
      break;
    case 9: //NFWPotential, 2 arguments
      potentialArgs->potentialEval= &NFWPotentialEval;
      potentialArgs->Rforce= &NFWPotentialRforce;
      potentialArgs->zforce= &NFWPotentialzforce;
      potentialArgs->phiforce= &ZeroForce;
      //potentialArgs->R2deriv= &NFWPotentialR2deriv;
      //potentialArgs->planarphi2deriv= &ZeroForce;
      //potentialArgs->planarRphideriv= &ZeroForce;
      potentialArgs->nargs= 2;
      break;
    case 10: //JaffePotential, 2 arguments
      potentialArgs->potentialEval= &JaffePotentialEval;
      potentialArgs->Rforce= &JaffePotentialRforce;
      potentialArgs->zforce= &JaffePotentialzforce;
      potentialArgs->phiforce= &ZeroForce;
      //potentialArgs->R2deriv= &JaffePotentialR2deriv;
      //potentialArgs->planarphi2deriv= &ZeroForce;
      //potentialArgs->planarRphideriv= &ZeroForce;
      potentialArgs->nargs= 2;
      break;
    case 11: //DoubleExponentialDiskPotential, XX arguments
      potentialArgs->potentialEval= &DoubleExponentialDiskPotentialEval;
      potentialArgs->Rforce= &DoubleExponentialDiskPotentialRforce;
      potentialArgs->zforce= &DoubleExponentialDiskPotentialzforce;
      potentialArgs->phiforce= &ZeroForce;
      //Look at pot_args to figure out the number of arguments
      potentialArgs->nargs= (int) (8 + 2 * *(*pot_args+5) + 4 * ( *(*pot_args+4) + 1 ));
      break;
    case 12: //FlattenedPowerPotential, 4 arguments
      potentialArgs->potentialEval= &FlattenedPowerPotentialEval;
      potentialArgs->Rforce= &FlattenedPowerPotentialRforce;
      potentialArgs->zforce= &FlattenedPowerPotentialzforce;
      potentialArgs->phiforce= &ZeroForce;
      potentialArgs->nargs= 4;
      break;
    case 13: //interpRZPotential, XX arguments
      //Grab the grids and the coefficients
      nR= (int) *(*pot_args)++;
      nz= (int) *(*pot_args)++;
      Rgrid= (double *) malloc ( nR * sizeof ( double ) );
      zgrid= (double *) malloc ( nz * sizeof ( double ) );
      potGrid_splinecoeffs= (double *) malloc ( nR * nz * sizeof ( double ) );
      for (kk=0; kk < nR; kk++)
	*(Rgrid+kk)= *(*pot_args)++;
      for (kk=0; kk < nz; kk++)
	*(zgrid+kk)= *(*pot_args)++;
      for (kk=0; kk < nR; kk++)
	put_row(potGrid_splinecoeffs,kk,*pot_args+kk*nz,nz);
      *pot_args+= nR*nz;
      potentialArgs->i2d= interp_2d_alloc(nR,nz);
      interp_2d_init(potentialArgs->i2d,Rgrid,zgrid,potGrid_splinecoeffs,
		     INTERP_2D_LINEAR); //latter bc we already calculated the coeffs
      potentialArgs->accx= gsl_interp_accel_alloc ();
      potentialArgs->accy= gsl_interp_accel_alloc ();
      for (kk=0; kk < nR; kk++)
	put_row(potGrid_splinecoeffs,kk,*pot_args+kk*nz,nz); 
      *pot_args+= nR*nz;
      potentialArgs->i2drforce= interp_2d_alloc(nR,nz);
      interp_2d_init(potentialArgs->i2drforce,Rgrid,zgrid,potGrid_splinecoeffs,
		     INTERP_2D_LINEAR); //latter bc we already calculated the coeffs
      potentialArgs->accxrforce= gsl_interp_accel_alloc ();
      potentialArgs->accyrforce= gsl_interp_accel_alloc ();
      for (kk=0; kk < nR; kk++)
	put_row(potGrid_splinecoeffs,kk,*pot_args+kk*nz,nz); 
      *pot_args+= nR*nz;    
      potentialArgs->i2dzforce= interp_2d_alloc(nR,nz);
      interp_2d_init(potentialArgs->i2dzforce,Rgrid,zgrid,potGrid_splinecoeffs,
		     INTERP_2D_LINEAR); //latter bc we already calculated the coeffs
      potentialArgs->accxzforce= gsl_interp_accel_alloc ();
      potentialArgs->accyzforce= gsl_interp_accel_alloc ();
      potentialArgs->potentialEval= &interpRZPotentialEval;
      potentialArgs->Rforce= &interpRZPotentialRforce;
      potentialArgs->zforce= &interpRZPotentialzforce;
      potentialArgs->phiforce= &ZeroForce;
      potentialArgs->nargs= 2;
      //clean up
      free(Rgrid);
      free(zgrid);
      free(potGrid_splinecoeffs);
      break;
    case 14: //IsochronePotential, 2 arguments
      potentialArgs->potentialEval= &IsochronePotentialEval;
      potentialArgs->Rforce= &IsochronePotentialRforce;
      potentialArgs->zforce= &IsochronePotentialzforce;
      potentialArgs->phiforce= &ZeroForce;
      potentialArgs->nargs= 2;
      break;
    case 15: //PowerSphericalwCutoffPotential, 3 arguments
      potentialArgs->potentialEval= &PowerSphericalPotentialwCutoffEval;
      potentialArgs->Rforce= &PowerSphericalPotentialwCutoffRforce;
      potentialArgs->zforce= &PowerSphericalPotentialwCutoffzforce;
      potentialArgs->phiforce= &ZeroForce;
      //potentialArgs->R2deriv= &PowerSphericalPotentialR2deriv;
      //potentialArgs->planarphi2deriv= &ZeroForce;
      //potentialArgs->planarRphideriv= &ZeroForce;
      potentialArgs->nargs= 3;
      break;
    case 16: //KuzminKutuzovStaeckelPotential, 3 arguments
      potentialArgs->potentialEval= &KuzminKutuzovStaeckelPotentialEval;
      potentialArgs->Rforce= &KuzminKutuzovStaeckelPotentialRforce;
      potentialArgs->zforce= &KuzminKutuzovStaeckelPotentialzforce;
      potentialArgs->phiforce= &ZeroForce;
      //potentialArgs->R2deriv= &KuzminKutuzovStaeckelPotentialR2deriv;
      potentialArgs->nargs= 3;
      break;
    case 17: //PlummerPotential, 2 arguments
      potentialArgs->potentialEval= &PlummerPotentialEval;
      potentialArgs->Rforce= &PlummerPotentialRforce;
      potentialArgs->zforce= &PlummerPotentialzforce;
      potentialArgs->phiforce= &ZeroForce;
      //potentialArgs->R2deriv= &PlummerPotentialR2deriv;
      potentialArgs->nargs= 2;
      break;
    case 18: //PseudoIsothermalPotential, 2 arguments
      potentialArgs->potentialEval= &PseudoIsothermalPotentialEval;
      potentialArgs->Rforce= &PseudoIsothermalPotentialRforce;
      potentialArgs->zforce= &PseudoIsothermalPotentialzforce;
      potentialArgs->phiforce= &ZeroForce;
      //potentialArgs->R2deriv= &PseudoIsothermalPotentialR2deriv;
      potentialArgs->nargs= 2;
      break;
    case 19: //KuzminDiskPotential, 2 arguments
      potentialArgs->potentialEval= &KuzminDiskPotentialEval;
      potentialArgs->Rforce= &KuzminDiskPotentialRforce;
      potentialArgs->zforce= &KuzminDiskPotentialzforce;
      potentialArgs->phiforce= &ZeroForce;
      potentialArgs->nargs= 2;
      break;
    case 20: //BurkertPotential, 2 arguments
      potentialArgs->potentialEval= &BurkertPotentialEval;
      potentialArgs->Rforce= &BurkertPotentialRforce;
      potentialArgs->zforce= &BurkertPotentialzforce;
      potentialArgs->phiforce= &ZeroForce;
      potentialArgs->nargs= 2;
      break;
    case 21: //TriaxialHernquistPotential, lots of arguments
      potentialArgs->potentialEval= &EllipsoidalPotentialEval;
      potentialArgs->Rforce = &EllipsoidalPotentialRforce;
      potentialArgs->zforce = &EllipsoidalPotentialzforce;
      potentialArgs->phiforce = &EllipsoidalPotentialphiforce;
      // Also assign functions specific to EllipsoidalPotential
      potentialArgs->psi= &TriaxialHernquistPotentialpsi;
      potentialArgs->mdens= &TriaxialHernquistPotentialmdens;
      potentialArgs->mdensDeriv= &TriaxialHernquistPotentialmdensDeriv;
      potentialArgs->nargs = (int) (21 + *(*pot_args+7) + 2 * *(*pot_args 
					    + (int) (*(*pot_args+7) + 20)));
      break;
    case 22: //TriaxialNFWPotential, lots of arguments
      potentialArgs->potentialEval= &EllipsoidalPotentialEval;
      potentialArgs->Rforce = &EllipsoidalPotentialRforce;
      potentialArgs->zforce = &EllipsoidalPotentialzforce;
      potentialArgs->phiforce = &EllipsoidalPotentialphiforce;
      // Also assign functions specific to EllipsoidalPotential
      potentialArgs->psi= &TriaxialNFWPotentialpsi;
      potentialArgs->mdens= &TriaxialNFWPotentialmdens;
      potentialArgs->mdensDeriv= &TriaxialNFWPotentialmdensDeriv;
      potentialArgs->nargs = (int) (21 + *(*pot_args+7) + 2 * *(*pot_args 
					    + (int) (*(*pot_args+7) + 20)));
      break;
    case 23: //TriaxialJaffePotential, lots of arguments
      potentialArgs->potentialEval= &EllipsoidalPotentialEval;
      potentialArgs->Rforce = &EllipsoidalPotentialRforce;
      potentialArgs->zforce = &EllipsoidalPotentialzforce;
      potentialArgs->phiforce = &EllipsoidalPotentialphiforce;
      // Also assign functions specific to EllipsoidalPotential
      potentialArgs->psi= &TriaxialJaffePotentialpsi;
      potentialArgs->mdens= &TriaxialJaffePotentialmdens;
      potentialArgs->mdensDeriv= &TriaxialJaffePotentialmdensDeriv;
      potentialArgs->nargs = (int) (21 + *(*pot_args+7) + 2 * *(*pot_args 
					    + (int) (*(*pot_args+7) + 20)));
      break;
    case 24: //SCFPotential, many arguments
      potentialArgs->potentialEval= &SCFPotentialEval;
      potentialArgs->Rforce= &SCFPotentialRforce;
      potentialArgs->zforce= &SCFPotentialzforce;
      potentialArgs->phiforce= &SCFPotentialphiforce;
      potentialArgs->nargs= (int) (5 + (1 + *(*pot_args + 1)) * *(*pot_args+2) * *(*pot_args+3)* *(*pot_args+4) + 7);
      break;
    case 25: //SoftenedNeedleBarPotential, 13 arguments
      potentialArgs->potentialEval= &SoftenedNeedleBarPotentialEval;
      potentialArgs->Rforce= &SoftenedNeedleBarPotentialRforce;
      potentialArgs->zforce= &SoftenedNeedleBarPotentialzforce;
      potentialArgs->phiforce= &SoftenedNeedleBarPotentialphiforce;
      potentialArgs->nargs= (int) 13;
      break;      
    case 26: //DiskSCFPotential, nsigma+3 arguments
      potentialArgs->potentialEval= &DiskSCFPotentialEval;
      potentialArgs->Rforce= &DiskSCFPotentialRforce;
      potentialArgs->zforce= &DiskSCFPotentialzforce;
      potentialArgs->phiforce= &ZeroForce;
      potentialArgs->nargs= (int) **pot_args + 3;
      break;
    case 27: // SpiralArmsPotential, 10 arguments + array of Cs
      potentialArgs->Rforce = &SpiralArmsPotentialRforce;
      potentialArgs->zforce = &SpiralArmsPotentialzforce;
      potentialArgs->phiforce = &SpiralArmsPotentialphiforce;
      //potentialArgs->R2deriv = &SpiralArmsPotentialR2deriv;
      //potentialArgs->z2deriv = &SpiralArmsPotentialz2deriv;
      potentialArgs->phi2deriv = &SpiralArmsPotentialphi2deriv;
      //potentialArgs->Rzderiv = &SpiralArmsPotentialRzderiv;
      potentialArgs->Rphideriv = &SpiralArmsPotentialRphideriv;
      potentialArgs->nargs = (int) 10 + **pot_args;
      break;    
    case 30: // PerfectEllipsoidPotential, lots of arguments
      potentialArgs->potentialEval= &EllipsoidalPotentialEval;
      potentialArgs->Rforce = &EllipsoidalPotentialRforce;
      potentialArgs->zforce = &EllipsoidalPotentialzforce;
      potentialArgs->phiforce = &EllipsoidalPotentialphiforce;
      //potentialArgs->R2deriv = &EllipsoidalPotentialR2deriv;
      //potentialArgs->z2deriv = &EllipsoidalPotentialz2deriv;
      //potentialArgs->phi2deriv = &EllipsoidalPotentialphi2deriv;
      //potentialArgs->Rzderiv = &EllipsoidalPotentialRzderiv;
      //potentialArgs->Rphideriv = &EllipsoidalPotentialRphideriv;
      // Also assign functions specific to EllipsoidalPotential
      potentialArgs->psi= &PerfectEllipsoidPotentialpsi;
      potentialArgs->mdens= &PerfectEllipsoidPotentialmdens;
      potentialArgs->mdensDeriv= &PerfectEllipsoidPotentialmdensDeriv;
      potentialArgs->nargs = (int) (21 + *(*pot_args+7) + 2 * *(*pot_args 
					    + (int) (*(*pot_args+7) + 20)));
      break;    
//////////////////////////////// WRAPPERS /////////////////////////////////////
    case -1: //DehnenSmoothWrapperPotential
      potentialArgs->potentialEval= &DehnenSmoothWrapperPotentialEval;
      potentialArgs->Rforce= &DehnenSmoothWrapperPotentialRforce;
      potentialArgs->zforce= &DehnenSmoothWrapperPotentialzforce;
      potentialArgs->phiforce= &DehnenSmoothWrapperPotentialphiforce;
      potentialArgs->nargs= (int) 3;
      break;
    case -2: //SolidBodyRotationWrapperPotential
      potentialArgs->Rforce= &SolidBodyRotationWrapperPotentialRforce;
      potentialArgs->zforce= &SolidBodyRotationWrapperPotentialzforce;
      potentialArgs->phiforce= &SolidBodyRotationWrapperPotentialphiforce;
      potentialArgs->nargs= (int) 3;
      break;
    case -4: //CorotatingRotationWrapperPotential
      potentialArgs->Rforce= &CorotatingRotationWrapperPotentialRforce;
      potentialArgs->zforce= &CorotatingRotationWrapperPotentialzforce;
      potentialArgs->phiforce= &CorotatingRotationWrapperPotentialphiforce;
      potentialArgs->nargs= (int) 5;
      break;
    case -5: //GaussianAmplitudeWrapperPotential
      potentialArgs->potentialEval= &GaussianAmplitudeWrapperPotentialEval;
      potentialArgs->Rforce= &GaussianAmplitudeWrapperPotentialRforce;
      potentialArgs->zforce= &GaussianAmplitudeWrapperPotentialzforce;
      potentialArgs->phiforce= &GaussianAmplitudeWrapperPotentialphiforce;
      potentialArgs->nargs= (int) 3;
      break;
    }
    if ( *(*pot_type-1) < 0 ) { // Parse wrapped potential for wrappers
      potentialArgs->nwrapped= (int) *(*pot_args)++;
      potentialArgs->wrappedPotentialArg= \
	(struct potentialArg *) malloc ( potentialArgs->nwrapped	\
					 * sizeof (struct potentialArg) );
      parse_leapFuncArgs_Full(potentialArgs->nwrapped,
			      potentialArgs->wrappedPotentialArg,
			      pot_type,pot_args);
    }
    potentialArgs->args= (double *) malloc( potentialArgs->nargs * sizeof(double));
    for (jj=0; jj < potentialArgs->nargs; jj++){
      *(potentialArgs->args)= *(*pot_args)++;
      potentialArgs->args++;
    }
    potentialArgs->args-= potentialArgs->nargs;
    potentialArgs++;
  }
  potentialArgs-= npot;
}
EXPORT void integrateFullOrbit(double *yo,
			       int nt, 
			       double *t,
			       int npot,
			       int * pot_type,
			       double * pot_args,
			       double dt,
			       double rtol,
			       double atol,
			       double *result,
			       int * err,
			       int odeint_type){
  //Set up the forces, first count
  int dim;
  struct potentialArg * potentialArgs= (struct potentialArg *) malloc ( npot * sizeof (struct potentialArg) );
  parse_leapFuncArgs_Full(npot,potentialArgs,&pot_type,&pot_args);
  //Integrate
  void (*odeint_func)(void (*func)(double, double *, double *,
			   int, struct potentialArg *),
		      int,
		      double *,
		      int, double, double *,
		      int, struct potentialArg *,
		      double, double,
		      double *,int *);
  void (*odeint_deriv_func)(double, double *, double *,
			    int,struct potentialArg *);
  switch ( odeint_type ) {
  case 0: //leapfrog
    odeint_func= &leapfrog;
    odeint_deriv_func= &evalRectForce;
    dim= 3;
    break;
  case 1: //RK4
    odeint_func= &bovy_rk4;
    odeint_deriv_func= &evalRectDeriv;
    dim= 6;
    break;
  case 2: //RK6
    odeint_func= &bovy_rk6;
    odeint_deriv_func= &evalRectDeriv;
    dim= 6;
    break;
  case 3: //symplec4
    odeint_func= &symplec4;
    odeint_deriv_func= &evalRectForce;
    dim= 3;
    break;
  case 4: //symplec6
    odeint_func= &symplec6;
    odeint_deriv_func= &evalRectForce;
    dim= 3;
    break;
  case 5: //DOPR54
    odeint_func= &bovy_dopr54;
    odeint_deriv_func= &evalRectDeriv;
    dim= 6;
    break;
  }
  odeint_func(odeint_deriv_func,dim,yo,nt,dt,t,npot,potentialArgs,rtol,atol,
	      result,err);
  //Free allocated memory
  free_potentialArgs(npot,potentialArgs);
  free(potentialArgs);
  //Done!
}
// LCOV_EXCL_START
void integrateOrbit_dxdv(double *yo,
			 int nt, 
			 double *t,
			 int npot,
			 int * pot_type,
			 double * pot_args,
			 double rtol,
			 double atol,
			 double *result,
			 int * err,
			 int odeint_type){
  //Set up the forces, first count
  int dim;
  struct potentialArg * potentialArgs= (struct potentialArg *) malloc ( npot * sizeof (struct potentialArg) );
  parse_leapFuncArgs_Full(npot,potentialArgs,&pot_type,&pot_args);
  //Integrate
  void (*odeint_func)(void (*func)(double, double *, double *,
			   int, struct potentialArg *),
		      int,
		      double *,
		      int, double, double *,
		      int, struct potentialArg *,
		      double, double,
		      double *,int *);
  void (*odeint_deriv_func)(double, double *, double *,
			    int,struct potentialArg *);
  switch ( odeint_type ) {
  case 0: //leapfrog
    odeint_func= &leapfrog;
    odeint_deriv_func= &evalRectForce;
    dim= 6;
    break;
  case 1: //RK4
    odeint_func= &bovy_rk4;
    odeint_deriv_func= &evalRectDeriv_dxdv;
    dim= 12;
    break;
  case 2: //RK6
    odeint_func= &bovy_rk6;
    odeint_deriv_func= &evalRectDeriv_dxdv;
    dim= 12;
    break;
  case 3: //symplec4
    odeint_func= &symplec4;
    odeint_deriv_func= &evalRectForce;
    dim= 6;
    break;
  case 4: //symplec6
    odeint_func= &symplec6;
    odeint_deriv_func= &evalRectForce;
    dim= 6;
    break;
  case 5: //DOPR54
    odeint_func= &bovy_dopr54;
    odeint_deriv_func= &evalRectDeriv_dxdv;
    dim= 12;
    break;
  }
  odeint_func(odeint_deriv_func,dim,yo,nt,-9999.99,t,npot,potentialArgs,
	      rtol,atol,result,err);
  //Free allocated memory
  free_potentialArgs(npot,potentialArgs);
  free(potentialArgs);
  //Done!
}
// LCOV_EXCL_STOP
void evalRectForce(double t, double *q, double *a,
		   int nargs, struct potentialArg * potentialArgs){
  double sinphi, cosphi, x, y, phi,R,Rforce,phiforce, z, zforce;
  //q is rectangular so calculate R and phi
  x= *q;
  y= *(q+1);
  z= *(q+2);
  R= sqrt(x*x+y*y);
  phi= acos(x/R);
  sinphi= y/R;
  cosphi= x/R;
  if ( y < 0. ) phi= 2.*M_PI-phi;
  //Calculate the forces
  Rforce= calcRforce(R,z,phi,t,nargs,potentialArgs);
  zforce= calczforce(R,z,phi,t,nargs,potentialArgs);
  phiforce= calcPhiforce(R,z,phi,t,nargs,potentialArgs);
  *a++= cosphi*Rforce-1./R*sinphi*phiforce;
  *a++= sinphi*Rforce+1./R*cosphi*phiforce;
  *a= zforce;
}
void evalRectDeriv(double t, double *q, double *a,
		   int nargs, struct potentialArg * potentialArgs){
  double sinphi, cosphi, x, y, phi,R,Rforce,phiforce,z,zforce;
  //first three derivatives are just the velocities
  *a++= *(q+3);
  *a++= *(q+4);
  *a++= *(q+5);
  //Rest is force
  //q is rectangular so calculate R and phi
  x= *q;
  y= *(q+1);
  z= *(q+2);
  R= sqrt(x*x+y*y);
  phi= acos(x/R);
  sinphi= y/R;
  cosphi= x/R;
  if ( y < 0. ) phi= 2.*M_PI-phi;
  //Calculate the forces
  Rforce= calcRforce(R,z,phi,t,nargs,potentialArgs);
  zforce= calczforce(R,z,phi,t,nargs,potentialArgs);
  phiforce= calcPhiforce(R,z,phi,t,nargs,potentialArgs);
  *a++= cosphi*Rforce-1./R*sinphi*phiforce;
  *a++= sinphi*Rforce+1./R*cosphi*phiforce;
  *a= zforce;
}

// LCOV_EXCL_START
void evalRectDeriv_dxdv(double t, double *q, double *a,
			int nargs, struct potentialArg * potentialArgs){
  double sinphi, cosphi, x, y, phi,R,Rforce,phiforce,z,zforce;
  double R2deriv, phi2deriv, Rphideriv, dFxdx, dFxdy, dFydx, dFydy;
  //first three derivatives are just the velocities
  *a++= *(q+3);
  *a++= *(q+4);
  *a++= *(q+5);
  //Rest is force
  //q is rectangular so calculate R and phi
  x= *q;
  y= *(q+1);
  z= *(q+2);
  R= sqrt(x*x+y*y);
  phi= acos(x/R);
  sinphi= y/R;
  cosphi= x/R;
  if ( y < 0. ) phi= 2.*M_PI-phi;
  //Calculate the forces
  Rforce= calcRforce(R,z,phi,t,nargs,potentialArgs);
  zforce= calczforce(R,z,phi,t,nargs,potentialArgs);
  phiforce= calcPhiforce(R,z,phi,t,nargs,potentialArgs);
  *a++= cosphi*Rforce-1./R*sinphi*phiforce;
  *a++= sinphi*Rforce+1./R*cosphi*phiforce;
  *a++= zforce;
  //dx derivatives are just dv
  *a++= *(q+9);
  *a++= *(q+10);
  *a++= *(q+11);
  //for the dv derivatives we need also R2deriv, phi2deriv, and Rphideriv
  R2deriv= calcR2deriv(R,z,phi,t,nargs,potentialArgs);
  phi2deriv= calcphi2deriv(R,z,phi,t,nargs,potentialArgs);
  Rphideriv= calcRphideriv(R,z,phi,t,nargs,potentialArgs);
  //..and dFxdx, dFxdy, dFydx, dFydy
  dFxdx= -cosphi*cosphi*R2deriv
    +2.*cosphi*sinphi/R/R*phiforce
    +sinphi*sinphi/R*Rforce
    +2.*sinphi*cosphi/R*Rphideriv
    -sinphi*sinphi/R/R*phi2deriv;
  dFxdy= -sinphi*cosphi*R2deriv
    +(sinphi*sinphi-cosphi*cosphi)/R/R*phiforce
    -cosphi*sinphi/R*Rforce
    -(cosphi*cosphi-sinphi*sinphi)/R*Rphideriv
    +cosphi*sinphi/R/R*phi2deriv;
  dFydx= -cosphi*sinphi*R2deriv
    +(sinphi*sinphi-cosphi*cosphi)/R/R*phiforce
    +(sinphi*sinphi-cosphi*cosphi)/R*Rphideriv
    -sinphi*cosphi/R*Rforce
    +sinphi*cosphi/R/R*phi2deriv;
  dFydy= -sinphi*sinphi*R2deriv
    -2.*sinphi*cosphi/R/R*phiforce
    -2.*sinphi*cosphi/R*Rphideriv
    +cosphi*cosphi/R*Rforce
    -cosphi*cosphi/R/R*phi2deriv;
  *a++= dFxdx * *(q+4) + dFxdy * *(q+5);
  *a++= dFydx * *(q+4) + dFydy * *(q+5);
  *a= 0; //BOVY: PUT IN Z2DERIVS
}

// LCOV_EXCL_STOP
