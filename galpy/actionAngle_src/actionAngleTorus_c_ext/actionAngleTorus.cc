/*
  C code wrapper around Dehnen/McMillan Torus code for actionAngle calculations
*/
#include <new>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <cstdio>
#include <ctime>
#include <gsl/gsl_spline.h>
#include "Torus.h"
#include "interp_2d.h"
#include "galpyPot.h"
#include <actionAngle.h>
#include <integrateFullOrbit.h>
#include <galpy_potentials.h>

extern "C"
{
  // Clean up function
  inline void cleanup(Torus * T,Potential * Phi,
		      int npot,struct potentialArg * actionAngleArgs)
  {
    delete Phi;
    delete T;
    int ii;
    for (ii=0; ii < npot; ii++) {
      if ( (actionAngleArgs+ii)->i2d )
	interp_2d_free((actionAngleArgs+ii)->i2d) ;
      if ((actionAngleArgs+ii)->accx )
	gsl_interp_accel_free ((actionAngleArgs+ii)->accx);
      if ((actionAngleArgs+ii)->accy )
	gsl_interp_accel_free ((actionAngleArgs+ii)->accy);
      if ( (actionAngleArgs+ii)->i2drforce )
	interp_2d_free((actionAngleArgs+ii)->i2drforce) ;
      if ((actionAngleArgs+ii)->accxrforce )
	gsl_interp_accel_free ((actionAngleArgs+ii)->accxrforce);
      if ((actionAngleArgs+ii)->accyrforce )
	gsl_interp_accel_free ((actionAngleArgs+ii)->accyrforce);
      if ( (actionAngleArgs+ii)->i2dzforce )
	interp_2d_free((actionAngleArgs+ii)->i2dzforce) ;
      if ((actionAngleArgs+ii)->accxzforce )
	gsl_interp_accel_free ((actionAngleArgs+ii)->accxzforce);
      if ((actionAngleArgs+ii)->accyzforce )
	gsl_interp_accel_free ((actionAngleArgs+ii)->accyzforce);
      free((actionAngleArgs+ii)->args);
    }
    free(actionAngleArgs);
  }
  // Calculate frequencies
  void actionAngleTorus_Freqs(double jr, double jphi, double jz,
			      int npot,
			      int * pot_type,
			      double * pot_args,
			      double tol,
			      double * Omegar,double * Omegaphi,double * Omegaz,
			      int * flag)
  {
    // set up Torus
    Torus *T;
    T= new(std::nothrow) Torus;
    
    // set up potential
    Potential *Phi;
    //Phi = new(std::nothrow) LogPotential(1.,0.8,0.,0.);
    struct potentialArg * actionAngleArgs= (struct potentialArg *) malloc ( npot * sizeof (struct potentialArg) );
    parse_actionAngleArgs(npot,actionAngleArgs,pot_type,pot_args,true);
    Phi = new(std::nothrow) galpyPotential(npot,actionAngleArgs);

    // Load actions and fit Torus
    Actions J;
    J[0]= jr;
    J[1]= jz;
    J[2]= jphi;
    *flag = T->AutoFit(J,Phi,tol);

    Phi->set_Lz(J(2));

    // Grab the frequencies
    Frequencies om=T->omega();
    *Omegar= om(0);
    *Omegaz= om(1);
    *Omegaphi= om(2);

    // Clean up
    cleanup(T,Phi,npot,actionAngleArgs);
  }
  // Calculate (x,v) for angles on a single torus; also returns the frequencies
  void actionAngleTorus_xvFreqs(double jr, double jphi, double jz,
				int na,
				double * angler, double * anglephi, double * anglez,
				int npot,
				int * pot_type,
				double * pot_args,
				double tol,
				double * R, double * vR, double * vT, 
				double * z, double * vz, double * phi,
				double * Omegar,double * Omegaphi,double * Omegaz,
				int * flag)
  {
    // set up Torus
    Torus *T;
    T= new(std::nothrow) Torus;
    
    // set up potential
    Potential *Phi;
    //Phi = new(std::nothrow) LogPotential(1.,0.8,0.,0.);
    struct potentialArg * actionAngleArgs= (struct potentialArg *) malloc ( npot * sizeof (struct potentialArg) );
    parse_actionAngleArgs(npot,actionAngleArgs,pot_type,pot_args,true);
    Phi = new(std::nothrow) galpyPotential(npot,actionAngleArgs);

    // Load actions and fit Torus
    Actions J;
    J[0]= jr;
    J[1]= jz;
    J[2]= jphi;
    *flag = T->AutoFit(J,Phi,tol);

    Phi->set_Lz(J(2));

    // Load angles and get (x,v)
    Angles A;
    PSPT Q;
    int ii;
    for (ii=0; ii < na; ii++) {
      // Load angles
      A[0]= *(angler+ii);
      A[1]= *(anglez+ii);
      A[2]= *(anglephi+ii);
      // get phase-space point
      Q= T->Map3D(A);
      *(R+ii)= Q(0);
      *(z+ii)= Q(1);
      *(phi+ii)= Q(2);
      *(vR+ii)= Q(3);
      *(vz+ii)= Q(4);
      *(vT+ii)= Q(5);
    }

    // Finally, grab the frequencies
    Frequencies om=T->omega();
    *Omegar= om(0);
    *Omegaz= om(1);
    *Omegaphi= om(2);

    // Clean up
    cleanup(T,Phi,npot,actionAngleArgs);
  }
}
