/*
  C code wrapper around Dehnen/McMillan Torus code for actionAngle calculations
*/
#include <new>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <cstdio>
#include <ctime>
#include <cmath>
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
    free_potentialArgs(npot,actionAngleArgs);
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
    parse_actionAngleArgs(npot,actionAngleArgs,&pot_type,&pot_args,true);
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
    parse_actionAngleArgs(npot,actionAngleArgs,&pot_type,&pot_args,true);
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
  // Calculate Hessian and frequencies
  void actionAngleTorus_hessianFreqs(double jr, double jphi, double jz,
				     int npot,
				     int * pot_type,
				     double * pot_args,
				     double tol,
				     double indJ,
				     double * dOdJT,
				     double * Omegar,
				     double * Omegaphi,
				     double * Omegaz,
				     int * flag)
  {
    int ii,jj;
    double dJ;
    // set up Torus
    Torus *T;
    T= new(std::nothrow) Torus;
    
    // set up potential
    Potential *Phi;
    //Phi = new(std::nothrow) LogPotential(1.,0.8,0.,0.);
    struct potentialArg * actionAngleArgs= (struct potentialArg *) malloc ( npot * sizeof (struct potentialArg) );
    parse_actionAngleArgs(npot,actionAngleArgs,&pot_type,&pot_args,true);
    Phi = new(std::nothrow) galpyPotential(npot,actionAngleArgs);

    // Load actions and fit Torus
    Actions J,JdJ;
    Frequencies om, omdom;
    J[0]= jr;
    J[1]= jz;
    J[2]= jphi;
    *flag = T->AutoFit(J,Phi,tol);

    Phi->set_Lz(J(2));

    // Grab the frequencies
    om=T->omega();
    *Omegar= om(0);
    *Omegaz= om(1);
    *Omegaphi= om(2);

    // Now compute the Jacobian
    for (ii=0;ii < 3; ii++){
      JdJ= J;
      dJ= J[ii]+indJ;
      dJ= dJ-J[ii];
      JdJ[ii]= J[ii]+dJ;
      T->AutoFit(JdJ,Phi,tol);
      Phi->set_Lz(JdJ(2));
      omdom=T->omega();
      for (jj=0;jj<3;jj++) *(dOdJT+ii*3+jj)= (omdom(jj)-om(jj)) / dJ;
    }

    // Clean up
    cleanup(T,Phi,npot,actionAngleArgs);
  }
  // Calculate Jacobian and frequencies
  void actionAngleTorus_jacobianFreqs(double jr,double jphi, 
				      double jz,int na,double * angler,
				      double * anglephi, double * anglez,
				      int npot,
				      int * pot_type,
				      double * pot_args,
				      double tol,
				      double indJ,
				      double * R, double * vR, double * vT, 
				      double * z, double * vz, double * phi,
				      double * dxvOdJaT,
				      double * dOdJT,
				      double * Omegar,
				      double * Omegaphi,
				      double * Omegaz,
				      int * flag)
  {
    int ii,jj,kk;
    double dJ, dA;
    // set up Torus
    Torus *T;
    T= new(std::nothrow) Torus;
    
    // set up potential
    Potential *Phi;
    struct potentialArg * actionAngleArgs= (struct potentialArg *) malloc ( npot * sizeof (struct potentialArg) );
    parse_actionAngleArgs(npot,actionAngleArgs,&pot_type,&pot_args,true);
    Phi = new(std::nothrow) galpyPotential(npot,actionAngleArgs);

    // Load actions and fit Torus
    Actions J,JdJ;
    Frequencies om, omdom;
    Angles A, AdA;
    PSPT Q, QdQ;
    PSPT * Qs= (PSPT *) malloc ( na * sizeof ( PSPT ) );
    J[0]= jr;
    J[1]= jz;
    J[2]= jphi;
    *flag = T->AutoFit(J,Phi,tol);

    Phi->set_Lz(J(2));

    // Grab the frequencies
    om=T->omega();
    *Omegar= om(0);
    *Omegaz= om(1);
    *Omegaphi= om(2);

    // Now compute the Jacobian: dangle changes
    for (ii=0;ii < na;ii++){
      // Load angles and get phase-space point
      A[0]= *(angler+ii);
      A[1]= *(anglez+ii);
      A[2]= *(anglephi+ii);
      Q= T->Map3D(A);
      *(Qs+ii)= Q; // store for dJ calc. below
      *(R+ii)= Q(0); // output x,v
      *(z+ii)= Q(1);
      *(phi+ii)= Q(2);
      *(vR+ii)= Q(3);
      *(vz+ii)= Q(4);
      *(vT+ii)= Q(5);
      for (jj=0;jj < 3;jj++){
	// Setup dangle
	AdA= A;
	dA= A[jj]+1.e-8;
	dA= dA-A[jj];
	AdA[jj]= A[jj]+dA;
	// get phase-space point
	QdQ= T->Map3D(AdA);
	for (kk=0;kk < 6;kk++)
	  *(dxvOdJaT+ii*36+(jj+3)*6+kk)= (QdQ(kk)-Q(kk)) / dA;
      }
    }
    // Now compute the Jacobian: dJ changes
    for (jj=0;jj < 3; jj++){
      // Setup dJ torus
      JdJ= J;
      dJ= J[jj]+indJ;
      dJ= dJ-J[jj];
      JdJ[jj]= J[jj]+dJ;
      T->AutoFit(JdJ,Phi,tol);
      Phi->set_Lz(JdJ(2));
      for (ii=0;ii < na;ii++){
	// Load angles and get phase-space point
	A[0]= *(angler+ii);
	A[1]= *(anglez+ii);
	A[2]= *(anglephi+ii);
	QdQ= T->Map3D(A);
	for (kk=0;kk < 6;kk++)
	  *(dxvOdJaT+ii*36+jj*6+kk)= (QdQ(kk)-(*(Qs+ii))(kk)) / dJ;
      }
      // and frequencies
      omdom=T->omega();
      for (kk=0;kk<3;kk++) *(dOdJT+jj*3+kk)= (omdom(kk)-om(kk)) / dJ;
    }

    // Clean up
    free(Qs);
    cleanup(T,Phi,npot,actionAngleArgs);
  }
}
