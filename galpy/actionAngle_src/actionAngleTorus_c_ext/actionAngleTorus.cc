/*
  C code wrapper around Dehnen/McMillan Torus code for actionAngle calculations
*/
#include <new>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <cstdio>
#include <ctime>
#include "Torus.h"
#include "LogPot.h"
#include "galpyPot.h"
#include <actionAngle.h>
#include <integrateFullOrbit.h>
#include <galpy_potentials.h>

extern "C"
{
  // Calculate (x,v) for angles on a single torus; also returns the frequencies
  void actionAngleTorus_xvFreqs(double jr, double jphi, double jz,
				int na,
				double * angler, double * anglephi, double * anglez,
				int npot,
				int * pot_type,
				double * pot_args,
				double * R, double * vR, double * vT, 
				double * z, double * vz, double * phi,
				double * Omegar,double * Omegaphi,double * Omegaz)
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
    double dJ= 0.003;
    int flag;
    flag = T->AutoFit(J,Phi,dJ);

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
  }
}
