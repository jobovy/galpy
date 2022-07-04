// -*- C++ -*-                                                                 |
//-----------------------------------------------------------------------------+
//
// PowSphwCut.cc:
//
//    Power-law potential with exponential cut-off
//    (galpy's PowerSphericalPotentialwCutoff)
//    Based on LogPot.cc in NEMO
//-----------------------------------------------------------------------------+
//#############################################################################
//Copyright (c) 2014, Jo Bovy
//All rights reserved.
//
//Redistribution and use in source and binary forms, with or without
//modification, are permitted provided that the following conditions are met:
//
//   Redistributions of source code must retain the above copyright notice,
//      this list of conditions and the following disclaimer.
//   Redistributions in binary form must reproduce the above copyright notice,
//      this list of conditions and the following disclaimer in the
//      documentation and/or other materials provided with the distribution.
//   The name of the author may not be used to endorse or promote products
//      derived from this software without specific prior written permission.
//
//THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
//"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
//LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
//A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
//HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
//INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
//BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
//OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED
//AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
//LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY
//WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
//POSSIBILITY OF SUCH DAMAGE.
//#############################################################################
#include <iostream>
#include <cmath>
#define POT_DEF
#include <defacc.h> // from NEMOINC
#include <gsl/gsl_sf_gamma.h>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
////////////////////////////////////////////////////////////////////////////////
using namespace std;
////////////////////////////////////////////////////////////////////////////////
namespace {
  //////////////////////////////////////////////////////////////////////////////
  // Function that gives the mass
  double mass(double r2,double alpha, double rc){
    return 2. * M_PI * pow ( rc , 3. - alpha ) * ( gsl_sf_gamma ( 1.5 - 0.5 * alpha ) - gsl_sf_gamma_inc ( 1.5 - 0.5 * alpha , r2 / rc / rc ) );
  }
  class PowSphwCut
  {
    double amp; // Potential amplityde
    double alpha; // Power-law exponent
    double rc; // Radius of exponential cut-off
  public:
    //--------------------------------------------------------------------------
    static const char* name() { return "PowSphwCut"; }
    bool NeedMass() const { return false; }
    bool NeedVels() const { return false; }
    //--------------------------------------------------------------------------
    PowSphwCut(const double*pars, int npar, const char*)
      : amp ( npar>1? pars[1] : 1.),
      	alpha ( npar>2? pars[2] : 1.),
      	rc ( npar>3? pars[3] : 1.)
    {
      if(amp<=0) error("PowSphwCut: amp <=0\n");
      if(rc< 0) error("PowSphwCut: cut-off radius <0\n");
      if((npar<3 && nemo_debug(1)) || nemo_debug(2) )
	std::cerr<<
	"### PowSphwCut: external power-law w/ exp. cut-off potential\n\n"
	"\n"
	"      rho = amp / r^alpha exp(-(r/rc)^2)\n"
	"\n\n"
	"      par[0] ignored\n"
	"      par[1] amp : amplitude (default: 1)\n"
	"      par[2] alpha : power-law exponent (default: 1)\n"
	"      par[3] rc: cut-off radius (default: 1)\n";
    }
    //--------------------------------------------------------------------------
    template<int NDIM, typename scalar>
    void set_time(double       ,
		  int          ,
		  const scalar*,
		  const scalar*,
		  const scalar*) const {}
    //--------------------------------------------------------------------------
    template<int NDIM, typename scalar>
    void acc(const scalar*,
	     const scalar*X,
	     const scalar*,
	     scalar      &P,
	     scalar      *A) const
    {
      if(NDIM == 3) {
	double r2 = X[0]*X[0]+X[1]*X[1]+X[2]*X[2];
	double rforceOverr = - amp * mass (r2,alpha,rc) / pow(r2,1.5);
	A[0] = rforceOverr * X[0];
	A[1] = rforceOverr * X[1];
	A[2] = rforceOverr * X[2];
	double r = sqrt(r2);
	P    = amp * 2. * M_PI * pow(rc,3.-alpha) / r * ( r / rc * ( gsl_sf_gamma ( 1. - 0.5 * alpha ) - gsl_sf_gamma_inc ( 1. - 0.5 * alpha , r2 / rc / rc ) ) - ( gsl_sf_gamma ( 1.5 - 0.5 * alpha ) - gsl_sf_gamma_inc ( 1.5 - 0.5 * alpha , r2 / rc / rc) ) );
      } else {
	P = A[0] = A[1] = A[2] = 0;
	error("PowSphwCut: wrong number (%d) of dimensions, only allow 3D\n",NDIM);}
    }
  };
} // namespace {
//------------------------------------------------------------------------------
__DEF__ACC(PowSphwCut)
__DEF__POT(PowSphwCut)
//------------------------------------------------------------------------------
