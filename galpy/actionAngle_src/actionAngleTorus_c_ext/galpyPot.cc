/***************************************************************************//**\file galpyPot.cc
\brief Contains class galpyPotential. 
General interface to galpy potentials
									     
*                                                                              *
* galpyPot.cc                                                                   *
*                                                                              *
* C++ code written by Jo Bovy, 2015                                            *
*******************************************************************************/
//#include <iostream>
#include <galpyPot.h>
#include <galpy_potentials.h>

 // Implementation for completeness, some parts never touched by galpy's use of TM
// LCOV_EXCL_START
void  galpyPotential::error(const char* msgs) const
{
  cerr << " Error in class galpyPotential: " << msgs << '\n';
  exit(1);
}

double galpyPotential::operator() (double R, double z) const
{
  return evaluatePotentials(R,z,nargs,potentialArgs);
}
// LCOV_EXCL_STOP

double galpyPotential::operator() (double R, double z,
				   double& dPdR,double& dPdz) const
{
  dPdR= -calcRforce(R,z,0.,0.,nargs,potentialArgs);
  dPdz= -calczforce(R,z,0.,0.,nargs,potentialArgs);
  return evaluatePotentials(R,z,nargs,potentialArgs);
}
// LCOV_EXCL_START
double galpyPotential::LfromRc(const double R, double* dR) const
{
  double dPR,dPz,P;
  P = (*this)(R,0.,dPR,dPz);
  return sqrt(R*R*R*dPR);  
}
// LCOV_EXCL_STOP
double galpyPotential::RfromLc(const double L_in, double* dR) const
{
  bool more=false;
  double R,lR=0.,dlR=0.001,dPR,dPz,P,LcR,oldL,L=fabs(L_in);
  R=exp(lR);
  P= (*this)(R,0.,dPR,dPz);
  LcR=sqrt(R*R*R*dPR);
  if(LcR == L) return R;
  if(L>LcR) more=true;
  oldL=LcR;
  
  for( ; ; ) {
    lR += (more)? dlR : -dlR;
    R=exp(lR);
    P= (*this)(R,0.,dPR,dPz);
    LcR=sqrt(R*R*R*dPR);
    if(LcR == L) return R;
    if((L< LcR && L>oldL) ||(L>LcR && L<oldL)){
      R=(more)? exp(lR-0.5*dlR) : exp(lR+0.5*dlR);
      return R;}
    oldL=LcR;
  }
  
}
// LCOV_EXCL_START
Frequencies galpyPotential::KapNuOm(const double R) const {
  Frequencies KNO = -9999.99;
  return KNO;
}
// LCOV_EXCL_STOP


