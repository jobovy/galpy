/***************************************************************************//**\file galpyPot.h
\brief Contains class galpyPotential. 
General interface to galpy potentials
									     
*                                                                              *
* galpyPot.h                                                                   *
*                                                                              *
* C++ code written by Jo Bovy, 2015                                            *
*******************************************************************************/
#ifndef __GALPY_GALPYPOT_H__
#define __GALPY_GALPYPOT_H__
#include "Potential.h"
#include <galpy_potentials.h>

/** \brief A general interface to galpy potentials in C++

 */
// Class declaration
class galpyPotential : public Potential {
  int nargs;
  struct potentialArg * potentialArgs;
  void  error(const char*) const;
 public:
  galpyPotential(int,struct potentialArg *);
  double operator() (const double, const double) const;
  double operator() (const double, double&, double&) const;//??
  double operator() (const double, const double, double&, double&) const;
  double operator() (const double, const double,
		     double&, double&, double&, double&, double&) const;//??
  double RfromLc(const double, double* = 0) const;
  double LfromRc(const double, double* = 0) const;
  Frequencies KapNuOm(const double) const;
};


inline galpyPotential::galpyPotential(int na,
				      struct potentialArg * inPotentialArgs) :
		      nargs(na), potentialArgs(inPotentialArgs)
{ 

}

#endif /* galpyPot.h */
