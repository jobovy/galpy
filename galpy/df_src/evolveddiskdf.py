###############################################################################
#   evolveddiskdf.py: module that builds a distribution function as a 
#                     steady-state DF + subsequent evolution
#
#   This module contains the following classes:
#
#      evolveddiskdf - top-level class that represents a distribution function
###############################################################################
_NSIGMA= 4.
_NTS= 1000
import sys
import math
import copy
import numpy as nu
from scipy import integrate
from galpy.orbit import Orbit
from galpy.potential import calcRotcurve
from galpy.util.bovy_quadpack import dblquad
from galpy.util import bovy_plot
_DEGTORAD= math.pi/180.
_RADTODEG= 180./math.pi
_NAN= nu.nan
class evolveddiskdf:
    """Class that represents a diskdf as initial DF + subsequent secular evolution"""
    def __init__(self,initdf,pot,to=0.):
        """
        NAME:
           __init__
        PURPOSE:
           initialize
        INPUT:
           initdf - the df at the start of the evolution (at to)
           pot - potential to integrate orbits in
           to= initial time (time at which initdf is evaluated; orbits are 
               integrated from current t back to to)
        OUTPUT:
        HISTORY:
           2011-03-30 - Written - Bovy (NYU)
        """
        self._initdf= initdf
        self._pot= pot
        self._to= to

    def __call__(self,*args,**kwargs):
        """
        NAME:
           __call__
        PURPOSE:
           evaluate the distribution function
        INPUT:
           Orbit instance:
              a) Orbit instance alone: use initial state and t=0
              b) Orbit instance + t: Orbit instance *NOT* called (i.e., Orbit's initial condition is used, call Orbit yourself)
                 If t is a list of t, DF is returned for each t, times must be in descending order (does not work with marginalize...)

        KWARGS:

           marginalizeVperp - marginalize over perpendicular velocity (only supported with 1a) above) + nsigma, +scipy.integrate.quad keywords

           marginalizeVlos - marginalize over line-of-sight velocity (only supported with 1a) above) + nsigma, +scipy.integrate.quad keywords

        OUTPUT:
           DF(orbit,t)
        HISTORY:
           2011-03-30 - Written - Bovy (NYU)
           2011-04-15 - Added list of times option - BOVY (NYU)
        """
        if isinstance(args[0],Orbit):
            if len(args) == 1:
                t= 0.
            else:
                t= args[1]
        else:
            raise IOError("Input to __call__ not understood; this has to be an Orbit instance with optional time")
        if isinstance(t,(list,nu.ndarray)): tlist= True
        else: tlist= False
        if kwargs.has_key('marginalizeVperp') and \
                kwargs['marginalizeVperp']:
            kwargs.pop('marginalizeVperp')
            if tlist: raise IOError("Input times to __call__ is a list; this is not supported in conjunction with marginalizeVperp")
            return self._call_marginalizevperp(args[0],**kwargs)
        elif kwargs.has_key('marginalizeVlos') and \
                kwargs['marginalizeVlos']:
            kwargs.pop('marginalizeVlos') 
            if tlist: raise IOError("Input times to __call__ is a list; this is not supported in conjunction with marginalizeVlos")
            return self._call_marginalizevlos(args[0],**kwargs)   
        #Integrate back
        if tlist:
            if self._to == t[0]:
                return [self._initdf(args[0])]
            ts= self._create_ts_tlist(t)
            o= args[0]
            #integrate orbit
            o.integrate(ts,self._pot)
            #Now evaluate the DF
            retval= []
            for time in t:
                retval.append(self._initdf(o(self._to+t[0]-time)))
            if isinstance(t,nu.ndarray): retval= nu.array(retval)
        else:
            if self._to == t:
                return self._initdf(args[0])
            ts= nu.linspace(t,self._to,_NTS)
            o= args[0]
            #integrate orbit
            o.integrate(ts,self._pot)
            #Now evaluate the DF
            retval= self._initdf(o(self._to-t))
            if nu.isnan(retval): print retval, o.vxvv, o(self._to-t).vxvv
        return retval

    def vmomentsurfacemass(self,R,n,m,t=0.,nsigma=None,deg=False,
                           epsrel=1.e-02,epsabs=1.e-05,phi=0.,
                           grid=None,gridpoints=101,returnGrid=False,
                           hierarchgrid=False,nlevels=2,
                           print_progress=False):
        """
        NAME:
           vmomentsurfacemass
        PURPOSE:
           calculate the an arbitrary moment of the velocity distribution 
           at (R,phi) times the surfacmass
        INPUT:
           R - radius at which to calculate the moment(/ro)
           phi= azimuth (rad unless deg=True)
           n - vR^n
           m - vT^m
           t= time at which to evaluate the DF (can be a list or ndarray)
        OPTIONAL INPUT:
           nsigma - number of sigma to integrate the velocities over (based on an estimate, so be generous)
           deg= azimuth is in degree (default=False)
           epsrel, epsabs - scipy.integrate keywords (the integration 
                            calculates the ration of this vmoment to that 
                            of the initial DF)
           grid= if set to True, build a grid and use that to evaluate 
                 integrals; if set to a grid-objects (such as returned by this 
                 procedure), use this grid; if this was created for a list of 
                 times, moments are calculated for each time
           gridpoints= number of points to use for the grid in 1D (default=101)
           returnGrid= if True, return the grid object (default=False)
           hierarchgrid= if True, use a hierarchical grid (default=False)
           nlevels= number of hierarchical levels for the hierarchical grid
           print_progress= if True, print progress updates
        OUTPUT:
           <vR^n vT^m  x surface-mass> at R,phi
        HISTORY:
           2011-03-30 - Written - Bovy (NYU)
        """
        #if we have already precalculated a grid, use that
        if not grid is None and isinstance(grid,evolveddiskdfGrid):
            if returnGrid:
                return (self._vmomentsurfacemassGrid(n,m,grid),grid)
            else:
                return self._vmomentsurfacemassGrid(n,m,grid)
        elif not grid is None \
                and isinstance(grid,evolveddiskdfHierarchicalGrid):
            if returnGrid:
                return (self._vmomentsurfacemassHierarchicalGrid(n,m,grid),
                        grid)
            else:
                return self._vmomentsurfacemassHierarchicalGrid(n,m,grid)
        #Otherwise we need to do some more work
        if deg: az= phi*_DEGTORAD
        else: az= phi
        if nsigma is None: nsigma= _NSIGMA
        sigmaR1= nu.sqrt(self._initdf.sigmaR2(R,phi=phi))
        sigmaT1= nu.sqrt(self._initdf.sigmaT2(R,phi=phi))
        meanvR= self._initdf.meanvR(R,phi=phi)
        meanvT= self._initdf.meanvT(R,phi=phi)
        if not grid is None and isinstance(grid,bool) and grid:
            if not hierarchgrid:
                grido= self._buildvgrid(R,az,nsigma,t,
                                        sigmaR1,sigmaT1,meanvR,meanvT,
                                        gridpoints,print_progress)
                if returnGrid:
                    return (self._vmomentsurfacemassGrid(n,m,grido),grido)
                else:
                    return self._vmomentsurfacemassGrid(n,m,grido)
            else: #hierarchical grid
                grido= evolveddiskdfHierarchicalGrid(self,R,az,nsigma,t,
                                                     sigmaR1,sigmaT1,meanvR,
                                                     meanvT,
                                                     gridpoints,nlevels,
                                                     print_progress=print_progress)
                if returnGrid:
                    return (self._vmomentsurfacemassHierarchicalGrid(n,m,
                                                                     grido),
                            grido)
                else:
                    return self._vmomentsurfacemassHierarchicalGrid(n,m,grido)
        #Calculate the initdf moment and then calculate the ratio
        initvmoment= self._initdf.vmomentsurfacemass(R,n,m,nsigma=nsigma,
                                                     phi=phi)
        if initvmoment == 0.: initvmoment= 1.
        norm= sigmaR1**(n+1)*sigmaT1**(m+1)*initvmoment
        if isinstance(t,(list,nu.ndarray)):
            raise IOError("list of times is only supported with grid-based calculation")            
        return dblquad(_vmomentsurfaceIntegrand,
                                 meanvT/sigmaT1-nsigma,
                                 meanvT/sigmaT1+nsigma,
                                 lambda x: meanvR/sigmaR1
                                 -nu.sqrt(nsigma**2.-(x-meanvT/sigmaT1)**2.),
                                 lambda x: meanvR/sigmaR1
                                 +nu.sqrt(nsigma**2.-(x-meanvT/sigmaT1)**2.), 
                                 (R,az,self,n,m,sigmaR1,sigmaT1,t,initvmoment),
                                 epsrel=epsrel,epsabs=epsabs)[0]*norm

    def vertexdev(self,R,t=0.,nsigma=None,deg=False,
                  epsrel=1.e-02,epsabs=1.e-05,phi=0.,
                  grid=None,gridpoints=101,returnGrid=False,
                  sigmaR2=None,sigmaT2=None,sigmaRT=None,surfacemass=None,
                  hierarchgrid=False,nlevels=2):
        """
        NAME:
           vertexdev
        PURPOSE:
           calculate the vertex deviation of the velocity distribution 
           at (R,phi)
        INPUT:
           R - radius at which to calculate the moment(/ro)
           phi= azimuth (rad unless deg=True)
           t= time at which to evaluate the DF (can be a list)
        OPTIONAL INPUT:
           sigmaR2, sigmaT2, sigmaRT= if set the vertex deviation is simply 
                                      calculated using these
           nsigma - number of sigma to integrate the velocities over (based on an estimate, so be generous)
           deg= azimuth is in degree (default=False)
           epsrel, epsabs - scipy.integrate keywords (the integration 
                            calculates the ration of this vmoment to that 
                            of the initial DF)
           grid= if set to True, build a grid and use that to evaluate 
                 integrals; if set to a grid-objects (such as returned by this 
                 procedure), use this grid
           gridpoints= number of points to use for the grid in 1D (default=101)
           returnGrid= if True, return the grid object (default=False)
           hierarchgrid= if True, use a hierarchical grid (default=False)
           nlevels= number of hierarchical levels for the hierarchical grid
        OUTPUT:
           vertex deviation in degree
        HISTORY:
           2011-03-31 - Written - Bovy (NYU)
        """
        #The following aren't actually the moments, but they are the moments
        #times the surface-mass density; that drops out
        if isinstance(grid,evolveddiskdfGrid) or \
                isinstance(grid,evolveddiskdfHierarchicalGrid):
            grido= grid           
        elif (sigmaR2 is None or sigmaT2 is None or sigmaRT is None) \
                and isinstance(grid,bool) and grid:
            #Precalculate the grid
            (sigmaR2,grido)= self.vmomentsurfacemass(R,2,0,deg=deg,t=t,
                                                     nsigma=nsigma,phi=phi,
                                                     epsrel=epsrel,
                                                     epsabs=epsabs,grid=grid,
                                                     gridpoints=gridpoints,
                                                     returnGrid=True,
                                                     hierarchgrid=hierarchgrid,
                                                     nlevels=nlevels)
        else:
            grido= False
        if surfacemass is None:
            surfacemass= self.vmomentsurfacemass(R,0,0,deg=deg,t=t,phi=phi,
                                                 nsigma=nsigma,epsrel=epsrel,
                                                 epsabs=epsabs,grid=grido,
                                                 gridpoints=gridpoints,
                                                 returnGrid=False,
                                                 hierarchgrid=hierarchgrid,
                                                 nlevels=nlevels)
        if sigmaR2 is None:
            sigmaR2= self.sigmaR2(R,deg=deg,t=t,phi=phi,
                                  nsigma=nsigma,epsrel=epsrel,
                                  epsabs=epsabs,grid=grido,
                                  gridpoints=gridpoints,
                                  returnGrid=False,
                                  hierarchgrid=hierarchgrid,
                                  nlevels=nlevels)/surfacemass
        if sigmaT2 is None:
            sigmaT2= self.sigmaT2(R,deg=deg,t=t,phi=phi,
                                  nsigma=nsigma,epsrel=epsrel,
                                  epsabs=epsabs,grid=grido,
                                  gridpoints=gridpoints,
                                  returnGrid=False,
                                  hierarchgrid=hierarchgrid,
                                  nlevels=nlevels)/surfacemass
        if sigmaRT is None:
            sigmaRT= self.sigmaRT(R,deg=deg,t=t,phi=phi,
                                  nsigma=nsigma,epsrel=epsrel,
                                  epsabs=epsabs,grid=grido,
                                  gridpoints=gridpoints,
                                  returnGrid=False,
                                  hierarchgrid=hierarchgrid,
                                  nlevels=nlevels)/surfacemass
        if returnGrid and ((isinstance(grid,bool) and grid) or 
                           isinstance(grid,evolveddiskdfGrid) or
                           isinstance(grid,evolveddiskdfHierarchicalGrid)):
            return (-nu.arctan(2.*sigmaRT/(sigmaR2-sigmaT2))/2.*_RADTODEG,
                     grido)
        else:
            return -nu.arctan(2.*sigmaRT/(sigmaR2-sigmaT2))/2.*_RADTODEG

    def meanvR(self,R,t=0.,nsigma=None,deg=False,phi=0.,
               epsrel=1.e-02,epsabs=1.e-05,
               grid=None,gridpoints=101,returnGrid=False,
               surfacemass=None,
               hierarchgrid=False,nlevels=2):
        """
        NAME:
           meanvR
        PURPOSE:
           calculate the mean vR of the velocity distribution 
           at (R,phi)
        INPUT:
           R - radius at which to calculate the moment(/ro)
           phi= azimuth (rad unless deg=True)
           t= time at which to evaluate the DF
        OPTIONAL INPUT:
           surfacemass= if set use this pre-calculated surfacemass
           nsigma - number of sigma to integrate the velocities over (based on an estimate, so be generous)
           deg= azimuth is in degree (default=False)
           epsrel, epsabs - scipy.integrate keywords (the integration 
                            calculates the ration of this vmoment to that 
                            of the initial DF)
           grid= if set to True, build a grid and use that to evaluate 
                 integrals; if set to a grid-objects (such as returned by this 
                 procedure), use this grid
           gridpoints= number of points to use for the grid in 1D (default=101)
           returnGrid= if True, return the grid object (default=False)
           hierarchgrid= if True, use a hierarchical grid (default=False)
           nlevels= number of hierarchical levels for the hierarchical grid
        OUTPUT:
           mean vR
        HISTORY:
           2011-03-31 - Written - Bovy (NYU)
        """
        if isinstance(grid,evolveddiskdfGrid) or \
                isinstance(grid,evolveddiskdfHierarchicalGrid):
            grido= grid           
            vmomentR= self.vmomentsurfacemass(R,1,0,deg=deg,t=t,phi=phi,
                                              nsigma=nsigma,
                                              epsrel=epsrel,
                                              epsabs=epsabs,grid=grid,
                                              gridpoints=gridpoints,
                                              returnGrid=False,
                                              hierarchgrid=hierarchgrid,
                                              nlevels=nlevels)
        elif isinstance(grid,bool) and grid:
            #Precalculate the grid
            (vmomentR,grido)= self.vmomentsurfacemass(R,1,0,deg=deg,t=t,
                                                      nsigma=nsigma,phi=phi,
                                                      epsrel=epsrel,
                                                      epsabs=epsabs,grid=grid,
                                                      gridpoints=gridpoints,
                                                      returnGrid=True,
                                                      hierarchgrid=hierarchgrid,
                                                      nlevels=nlevels)
        else:
            grido= False
            vmomentR= self.vmomentsurfacemass(R,1,0,deg=deg,t=t,phi=phi,
                                              nsigma=nsigma,
                                              epsrel=epsrel,
                                              epsabs=epsabs,grid=grid,
                                              gridpoints=gridpoints,
                                              returnGrid=False,
                                              hierarchgrid=hierarchgrid,
                                              nlevels=nlevels)
        if surfacemass is None:
            surfacemass= self.vmomentsurfacemass(R,0,0,deg=deg,t=t,phi=phi,
                                                 nsigma=nsigma,epsrel=epsrel,
                                                 epsabs=epsabs,grid=grido,
                                                 gridpoints=gridpoints,
                                                 returnGrid=False,
                                                 hierarchgrid=hierarchgrid,
                                                 nlevels=nlevels)
        out= vmomentR/surfacemass
        if returnGrid and ((isinstance(grid,bool) and grid) or 
                           isinstance(grid,evolveddiskdfGrid) or
                           isinstance(grid,evolveddiskdfHierarchicalGrid)):
            return (out,grido)
        else:
            return out

    def meanvT(self,R,t=0.,nsigma=None,deg=False,phi=0.,
               epsrel=1.e-02,epsabs=1.e-05,
               grid=None,gridpoints=101,returnGrid=False,
               surfacemass=None,
               hierarchgrid=False,nlevels=2):
        """
        NAME:
           meanvT
        PURPOSE:
           calculate the mean vT of the velocity distribution 
           at (R,phi)
        INPUT:
           R - radius at which to calculate the moment(/ro)
           phi= azimuth (rad unless deg=True)
           t= time at which to evaluate the DF
        OPTIONAL INPUT:
           surfacemass= if set use this pre-calculated surfacemass
           nsigma - number of sigma to integrate the velocities over (based on an estimate, so be generous)
           deg= azimuth is in degree (default=False)
           epsrel, epsabs - scipy.integrate keywords (the integration 
                            calculates the ration of this vmoment to that 
                            of the initial DF)
           grid= if set to True, build a grid and use that to evaluate 
                 integrals; if set to a grid-objects (such as returned by this 
                 procedure), use this grid
           gridpoints= number of points to use for the grid in 1D (default=101)
           returnGrid= if True, return the grid object (default=False)
           hierarchgrid= if True, use a hierarchical grid (default=False)
           nlevels= number of hierarchical levels for the hierarchical grid
        OUTPUT:
           mean vT
        HISTORY:
           2011-03-31 - Written - Bovy (NYU)
        """
        if isinstance(grid,evolveddiskdfGrid) or \
                isinstance(grid,evolveddiskdfHierarchicalGrid):
            grido= grid           
            vmomentT= self.vmomentsurfacemass(R,0,1,deg=deg,t=t,
                                              nsigma=nsigma,phi=phi,
                                              epsrel=epsrel,
                                              epsabs=epsabs,grid=grid,
                                              gridpoints=gridpoints,
                                              returnGrid=False,
                                              hierarchgrid=hierarchgrid,
                                              nlevels=nlevels)
        elif isinstance(grid,bool) and grid:
            #Precalculate the grid
            (vmomentT,grido)= self.vmomentsurfacemass(R,0,1,deg=deg,t=t,
                                                      nsigma=nsigma,phi=phi,
                                                      epsrel=epsrel,
                                                      epsabs=epsabs,grid=grid,
                                                      gridpoints=gridpoints,
                                                      returnGrid=True,
                                                      hierarchgrid=hierarchgrid,
                                                      nlevels=nlevels)
        else:
            grido= False
            vmomentT= self.vmomentsurfacemass(R,0,1,deg=deg,t=t,
                                              nsigma=nsigma,phi=phi,
                                              epsrel=epsrel,
                                              epsabs=epsabs,grid=grid,
                                              gridpoints=gridpoints,
                                              returnGrid=False,
                                              hierarchgrid=hierarchgrid,
                                              nlevels=nlevels)
        if surfacemass is None:
            surfacemass= self.vmomentsurfacemass(R,0,0,deg=deg,t=t,phi=phi,
                                                 nsigma=nsigma,epsrel=epsrel,
                                                 epsabs=epsabs,grid=grido,
                                                 gridpoints=gridpoints,
                                                 returnGrid=False,
                                                 hierarchgrid=hierarchgrid,
                                                 nlevels=nlevels)
        out= vmomentT/surfacemass
        if returnGrid and ((isinstance(grid,bool) and grid) or 
                           isinstance(grid,evolveddiskdfGrid) or
                           isinstance(grid,evolveddiskdfHierarchicalGrid)):
            return (out,grido)
        else:
            return out

    def sigmaR2(self,R,t=0.,nsigma=None,deg=False,phi=0.,
                epsrel=1.e-02,epsabs=1.e-05,
                grid=None,gridpoints=101,returnGrid=False,
                surfacemass=None,meanvR=None,
                hierarchgrid=False,nlevels=2):
        """
        NAME:
           sigmaR2
        PURPOSE:
           calculate the radial variance of the velocity distribution 
           at (R,phi)
        INPUT:
           R - radius at which to calculate the moment(/ro)
           phi= azimuth (rad unless deg=True)
           t= time at which to evaluate the DF
        OPTIONAL INPUT:
           surfacemass, meanvR= if set use this pre-calculated surfacemass and
                                mean vR
           nsigma - number of sigma to integrate the velocities over (based on an estimate, so be generous)
           deg= azimuth is in degree (default=False)
           epsrel, epsabs - scipy.integrate keywords (the integration 
                            calculates the ration of this vmoment to that 
                            of the initial DF)
           grid= if set to True, build a grid and use that to evaluate 
                 integrals; if set to a grid-objects (such as returned by this 
                 procedure), use this grid
           gridpoints= number of points to use for the grid in 1D (default=101)
           returnGrid= if True, return the grid object (default=False)
           hierarchgrid= if True, use a hierarchical grid (default=False)
           nlevels= number of hierarchical levels for the hierarchical grid
        OUTPUT:
           variance of vR
        HISTORY:
           2011-03-31 - Written - Bovy (NYU)
        """
        #The following aren't actually the moments, but they are the moments
        #times the surface-mass density
        if isinstance(grid,evolveddiskdfGrid) or \
                isinstance(grid,evolveddiskdfHierarchicalGrid):
            grido= grid
            sigmaR2= self.vmomentsurfacemass(R,2,0,deg=deg,t=t,phi=phi,
                                             nsigma=nsigma,
                                             epsrel=epsrel,
                                             epsabs=epsabs,grid=grido,
                                             gridpoints=gridpoints,
                                             returnGrid=False,
                                             hierarchgrid=hierarchgrid,
                                             nlevels=nlevels)
        elif (meanvR is None or surfacemass is None ) \
                and isinstance(grid,bool) and grid:
            #Precalculate the grid
            (sigmaR2,grido)= self.vmomentsurfacemass(R,2,0,deg=deg,t=t,
                                                     nsigma=nsigma,phi=phi,
                                                     epsrel=epsrel,
                                                     epsabs=epsabs,grid=grid,
                                                     gridpoints=gridpoints,
                                                     returnGrid=True,
                                                     hierarchgrid=hierarchgrid,
                                                     nlevels=nlevels)
        else:
            grido= False
            sigmaR2= self.vmomentsurfacemass(R,2,0,deg=deg,t=t,
                                             nsigma=nsigma,phi=phi,
                                             epsrel=epsrel,
                                             epsabs=epsabs,grid=grido,
                                             gridpoints=gridpoints,
                                             returnGrid=False,
                                             hierarchgrid=hierarchgrid,
                                             nlevels=nlevels)
        if surfacemass is None:
            surfacemass= self.vmomentsurfacemass(R,0,0,deg=deg,t=t,phi=phi,
                                                 nsigma=nsigma,epsrel=epsrel,
                                                 epsabs=epsabs,grid=grido,
                                                 gridpoints=gridpoints,
                                                 returnGrid=False,
                                                 hierarchgrid=hierarchgrid,
                                                 nlevels=nlevels)
        if meanvR is None:
            meanvR= self.vmomentsurfacemass(R,1,0,deg=deg,t=t,phi=phi,
                                             nsigma=nsigma,epsrel=epsrel,
                                             epsabs=epsabs,grid=grido,
                                             gridpoints=gridpoints,
                                             returnGrid=False,
                                            hierarchgrid=hierarchgrid,
                                            nlevels=nlevels)/surfacemass
        out= sigmaR2/surfacemass-meanvR**2.
        if returnGrid and ((isinstance(grid,bool) and grid) or 
                           isinstance(grid,evolveddiskdfGrid) or
                           isinstance(grid,evolveddiskdfHierarchicalGrid)):
            return (out,grido)
        else:
            return out

    def sigmaT2(self,R,t=0.,nsigma=None,deg=False,phi=0.,
                epsrel=1.e-02,epsabs=1.e-05,
                grid=None,gridpoints=101,returnGrid=False,
                surfacemass=None,meanvT=None,
                hierarchgrid=False,nlevels=2):
        """
        NAME:
           sigmaT2
        PURPOSE:
           calculate the tangential variance of the velocity distribution 
           at (R,phi)
        INPUT:
           R - radius at which to calculate the moment(/ro)
           phi= azimuth (rad unless deg=True)
           t= time at which to evaluate the DF
        OPTIONAL INPUT:
           surfacemass, meanvT= if set use this pre-calculated surfacemass
                                and mean tangential velocity
           nsigma - number of sigma to integrate the velocities over (based on an estimate, so be generous)
           deg= azimuth is in degree (default=False)
           epsrel, epsabs - scipy.integrate keywords (the integration 
                            calculates the ration of this vmoment to that 
                            of the initial DF)
           grid= if set to True, build a grid and use that to evaluate 
                 integrals; if set to a grid-objects (such as returned by this 
                 procedure), use this grid
           gridpoints= number of points to use for the grid in 1D (default=101)
           returnGrid= if True, return the grid object (default=False)
           hierarchgrid= if True, use a hierarchical grid (default=False)
           nlevels= number of hierarchical levels for the hierarchical grid
        OUTPUT:
           variance of vT
        HISTORY:
           2011-03-31 - Written - Bovy (NYU)
        """
        if isinstance(grid,evolveddiskdfGrid) or \
                isinstance(grid,evolveddiskdfHierarchicalGrid):
            grido= grid
            sigmaT2= self.vmomentsurfacemass(R,0,2,deg=deg,t=t,phi=phi,
                                             nsigma=nsigma,
                                             epsrel=epsrel,
                                             epsabs=epsabs,grid=grido,
                                             gridpoints=gridpoints,
                                             returnGrid=False,
                                             hierarchgrid=hierarchgrid,
                                             nlevels=nlevels)
        elif (meanvT is None or surfacemass is None ) \
                and isinstance(grid,bool) and grid:
            #Precalculate the grid
            (sigmaT2,grido)= self.vmomentsurfacemass(R,0,2,deg=deg,t=t,
                                                     nsigma=nsigma,phi=phi,
                                                     epsrel=epsrel,
                                                     epsabs=epsabs,grid=grid,
                                                     gridpoints=gridpoints,
                                                     returnGrid=True,
                                                     hierarchgrid=hierarchgrid,
                                                     nlevels=nlevels)
        else:
            grido= False
            sigmaT2= self.vmomentsurfacemass(R,0,2,deg=deg,t=t,
                                             nsigma=nsigma,phi=phi,
                                             epsrel=epsrel,
                                             epsabs=epsabs,grid=grido,
                                             gridpoints=gridpoints,
                                             returnGrid=False,
                                             hierarchgrid=hierarchgrid,
                                             nlevels=nlevels)
        if surfacemass is None:
            surfacemass= self.vmomentsurfacemass(R,0,0,deg=deg,t=t,phi=phi,
                                                 nsigma=nsigma,epsrel=epsrel,
                                                 epsabs=epsabs,grid=grido,
                                                 gridpoints=gridpoints,
                                                 returnGrid=False,
                                                 hierarchgrid=hierarchgrid,
                                                 nlevels=nlevels)
        if meanvT is None:
            meanvT= self.vmomentsurfacemass(R,0,1,deg=deg,t=t,phi=phi,
                                            nsigma=nsigma,epsrel=epsrel,
                                            epsabs=epsabs,grid=grido,
                                            gridpoints=gridpoints,
                                            returnGrid=False,
                                            hierarchgrid=hierarchgrid,
                                            nlevels=nlevels)/surfacemass
        out= sigmaT2/surfacemass-meanvT**2.
        if returnGrid and ((isinstance(grid,bool) and grid) or 
                           isinstance(grid,evolveddiskdfGrid) or
                           isinstance(grid,evolveddiskdfHierarchicalGrid)):
            return (out,grido)
        else:
            return out

    def sigmaRT(self,R,t=0.,nsigma=None,deg=False,
                epsrel=1.e-02,epsabs=1.e-05,phi=0.,
                grid=None,gridpoints=101,returnGrid=False,
                surfacemass=None,meanvR=None,meanvT=None,
                hierarchgrid=False,nlevels=2):
        """
        NAME:
           sigmaRT
        PURPOSE:
           calculate the radial-tangential co-variance of the velocity 
           distribution at (R,phi)
        INPUT:
           R - radius at which to calculate the moment(/ro)
           phi= azimuth (rad unless deg=True)
           t= time at which to evaluate the DF
        OPTIONAL INPUT:
           surfacemass, meanvR, meavT= if set use this pre-calculated 
                                       surfacemass and mean vR and vT
           nsigma - number of sigma to integrate the velocities over (based on an estimate, so be generous)
           deg= azimuth is in degree (default=False)
           epsrel, epsabs - scipy.integrate keywords (the integration 
                            calculates the ration of this vmoment to that 
                            of the initial DF)
           grid= if set to True, build a grid and use that to evaluate 
                 integrals; if set to a grid-objects (such as returned by this 
                 procedure), use this grid
           gridpoints= number of points to use for the grid in 1D (default=101)
           returnGrid= if True, return the grid object (default=False)
           hierarchgrid= if True, use a hierarchical grid (default=False)
           nlevels= number of hierarchical levels for the hierarchical grid
        OUTPUT:
           covariance of vR and vT
        HISTORY:
           2011-03-31 - Written - Bovy (NYU)
        """
        #The following aren't actually the moments, but they are the moments
        #times the surface-mass density
        if isinstance(grid,evolveddiskdfGrid) or \
                isinstance(grid,evolveddiskdfHierarchicalGrid):
            grido= grid
            sigmaRT= self.vmomentsurfacemass(R,1,1,deg=deg,t=t,phi=phi,
                                             nsigma=nsigma,
                                             epsrel=epsrel,
                                             epsabs=epsabs,grid=grido,
                                             gridpoints=gridpoints,
                                             returnGrid=False,
                                             hierarchgrid=hierarchgrid,
                                             nlevels=nlevels)
        elif (meanvR is None or surfacemass is None ) \
                and isinstance(grid,bool) and grid:
            #Precalculate the grid
            (sigmaRT,grido)= self.vmomentsurfacemass(R,1,1,deg=deg,t=t,
                                                     nsigma=nsigma,phi=phi,
                                                     epsrel=epsrel,
                                                     epsabs=epsabs,grid=grid,
                                                     gridpoints=gridpoints,
                                                     returnGrid=True,
                                                     hierarchgrid=hierarchgrid,
                                                     nlevels=nlevels)
        else:
            grido= False
            sigmaRT= self.vmomentsurfacemass(R,1,1,deg=deg,t=t,
                                             nsigma=nsigma,phi=phi,
                                             epsrel=epsrel,
                                             epsabs=epsabs,grid=grido,
                                             gridpoints=gridpoints,
                                             returnGrid=False,
                                             hierarchgrid=hierarchgrid,
                                             nlevels=nlevels)
        if surfacemass is None:
            surfacemass= self.vmomentsurfacemass(R,0,0,deg=deg,t=t,phi=phi,
                                                 nsigma=nsigma,epsrel=epsrel,
                                                 epsabs=epsabs,grid=grido,
                                                 gridpoints=gridpoints,
                                                 returnGrid=False,
                                                 hierarchgrid=hierarchgrid,
                                                 nlevels=nlevels)
        if meanvR is None:
            meanvR= self.vmomentsurfacemass(R,1,0,deg=deg,t=t,phi=phi,
                                            nsigma=nsigma,epsrel=epsrel,
                                            epsabs=epsabs,grid=grido,
                                            gridpoints=gridpoints,
                                            returnGrid=False,
                                            hierarchgrid=hierarchgrid,
                                            nlevels=nlevels)/surfacemass
        if meanvT is None:
            meanvT= self.vmomentsurfacemass(R,0,1,deg=deg,t=t,phi=phi,
                                            nsigma=nsigma,epsrel=epsrel,
                                            epsabs=epsabs,grid=grido,
                                            gridpoints=gridpoints,
                                            returnGrid=False,
                                            hierarchgrid=hierarchgrid,
                                            nlevels=nlevels)/surfacemass
        out= sigmaRT/surfacemass-meanvR*meanvT
        if returnGrid and ((isinstance(grid,bool) and grid) or 
                           isinstance(grid,evolveddiskdfGrid) or
                           isinstance(grid,evolveddiskdfHierarchicalGrid)):
            return (out,grido)
        else:
            return out

    def _vmomentsurfacemassGrid(self,n,m,grid):
        """Internal function to evaluate vmomentsurfacemass using a grid 
        rather than direct integration"""
        if len(grid.df.shape) == 3: tlist= True
        else: tlist= False
        if tlist: 
            nt= grid.df.shape[2]
            out= []
            for ii in range(nt):
                out.append(nu.dot(grid.vRgrid**n,nu.dot(grid.df[:,:,ii],grid.vTgrid**m))*\
                    (grid.vRgrid[1]-grid.vRgrid[0])*(grid.vTgrid[1]-grid.vTgrid[0]))
            return nu.array(out)
        else:
            return nu.dot(grid.vRgrid**n,nu.dot(grid.df,grid.vTgrid**m))*\
                (grid.vRgrid[1]-grid.vRgrid[0])*(grid.vTgrid[1]-grid.vTgrid[0])
        
    def _buildvgrid(self,R,phi,nsigma,t,sigmaR1,sigmaT1,meanvR,meanvT,
                    gridpoints,print_progress):
        """Internal function to grid the vDF at a given location"""
        out= evolveddiskdfGrid()
        out.sigmaR1= sigmaR1
        out.sigmaT1= sigmaT1
        out.meanvR= meanvR
        out.meanvT= meanvT
        out.vRgrid= nu.linspace(meanvR-nsigma*sigmaR1,meanvR+nsigma*sigmaR1,
                                gridpoints)
        out.vTgrid= nu.linspace(meanvT-nsigma*sigmaT1,meanvT+nsigma*sigmaT1,
                                gridpoints)
        if isinstance(t,(list,nu.ndarray)):
            nt= len(t)
            out.df= nu.zeros((gridpoints,gridpoints,nt))
            for ii in range(gridpoints):
                for jj in range(gridpoints):
                    if print_progress:
                        sys.stdout.write('\r'+"Velocity gridpoint %i out of %i" % \
                                             (jj+ii*gridpoints+1,gridpoints*gridpoints))
                        sys.stdout.flush()
                    thiso= Orbit([R,out.vRgrid[ii],out.vTgrid[jj],phi])
                    out.df[ii,jj,:]= self(thiso,nu.array(t).flatten())
                    out.df[ii,jj,nu.isnan(out.df[ii,jj,:])]= 0. #BOVY: for now
            if print_progress: sys.stdout.write('\n')
        else:
            out.df= nu.zeros((gridpoints,gridpoints))
            for ii in range(gridpoints):
                for jj in range(gridpoints):
                    if print_progress:
                        sys.stdout.write('\r'+"Velocity gridpoint %i out of %i" % \
                                             (jj+ii*gridpoints+1,gridpoints*gridpoints))
                        sys.stdout.flush()
                    thiso= Orbit([R,out.vRgrid[ii],out.vTgrid[jj],phi])
                    out.df[ii,jj]= self(thiso,t)
                    if nu.isnan(out.df[ii,jj]): out.df[ii,jj]= 0. #BOVY: for now
            if print_progress: sys.stdout.write('\n')
        return out

    def _create_ts_tlist(self,t):
        #Check input
        if not all(t == sorted(t,reverse=True)): raise IOError("List of times has to be sorted in descending order")
        #Initialize
        ts= nu.linspace(t[0],self._to,_NTS)
        #Add other t
        ts= list(ts)
        ts.extend([self._to+t[0]-ti for ti in t[1:len(t)]])
        #sort
        ts.sort(reverse=True)
        return nu.array(ts)

    def _call_marginalizevperp(self,o,**kwargs):
        """Call the DF, marginalizing over perpendicular velocity"""
        #Get d, l, vlos
        d= o.dist(ro=1.,obs=[1.,0.,0.])
        l= o.ll(obs=[1.,0.,0.],ro=1.)*_DEGTORAD
        vlos= o.vlos(ro=1.,vo=1.,obs=[1.,0.,0.,0.,0.,0.])
        R= o.R()
        phi= o.phi()
        #Get local circular velocity, projected onto the los
        if isinstance(self._pot,list):
            vcirc= calcRotcurve([p for p in self._pot if not p.isNonAxi],R)[0]
        else:
            vcirc= calcRotcurve(self._pot,R)[0]
        vcirclos= vcirc*math.sin(phi+l)
        print R, vlos, vlos-vcirclos
        #Marginalize
        alphalos= phi+l
        if not kwargs.has_key('nsigma') or (kwargs.has_key('nsigma') and \
                                                kwargs['nsigma'] is None):
            nsigma= _NSIGMA
        else:
            nsigma= kwargs['nsigma']
        if kwargs.has_key('nsigma'): kwargs.pop('nsigma')
        #BOVY: add asymmetric drift here?
        if math.fabs(math.sin(alphalos)) < math.sqrt(1./2.):
            sigmaR1= nu.sqrt(self._initdf.sigmaT2(R,phi=phi)) #Slight abuse
            cosalphalos= math.cos(alphalos)
            tanalphalos= math.tan(alphalos)
            return integrate.quad(_marginalizeVperpIntegrandSinAlphaSmall,
                                  -nsigma,nsigma,
                                  args=(self,R,cosalphalos,tanalphalos,
                                        vlos-vcirclos,vcirc,
                                        sigmaR1,phi),
                                  **kwargs)[0]/math.fabs(cosalphalos)
        else:
            sigmaR1= nu.sqrt(self._initdf.sigmaR2(R,phi=phi))
            sinalphalos= math.sin(alphalos)
            cotalphalos= 1./math.tan(alphalos)
            return integrate.quad(_marginalizeVperpIntegrandSinAlphaLarge,
                                  -nsigma,nsigma,
                                  args=(self,R,sinalphalos,cotalphalos,
                                        vlos-vcirclos,vcirc,sigmaR1,phi),
                                  **kwargs)[0]/math.fabs(sinalphalos)
        
    def _call_marginalizevlos(self,o,**kwargs):
        """Call the DF, marginalizing over line-of-sight velocity"""
        #Get d, l, vperp
        d= o.dist(ro=1.,obs=[1.,0.,0.])
        l= o.ll(obs=[1.,0.,0.],ro=1.)*_DEGTORAD
        vperp= o.vll(ro=1.,vo=1.,obs=[1.,0.,0.,0.,0.,0.])
        R= o.R()
        phi= o.phi()
        #Get local circular velocity, projected onto the perpendicular 
        #direction
        if isinstance(self._pot,list):
            vcirc= calcRotcurve([p for p in self._pot if not p.isNonAxi],R)[0]
        else:
            vcirc= calcRotcurve(self._pot,R)[0]
        vcirclos= vcirc*math.sin(phi+l)
        #Marginalize
        alphaperp= math.pi/2.+phi+l
        if not kwargs.has_key('nsigma') or (kwargs.has_key('nsigma') and \
                                                kwargs['nsigma'] is None):
            nsigma= _NSIGMA
        else:
            nsigma= kwargs['nsigma']
        if kwargs.has_key('nsigma'): kwargs.pop('nsigma')
        #BOVY: Put asymmetric drift in here?
        if math.fabs(math.sin(alphaperp)) < math.sqrt(1./2.):
            sigmaR1= nu.sqrt(self._initdf.sigmaT2(R,phi=phi)) #slight abuse
            cosalphaperp= math.cos(alphaperp)
            tanalphaperp= math.tan(alphaperp)
            #we can reuse the VperpIntegrand, since it is just another angle
            return integrate.quad(_marginalizeVperpIntegrandSinAlphaSmall,
                                  -nsigma,nsigma,
                                  args=(self,R,cosalphaperp,tanalphaperp,
                                        vperp-vcircperp,vcirc,
                                        sigmaR1,phi),
                                  **kwargs)[0]/math.fabs(cosalphaperp)
        else:
            sigmaR1= nu.sqrt(self._initdf.sigmaR2(R,phi=phi))
            sinalphaperp= math.sin(alphaperp)
            cotalphaperp= 1./math.tan(alphaperp)
            #we can reuse the VperpIntegrand, since it is just another angle
            return integrate.quad(_marginalizeVperpIntegrandSinAlphaLarge,
                                  -nsigma,nsigma,
                                  args=(self,R,sinalphaperp,cotalphaperp,
                                        vperp-vcircperp,vcirc,sigmaR1,phi),
                                  **kwargs)[0]/math.fabs(sinalphaperp)

    def _vmomentsurfacemassHierarchicalGrid(self,n,m,grid):
        """Internal function to evaluate vmomentsurfacemass using a 
        hierarchical grid rather than direct integration,
        rather unnecessary""" 
        return grid(n,m)       
        
class evolveddiskdfGrid:
    """(not quite) Empty class since it is only used to store some stuff"""
    def __init__(self):
        return None

    def plot(self,tt=0):
        """
        NAME:
           plot
        PURPOSE:
           plot the velocity distribution
        INPUT:
           t= optional time index
        OUTPUT:
           plot of velocity distribution to output device
        HISTORY:
           2011-06-27 - Written - Bovy (NYU)
        """
        xrange= [self.vRgrid[0],self.vRgrid[len(self.vRgrid)-1]]
        yrange= [self.vTgrid[0],self.vTgrid[len(self.vTgrid)-1]]
        if len(self.df.shape) == 3:
            plotthis= self.df[:,:,tt]
        else:
            plotthis= self.df
        bovy_plot.bovy_dens2d(plotthis.T,cmap='gist_yarg',origin='lower',
                              aspect=(xrange[1]-xrange[0])/\
                                  (yrange[1]-yrange[0]),
                              extent=[xrange[0],xrange[1],
                                      yrange[0],yrange[1]],
                              xlabel=r'$v_R / v_0$',
                              ylabel=r'$v_T / v_0$')

class evolveddiskdfHierarchicalGrid:
    """Class that holds a hierarchical velocity grid"""
    def __init__(self,edf,R,phi,nsigma,t,sigmaR1,sigmaT1,meanvR,meanvT,
                 gridpoints,nlevels,upperdxdy=None,print_progress=False,
                 nlevelsTotal=None):
        """
        NAME:
            __init__
        PURPOSE:
            Initialize a hierarchical grid
        INPUT:
            edf - evolveddiskdf instance
            R - Radius
            phi- azimuth
            nsigma - number of sigma to integrate over
            t- time
            sigmaR1 - radial dispersion
            sigmaT1 - tangential dispersion
            meanvR - mean of radial velocity
            meanvT - mean of tangential velocity
            gridpoints- number of gridpoints
            nlevels- number of levels to build
            upperdxdy= area element of previous hierarchical level
            print_progress= if True, print progress on building the grid
        OUTPUT:
           object
        HISTORY:
           2011-04-21 - Written - Bovy (NYU)
        """
        self.sigmaR1= sigmaR1
        self.sigmaT1= sigmaT1
        self.meanvR= meanvR
        self.meanvT= meanvT
        self.gridpoints= gridpoints
        self.vRgrid= nu.linspace(self.meanvR-nsigma*self.sigmaR1,
                                 self.meanvR+nsigma*self.sigmaR1,
                                 self.gridpoints)
        self.vTgrid= nu.linspace(self.meanvT-nsigma*self.sigmaT1,
                                 self.meanvT+nsigma*self.sigmaT1,
                                 self.gridpoints)
        self.t= t
        if nlevelsTotal is None:
            nlevelsTotal= nlevels
        self.nlevels= nlevels
        self.nlevelsTotal= nlevelsTotal
        if isinstance(t,(list,nu.ndarray)):
            nt= len(t)
            self.df= nu.zeros((gridpoints,gridpoints,nt))
            dxdy= (self.vRgrid[1]-self.vRgrid[0])\
                *(self.vTgrid[1]-self.vTgrid[0])
            if nlevels > 0:
                xsubmin= int(gridpoints)/4
                xsubmax= gridpoints-int(gridpoints)/4
            else:
                xsubmin= gridpoints
                xsubmax= 0
            ysubmin, ysubmax= xsubmin, xsubmax
            for ii in range(gridpoints):
                for jj in range(gridpoints):
                    if print_progress:
                        sys.stdout.write('\r'+"Velocity gridpoint %i out of %i" % \
                                             (jj+ii*gridpoints+1,gridpoints*gridpoints))
                        sys.stdout.flush()
                    #If this is part of a subgrid, ignore
                    if nlevels > 1 and ii >= xsubmin and ii < xsubmax \
                            and jj >= ysubmin and jj < ysubmax:
                        continue
                    thiso= Orbit([R,self.vRgrid[ii],self.vTgrid[jj],phi])
                    self.df[ii,jj,:]= edf(thiso,nu.array(t).flatten())
                    self.df[ii,jj,nu.isnan(self.df[ii,jj,:])]= 0.#BOVY: for now
                    #Multiply in area, somewhat tricky for edge objects
                    if upperdxdy is None or (ii != 0 and ii != gridpoints-1\
                                                 and jj != 0 
                                             and jj != gridpoints-1):
                        self.df[ii,jj,:]*= dxdy
                    elif ((ii == 0 or ii == gridpoints-1) and \
                            (jj != 0 and jj != gridpoints-1))\
                            or \
                            ((jj == 0 or jj == gridpoints-1) and \
                                 (ii != 0 and ii != gridpoints-1)): #edge
                        self.df[ii,jj,:]*= 1.5*dxdy
                    else: #corner
                        self.df[ii,jj,:]*= 2.25*dxdy
            if print_progress: sys.stdout.write('\n')
        else:
            self.df= nu.zeros((gridpoints,gridpoints))
            dxdy= (self.vRgrid[1]-self.vRgrid[0])\
                *(self.vTgrid[1]-self.vTgrid[0])
            if nlevels > 0:
                xsubmin= int(gridpoints)/4
                xsubmax= gridpoints-int(gridpoints)/4
            else:
                xsubmin= gridpoints
                xsubmax= 0
            ysubmin, ysubmax= xsubmin, xsubmax
            for ii in range(gridpoints):
                for jj in range(gridpoints):
                    if print_progress:
                        sys.stdout.write('\r'+"Velocity gridpoint %i out of %i" % \
                                             (jj+ii*gridpoints+1,gridpoints*gridpoints))
                        sys.stdout.flush()
                    #If this is part of a subgrid, ignore
                    if nlevels > 1 and ii >= xsubmin and ii < xsubmax \
                            and jj >= ysubmin and jj < ysubmax:
                        continue
                    thiso= Orbit([R,self.vRgrid[ii],self.vTgrid[jj],phi])
                    self.df[ii,jj]= edf(thiso,t)
                    if nu.isnan(self.df[ii,jj]): self.df[ii,jj]= 0. #BOVY: for now
                    #Multiply in area, somewhat tricky for edge objects
                    if upperdxdy is None or (ii != 0 and ii != gridpoints-1\
                                                 and jj != 0 
                                             and jj != gridpoints-1):
                        self.df[ii,jj]*= dxdy
                    elif ((ii == 0 or ii == gridpoints-1) and \
                            (jj != 0 and jj != gridpoints-1))\
                            or \
                            ((jj == 0 or jj == gridpoints-1) and \
                                 (ii != 0 and ii != gridpoints-1)): #edge
                        self.df[ii,jj]*= 1.5*dxdy
                    else: #corner
                        self.df[ii,jj]*= 2.25*dxdy
            if print_progress: sys.stdout.write('\n')
        if nlevels > 1:
            #Set up subgrid
            subnsigma= (self.meanvR-self.vRgrid[xsubmin])/self.sigmaR1
            self.subgrid= evolveddiskdfHierarchicalGrid(edf,R,phi,
                                                        subnsigma,t,
                                                        sigmaR1,
                                                        sigmaT1,
                                                        meanvR,
                                                        meanvT,
                                                        gridpoints,
                                                        nlevels-1,
                                                        upperdxdy=dxdy,
                                                        print_progress=print_progress,
                                                        nlevelsTotal=nlevelsTotal)
        else:
            self.subgrid= None
        return None
                
    def __call__(self,n,m):
        """Call"""
        if isinstance(self.t,(list,nu.ndarray)): tlist= True
        else: tlist= False
        if tlist: 
            nt= self.df.shape[2]
            out= []
            for ii in range(nt):
                #We already multiplied in the area
                out.append(nu.dot(self.vRgrid**n,nu.dot(self.df[:,:,ii],
                                                        self.vTgrid**m)))

            if self.subgrid is None: return nu.array(out)
            else: return nu.array(out)+ self.subgrid(n,m)
        else:
           #We already multiplied in the area
            thislevel= nu.dot(self.vRgrid**n,nu.dot(self.df,self.vTgrid**m))
            if self.subgrid is None: return thislevel
            else: return thislevel+self.subgrid(n,m)

    def plot(self,tt=0,vmax=None,aspect=None,extent=None):
        """
        NAME:
           plot
        PURPOSE:
           plot the velocity distribution
        INPUT:
           t= optional time index
        OUTPUT:
           plot of velocity distribution to output device
        HISTORY:
           2011-06-27 - Written - Bovy (NYU)
        """
        if vmax is None:
            print "You want to figure out a good vmax= using the max(tt=) member function ..."
        #Figure out how big of a grid we need
        dvR= (self.vRgrid[1]-self.vRgrid[0])
        dvT= (self.vTgrid[1]-self.vTgrid[0])
        nvR= len(self.vRgrid)
        nvT= len(self.vTgrid)
        nUpperLevels= self.nlevelsTotal-self.nlevels
        for ii in range(nUpperLevels):
            nvR*= 2
            nvT*= 2
        plotthis= nu.zeros((nvR,nvT))
        if len(self.df.shape) == 3:
            plotdf= copy.copy(self.df[:,:,tt])
        else:
            plotdf= copy.copy(self.df)
        print plotdf
        plotdf[(plotdf == 0.)]= _NAN
        #Fill up the grid
        xsubmin= 0
        xsubmax= nvR
        ysubmin= 0
        ysubmax= nvT
        for ii in range(nUpperLevels):
            xsubmin= int(nvR)/4
            xsubmax= nvR-int(nvR)/4
            ysubmin= int(nvT)/4
            ysubmax= nvT-int(nvT)/4
            for jj in range(nvR):
                for kk in range(nvT):
                    #If this is part of a subgrid, ignore
                    if jj >= xsubmin and jj < xsubmax \
                            and kk >= ysubmin and kk < ysubmax:
                        continue
                    plotthis[jj,kk]= _NAN
            nvR/= 2
            nvT/= 2
        #Fill in this level
        plotthis[xsubmin:xsubmax,ysubmin:ysubmax]= plotdf
        #print plotthis
        #Plot
        if nUpperLevels == 0:
            xrange= [self.vRgrid[0],self.vRgrid[len(self.vRgrid)-1]]
            yrange= [self.vTgrid[0],self.vTgrid[len(self.vTgrid)-1]]
            aspect=(xrange[1]-xrange[0])/\
                (yrange[1]-yrange[0])
            extent=[xrange[0],xrange[1],
                    yrange[0],yrange[1]]
            bovy_plot.bovy_dens2d(plotthis.T,cmap='gist_yarg',origin='lower',
                                          interpolation='nearest',
                                  aspect=aspect,
                                  extent=extent,
                                  xlabel=r'$v_R / v_0$',
                                  ylabel=r'$v_T / v_0$',
                                  vmin=0.,vmax=vmax)
        else:
            bovy_plot.bovy_dens2d(plotthis.T,cmap='gist_yarg',origin='lower',
                                  aspect=aspect,extent=extent,
                                  interpolation='nearest',
                                  overplot=True,vmin=0.,vmax=vmax)
        if not self.subgrid is None:
            self.subgrid.plot(tt=tt,vmax=vmax,aspect=aspect,extent=extent)

    def max(self,tt=0):
        if not self.subgrid is None:
            if len(self.df.shape) == 3:
                return nu.amax([nu.amax(self.df[:,:,tt]),
                                self.subgrid.max(tt)])
            else:
                return nu.amax([nu.amax(self.df[:,:]),
                                self.subgrid.max()])
        else:
            if len(self.df.shape) == 3:
                return nu.amax(self.df[:,:,tt])
            else:
                return nu.amax(self.df[:,:])


def _vmomentsurfaceIntegrand(vR,vT,R,az,df,n,m,sigmaR1,sigmaT1,t,initvmoment):
    """Internal function that is the integrand for the velocity moment times
    surface mass integration"""
    o= Orbit([R,vR*sigmaR1,vT*sigmaT1,az])
    return vR**n*vT**m*df(o,t)/initvmoment

def _marginalizeVperpIntegrandSinAlphaLarge(vR,df,R,sinalpha,cotalpha,
                                            vlos,vcirc,sigma,phi):
    return df(Orbit([R,vR*sigma,cotalpha*vR*sigma+vlos/sinalpha+vcirc,phi]))
                     

def _marginalizeVperpIntegrandSinAlphaSmall(vT,df,R,cosalpha,tanalpha,
                                            vlos,vcirc,sigma,phi):
    return df(Orbit([R,tanalpha*vT*sigma-vlos/cosalpha,vT*sigma+vcirc,phi]))

