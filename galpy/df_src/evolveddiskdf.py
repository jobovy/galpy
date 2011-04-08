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
import math as m
import numpy as nu
from scipy import integrate
from galpy.orbit import Orbit
from galpy.util.bovy_quadpack import dblquad
_DEGTORAD= m.pi/180.
_RADTODEG= 180./m.pi
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

    def __call__(self,*args):
        """
        NAME:
           __call__
        PURPOSE:
           evaluate the distribution function
        INPUT:
           Orbit instance:
              a) Orbit instance alone: use initial state and t=0
              b) Orbit instance + t: Orbit instance *NOT* called (i.e., Orbit's initial condition is used, call Orbit yourself)
        OUTPUT:
           DF(orbit,t)
        HISTORY:
           2011-03-30 - Written - Bovy (NYU)
        """
        if isinstance(args[0],Orbit):
            if len(args) == 1:
                t= 0.
            else:
                t= args[1]
        else:
            raise IOError("Input to __call__ not understood; this has to be an Orbit instance with optional time")
        #Integrate back
        if self._to == t:
            return self._initdf(args[0])
        ts= nu.linspace(0.,self._to-t,_NTS)
        o= args[0]
        #integrate orbit
        o.integrate(ts,self._pot)
        #Now evaluate the DF
        retval= self._initdf(o(self._to-t))
        if nu.isnan(retval): print retval, o.vxvv, o(self._to-t).vxvv
        return retval

    def vmomentsurfacemass(self,R,phi,n,m,t=0.,nsigma=None,deg=False,
                           epsrel=1.e-02,epsabs=1.e-05,
                           grid=None,gridpoints=101,returnGrid=False):
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
           t= time at which to evaluate the DF
        OPTIONAL INPUT:
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
        #Otherwise we need to do some more work
        if deg: az= phi*_DEGTORAD
        else: az= phi
        if nsigma == None: nsigma= _NSIGMA
        sigmaR1= nu.sqrt(self._initdf.sigmaR2(R))
        sigmaT1= nu.sqrt(self._initdf.sigmaT2(R))
        meanvR= self._initdf.meanvR(R)
        meanvT= self._initdf.meanvT(R)
        if not grid is None and isinstance(grid,bool) and grid:
            grido= self._buildvgrid(R,az,nsigma,t,
                                    sigmaR1,sigmaT1,meanvR,meanvT,gridpoints)
            if returnGrid:
                return (self._vmomentsurfacemassGrid(n,m,grido),grido)
            else:
                return self._vmomentsurfacemassGrid(n,m,grido)
        #Calculate the initdf moment and then calculate the ratio
        initvmoment= self._initdf.vmomentsurfacemass(R,n,m,nsigma=nsigma)
        if initvmoment == 0.: initvmoment= 1.
        norm= sigmaR1**(n+1)*sigmaT1**(m+1)*initvmoment
        return dblquad(_vmomentsurfaceIntegrand,
                                 meanvT/sigmaT1-nsigma,
                                 meanvT/sigmaT1+nsigma,
                                 lambda x: meanvR/sigmaR1
                                 -nu.sqrt(nsigma**2.-(x-meanvT/sigmaT1)**2.),
                                 lambda x: meanvR/sigmaR1
                                 +nu.sqrt(nsigma**2.-(x-meanvT/sigmaT1)**2.), 
                                 (R,az,self,n,m,sigmaR1,sigmaT1,t,initvmoment),
                                 epsrel=epsrel,epsabs=epsabs)[0]*norm

    def vertexdev(self,R,phi,t=0.,nsigma=None,deg=False,
                  epsrel=1.e-02,epsabs=1.e-05,
                  grid=None,gridpoints=101,returnGrid=False,
                  sigmaR2=None,sigmaT2=None,sigmaRT=None):
        """
        NAME:
           vertexdev
        PURPOSE:
           calculate the vertex deviation of the velocity distribution 
           at (R,phi)
        INPUT:
           R - radius at which to calculate the moment(/ro)
           phi= azimuth (rad unless deg=True)
           t= time at which to evaluate the DF
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
        OUTPUT:
           vertex deviation in degree
        HISTORY:
           2011-03-31 - Written - Bovy (NYU)
        """
        #The following aren't actually the moments, but they are the moments
        #times the surface-mass density; that drops out
        if isinstance(grid,evolveddiskdfGrid):
            grido= grid           
        elif (sigmaR2 is None or sigmaT2 is None or sigmaRT is None) \
                and isinstance(grid,bool) and grid:
            #Precalculate the grid and the surfacemass
            (sigmaR2,grido)= self.vmomentsurfacemass(R,phi,2,0,deg=deg,t=t,
                                                     nsigma=nsigma,
                                                     epsrel=epsrel,
                                                     epsabs=epsabs,grid=grid,
                                                     gridpoints=gridpoints,
                                                     returnGrid=True)
            surfacemass= self.vmomentsurfacemass(R,phi,0,0,deg=deg,t=t,
                                                 nsigma=nsigma,epsrel=epsrel,
                                                 epsabs=epsabs,grid=grido,
                                                 gridpoints=gridpoints,
                                                 returnGrid=False)
        else:
            grido= False
        if sigmaR2 is None:
            sigmaR2= self.sigmaR2(R,phi,deg=deg,t=t,
                                  nsigma=nsigma,epsrel=epsrel,
                                  epsabs=epsabs,grid=grido,
                                  gridpoints=gridpoints,
                                  returnGrid=False)/surfacemass
        if sigmaT2 is None:
            sigmaT2= self.sigmaT2(R,phi,deg=deg,t=t,
                                  nsigma=nsigma,epsrel=epsrel,
                                  epsabs=epsabs,grid=grido,
                                  gridpoints=gridpoints,
                                  returnGrid=False)/surfacemass
        if sigmaRT is None:
            sigmaRT= self.sigmaRT(R,phi,deg=deg,t=t,
                                  nsigma=nsigma,epsrel=epsrel,
                                  epsabs=epsabs,grid=grido,
                                  gridpoints=gridpoints,
                                  returnGrid=False)/surfacemass
        if returnGrid and ((isinstance(grid,bool) and grid) or 
                           isinstance(grid,evolveddiskdfGrid)):
            return (-nu.arctan(2.*sigmaRT/(sigmaR2-sigmaT2))/2.*_RADTODEG,
                     grido)
        else:
            return -nu.arctan(2.*sigmaRT/(sigmaR2-sigmaT2))/2.*_RADTODEG

    def meanvR(self,R,phi,t=0.,nsigma=None,deg=False,
               epsrel=1.e-02,epsabs=1.e-05,
               grid=None,gridpoints=101,returnGrid=False,
               surfacemass=None):
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
        OUTPUT:
           mean vR
        HISTORY:
           2011-03-31 - Written - Bovy (NYU)
        """
        if isinstance(grid,evolveddiskdfGrid):
            grido= grid           
            vmomentR= self.vmomentsurfacemass(R,phi,1,0,deg=deg,t=t,
                                              nsigma=nsigma,
                                              epsrel=epsrel,
                                              epsabs=epsabs,grid=grid,
                                              gridpoints=gridpoints,
                                              returnGrid=False)
        elif isinstance(grid,bool) and grid:
            #Precalculate the grid
            (vmomentR,grido)= self.vmomentsurfacemass(R,phi,1,0,deg=deg,t=t,
                                                      nsigma=nsigma,
                                                      epsrel=epsrel,
                                                      epsabs=epsabs,grid=grid,
                                                      gridpoints=gridpoints,
                                                      returnGrid=True)
        else:
            grido= False
            vmomentR= self.vmomentsurfacemass(R,phi,1,0,deg=deg,t=t,
                                              nsigma=nsigma,
                                              epsrel=epsrel,
                                              epsabs=epsabs,grid=grid,
                                              gridpoints=gridpoints,
                                              returnGrid=False)
        if surfacemass is None:
            surfacemass= self.vmomentsurfacemass(R,phi,0,0,deg=deg,t=t,
                                                 nsigma=nsigma,epsrel=epsrel,
                                                 epsabs=epsabs,grid=grido,
                                                 gridpoints=gridpoints,
                                                 returnGrid=False)
        out= vmomentR/surfacemass
        if returnGrid and ((isinstance(grid,bool) and grid) or 
                           isinstance(grid,evolveddiskdfGrid)):
            return (out,grido)
        else:
            return out

    def meanvT(self,R,phi,t=0.,nsigma=None,deg=False,
               epsrel=1.e-02,epsabs=1.e-05,
               grid=None,gridpoints=101,returnGrid=False,
               surfacemass=None):
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
        OUTPUT:
           mean vT
        HISTORY:
           2011-03-31 - Written - Bovy (NYU)
        """
        if isinstance(grid,evolveddiskdfGrid):
            grido= grid           
            vmomentT= self.vmomentsurfacemass(R,phi,0,1,deg=deg,t=t,
                                              nsigma=nsigma,
                                              epsrel=epsrel,
                                              epsabs=epsabs,grid=grid,
                                              gridpoints=gridpoints,
                                              returnGrid=False)
        elif isinstance(grid,bool) and grid:
            #Precalculate the grid
            (vmomentT,grido)= self.vmomentsurfacemass(R,phi,0,1,deg=deg,t=t,
                                                      nsigma=nsigma,
                                                      epsrel=epsrel,
                                                      epsabs=epsabs,grid=grid,
                                                      gridpoints=gridpoints,
                                                      returnGrid=True)
        else:
            grido= False
            vmomentT= self.vmomentsurfacemass(R,phi,0,1,deg=deg,t=t,
                                              nsigma=nsigma,
                                              epsrel=epsrel,
                                              epsabs=epsabs,grid=grid,
                                              gridpoints=gridpoints,
                                              returnGrid=False)
        if surfacemass is None:
            surfacemass= self.vmomentsurfacemass(R,phi,0,0,deg=deg,t=t,
                                                 nsigma=nsigma,epsrel=epsrel,
                                                 epsabs=epsabs,grid=grido,
                                                 gridpoints=gridpoints,
                                                 returnGrid=False)
        out= vmomentT/surfacemass
        if returnGrid and ((isinstance(grid,bool) and grid) or 
                           isinstance(grid,evolveddiskdfGrid)):
            return (out,grido)
        else:
            return out

    def sigmaR2(self,R,phi,t=0.,nsigma=None,deg=False,
                epsrel=1.e-02,epsabs=1.e-05,
                grid=None,gridpoints=101,returnGrid=False,
                surfacemass=None,meanvR=None):
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
        OUTPUT:
           variance of vR
        HISTORY:
           2011-03-31 - Written - Bovy (NYU)
        """
        #The following aren't actually the moments, but they are the moments
        #times the surface-mass density
        if isinstance(grid,evolveddiskdfGrid):
            grido= grid
            sigmaR2= self.vmomentsurfacemass(R,phi,2,0,deg=deg,t=t,
                                             nsigma=nsigma,
                                             epsrel=epsrel,
                                             epsabs=epsabs,grid=grido,
                                             gridpoints=gridpoints,
                                             returnGrid=False)
        elif (meanvR is None or surfacemass is None ) \
                and isinstance(grid,bool) and grid:
            #Precalculate the grid
            (sigmaR2,grido)= self.vmomentsurfacemass(R,phi,2,0,deg=deg,t=t,
                                                     nsigma=nsigma,
                                                     epsrel=epsrel,
                                                     epsabs=epsabs,grid=grid,
                                                     gridpoints=gridpoints,
                                                     returnGrid=True)
        else:
            grido= False
            sigmaR2= self.vmomentsurfacemass(R,phi,2,0,deg=deg,t=t,
                                             nsigma=nsigma,
                                             epsrel=epsrel,
                                             epsabs=epsabs,grid=grido,
                                             gridpoints=gridpoints,
                                             returnGrid=False)
        if surfacemass is None:
            surfacemass= self.vmomentsurfacemass(R,phi,0,0,deg=deg,t=t,
                                                 nsigma=nsigma,epsrel=epsrel,
                                                 epsabs=epsabs,grid=grido,
                                                 gridpoints=gridpoints,
                                                 returnGrid=False)
        if meanvR is None:
            meanvR= self.vmomentsurfacemass(R,phi,1,0,deg=deg,t=t,
                                             nsigma=nsigma,epsrel=epsrel,
                                             epsabs=epsabs,grid=grido,
                                             gridpoints=gridpoints,
                                             returnGrid=False)/surfacemass
        out= sigmaR2/surfacemass-meanvR**2.
        if returnGrid and ((isinstance(grid,bool) and grid) or 
                           isinstance(grid,evolveddiskdfGrid)):
            return (out,grido)
        else:
            return out

    def sigmaT2(self,R,phi,t=0.,nsigma=None,deg=False,
                epsrel=1.e-02,epsabs=1.e-05,
                grid=None,gridpoints=101,returnGrid=False,
                surfacemass=None,meanvT=None):
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
        OUTPUT:
           variance of vT
        HISTORY:
           2011-03-31 - Written - Bovy (NYU)
        """
        #The following aren't actually the moments, but they are the moments
        #times the surface-mass density
        if isinstance(grid,evolveddiskdfGrid):
            grido= grid
            sigmaT2= self.vmomentsurfacemass(R,phi,0,2,deg=deg,t=t,
                                             nsigma=nsigma,
                                             epsrel=epsrel,
                                             epsabs=epsabs,grid=grido,
                                             gridpoints=gridpoints,
                                             returnGrid=False)
        elif (meanvT is None or surfacemass is None ) \
                and isinstance(grid,bool) and grid:
            #Precalculate the grid
            (sigmaT2,grido)= self.vmomentsurfacemass(R,phi,0,2,deg=deg,t=t,
                                                     nsigma=nsigma,
                                                     epsrel=epsrel,
                                                     epsabs=epsabs,grid=grid,
                                                     gridpoints=gridpoints,
                                                     returnGrid=True)
        else:
            grido= False
            sigmaT2= self.vmomentsurfacemass(R,phi,0,2,deg=deg,t=t,
                                             nsigma=nsigma,
                                             epsrel=epsrel,
                                             epsabs=epsabs,grid=grido,
                                             gridpoints=gridpoints,
                                             returnGrid=False)
        if surfacemass is None:
            surfacemass= self.vmomentsurfacemass(R,phi,0,0,deg=deg,t=t,
                                                 nsigma=nsigma,epsrel=epsrel,
                                                 epsabs=epsabs,grid=grido,
                                                 gridpoints=gridpoints,
                                                 returnGrid=False)
        if meanvT is None:
            meanvT= self.vmomentsurfacemass(R,phi,0,1,deg=deg,t=t,
                                             nsigma=nsigma,epsrel=epsrel,
                                             epsabs=epsabs,grid=grido,
                                             gridpoints=gridpoints,
                                             returnGrid=False)/surfacemass
        out= sigmaT2/surfacemass-meanvT**2.
        if returnGrid and ((isinstance(grid,bool) and grid) or 
                           isinstance(grid,evolveddiskdfGrid)):
            return (out,grido)
        else:
            return out

    def sigmaRT(self,R,phi,t=0.,nsigma=None,deg=False,
                epsrel=1.e-02,epsabs=1.e-05,
                grid=None,gridpoints=101,returnGrid=False,
                surfacemass=None,meanvR=None,meanvT=None):
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
        OUTPUT:
           variance of vR
        HISTORY:
           2011-03-31 - Written - Bovy (NYU)
        """
        #The following aren't actually the moments, but they are the moments
        #times the surface-mass density
        if isinstance(grid,evolveddiskdfGrid):
            grido= grid
            sigmaRT= self.vmomentsurfacemass(R,phi,1,1,deg=deg,t=t,
                                             nsigma=nsigma,
                                             epsrel=epsrel,
                                             epsabs=epsabs,grid=grido,
                                             gridpoints=gridpoints,
                                             returnGrid=False)
        elif (meanvR is None or surfacemass is None ) \
                and isinstance(grid,bool) and grid:
            #Precalculate the grid
            (sigmaRT,grido)= self.vmomentsurfacemass(R,phi,1,1,deg=deg,t=t,
                                                     nsigma=nsigma,
                                                     epsrel=epsrel,
                                                     epsabs=epsabs,grid=grid,
                                                     gridpoints=gridpoints,
                                                     returnGrid=True)
        else:
            grido= False
            sigmaRT= self.vmomentsurfacemass(R,phi,1,1,deg=deg,t=t,
                                             nsigma=nsigma,
                                             epsrel=epsrel,
                                             epsabs=epsabs,grid=grido,
                                             gridpoints=gridpoints,
                                             returnGrid=False)
        if surfacemass is None:
            surfacemass= self.vmomentsurfacemass(R,phi,0,0,deg=deg,t=t,
                                                 nsigma=nsigma,epsrel=epsrel,
                                                 epsabs=epsabs,grid=grido,
                                                 gridpoints=gridpoints,
                                                 returnGrid=False)
        if meanvR is None:
            meanvR= self.vmomentsurfacemass(R,phi,1,0,deg=deg,t=t,
                                             nsigma=nsigma,epsrel=epsrel,
                                             epsabs=epsabs,grid=grido,
                                             gridpoints=gridpoints,
                                             returnGrid=False)/surfacemass
        if meanvT is None:
            meanvT= self.vmomentsurfacemass(R,phi,0,1,deg=deg,t=t,
                                             nsigma=nsigma,epsrel=epsrel,
                                             epsabs=epsabs,grid=grido,
                                             gridpoints=gridpoints,
                                             returnGrid=False)/surfacemass
        out= sigmaRT/surfacemass-meanvR*meanvT
        if returnGrid and ((isinstance(grid,bool) and grid) or 
                           isinstance(grid,evolveddiskdfGrid)):
            return (out,grido)
        else:
            return out

    def _vmomentsurfacemassGrid(self,n,m,grid):
        """Internal function to evaluate vmomentsurfacemass using a grid 
        rather than direct integration"""
        return nu.dot(grid.vRgrid**n,nu.dot(grid.df,grid.vTgrid**m))*\
            (grid.vRgrid[1]-grid.vRgrid[0])*(grid.vTgrid[1]-grid.vTgrid[0])
        
    def _buildvgrid(self,R,phi,nsigma,t,sigmaR1,sigmaT1,meanvR,meanvT,
                    gridpoints):
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
        out.df= nu.zeros((gridpoints,gridpoints))
        for ii in range(gridpoints):
            for jj in range(gridpoints):
                thiso= Orbit([R,out.vRgrid[ii],out.vTgrid[jj],phi])
                out.df[ii,jj]= self(thiso,t)
                if nu.isnan(out.df[ii,jj]): out.df[ii,jj]= 0.
        return out

class evolveddiskdfGrid:
    """Empty class since it is only used to store some stuff"""
    pass

def _vmomentsurfaceIntegrand(vR,vT,R,az,df,n,m,sigmaR1,sigmaT1,t,initvmoment):
    """Internal function that is the integrand for the velocity moment times
    surface mass integration"""
    o= Orbit([R,vR*sigmaR1,vT*sigmaT1,az])
    return vR**n*vT**m*df(o,t)/initvmoment

