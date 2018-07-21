###############################################################################
#   evolveddiskdf.py: module that builds a distribution function as a 
#                     steady-state DF + subsequent evolution
#
#   This module contains the following classes:
#
#      evolveddiskdf - top-level class that represents a distribution function
###############################################################################
from __future__ import print_function
_NSIGMA= 4.
_NTS= 1000
_PROFILE= False
import sys
import math
import copy
import time as time_module
import warnings
import numpy as nu
from scipy import integrate
from galpy.util import galpyWarning
from galpy.orbit import Orbit
from galpy.potential import calcRotcurve
from .df import df, _APY_LOADED
from galpy.potential.Potential import _check_c
from galpy.util.bovy_quadpack import dblquad
from galpy.util import bovy_plot
from galpy.util.bovy_conversion import physical_conversion, \
    potential_physical_input, time_in_Gyr
if _APY_LOADED:
    from astropy import units
_DEGTORAD= math.pi/180.
_RADTODEG= 180./math.pi
_NAN= nu.nan
class evolveddiskdf(df):
    """Class that represents a diskdf as initial DF + subsequent secular evolution"""
    def __init__(self,initdf,pot,to=0.):
        """
        NAME:

           __init__

        PURPOSE:

           initialize

        INPUT:

           initdf - the df at the start of the evolution (at to) (units are transferred)

           pot - potential to integrate orbits in

           to= initial time (time at which initdf is evaluated; orbits are integrated from current t back to to) (can be Quantity)

        OUTPUT:

           instance

        HISTORY:

           2011-03-30 - Written - Bovy (NYU)

        """
        if initdf._roSet: ro= initdf._ro
        else: ro= None
        if initdf._voSet: vo= initdf._vo
        else: vo= None
        df.__init__(self,ro=ro,vo=vo)
        self._initdf= initdf
        self._pot= pot
        if _APY_LOADED and isinstance(to,units.Quantity):
            to= to.to(units.Gyr).value/time_in_Gyr(self._vo,self._ro)
        self._to= to

    @physical_conversion('phasespacedensity2d',pop=True)
    def __call__(self,*args,**kwargs):
        """
        NAME:

           __call__

        PURPOSE:

           evaluate the distribution function

        INPUT:

           Orbit instance:

              a) Orbit instance alone: use initial state and t=0

              b) Orbit instance + t: Orbit instance *NOT* called (i.e., Orbit's initial condition is used, call Orbit yourself), t can be Quantity

                 If t is a list of t, DF is returned for each t, times must be in descending order and equally spaced (does not work with marginalize...)

           marginalizeVperp - marginalize over perpendicular velocity (only supported with 1a) above) + nsigma, +scipy.integrate.quad keywords

           marginalizeVlos - marginalize over line-of-sight velocity (only supported with 1a) above) + nsigma, +scipy.integrate.quad keywords

           log= if True, return the log (not for deriv, bc that can be negative)

           integrate_method= method argument of orbit.integrate

           deriv= None, 'R', or 'phi': calculates derivative of the moment wrt R or phi **not with the marginalize options**

        OUTPUT:

           DF(orbit,t)

        HISTORY:

           2011-03-30 - Written - Bovy (NYU)

           2011-04-15 - Added list of times option - Bovy (NYU)

        """
        integrate_method= kwargs.pop('integrate_method','dopr54_c')
        # Must match Python fallback for non-C potentials here, bc odeint needs
        # custom t list to avoid numerically instabilities
        if '_c' in integrate_method and not _check_c(self._pot):
            if ('leapfrog' in integrate_method \
                    or 'symplec' in integrate_method):
                integrate_method= 'leapfrog'
            else:
                integrate_method= 'odeint'
        deriv= kwargs.get('deriv',None)
        if isinstance(args[0],Orbit):
            if len(args) == 1:
                t= 0.
            else:
                t= args[1]
        else:
            raise IOError("Input to __call__ not understood; this has to be an Orbit instance with optional time")
        if isinstance(t,list):
            t= nu.array(t)
            tlist= True
        elif isinstance(t,nu.ndarray) and \
                not (hasattr(t,'isscalar') and t.isscalar):
            tlist= True
        else: tlist= False
        if _APY_LOADED and isinstance(t,units.Quantity):
            t= t.to(units.Gyr).value/time_in_Gyr(self._vo,self._ro)
        if kwargs.pop('marginalizeVperp',False):
            if tlist: raise IOError("Input times to __call__ is a list; this is not supported in conjunction with marginalizeVperp")
            if kwargs.pop('log',False):
                return nu.log(self._call_marginalizevperp(args[0],integrate_method=integrate_method,**kwargs))
            else:
                return self._call_marginalizevperp(args[0],integrate_method=integrate_method,**kwargs)
        elif kwargs.pop('marginalizeVlos',False):
            if tlist: raise IOError("Input times to __call__ is a list; this is not supported in conjunction with marginalizeVlos")
            if kwargs.pop('log',False):
                return nu.log(self._call_marginalizevlos(args[0],integrate_method=integrate_method,**kwargs))
            else:
                return self._call_marginalizevlos(args[0],integrate_method=integrate_method,**kwargs)   
        #Integrate back
        if tlist:
            if self._to == t[0]:
                if kwargs.get('log',False):
                    return nu.log([self._initdf(args[0],use_physical=False)])
                else:
                    return [self._initdf(args[0],use_physical=False)]
            ts= self._create_ts_tlist(t,integrate_method)
            o= args[0]
            #integrate orbit
            if _PROFILE: #pragma: no cover
                start= time_module.time()
            if not deriv is None:
                #Also calculate the derivative of the initial df with respect to R, phi, vR, and vT, and the derivative of Ro wrt R/phi etc., to calculate the derivative; in this case we also integrate a small area of phase space
                if deriv.lower() == 'r':
                    dderiv= 10.**-10.
                    tmp= o.R(use_physical=False)+dderiv
                    dderiv= tmp-o.R(use_physical=False)
                    msg= o._orb.integrate_dxdv([dderiv,0.,0.,0.],ts,self._pot,method=integrate_method)
                elif deriv.lower() == 'phi':
                    dderiv= 10.**-10.
                    tmp= o.phi(use_physical=False)+dderiv
                    dderiv= tmp-o.phi(use_physical=False)
                    msg= o._orb.integrate_dxdv([0.,0.,0.,dderiv],ts,self._pot,method=integrate_method)
                if msg > 0.: # pragma: no cover
                    print("Warning: dxdv integration inaccurate, returning zero everywhere ... result might not be correct ...")
                    if kwargs.get('log',False) and deriv is None: return nu.zeros(len(t))-nu.finfo(nu.dtype(nu.float64)).max
                    else: return nu.zeros(len(t))
                o._orb.orbit= o._orb.orbit_dxdv[:,0:4]
            else:
                o.integrate(ts,self._pot,method=integrate_method)
            if _PROFILE: #pragma: no cover
                int_time= (time_module.time()-start)
            #Now evaluate the DF
            if _PROFILE: #pragma: no cover
                start= time_module.time()
            if integrate_method == 'odeint':
                retval= []
                os= [o(self._to+t[0]-ti,use_physical=False) for ti in t]
                retval= nu.array(self._initdf(os,use_physical=False))
            else:
                if len(t) == 1:
                    orb_array= o.getOrbit().T
                    orb_array= orb_array[:,1]
                else:
                    orb_array= o.getOrbit().T
                retval= self._initdf(orb_array,use_physical=False)
                if (isinstance(retval,float) or len(retval.shape) == 0) \
                        and nu.isnan(retval):
                    retval= 0.
                elif not isinstance(retval,float) and len(retval.shape) > 0:
                    retval[(nu.isnan(retval))]= 0.
                if len(t) > 1: retval= retval[::-1]
            if _PROFILE: #pragma: no cover
                df_time= (time_module.time()-start)
                tot_time= int_time+df_time
                print(int_time/tot_time, df_time/tot_time, tot_time)
            if not deriv is None:
                if integrate_method == 'odeint':
                    dlnfdRo= nu.array([self._initdf._dlnfdR(o.R(self._to+t[0]-ti,use_physical=False),
                                                            o.vR(self._to+t[0]-ti,use_physical=False),
                                                            o.vT(self._to+t[0]-ti,use_physical=False))
                                       for ti in t])
                    dlnfdvRo= nu.array([self._initdf._dlnfdvR(o.R(self._to+t[0]-ti,use_physical=False),
                                                              o.vR(self._to+t[0]-ti,use_physical=False),
                                                              o.vT(self._to+t[0]-ti,use_physical=False))
                                        for ti in t])
                    dlnfdvTo= nu.array([self._initdf._dlnfdvT(o.R(self._to+t[0]-ti,use_physical=False),
                                                              o.vR(self._to+t[0]-ti,use_physical=False),
                                                              o.vT(self._to+t[0]-ti,use_physical=False))
                                        for ti in t])
                    dRo= nu.array([o._orb.orbit_dxdv[list(ts).index(self._to+t[0]-ti),4] for ti in t])/dderiv
                    dvRo= nu.array([o._orb.orbit_dxdv[list(ts).index(self._to+t[0]-ti),5] for ti in t])/dderiv
                    dvTo= nu.array([o._orb.orbit_dxdv[list(ts).index(self._to+t[0]-ti),6] for ti in t])/dderiv
                    #print(dRo, dvRo, dvTo)
                    dlnfderiv= dlnfdRo*dRo+dlnfdvRo*dvRo+dlnfdvTo*dvTo
                    retval*= dlnfderiv
                else:
                    if len(t) == 1:
                        dlnfdRo= self._initdf._dlnfdR(orb_array[0],
                                                      orb_array[1],
                                                      orb_array[2])
                        dlnfdvRo= self._initdf._dlnfdvR(orb_array[0],
                                                        orb_array[1],
                                                        orb_array[2])
                        dlnfdvTo= self._initdf._dlnfdvT(orb_array[0],
                                                        orb_array[1],
                                                        orb_array[2])
                    else:
                        dlnfdRo= nu.array([self._initdf._dlnfdR(orb_array[0,ii],
                                                                orb_array[1,ii],
                                                                orb_array[2,ii])
                                           for ii in range(len(t))])
                        dlnfdvRo= nu.array([self._initdf._dlnfdvR(orb_array[0,ii],
                                                                  orb_array[1,ii],
                                                                  orb_array[2,ii])
                                            for ii in range(len(t))])
                        dlnfdvTo= nu.array([self._initdf._dlnfdvT(orb_array[0,ii],
                                                                  orb_array[1,ii],
                                                                  orb_array[2,ii])
                                            for ii in range(len(t))])
                    dorb_array= o._orb.orbit_dxdv.T
                    if len(t) == 1: dorb_array= dorb_array[:,1]
                    dRo= dorb_array[4]/dderiv
                    dvRo= dorb_array[5]/dderiv
                    dvTo= dorb_array[6]/dderiv
                    #print(dRo, dvRo, dvTo)
                    dlnfderiv= dlnfdRo*dRo+dlnfdvRo*dvRo+dlnfdvTo*dvTo
                    if len(t) > 1: dlnfderiv= dlnfderiv[::-1]
                    retval*= dlnfderiv
        else:
            if self._to == t and deriv is None:
                if kwargs.get('log',False):
                    return nu.log(self._initdf(args[0],use_physical=False))
                else:
                    return self._initdf(args[0],use_physical=False)
            elif self._to == t and not deriv is None:
                if deriv.lower() == 'r':
                    return self._initdf(args[0])*self._initdf._dlnfdR(args[0]._orb.vxvv[0],
                                                                      args[0]._orb.vxvv[1],
                                                                      args[0]._orb.vxvv[2])
                elif deriv.lower() == 'phi':
                    return 0.
            if integrate_method == 'odeint':
                ts= nu.linspace(t,self._to,_NTS)
            else:
                ts= nu.linspace(t,self._to,2)
            o= args[0]
            #integrate orbit
            if not deriv is None:
                ts= nu.linspace(t,self._to,_NTS)
                #Also calculate the derivative of the initial df with respect to R, phi, vR, and vT, and the derivative of Ro wrt R/phi etc., to calculate the derivative; in this case we also integrate a small area of phase space
                if deriv.lower() == 'r':
                    dderiv= 10.**-10.
                    tmp= o.R(use_physical=False)+dderiv
                    dderiv= tmp-o.R(use_physical=False)
                    o._orb.integrate_dxdv([dderiv,0.,0.,0.],ts,self._pot,method=integrate_method)
                elif deriv.lower() == 'phi':
                    dderiv= 10.**-10.
                    tmp= o.phi(use_physical=False)+dderiv
                    dderiv= tmp-o.phi(use_physical=False)
                    o._orb.integrate_dxdv([0.,0.,0.,dderiv],ts,self._pot,method=integrate_method)
                o._orb.orbit= o._orb.orbit_dxdv[:,0:4]
            else:
                o.integrate(ts,self._pot,method=integrate_method)
            #int_time= (time.time()-start)
            #Now evaluate the DF
            if o.R(self._to-t,use_physical=False) <= 0.: 
                if kwargs.get('log',False):
                    return -nu.finfo(nu.dtype(nu.float64)).max
                else:
                    return nu.finfo(nu.dtype(nu.float64)).eps
            #start= time.time()
            retval= self._initdf(o(self._to-t,use_physical=False),
                                 use_physical=False)
            #print( int_time/(time.time()-start))
            if nu.isnan(retval): print(retval, o._orb.vxvv, o(self._to-t)._orb.vxvv)
            if not deriv is None:
                thisorbit= o(self._to-t)._orb.vxvv
                dlnfdRo= self._initdf._dlnfdR(thisorbit[0],
                                              thisorbit[1],
                                              thisorbit[2])
                dlnfdvRo= self._initdf._dlnfdvR(thisorbit[0],
                                                thisorbit[1],
                                                thisorbit[2])
                dlnfdvTo= self._initdf._dlnfdvT(thisorbit[0],
                                                thisorbit[1],
                                                thisorbit[2])
                indx= list(ts).index(self._to-t)
                dRo= o._orb.orbit_dxdv[indx,4]/dderiv
                dvRo= o._orb.orbit_dxdv[indx,5]/dderiv
                dvTo= o._orb.orbit_dxdv[indx,6]/dderiv
                dlnfderiv= dlnfdRo*dRo+dlnfdvRo*dvRo+dlnfdvTo*dvTo
                retval*= dlnfderiv
        if kwargs.get('log',False) and deriv is None:
            if tlist:
                out= nu.log(retval)
                out[retval == 0.]= -nu.finfo(nu.dtype(nu.float64)).max
            else:
                if retval == 0.: out= -nu.finfo(nu.dtype(nu.float64)).max
                else: out= nu.log(retval)
            return out
        else:
            return retval

    def vmomentsurfacemass(self,R,n,m,t=0.,nsigma=None,deg=False,
                           epsrel=1.e-02,epsabs=1.e-05,phi=0.,
                           grid=None,gridpoints=101,returnGrid=False,
                           hierarchgrid=False,nlevels=2,
                           print_progress=False,
                           integrate_method='dopr54_c',
                           deriv=None):
        """
        NAME:

           vmomentsurfacemass

        PURPOSE:

           calculate the an arbitrary moment of the velocity distribution at (R,phi) times the surfacmass

        INPUT:

           R - radius at which to calculate the moment (in natural units)

           phi= azimuth (rad unless deg=True)

           n - vR^n

           m - vT^m

           t= time at which to evaluate the DF (can be a list or ndarray; if this is the case, list needs to be in descending order and equally spaced)

           nsigma - number of sigma to integrate the velocities over (based on an estimate, so be generous, but not too generous)

           deg= azimuth is in degree (default=False)

           epsrel, epsabs - scipy.integrate keywords (the integration calculates the ratio of this vmoment to that of the initial DF)

           grid= if set to True, build a grid and use that to evaluate integrals; if set to a grid-objects (such as returned by this procedure), use this grid; if this was created for a list of times, moments are calculated for each time

           gridpoints= number of points to use for the grid in 1D (default=101)

           returnGrid= if True, return the grid object (default=False)

           hierarchgrid= if True, use a hierarchical grid (default=False)

           nlevels= number of hierarchical levels for the hierarchical grid

           print_progress= if True, print progress updates

           integrate_method= orbit.integrate method argument

           deriv= None, 'R', or 'phi': calculates derivative of the moment wrt R or phi **onnly with grid options**

        OUTPUT:

           <vR^n vT^m  x surface-mass> at R,phi (no support for units)

        COMMENT:

           grid-based calculation is the only one that is heavily tested (although the test suite also tests the direct calculation)

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
        if _PROFILE: #pragma: no cover
            start= time_module.time()
        if hasattr(self._initdf,'_estimatemeanvR') \
           and hasattr(self._initdf,'_estimatemeanvT') \
           and hasattr(self._initdf,'_estimateSigmaR2') \
           and hasattr(self._initdf,'_estimateSigmaT2'):
            sigmaR1= nu.sqrt(self._initdf._estimateSigmaR2(R,phi=az))
            sigmaT1= nu.sqrt(self._initdf._estimateSigmaT2(R,phi=az))
            meanvR= self._initdf._estimatemeanvR(R,phi=az)
            meanvT= self._initdf._estimatemeanvT(R,phi=az)
        else:
            warnings.warn("No '_estimateSigmaR2' etc. functions found for initdf in evolveddf; thus using potentially slow sigmaR2 etc functions",
                          galpyWarning)
            sigmaR1= nu.sqrt(self._initdf.sigmaR2(R,phi=az,use_physical=False))
            sigmaT1= nu.sqrt(self._initdf.sigmaT2(R,phi=az,use_physical=False))
            meanvR= self._initdf.meanvR(R,phi=az,use_physical=False)
            meanvT= self._initdf.meanvT(R,phi=az,use_physical=False)
        if _PROFILE: #pragma: no cover
            setup_time= (time_module.time()-start)
        if not grid is None and isinstance(grid,bool) and grid:
            if not hierarchgrid:
                if _PROFILE: #pragma: no cover
                    start= time_module.time()
                grido= self._buildvgrid(R,az,nsigma,t,
                                        sigmaR1,sigmaT1,meanvR,meanvT,
                                        gridpoints,print_progress,
                                        integrate_method,deriv)
                if _PROFILE: #pragma: no cover
                    grid_time= (time_module.time()-start)
                    print(setup_time/(setup_time+grid_time), \
                          grid_time/(setup_time+grid_time), \
                          setup_time+grid_time)
                if returnGrid:
                    return (self._vmomentsurfacemassGrid(n,m,grido),grido)
                else:
                    return self._vmomentsurfacemassGrid(n,m,grido)
            else: #hierarchical grid
                grido= evolveddiskdfHierarchicalGrid(self,R,az,nsigma,t,
                                                     sigmaR1,sigmaT1,meanvR,
                                                     meanvT,
                                                     gridpoints,nlevels,deriv,
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

    @potential_physical_input
    @physical_conversion('angle',pop=True)
    def vertexdev(self,R,t=0.,nsigma=None,deg=False,
                  epsrel=1.e-02,epsabs=1.e-05,phi=0.,
                  grid=None,gridpoints=101,returnGrid=False,
                  sigmaR2=None,sigmaT2=None,sigmaRT=None,surfacemass=None,
                  hierarchgrid=False,nlevels=2,
                  integrate_method='dopr54_c'):
        """
        NAME:

           vertexdev

        PURPOSE:

           calculate the vertex deviation of the velocity distribution at (R,phi)

        INPUT:

           R - radius at which to calculate the moment (can be Quantity)

           phi= azimuth (rad unless deg=True; can be Quantity)

           t= time at which to evaluate the DF (can be a list or ndarray; if this is the case, list needs to be in descending order and equally spaced) (can be Quantity)

           sigmaR2, sigmaT2, sigmaRT= if set the vertex deviation is simply calculated using these

           nsigma - number of sigma to integrate the velocities over (based on an estimate, so be generous)

           deg= azimuth is in degree (default=False); do not set this when giving phi as a Quantity

           epsrel, epsabs - scipy.integrate keywords (the integration calculates the ratio of this vmoment to that of the initial DF)

           grid= if set to True, build a grid and use that to evaluate integrals; if set to a grid-objects (such as returned by this procedure), use this grid

           gridpoints= number of points to use for the grid in 1D (default=101)

           returnGrid= if True, return the grid object (default=False)

           hierarchgrid= if True, use a hierarchical grid (default=False)

           nlevels= number of hierarchical levels for the hierarchical grid

           integrate_method= orbit.integrate method argument

        OUTPUT:

           vertex deviation in rad

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
            (sigmaR2_tmp,grido)= self.vmomentsurfacemass(R,2,0,deg=deg,t=t,
                                                         nsigma=nsigma,phi=phi,
                                                         epsrel=epsrel,
                                                         epsabs=epsabs,grid=grid,
                                                         gridpoints=gridpoints,
                                                         returnGrid=True,
                                                         hierarchgrid=hierarchgrid,
                                                         nlevels=nlevels,
                                                         integrate_method=integrate_method)
        else:
            grido= False
        if sigmaR2 is None:
            sigmaR2= self.sigmaR2(R,deg=deg,t=t,phi=phi,
                                  nsigma=nsigma,epsrel=epsrel,
                                  epsabs=epsabs,grid=grido,
                                  gridpoints=gridpoints,
                                  returnGrid=False,
                                  hierarchgrid=hierarchgrid,
                                  nlevels=nlevels,
                                  integrate_method=integrate_method,
                                  use_physical=False)
        if sigmaT2 is None:
            sigmaT2= self.sigmaT2(R,deg=deg,t=t,phi=phi,
                                  nsigma=nsigma,epsrel=epsrel,
                                  epsabs=epsabs,grid=grido,
                                  gridpoints=gridpoints,
                                  returnGrid=False,
                                  hierarchgrid=hierarchgrid,
                                  nlevels=nlevels,
                                  integrate_method=integrate_method,
                                  use_physical=False)
        if sigmaRT is None:
            sigmaRT= self.sigmaRT(R,deg=deg,t=t,phi=phi,
                                  nsigma=nsigma,epsrel=epsrel,
                                  epsabs=epsabs,grid=grido,
                                  gridpoints=gridpoints,
                                  returnGrid=False,
                                  hierarchgrid=hierarchgrid,
                                  nlevels=nlevels,
                                  integrate_method=integrate_method,
                                  use_physical=False)
        warnings.warn("In versions >1.3, the output unit of evolveddiskdf.vertexdev has been changed to radian (from degree before)",galpyWarning)
        if returnGrid and ((isinstance(grid,bool) and grid) or 
                           isinstance(grid,evolveddiskdfGrid) or
                           isinstance(grid,evolveddiskdfHierarchicalGrid)):
            return (-nu.arctan(2.*sigmaRT/(sigmaR2-sigmaT2))/2.,grido)
        else:
            return -nu.arctan(2.*sigmaRT/(sigmaR2-sigmaT2))/2.

    @potential_physical_input
    @physical_conversion('velocity',pop=True)
    def meanvR(self,R,t=0.,nsigma=None,deg=False,phi=0.,
               epsrel=1.e-02,epsabs=1.e-05,
               grid=None,gridpoints=101,returnGrid=False,
               surfacemass=None,
               hierarchgrid=False,nlevels=2,integrate_method='dopr54_c'):
        """
        NAME:

           meanvR

        PURPOSE:

           calculate the mean vR of the velocity distribution at (R,phi)

        INPUT:

           R - radius at which to calculate the moment(/ro) (can be Quantity)

           phi= azimuth (rad unless deg=True; can be Quantity)

           t= time at which to evaluate the DF (can be a list or ndarray; if this is the case, list needs to be in descending order and equally spaced) (can be Quantity)

           surfacemass= if set use this pre-calculated surfacemass

           nsigma - number of sigma to integrate the velocities over (based on an estimate, so be generous)

           deg= azimuth is in degree (default=False); do not set this when giving phi as a Quantity

           epsrel, epsabs - scipy.integrate keywords (the integration calculates the ratio of this vmoment to that of the initial DF)

           grid= if set to True, build a grid and use that to evaluate integrals; if set to a grid-objects (such as returned by this procedure), use this grid

           gridpoints= number of points to use for the grid in 1D (default=101)

           returnGrid= if True, return the grid object (default=False)

           hierarchgrid= if True, use a hierarchical grid (default=False)

           nlevels= number of hierarchical levels for the hierarchical grid

           integrate_method= orbit.integrate method argument

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
                                              nlevels=nlevels,
                                              integrate_method=integrate_method)
        elif isinstance(grid,bool) and grid:
            #Precalculate the grid
            (vmomentR,grido)= self.vmomentsurfacemass(R,1,0,deg=deg,t=t,
                                                      nsigma=nsigma,phi=phi,
                                                      epsrel=epsrel,
                                                      epsabs=epsabs,grid=grid,
                                                      gridpoints=gridpoints,
                                                      returnGrid=True,
                                                      hierarchgrid=hierarchgrid,
                                                      nlevels=nlevels,
                                                      integrate_method=integrate_method)
        else:
            grido= False
            vmomentR= self.vmomentsurfacemass(R,1,0,deg=deg,t=t,phi=phi,
                                              nsigma=nsigma,
                                              epsrel=epsrel,
                                              epsabs=epsabs,grid=grid,
                                              gridpoints=gridpoints,
                                              returnGrid=False,
                                              hierarchgrid=hierarchgrid,
                                              nlevels=nlevels,integrate_method=integrate_method)
        if surfacemass is None:
            surfacemass= self.vmomentsurfacemass(R,0,0,deg=deg,t=t,phi=phi,
                                                 nsigma=nsigma,epsrel=epsrel,
                                                 epsabs=epsabs,grid=grido,
                                                 gridpoints=gridpoints,
                                                 returnGrid=False,
                                                 hierarchgrid=hierarchgrid,
                                                 nlevels=nlevels,integrate_method=integrate_method)
        out= vmomentR/surfacemass
        if returnGrid and ((isinstance(grid,bool) and grid) or 
                           isinstance(grid,evolveddiskdfGrid) or
                           isinstance(grid,evolveddiskdfHierarchicalGrid)):
            return (out,grido)
        else:
            return out

    @potential_physical_input
    @physical_conversion('velocity',pop=True)
    def meanvT(self,R,t=0.,nsigma=None,deg=False,phi=0.,
               epsrel=1.e-02,epsabs=1.e-05,
               grid=None,gridpoints=101,returnGrid=False,
               surfacemass=None,
               hierarchgrid=False,nlevels=2,integrate_method='dopr54_c'):
        """
        NAME:

           meanvT

        PURPOSE:

           calculate the mean vT of the velocity distribution at (R,phi)

        INPUT:

           R - radius at which to calculate the moment (can be Quantity)

           phi= azimuth (rad unless deg=True; can be Quantity)

           t= time at which to evaluate the DF (can be a list or ndarray; if this is the case, list needs to be in descending order and equally spaced) (can be Quantity)

           surfacemass= if set use this pre-calculated surfacemass

           nsigma - number of sigma to integrate the velocities over (based on an estimate, so be generous)

           deg= azimuth is in degree (default=False); do not set this when giving phi as a Quantity

           epsrel, epsabs - scipy.integrate keywords (the integration calculates the ratio of this vmoment to that of the initial DF)

           grid= if set to True, build a grid and use that to evaluate integrals; if set to a grid-objects (such as returned by this procedure), use this grid

           gridpoints= number of points to use for the grid in 1D (default=101)

           returnGrid= if True, return the grid object (default=False)

           hierarchgrid= if True, use a hierarchical grid (default=False)

           nlevels= number of hierarchical levels for the hierarchical grid

           integrate_method= orbit.integrate method argument

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
                                              nlevels=nlevels,
                                              integrate_method=integrate_method)
        elif isinstance(grid,bool) and grid:
            #Precalculate the grid
            (vmomentT,grido)= self.vmomentsurfacemass(R,0,1,deg=deg,t=t,
                                                      nsigma=nsigma,phi=phi,
                                                      epsrel=epsrel,
                                                      epsabs=epsabs,grid=grid,
                                                      gridpoints=gridpoints,
                                                      returnGrid=True,
                                                      hierarchgrid=hierarchgrid,
                                                      nlevels=nlevels,
                                                      integrate_method=integrate_method)
        else:
            grido= False
            vmomentT= self.vmomentsurfacemass(R,0,1,deg=deg,t=t,
                                              nsigma=nsigma,phi=phi,
                                              epsrel=epsrel,
                                              epsabs=epsabs,grid=grid,
                                              gridpoints=gridpoints,
                                              returnGrid=False,
                                              hierarchgrid=hierarchgrid,
                                              nlevels=nlevels,integrate_method=integrate_method)
        if surfacemass is None:
            surfacemass= self.vmomentsurfacemass(R,0,0,deg=deg,t=t,phi=phi,
                                                 nsigma=nsigma,epsrel=epsrel,
                                                 epsabs=epsabs,grid=grido,
                                                 gridpoints=gridpoints,
                                                 returnGrid=False,
                                                 hierarchgrid=hierarchgrid,
                                                 nlevels=nlevels,
                                                 integrate_method=integrate_method)
        out= vmomentT/surfacemass
        if returnGrid and ((isinstance(grid,bool) and grid) or 
                           isinstance(grid,evolveddiskdfGrid) or
                           isinstance(grid,evolveddiskdfHierarchicalGrid)):
            return (out,grido)
        else:
            return out

    @potential_physical_input
    @physical_conversion('velocity2',pop=True)
    def sigmaR2(self,R,t=0.,nsigma=None,deg=False,phi=0.,
                epsrel=1.e-02,epsabs=1.e-05,
                grid=None,gridpoints=101,returnGrid=False,
                surfacemass=None,meanvR=None,
                hierarchgrid=False,nlevels=2,
                integrate_method='dopr54_c'):
        """
        NAME:

           sigmaR2

        PURPOSE:

           calculate the radial variance of the velocity distribution at (R,phi)

        INPUT:

           R - radius at which to calculate the moment (can be Quantity)

           phi= azimuth (rad unless deg=True; can be Quantity)

           t= time at which to evaluate the DF (can be a list or ndarray; if this is the case, list needs to be in descending order and equally spaced) (can be Quantity)

           surfacemass, meanvR= if set use this pre-calculated surfacemass and mean vR

           nsigma - number of sigma to integrate the velocities over (based on an estimate, so be generous)

           deg= azimuth is in degree (default=False); do not set this when giving phi as a Quantity

           epsrel, epsabs - scipy.integrate keywords (the integration calculates the ratio of this vmoment to that of the initial DF)

           grid= if set to True, build a grid and use that to evaluate integrals; if set to a grid-objects (such as returned by this procedure), use this grid

           gridpoints= number of points to use for the grid in 1D (default=101)

           returnGrid= if True, return the grid object (default=False)

           hierarchgrid= if True, use a hierarchical grid (default=False)

           nlevels= number of hierarchical levels for the hierarchical grid

           integrate_method= orbit.integrate method argument

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
                                             nlevels=nlevels,
                                             integrate_method=integrate_method)
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
                                                     nlevels=nlevels,
                                                     integrate_method=integrate_method)
        else:
            grido= False
            sigmaR2= self.vmomentsurfacemass(R,2,0,deg=deg,t=t,
                                             nsigma=nsigma,phi=phi,
                                             epsrel=epsrel,
                                             epsabs=epsabs,grid=grido,
                                             gridpoints=gridpoints,
                                             returnGrid=False,
                                             hierarchgrid=hierarchgrid,
                                             nlevels=nlevels,
                                             integrate_method=integrate_method)
        if surfacemass is None:
            surfacemass= self.vmomentsurfacemass(R,0,0,deg=deg,t=t,phi=phi,
                                                 nsigma=nsigma,epsrel=epsrel,
                                                 epsabs=epsabs,grid=grido,
                                                 gridpoints=gridpoints,
                                                 returnGrid=False,
                                                 hierarchgrid=hierarchgrid,
                                                 nlevels=nlevels,
                                                 integrate_method=integrate_method)
        if meanvR is None:
            meanvR= self.vmomentsurfacemass(R,1,0,deg=deg,t=t,phi=phi,
                                            nsigma=nsigma,epsrel=epsrel,
                                            epsabs=epsabs,grid=grido,
                                            gridpoints=gridpoints,
                                            returnGrid=False,
                                            hierarchgrid=hierarchgrid,
                                            nlevels=nlevels,
                                            integrate_method=integrate_method)/surfacemass
        out= sigmaR2/surfacemass-meanvR**2.
        if returnGrid and ((isinstance(grid,bool) and grid) or 
                           isinstance(grid,evolveddiskdfGrid) or
                           isinstance(grid,evolveddiskdfHierarchicalGrid)):
            return (out,grido)
        else:
            return out

    @potential_physical_input
    @physical_conversion('velocity2',pop=True)
    def sigmaT2(self,R,t=0.,nsigma=None,deg=False,phi=0.,
                epsrel=1.e-02,epsabs=1.e-05,
                grid=None,gridpoints=101,returnGrid=False,
                surfacemass=None,meanvT=None,
                hierarchgrid=False,nlevels=2,
                integrate_method='dopr54_c'):
        """
        NAME:

           sigmaT2

        PURPOSE:

           calculate the tangential variance of the velocity distribution at (R,phi)

        INPUT:

           R - radius at which to calculate the moment (can be Quantity)

           phi= azimuth (rad unless deg=True; can be Quantity)

           t= time at which to evaluate the DF (can be a list or ndarray; if this is the case, list needs to be in descending order and equally spaced) (can be Quantity)

           surfacemass, meanvT= if set use this pre-calculated surfacemass and mean tangential velocity

           nsigma - number of sigma to integrate the velocities over (based on an estimate, so be generous)

           deg= azimuth is in degree (default=False); do not set this when giving phi as a Quantity

           epsrel, epsabs - scipy.integrate keywords (the integration calculates the ratio of this vmoment to that of the initial DF)

           grid= if set to True, build a grid and use that to evaluate integrals; if set to a grid-objects (such as returned by this procedure), use this grid

           gridpoints= number of points to use for the grid in 1D (default=101)

           returnGrid= if True, return the grid object (default=False)

           hierarchgrid= if True, use a hierarchical grid (default=False)

           nlevels= number of hierarchical levels for the hierarchical grid

           integrate_method= orbit.integrate method argument

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
                                             nlevels=nlevels,
                                             integrate_method=integrate_method)
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
                                                     nlevels=nlevels,
                                                     integrate_method=integrate_method)
        else:
            grido= False
            sigmaT2= self.vmomentsurfacemass(R,0,2,deg=deg,t=t,
                                             nsigma=nsigma,phi=phi,
                                             epsrel=epsrel,
                                             epsabs=epsabs,grid=grido,
                                             gridpoints=gridpoints,
                                             returnGrid=False,
                                             hierarchgrid=hierarchgrid,
                                             nlevels=nlevels,
                                             integrate_method=integrate_method)
        if surfacemass is None:
            surfacemass= self.vmomentsurfacemass(R,0,0,deg=deg,t=t,phi=phi,
                                                 nsigma=nsigma,epsrel=epsrel,
                                                 epsabs=epsabs,grid=grido,
                                                 gridpoints=gridpoints,
                                                 returnGrid=False,
                                                 hierarchgrid=hierarchgrid,
                                                 nlevels=nlevels,
                                                 integrate_method=integrate_method)
        if meanvT is None:
            meanvT= self.vmomentsurfacemass(R,0,1,deg=deg,t=t,phi=phi,
                                            nsigma=nsigma,epsrel=epsrel,
                                            epsabs=epsabs,grid=grido,
                                            gridpoints=gridpoints,
                                            returnGrid=False,
                                            hierarchgrid=hierarchgrid,
                                            nlevels=nlevels,
                                            integrate_method=integrate_method)/surfacemass
        out= sigmaT2/surfacemass-meanvT**2.
        if returnGrid and ((isinstance(grid,bool) and grid) or 
                           isinstance(grid,evolveddiskdfGrid) or
                           isinstance(grid,evolveddiskdfHierarchicalGrid)):
            return (out,grido)
        else:
            return out

    @potential_physical_input
    @physical_conversion('velocity2',pop=True)
    def sigmaRT(self,R,t=0.,nsigma=None,deg=False,
                epsrel=1.e-02,epsabs=1.e-05,phi=0.,
                grid=None,gridpoints=101,returnGrid=False,
                surfacemass=None,meanvR=None,meanvT=None,
                hierarchgrid=False,nlevels=2,
                integrate_method='dopr54_c'):
        """
        NAME:

           sigmaRT

        PURPOSE:

           calculate the radial-tangential co-variance of the velocity distribution at (R,phi)

        INPUT:

           R - radius at which to calculate the moment (can be Quantity)

           phi= azimuth (rad unless deg=True; can be Quantity)

           t= time at which to evaluate the DF (can be a list or ndarray; if this is the case, list needs to be in descending order and equally spaced) (can be Quantity)

           surfacemass, meanvR, meavT= if set use this pre-calculated surfacemass and mean vR and vT

           nsigma - number of sigma to integrate the velocities over (based on an estimate, so be generous)

           deg= azimuth is in degree (default=False); do not set this when giving phi as a Quantity

           epsrel, epsabs - scipy.integrate keywords (the integration calculates the ration of this vmoment to that of the initial DF)

           grid= if set to True, build a grid and use that to evaluate integrals; if set to a grid-objects (such as returned by this procedure), use this grid

           gridpoints= number of points to use for the grid in 1D (default=101)

           returnGrid= if True, return the grid object (default=False)

           hierarchgrid= if True, use a hierarchical grid (default=False)

           nlevels= number of hierarchical levels for the hierarchical grid

           integrate_method= orbit.integrate method argument

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
                                             nlevels=nlevels,
                                             integrate_method=integrate_method)
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
                                                     nlevels=nlevels,
                                                     integrate_method=integrate_method)
        else:
            grido= False
            sigmaRT= self.vmomentsurfacemass(R,1,1,deg=deg,t=t,
                                             nsigma=nsigma,phi=phi,
                                             epsrel=epsrel,
                                             epsabs=epsabs,grid=grido,
                                             gridpoints=gridpoints,
                                             returnGrid=False,
                                             hierarchgrid=hierarchgrid,
                                             nlevels=nlevels,
                                             integrate_method=integrate_method)
        if surfacemass is None:
            surfacemass= self.vmomentsurfacemass(R,0,0,deg=deg,t=t,phi=phi,
                                                 nsigma=nsigma,epsrel=epsrel,
                                                 epsabs=epsabs,grid=grido,
                                                 gridpoints=gridpoints,
                                                 returnGrid=False,
                                                 hierarchgrid=hierarchgrid,
                                                 nlevels=nlevels,
                                                 integrate_method=integrate_method)
        if meanvR is None:
            meanvR= self.vmomentsurfacemass(R,1,0,deg=deg,t=t,phi=phi,
                                            nsigma=nsigma,epsrel=epsrel,
                                            epsabs=epsabs,grid=grido,
                                            gridpoints=gridpoints,
                                            returnGrid=False,
                                            hierarchgrid=hierarchgrid,
                                            nlevels=nlevels,
                                            integrate_method=integrate_method)/surfacemass
        if meanvT is None:
            meanvT= self.vmomentsurfacemass(R,0,1,deg=deg,t=t,phi=phi,
                                            nsigma=nsigma,epsrel=epsrel,
                                            epsabs=epsabs,grid=grido,
                                            gridpoints=gridpoints,
                                            returnGrid=False,
                                            hierarchgrid=hierarchgrid,
                                            nlevels=nlevels,
                                            integrate_method=integrate_method)/surfacemass
        out= sigmaRT/surfacemass-meanvR*meanvT
        if returnGrid and ((isinstance(grid,bool) and grid) or 
                           isinstance(grid,evolveddiskdfGrid) or
                           isinstance(grid,evolveddiskdfHierarchicalGrid)):
            return (out,grido)
        else:
            return out

    @potential_physical_input
    @physical_conversion('frequency-kmskpc',pop=True)
    def oortA(self,R,t=0.,nsigma=None,deg=False,phi=0.,
              epsrel=1.e-02,epsabs=1.e-05,
              grid=None,gridpoints=101,returnGrids=False,
              derivRGrid=None,derivphiGrid=None,derivGridpoints=101,
              derivHierarchgrid=False,
              hierarchgrid=False,nlevels=2,integrate_method='dopr54_c'):
        """
        NAME:

           oortA

        PURPOSE:

           calculate the Oort function A at (R,phi,t)

        INPUT:

           R - radius at which to calculate A (can be Quantity)

           phi= azimuth (rad unless deg=True; can be Quantity

           t= time at which to evaluate the DF (can be a list or ndarray; if this is the case, list needs to be in descending order and equally spaced) (can be Quantity)

           nsigma - number of sigma to integrate the velocities over (based on an estimate, so be generous)

           deg= azimuth is in degree (default=False); do not set this when giving phi as a Quantity

           epsrel, epsabs - scipy.integrate keywords

           grid= if set to True, build a grid and use that to evaluate integrals; if set to a grid-objects (such as returned by this procedure), use this grid

           derivRGrid, derivphiGrid= if set to True, build a grid and use that to evaluate integrals of the derivatives of the DF;if set to a grid-objects (such as returned by this procedure), use this grid
           
           gridpoints= number of points to use for the grid in 1D (default=101)

           derivGridpoints= number of points to use for the grid in 1D (default=101)

           returnGrid= if True, return the grid objects (default=False)

           hierarchgrid= if True, use a hierarchical grid (default=False)

           derivHierarchgrid= if True, use a hierarchical grid (default=False)

           nlevels= number of hierarchical levels for the hierarchical grid

           integrate_method= orbit.integrate method argument

        OUTPUT:

           Oort A at R,phi,t

        HISTORY:

           2011-10-16 - Written - Bovy (NYU)

        """
        #First calculate the grids if they are not given
        if isinstance(grid,bool) and grid:
            (surfacemass,grid)= self.vmomentsurfacemass(R,0,0,deg=deg,t=t,phi=phi,
                                                        nsigma=nsigma,epsrel=epsrel,
                                                        epsabs=epsabs,grid=True,
                                                        gridpoints=gridpoints,
                                                        returnGrid=True,
                                                        hierarchgrid=hierarchgrid,
                                                        nlevels=nlevels,
                                                        integrate_method=integrate_method)
        elif isinstance(grid,evolveddiskdfGrid) or \
                isinstance(grid,evolveddiskdfHierarchicalGrid):
            surfacemass= self.vmomentsurfacemass(R,0,0,deg=deg,t=t,phi=phi,
                                                 nsigma=nsigma,epsrel=epsrel,
                                                 epsabs=epsabs,grid=grid,
                                                 gridpoints=gridpoints,
                                                 returnGrid=False,
                                                 hierarchgrid=hierarchgrid,
                                                 nlevels=nlevels,
                                                 integrate_method=integrate_method)
        if isinstance(derivRGrid,bool) and derivRGrid:
            (dsurfacemassdR,derivRGrid)= self.vmomentsurfacemass(R,0,0,deg=deg,t=t,phi=phi,
                                                                 nsigma=nsigma,epsrel=epsrel,
                                                                 epsabs=epsabs,grid=True,
                                                                 gridpoints=derivGridpoints,
                                                                 returnGrid=True,
                                                                 hierarchgrid=derivHierarchgrid,
                                                                 nlevels=nlevels,
                                                                 integrate_method=integrate_method,deriv='R')
        elif isinstance(derivRGrid,evolveddiskdfGrid) or \
                isinstance(derivRGrid,evolveddiskdfHierarchicalGrid):
            dsurfacemassdR= self.vmomentsurfacemass(R,0,0,deg=deg,t=t,phi=phi,
                                                    nsigma=nsigma,epsrel=epsrel,
                                                    epsabs=epsabs,grid=derivRGrid,
                                                    gridpoints=derivGridpoints,
                                                    returnGrid=False,
                                                    hierarchgrid=derivHierarchgrid,
                                                    nlevels=nlevels,
                                                    integrate_method=integrate_method,deriv='R')
        if isinstance(derivphiGrid,bool) and derivphiGrid:
            (dsurfacemassdphi,derivphiGrid)= self.vmomentsurfacemass(R,0,0,deg=deg,t=t,phi=phi,
                                                                     nsigma=nsigma,epsrel=epsrel,
                                                                     epsabs=epsabs,grid=True,
                                                                     gridpoints=derivGridpoints,
                                                                     returnGrid=True,
                                                                     hierarchgrid=derivHierarchgrid,
                                                                     nlevels=nlevels,
                                                                     integrate_method=integrate_method,deriv='phi')
        elif isinstance(derivphiGrid,evolveddiskdfGrid) or \
                isinstance(derivphiGrid,evolveddiskdfHierarchicalGrid):
            dsurfacemassdphi= self.vmomentsurfacemass(R,0,0,deg=deg,t=t,phi=phi,
                                                      nsigma=nsigma,epsrel=epsrel,
                                                      epsabs=epsabs,grid=derivphiGrid,
                                                      gridpoints=derivGridpoints,
                                                      returnGrid=False,
                                                      hierarchgrid=derivHierarchgrid,
                                                      nlevels=nlevels,
                                                      integrate_method=integrate_method,deriv='phi')
        #2A= meanvT/R-dmeanvR/R/dphi-dmeanvphi/dR
        #meanvT
        meanvT= self.meanvT(R,t=t,nsigma=nsigma,deg=deg,phi=phi,
                            epsrel=epsrel,epsabs=epsabs,
                            grid=grid,gridpoints=gridpoints,returnGrid=False,
                            surfacemass=surfacemass,
                            hierarchgrid=hierarchgrid,
                            nlevels=nlevels,integrate_method=integrate_method,
                            use_physical=False)
        dmeanvRdphi= (self.vmomentsurfacemass(R,1,0,deg=deg,t=t,phi=phi,
                                              nsigma=nsigma,epsrel=epsrel,
                                              epsabs=epsabs,grid=derivphiGrid,
                                              gridpoints=derivGridpoints,
                                              returnGrid=False,
                                              hierarchgrid=derivHierarchgrid,
                                              nlevels=nlevels,
                                              integrate_method=integrate_method,deriv='phi')
                      /surfacemass
                      -self.vmomentsurfacemass(R,1,0,deg=deg,t=t,phi=phi,
                                               nsigma=nsigma,epsrel=epsrel,
                                               epsabs=epsabs,grid=grid,
                                               gridpoints=gridpoints,
                                               returnGrid=False,
                                               hierarchgrid=hierarchgrid,
                                               nlevels=nlevels,
                                               integrate_method=integrate_method)
                      /surfacemass**2.*dsurfacemassdphi)
        dmeanvTdR= (self.vmomentsurfacemass(R,0,1,deg=deg,t=t,phi=phi,
                                            nsigma=nsigma,epsrel=epsrel,
                                            epsabs=epsabs,grid=derivRGrid,
                                            gridpoints=derivGridpoints,
                                            returnGrid=False,
                                            hierarchgrid=derivHierarchgrid,
                                            nlevels=nlevels,
                                            integrate_method=integrate_method,deriv='R')
                    /surfacemass
                    -self.vmomentsurfacemass(R,0,1,deg=deg,t=t,phi=phi,
                                               nsigma=nsigma,epsrel=epsrel,
                                               epsabs=epsabs,grid=grid,
                                               gridpoints=gridpoints,
                                               returnGrid=False,
                                               hierarchgrid=hierarchgrid,
                                               nlevels=nlevels,
                                               integrate_method=integrate_method)
                      /surfacemass**2.*dsurfacemassdR)
        if returnGrids:
            return (0.5*(meanvT/R-dmeanvRdphi/R-dmeanvTdR),grid,
                    derivRGrid,derivphiGrid)
        else:
            return 0.5*(meanvT/R-dmeanvRdphi/R-dmeanvTdR)

    @potential_physical_input
    @physical_conversion('frequency-kmskpc',pop=True)
    def oortB(self,R,t=0.,nsigma=None,deg=False,phi=0.,
              epsrel=1.e-02,epsabs=1.e-05,
              grid=None,gridpoints=101,returnGrids=False,
              derivRGrid=None,derivphiGrid=None,derivGridpoints=101,
              derivHierarchgrid=False,
              hierarchgrid=False,nlevels=2,integrate_method='dopr54_c'):
        """
        NAME:

           oortB

        PURPOSE:

           calculate the Oort function B at (R,phi,t)

        INPUT:

           R - radius at which to calculate B (can be Quantity)

           phi= azimuth (rad unless deg=True; can be Quantity)

           t= time at which to evaluate the DF (can be a list or ndarray; if this is the case, list needs to be in descending order and equally spaced) (can be Quantity)

           nsigma - number of sigma to integrate the velocities over (based on an estimate, so be generous)

           deg= azimuth is in degree (default=False); do not set this when giving phi as a Quantity

           epsrel, epsabs - scipy.integrate keywords

           grid= if set to True, build a grid and use that to evaluate integrals; if set to a grid-objects (such as returned by this procedure), use this grid
           
           derivRGrid, derivphiGrid= if set to True, build a grid and use that to evaluat integrals of the derivatives of the DF: if set to a grid-objects (such as returned by this procedure), use this grid

           gridpoints= number of points to use for the grid in 1D (default=101)

           derivGridpoints= number of points to use for the grid in 1D (default=101)

           returnGrid= if True, return the grid objects (default=False)

           hierarchgrid= if True, use a hierarchical grid (default=False)

           derivHierarchgrid= if True, use a hierarchical grid (default=False)

           nlevels= number of hierarchical levels for the hierarchical grid

           integrate_method= orbit.integrate method argument

        OUTPUT:

           Oort B at R,phi,t

        HISTORY:

           2011-10-16 - Written - Bovy (NYU)

        """
        #First calculate the grids if they are not given
        if isinstance(grid,bool) and grid:
            (surfacemass,grid)= self.vmomentsurfacemass(R,0,0,deg=deg,t=t,phi=phi,
                                                        nsigma=nsigma,epsrel=epsrel,
                                                        epsabs=epsabs,grid=True,
                                                        gridpoints=gridpoints,
                                                        returnGrid=True,
                                                        hierarchgrid=hierarchgrid,
                                                        nlevels=nlevels,
                                                        integrate_method=integrate_method)
        elif isinstance(grid,evolveddiskdfGrid) or \
                isinstance(grid,evolveddiskdfHierarchicalGrid):
            surfacemass= self.vmomentsurfacemass(R,0,0,deg=deg,t=t,phi=phi,
                                                 nsigma=nsigma,epsrel=epsrel,
                                                 epsabs=epsabs,grid=grid,
                                                 gridpoints=gridpoints,
                                                 returnGrid=False,
                                                 hierarchgrid=hierarchgrid,
                                                 nlevels=nlevels,
                                                 integrate_method=integrate_method)
        if isinstance(derivRGrid,bool) and derivRGrid:
            (dsurfacemassdR,derivRGrid)= self.vmomentsurfacemass(R,0,0,deg=deg,t=t,phi=phi,
                                                                 nsigma=nsigma,epsrel=epsrel,
                                                                 epsabs=epsabs,grid=True,
                                                                 gridpoints=derivGridpoints,
                                                                 returnGrid=True,
                                                                 hierarchgrid=derivHierarchgrid,
                                                                 nlevels=nlevels,
                                                                 integrate_method=integrate_method,deriv='R')
        elif isinstance(derivRGrid,evolveddiskdfGrid) or \
                isinstance(derivRGrid,evolveddiskdfHierarchicalGrid):
            dsurfacemassdR= self.vmomentsurfacemass(R,0,0,deg=deg,t=t,phi=phi,
                                                    nsigma=nsigma,epsrel=epsrel,
                                                    epsabs=epsabs,grid=derivRGrid,
                                                    gridpoints=derivGridpoints,
                                                    returnGrid=False,
                                                    hierarchgrid=derivHierarchgrid,
                                                    nlevels=nlevels,
                                                    integrate_method=integrate_method,deriv='R')
        if isinstance(derivphiGrid,bool) and derivphiGrid:
            (dsurfacemassdphi,derivphiGrid)= self.vmomentsurfacemass(R,0,0,deg=deg,t=t,phi=phi,
                                                                     nsigma=nsigma,epsrel=epsrel,
                                                                     epsabs=epsabs,grid=True,
                                                                     gridpoints=derivGridpoints,
                                                                     returnGrid=True,
                                                                     hierarchgrid=derivHierarchgrid,
                                                                     nlevels=nlevels,
                                                                     integrate_method=integrate_method,deriv='phi')
        elif isinstance(derivphiGrid,evolveddiskdfGrid) or \
                isinstance(derivphiGrid,evolveddiskdfHierarchicalGrid):
            dsurfacemassdphi= self.vmomentsurfacemass(R,0,0,deg=deg,t=t,phi=phi,
                                                      nsigma=nsigma,epsrel=epsrel,
                                                      epsabs=epsabs,grid=derivphiGrid,
                                                      gridpoints=derivGridpoints,
                                                      returnGrid=False,
                                                      hierarchgrid=derivHierarchgrid,
                                                      nlevels=nlevels,
                                                      integrate_method=integrate_method,deriv='phi')
        #2B= -meanvT/R+dmeanvR/R/dphi-dmeanvphi/dR
        #meanvT
        meanvT= self.meanvT(R,t=t,nsigma=nsigma,deg=deg,phi=phi,
                            epsrel=epsrel,epsabs=epsabs,
                            grid=grid,gridpoints=gridpoints,returnGrid=False,
                            surfacemass=surfacemass,
                            hierarchgrid=hierarchgrid,
                            nlevels=nlevels,integrate_method=integrate_method,
                            use_physical=False)
        dmeanvRdphi= (self.vmomentsurfacemass(R,1,0,deg=deg,t=t,phi=phi,
                                              nsigma=nsigma,epsrel=epsrel,
                                              epsabs=epsabs,grid=derivphiGrid,
                                              gridpoints=derivGridpoints,
                                              returnGrid=False,
                                              hierarchgrid=derivHierarchgrid,
                                              nlevels=nlevels,
                                              integrate_method=integrate_method,deriv='phi')
                      /surfacemass
                      -self.vmomentsurfacemass(R,1,0,deg=deg,t=t,phi=phi,
                                               nsigma=nsigma,epsrel=epsrel,
                                               epsabs=epsabs,grid=grid,
                                               gridpoints=gridpoints,
                                               returnGrid=False,
                                               hierarchgrid=hierarchgrid,
                                               nlevels=nlevels,
                                               integrate_method=integrate_method)
                      /surfacemass**2.*dsurfacemassdphi)
        dmeanvTdR= (self.vmomentsurfacemass(R,0,1,deg=deg,t=t,phi=phi,
                                            nsigma=nsigma,epsrel=epsrel,
                                            epsabs=epsabs,grid=derivRGrid,
                                            gridpoints=derivGridpoints,
                                            returnGrid=False,
                                            hierarchgrid=derivHierarchgrid,
                                            nlevels=nlevels,
                                            integrate_method=integrate_method,deriv='R')
                    /surfacemass
                    -self.vmomentsurfacemass(R,0,1,deg=deg,t=t,phi=phi,
                                               nsigma=nsigma,epsrel=epsrel,
                                               epsabs=epsabs,grid=grid,
                                               gridpoints=gridpoints,
                                               returnGrid=False,
                                               hierarchgrid=hierarchgrid,
                                               nlevels=nlevels,
                                               integrate_method=integrate_method)
                      /surfacemass**2.*dsurfacemassdR)
        if returnGrids:
            return (0.5*(-meanvT/R+dmeanvRdphi/R-dmeanvTdR),grid,
                    derivRGrid,derivphiGrid)
        else:
            return 0.5*(-meanvT/R+dmeanvRdphi/R-dmeanvTdR)

    @potential_physical_input
    @physical_conversion('frequency-kmskpc',pop=True)
    def oortC(self,R,t=0.,nsigma=None,deg=False,phi=0.,
              epsrel=1.e-02,epsabs=1.e-05,
              grid=None,gridpoints=101,returnGrids=False,
              derivRGrid=None,derivphiGrid=None,derivGridpoints=101,
              derivHierarchgrid=False,
              hierarchgrid=False,nlevels=2,integrate_method='dopr54_c'):
        """
        NAME:

           oortC

        PURPOSE:

           calculate the Oort function C at (R,phi,t)

        INPUT:

           R - radius at which to calculate C (can be Quantity)

           phi= azimuth (rad unless deg=True; can be Quantity)

           t= time at which to evaluate the DF (can be a list or ndarray; if this is the case, list needs to be in descending order and equally spaced) (can be Quantity)

           nsigma - number of sigma to integrate the velocities over (based on an estimate, so be generous)

           deg= azimuth is in degree (default=False); do not set this when giving phi as a Quantity

           epsrel, epsabs - scipy.integrate keywords

           grid= if set to True, build a grid and use that to evaluate integrals; if set to a grid-objects (such as returned by this procedure), use this grid
           
           derivRGrid, derivphiGrid= if set to True, build a grid and use that to evaluate integrals of the derivatives of the DF; if set to a grid-objects (such as returned by this procedure), use this grid

           gridpoints= number of points to use for the grid in 1D (default=101)

           derivGridpoints= number of points to use for the grid in 1D (default=101)

           returnGrid= if True, return the grid objects (default=False)

           hierarchgrid= if True, use a hierarchical grid (default=False)

           derivHierarchgrid= if True, use a hierarchical grid (default=False)

           nlevels= number of hierarchical levels for the hierarchical grid

           integrate_method= orbit.integrate method argument

        OUTPUT:

           Oort C at R,phi,t

        HISTORY:

           2011-10-16 - Written - Bovy (NYU)

        """
        #First calculate the grids if they are not given
        if isinstance(grid,bool) and grid:
            (surfacemass,grid)= self.vmomentsurfacemass(R,0,0,deg=deg,t=t,phi=phi,
                                                        nsigma=nsigma,epsrel=epsrel,
                                                        epsabs=epsabs,grid=True,
                                                        gridpoints=gridpoints,
                                                        returnGrid=True,
                                                        hierarchgrid=hierarchgrid,
                                                        nlevels=nlevels,
                                                        integrate_method=integrate_method)
        elif isinstance(grid,evolveddiskdfGrid) or \
                isinstance(grid,evolveddiskdfHierarchicalGrid):
            surfacemass= self.vmomentsurfacemass(R,0,0,deg=deg,t=t,phi=phi,
                                                 nsigma=nsigma,epsrel=epsrel,
                                                 epsabs=epsabs,grid=grid,
                                                 gridpoints=gridpoints,
                                                 returnGrid=False,
                                                 hierarchgrid=hierarchgrid,
                                                 nlevels=nlevels,
                                                 integrate_method=integrate_method)
        if isinstance(derivRGrid,bool) and derivRGrid:
            (dsurfacemassdR,derivRGrid)= self.vmomentsurfacemass(R,0,0,deg=deg,t=t,phi=phi,
                                                                 nsigma=nsigma,epsrel=epsrel,
                                                                 epsabs=epsabs,grid=True,
                                                                 gridpoints=derivGridpoints,
                                                                 returnGrid=True,
                                                                 hierarchgrid=derivHierarchgrid,
                                                                 nlevels=nlevels,
                                                                 integrate_method=integrate_method,deriv='R')
        elif isinstance(derivRGrid,evolveddiskdfGrid) or \
                isinstance(derivRGrid,evolveddiskdfHierarchicalGrid):
            dsurfacemassdR= self.vmomentsurfacemass(R,0,0,deg=deg,t=t,phi=phi,
                                                    nsigma=nsigma,epsrel=epsrel,
                                                    epsabs=epsabs,grid=derivRGrid,
                                                    gridpoints=derivGridpoints,
                                                    returnGrid=False,
                                                    hierarchgrid=derivHierarchgrid,
                                                    nlevels=nlevels,
                                                    integrate_method=integrate_method,deriv='R')
        if isinstance(derivphiGrid,bool) and derivphiGrid:
            (dsurfacemassdphi,derivphiGrid)= self.vmomentsurfacemass(R,0,0,deg=deg,t=t,phi=phi,
                                                                     nsigma=nsigma,epsrel=epsrel,
                                                                     epsabs=epsabs,grid=True,
                                                                     gridpoints=derivGridpoints,
                                                                     returnGrid=True,
                                                                     hierarchgrid=derivHierarchgrid,
                                                                     nlevels=nlevels,
                                                                     integrate_method=integrate_method,deriv='phi')
        elif isinstance(derivphiGrid,evolveddiskdfGrid) or \
                isinstance(derivphiGrid,evolveddiskdfHierarchicalGrid):
            dsurfacemassdphi= self.vmomentsurfacemass(R,0,0,deg=deg,t=t,phi=phi,
                                                      nsigma=nsigma,epsrel=epsrel,
                                                      epsabs=epsabs,grid=derivphiGrid,
                                                      gridpoints=derivGridpoints,
                                                      returnGrid=False,
                                                      hierarchgrid=derivHierarchgrid,
                                                      nlevels=nlevels,
                                                      integrate_method=integrate_method,deriv='phi')
        #2C= -meanvR/R-dmeanvT/R/dphi+dmeanvR/dR
        #meanvR
        meanvR= self.meanvR(R,t=t,nsigma=nsigma,deg=deg,phi=phi,
                            epsrel=epsrel,epsabs=epsabs,
                            grid=grid,gridpoints=gridpoints,returnGrid=False,
                            surfacemass=surfacemass,
                            hierarchgrid=hierarchgrid,
                            nlevels=nlevels,integrate_method=integrate_method,
                            use_physical=False)
        dmeanvTdphi= (self.vmomentsurfacemass(R,0,1,deg=deg,t=t,phi=phi,
                                              nsigma=nsigma,epsrel=epsrel,
                                              epsabs=epsabs,grid=derivphiGrid,
                                              gridpoints=derivGridpoints,
                                              returnGrid=False,
                                              hierarchgrid=derivHierarchgrid,
                                              nlevels=nlevels,
                                              integrate_method=integrate_method,deriv='phi')
                      /surfacemass
                      -self.vmomentsurfacemass(R,0,1,deg=deg,t=t,phi=phi,
                                               nsigma=nsigma,epsrel=epsrel,
                                               epsabs=epsabs,grid=grid,
                                               gridpoints=gridpoints,
                                               returnGrid=False,
                                               hierarchgrid=hierarchgrid,
                                               nlevels=nlevels,
                                               integrate_method=integrate_method)
                      /surfacemass**2.*dsurfacemassdphi)
        dmeanvRdR= (self.vmomentsurfacemass(R,1,0,deg=deg,t=t,phi=phi,
                                            nsigma=nsigma,epsrel=epsrel,
                                            epsabs=epsabs,grid=derivRGrid,
                                            gridpoints=derivGridpoints,
                                            returnGrid=False,
                                            hierarchgrid=derivHierarchgrid,
                                            nlevels=nlevels,
                                            integrate_method=integrate_method,deriv='R')
                    /surfacemass
                    -self.vmomentsurfacemass(R,1,0,deg=deg,t=t,phi=phi,
                                               nsigma=nsigma,epsrel=epsrel,
                                               epsabs=epsabs,grid=grid,
                                               gridpoints=gridpoints,
                                               returnGrid=False,
                                               hierarchgrid=hierarchgrid,
                                               nlevels=nlevels,
                                               integrate_method=integrate_method)
                      /surfacemass**2.*dsurfacemassdR)
        if returnGrids:
            return (0.5*(-meanvR/R-dmeanvTdphi/R+dmeanvRdR),grid,
                    derivRGrid,derivphiGrid)
        else:
            return 0.5*(-meanvR/R-dmeanvTdphi/R+dmeanvRdR)

    @potential_physical_input
    @physical_conversion('frequency-kmskpc',pop=True)
    def oortK(self,R,t=0.,nsigma=None,deg=False,phi=0.,
              epsrel=1.e-02,epsabs=1.e-05,
              grid=None,gridpoints=101,returnGrids=False,
              derivRGrid=None,derivphiGrid=None,derivGridpoints=101,
              derivHierarchgrid=False,
              hierarchgrid=False,nlevels=2,integrate_method='dopr54_c'):
        """
        NAME:

           oortK

        PURPOSE:

           calculate the Oort function K at (R,phi,t)

        INPUT:

           R - radius at which to calculate K (can be Quantity)

           phi= azimuth (rad unless deg=True; can be Quantity)

           t= time at which to evaluate the DF (can be a list or ndarray; if this is the case, list needs to be in descending order and equally spaced) (can be Quantity)

           nsigma - number of sigma to integrate the velocities over (based on an estimate, so be generous)

           deg= azimuth is in degree (default=False); do not set this when giving phi as a Quantity

           epsrel, epsabs - scipy.integrate keywords

           grid= if set to True, build a grid and use that to evaluate integrals; if set to a grid-objects (such as returned by this procedure), use this grid

           derivRGrid, derivphiGrid= if set to True, build a grid and use that to evaluate integrals of the derivatives of the DF; if set to a grid-objects (such as returned by this procedure), use this grid

           gridpoints= number of points to use for the grid in 1D (default=101)

           derivGridpoints= number of points to use for the grid in 1D (default=101)

           returnGrid= if True, return the grid objects (default=False)

           hierarchgrid= if True, use a hierarchical grid (default=False)

           derivHierarchgrid= if True, use a hierarchical grid (default=False)

           nlevels= number of hierarchical levels for the hierarchical grid

           integrate_method= orbit.integrate method argument

        OUTPUT:

           Oort K at R,phi,t

        HISTORY:

           2011-10-16 - Written - Bovy (NYU)

        """
        #First calculate the grids if they are not given
        if isinstance(grid,bool) and grid:
            (surfacemass,grid)= self.vmomentsurfacemass(R,0,0,deg=deg,t=t,phi=phi,
                                                        nsigma=nsigma,epsrel=epsrel,
                                                        epsabs=epsabs,grid=True,
                                                        gridpoints=gridpoints,
                                                        returnGrid=True,
                                                        hierarchgrid=hierarchgrid,
                                                        nlevels=nlevels,
                                                        integrate_method=integrate_method)
        elif isinstance(grid,evolveddiskdfGrid) or \
                isinstance(grid,evolveddiskdfHierarchicalGrid):
            surfacemass= self.vmomentsurfacemass(R,0,0,deg=deg,t=t,phi=phi,
                                                 nsigma=nsigma,epsrel=epsrel,
                                                 epsabs=epsabs,grid=grid,
                                                 gridpoints=gridpoints,
                                                 returnGrid=False,
                                                 hierarchgrid=hierarchgrid,
                                                 nlevels=nlevels,
                                                 integrate_method=integrate_method)
        if isinstance(derivRGrid,bool) and derivRGrid:
            (dsurfacemassdR,derivRGrid)= self.vmomentsurfacemass(R,0,0,deg=deg,t=t,phi=phi,
                                                                 nsigma=nsigma,epsrel=epsrel,
                                                                 epsabs=epsabs,grid=True,
                                                                 gridpoints=derivGridpoints,
                                                                 returnGrid=True,
                                                                 hierarchgrid=derivHierarchgrid,
                                                                 nlevels=nlevels,
                                                                 integrate_method=integrate_method,deriv='R')
        elif isinstance(derivRGrid,evolveddiskdfGrid) or \
                isinstance(derivRGrid,evolveddiskdfHierarchicalGrid):
            dsurfacemassdR= self.vmomentsurfacemass(R,0,0,deg=deg,t=t,phi=phi,
                                                    nsigma=nsigma,epsrel=epsrel,
                                                    epsabs=epsabs,grid=derivRGrid,
                                                    gridpoints=derivGridpoints,
                                                    returnGrid=False,
                                                    hierarchgrid=derivHierarchgrid,
                                                    nlevels=nlevels,
                                                    integrate_method=integrate_method,deriv='R')
        if isinstance(derivphiGrid,bool) and derivphiGrid:
            (dsurfacemassdphi,derivphiGrid)= self.vmomentsurfacemass(R,0,0,deg=deg,t=t,phi=phi,
                                                                     nsigma=nsigma,epsrel=epsrel,
                                                                     epsabs=epsabs,grid=True,
                                                                     gridpoints=derivGridpoints,
                                                                     returnGrid=True,
                                                                     hierarchgrid=derivHierarchgrid,
                                                                     nlevels=nlevels,
                                                                     integrate_method=integrate_method,deriv='phi')
        elif isinstance(derivphiGrid,evolveddiskdfGrid) or \
                isinstance(derivphiGrid,evolveddiskdfHierarchicalGrid):
            dsurfacemassdphi= self.vmomentsurfacemass(R,0,0,deg=deg,t=t,phi=phi,
                                                      nsigma=nsigma,epsrel=epsrel,
                                                      epsabs=epsabs,grid=derivphiGrid,
                                                      gridpoints=derivGridpoints,
                                                      returnGrid=False,
                                                      hierarchgrid=derivHierarchgrid,
                                                      nlevels=nlevels,
                                                      integrate_method=integrate_method,deriv='phi')
        #2K= meanvR/R+dmeanvT/R/dphi+dmeanvR/dR
        #meanvR
        meanvR= self.meanvR(R,t=t,nsigma=nsigma,deg=deg,phi=phi,
                            epsrel=epsrel,epsabs=epsabs,
                            grid=grid,gridpoints=gridpoints,returnGrid=False,
                            surfacemass=surfacemass,
                            hierarchgrid=hierarchgrid,
                            nlevels=nlevels,integrate_method=integrate_method,
                            use_physical=False)
        dmeanvTdphi= (self.vmomentsurfacemass(R,0,1,deg=deg,t=t,phi=phi,
                                              nsigma=nsigma,epsrel=epsrel,
                                              epsabs=epsabs,grid=derivphiGrid,
                                              gridpoints=derivGridpoints,
                                              returnGrid=False,
                                              hierarchgrid=derivHierarchgrid,
                                              nlevels=nlevels,
                                              integrate_method=integrate_method,deriv='phi')
                      /surfacemass
                      -self.vmomentsurfacemass(R,0,1,deg=deg,t=t,phi=phi,
                                               nsigma=nsigma,epsrel=epsrel,
                                               epsabs=epsabs,grid=grid,
                                               gridpoints=gridpoints,
                                               returnGrid=False,
                                               hierarchgrid=hierarchgrid,
                                               nlevels=nlevels,
                                               integrate_method=integrate_method)
                      /surfacemass**2.*dsurfacemassdphi)
        dmeanvRdR= (self.vmomentsurfacemass(R,1,0,deg=deg,t=t,phi=phi,
                                            nsigma=nsigma,epsrel=epsrel,
                                            epsabs=epsabs,grid=derivRGrid,
                                            gridpoints=derivGridpoints,
                                            returnGrid=False,
                                            hierarchgrid=derivHierarchgrid,
                                            nlevels=nlevels,
                                            integrate_method=integrate_method,deriv='R')
                    /surfacemass
                    -self.vmomentsurfacemass(R,1,0,deg=deg,t=t,phi=phi,
                                               nsigma=nsigma,epsrel=epsrel,
                                               epsabs=epsabs,grid=grid,
                                               gridpoints=gridpoints,
                                               returnGrid=False,
                                               hierarchgrid=hierarchgrid,
                                               nlevels=nlevels,
                                               integrate_method=integrate_method)
                      /surfacemass**2.*dsurfacemassdR)
        if returnGrids:
            return (0.5*(meanvR/R+dmeanvTdphi/R+dmeanvRdR),grid,
                    derivRGrid,derivphiGrid)
        else:
            return 0.5*(meanvR/R+dmeanvTdphi/R+dmeanvRdR)

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
                    gridpoints,print_progress,integrate_method,deriv):
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
                for jj in range(gridpoints-1,-1,-1):#Reverse, so we get the peak before we get to the extreme lags NOT NECESSARY
                    if print_progress: #pragma: no cover
                        sys.stdout.write('\r'+"Velocity gridpoint %i out of %i" % \
                                             (jj+ii*gridpoints+1,gridpoints*gridpoints))
                        sys.stdout.flush()
                    thiso= Orbit([R,out.vRgrid[ii],out.vTgrid[jj],phi])
                    out.df[ii,jj,:]= self(thiso,nu.array(t).flatten(),
                                          integrate_method=integrate_method,
                                          deriv=deriv,use_physical=False)
                    out.df[ii,jj,nu.isnan(out.df[ii,jj,:])]= 0. #BOVY: for now
            if print_progress: sys.stdout.write('\n') #pragma: no cover
        else:
            out.df= nu.zeros((gridpoints,gridpoints))
            for ii in range(gridpoints):
                for jj in range(gridpoints):
                    if print_progress: #pragma: no cover
                        sys.stdout.write('\r'+"Velocity gridpoint %i out of %i" % \
                                             (jj+ii*gridpoints+1,gridpoints*gridpoints))
                        sys.stdout.flush()
                    thiso= Orbit([R,out.vRgrid[ii],out.vTgrid[jj],phi])
                    out.df[ii,jj]= self(thiso,t,
                                        integrate_method=integrate_method,
                                        deriv=deriv,use_physical=False)
                    if nu.isnan(out.df[ii,jj]): out.df[ii,jj]= 0. #BOVY: for now
            if print_progress: sys.stdout.write('\n') #pragma: no cover
        return out

    def _create_ts_tlist(self,t,integrate_method):
        #Check input
        if not all(t == sorted(t,reverse=True)): raise IOError("List of times has to be sorted in descending order")
        #Initialize
        if integrate_method == 'odeint':
            _NTS= 1000
            tmax= nu.amax(t)
            ts= nu.linspace(tmax,self._to,_NTS)
            #Add other t
            ts= list(ts)
            ts.extend([self._to+tmax-ti for ti in t if ti != tmax])
        else:
            if len(t) == 1: #Special case this because it is confusing
                ts= nu.array([t[0],self._to])
            else:
                ts= -t+self._to+nu.amax(t)
        #sort
        ts= list(ts)
        ts.sort(reverse=True)
        return nu.array(ts)

    def _call_marginalizevperp(self,o,integrate_method='dopr54_c',**kwargs):
        """Call the DF, marginalizing over perpendicular velocity"""
        #Get d, l, vlos
        l= o.ll(obs=[1.,0.,0.],ro=1.)*_DEGTORAD
        vlos= o.vlos(ro=1.,vo=1.,obs=[1.,0.,0.,0.,0.,0.])
        R= o.R(use_physical=False)
        phi= o.phi(use_physical=False)
        #Get local circular velocity, projected onto the los
        if isinstance(self._pot,list):
            vcirc= calcRotcurve([p for p in self._pot if not p.isNonAxi],R)[0]
        else:
            vcirc= calcRotcurve(self._pot,R)[0]
        vcirclos= vcirc*math.sin(phi+l)
        #Marginalize
        alphalos= phi+l
        if not 'nsigma' in kwargs or ('nsigma' in kwargs and \
                                          kwargs['nsigma'] is None):
            nsigma= _NSIGMA
        else:
            nsigma= kwargs['nsigma']
        kwargs.pop('nsigma',None)
        #BOVY: add asymmetric drift here?
        if math.fabs(math.sin(alphalos)) < math.sqrt(1./2.):
            sigmaR1= nu.sqrt(self._initdf.sigmaT2(R,phi=phi,
                                                  use_physical=False)) #Slight abuse
            cosalphalos= math.cos(alphalos)
            tanalphalos= math.tan(alphalos)
            return integrate.quad(_marginalizeVperpIntegrandSinAlphaSmall,
                                  -nsigma,nsigma,
                                  args=(self,R,cosalphalos,tanalphalos,
                                        vlos-vcirclos,vcirc,
                                        sigmaR1,phi),
                                  **kwargs)[0]/math.fabs(cosalphalos)*sigmaR1
        else:
            sigmaR1= nu.sqrt(self._initdf.sigmaR2(R,phi=phi,
                                                  use_physical=False))
            sinalphalos= math.sin(alphalos)
            cotalphalos= 1./math.tan(alphalos)
            return integrate.quad(_marginalizeVperpIntegrandSinAlphaLarge,
                                  -nsigma,nsigma,
                                  args=(self,R,sinalphalos,cotalphalos,
                                        vlos-vcirclos,vcirc,sigmaR1,phi),
                                  **kwargs)[0]/math.fabs(sinalphalos)*sigmaR1
        
    def _call_marginalizevlos(self,o,integrate_method='dopr54_c',**kwargs):
        """Call the DF, marginalizing over line-of-sight velocity"""
        #Get d, l, vperp
        l= o.ll(obs=[1.,0.,0.],ro=1.)*_DEGTORAD
        vperp= o.vll(ro=1.,vo=1.,obs=[1.,0.,0.,0.,0.,0.])
        R= o.R(use_physical=False)
        phi= o.phi(use_physical=False)
        #Get local circular velocity, projected onto the perpendicular 
        #direction
        if isinstance(self._pot,list):
            vcirc= calcRotcurve([p for p in self._pot if not p.isNonAxi],R)[0]
        else:
            vcirc= calcRotcurve(self._pot,R)[0]
        vcircperp= vcirc*math.cos(phi+l) 
        #Marginalize
        alphaperp= math.pi/2.+phi+l
        if not 'nsigma' in kwargs or ('nsigma' in kwargs and \
                                          kwargs['nsigma'] is None):
            nsigma= _NSIGMA
        else:
            nsigma= kwargs['nsigma']
        kwargs.pop('nsigma',None)
        if math.fabs(math.sin(alphaperp)) < math.sqrt(1./2.):
            sigmaR1= nu.sqrt(self._initdf.sigmaT2(R,phi=phi,
                                                  use_physical=False)) #slight abuse
            va= vcirc-self._initdf.meanvT(R,phi=phi,use_physical=False)
            cosalphaperp= math.cos(alphaperp)
            tanalphaperp= math.tan(alphaperp)
            #we can reuse the VperpIntegrand, since it is just another angle
            return integrate.quad(_marginalizeVperpIntegrandSinAlphaSmall,
                                  -va/sigmaR1-nsigma,
                                  -va/sigmaR1+nsigma,
                                  args=(self,R,cosalphaperp,tanalphaperp,
                                        vperp-vcircperp,vcirc,
                                        sigmaR1,phi),
                                  **kwargs)[0]/math.fabs(cosalphaperp)*sigmaR1
        else:
            sigmaR1= nu.sqrt(self._initdf.sigmaR2(R,phi=phi,
                                                  use_physical=False))
            sinalphaperp= math.sin(alphaperp)
            cotalphaperp= 1./math.tan(alphaperp)
            #we can reuse the VperpIntegrand, since it is just another angle
            return integrate.quad(_marginalizeVperpIntegrandSinAlphaLarge,
                                  -nsigma,nsigma,
                                  args=(self,R,sinalphaperp,cotalphaperp,
                                        vperp-vcircperp,vcirc,sigmaR1,phi),
                                  **kwargs)[0]/math.fabs(sinalphaperp)*sigmaR1

    def _vmomentsurfacemassHierarchicalGrid(self,n,m,grid):
        """Internal function to evaluate vmomentsurfacemass using a 
        hierarchical grid rather than direct integration,
        rather unnecessary""" 
        return grid(n,m)       
        
class evolveddiskdfGrid(object):
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

class evolveddiskdfHierarchicalGrid(object):
    """Class that holds a hierarchical velocity grid"""
    def __init__(self,edf,R,phi,nsigma,t,sigmaR1,sigmaT1,meanvR,meanvT,
                 gridpoints,nlevels,deriv,upperdxdy=None,print_progress=False,
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
            deriv- None, 'R', or 'phi': calculates derivative of the moment wrt
                  R or phi
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
                xsubmin= int(gridpoints)//4
                xsubmax= gridpoints-int(gridpoints)//4
            else:
                xsubmin= gridpoints
                xsubmax= 0
            ysubmin, ysubmax= xsubmin, xsubmax
            for ii in range(gridpoints):
                for jj in range(gridpoints):
                    if print_progress: #pragma: no cover
                        sys.stdout.write('\r'+"Velocity gridpoint %i out of %i" % \
                                             (jj+ii*gridpoints+1,gridpoints*gridpoints))
                        sys.stdout.flush()
                    #If this is part of a subgrid, ignore
                    if nlevels > 1 and ii >= xsubmin and ii < xsubmax \
                            and jj >= ysubmin and jj < ysubmax:
                        continue
                    thiso= Orbit([R,self.vRgrid[ii],self.vTgrid[jj],phi])
                    self.df[ii,jj,:]= edf(thiso,nu.array(t).flatten(),
                                          deriv=deriv)
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
                        self.df[ii,jj,:]*= 1.5*dxdy/1.5 #turn this off for now
                    else: #corner
                        self.df[ii,jj,:]*= 2.25*dxdy/2.25 #turn this off for now
            if print_progress: sys.stdout.write('\n') #pragma: no cover
        else:
            self.df= nu.zeros((gridpoints,gridpoints))
            dxdy= (self.vRgrid[1]-self.vRgrid[0])\
                *(self.vTgrid[1]-self.vTgrid[0])
            if nlevels > 0:
                xsubmin= int(gridpoints)//4
                xsubmax= gridpoints-int(gridpoints)//4
            else:
                xsubmin= gridpoints
                xsubmax= 0
            ysubmin, ysubmax= xsubmin, xsubmax
            for ii in range(gridpoints):
                for jj in range(gridpoints):
                    if print_progress: #pragma: no cover
                        sys.stdout.write('\r'+"Velocity gridpoint %i out of %i" % \
                                             (jj+ii*gridpoints+1,gridpoints*gridpoints))
                        sys.stdout.flush()
                    #If this is part of a subgrid, ignore
                    if nlevels > 1 and ii >= xsubmin and ii < xsubmax \
                            and jj >= ysubmin and jj < ysubmax:
                        continue
                    thiso= Orbit([R,self.vRgrid[ii],self.vTgrid[jj],phi])
                    self.df[ii,jj]= edf(thiso,t,deriv=deriv)
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
                        self.df[ii,jj]*= 1.5*dxdy/1.5#turn this off for now
                    else: #corner
                        self.df[ii,jj]*= 2.25*dxdy/2.25#turn this off for now
            if print_progress: sys.stdout.write('\n') #pragma: no cover
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
                                                        deriv,
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
            vmax= self.max(tt=tt)*2.
        #Figure out how big of a grid we need
        dvR= (self.vRgrid[1]-self.vRgrid[0])
        dvT= (self.vTgrid[1]-self.vTgrid[0])
        nvR= len(self.vRgrid)
        nvT= len(self.vTgrid)
        nUpperLevels= self.nlevelsTotal-self.nlevels
        nvRTot= nvR*2**nUpperLevels
        nvTTot= nvT*2**nUpperLevels
        plotthis= nu.zeros((nvRTot,nvTTot))
        if len(self.df.shape) == 3:
            plotdf= copy.copy(self.df[:,:,tt])
        else:
            plotdf= copy.copy(self.df)
        plotdf[(plotdf == 0.)]= _NAN
        #Fill up the grid
        if nUpperLevels > 0:
            xsubmin= nvRTot//2-nvR//2-1
            xsubmax= nvRTot//2+nvR//2
            ysubmin= nvTTot//2-nvT//2-1
            ysubmax= nvTTot//2+nvT//2
            #Set outside this subgrid to NaN
            plotthis[:,:]= _NAN #within the grid gets filled in below
        else:
            xsubmin= 0
            xsubmax= nvR
            ysubmin= 0
            ysubmax= nvT
        #Fill in this level
        plotthis[xsubmin:xsubmax,ysubmin:ysubmax]= plotdf/dvR/dvT/nvR/nvT
        #Plot
        if nUpperLevels == 0:
            xrange= [self.vRgrid[0]+dvR/2.,self.vRgrid[len(self.vRgrid)-1]-dvR/2.]
            yrange= [self.vTgrid[0]+dvT/2.,self.vTgrid[len(self.vTgrid)-1]-dvT/2.]
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
