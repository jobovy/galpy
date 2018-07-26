from __future__ import division, print_function

import os, os.path
import pickle
import numpy as nu
import galpy.util.bovy_plot as plot
from galpy.util import config
from .Potential import PotentialError, flatten
from galpy.util.bovy_conversion import physical_conversion,\
    potential_physical_input
_APY_LOADED= True
try:
    from astropy import units
except ImportError:
    _APY_LOADED= False
class linearPotential(object):
    """Class representing 1D potentials"""
    def __init__(self,amp=1.,ro=None,vo=None):
        self._amp= amp
        self.dim= 1
        self.isRZ= False
        self.hasC= False
        # Parse ro and vo
        if ro is None:
            self._ro= config.__config__.getfloat('normalization','ro')
            self._roSet= False
        else:
            if _APY_LOADED and isinstance(ro,units.Quantity):
                ro= ro.to(units.kpc).value
            self._ro= ro
            self._roSet= True
        if vo is None:
            self._vo= config.__config__.getfloat('normalization','vo')
            self._voSet= False
        else:
            if _APY_LOADED and isinstance(vo,units.Quantity):
                vo= vo.to(units.km/units.s).value
            self._vo= vo
            self._voSet= True
        return None

    def turn_physical_off(self):
        """
        NAME:

           turn_physical_off

        PURPOSE:

           turn off automatic returning of outputs in physical units

        INPUT:

           (none)

        OUTPUT:

           (none)

        HISTORY:

           2016-01-30 - Written - Bovy (UofT)

        """
        self._roSet= False
        self._voSet= False
        return None

    def turn_physical_on(self,ro=None,vo=None):
        """
        NAME:

           turn_physical_on

        PURPOSE:

           turn on automatic returning of outputs in physical units

        INPUT:

           ro= reference distance (kpc; can be Quantity)

           vo= reference velocity (km/s; can be Quantity)

        OUTPUT:

           (none)

        HISTORY:

           2016-01-30 - Written - Bovy (UofT)

        """
        self._roSet= True
        self._voSet= True
        if not ro is None:
            if _APY_LOADED and isinstance(ro,units.Quantity):
                ro= ro.to(units.kpc).value
            self._ro= ro
        if not vo is None:
            if _APY_LOADED and isinstance(vo,units.Quantity):
                vo= vo.to(units.km/units.s).value
            self._vo= vo
        return None

    @potential_physical_input
    @physical_conversion('energy',pop=True)
    def __call__(self,x,t=0.):
        """
        NAME:
           __call__
        PURPOSE:

           evaluate the potential

        INPUT:

           x - position (can be Quantity)

           t= time (optional; can be Quantity)

        OUTPUT:

           Phi(x,t)

        HISTORY:

           2010-07-12 - Written - Bovy (NYU)

        """
        return self._call_nodecorator(x,t=t)

    def _call_nodecorator(self,x,t=0.):
        # Separate, so it can be used during orbit integration
        try:
            return self._amp*self._evaluate(x,t=t)
        except AttributeError: #pragma: no cover
            raise PotentialError("'_evaluate' function not implemented for this potential")

    @potential_physical_input
    @physical_conversion('force',pop=True)
    def force(self,x,t=0.):
        """
        NAME:

           force

        PURPOSE:

           evaluate the force

        INPUT:

           x - position (can be Quantity)

           t= time (optional; can be Quantity)

        OUTPUT:

           F(x,t)

        HISTORY:

           2010-07-12 - Written - Bovy (NYU)

        """
        return self._force_nodecorator(x,t=t)

    def _force_nodecorator(self,x,t=0.):
        # Separate, so it can be used during orbit integration
        try:
            return self._amp*self._force(x,t=t)
        except AttributeError: #pragma: no cover
            raise PotentialError("'_force' function not implemented for this potential")

    def plot(self,t=0.,min=-15.,max=15,ns=21,savefilename=None):
        """
        NAME:

           plot

        PURPOSE:

           plot the potential

        INPUT:

           t - time to evaluate the potential at

           min - minimum x

           max - maximum x

           ns - grid in x

           savefilename - save to or restore from this savefile (pickle)

        OUTPUT:

           plot to output device

        HISTORY:

           2010-07-13 - Written - Bovy (NYU)

        """
        if not savefilename == None and os.path.exists(savefilename):
            print("Restoring savefile "+savefilename+" ...")
            savefile= open(savefilename,'rb')
            potx= pickle.load(savefile)
            xs= pickle.load(savefile)
            savefile.close()
        else:
            xs= nu.linspace(min,max,ns)
            potx= nu.zeros(ns)
            for ii in range(ns):
                potx[ii]= self._evaluate(xs[ii],t=t)
            if not savefilename == None:
                print("Writing savefile "+savefilename+" ...")
                savefile= open(savefilename,'wb')
                pickle.dump(potx,savefile)
                pickle.dump(xs,savefile)
                savefile.close()
        return plot.bovy_plot(xs,potx,
                              xlabel=r"$x/x_0$",ylabel=r"$\Phi(x)$",
                              xrange=[min,max])

@potential_physical_input
@physical_conversion('energy',pop=True)
def evaluatelinearPotentials(Pot,x,t=0.):
    """
    NAME:

       evaluatelinearPotentials

    PURPOSE:

       evaluate the sum of a list of potentials

    INPUT:

       Pot - (list of) linearPotential instance(s)

       x - evaluate potentials at this position (can be Quantity)

       t - time to evaluate at  (can be Quantity)

    OUTPUT:

       pot(x,t)

    HISTORY:

       2010-07-13 - Written - Bovy (NYU)

    """
    return _evaluatelinearPotentials(Pot,x,t=t)

def _evaluatelinearPotentials(Pot,x,t=0.):
    """Raw, undecorated function for internal use"""
    if isinstance(Pot,list):
        sum= 0.
        for pot in Pot:
            sum+= pot._call_nodecorator(x,t=t)
        return sum
    elif isinstance(Pot,linearPotential):
        return Pot._call_nodecorator(x,t=t)
    else: #pragma: no cover
        raise PotentialError("Input to 'evaluatelinearPotentials' is neither a linearPotential-instance or a list of such instances")

@potential_physical_input
@physical_conversion('force',pop=True)
def evaluatelinearForces(Pot,x,t=0.):
    """
    NAME:

       evaluatelinearForces

    PURPOSE:

       evaluate the forces due to a list of potentials

    INPUT:

       Pot - (list of) linearPotential instance(s)

       x - evaluate forces at this position (can be Quantity)

       t - time to evaluate at (can be Quantity)

    OUTPUT:

       force(x,t)

    HISTORY:

       2010-07-13 - Written - Bovy (NYU)

    """
    return _evaluatelinearForces(Pot,x,t=t)

def _evaluatelinearForces(Pot,x,t=0.):
    """Raw, undecorated function for internal use"""
    if isinstance(Pot,list):
        sum= 0.
        for pot in Pot:
            sum+= pot._force_nodecorator(x,t=t)
        return sum
    elif isinstance(Pot,linearPotential):
        return Pot._force_nodecorator(x,t=t)
    else: #pragma: no cover
        raise PotentialError("Input to 'evaluateForces' is neither a linearPotential-instance or a list of such instances")

def plotlinearPotentials(Pot,t=0.,min=-15.,max=15,ns=21,savefilename=None):
    """
    NAME:

       plotlinearPotentials

    PURPOSE:

       plot a combination of potentials

    INPUT:

       t - time to evaluate potential at

       min - minimum x

       max - maximum x

       ns - grid in x

       savefilename - save to or restore from this savefile (pickle)

    OUTPUT:

       plot to output device

    HISTORY:

       2010-07-13 - Written - Bovy (NYU)

    """
    Pot= flatten(Pot)
    if not savefilename == None and os.path.exists(savefilename):
        print("Restoring savefile "+savefilename+" ...")
        savefile= open(savefilename,'rb')
        potx= pickle.load(savefile)
        xs= pickle.load(savefile)
        savefile.close()
    else:
        xs= nu.linspace(min,max,ns)
        potx= nu.zeros(ns)
        for ii in range(ns):
            potx[ii]= evaluatelinearPotentials(Pot,xs[ii],t=t)
        if not savefilename == None:
            print("Writing savefile "+savefilename+" ...")
            savefile= open(savefilename,'wb')
            pickle.dump(potx,savefile)
            pickle.dump(xs,savefile)
            savefile.close()
    return plot.bovy_plot(xs,potx,
                          xlabel=r"$x/x_0$",ylabel=r"$\Phi(x)$",
                          xrange=[min,max])

