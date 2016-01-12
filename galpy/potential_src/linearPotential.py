from __future__ import division, print_function

import os, os.path
import pickle
import numpy as nu
import galpy.util.bovy_plot as plot
from galpy.util import config
from galpy.potential_src.Potential import PotentialError
from galpy.util.bovy_conversion import physical_conversion
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
            self._ro= ro
            self._roSet= True
        if vo is None:
            self._vo= config.__config__.getfloat('normalization','vo')
            self._voSet= False
        else:
            self._vo= vo
            self._voSet= True
        return None

    @physical_conversion('energy',pop=True)
    def __call__(self,x,t=0.):
        """
        NAME:
           __call__
        PURPOSE:

           evaluate the potential

        INPUT:

           x - position

           t= time (optional)

        OUTPUT:

           Phi(x,t)

        HISTORY:

           2010-07-12 - Written - Bovy (NYU)

        """
        try:
            return self._amp*self._evaluate(x,t=t)
        except AttributeError: #pragma: no cover
            raise PotentialError("'_evaluate' function not implemented for this potential")

    @physical_conversion('force',pop=True)
    def force(self,x,t=0.):
        """
        NAME:

           force

        PURPOSE:

           evaluate the force

        INPUT:

           x - position

           t= time (optional)

        OUTPUT:

           F(x,t)

        HISTORY:

           2010-07-12 - Written - Bovy (NYU)

        """
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

@physical_conversion('energy',pop=True)
def evaluatelinearPotentials(Pot,x,t=0.):
    """
    NAME:

       evaluatelinearPotentials

    PURPOSE:

       evaluate the sum of a list of potentials

    INPUT:

       Pot - (list of) linearPotential instance(s)

       x - evaluate potentials at this position

       t - time to evaluate at

    OUTPUT:

       pot(x,t)

    HISTORY:

       2010-07-13 - Written - Bovy (NYU)

    """
    if isinstance(Pot,list):
        sum= 0.
        for pot in Pot:
            sum+= pot(x,t=t,use_physical=False)
        return sum
    elif isinstance(Pot,linearPotential):
        return Pot(x,t=t,use_physical=False)
    else: #pragma: no cover
        raise PotentialError("Input to 'evaluatelinearPotentials' is neither a linearPotential-instance or a list of such instances")

@physical_conversion('force',pop=True)
def evaluatelinearForces(Pot,x,t=0.):
    """
    NAME:

       evaluatelinearForces

    PURPOSE:

       evaluate the forces due to a list of potentials

    INPUT:

       Pot - (list of) linearPotential instance(s)

       x - evaluate forces at this position

       t - time to evaluate at

    OUTPUT:

       force(x,t)

    HISTORY:

       2010-07-13 - Written - Bovy (NYU)

    """
    if isinstance(Pot,list):
        sum= 0.
        for pot in Pot:
            sum+= pot.force(x,t=t,use_physical=False)
        return sum
    elif isinstance(Pot,linearPotential):
        return Pot.force(x,t=t,use_physical=False)
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

