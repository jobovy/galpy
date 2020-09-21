from __future__ import division, print_function

import os, os.path
import copy
import pickle
import numpy
from ..util import plot, conversion, config
from .Potential import PotentialError, flatten
from ..util.conversion import physical_conversion,\
    potential_physical_input, physical_compatible
class linearPotential(object):
    """Class representing 1D potentials"""
    def __init__(self,amp=1.,ro=None,vo=None):
        self._amp= amp
        self.dim= 1
        self.isRZ= False
        self.hasC= False
        self.hasC_dxdv= False
        self.hasC_dens= False
        # Parse ro and vo
        if ro is None:
            self._ro= config.__config__.getfloat('normalization','ro')
            self._roSet= False
        else:
            ro= conversion.parse_length_kpc(ro)
            self._ro= ro
            self._roSet= True
        if vo is None:
            self._vo= config.__config__.getfloat('normalization','vo')
            self._voSet= False
        else:
            vo= conversion.parse_velocity_kms(vo)
            self._vo= vo
            self._voSet= True
        return None

    def __mul__(self,b):
        """
        NAME:

           __mul__

        PURPOSE:

           Multiply a linearPotential's amplitude by a number

        INPUT:

           b - number

        OUTPUT:

           New instance with amplitude = (old amplitude) x b

        HISTORY:

           2019-01-27 - Written - Bovy (UofT)

        """
        if not isinstance(b,(int,float)):
            raise TypeError("Can only multiply a planarPotential instance with a number")
        out= copy.deepcopy(self)
        out._amp*= b
        return out
    # Similar functions
    __rmul__= __mul__
    def __div__(self,b): return self.__mul__(1./b)
    __truediv__= __div__

    def __add__(self,b):
        """
        NAME:

           __add__

        PURPOSE:

           Add linearPotential instances together to create a multi-component potential (e.g., pot= pot1+pot2+pot3)

        INPUT:

           b - linearPotential instance or a list thereof

        OUTPUT:

           List of linearPotential instances that represents the combined potential

        HISTORY:

           2019-01-27 - Written - Bovy (UofT)

        """
        from ..potential import flatten as flatten_pot
        if not isinstance(flatten_pot([b])[0],linearPotential):
            raise TypeError("""Can only combine galpy linearPotential"""
                            """ objects with """
                            """other such objects or lists thereof""")
        assert physical_compatible(self,b), \
            """Physical unit conversion parameters (ro,vo) are not """\
            """compatible between potentials to be combined"""
        if isinstance(b,list):
            return [self]+b
        else:
            return [self,b]
    # Define separately to keep order
    def __radd__(self,b):
        from ..potential import flatten as flatten_pot
        if not isinstance(flatten_pot([b])[0],linearPotential):
            raise TypeError("""Can only combine galpy linearPotential"""
                            """ objects with """
                            """other such objects or lists thereof""")
        assert physical_compatible(self,b), \
            """Physical unit conversion parameters (ro,vo) are not """\
            """compatible between potentials to be combined"""
        # If we get here, b has to be a list
        return b+[self]

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

           2020-04-22 - Don't turn on a parameter when it is False - Bovy (UofT)

        """
        if not ro is False: self._roSet= True
        if not vo is False: self._voSet= True
        if not ro is None and ro:
            self._ro= conversion.parse_length_kpc(ro)
        if not vo is None and vo:
            self._vo= conversion.parse_velocity_kms(vo)
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
            xs= numpy.linspace(min,max,ns)
            potx= numpy.zeros(ns)
            for ii in range(ns):
                potx[ii]= self._evaluate(xs[ii],t=t)
            if not savefilename == None:
                print("Writing savefile "+savefilename+" ...")
                savefile= open(savefilename,'wb')
                pickle.dump(potx,savefile)
                pickle.dump(xs,savefile)
                savefile.close()
        return plot.plot(xs,potx,
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
        xs= numpy.linspace(min,max,ns)
        potx= numpy.zeros(ns)
        for ii in range(ns):
            potx[ii]= evaluatelinearPotentials(Pot,xs[ii],t=t)
        if not savefilename == None:
            print("Writing savefile "+savefilename+" ...")
            savefile= open(savefilename,'wb')
            pickle.dump(potx,savefile)
            pickle.dump(xs,savefile)
            savefile.close()
    return plot.plot(xs,potx,
                     xlabel=r"$x/x_0$",ylabel=r"$\Phi(x)$",
                     xrange=[min,max])

