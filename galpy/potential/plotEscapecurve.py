from __future__ import division, print_function

import os
import pickle
import numpy
from ..util import plot, conversion
from ..util.conversion import physical_conversion,\
    potential_physical_input
_INF= 10**12.
def plotEscapecurve(Pot,*args,**kwargs):
    """
    NAME:

       plotEscapecurve

    PURPOSE:

       plot the escape velocity curve for this potential (in the z=0 plane for
       non-spherical potentials)

    INPUT:

       Pot - Potential or list of Potential instances

       Rrange= Range in R to consider (can be Quantity)

       grid= grid in R

       savefilename= save to or restore from this savefile (pickle)

       +galpy.util.plot.plot args and kwargs

    OUTPUT:

       plot to output device

    HISTORY:

       2010-08-08 - Written - Bovy (NYU)

    """
    # Using physical units or not?
    if isinstance(Pot,list):
        potro= Pot[0]._ro
        roSet= Pot[0]._roSet
        potvo= Pot[0]._vo
        voSet= Pot[0]._voSet
    else:
        potro= Pot._ro
        roSet= Pot._roSet
        potvo= Pot._vo
        voSet= Pot._voSet
    if (kwargs.get('use_physical',False) \
            and kwargs.get('ro',roSet) and kwargs.get('vo',voSet)) or \
            (not 'use_physical' in kwargs \
                 and kwargs.get('ro',roSet) and kwargs.get('vo',voSet)):
        use_physical= True
        potro= kwargs.get('ro',potro)
        potvo= kwargs.get('vo',potvo)
        xlabel= r'$R\,(\mathrm{kpc})$'
        ylabel= r"$v_e(R)\,(\mathrm{km\,s}^{-1})$"
        Rrange= kwargs.pop('Rrange',[0.01*potro,5.*potro])
    else:
        use_physical= False
        xlabel= r"$R/R_0$"
        ylabel= r"$v_e(R)/v_c(R_0)$"
        Rrange= kwargs.pop('Rrange',[0.01,5.])
    # Parse ro
    potro= conversion.parse_length_kpc(potro)
    potvo= conversion.parse_velocity_kms(potvo)
    Rrange[0]= conversion.parse_length_kpc(Rrange[0])
    Rrange[1]= conversion.parse_length_kpc(Rrange[1])
    if use_physical:
        Rrange[0]/= potro
        Rrange[1]/= potro
    grid= kwargs.pop('grid',1001)
    savefilename= kwargs.pop('savefilename',None)
    if not savefilename == None and os.path.exists(savefilename):
        print("Restoring savefile "+savefilename+" ...")
        savefile= open(savefilename,'rb')
        esccurve= pickle.load(savefile)
        Rs= pickle.load(savefile)
        savefile.close()
    else:
        Rs= numpy.linspace(Rrange[0],Rrange[1],grid)
        esccurve= calcEscapecurve(Pot,Rs)
        if not savefilename == None:
            print("Writing savefile "+savefilename+" ...")
            savefile= open(savefilename,'wb')
            pickle.dump(esccurve,savefile)
            pickle.dump(Rs,savefile)
            savefile.close()
    if use_physical:
        Rs*= potro
        esccurve*= potvo
        Rrange[0]*= potro
        Rrange[1]*= potro
    if not 'xlabel' in kwargs:
        kwargs['xlabel']= xlabel
    if not 'ylabel' in kwargs:
        kwargs['ylabel']= ylabel
    if not 'xrange' in kwargs:
        kwargs['xrange']= Rrange
    if not 'yrange' in kwargs:
        kwargs['yrange']=\
            [0.,1.2*numpy.amax(esccurve[True^numpy.isnan(esccurve)])]
    kwargs.pop('ro',None)
    kwargs.pop('vo',None)
    kwargs.pop('use_physical',None)
    return plot.plot(Rs,esccurve,*args,**kwargs)

def calcEscapecurve(Pot,Rs,t=0.):
    """
    NAME:
       calcEscapecurve
    PURPOSE:
       calculate the escape velocity curve for this potential (in the 
       z=0 plane for non-spherical potentials)
    INPUT:
       Pot - Potential or list of Potential instances

       Rs - (array of) radius(i)

       t - instantaneous time (optional)

    OUTPUT:
       array of v_esc
    HISTORY:
       2011-04-16 - Written - Bovy (NYU)
    """
    isList= isinstance(Pot,list)
    isNonAxi= ((isList and Pot[0].isNonAxi) or (not isList and Pot.isNonAxi))
    if isNonAxi:
        raise AttributeError("Escape velocity curve plotting for non-axisymmetric potentials is not currently supported")
    try:
        grid= len(Rs)
    except TypeError:
        grid=1
        Rs= numpy.array([Rs])
    esccurve= numpy.zeros(grid)
    for ii in range(grid):
        esccurve[ii]= vesc(Pot,Rs[ii],t=t,use_physical=False)
    return esccurve

@potential_physical_input
@physical_conversion('velocity',pop=True)
def vesc(Pot,R,t=0.):
    """

    NAME:

        vesc

    PURPOSE:

       calculate the escape velocity at R for potential Pot

    INPUT:

       Pot - Potential instances or list thereof

       R - Galactocentric radius (can be Quantity)

       t - time (optional; can be Quantity)

    OUTPUT:

       escape velocity

    HISTORY:

       2011-10-09 - Written - Bovy (IAS)

    """
    from ..potential import evaluateplanarPotentials
    from ..potential import PotentialError
    try:
        return numpy.sqrt(2.*(evaluateplanarPotentials(Pot,_INF,t=t,use_physical=False)-evaluateplanarPotentials(Pot,R,t=t,use_physical=False)))
    except PotentialError:
        from ..potential import RZToplanarPotential
        Pot= RZToplanarPotential(Pot)
        return numpy.sqrt(2.*(evaluateplanarPotentials(Pot,_INF,t=t,use_physical=False)-evaluateplanarPotentials(Pot,R,t=t,use_physical=False)))
        
