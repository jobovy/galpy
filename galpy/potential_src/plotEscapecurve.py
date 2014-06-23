import os
import pickle
import numpy as nu
import galpy.util.bovy_plot as plot
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

       Rrange= Range in R to consider

       grid= grid in R

       savefilename= save to or restore from this savefile (pickle)

       +bovy_plot.bovy_plot args and kwargs

    OUTPUT:

       plot to output device

    HISTORY:

       2010-08-08 - Written - Bovy (NYU)

    """
    if kwargs.has_key('Rrange'):
        Rrange= kwargs['Rrange']
        kwargs.pop('Rrange')
    else:
        Rrange= [0.01,5.]
    if kwargs.has_key('grid'):
        grid= kwargs['grid']
        kwargs.pop('grid')
    else:
        grid= 1001
    if kwargs.has_key('savefilename'):
        savefilename= kwargs['savefilename']
        kwargs.pop('savefilename')
    else:
        savefilename= None
    if not savefilename == None and os.path.exists(savefilename):
        print "Restoring savefile "+savefilename+" ..."
        savefile= open(savefilename,'rb')
        esccurve= pickle.load(savefile)
        Rs= pickle.load(savefile)
        savefile.close()
    else:
        Rs= nu.linspace(Rrange[0],Rrange[1],grid)
        esccurve= calcEscapecurve(Pot,Rs)
        if not savefilename == None:
            print "Writing savefile "+savefilename+" ..."
            savefile= open(savefilename,'wb')
            pickle.dump(esccurve,savefile)
            pickle.dump(Rs,savefile)
            savefile.close()
    if not kwargs.has_key('xlabel'):
        kwargs['xlabel']= r"$R/R_0$"
    if not kwargs.has_key('ylabel'):
        kwargs['ylabel']= r"$v_e(R)/v_c(R_0)$"
    if not kwargs.has_key('xrange'):
        kwargs['xrange']= Rrange
    if not kwargs.has_key('yrange'):
        kwargs['yrange']= [0.,1.2*nu.amax(esccurve)]
    return plot.bovy_plot(Rs,esccurve,*args,
                          **kwargs)

def calcEscapecurve(Pot,Rs):
    """
    NAME:
       calcEscapecurve
    PURPOSE:
       calculate the escape velocity curve for this potential (in the 
       z=0 plane for non-spherical potentials)
    INPUT:
       Pot - Potential or list of Potential instances

       Rs - (array of) radius(i)
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
        Rs= nu.array([Rs])
    esccurve= nu.zeros(grid)
    from planarPotential import evaluateplanarPotentials
    from Potential import PotentialError
    for ii in range(grid):
        try:
            esccurve[ii]= nu.sqrt(2.*(evaluateplanarPotentials(_INF,Pot)-evaluateplanarPotentials(Rs[ii],Pot)))
        except PotentialError:
            from planarPotential import RZToplanarPotential
            Pot= RZToplanarPotential(Pot)
            esccurve[ii]= nu.sqrt(2.*(evaluateplanarPotentials(_INF,Pot)-evaluateplanarPotentials(Rs[ii],Pot)))
    return esccurve

def vesc(Pot,R):
    """

    NAME:

        vesc

    PURPOSE:

       calculate the escape velocity at R for potential Pot

    INPUT:

       Pot - Potential instances or list thereof

       R - Galactocentric radius

    OUTPUT:

       escape velocity

    HISTORY:

       2011-10-09 - Written - Bovy (IAS)

    """
    from planarPotential import evaluateplanarPotentials
    from Potential import PotentialError
    try:
        return nu.sqrt(2.*(evaluateplanarPotentials(_INF,Pot)-evaluateplanarPotentials(R,Pot)))
    except PotentialError:
        from planarPotential import RZToplanarPotential
        Pot= RZToplanarPotential(Pot)
        return nu.sqrt(2.*(evaluateplanarPotentials(_INF,Pot)-evaluateplanarPotentials(R,Pot)))
        
