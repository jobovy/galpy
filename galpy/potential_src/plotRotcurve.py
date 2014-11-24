import os
import pickle
import numpy as nu
import galpy.util.bovy_plot as plot
def plotRotcurve(Pot,*args,**kwargs):
    """
    NAME:

       plotRotcurve

    PURPOSE:

       plot the rotation curve for this potential (in the z=0 plane for
       non-spherical potentials)

    INPUT:

       Pot - Potential or list of Potential instances

       Rrange - Range in R to consider

       grid= grid in R

       savefilename= save to or restore from this savefile (pickle)

       +bovy_plot.bovy_plot args and kwargs

    OUTPUT:

       plot to output device

    HISTORY:

       2010-07-10 - Written - Bovy (NYU)

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
    if not savefilename is None and os.path.exists(savefilename):
        print "Restoring savefile "+savefilename+" ..."
        savefile= open(savefilename,'rb')
        rotcurve= pickle.load(savefile)
        Rs= pickle.load(savefile)
        savefile.close()
    else:
        Rs= nu.linspace(Rrange[0],Rrange[1],grid)
        rotcurve= calcRotcurve(Pot,Rs)
        if not savefilename == None:
            print "Writing savefile "+savefilename+" ..."
            savefile= open(savefilename,'wb')
            pickle.dump(rotcurve,savefile)
            pickle.dump(Rs,savefile)
            savefile.close()
    if not kwargs.has_key('xlabel'):
        kwargs['xlabel']= r"$R/R_0$"
    if not kwargs.has_key('ylabel'):
        kwargs['ylabel']= r"$v_c(R)/v_c(R_0)$"
    if not kwargs.has_key('xrange'):
        kwargs['xrange']= Rrange
    if not kwargs.has_key('yrange'):
        kwargs['yrange']= [0.,1.2*nu.amax(rotcurve)]
    return plot.bovy_plot(Rs,rotcurve,*args,
                          **kwargs)

def calcRotcurve(Pot,Rs):
    """
    NAME:
       calcRotcurve
    PURPOSE:
       calculate the rotation curve for this potential (in the z=0 plane for
       non-spherical potentials)
    INPUT:
       Pot - Potential or list of Potential instances

       Rs - (array of) radius(i)
    OUTPUT:
       array of vc
    HISTORY:
       2011-04-13 - Written - Bovy (NYU)
    """
    isList= isinstance(Pot,list)
    isNonAxi= ((isList and Pot[0].isNonAxi) or (not isList and Pot.isNonAxi))
    if isNonAxi:
        raise AttributeError("Rotation curve plotting for non-axisymmetric potentials is not currently supported")
    try:
        grid= len(Rs)
    except TypeError:
        grid=1
        Rs= nu.array([Rs])
    rotcurve= nu.zeros(grid)
    for ii in range(grid):
        rotcurve[ii]= vcirc(Pot,Rs[ii])
    return rotcurve

def vcirc(Pot,R):
    """

    NAME:

       vcirc

    PURPOSE:

       calculate the circular velocity at R in potential Pot

    INPUT:

       Pot - Potential instance or list of such instances

       R - Galactocentric radius

    OUTPUT:

       circular rotation velocity

    HISTORY:

       2011-10-09 - Written - Bovy (IAS)

    """
    from planarPotential import evaluateplanarRforces
    from Potential import PotentialError
    try:
        return nu.sqrt(-R*evaluateplanarRforces(R,Pot))
    except PotentialError:
        from planarPotential import RZToplanarPotential
        Pot= RZToplanarPotential(Pot)
        return nu.sqrt(-R*evaluateplanarRforces(R,Pot))

def dvcircdR(Pot,R):
    """

    NAME:

       dvcircdR

    PURPOSE:

       calculate the derivative of the circular velocity wrt R at R in potential Pot

    INPUT:

       Pot - Potential instance or list of such instances

       R - Galactocentric radius

    OUTPUT:

       derivative of the circular rotation velocity wrt R

    HISTORY:

       2013-01-08 - Written - Bovy (IAS)

    """
    from planarPotential import evaluateplanarRforces
    from planarPotential import evaluateplanarR2derivs
    from Potential import PotentialError
    tvc= vcirc(Pot,R)
    try:
        return 0.5*(-evaluateplanarRforces(R,Pot)+R*evaluateplanarR2derivs(R,Pot))/tvc
    except PotentialError:
        from planarPotential import RZToplanarPotential
        Pot= RZToplanarPotential(Pot)
        return 0.5*(-evaluateplanarRforces(R,Pot)+R*evaluateplanarR2derivs(R,Pot))/tvc

