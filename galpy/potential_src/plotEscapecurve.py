import numpy as nu
import galpy.util.bovy_plot as plot
_INF= 1000000.
def plotEscapecurve(Pot,*args,**kwargs):
    """
    NAME:
       plotEscapecurve
    PURPOSE:
       plot the escape velocity curve for this potential (in the z=0 plane for
       non-spherical potentials)
    INPUT:
       Pot - Potential or list of Potential instances
       Rrange - 
       grid - grid in R
       savefilename - save to or restore from this savefile (pickle)
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
    isList= isinstance(Pot,list)
    isNonAxi= ((isList and Pot[0].isNonAxi) or (not isList and Pot.isNonAxi))
    if isNonAxi:
        raise AttributeError("Escape velocity curve plotting for non-axisymmetric potentials is not currently supported")
    if not savefilename == None and os.path.exists(savefilename):
        print "Restoring savefile "+savefilename+" ..."
        savefile= open(savefilename,'rb')
        rotcurve= pickle.load(savefile)
        Rs= pickle.load(savefile)
        savefile.close()
    else:
        Rs= nu.linspace(Rrange[0],Rrange[1],grid)
        rotcurve= nu.zeros(grid)
        from planarPotential import evaluateplanarPotentials
        for ii in range(grid):
            try:
                rotcurve[ii]= nu.sqrt(2.*(evaluateplanarPotentials(_INF,Pot)-evaluateplanarPotentials(Rs[ii],Pot)))
            except TypeError:
                from planarPotential import RZToplanarPotential
                Pot= RZToplanarPotential(Pot)
                rotcurve[ii]= nu.sqrt(2.*(evaluateplanarPotentials(_INF,Pot)-evaluateplanarPotentials(Rs[ii],Pot)))
        if not savefilename == None:
            print "Writing savefile "+savefilename+" ..."
            savefile= open(savefilename,'wb')
            pickle.dump(rotcurve,savefile)
            pickle.dump(Rs,savefile)
            savefile.close()
    if not kwargs.has_key('xlabel'):
        kwargs['xlabel']= r"$R/R_0$"
    if not kwargs.has_key('ylabel'):
        kwargs['ylabel']= r"$v_e(R)/v_c(R_0)$"
    kwargs['xrange']= Rrange
    return plot.bovy_plot(Rs,rotcurve,*args,
                          **kwargs)

