from RZOrbit import RZOrbit
from planarOrbit import planarOrbit, planarROrbit
from linearOrbit import linearOrbit
class Orbit:
    """General orbit class representing an orbit"""
    def __init__(self,vxvv=None):
        """
        NAME:
           __init__
        PURPOSE:
           Initialize an Orbit instance
        INPUT:
           vxvv - initial conditions
        OUTPUT:
           instance
        HISTORY:
           2010-07-20 - Written - Bovy (NYU)
        """
        if len(vxvv) == 2:
            self._orb= linearOrbit(vxvv=vxvv)
        elif len(vxvv) == 3:
            self._orb= planarROrbit(vxvv=vxvv)
        elif len(vxvv) == 4:
            self._orb= planarOrbit(vxvv=vxvv)
        elif len(vxvv) == 5:
            self._orb= RZOrbit(vxvv=vxvv)
        elif len(vxvv) == 6:
            pass

    def integrate(self,t,pot):
        """
        NAME:
           integrate
        PURPOSE:
           integrate the orbit
        INPUT:
           t - list of times at which to output (0 has to be in this!)
           pot - potential instance or list of instances
        OUTPUT:
           (none) (get the actual orbit using getOrbit()
        HISTORY:
           2010-07-10
        """
        self._orb.integrate(t,pot)

    def plot(self,*args,**kwargs):
        """
        NAME:
           plot
        PURPOSE:
           plot a previously calculated orbit
        INPUT:
           matplotlib.plot inputs+bovy_plot.plot inputs
        OUTPUT:
           sends plot to output device
        HISTORY:
           2010-07-10 - Written - Bovy (NYU)
        """
        self._orb.plot(*args,**kwargs)

    def plotEt(self,pot,*args,**kwargs):
        """
        NAME:
           plotEt
        PURPOSE:
           plot E(t) along the orbit
        INPUT:
           pot - Potential instance or list of instances in which the orbit was
                 integrated
           +bovy_plot.bovy_plot inputs
        OUTPUT:
           figure to output device
        HISTORY:
           2010-07-10 - Written - Bovy (NYU)
        """
        self._orb.plotEt(pot,*args,**kwargs)

    def plotEzt(self,pot,*args,**kwargs):
        """
        NAME:
           plotEzt
        PURPOSE:
           plot E_z(t) along the orbit
        INPUT:
           pot - Potential instance or list of instances in which the orbit was
                 integrated
           +bovy_plot.bovy_plot inputs
        OUTPUT:
           figure to output device
        HISTORY:
           2010-07-10 - Written - Bovy (NYU)
        """
        self._orb.plotEzt(pot,*args,**kwargs)

    def __call__(self,t):
        """
        NAME:
           __call__
        PURPOSE:
           return the orbit vector at time t
        INPUT:
           t - desired time
        OUTPUT:
           [x,vx], [R,vR,vT,z,vz(,phi)] or [R,vR,vT(,phi)] depending on 
           the orbit
        BUGS:
           currently only works for times at which the orbit was requested
           during integration; use interpolation in between?
        HISTORY:
           2010-07-10 - Written - Bovy (NYU)
        """
        return self._orb(t)

    def plotRt(self,*args,**kwargs):
        """
        NAME:
           plotRt
        PURPOSE:
           plot R(t) along the orbit
        INPUT:
           bovy_plot.bovy_plot inputs
        OUTPUT:
           figure to output device
        HISTORY:
           2010-07-10 - Written - Bovy (NYU)
        """
        self._orb.plotRt(*args,**kwargs)

    def plotzt(self,*args,**kwargs):
        """
        NAME:
           plotzt
        PURPOSE:
           plot z(t) along the orbit
        INPUT:
           bovy_plot.bovy_plot inputs
        OUTPUT:
           figure to output device
        HISTORY:
           2010-07-10 - Written - Bovy (NYU)
        """
        self._orb.plotzt(*args,**kwargs)

    def plotvRt(self,*args,**kwargs):
        """
        NAME:
           plotvRt
        PURPOSE:
           plot vR(t) along the orbit
        INPUT:
           bovy_plot.bovy_plot inputs
        OUTPUT:
           figure to output device
        HISTORY:
           2010-07-10 - Written - Bovy (NYU)
        """
        self._orb.plotvRt(*args,**kwargs)

    def plotvTt(self,*args,**kwargs):
        """
        NAME:
           plotvTt
        PURPOSE:
           plot vT(t) along the orbit
        INPUT:
           bovy_plot.bovy_plot inputs
        OUTPUT:
           figure to output device
        HISTORY:
           2010-07-10 - Written - Bovy (NYU)
        """
        self._orb.plotvTt(*args,**kwargs)

    def plotphit(self,*args,**kwargs):
        """
        NAME:
           plotphit
        PURPOSE:
           plot \phi(t) along the orbit
        INPUT:
           bovy_plot.bovy_plot inputs
        OUTPUT:
           figure to output device
        HISTORY:
           2010-07-10 - Written - Bovy (NYU)
        """
        self._orb.plotphit(*args,**kwargs)

    def plotvzt(self,*args,**kwargs):
        """
        NAME:
           plotvzt
        PURPOSE:
           plot vz(t) along the orbit
        INPUT:
           bovy_plot.bovy_plot inputs
        OUTPUT:
           figure to output device
        HISTORY:
           2010-07-10 - Written - Bovy (NYU)
        """
        self._orb.plotvzt(*args,**kwargs)

    def plotxt(self,*args,**kwargs):
        """
        NAME:
           plotxt
        PURPOSE:
           plot a one-dimensional orbit position
        INPUT:
           bovy_plot.bovy_plot inputs
        OUTPUT:
           figure to output device
        HISTORY:
           2010-07-21 - Written - Bovy (NYU)
        """
        self._orb.plotxt(*args,**kwargs)

    def plotvxt(self,*args,**kwargs):
        """
        NAME:
           plotvxt
        PURPOSE:
           plot a one-dimensional orbit velocity
        INPUT:
           bovy_plot.bovy_plot inputs
        OUTPUT:
           figure to output device
        HISTORY:
           2010-07-21 - Written - Bovy (NYU)
        """
        self._orb.plotvxt(*args,**kwargs)

    def __add__(self,linOrb):
        if not isinstance(self._orb,planarROrbit) or \
                not isinstance(linOrb._orb,linearOrbit):
            raise AttributeError("Only planarROrbit+linearOrbit is supported")
        return Orbit(vxvv=[self._orb.vxvv[0],self._orb.vxvv[1],
                           self._orb.vxvv[2],
                           linOrb._orb.vxvv[0],linOrb._orb.vxvv[1]])
