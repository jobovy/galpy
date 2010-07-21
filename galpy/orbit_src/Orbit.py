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
