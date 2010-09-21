from FullOrbit import FullOrbit
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
        self.vxvv= vxvv
        if len(vxvv) == 2:
            self._orb= linearOrbit(vxvv=vxvv)
        elif len(vxvv) == 3:
            self._orb= planarROrbit(vxvv=vxvv)
        elif len(vxvv) == 4:
            self._orb= planarOrbit(vxvv=vxvv)
        elif len(vxvv) == 5:
            self._orb= RZOrbit(vxvv=vxvv)
        elif len(vxvv) == 6:
            self._orb= FullOrbit(vxvv=vxvv)

    def setphi(self,phi):
        """
        NAME:
           setphi
        PURPOSE:
           set initial azimuth
        INPUT:
           phi - desired azimuth
        OUTPUT:
           (none)
        HISTORY:
           2010-08-01 - Written - Bovy (NYU)
        BUGS:
           Should perform check that this orbit has phi
        """
        if len(self.vxvv) == 2:
            raise AttributeError("One-dimensional orbit has no azimuth")
        elif len(self.vxvv) == 3:
            #Upgrade
            vxvv= [self.vxvv[0],self.vxvv[1],self.vxvv[2],phi]
            self.vxvv= vxvv
            self._orb= planarROrbit(vxvv=vxvv)
        elif len(self.vxvv) == 4:
            self.vxvv[-1]= phi
            self._orb.vxvv[-1]= phi
        elif len(self.vxvv) == 5:
            #Upgrade
            vxvv= [self.vxvv[0],self.vxvv[1],self.vxvv[2],self.vxvv[3],
                   self.vxvv[4],phi]
            self.vxvv= vxvv
            self._orb= FullOrbit(vxvv=vxvv)
        elif len(self.vxvv) == 6:
            self.vxvv[-1]= phi
            self._orb.vxvv[-1]= phi

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

    def E(self,pot=None):
        """
        NAME:
           E
        PURPOSE:
           calculate the energy
        INPUT:
           pot=
        OUTPUT:
           energy
        HISTORY:
           2010-09-15 - Written - Bovy (NYU)
        """
        return self._orb.E(pot=pot)

    def e(self):
        """
        NAME:
           e
        PURPOSE:
           calculate the eccentricity
        INPUT:
        OUTPUT:
           eccentricity
        HISTORY:
           2010-09-15 - Written - Bovy (NYU)
        """
        return self._orb.e()

    def rap(self):
        """
        NAME:
           rap
        PURPOSE:
           calculate the apocenter radius
        INPUT:
        OUTPUT:
           R_ap
        HISTORY:
           2010-09-20 - Written - Bovy (NYU)
        """
        return self._orb.rap()

    def rperi(self):
        """
        NAME:
           rperi
        PURPOSE:
           calculate the pericenter radius
        INPUT:
        OUTPUT:
           R_peri
        HISTORY:
           2010-09-20 - Written - Bovy (NYU)
        """
        return self._orb.rperi()

    def zmax(self):
        """
        NAME:
           zmax
        PURPOSE:
           calculate the maximum vertical height
        INPUT:
        OUTPUT:
           Z_max
        HISTORY:
           2010-09-20 - Written - Bovy (NYU)
        """
        return self._orb.zmax()

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

    def plotEz(self,pot,*args,**kwargs):
        """
        NAME:
           plotEz
        PURPOSE:
           plot E_z(.) along the orbit
        INPUT:
           pot - Potential instance or list of instances in which the orbit was
                 integrated
           d1= - plot Ez vs d1: e.g., 't', 'z', 'R'
           +bovy_plot.bovy_plot inputs
        OUTPUT:
           figure to output device
        HISTORY:
           2010-07-10 - Written - Bovy (NYU)
        """
        self._orb.plotEz(pot,*args,**kwargs)

    def plotEzJz(self,pot,*args,**kwargs):
        """
        NAME:
           plotEzJzt
        PURPOSE:
           plot E_z(t)/sqrt(dens(R)) along the orbit
        INPUT:
           pot - Potential instance or list of instances in which the orbit was
                 integrated
           d1= - plot Ez vs d1: e.g., 't', 'z', 'R'
           +bovy_plot.bovy_plot inputs
        OUTPUT:
           figure to output device
        HISTORY:
           2010-08-08 - Written - Bovy (NYU)
        """
        self._orb.plotEzJz(pot,*args,**kwargs)

    def __call__(self,*args,**kwargs):
        """
        NAME:
           __call__
        PURPOSE:
           return the orbit vector at time t
        INPUT:
           t - desired time
           rect - if true, return rectangular coordinates
        OUTPUT:
           [x,vx], [R,vR,vT,z,vz(,phi)] or [R,vR,vT(,phi)] depending on 
           the orbit
           if rect: [x,y(,z),vx,vy(,vz)]
        BUGS:
           currently only works for times at which the orbit was requested
           during integration; use interpolation in between?
        HISTORY:
           2010-07-10 - Written - Bovy (NYU)
        """
        return self._orb(*args,**kwargs)

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
        if (not (isinstance(self._orb,planarROrbit) and 
                isinstance(linOrb._orb,linearOrbit)) and
            not (isinstance(self._orb,linearOrbit) and 
                 isinstance(linOrb._orb,planarROrbit))):
            raise AttributeError("Only planarROrbit+linearOrbit is supported")
        if isinstance(self._orb,planarROrbit):
            return Orbit(vxvv=[self._orb.vxvv[0],self._orb.vxvv[1],
                               self._orb.vxvv[2],
                               linOrb._orb.vxvv[0],linOrb._orb.vxvv[1]])
        else:
            return Orbit(vxvv=[linOrb._orb.vxvv[0],linOrb._orb.vxvv[1],
                               linOrb._orb.vxvv[2],
                               self._orb.vxvv[0],self._orb.vxvv[1]])

    #4 pickling
    def __getinitargs__(self):
        return (self.vxvv,)

    def __getstate__(self):
        return self.vxvv
    
    def __setstate__(self,state):
        self.vxvv= state
