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

    def R(self,*args,**kwargs):
        """
        NAME:
           R
        PURPOSE:
           return cylindrical radius at time t
        INPUT:
           t - (optional) time at which to get the radius
        OUTPUT:
           R(t)
        HISTORY:
           2010-09-21 - Written - Bovy (NYU)
        """
        return self._orb.R(*args,**kwargs)

    def vR(self,*args,**kwargs):
        """
        NAME:
           vR
        PURPOSE:
           return radial velocity at time t
        INPUT:
           t - (optional) time at which to get the radial velocity
        OUTPUT:
           vR(t)
        HISTORY:
           2010-09-21 - Written - Bovy (NYU)
        """
        return self._orb.vR(*args,**kwargs)

    def vT(self,*args,**kwargs):
        """
        NAME:
           vT
        PURPOSE:
           return tangential velocity at time t
        INPUT:
           t - (optional) time at which to get the tangential velocity
        OUTPUT:
           vT(t)
        HISTORY:
           2010-09-21 - Written - Bovy (NYU)
        """
        return self._orb.vT(*args,**kwargs)

    def z(self,*args,**kwargs):
        """
        NAME:
           z
        PURPOSE:
           return vertical height
        INPUT:
           t - (optional) time at which to get the vertical height
        OUTPUT:
           z(t)
        HISTORY:
           2010-09-21 - Written - Bovy (NYU)
        """
        return self._orb.z(*args,**kwargs)

    def vz(self,*args,**kwargs):
        """
        NAME:
           vz
        PURPOSE:
           return vertical velocity
        INPUT:
           t - (optional) time at which to get the vertical velocity
        OUTPUT:
           vz(t)
        HISTORY:
           2010-09-21 - Written - Bovy (NYU)
        """
        return self._orb.vz(*args,**kwargs)

    def phi(self,*args,**kwargs):
        """
        NAME:
           phi
        PURPOSE:
           return azimuth
        INPUT:
           t - (optional) time at which to get the azimuth
        OUTPUT:
           phi(t)
        HISTORY:
           2010-09-21 - Written - Bovy (NYU)
        """
        return self._orb.phi(*args,**kwargs)

    def vphi(self,*args,**kwargs):
        """
        NAME:
           vphi
        PURPOSE:
           return angular velocity
        INPUT:
           t - (optional) time at which to get the angular velocity
        OUTPUT:
           vphi(t)
        HISTORY:
           2010-09-21 - Written - Bovy (NYU)
        """
        return self._orb.vphi(*args,**kwargs)

    def x(self,*args,**kwargs):
        """
        NAME:
           x
        PURPOSE:
           return x
        INPUT:
           t - (optional) time at which to get x
        OUTPUT:
           x(t)
        HISTORY:
           2010-09-21 - Written - Bovy (NYU)
        """
        return self._orb.x(*args,**kwargs)

    def y(self,*args,**kwargs):
        """
        NAME:
           y
        PURPOSE:
           return y
        INPUT:
           t - (optional) time at which to get y
        OUTPUT:
           y(t)
        HISTORY:
           2010-09-21 - Written - Bovy (NYU)
        """
        return self._orb.y(*args,**kwargs)

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

    def plotE(self,*args,**kwargs):
        """
        NAME:
           plotEt
        PURPOSE:
           plot E(.) along the orbit
        INPUT:
           pot= - Potential instance or list of instances in which the orbit 
                 was integrated
           d1= - plot Ez vs d1: e.g., 't', 'z', 'R', 'vR', 'vT', 'vz'      
           +bovy_plot.bovy_plot inputs
        OUTPUT:
           figure to output device
        HISTORY:
           2010-07-10 - Written - Bovy (NYU)
        """
        self._orb.plotE(*args,**kwargs)

    def plotEz(self,*args,**kwargs):
        """
        NAME:
           plotEz
        PURPOSE:
           plot E_z(.) along the orbit
        INPUT:
           po= - Potential instance or list of instances in which the orbit was
                 integrated
           d1= - plot Ez vs d1: e.g., 't', 'z', 'R'
           +bovy_plot.bovy_plot inputs
        OUTPUT:
           figure to output device
        HISTORY:
           2010-07-10 - Written - Bovy (NYU)
        """
        self._orb.plotEz(*args,**kwargs)

    def plotEzJz(self,*args,**kwargs):
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
        self._orb.plotEzJz(*args,**kwargs)

    def plotR(self,*args,**kwargs):
        """
        NAME:
           plotR
        PURPOSE:
           plot R(.) along the orbit
        INPUT:
           bovy_plot.bovy_plot inputs
        OUTPUT:
           figure to output device
        HISTORY:
           2010-07-10 - Written - Bovy (NYU)
        """
        self._orb.plotR(*args,**kwargs)

    def plotz(self,*args,**kwargs):
        """
        NAME:
           plotz
        PURPOSE:
           plot z(.) along the orbit
        INPUT:
           bovy_plot.bovy_plot inputs
        OUTPUT:
           figure to output device
        HISTORY:
           2010-07-10 - Written - Bovy (NYU)
        """
        self._orb.plotz(*args,**kwargs)

    def plotvR(self,*args,**kwargs):
        """
        NAME:
           plotvR
        PURPOSE:
           plot vR(.) along the orbit
        INPUT:
           bovy_plot.bovy_plot inputs
        OUTPUT:
           figure to output device
        HISTORY:
           2010-07-10 - Written - Bovy (NYU)
        """
        self._orb.plotvR(*args,**kwargs)

    def plotvT(self,*args,**kwargs):
        """
        NAME:
           plotvT
        PURPOSE:
           plot vT(.) along the orbit
        INPUT:
           bovy_plot.bovy_plot inputs
        OUTPUT:
           figure to output device
        HISTORY:
           2010-07-10 - Written - Bovy (NYU)
        """
        self._orb.plotvT(*args,**kwargs)

    def plotphi(self,*args,**kwargs):
        """
        NAME:
           plotphi
        PURPOSE:
           plot \phi(.) along the orbit
        INPUT:
           bovy_plot.bovy_plot inputs
        OUTPUT:
           figure to output device
        HISTORY:
           2010-07-10 - Written - Bovy (NYU)
        """
        self._orb.plotphi(*args,**kwargs)

    def plotvz(self,*args,**kwargs):
        """
        NAME:
           plotvz
        PURPOSE:
           plot vz(.) along the orbit
        INPUT:
           bovy_plot.bovy_plot inputs
        OUTPUT:
           figure to output device
        HISTORY:
           2010-07-10 - Written - Bovy (NYU)
        """
        self._orb.plotvz(*args,**kwargs)

    def plotx(self,*args,**kwargs):
        """
        NAME:
           plotx
        PURPOSE:
           plot a one-dimensional orbit position
        INPUT:
           bovy_plot.bovy_plot inputs
        OUTPUT:
           figure to output device
        HISTORY:
           2010-07-21 - Written - Bovy (NYU)
        """
        self._orb.plotx(*args,**kwargs)

    def plotvx(self,*args,**kwargs):
        """
        NAME:
           plotvx
        PURPOSE:
           plot a one-dimensional orbit velocity
        INPUT:
           bovy_plot.bovy_plot inputs
        OUTPUT:
           figure to output device
        HISTORY:
           2010-07-21 - Written - Bovy (NYU)
        """
        self._orb.plotvx(*args,**kwargs)

    def ploty(self,*args,**kwargs):
        """
        NAME:
           ploty
        PURPOSE:
           plot a one-dimensional orbit position
        INPUT:
           bovy_plot.bovy_plot inputs
        OUTPUT:
           figure to output device
        HISTORY:
           2010-07-21 - Written - Bovy (NYU)
        """
        self._orb.ploty(*args,**kwargs)

    def plotvy(self,*args,**kwargs):
        """
        NAME:
           plotvy
        PURPOSE:
           plot a one-dimensional orbit velocity
        INPUT:
           bovy_plot.bovy_plot inputs
        OUTPUT:
           figure to output device
        HISTORY:
           2010-07-21 - Written - Bovy (NYU)
        """
        self._orb.plotvy(*args,**kwargs)

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
