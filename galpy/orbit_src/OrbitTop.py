import math as m
import numpy as nu
from scipy import integrate, interpolate
import galpy.util.bovy_plot as plot
class OrbitTop:
    """General class that holds orbits and integrates them"""
    def __init__(self,vxvv=None):
        """
        NAME:
           __init__
        PURPOSE:
           Initialize an orbit instance
        INPUT:
           vxvv - initial condition
        OUTPUT:
           (none)
        HISTORY:
           2010-07-10 - Written - Bovy (NYU)
        """
        return None

    def integrate(self,t,pot):
        """
        NAME:
           integrate
        PURPOSE:
           integrate the orbit
        INPUT:
           t - list of times at which to output (0 has to be in this!)
           pot - Potential instance or list of instances
        OUTPUT:
           (none) (get the actual orbit using self.getOrbit()
        HISTORY:
           2010-07-10
        """
        raise AttributeError

    def getOrbit(self):
        """
        NAME:
           getOrbit
        PURPOSE:
           return a previously calculated orbit
        INPUT:
           (none)
        OUTPUT:
           (none)
        HISTORY:
           2010-07-10 - Written - Bovy (NYU)
        """
        return self.orbit

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
        thiso= self(*args,**kwargs)
        return thiso[0]

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
        thiso= self(*args,**kwargs)
        return thiso[1]

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
        thiso= self(*args,**kwargs)
        return thiso[2]

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
        thiso= self(*args,**kwargs)
        if len(thiso) < 5:
            raise AttributeError("linear and planar orbits do not have z()")
        return thiso[3]

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
        thiso= self(*args,**kwargs)
        if len(thiso) < 5:
            raise AttributeError("linear and planar orbits do not have vz()")
        return thiso[4]

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
        thiso= self(*args,**kwargs)
        if len(thiso) != 4 and len(thiso) != 6:
            raise AttributeError("orbit must track azimuth to use phi()")
        elif len(thiso) == 4:
            return thiso[3]
        else:
            return thiso[5]

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
        thiso= self(*args,**kwargs)
        if len(thiso) == 2:
            return thiso[0]
        if len(thiso) != 4 and len(thiso) != 6:
            raise AttributeError("orbit must track azimuth to use x()")
        elif len(thiso) == 4:
            return thiso[0]*m.cos(thiso[3])
        else:
            return thiso[0]*m.cos(thiso[5])

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
        thiso= self(*args,**kwargs)
        if len(thiso) != 4 and len(thiso) != 6:
            raise AttributeError("orbit must track azimuth to use y()")
        elif len(thiso) == 4:
            return thiso[0]*m.sin(thiso[3])
        else:
            return thiso[0]*m.sin(thiso[5])

    def vx(self,*args,**kwargs):
        """
        NAME:
           vx
        PURPOSE:
           return x velocity at time t
        INPUT:
           t - (optional) time at which to get the velocity
        OUTPUT:
           vx(t)
        HISTORY:
           2010-11-30 - Written - Bovy (NYU)
        """
        thiso= self(*args,**kwargs)
        if len(thiso) == 2:
            return thiso[1]
        if len(thiso) != 4 and len(thiso) != 6:
            raise AttributeError("orbit must track azimuth to use vx()")
        elif len(thiso) == 4:
            theta= thiso[3]
        else:
            theta= thiso[5]
        return thiso[1]*m.cos(theta)-thiso[2]*m.sin(theta)

    def vy(self,*args,**kwargs):
        """
        NAME:
           vy
        PURPOSE:
           return y velocity at time t
        INPUT:
           t - (optional) time at which to get the velocity
        OUTPUT:
           vy(t)
        HISTORY:
           2010-11-30 - Written - Bovy (NYU)
        """
        thiso= self(*args,**kwargs)
        if len(thiso) == 2:
            return thiso[1]
        if len(thiso) != 4 and len(thiso) != 6:
            raise AttributeError("orbit must track azimuth to use vx()")
        elif len(thiso) == 4:
            theta= thiso[3]
        else:
            theta= thiso[5]
        return thiso[2]*m.cos(theta)+thiso[1]*m.sin(theta)

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
        thiso= self(*args,**kwargs)
        return thiso[2]/thiso[0]

    def L(self,*args,**kwargs):
        """
        NAME:
           L
        PURPOSE:
           calculate the angular momentum
        INPUT:
           (none)
        OUTPUT:
           angular momentum
        HISTORY:
           2010-09-15 - Written - Bovy (NYU)
        """
        thiso= self(*args,**kwargs)
        if len(thiso) < 3:
            raise AttributeError("'linearOrbit has no angular momentum")
        elif len(thiso) == 3 or len(thiso) == 4:
            return thiso[0]*thiso[2]
        elif len(thiso) == 5:
            raise AttributeError("You must track the azimuth to get the angular momentum of a 3D Orbit")
        else: #len(thiso)==6
            vx= self.vx()
            vy= self.vy()
            vz= self.vz()
            x= self.x()
            y= self.y()
            z= self.z()
            return nu.array([y*vz-z*vy,z*vx-x*vz,x*vy-y*vx])

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
           [R,vR,vT,z,vz(,phi)] or [R,vR,vT(,phi)] depending on the orbit
        HISTORY:
           2010-07-10 - Written - Bovy (NYU)
        """
        #Options
        if kwargs.has_key('rect'):
            rect= kwargs['rect']
        else:
            rect= False
        if rect:
            return self._callRect(*args)
        if len(args) == 0:
            return self.vxvv
        else:
            t= args
        if t in list(self.t):
            return self.orbit[list(self.t).index(t),:]
        else:
            dim= len(self.vxvv)
            if not hasattr(self,"_orbInterp"):
                orbInterp= []
                for ii in range(dim):
                    orbInterp.append(interpolate.InterpolatedUnivariateSpline(\
                            self.t,self.orbit[:,ii]))
                self._orbInterp= orbInterp
            out= []
            for ii in range(dim):
                out.append(self._orbInterp[ii](t))
            return nu.array(out).reshape(dim)

    def plotE(self,pot,*args,**kwargs):
        """
        NAME:
           plotE
        PURPOSE:
           plot E(.) along the orbit
        INPUT:
           pot - Potential instance or list of instances in which the orbit was
                 integrated
           d1= - plot E vs d1: e.g., 't', 'z', 'R', 'vR', 'vT', 'vz'      
           +bovy_plot.bovy_plot inputs
        OUTPUT:
           figure to output device
        HISTORY:
           2010-07-10 - Written - Bovy (NYU)
        """
        raise AttributeError

    def plot(self,*args,**kwargs):
        """
        NAME:
           plot
        PURPOSE:
           plot aspects of an Orbit
        INPUT:
           bovy_plot args and kwargs
        OUTPUT:
           plot
        HISTORY:
           2010-07-26 - Written - Bovy (NYU)
           2010-09-22 - Adapted to more general framework - Bovy (NYU)
        """
        labeldict= {'t':r'$t$','R':r'$R$','vR':r'$v_R$','vT':r'$v_T$',
                    'z':r'$z$','vz':r'$v_z$','phi':r'$\phi$',
                    'x':r'$x$','y':r'$y$','vx':r'$v_x$','vy':r'$v_y$'}
        #Defaults
        if not kwargs.has_key('d1') and not kwargs.has_key('d2'):
            if len(self.vxvv) == 3:
                d1= 'R'
                d2= 'vR'
            elif len(self.vxvv) == 4:
                d1= 'x'
                d2= 'y'
            elif len(self.vxvv) == 2:
                d1= 'x'
                d2= 'vx'
            elif len(self.vxvv) == 5 or len(self.vxvv) == 6:
                d1= 'R'
                d2= 'z'
        elif not kwargs.has_key('d1'):
            d2= kwargs['d2']
            kwargs.pop('d2')
            d1= 't'
        elif not kwargs.has_key('d2'):
            d1= kwargs['d1']
            kwargs.pop('d1')
            d2= 't'
        else:
            d1= kwargs['d1']
            kwargs.pop('d1')
            d2= kwargs['d2']
            kwargs.pop('d2')
        #Get x and y
        if d1 == 't':
            x= nu.array(self.t)
        elif d1 == 'R':
            x= self.orbit[:,0]
        elif d1 == 'z':
            x= self.orbit[:,3]
        elif d1 == 'vz':
            x= self.orbit[:,4]
        elif d1 == 'vR':
            x= self.orbit[:,1]
        elif d1 == 'vT':
            x= self.orbit[:,2]
        elif d1 == 'x':
            if len(self.vxvv) == 2:
                x= self.orbit[:,0]
            elif len(self.vxvv) != 4 and len(self.vxvv) != 6:
                raise AttributeError("If you want x you need to track phi")
            elif len(self.vxvv) == 4:
                x= self.orbit[:,0]*nu.cos(self.orbit[:,3])
            else:
                x= self.orbit[:,0]*nu.cos(self.orbit[:,5])                
        elif d1 == 'y':
            if len(self.vxvv) != 4 and len(self.vxvv) != 6:
                raise AttributeError("If you want y you need to track phi")
            elif len(self.vxvv) == 4:
                x= self.orbit[:,0]*nu.sin(self.orbit[:,3])
            else:
                x= self.orbit[:,0]*nu.sin(self.orbit[:,5])                
        elif d1 == 'vx':
            x= self.vx()
        elif d1 == 'vy':
            x= self.vy()
        if d2 == 't':
            y= nu.array(self.t)
        elif d2 == 'R':
            y= self.orbit[:,0]
        elif d2 == 'z':
            y= self.orbit[:,3]
        elif d2 == 'vz':
            y= self.orbit[:,4]
        elif d2 == 'vR':
            y= self.orbit[:,1]
        elif d2 == 'vT':
            y= self.orbit[:,2]
        elif d2 == 'x':
            if len(self.vxvv) == 2:
                y= self.orbit[:,0]
            elif len(self.vxvv) != 4 and len(self.vxvv) != 6:
                raise AttributeError("If you want x you need to track phi")
            elif len(self.vxvv) == 4:
                y= self.orbit[:,0]*nu.cos(self.orbit[:,3])
            else:
                y= self.orbit[:,0]*nu.cos(self.orbit[:,5])                
        elif d2 == 'y':
            if len(self.vxvv) != 4 and len(self.vxvv) != 6:
                raise AttributeError("If you want y you need to track phi")
            elif len(self.vxvv) == 4:
                y= self.orbit[:,0]*nu.sin(self.orbit[:,3])
            else:
                y= self.orbit[:,0]*nu.sin(self.orbit[:,5])                
        elif d2 == 'vx':
            y= self.vx()
        elif d2 == 'vy':
            y= self.vy()

        #Plot
        if not kwargs.has_key('xlabel'):
            kwargs['xlabel']= labeldict[d1]
        if not kwargs.has_key('ylabel'):
            kwargs['ylabel']= labeldict[d2]
        plot.bovy_plot(x,y,*args,**kwargs)

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
        kwargs['d2']= 'R'
        self.plot(*args,**kwargs)

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
        kwargs['d2']= 'z'
        self.plot(*args,**kwargs)

    def plotx(self,*args,**kwargs):
        """
        NAME:
           plotx
        PURPOSE:
           plot x(.) along the orbit
        INPUT:
           bovy_plot.bovy_plot inputs
        OUTPUT:
           figure to output device
        HISTORY:
           2010-07-10 - Written - Bovy (NYU)
        """
        kwargs['d2']= 'x'
        self.plot(*args,**kwargs)

    def plotvx(self,*args,**kwargs):
        """
        NAME:
           plotvx
        PURPOSE:
           plot vx(.) along the orbit
        INPUT:
           bovy_plot.bovy_plot inputs
        OUTPUT:
           figure to output device
        HISTORY:
           2010-07-10 - Written - Bovy (NYU)
        """
        kwargs['d2']= 'vx'
        self.plot(*args,**kwargs)

    def ploty(self,*args,**kwargs):
        """
        NAME:
           ploty
        PURPOSE:
           plot y(.) along the orbit
        INPUT:
           bovy_plot.bovy_plot inputs
        OUTPUT:
           figure to output device
        HISTORY:
           2010-07-10 - Written - Bovy (NYU)
        """
        kwargs['d2']= 'y'
        self.plot(*args,**kwargs)

    def plotvy(self,*args,**kwargs):
        """
        NAME:
           plotvy
        PURPOSE:
           plot vy(.) along the orbit
        INPUT:
           bovy_plot.bovy_plot inputs
        OUTPUT:
           figure to output device
        HISTORY:
           2010-07-10 - Written - Bovy (NYU)
        """
        kwargs['d2']= 'vy'
        self.plot(*args,**kwargs)

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
        kwargs['d2']= 'vR'
        self.plot(*args,**kwargs)

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
        kwargs['d2']= 'vT'
        self.plot(*args,**kwargs)
        
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
        kwargs['d2']= 'phi'
        self.plot(*args,**kwargs)

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
        kwargs['d2']= 'phi'
        self.plot(*args,**kwargs)
        
