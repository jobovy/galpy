# galpy.potential.amuse: AMUSE representation of galpy potentials
import numpy
from amuse.units import units
from amuse.units.quantities import ScalarQuantity
from amuse.support.literature import LiteratureReferencesMixIn
from .. import potential
from ..util import bovy_conversion
class galpy_profile(LiteratureReferencesMixIn):
    """
    User-defined potential from galpy
    
    .. [#] Bovy, J, 2015, galpy: A Python Library for Galactic Dynamics, Astrophys. J. Supp. 216, 29 [2015ApJS..216...29B]
    
    """
    def __init__(self,pot,t = 0.,tgalpy = 0.,ro=8,vo=220.,reverse=False):
        """
        NAME:

           __init__

        PURPOSE:

           initialize a galpy potential for use with AMUSE

        INPUT:

           pot - galpy potential object or list of such objects

           t - start time for AMUSE simulation (can be an AMUSE Quantity)

           tgalpy - start time for galpy potential, can be less than zero (can be Quantity)

           ro=, vo= distance and velocity scales for translation into internal units (default from configuration file)

           reverse - set whether galpy potential evolves forwards or backwards in time (default: False)

        OUTPUT:

           (none)

        HISTORY:

           2019-08-12 - Written - Webb (UofT)

        """
        LiteratureReferencesMixIn.__init__(self)
        self.pot= pot
        self.ro= ro
        self.vo= vo
        self.reverse= reverse
        #Initialize model time
        if isinstance(t,ScalarQuantity):
            self.model_time= t
        else:
            self.model_time= \
                t*bovy_conversion.time_in_Gyr(ro=self.ro,vo=self.vo) \
                | units.Gyr
        #Initialize galpy time
        if isinstance(tgalpy,ScalarQuantity):
            self.tgalpy= tgalpy.value_in(units.Gyr)\
                /bovy_conversion.time_in_Gyr(ro=self.ro,vo=self.vo)
        else:
            self.tgalpy= tgalpy

    def evolve_model(self,time):
        """
        NAME:
           evolve_model
        PURPOSE:
           evolve time parameters to t_end
        INPUT:
           time - time to evolve potential to
        OUTPUT:
           None
        HISTORY:
           2019-08-12 - Written - Webb (UofT)
        """
        dt= time-self.model_time
        self.model_time= time  
        if self.reverse:
            self.tgalpy-= dt.value_in(units.Gyr)\
                /bovy_conversion.time_in_Gyr(ro=self.ro,vo=self.vo)
        else:
            self.tgalpy+= dt.value_in(units.Gyr)\
                /bovy_conversion.time_in_Gyr(ro=self.ro,vo=self.vo)

    def get_potential_at_point(self,eps,x,y,z):
        """
        NAME:
           get_potential_at_point
        PURPOSE:
           Get potenial at a given location in the potential
        INPUT:
           eps - softening length (necessary for AMUSE, but not used by galpy potential)
           x,y,z - position in the potential
        OUTPUT:
           Phi(x,y,z)
        HISTORY:
           2019-08-12 - Written - Webb (UofT)
        """
        R= numpy.sqrt(x.value_in(units.kpc)**2.+y.value_in(units.kpc)**2.)
        zed= z.value_in(units.kpc)
        phi= numpy.arctan2(y.value_in(units.kpc),x.value_in(units.kpc))
        return potential.evaluatePotentials(self.pot,R/self.ro,zed/self.ro,
                                            phi=phi,t=self.tgalpy,
                                            ro=self.ro,vo=self.vo) \
                                            | units.km**2*units.s**-2

    def get_gravity_at_point(self,eps,x,y,z):
        """
        NAME:
           get_gravity_at_point
        PURPOSE:
           Get acceleration due to potential at a given location in the potential
        INPUT:
           eps - softening length (necessary for AMUSE, but not used by galpy potential)
           x,y,z - position in the potential
        OUTPUT:
           ax,ay,az
        HISTORY:
           2019-08-12 - Written - Webb (UofT)
        """
        R= numpy.sqrt(x.value_in(units.kpc)**2.+y.value_in(units.kpc)**2.)
        zed= z.value_in(units.kpc)
        phi= numpy.arctan2(y.value_in(units.kpc),x.value_in(units.kpc))
        # Cylindrical force
        Rforce= potential.evaluateRforces(self.pot,R/self.ro,zed/self.ro,
                                          phi=phi,t=self.tgalpy)
        phiforce= potential.evaluatephiforces(self.pot,R/self.ro,zed/self.ro,
                                              phi=phi,t=self.tgalpy)\
                                              /(R/self.ro)
        zforce=potential.evaluatezforces(self.pot,R/self.ro,zed/self.ro,
                                         phi=phi,t=self.tgalpy)
        # Convert cylindrical force --> rectangular
        cp, sp= numpy.cos(phi), numpy.sin(phi)
        ax= (Rforce*cp - phiforce*sp)\
            *bovy_conversion.force_in_kmsMyr(ro=self.ro,vo=self.vo) \
            | units.kms * units.myr**-1
        ay= (Rforce*sp + phiforce*cp)\
            *bovy_conversion.force_in_kmsMyr(ro=self.ro,vo=self.vo) \
            | units.kms * units.myr**-1
        az= zforce\
            *bovy_conversion.force_in_kmsMyr(ro=self.ro,vo=self.vo) \
            | units.kms * units.myr**-1
        return ax,ay,az

    def mass_density(self,x,y,z):
        """
        NAME:
           mass_density
        PURPOSE:
           Get mass density at a given location in the potential
        INPUT:
           eps - softening length (necessary for AMUSE, but not used by galpy potential)
           x,y,z - position in the potential
        OUTPUT:
           the density
        HISTORY:
           2019-08-12 - Written - Webb (UofT)
        """
        R= numpy.sqrt(x.value_in(units.kpc)**2.+y.value_in(units.kpc)**2.)
        zed= z.value_in(units.kpc)
        phi= numpy.arctan2(y.value_in(units.kpc),x.value_in(units.kpc))
        return potential.evaluateDensities(self.pot,R/self.ro,zed/self.ro,
                                           phi=phi,t=self.tgalpy,
                                           ro=self.ro,vo=self.vo) \
                                           | units.MSun/(units.parsec**3.)

    def circular_velocity(self,r):
        """
        NAME:
           circular_velocity
        PURPOSE:
           Get circular velocity at a given radius in the potential
        INPUT:
           r - radius in the potential
        OUTPUT:
           the circular velocity
        HISTORY:
           2019-08-12 - Written - Webb (UofT)
        """
        return potential.vcirc(self.pot,r.value_in(units.kpc)/self.ro,phi=0,
                               t=self.tgalpy,ro=self.ro,vo=self.vo) | units.kms
        
    def enclosed_mass(self,r):
        """
        NAME:
           enclosed_mass
        PURPOSE:
           Get mass enclosed within a given radius in the potential
        INPUT:
           r - radius in the potential
        OUTPUT:
           the mass enclosed
        HISTORY:
           2019-08-12 - Written - Webb (UofT)
        """
        vc2= potential.vcirc(self.pot,r.value_in(units.kpc)/self.ro,phi=0,
                             t=self.tgalpy,ro=self.ro,vo=self.vo)**2.
        return vc2*r.value_in(units.kpc)/bovy_conversion._G*1000. | units.MSun

    def stop(self):
        """
        NAME:
           stop
        PURPOSE:
           Stop the potential model (necessary function for AMUSE)
        INPUT:
           None
        OUTPUT:
           None
        HISTORY:
           2019-08-12 - Written - Webb (UofT)
        """
        pass
