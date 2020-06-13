import numpy
from galpy.orbit import Orbit
from galpy.df.df import df, _APY_LOADED
from galpy.potential import flatten as flatten_potential
from galpy.potential import rtide, evaluateRforces
from galpy.util import bovy_coords, bovy_conversion, \
    _rotate_to_arbitrary_vector
if _APY_LOADED:
    from astropy import units
class streamspraydf(df):
    def __init__(self,progenitor_mass,progenitor=None,pot=None,
                 tdisrupt=None,leading=True,
                 meankvec=[2.,0.,0.3,0.,0.,0.],
                 sigkvec=[0.4,0.,0.4,0.5,0.5,0.],
                 ro=None,vo=None):
        """
        NAME:
        
           __init__
        PURPOSE:

           Initialize a stream spray DF model of a tidal stream

        INPUT:

           progenitor_mass - mass of the progenitor (can be Quantity)

           tdisrupt= (5 Gyr) time since start of disruption (can be Quantity)

           leading= (True) if True, model the leading part of the stream
                           if False, model the trailing part

           progenitor= progenitor orbit as Orbit instance (will be re-integrated, so don't bother integrating the orbit before)
           
           meankvec= (Fardal+2015-ish defaults) 
           
           sigkvec= (Fardal+2015-ish defaults) 
           
        OUTPUT:
        
            Instance
            
        HISTORY:
        
           2018-07-31 - Written - Bovy (UofT)

        """
        df.__init__(self,ro=ro,vo=vo)
        if _APY_LOADED and isinstance(progenitor_mass,units.Quantity):
            progenitor_mass= progenitor_mass.to(units.Msun).value\
                            /bovy_conversion.mass_in_msol(self._vo,self._ro)
        self._progenitor_mass= progenitor_mass
        if tdisrupt is None:
            self._tdisrupt= 5./bovy_conversion.time_in_Gyr(self._vo,self._ro)
        else:
            if _APY_LOADED and isinstance(tdisrupt,units.Quantity):
                tdisrupt= tdisrupt.to(units.Gyr).value\
                    /bovy_conversion.time_in_Gyr(self._vo,self._ro)
            self._tdisrupt= tdisrupt
        if pot is None: #pragma: no cover
            raise IOError("pot= must be set")
        self._pot= flatten_potential(pot)
        self._progenitor= progenitor()
        self._progenitor_times= numpy.linspace(0.,-self._tdisrupt,10001)
        self._progenitor.integrate(self._progenitor_times,self._pot)
        self._meankvec= numpy.array(meankvec)
        self._sigkvec= numpy.array(sigkvec)
        if leading: self._meankvec*= -1.
        return None
    
    def sample(self,n,returndt=False,integrate=True,xy=False,lb=False):
        """
        NAME:

            sample

        PURPOSE:

            sample from the DF

        INPUT:

            n - number of points to return

            returndt= (False) if True, also return the time since the star was stripped
            
            integrate= (True) if True, integrate the orbits to the present time, if False, return positions at stripping (probably want to combine with returndt=True then to make sense of them!)

            xy= (False) if True, return Galactocentric rectangular coordinates

            lb= (False) if True, return Galactic l,b,d,vlos,pmll,pmbb coordinates

        OUTPUT:

            (R,vR,vT,z,vz,phi) of points on the stream in 6,N array

        HISTORY:

            2018-07-31 - Written - Bovy (IAS)

        """
        if xy or lb:
            raise NotImplementedError("xy=True and lb=True options currently not implemented")
        # First sample times
        dt= numpy.random.uniform(size=n)*self._tdisrupt
        # Build all rotation matrices
        rot, rot_inv= self._setup_rot(dt)
        # Compute progenitor position in the instantaneous frame
        xyzpt= numpy.einsum('ijk,ik->ij',rot,
                            numpy.array([self._progenitor.x(-dt),
                                         self._progenitor.y(-dt),
                                         self._progenitor.z(-dt)]).T)
        vxyzpt= numpy.einsum('ijk,ik->ij',rot,
                             numpy.array([self._progenitor.vx(-dt),
                                          self._progenitor.vy(-dt),
                                          self._progenitor.vz(-dt)]).T)
        Rpt,phipt,Zpt= bovy_coords.rect_to_cyl(xyzpt[:,0],xyzpt[:,1],xyzpt[:,2])
        vRpt,vTpt,vZpt= bovy_coords.rect_to_cyl_vec(vxyzpt[:,0],vxyzpt[:,1],vxyzpt[:,2],
                                                    Rpt,phipt,Zpt,cyl=True)
        # Sample positions and velocities in the instantaneous frame
        k= self._meankvec+numpy.random.normal(size=n)[:,numpy.newaxis]*self._sigkvec
        try:
            rtides= rtide(self._pot,Rpt,Zpt,phi=phipt,
                          t=-dt,M=self._progenitor_mass,use_physical=False)
            vcs= numpy.sqrt(-Rpt
                            *evaluateRforces(self._pot,Rpt,Zpt,phi=phipt,t=-dt,
                                             use_physical=False))
        except (ValueError,TypeError):
            rtides= numpy.array([rtide(self._pot,Rpt[ii],Zpt[ii],phi=phipt[ii],
                                  t=-dt[ii],M=self._progenitor_mass,use_physical=False)
                                for ii in range(len(Rpt))])
            vcs= numpy.array([numpy.sqrt(-Rpt[ii]
                                *evaluateRforces(self._pot,Rpt[ii],Zpt[ii],phi=phipt[ii],t=-dt[ii],
                                                 use_physical=False))
                              for ii in range(len(Rpt))])
        rtides_as_frac= rtides/Rpt
        RpZst= numpy.array([Rpt+k[:,0]*rtides,
                            phipt+k[:,5]*rtides_as_frac,
                            k[:,3]*rtides_as_frac]).T
        vRTZst= numpy.array([vRpt*(1.+k[:,1]),
                             vTpt+k[:,2]*vcs*rtides_as_frac,
                             k[:,4]*vcs*rtides_as_frac]).T
        # Now rotate these back to the galactocentric frame
        xst,yst,zst= bovy_coords.cyl_to_rect(RpZst[:,0],RpZst[:,1],RpZst[:,2])
        vxst,vyst,vzst= bovy_coords.cyl_to_rect_vec(vRTZst[:,0],vRTZst[:,1],vRTZst[:,2],RpZst[:,1])
        xyzs= numpy.einsum('ijk,ik->ij',rot_inv,numpy.array([xst,yst,zst]).T)
        vxyzs= numpy.einsum('ijk,ik->ij',rot_inv,numpy.array([vxst,vyst,vzst]).T)
        Rs,phis,Zs= bovy_coords.rect_to_cyl(xyzs[:,0],xyzs[:,1],xyzs[:,2])
        vRs,vTs,vZs= bovy_coords.rect_to_cyl_vec(vxyzs[:,0],vxyzs[:,1],vxyzs[:,2],
                                                 Rs,phis,Zs,cyl=True)
        out= numpy.empty((6,n))
        if integrate:
            # Now integrate the orbits
            for ii in range(n):
                o= Orbit([Rs[ii],vRs[ii],vTs[ii],Zs[ii],vZs[ii],phis[ii]])
                o.integrate(numpy.linspace(-dt[ii],0.,10001),self._pot)
                o= o(0.)
                out[:,ii]= [o.R(),o.vR(),o.vT(),o.z(),o.vz(),o.phi()]
        else:
            out[0]= Rs
            out[1]= vRs
            out[2]= vTs
            out[3]= Zs
            out[4]= vZs
            out[5]= phis
        if returndt:
            return (out,dt)
        else:
            return out

    def _setup_rot(self,dt):
        n= len(dt)
        L= self._progenitor.L(-dt)
        Lnorm= L/numpy.tile(numpy.sqrt(numpy.sum(L**2.,axis=1)),(3,1)).T
        z_rot= numpy.swapaxes(_rotate_to_arbitrary_vector(numpy.atleast_2d(Lnorm),[0.,0.,1],inv=True),1,2)
        z_rot_inv= numpy.swapaxes(_rotate_to_arbitrary_vector(numpy.atleast_2d(Lnorm),[0.,0.,1],inv=False),1,2)
        xyzt= numpy.einsum('ijk,ik->ij',z_rot,
                           numpy.array([self._progenitor.x(-dt),
                                        self._progenitor.y(-dt),
                                        self._progenitor.z(-dt)]).T)
        Rt= numpy.sqrt(xyzt[:,0]**2.+xyzt[:,1]**2.)
        cosphi, sinphi= xyzt[:,0]/Rt,xyzt[:,1]/Rt
        pa_rot= numpy.array([[cosphi,-sinphi,numpy.zeros(n)],
                             [sinphi,cosphi,numpy.zeros(n)],
                             [numpy.zeros(n),numpy.zeros(n),numpy.ones(n)]]).T
        pa_rot_inv= numpy.array([[cosphi,sinphi,numpy.zeros(n)],
                                 [-sinphi,cosphi,numpy.zeros(n)],
                                 [numpy.zeros(n),numpy.zeros(n),numpy.ones(n)]]).T
        rot= numpy.einsum('ijk,ikl->ijl',pa_rot,z_rot)
        rot_inv= numpy.einsum('ijk,ikl->ijl',z_rot_inv,pa_rot_inv)
        return (rot,rot_inv)
