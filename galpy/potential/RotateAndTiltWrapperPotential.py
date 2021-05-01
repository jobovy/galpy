###############################################################################
#   RotateAndTiltWrapperPotential.py: Wrapper to rotate and tilt the z-axis
#   of a potential
###############################################################################
import numpy
from .Potential import check_potential_inputs_not_arrays, \
    _evaluatePotentials, _evaluateRforces, _evaluatezforces, \
    _evaluatephiforces, evaluateDensities, evaluateR2derivs, \
    evaluatez2derivs, evaluatephi2derivs, evaluateRzderivs, \
    evaluateRphiderivs, evaluatephizderivs
from .WrapperPotential import WrapperPotential
from ..util import conversion
from ..util import _rotate_to_arbitrary_vector
from ..util import coords
# Only implement 3D wrapper
class RotateAndTiltWrapperPotential(WrapperPotential):
    """Potential wrapper that allows a potential to be rotated in 3D
according to three orientation angles. These angles can either be
specified using:

* A rotation around the original z-axis (`galaxy_pa`) and the new direction of the z-axis (`zvec`) or

* A rotation around the original z-axis (`galaxy_pa`), the `inclination`, and a rotation around the new z axis (`sky_pa`).

The second option allows one to specify the inclination and sky position angle (measured from North) in the usual manner in extragalactic observations."""
    def __init__(self,amp=1.,inclination=None,galaxy_pa=None,sky_pa=None,
                 zvec=None,pot=None,
                 ro=None,vo=None):
        """
        NAME:

           __init__

        PURPOSE:

           initialize a RotateAndTiltWrapper Potential

        INPUT:

           amp= (1.) overall amplitude to apply to the potential

           pot= Potential instance or list thereof for the potential to rotate and tilt

           Orientation angles as
        
              galaxy_pa= rotation angle of the original potential around the original z axis (can be a Quantity)

           and either


              1) zvec= 3D vector specifying the direction of the rotated z axis

              2) inclination= usual inclination angle (with the line-of-sight being the z axis)

                 sky_pa= rotation angle around the inclined z axis (usual sky position angle measured from North)

        OUTPUT:

           (none)

        HISTORY:

           2021-03-29 - Started - Mackereth (UofT)

           2021-04-18 - Added inclination, sky_pa, galaxy_pa setup - Bovy (UofT)

        """
        WrapperPotential.__init__(self,amp=amp,pot=pot,ro=ro,vo=vo,
                                  _init=True)
        inclination= conversion.parse_angle(inclination)
        sky_pa= conversion.parse_angle(sky_pa)
        galaxy_pa= conversion.parse_angle(galaxy_pa)
        zvec, galaxy_pa= self._parse_inclination(inclination,sky_pa,
                                                 zvec,galaxy_pa)
        self._setup_zvec_pa(zvec,galaxy_pa)
        self.hasC= True
        self.hasC_dxdv= True
        self.isNonAxi= True

    def _parse_inclination(self,inclination,sky_pa,zvec,galaxy_pa):
        if inclination is None:
            return (zvec,galaxy_pa)
        if sky_pa is None:
            sky_pa= 0.
        zvec_rot= numpy.dot(\
            numpy.array([[numpy.sin(sky_pa),numpy.cos(sky_pa),0.],
                         [-numpy.cos(sky_pa),numpy.sin(sky_pa),0.],
                         [0.,0.,1]]),
            numpy.array([[1.,0.,0.],
                         [0.,-numpy.cos(inclination),-numpy.sin(inclination)],
                         [0.,-numpy.sin(inclination),numpy.cos(inclination)]]))
        zvec= numpy.dot(zvec_rot,numpy.array([0.,0.,1.]))
        int_rot= _rotate_to_arbitrary_vector(numpy.array([[0.,0.,1.]]),
                                             zvec,inv=False)[0]
        pa= numpy.dot(int_rot,numpy.dot(zvec_rot,[1.,0.,0.]))
        return (zvec,galaxy_pa+numpy.arctan2(pa[1],pa[0]))
        
    def _setup_zvec_pa(self,zvec,pa):
        if not pa is None:
            pa_rot= numpy.array([[numpy.cos(pa),numpy.sin(pa),0.],
                                 [-numpy.sin(pa),numpy.cos(pa),0.],
                                 [0.,0.,1.]])
        else:
            pa_rot = numpy.eye(3)
        if not zvec is None:
            if not isinstance(zvec,numpy.ndarray):
                zvec= numpy.array(zvec)
            zvec/= numpy.sqrt(numpy.sum(zvec**2.))
            zvec_rot= _rotate_to_arbitrary_vector(\
                numpy.array([[0.,0.,1.]]),zvec,inv=True)[0]
        else:
            zvec_rot = numpy.eye(3)
        self._rot = numpy.dot(pa_rot,zvec_rot)
        self._inv_rot = numpy.linalg.inv(self._rot)
        return None

    def __getattr__(self,attribute):
        return super(RotateAndTiltWrapperPotential,self)\
            .__getattr__(attribute)

    @check_potential_inputs_not_arrays
    def _evaluate(self,R,z,phi=0.,t=0.):
        """
        NAME:
           _evaluate
        PURPOSE:
           evaluate the potential at R,z
        INPUT:
           R - Galactocentric cylindrical radius
           z - vertical height
           phi - azimuth
           t - time
        OUTPUT:
           Phi(R,z)
        HISTORY:
           2021-04-18 - Written - Bovy (UofT)
        """
        x,y,z= coords.cyl_to_rect(R,phi,z)
        if numpy.isinf(R): y= 0.
        xyzp= numpy.dot(self._rot,numpy.array([x,y,z]))
        Rp,phip,zp = coords.rect_to_cyl(xyzp[0],xyzp[1],xyzp[2])
        return _evaluatePotentials(self._pot,Rp,zp,phi=phip,t=t)

    @check_potential_inputs_not_arrays
    def _Rforce(self,R,z,phi=0.,t=0.):
        """
        NAME:
           _Rforce
        PURPOSE:
           evaluate the radial force for this potential
        INPUT:
           R - Galactocentric cylindrical radius
           z - vertical height
           phi - azimuth
           t - time
        OUTPUT:
           the radial force
        HISTORY:
           2021-04-18 - Written - Bovy (UofT)
        """
        Fxyz= self._force_xyz(R,z,phi=phi,t=t)
        return numpy.cos(phi)*Fxyz[0]+numpy.sin(phi)*Fxyz[1]

    @check_potential_inputs_not_arrays
    def _phiforce(self,R,z,phi=0.,t=0.):
        """
        NAME:
           _phiforce
        PURPOSE:
           evaluate the azimuthal force (torque) for this potential
        INPUT:
           R - Galactocentric cylindrical radius
           z - vertical height
           phi - azimuth
           t - time
        OUTPUT:
           the azimuthal force (torque)
        HISTORY:
           2021-04-18 - Written - Bovy (UofT)
        """
        Fxyz= self._force_xyz(R,z,phi=phi,t=t)
        return R*(-numpy.sin(phi)*Fxyz[0] + numpy.cos(phi)*Fxyz[1])

    @check_potential_inputs_not_arrays
    def _zforce(self,R,z,phi=0.,t=0.):
        """
        NAME:
           _zforce
        PURPOSE:
           evaluate the vertical force for this potential
        INPUT:
           R - Galactocentric cylindrical radius
           z - vertical height
           phi - azimuth
           t - time
        OUTPUT:
           the vertical force
        HISTORY:
           2021-04-18 - Written - Bovy (UofT)
        """
        return self._force_xyz(R,z,phi=phi,t=t)[2]

    def _force_xyz(self,R,z,phi=0.,t=0.):
        """Get the rectangular forces in the transformed frame"""
        x,y,z= coords.cyl_to_rect(R,phi,z)
        xyzp= numpy.dot(self._rot,numpy.array([x,y,z]))
        Rp,phip,zp =coords.rect_to_cyl(xyzp[0],xyzp[1],xyzp[2])
        Rforcep= _evaluateRforces(self._pot,Rp,zp,phi=phip,t=t)
        phiforcep= _evaluatephiforces(self._pot,Rp,zp,phi=phip,t=t)
        zforcep= _evaluatezforces(self._pot,Rp,zp,phi=phip,t=t)
        xforcep= numpy.cos(phip)*Rforcep-numpy.sin(phip)*phiforcep/Rp
        yforcep= numpy.sin(phip)*Rforcep+numpy.cos(phip)*phiforcep/Rp
        return numpy.dot(self._inv_rot,
                         numpy.array([xforcep,yforcep,zforcep]))   

    @check_potential_inputs_not_arrays
    def _R2deriv(self,R,z,phi=0.,t=0.):
        """
        NAME:
           _R2deriv
        PURPOSE:
           evaluate the second radial derivative for this potential
        INPUT:
           R - Galactocentric cylindrical radius
           z - vertical height
           phi - azimuth
           t - time
        OUTPUT:
           the second radial derivative
        HISTORY:
           2021-04-19 - Written - Bovy (UofT)
        """
        phi2= self._2ndderiv_xyz(R,z,phi=phi,t=t)
        return numpy.cos(phi)**2.*phi2[0,0]+numpy.sin(phi)**2.*phi2[1,1]\
            +2.*numpy.cos(phi)*numpy.sin(phi)*phi2[0,1]

    @check_potential_inputs_not_arrays
    def _Rzderiv(self,R,z,phi=0.,t=0.):
        """
        NAME:
           _Rzderiv
        PURPOSE:
           evaluate the mixed radial, vertical derivative for this potential
        INPUT:
           R - Galactocentric cylindrical radius
           z - vertical height
           phi - azimuth
           t - time
        OUTPUT:
           the mixed radial, vertical derivative
        HISTORY:
           2021-04-19 - Written - Bovy (UofT)
        """
        phi2= self._2ndderiv_xyz(R,z,phi=phi,t=t)
        return numpy.cos(phi)*phi2[0,2]+numpy.sin(phi)*phi2[1,2]

    @check_potential_inputs_not_arrays
    def _z2deriv(self,R,z,phi=0.,t=0.):
        """
        NAME:
           _z2deriv
        PURPOSE:
           evaluate the second vertical derivative for this potential
        INPUT:
           R - Galactocentric cylindrical radius
           z - vertical height
           phi - azimuth
           t - time
        OUTPUT:
           the second vertical derivative
        HISTORY:
           2021-04-19 - Written - Bovy (UofT)
        """
        return self._2ndderiv_xyz(R,z,phi=phi,t=t)[2,2]

    @check_potential_inputs_not_arrays
    def _phi2deriv(self,R,z,phi=0.,t=0.):
        """
        NAME:
           _phi2deriv
        PURPOSE:
           evaluate the second azimuthal derivative for this potential
        INPUT:
           R - Galactocentric cylindrical radius
           z - vertical height
           phi - azimuth
           t - time
        OUTPUT:
           the second azimuthal derivative
        HISTORY:
           2021-04-19 - Written - Bovy (UofT)
        """
        Fxyz= self._force_xyz(R,z,phi=phi,t=t)
        phi2= self._2ndderiv_xyz(R,z,phi=phi,t=t)
        return R**2.*(numpy.sin(phi)**2.*phi2[0,0]
                      +numpy.cos(phi)**2.*phi2[1,1]\
                      -2.*numpy.cos(phi)*numpy.sin(phi)*phi2[0,1])\
                      +R*(numpy.cos(phi)*Fxyz[0]+numpy.sin(phi)*Fxyz[1])

    @check_potential_inputs_not_arrays
    def _Rphideriv(self,R,z,phi=0.,t=0.):
        """
        NAME:
           _Rphideriv
        PURPOSE:
           evaluate the mixed radial, azimuthal derivative for this potential
        INPUT:
           R - Galactocentric cylindrical radius
           z - vertical height
           phi - azimuth
           t - time
        OUTPUT:
           the mixed radial, azimuthal derivative
        HISTORY:
           2021-04-19 - Written - Bovy (UofT)
        """
        Fxyz= self._force_xyz(R,z,phi=phi,t=t)
        phi2= self._2ndderiv_xyz(R,z,phi=phi,t=t)
        return R*numpy.cos(phi)*numpy.sin(phi)*\
            (phi2[1,1]-phi2[0,0])+R*numpy.cos(2.*phi)*phi2[0,1]\
            +numpy.sin(phi)*Fxyz[0]-numpy.cos(phi)*Fxyz[1]

    @check_potential_inputs_not_arrays
    def _phizderiv(self,R,z,phi=0.,t=0.):
        """
        NAME:
           _phizderiv
        PURPOSE:
           evaluate the mixed azimuthal, vertical derivative for this potential
        INPUT:
           R - Galactocentric cylindrical radius
           z - vertical height
           phi - azimuth
           t - time
        OUTPUT:
           the mixed azimuthal, vertical derivative
        HISTORY:
           2021-04-30 - Written - Bovy (UofT)
        """
        phi2= self._2ndderiv_xyz(R,z,phi=phi,t=t)
        return R*(numpy.cos(phi)*phi2[1,2]-numpy.sin(phi)*phi2[0,2])

    def _2ndderiv_xyz(self,R,z,phi=0.,t=0.):
        """Get the rectangular forces in the transformed frame"""
        x,y,z= coords.cyl_to_rect(R,phi,z)
        xyzp= numpy.dot(self._rot,numpy.array([x,y,z]))
        Rp,phip,zp =coords.rect_to_cyl(xyzp[0],xyzp[1],xyzp[2])
        Rforcep= _evaluateRforces(self._pot,Rp,zp,phi=phip,t=t)
        phiforcep= _evaluatephiforces(self._pot,Rp,zp,phi=phip,t=t)
        R2derivp= evaluateR2derivs(self._pot,Rp,zp,phi=phip,t=t,
                                   use_physical=False)
        phi2derivp= evaluatephi2derivs(self._pot,Rp,zp,phi=phip,t=t,
                                       use_physical=False)
        z2derivp= evaluatez2derivs(self._pot,Rp,zp,phi=phip,t=t,
                                   use_physical=False)
        Rzderivp= evaluateRzderivs(self._pot,Rp,zp,phi=phip,t=t,
                                   use_physical=False)
        Rphiderivp= evaluateRphiderivs(self._pot,Rp,zp,phi=phip,t=t,
                                       use_physical=False)
        phizderivp= evaluatephizderivs(self._pot,Rp,zp,phi=phip,t=t,
                                       use_physical=False)
        cp, sp= numpy.cos(phip), numpy.sin(phip)
        cp2, sp2, cpsp= cp**2., sp**2., cp*sp
        Rp2= Rp*Rp
        x2derivp= R2derivp*cp2-2.*Rphiderivp*cpsp/Rp+phi2derivp*sp2/Rp2\
            -Rforcep*sp2/Rp-2.*phiforcep*cpsp/Rp2
        y2derivp= R2derivp*sp2+2.*Rphiderivp*cpsp/Rp+phi2derivp*cp2/Rp2\
            -Rforcep*cp2/Rp+2.*phiforcep*cpsp/Rp2
        xyderivp= R2derivp*cpsp+Rphiderivp*(cp2-sp2)/Rp-phi2derivp*cpsp/Rp2\
            +Rforcep*cpsp/Rp+phiforcep*(cp2-sp2)/Rp2
        xzderivp= Rzderivp*cp-phizderivp*sp/Rp
        yzderivp= Rzderivp*sp+phizderivp*cp/Rp
        return numpy.dot(self._inv_rot,numpy.dot(\
                                numpy.array([[x2derivp,xyderivp,xzderivp],
                                             [xyderivp,y2derivp,yzderivp],
                                             [xzderivp,yzderivp,z2derivp]]),
                                self._inv_rot.T))

    @check_potential_inputs_not_arrays
    def _dens(self,R,z,phi=0.,t=0.):
        """
        NAME:
           _dens
        PURPOSE:
           evaluate the density for this potential
        INPUT:
           R - Galactocentric cylindrical radius
           z - vertical height
           phi - azimuth
           t - time
        OUTPUT:
           the density
        HISTORY:
           2021-04-18 - Written - Bovy (UofT)
        """
        x,y,z= coords.cyl_to_rect(R,phi,z)
        if numpy.isinf(R): y= 0.
        xyzp= numpy.dot(self._rot,numpy.array([x,y,z]))
        Rp,phip,zp = coords.rect_to_cyl(xyzp[0],xyzp[1],xyzp[2])
        return evaluateDensities(self._pot,Rp,zp,phi=phip,t=t,
                                 use_physical=False)

   
