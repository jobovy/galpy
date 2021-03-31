###############################################################################
#   RotateAndTiltWrapperPotential.py: Wrapper to rotate and tilt the z-axis
#   of a potential
###############################################################################
import numpy
from .WrapperPotential import parentWrapperPotential
from ..util import conversion
from ..util import _rotate_to_arbitrary_vector
from ..util import coords
class RotateAndTiltWrapperPotential(parentWrapperPotential):
    """ Potential wrapper class that implements an adjustment to the rotation
        and z-axis vector (tilt) of a given Potential. This can be used,
        for example, to tilt a disc to a desired inclination angle
    """
    def __init__(self,amp=1.,zvec=None,pa=None,pot=None,
                 ro=None,vo=None):
        """
        NAME:

           __init__

        PURPOSE:

           initialize a RotateAndTiltWrapper Potential

        INPUT:

           zvec - the vector along the required final z-axis

        OUTPUT:

           (none)

        HISTORY:

           2020-03-29 - Started - Mackereth (UofT)

        """
        self._setup_zvec_pa(zvec, pa)
        self._inv_rot = numpy.linalg.inv(self._rot)
        self.hasC= True
        self.hasC_dxdv= True
        self.isNonAxi = True

    def _setup_zvec_pa(self,zvec,pa):
        """ taken from EllipsoidalPotential """
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
        return None

    def _wrap(self,attribute,*args,**kwargs):
        #need to convert input R,phi,z to x,y,z for rotation
        R,phi,z = args[0],kwargs.get('phi',0.),0 if len(args) == 1 else args[1]
        if phi is None:
            phi = 0.
        x,y,z= coords.cyl_to_rect(R, phi, z)
        #apply rotation matrix
        xyzp= numpy.dot(self._rot,numpy.array([x,y,z]))
        #back to R,phi,z
        R, phi, z = coords.rect_to_cyl(xyzp[0], xyzp[1], xyzp[2])
        args = (R, z)
        kwargs['phi'] = phi
        return self._wrap_pot_func(attribute)(self._pot,*args,**kwargs)

    def _force_xyz(self,*args,**kwargs):
        """ get the rectangular forces in the transformed frame """
        #first figure out the R,phi,z force in the aligned frame...
        o_R,o_phi,o_z = args[0],kwargs.get('phi',0.),0 if len(args) == 1 else args[1]
        if o_phi is None:
            o_phi = 0.
        x,y,z= coords.cyl_to_rect(o_R, o_phi, o_z)
        #apply rotation matrix
        xyzp= numpy.dot(self._rot,numpy.array([x,y,z]))
        #back to R,phi,z
        t_R, t_phi, t_z = coords.rect_to_cyl(xyzp[0], xyzp[1], xyzp[2])
        args = (t_R, t_z)
        kwargs['phi'] = t_phi
        #get the forces
        Rforce_a = self._wrap_pot_func('_Rforce')(self._pot,*args,**kwargs)
        phiforce_a = self._wrap_pot_func('_phiforce')(self._pot,*args,**kwargs)
        zforce_a = self._wrap_pot_func('_zforce')(self._pot,*args,**kwargs)
        #get the forces in x,y
        xforce_a = numpy.cos(t_phi)*Rforce_a - numpy.sin(t_phi)*phiforce_a
        yforce_a = numpy.sin(t_phi)*Rforce_a + numpy.cos(t_phi)*phiforce_a
        #rotate back
        Fxyz = numpy.dot(self._inv_rot, numpy.array([xforce_a,yforce_a,zforce_a]))
        return Fxyz

    def _Rforce(self,*args,**kwargs):
        Fxyz = self._force_xyz(*args,**kwargs)
        o_R,o_phi,o_z = args[0],kwargs.get('phi',0.),0 if len(args) == 1 else args[1]
        if o_phi is None:
            o_phi = 0.
        return numpy.cos(o_phi)*Fxyz[0] + numpy.sin(o_phi)*Fxyz[1]

    def _phiforce(self,*args,**kwargs):
        Fxyz = self._force_xyz(*args,**kwargs)
        o_R,o_phi,o_z = args[0],kwargs.get('phi',0.),0 if len(args) == 1 else args[1]
        if o_phi is None:
            o_phi = 0.
        return -numpy.sin(o_phi)*Fxyz[0] + numpy.cos(o_phi)*Fxyz[1]

    def _zforce(self,*args,**kwargs):
        Fxyz = self._force_xyz(*args,**kwargs)
        return Fxyz[2]
