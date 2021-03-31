###############################################################################
#   RotateAndTiltWrapperPotential.py: Wrapper to tilt the z-axis of a potential
###############################################################################
import numpy
from .WrapperPotential import parentWrapperPotential
from ..util import conversion
from ..util import _rotate_to_arbitrary_vector
from ..util import coords
class RotateAndTiltWrapperPotential(parentWrapperPotential):
    """ Potential wrapper class that implements an adjustment to the z-axis vector of a given Potential. This can be used, for example, to tilt a disc to a desired inclination angle
    """
    def __init__(self,amp=1.,zvec=None,pot=None,
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
        self._setup_zvec(zvec)
        self.hasC= True
        self.hasC_dxdv= True
        self.isNonAxi = True

    def _setup_zvec(self,zvec):
        """ taken from EllipsoidalPotential """
        if not zvec is None:
            if not isinstance(zvec,numpy.ndarray):
                zvec= numpy.array(zvec)
            zvec/= numpy.sqrt(numpy.sum(zvec**2.))
            zvec_rot= _rotate_to_arbitrary_vector(\
                numpy.array([[0.,0.,1.]]),zvec,inv=True)[0]
            self._rot = zvec_rot
        else:
            self._rot = numpy.eye(3)

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
