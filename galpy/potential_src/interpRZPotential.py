import numpy as nu
from scipy import interpolate
from Potential import Potential
_DEBUG= True
class interpRZPotential(Potential):
    """Class that interpolates a given potential on a grid for fast orbit integration"""
    def __init__(self,RZPot,rgrid=(0.01,2.,101),zgrid=(-0.2,0.2,101),
                 logR=False):
        """
        NAME:
           __init__
        PURPOSE:
           Initialize an interpRZPotential instance
        INPUT:
           RZPot - RZPotential to be interpolated
           rgrid - R grid to be given to linspace
           zgrid - z grid to be given to linspace
           logR - if True, rgrid is in the log of R
        OUTPUT:
           instance
        HISTORY:
           2010-07-21 - Written - Bovy (NYU)
        """
        Potential.__init__(self,amp=1.)
        self._origPot= RZPot
        self._rgrid= nu.linspace(*rgrid)
        if logR:
            self._rgrid= nu.exp(self._rgrid)
        self._zgrid= nu.linspace(*zgrid)
        R= nu.zeros(len(self._rgrid)*len(self._zgrid))
        z= nu.zeros(len(self._rgrid)*len(self._zgrid))
        Rforce= nu.zeros(len(self._rgrid)*len(self._zgrid))
        zforce= nu.zeros(len(self._rgrid)*len(self._zgrid))
        if _DEBUG:
            print "Computing forces on grid ..."
        for ii in range(len(self._rgrid)):
            for jj in range(len(self._zgrid)):
                Rforce[ii*len(self._zgrid)+jj]= RZPot.Rforce(self._rgrid[ii],
                                                             self._zgrid[jj])
                zforce[ii*len(self._zgrid)+jj]= RZPot.zforce(self._rgrid[ii],
                                                             self._zgrid[jj])
                R[ii*len(self._zgrid)+jj]= self._rgrid[ii]
                z[ii*len(self._zgrid)+jj]= self._zgrid[jj]
        if _DEBUG:
            print "Interpolating ..."
        self._interpRforce= interpolate.interp2d(R,z,Rforce,bounds_error=True)
        self._interpzforce= interpolate.interp2d(R,z,zforce,bounds_error=True)
                                                 

    def _Rforce(self,R,z,phi=0.):
        if R < self._rgrid[0] or R > self._rgrid[-1] \
                or z < self._zgrid[0] or z > self._zgrid[-1]:
            print "Current position out of range of interpolation, consider interpolating on a larger range"
            return self._origPot.Rforce(R,z)
        else:
            return self._interpRforce(R,z)

    def _zforce(self,R,z,phi=0.):
        if R < self._rgrid[0] or R > self._rgrid[-1] \
                or z < self._zgrid[0] or z > self._zgrid[-1]:
            print "Current position out of range of interpolation, consider interpolating on a larger range"
            return self._origPot.zforce(R,z)
        else:
            return self._interpzforce(R,z)
