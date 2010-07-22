import numpy as nu
from scipy import interpolate
from Potential import Potential
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
        Rforce= nu.zeros((len(self._rgrid),len(self._zgrid)))
        zforce= nu.zeros((len(self._rgrid),len(self._zgrid)))
        print "Computing forces on grid"
        for ii in range(len(self._rgrid)):
            for jj in range(len(self._zgrid)):
                Rforce[ii,jj]= RZPot.Rforce(self._rgrid[ii],self._zgrid[jj])
                zforce[ii,jj]= RZPot.zforce(self._rgrid[ii],self._zgrid[jj])
        self._interpRforce= interpolate.bisplrep(self._rgrid,self._zgrid,
                                                 Rforce,s=0)
        self._interpzforce= interpolate.bisplrep(self._rgrid,self._zgrid,
                                                 zforce,s=0)
                                                 

    def _Rforce(R,z):
        if R < self._rgrid[0] or R > self._rgrid[-1] \
                or z < self._zgrid[0] or z > self._zgrid[-1]:
            print "Current position out of range of interpolation, consider interpolating on a larger range"
            return self._RZPot.Rforce(R,z)
        else:
            return interpolate.bisplev(R,z,self._interpRforce)

    def _zforce(R,z):
        if R < self._rgrid[0] or R > self._rgrid[-1] \
                or z < self._zgrid[0] or z > self._zgrid[-1]:
            print "Current position out of range of interpolation, consider interpolating on a larger range"
            return self._RZPot.zforce(R,z)
        else:
            return interpolate.bisplev(R,z,self._interpzforce)
