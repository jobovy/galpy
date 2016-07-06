import sys
import numpy
from galpy.orbit import Orbit
from galpy.potential import DoubleExponentialDiskPotential

if __name__ == '__main__':
    # python orbitint4sigint.py symplec4_c full
    dp= DoubleExponentialDiskPotential(normalize=1.,hr=0.5,hz=0.05)
    if sys.argv[2] == 'full':
        o= Orbit([1.,0.1,1.1,0.1,0.1,0.])
    elif sys.argv[2] == 'planar':
        o= Orbit([1.,0.1,1.1,0.1])
    ts= numpy.linspace(0.,100000.,10001)
    o.integrate(ts,dp,method=sys.argv[1])
    sys.exit(0)
