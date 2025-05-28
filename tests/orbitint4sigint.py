import sys

import numpy

from galpy.orbit import Orbit
from galpy.potential import MiyamotoNagaiPotential

if __name__ == "__main__":
    # python orbitint4sigint.py symplec4_c full
    mp = MiyamotoNagaiPotential(normalize=1.0, a=0.5, b=0.05)
    ts = numpy.linspace(0.0, 10000000.0, 1000001)
    if sys.argv[2] == "full" or sys.argv[2] == "planar":
        if sys.argv[2] == "full":
            o = Orbit([1.0, 0.1, 1.1, 0.1, 0.1, 0.0])
        elif sys.argv[2] == "planar":
            o = Orbit([1.0, 0.1, 1.1, 0.1])
        print("Starting long C integration ...")
        sys.stdout.flush()
        o.integrate(ts, mp, method=sys.argv[1])
    elif sys.argv[2] == "planardxdv":
        o = Orbit([1.0, 0.1, 1.1, 0.1])
        print("Starting long C integration ...")
        sys.stdout.flush()
        o.integrate_dxdv([0.1, 0.1, 0.1, 0.1], ts, mp, method=sys.argv[1])
    sys.exit(0)
