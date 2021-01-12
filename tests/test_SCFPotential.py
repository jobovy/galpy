from __future__ import print_function, division
import numpy
from galpy import potential

def test_spherical():
    #Test that the SCF coefficient estimators make a spherical potential
    
    N= int(1e6)
    Mh= 10.
    ah= 50./8.
    m= Mh/N

    hp= HernquistPotential(amp=2*Mh,a=ah)
    hdf= isotropicHernquistdf(hp)
    samp= hdf.sample(n=N)
    
    positions= numpy.array([samp.x(),samp.y(),samp.z()])

    Acdens, Asdens= scf_compute_coeffs(hp.dens,10,10,a=50./8.)
    Acnbdy, Asnbdy= scf_compute_coeffs_nbody(positions,m*numpy.ones(N),10,10,a=50./8.)
    
    potdens= potential.SCFPotential(Acos=Acos1,Asin=Asin1,a=50./8.)
    potnbdy= potential.SCFPotential(Acos=Acos2,Asin=Asin2,a=50./8.)

    potdens.turn_physical_off()
    potnbdy.turn_physical_off()
    pothern.turn_physical_off()
    
    r= np.linspace(0,1,100)
    z= np.linspace(-0.5,0.5,100)

    ppot= hp.dens(r[:,None],z[None,:])
    pdens= pot1.dens(r[:,None],z[None,:])
    pnbdy= pot2.dens(r[:,None],z[None,:])

    assert numpy.max(numpy.abs((ppot-pdens)/ppot))<1.e-3, 'scf_compute_coeffs did not meet the required tolerance of 1e-3'
    assert numpy.max(numpy.abs((ppot-pnbdy)/ppot))<0.1, 'scf_compute_coeffs_nbody did not meet the required tolerance of 0.1'