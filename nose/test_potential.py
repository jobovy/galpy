############################TESTS ON POTENTIALS################################

#Test whether the normalization of the potential works
def test_normalize_potential():
    from galpy import potential
    #Grab all of the potentials
    pots= [p for p in dir(potential) 
           if ('Potential' in p and not 'plot' in p and not 'RZTo' in p 
               and not 'evaluate' in p)]
    rmpots= ['Potential','MWPotential','MovingObjectPotential',
             'interpRZPotential', 'linearPotential', 'planarAxiPotential',
             'planarPotential', 'verticalPotential']
    for p in rmpots:
        pots.remove(p)
    for p in pots:
        #if not 'NFW' in p: continue #For testing the test
        #Setup instance of potential
        tclass= getattr(potential,p)
        tp= tclass()
        if not hasattr(tp,'normalize'): continue
        tp.normalize(1.)
        try:
            assert((tp.Rforce(1.,0.)+1.)**2. < 10.**-16.)
        except AssertionError:
            raise AssertionError("Normalization of %s potential does not work" % p)
        tp.normalize(.5)
        try:
            assert((tp.Rforce(1.,0.)+.5)**2. < 10.**-16.)
        except AssertionError:
            raise AssertionError("Normalization of %s potential does not work" % p)

