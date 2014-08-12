# Tests of the quasiisothermaldf module
import numpy
#fiducial setup uses these
from galpy.potential import MWPotential
from galpy.actionAngle import actionAngleAdiabatic, actionAngleStaeckel
from galpy.df import quasiisothermaldf
aAA= actionAngleAdiabatic(pot=MWPotential,c=True)
aAS= actionAngleStaeckel(pot=MWPotential,c=True,delta=0.5)

def test_meanvR_adiabatic_gl():
    qdf= quasiisothermaldf(1./4.,0.2,0.1,1.,1.,
                           pot=MWPotential,aA=aAA,cutcounter=True)
    #In the mid-plane
    assert numpy.fabs(qdf.meanvR(0.9,0.,gl=True)) < 0.01, "qdf's meanvr is not equal to zero for adiabatic approx."
    #higher up
    assert numpy.fabs(qdf.meanvR(0.9,0.2,gl=True)) < 0.01, "qdf's meanvr is not equal to zero for adiabatic approx."
    assert numpy.fabs(qdf.meanvR(0.9,-0.25,gl=True)) < 0.01, "qdf's meanvr is not equal to zero for adiabatic approx."
    return None

def test_meanvR_adiabatic_mc():
    qdf= quasiisothermaldf(1./4.,0.2,0.1,1.,1.,
                           pot=MWPotential,aA=aAA,cutcounter=True)
    #In the mid-plane
    assert numpy.fabs(qdf.meanvR(0.9,0.,mc=True)) < 0.01, "qdf's meanvr is not equal to zero for adiabatic approx."
    #higher up
    assert numpy.fabs(qdf.meanvR(0.9,0.2,mc=True)) < 0.01, "qdf's meanvr is not equal to zero for adiabatic approx."
    assert numpy.fabs(qdf.meanvR(0.9,-0.25,mc=True)) < 0.01, "qdf's meanvr is not equal to zero for adiabatic approx."
    return None

def test_meanvR_staeckel_gl():
    qdf= quasiisothermaldf(1./4.,0.2,0.1,1.,1.,
                           pot=MWPotential,aA=aAS,cutcounter=True)
    #In the mid-plane
    assert numpy.fabs(qdf.meanvR(0.9,0.,gl=True)) < 0.01, "qdf's meanvr is not equal to zero for staeckel approx."
    #higher up
    assert numpy.fabs(qdf.meanvR(0.9,0.2,gl=True)) < 0.01, "qdf's meanvr is not equal to zero for staeckel approx."
    assert numpy.fabs(qdf.meanvR(0.9,-0.25,gl=True)) < 0.01, "qdf's meanvr is not equal to zero for staeckel approx."
    return None

def test_meanvR_staeckel_mc():
    qdf= quasiisothermaldf(1./4.,0.2,0.1,1.,1.,
                           pot=MWPotential,aA=aAS,cutcounter=True)
    #In the mid-plane
    assert numpy.fabs(qdf.meanvR(0.9,0.,mc=True)) < 0.01, "qdf's meanvr is not equal to zero for staeckel approx."
    #higher up
    assert numpy.fabs(qdf.meanvR(0.9,0.2,mc=True)) < 0.01, "qdf's meanvr is not equal to zero for staeckel approx."
    assert numpy.fabs(qdf.meanvR(0.9,-0.25,mc=True)) < 0.01, "qdf's meanvr is not equal to zero for staeckel approx."
    return None

