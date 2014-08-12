# Tests of the quasiisothermaldf module
import numpy
#fiducial setup uses these
from galpy.potential import MWPotential, vcirc
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
    numpy.random.seed(1)
    qdf= quasiisothermaldf(1./4.,0.2,0.1,1.,1.,
                           pot=MWPotential,aA=aAA,cutcounter=True)
    #In the mid-plane
    assert numpy.fabs(qdf.meanvR(0.9,0.,mc=True)) < 0.01, "qdf's meanvr is not equal to zero for adiabatic approx."
    #higher up
    assert numpy.fabs(qdf.meanvR(0.9,0.2,mc=True)) < 0.05, "qdf's meanvr is not equal to zero for adiabatic approx."
    assert numpy.fabs(qdf.meanvR(0.9,-0.25,mc=True)) < 0.05, "qdf's meanvr is not equal to zero for adiabatic approx."
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
    numpy.random.seed(1)
    qdf= quasiisothermaldf(1./4.,0.2,0.1,1.,1.,
                           pot=MWPotential,aA=aAS,cutcounter=True)
    #In the mid-plane
    assert numpy.fabs(qdf.meanvR(0.9,0.,mc=True)) < 0.01, "qdf's meanvr is not equal to zero for staeckel approx."
    #higher up
    assert numpy.fabs(qdf.meanvR(0.9,0.2,mc=True)) < 0.05, "qdf's meanvr is not equal to zero for staeckel approx."
    assert numpy.fabs(qdf.meanvR(0.9,-0.25,mc=True)) < 0.05, "qdf's meanvr is not equal to zero for staeckel approx."
    return None

def test_meanvT_adiabatic_gl():
    qdf= quasiisothermaldf(1./4.,0.2,0.1,1.,1.,
                           pot=MWPotential,aA=aAA,cutcounter=True)
    from galpy.df import dehnendf #baseline
    dfc= dehnendf(profileParams=(1./4.,1.0, 0.2),
                  beta=0.,correct=False)
    #In the mid-plane
    vtp9= qdf.meanvT(0.9,0.,gl=True)
    assert numpy.fabs(vtp9-dfc.meanvT(0.9)) < 0.05, "qdf's meanvT is not close to that of dehnendf"
    assert vtp9 <  vcirc(MWPotential,0.9), "qdf's meanvT is not less than the circular velocity (which we expect)"
    #higher up
    assert qdf.meanvR(0.9,0.2,gl=True) < vtp9, "qdf's meanvT above the plane is not less than in the plane (which we expect)"
    assert qdf.meanvR(0.9,-0.25,gl=True) < vtp9, "qdf's meanvT above the plane is not less than in the plane (which we expect)"
    return None

def test_meanvT_adiabatic_mc():
    numpy.random.seed(1)
    qdf= quasiisothermaldf(1./4.,0.2,0.1,1.,1.,
                           pot=MWPotential,aA=aAA,cutcounter=True)
    from galpy.df import dehnendf #baseline
    dfc= dehnendf(profileParams=(1./4.,1.0, 0.2),
                  beta=0.,correct=False)
    #In the mid-plane
    vtp9= qdf.meanvT(0.9,0.,mc=True)
    assert numpy.fabs(vtp9-dfc.meanvT(0.9)) < 0.05, "qdf's meanvT is not close to that of dehnendf"
    assert vtp9 <  vcirc(MWPotential,0.9), "qdf's meanvT is not less than the circular velocity (which we expect)"
    #higher up
    assert qdf.meanvR(0.9,0.2,mc=True) < vtp9, "qdf's meanvT above the plane is not less than in the plane (which we expect)"
    assert qdf.meanvR(0.9,-0.25,mc=True) < vtp9, "qdf's meanvT above the plane is not less than in the plane (which we expect)"
    return None

def test_meanvT_staeckel_gl():
    qdf= quasiisothermaldf(1./4.,0.2,0.1,1.,1.,
                           pot=MWPotential,aA=aAS,cutcounter=True)
    from galpy.df import dehnendf #baseline
    dfc= dehnendf(profileParams=(1./4.,1.0, 0.2),
                  beta=0.,correct=False)
    #In the mid-plane
    vtp9= qdf.meanvT(0.9,0.,gl=True)
    assert numpy.fabs(vtp9-dfc.meanvT(0.9)) < 0.05, "qdf's meanvT is not close to that of dehnendf"
    assert vtp9 <  vcirc(MWPotential,0.9), "qdf's meanvT is not less than the circular velocity (which we expect)"
    #higher up
    assert qdf.meanvR(0.9,0.2,gl=True) < vtp9, "qdf's meanvT above the plane is not less than in the plane (which we expect)"
    assert qdf.meanvR(0.9,-0.25,gl=True) < vtp9, "qdf's meanvT above the plane is not less than in the plane (which we expect)"
    return None

def test_meanvT_staeckel_mc():
    numpy.random.seed(1)
    qdf= quasiisothermaldf(1./4.,0.2,0.1,1.,1.,
                           pot=MWPotential,aA=aAS,cutcounter=True)
    from galpy.df import dehnendf #baseline
    dfc= dehnendf(profileParams=(1./4.,1.0, 0.2),
                  beta=0.,correct=False)
    #In the mid-plane
    vtp9= qdf.meanvT(0.9,0.,mc=True)
    assert numpy.fabs(vtp9-dfc.meanvT(0.9)) < 0.05, "qdf's meanvT is not close to that of dehnendf"
    assert vtp9 <  vcirc(MWPotential,0.9), "qdf's meanvT is not less than the circular velocity (which we expect)"
    #higher up
    assert qdf.meanvR(0.9,0.2,mc=True) < vtp9, "qdf's meanvT above the plane is not less than in the plane (which we expect)"
    assert qdf.meanvR(0.9,-0.25,mc=True) < vtp9, "qdf's meanvT above the plane is not less than in the plane (which we expect)"
    return None

def test_meanvz_adiabatic_gl():
    qdf= quasiisothermaldf(1./4.,0.2,0.1,1.,1.,
                           pot=MWPotential,aA=aAA,cutcounter=True)
    #In the mid-plane
    assert numpy.fabs(qdf.meanvz(0.9,0.,gl=True)) < 0.01, "qdf's meanvr is not equal to zero for adiabatic approx."
    #higher up
    assert numpy.fabs(qdf.meanvz(0.9,0.2,gl=True)) < 0.01, "qdf's meanvr is not equal to zero for adiabatic approx."
    assert numpy.fabs(qdf.meanvz(0.9,-0.25,gl=True)) < 0.01, "qdf's meanvr is not equal to zero for adiabatic approx."
    return None

def test_meanvz_adiabatic_mc():
    numpy.random.seed(1)
    qdf= quasiisothermaldf(1./4.,0.2,0.1,1.,1.,
                           pot=MWPotential,aA=aAA,cutcounter=True)
    #In the mid-plane
    assert numpy.fabs(qdf.meanvz(0.9,0.,mc=True)) < 0.01, "qdf's meanvr is not equal to zero for adiabatic approx."
    #higher up
    assert numpy.fabs(qdf.meanvz(0.9,0.2,mc=True)) < 0.05, "qdf's meanvr is not equal to zero for adiabatic approx."
    assert numpy.fabs(qdf.meanvz(0.9,-0.25,mc=True)) < 0.05, "qdf's meanvr is not equal to zero for adiabatic approx."
    return None

def test_meanvz_staeckel_gl():
    qdf= quasiisothermaldf(1./4.,0.2,0.1,1.,1.,
                           pot=MWPotential,aA=aAS,cutcounter=True)
    #In the mid-plane
    assert numpy.fabs(qdf.meanvz(0.9,0.,gl=True)) < 0.01, "qdf's meanvr is not equal to zero for staeckel approx."
    #higher up
    assert numpy.fabs(qdf.meanvz(0.9,0.2,gl=True)) < 0.01, "qdf's meanvr is not equal to zero for staeckel approx."
    assert numpy.fabs(qdf.meanvz(0.9,-0.25,gl=True)) < 0.01, "qdf's meanvr is not equal to zero for staeckel approx."
    return None

def test_meanvz_staeckel_mc():
    numpy.random.seed(1)
    qdf= quasiisothermaldf(1./4.,0.2,0.1,1.,1.,
                           pot=MWPotential,aA=aAS,cutcounter=True)
    #In the mid-plane
    assert numpy.fabs(qdf.meanvz(0.9,0.,mc=True)) < 0.01, "qdf's meanvr is not equal to zero for staeckel approx."
    #higher up
    assert numpy.fabs(qdf.meanvz(0.9,0.2,mc=True)) < 0.05, "qdf's meanvr is not equal to zero for staeckel approx."
    assert numpy.fabs(qdf.meanvz(0.9,-0.25,mc=True)) < 0.05, "qdf's meanvr is not equal to zero for staeckel approx."
    return None

def test_sigmar_staeckel_gl():
    qdf= quasiisothermaldf(1./4.,0.2,0.1,1.,1.,
                           pot=MWPotential,aA=aAS,cutcounter=True)
    #In the mid-plane
    assert numpy.fabs(numpy.log(qdf.sigmaR2(0.9,0.,gl=True))-2.*numpy.log(0.2)-0.2) < 0.2, "qdf's sigmaR2 deviates more than expected from input for staeckel approx."
    #higher up
    assert numpy.fabs(numpy.log(qdf.sigmaR2(0.9,0.2,gl=True))-2.*numpy.log(0.2)-0.2) < 0.3, "qdf's sigmaR2 deviates more than expected from input for staeckel approx."
    assert numpy.fabs(numpy.log(qdf.sigmaR2(0.9,-0.25,gl=True))-2.*numpy.log(0.2)-0.2) < 0.3, "qdf's sigmaR2 deviates more than expected from input for staeckel approx."
    return None

def test_sigmar_staeckel_mc():
    qdf= quasiisothermaldf(1./4.,0.2,0.1,1.,1.,
                           pot=MWPotential,aA=aAS,cutcounter=True)
    #In the mid-plane
    assert numpy.fabs(numpy.log(qdf.sigmaR2(0.9,0.,mc=True))-2.*numpy.log(0.2)-0.2) < 0.2, "qdf's sigmaR2 deviates more than expected from input for staeckel approx."
    #higher up
    assert numpy.fabs(numpy.log(qdf.sigmaR2(0.9,0.2,mc=True))-2.*numpy.log(0.2)-0.2) < 0.4, "qdf's sigmaR2 deviates more than expected from input for staeckel approx."
    assert numpy.fabs(numpy.log(qdf.sigmaR2(0.9,-0.25,mc=True))-2.*numpy.log(0.2)-0.2) < 0.3, "qdf's sigmaR2 deviates more than expected from input for staeckel approx."
    return None

