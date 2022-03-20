# Tests of integrating orbits in non-inertial frames
import pytest
import numpy
from galpy import potential
from galpy.orbit import Orbit
from galpy.util import coords

def test_lsrframe_scalaromegaz():
    # Test that integrating an orbit in the LSR frame is equivalent to 
    # normal orbit integration
    lp= potential.LogarithmicHaloPotential(normalize=1.)
    omega= lp.omegac(1.)
    dp= potential.DehnenBarPotential(omegab=1.8,rb=0.5,Af=0.03)
    diskpot= lp+dp
    framepot= potential.NonInertialFrameForce(Omega=omega)
    dp_frame= potential.DehnenBarPotential(omegab=1.8-omega,rb=0.5,Af=0.03)
    diskframepot= lp+dp_frame+framepot
    # Now integrate the orbit of the Sun in both the inertial and the lsr frame
    def check_orbit(method='odeint',tol=1e-9):
        o= Orbit()
        o.turn_physical_off()
        # Inertial frame
        ts= numpy.linspace(0.,20.,1001)
        o.integrate(ts,diskpot)
        # Non-inertial frame
        op= Orbit([o.R(),o.vR(),o.vT()-omega*o.R(),o.z(),o.vz(),o.phi()])
        op.integrate(ts,diskframepot,method=method)
        # Compare
        o_xs= o.R(ts)*numpy.cos(o.phi(ts)-omega*ts)
        o_ys= o.R(ts)*numpy.sin(o.phi(ts)-omega*ts)
        op_xs= op.x(ts)
        op_ys= op.y(ts)
        assert numpy.amax(numpy.fabs(o_xs-op_xs)) <tol, 'Integrating an orbit in the rotating LSR frame does not agree with the equivalent orbit in the intertial frame for integration method {}'.format(method)
        assert numpy.amax(numpy.fabs(o_ys-op_ys)) < tol, 'Integrating an orbit in the rotating LSR frame does not agree with the equivalent orbit in the intertial frame for integration method {}'.format(method)
    check_orbit(method='odeint',tol=1e-6)
    check_orbit(method='dop853_c',tol=1e-9)
    return None
    
def test_lsrframe_vecomegaz():
    # Test that integrating an orbit in the LSR frame is equivalent to 
    # normal orbit integration
    lp= potential.LogarithmicHaloPotential(normalize=1.)
    omega= lp.omegac(1.)
    dp= potential.DehnenBarPotential(omegab=1.8,rb=0.5,Af=0.03)
    diskpot= lp+dp
    framepot= potential.NonInertialFrameForce(Omega=numpy.array([0.,0.,omega]))
    dp_frame= potential.DehnenBarPotential(omegab=1.8-omega,rb=0.5,Af=0.03)
    diskframepot= lp+dp_frame+framepot
    # Now integrate the orbit of the Sun in both the inertial and the lsr frame
    def check_orbit(method='odeint',tol=1e-9):
        o= Orbit()
        o.turn_physical_off()
        # Inertial frame
        ts= numpy.linspace(0.,20.,1001)
        o.integrate(ts,diskpot)
        # Non-inertial frame
        op= Orbit([o.R(),o.vR(),o.vT()-omega*o.R(),o.z(),o.vz(),o.phi()])
        op.integrate(ts,diskframepot,method=method)
        # Compare
        o_xs= o.R(ts)*numpy.cos(o.phi(ts)-omega*ts)
        o_ys= o.R(ts)*numpy.sin(o.phi(ts)-omega*ts)
        op_xs= op.x(ts)
        op_ys= op.y(ts)
        assert numpy.amax(numpy.fabs(o_xs-op_xs)) < tol, 'Integrating an orbit in the rotating LSR frame does not agree with the equivalent orbit in the intertial frame for method {}'.format(method)
        assert numpy.amax(numpy.fabs(o_ys-op_ys)) < tol, 'Integrating an orbit in the rotating LSR frame does not agree with the equivalent orbit in the intertial frame for method {}'.format(method)
    check_orbit(method='odeint',tol=1e-6)
    check_orbit(method='dop853_c',tol=1e-9)
    return None    
        
def test_accellsrframe_scalaromegaz():
    # Test that integrating an orbit in an LSR frame that is accelerating
    # is equivalent to normal orbit integration
    lp= potential.LogarithmicHaloPotential(normalize=1.)
    omega= lp.omegac(1.)
    omegadot= 0.02
    diskpot= lp
    framepot= potential.NonInertialFrameForce(Omega=omega,Omegadot=omegadot)
    diskframepot= lp+framepot
    # Now integrate the orbit of the Sun in both the inertial and the lsr frame
    def check_orbit(method='odeint',tol=1e-9):
        o= Orbit()
        o.turn_physical_off()
        # Inertial frame
        ts= numpy.linspace(0.,20.,1001)
        o.integrate(ts,diskpot)
        # Non-inertial frame
        op= Orbit([o.R(),o.vR(),o.vT()-omega*o.R(),o.z(),o.vz(),o.phi()])
        op.integrate(ts,diskframepot,method=method)
        # Compare
        o_xs= o.R(ts)*numpy.cos(o.phi(ts)-omega*ts-omegadot*ts**2./2.)
        o_ys= o.R(ts)*numpy.sin(o.phi(ts)-omega*ts-omegadot*ts**2./2.)
        op_xs= op.x(ts)
        op_ys= op.y(ts)
        assert numpy.amax(numpy.fabs(o_xs-op_xs)) < tol, 'Integrating an orbit in the acceleratingly-rotating LSR frame does not agree with the equivalent orbit in the intertial frame for method {}'.format(method)
        assert numpy.amax(numpy.fabs(o_ys-op_ys)) < tol, 'Integrating an orbit in the acceleratingly-rotating LSR frame does not agree with the equivalent orbit in the intertial frame for method {}'.format(method)
    check_orbit(method='odeint',tol=1e-6)
    check_orbit(method='dop853_c',tol=1e-9)
    return None   

def test_accellsrframe_vecomegaz():
    # Test that integrating an orbit in an LSR frame that is accelerating
    # is equivalent to normal orbit integration
    lp= potential.LogarithmicHaloPotential(normalize=1.)
    omega= lp.omegac(1.)
    omegadot= 0.02
    diskpot= lp
    framepot= potential.NonInertialFrameForce(Omega=numpy.array([0.,0.,omega]),
                                              Omegadot=numpy.array([0.,0.,omegadot]))
    diskframepot= lp+framepot
    # Now integrate the orbit of the Sun in both the inertial and the lsr frame
    def check_orbit(method='odeint',tol=1e-9):
        o= Orbit()
        o.turn_physical_off()
        # Inertial frame
        ts= numpy.linspace(0.,20.,1001)
        o.integrate(ts,diskpot)
        # Non-inertial frame
        op= Orbit([o.R(),o.vR(),o.vT()-omega*o.R(),o.z(),o.vz(),o.phi()])
        op.integrate(ts,diskframepot,method=method)
        # Compare
        o_xs= o.R(ts)*numpy.cos(o.phi(ts)-omega*ts-omegadot*ts**2./2.)
        o_ys= o.R(ts)*numpy.sin(o.phi(ts)-omega*ts-omegadot*ts**2./2.)
        op_xs= op.x(ts)
        op_ys= op.y(ts)
        assert numpy.amax(numpy.fabs(o_xs-op_xs)) < tol, 'Integrating an orbit in the acceleratingly-rotating LSR frame does not agree with the equivalent orbit in the intertial frame for method {}'.format(method)
        assert numpy.amax(numpy.fabs(o_ys-op_ys)) < tol, 'Integrating an orbit in the acceleratingly-rotating LSR frame does not agree with the equivalent orbit in the intertial frame for method {}'.format(method)
    check_orbit(method='odeint',tol=1e-6)
    check_orbit(method='dop853_c',tol=1e-9)
    return None      

def test_linacc_vertical_constantacc_z():
    # Test that a linearly-accelerating frame along the z direction works
    # with a constant acceleration
    lp= potential.LogarithmicHaloPotential(normalize=1.)
    dp= potential.DehnenBarPotential(omegab=1.8,rb=0.5,Af=0.03)
    diskpot= lp+dp
    az= 0.02
    intaz= lambda t: 0.02*t**2./2.
    framepot= potential.NonInertialFrameForce(RTacm=[0.,0.,az])
    diskframepot= AcceleratingPotentialWrapperPotential(pot=diskpot,
                                                        a=[lambda t: 0.,
                                                           lambda t: 0.,
                                                           intaz])\
                    +framepot
    def check_orbit(method='odeint',tol=1e-9):
        o= Orbit()
        o.turn_physical_off()
        # Inertial frame
        ts= numpy.linspace(0.,20.,1001)
        o.integrate(ts,diskpot,method=method)
        # Non-inertial frame
        op= o()
        op.integrate(ts,diskframepot,method=method)
        # Compare
        o_xs= o.x(ts)
        o_ys= o.y(ts)
        o_zs= o.z(ts)
        op_xs= op.x(ts)
        op_ys= op.y(ts)
        op_zs= op.z(ts)+intaz(ts)
        assert numpy.amax(numpy.fabs(o_xs-op_xs)) < tol, 'Integrating an orbit in a linearly-accelerating frame with constant acceleration does not agree with the equivalent orbit in the intertial frame for method {}'.format(method)
        assert numpy.amax(numpy.fabs(o_ys-op_ys)) < tol, 'Integrating an orbit in a linearly-accelerating frame with constant acceleration does not agree with the equivalent orbit in the intertial frame for method {}'.format(method)
        assert numpy.amax(numpy.fabs(o_zs-op_zs)) < tol, 'Integrating an orbit in a linearly-accelerating frame with constant acceleration does not agree with the equivalent orbit in the intertial frame for method {}'.format(method)
    check_orbit(method='odeint',tol=1e-9)
    check_orbit(method='dop853',tol=1e-9)
    check_orbit(method='dop853_c',tol=1e-5) # Lower tol, because diff integrators for inertial and non-inertial, bc wrapper not implemented in C
    return None

def test_linacc_vertical_constantacc_xyz():
    # Test that a linearly-accelerating frame along the z direction works
    # with a constant acceleration
    lp= potential.LogarithmicHaloPotential(normalize=1.)
    dp= potential.DehnenBarPotential(omegab=1.8,rb=0.5,Af=0.03)
    diskpot= lp+dp
    ax,ay,az= -0.03,0.04,0.02
    inta= [lambda t: -0.03*t**2./2.,lambda t: 0.04*t**2./2.,lambda t: 0.02*t**2./2.]
    framepot= potential.NonInertialFrameForce(RTacm=[ax,ay,az])
    diskframepot= AcceleratingPotentialWrapperPotential(pot=diskpot,a=inta)\
                    +framepot
    def check_orbit(method='odeint',tol=1e-9):
        o= Orbit()
        o.turn_physical_off()
        # Inertial frame
        ts= numpy.linspace(0.,20.,1001)
        o.integrate(ts,diskpot,method=method)
        # Non-inertial frame
        op= o()
        op.integrate(ts,diskframepot,method=method)
        # Compare
        o_xs= o.x(ts)
        o_ys= o.y(ts)
        o_zs= o.z(ts)
        op_xs= op.x(ts)+inta[0](ts)
        op_ys= op.y(ts)+inta[1](ts)
        op_zs= op.z(ts)+inta[2](ts)
        assert numpy.amax(numpy.fabs(o_xs-op_xs)) < tol, 'Integrating an orbit in a linearly-accelerating frame with constant acceleration does not agree with the equivalent orbit in the intertial frame for method {}'.format(method)
        assert numpy.amax(numpy.fabs(o_ys-op_ys)) < tol, 'Integrating an orbit in a linearly-accelerating frame with constant acceleration does not agree with the equivalent orbit in the intertial frame for method {}'.format(method)
        assert numpy.amax(numpy.fabs(o_zs-op_zs)) < tol, 'Integrating an orbit in a linearly-accelerating frame with constant acceleration does not agree with the equivalent orbit in the intertial frame for method {}'.format(method)
    check_orbit(method='odeint',tol=1e-5)
    check_orbit(method='dop853',tol=1e-9)
    check_orbit(method='dop853_c',tol=1e-5) # Lower tol, because diff integrators for inertial and non-inertial, bc wrapper not implemented in C
    return None

# Utility wrapers
from galpy.potential.WrapperPotential import parentWrapperPotential
from galpy.potential.Potential import _evaluateRforces, _evaluatephiforces, _evaluatezforces
class AcceleratingPotentialWrapperPotential(parentWrapperPotential):
    def __init__(self,amp=1.,pot=None,
                 a=[lambda t: 0., lambda t: 0., lambda t: 0.],
                 ro=None,vo=None):
        # a = accelerated x
        # that is
        # x -> x + a(t)
        # so a isn't really a...
        self._a= a
        
    def _Rforce(self,R,z,phi=0.,t=0.):
         Fxyz= self._force_xyz(R,z,phi=phi,t=t)
         return numpy.cos(phi)*Fxyz[0]+numpy.sin(phi)*Fxyz[1]

    def _phiforce(self,R,z,phi=0.,t=0.):
        Fxyz= self._force_xyz(R,z,phi=phi,t=t)
        return R*(-numpy.sin(phi)*Fxyz[0] + numpy.cos(phi)*Fxyz[1])

    def _zforce(self,R,z,phi=0.,t=0.):
        return self._force_xyz(R,z,phi=phi,t=t)[2]

    def _force_xyz(self,R,z,phi=0.,t=0.):
        """Get the rectangular forces in the transformed frame"""
        x,y,_= coords.cyl_to_rect(R,phi,z)
        xp= x+self._a[0](t)
        yp= y+self._a[1](t)
        zp= z+self._a[2](t)
        Rp,phip,zp= coords.rect_to_cyl(xp,yp,zp)
        Rforcep= _evaluateRforces(self._pot,Rp,zp,phi=phip,t=t)
        phiforcep= _evaluatephiforces(self._pot,Rp,zp,phi=phip,t=t)
        zforcep= _evaluatezforces(self._pot,Rp,zp,phi=phip,t=t)
        xforcep= numpy.cos(phip)*Rforcep-numpy.sin(phip)*phiforcep/Rp
        yforcep= numpy.sin(phip)*Rforcep+numpy.cos(phip)*phiforcep/Rp
        return numpy.array([xforcep,yforcep,zforcep])
    