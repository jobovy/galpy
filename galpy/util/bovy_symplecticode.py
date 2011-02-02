#Symplectic ODE integrators
#Follows scipy.integrate.odeint inputs
def leapfrog(func,yo,t,args=(),rtol=None,atol=None):
    if rtol is None and atol is None:
        rtol= 1.49012e-8
        atol= 0.


def leapfrog_step(func,yo,to,dt,args=()):
    pass
