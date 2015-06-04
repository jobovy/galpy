# The DF of a gap in a tidal stream
import numpy


def impulse_deltav_plummer(v,y,b,w,GM,rs):
    """
    NAME:
       impulse_deltav_plummer
    PURPOSE:
       calculate the delta velocity to due an encounter with a Plummer sphere in the impulse approximation; allows for arbitrary velocity vectors, but y is input as the position along the stream
    INPUT:
       v - velocity of the stream (nstar,3)
       y - position along the stream (nstar)
       b - impact parameter
       w - velocity of the Plummer sphere (3)
       GM - mass of the Plummer sphere (in natural units)
       rs - size of the Plummer sphere
    OUTPUT:
       deltav (nstar,3)
    HISTORY:
       2015-04-30 - Written based on Erkal's expressions - Bovy (IAS)
    """
    if len(v.shape) == 1: v= numpy.reshape(v,(1,3))
    nv= v.shape[0]
    # Build the rotation matrices and their inverse
    rot= _rotation_vy(v)
    rotinv= _rotation_vy(v,inv=True)
    # Rotate the Plummer sphere's velocity to the stream frames
    tilew= numpy.sum(rot*numpy.tile(w,(nv,3,1)),axis=-1)
    # Use Denis' expressions
    wperp= numpy.sqrt(tilew[:,0]**2.+tilew[:,2]**2.)
    wpar= numpy.sqrt(numpy.sum(v**2.,axis=1))-tilew[:,1]
    wmag2= wpar**2.+wperp**2.
    wmag= numpy.sqrt(wmag2)
    out= numpy.empty_like(v)
    denom= wmag*((b**2.+rs**2.)*wmag2+wperp**2.*y**2.)
    out[:,0]= (b*wmag2*tilew[:,2]/wperp-y*wpar*tilew[:,0])/denom
    out[:,1]= -wperp**2.*y/denom
    out[:,2]= -(b*wmag2*tilew[:,0]/wperp+y*wpar*tilew[:,2])/denom
    # Rotate back to the original frame
    return 2.0*GM*numpy.sum(rotinv*numpy.tile(out.T,(3,1,1)).T,axis=-1)

def impulse_deltav_plummer_general(v,x,b,w,x0,v0,GM,rs):
    """
    NAME:
       impulse_deltav_plummer_general
    PURPOSE:
       calculate the delta velocity to due an encounter with a Plummer sphere in the impulse approximation; allows for arbitrary velocity vectors, and arbitrary position along the stream
    INPUT:
       v - velocity of the stream (nstar,3)
       x - position along the stream (nstar,3)
       b - impact parameter
       w - velocity of the Plummer sphere (3)
       x0 - point of closest approach
       v0 - velocity of point of closest approach
       GM - mass of the Plummer sphere (in natural units)
       rs - size of the Plummer sphere
    OUTPUT:
       deltav (nstar,3)
    HISTORY:
       2015-05-04 - Written based on above - SANDERS
    """
    if len(v.shape) == 1: v= numpy.reshape(v,(1,3))
    if len(x.shape) == 1: x= numpy.reshape(x,(1,3))
    b0 = numpy.cross(w,v0)
    b0 *= b/numpy.sqrt(numpy.sum(b0**2))
    b_ = b0+x-x0
    w = w-v
    wmag = numpy.sqrt(numpy.sum(w**2,axis=1))
    bdotw=numpy.sum(b_*w,axis=1)/wmag
    denom= wmag*(numpy.sum(b_**2,axis=1)+rs**2-bdotw**2)
    denom = 1./denom
    return -2.0*GM*((b_.T-bdotw*w.T/wmag)*denom).T

def _a_integrand(T,y,b,w,pot,compt):
    t = T/(1-T*T)
    X = b+w*t+y*numpy.array([0,1,0])
    r = numpy.sqrt(numpy.sum(X**2))
    return (1+T*T)/(1-T*T)**2*pot.forces(r)*X[compt]/r

from scipy.integrate import quad

def _deltav_integrate(y,b,w,pot):
    return numpy.array([quad(_a_integrand,-1.,1.,args=(y,b,w,pot,i))[0] for i in range(3)])

def impulse_deltav_general(v,y,b,w,pot):
    """
    NAME:
       impulse_deltav_general
    PURPOSE:
       calculate the delta velocity to due an encounter with a general spherical potential in the impulse approximation; allows for arbitrary velocity vectors, but y is input as the position along the stream
    INPUT:
       v - velocity of the stream (nstar,3)
       y - position along the stream (nstar)
       b - impact parameter
       w - velocity of the subhalo (3)
       pot - object that has method forces(r) = -d\Phi/dr where \Phi is the subhalo potential
    OUTPUT:
       deltav (nstar,3)
    HISTORY:
       2015-05-04 - SANDERS
    """
    if len(v.shape) == 1: v= numpy.reshape(v,(1,3))
    nv= v.shape[0]
    # Build the rotation matrices and their inverse
    rot= _rotation_vy(v)
    rotinv= _rotation_vy(v,inv=True)
    # Rotate the subhalo's velocity to the stream frames
    tilew= numpy.sum(rot*numpy.tile(w,(nv,3,1)),axis=-1)
    tilew[:,1]-=numpy.sqrt(numpy.sum(v**2.,axis=1))
    wmag = numpy.sqrt(w[0]**2+w[2]**2)
    b0 = b*numpy.array([-w[2]/wmag,0,w[0]/wmag])
    return numpy.array(map(lambda i:numpy.sum(i[2]
                       *_deltav_integrate(i[0],b0,i[1],pot).T,axis=-1)
                        ,zip(y,tilew,rotinv)))

def impulse_deltav_general_curvedstream(v,x,b,w,x0,v0,pot):
    """
    NAME:
       impulse_deltav_general_curvedstream
    PURPOSE:
       calculate the delta velocity to due an encounter with a general spherical potential in the impulse approximation; allows for arbitrary velocity vectors and arbitrary shaped streams
    INPUT:
       v - velocity of the stream (nstar,3)
       x - position along the stream (nstar,3)
       b - impact parameter
       w - velocity of the subhalo (3)
       x0 - position of closest approach (3)
       v0 - velocity of stream at closest approach (3)
       pot - object that has method forces(r) = -d\Phi/dr where \Phi is the subhalo potential
    OUTPUT:
       deltav (nstar,3)
    HISTORY:
       2015-05-04 - SANDERS
    """
    if len(v.shape) == 1: v= numpy.reshape(v,(1,3))
    if len(x.shape) == 1: x= numpy.reshape(x,(1,3))
    b0 = numpy.cross(w,v0)
    b0 *= b/numpy.sqrt(numpy.sum(b0**2))
    b_ = b0+x-x0
    return numpy.array(map(lambda i:_deltav_integrate(0.,i[1],i[0],pot)
                        ,zip(w-v,b_)))

def _rotation_vy(v,inv=False):
    return _rotate_to_arbitrary_vector(v,[0,1,0],inv)

def _rotate_to_arbitrary_vector(v,a,inv=False):
    """ Return a rotation matrix that rotates v to align with vector a
        i.e. R . v = |v|\hat{a} """
    normv= v/numpy.tile(numpy.sqrt(numpy.sum(v**2.,axis=1)),(3,1)).T
    rotaxis= numpy.cross(normv,a)
    rotaxis/= numpy.tile(numpy.sqrt(numpy.sum(rotaxis**2.,axis=1)),(3,1)).T
    crossmatrix= numpy.empty((len(v),3,3))
    crossmatrix[:,0,:]= numpy.cross(rotaxis,[1,0,0])
    crossmatrix[:,1,:]= numpy.cross(rotaxis,[0,1,0])
    crossmatrix[:,2,:]= numpy.cross(rotaxis,[0,0,1])
    costheta= normv[:,1]
    sintheta= numpy.sqrt(1.-costheta**2.)
    if inv: sgn= 1.
    else: sgn= -1.
    out= numpy.tile(costheta,(3,3,1)).T*numpy.tile(numpy.eye(3),(len(v),1,1))\
        +sgn*numpy.tile(sintheta,(3,3,1)).T*crossmatrix\
        +numpy.tile(1.-costheta,(3,3,1)).T\
        *(rotaxis[:,:,numpy.newaxis]*rotaxis[:,numpy.newaxis,:])
    return out
