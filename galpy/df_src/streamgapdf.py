# The DF of a gap in a tidal stream
import numpy
from galpy.df import streamdf
class streamgapdf(streamdf):
    """The DF of a tidal stream"""
    def __init__(self,*args,**kwargs):
        """
        NAME:

           __init__

        PURPOSE:

           Initialize the DF of a gap in a stellar stream

        INPUT:

           streamdf args and kwargs

           Subhalo and impact parameters:

              impactb= impact parameter

              subhalovel= velocity of the subhalo shape=(3)

              GM= mass of the subhalo (in natural units)

              rs= size parameter of the subhalo
             
           deltaAngleTrackImpact= (None) angle to estimate the stream track over to determine the effect of the impact [similar to deltaAngleTrack] (rad)

           nTrackChunksImpact= (floor(deltaAngleTrack/0.15)+1) number of chunks to divide the progenitor track in near the impact [similar to nTrackChunks]

        OUTPUT:

           object

        HISTORY:

           2015-06-02 - Started - Bovy (IAS)

        """
        # Parse kwargs
        impactb= kwargs.pop('impactb',1.)
        subhalovel= kwargs.pop('subhalovel',1.)
        GM= kwargs.pop('GM',1.)
        rs= kwargs.pop('rs',1.)
        deltaAngleTrackImpact= kwargs.pop('deltaAngleTrackImpact',None)
        nTrackChunksImpact= kwargs.pop('nTrackChunksImpact',None)      
        return None

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

def _rotation_vy(v,inv=False):
    """Return a rotation matrix that rotates v to point to (0,1,0)"""
    normv= v/numpy.tile(numpy.sqrt(numpy.sum(v**2.,axis=1)),(3,1)).T
    rotaxis= numpy.cross(normv,[0,1,0])
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
