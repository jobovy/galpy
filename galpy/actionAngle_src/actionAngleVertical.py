###############################################################################
#   actionAngle: a Python module to calculate  actions, angles, and frequencies
#
#      class: actionAngleVertical
#
#      methods:
#             __call__: returns (j)
#             actionsFreqs: returns (j,omega)
#             actionsFreqsAngles: returns (j,omega,a)
#             calcxmax
###############################################################################
import numpy as nu
from scipy import optimize, integrate
from galpy.actionAngle_src.actionAngle import actionAngle
from galpy.potential_src.linearPotential import evaluatelinearPotentials
class actionAngleVertical(actionAngle):
    """Action-angle formalism for one-dimensional potentials (or of the vertical potential in a galactic disk in the adiabatic approximation, hence the name)"""
    def __init__(self,*args,**kwargs):
        """
        NAME:

           __init__

        PURPOSE:

           initialize an actionAngleVertical object

        INPUT:

           pot= potential or list of potentials (planarPotentials)
           
           ro= distance from vantage point to GC (kpc; can be Quantity)

           vo= circular velocity at ro (km/s; can be Quantity)

        OUTPUT:
        
           instance

        HISTORY:

           2012-06-01 - Written - Bovy (IAS)
           2018-05-19 - Conformed to the general actionAngle framework - Bovy (UofT)

           Either:
              a) z,vz
              b) Orbit instance: initial condition used if that's it, orbit(t)
                 if there is a time given as well
        """
        actionAngle.__init__(self,
                             ro=kwargs.get('ro',None),vo=kwargs.get('vo',None))
        if not 'pot' in kwargs: #pragma: no cover
            raise IOError("Must specify pot= for actionAngleVertical")
        if not 'pot' in kwargs: #pragma: no cover
            raise IOError("Must specify pot= for actionAngleVertical")
        self._pot= kwargs['pot']
        return None
        """
        self._parse_eval_args(*args,_noOrbUnitsCheck=True,**kwargs)
        self._z= self._eval_z
        self._vz= self._eval_vz
        self._verticalpot= kwargs['pot']
        return None
        """

    def _evaluate(self,*args,**kwargs):
        """
        NAME:
           __call__ (_evaluate)
        PURPOSE:
           evaluate the action
        INPUT:
           Either:
              a) x,vx:
                 1) floats: phase-space value for single object (each can be a Quantity)
                 2) numpy.ndarray: [N] phase-space values for N objects (each can be a Quantity)
        OUTPUT:
           action
        HISTORY:
           2018-05-19 - Written based on re-write of existing code - Bovy (UofT)
        """
        if len(args) == 2: # x,vx
            x,vx= args
            if isinstance(x,float):
                x= nu.array([x])
                vx= nu.array([vx])
            J= nu.empty(len(x))
            for ii in range(len(x)):
                E= vx[ii]**2./2.\
                    +evaluatelinearPotentials(self._pot,x[ii],
                                              use_physical=False)
                xmax= self.calcxmax(x[ii],vx[ii],E)
                if xmax == -9999.99:
                    J[ii]= 9999.99
                else:
                    J[ii]= 2.*integrate.quad(\
                        lambda xi: nu.sqrt(2.*(E\
                              -evaluatelinearPotentials(self._pot,xi,
                                                        use_physical=False))),
                        0.,xmax)[0]/nu.pi
            return J
        else: # pragma: no cover
            raise ValueError('actionAngleVertical __call__ input not understood')

    def _actionsFreqs(self,*args,**kwargs):
        """
        NAME:
           actionsFreqs (_actionsFreqs)
        PURPOSE:
           evaluate the action and frequency
        INPUT:
           Either:
              a) x,vx:
                 1) floats: phase-space value for single object (each can be a Quantity)
                 2) numpy.ndarray: [N] phase-space values for N objects (each can be a Quantity)
        OUTPUT:
           action,frequency
        HISTORY:
           2018-05-19 - Written based on re-write of existing code - Bovy (UofT)
        """
        if len(args) == 2: # x,vx
            x,vx= args
            if isinstance(x,float):
                x= nu.array([x])
                vx= nu.array([vx])
            J= nu.empty(len(x))
            Omega= nu.empty(len(x))
            for ii in range(len(x)):
                E= vx[ii]**2./2.\
                    +evaluatelinearPotentials(self._pot,x[ii],
                                              use_physical=False)
                xmax= self.calcxmax(x[ii],vx[ii],E)
                if xmax == -9999.99:
                    J[ii]= 9999.99
                    Omega[ii]= 9999.99
                else:
                    J[ii]= 2.*integrate.quad(\
                        lambda xi: nu.sqrt(2.*(E\
                                       -evaluatelinearPotentials(self._pot,xi,
                                                         use_physical=False))),
                        0.,xmax,)[0]/nu.pi
                    # Transformed x = xmax-t^2 for singularity
                    Omega[ii]= nu.pi/2./integrate.quad(\
                        lambda t: 2.*t/nu.sqrt(2.*(E\
                                       -evaluatelinearPotentials(self._pot,
                                                                 xmax-t**2.,
                                                         use_physical=False))),
                        0,nu.sqrt(xmax))[0]
            return (J,Omega)
        else: # pragma: no cover
            raise ValueError('actionAngleVertical __call__ input not understood')

    def _actionsFreqsAngles(self,*args,**kwargs):
        """
        NAME:
           actionsFreqsAngles (_actionsFreqsAngles)
        PURPOSE:
           evaluate the action, frequency, and angle
        INPUT:
           Either:
              a) x,vx:
                 1) floats: phase-space value for single object (each can be a Quantity)
                 2) numpy.ndarray: [N] phase-space values for N objects (each can be a Quantity)
        OUTPUT:
           action,frequency,angle
        HISTORY:
           2018-05-19 - Written based on re-write of existing code - Bovy (UofT)
        """
        if len(args) == 2: # x,vx
            x,vx= args
            if isinstance(x,float):
                x= nu.array([x])
                vx= nu.array([vx])
            J= nu.empty(len(x))
            Omega= nu.empty(len(x))
            angle= nu.empty(len(x))
            for ii in range(len(x)):
                E= vx[ii]**2./2.\
                    +evaluatelinearPotentials(self._pot,x[ii],
                                              use_physical=False)
                xmax= self.calcxmax(x[ii],vx[ii],E)
                if xmax == -9999.99:
                    J[ii]= 9999.99
                    Omega[ii]= 9999.99
                    angle[ii]= 9999.99
                else:
                    J[ii]= 2.*integrate.quad(\
                        lambda xi: nu.sqrt(2.*(E\
                                       -evaluatelinearPotentials(self._pot,xi,
                                                         use_physical=False))),
                        0.,xmax)[0]/nu.pi
                    Omega[ii]= nu.pi/2./integrate.quad(\
                        lambda t: 2.*t/nu.sqrt(2.*(E\
                                       -evaluatelinearPotentials(self._pot,
                                                                 xmax-t**2.,
                                                         use_physical=False))),
                        0,nu.sqrt(xmax))[0]
                    angle[ii]= integrate.quad(\
                        lambda xi: 1./nu.sqrt(2.*(E\
                                       -evaluatelinearPotentials(self._pot,xi,
                                                         use_physical=False))),
                                        0,nu.fabs(x[ii]))[0]
            angle*= Omega
            angle[(x >= 0.)*(vx < 0.)]= nu.pi-angle[(x >= 0.)*(vx < 0.)]
            angle[(x < 0.)*(vx <= 0.)]= nu.pi+angle[(x < 0.)*(vx <= 0.)]
            angle[(x < 0.)*(vx > 0.)]= 2.*nu.pi-angle[(x < 0.)*(vx > 0.)]
            return (J,Omega,angle % (2.*nu.pi))
        else: # pragma: no cover
            raise ValueError('actionAngleVertical __call__ input not understood')

    def calcxmax(self,x,vx,E=None):
        """
        NAME:
           calcxmax
        PURPOSE:
           calculate the maximum height
        INPUT:
           x - position
           vx - velocity
        OUTPUT:
           zmax
        HISTORY:
           2012-06-01 - Written - Bovy (IAS)
           2018-05-19 - Re-written for new framework - Bovy (UofT)
        """
        if E is None:
            E= E= vx**2./2.\
                +evaluatelinearPotentials(self._pot,x,use_physical=False)
        if vx == 0.: #We are exactly at the maximum height
            xmax= nu.fabs(x)
        else:
            xstart= x
            try:
                if x == 0.: xend= 0.00001
                else: xend= 2.*nu.fabs(x)
                while (E-evaluatelinearPotentials(self._pot,xend,
                                                  use_physical=False)) > 0.:
                    xend*= 2.
                    if xend > 100.: #pragma: no cover
                        raise OverflowError
            except OverflowError: #pragma: no cover
                xmax= -9999.99
            else:
                xmax= optimize.brentq(\
                    lambda xm: E-evaluatelinearPotentials(self._pot,xm,
                                                          use_physical=False),
                    xstart,xend,xtol=1e-14)
                while (E-evaluatelinearPotentials(self._pot,xmax,
                                                          use_physical=False)) < 0:
                    xmax-= 1e-14
        return xmax
