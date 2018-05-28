###############################################################################
#   actionAngle: a Python module to calculate  actions, angles, and frequencies
#
#      class: actionAngleVerticalInverse
#
#             Calculate (x,v) coordinates for a one-dimensional potental
#             given actions-angle coordinates
#
###############################################################################
import copy
import numpy
import warnings
from scipy import interpolate,ndimage
from galpy.potential import evaluatelinearPotentials, \
    evaluatelinearForces
from galpy.util import bovy_plot, galpyWarning
from matplotlib import pyplot, gridspec
from matplotlib.ticker import NullFormatter
from galpy.actionAngle_src.actionAngleHarmonic import actionAngleHarmonic
from galpy.actionAngle_src.actionAngleHarmonicInverse import \
    actionAngleHarmonicInverse
from galpy.actionAngle_src.actionAngleVertical import actionAngleVertical
from galpy.actionAngle_src.actionAngleInverse import actionAngleInverse
class actionAngleVerticalInverse(actionAngleInverse):
    """Inverse action-angle formalism for one dimensional systems"""
    def __init__(self,pot=None,Es=[0.1,0.3],nta=128,setup_interp=False,
                 maxiter=100,angle_tol=1e-12,bisect=False):
        """
        NAME:

           __init__

        PURPOSE:

           initialize an actionAngleVerticalInverse object

        INPUT:

           pot= a linearPotential or list thereof

           Es= energies of the orbits to map the tori for, will be forcibly sorted (needs to be a dense grid when setting up the object for interpolation with setup_interp=True)

           nta= (128) number of auxiliary angles to sample the torus at when mapping the torus

           setup_interp= (False) if True, setup interpolation grids that allow any torus within the E range to be accessed through interpolation

           maxiter= (100) maximum number of iterations of root-finding algorithms

           angle_tol= (1e-12) tolerance for angle root-finding (f(x) is within tol of desired value)

           bisect= (False) if True, use simple bisection for root-finding, otherwise first try Newton-Raphson (mainly useful for testing the bisection fallback)
           
        OUTPUT:

           instance

        HISTORY:

           2018-04-11 - Started - Bovy (UofT)

        """
        #actionAngleInverse.__init__(self,*args,**kwargs)
        if pot is None: #pragma: no cover
            raise IOError("Must specify pot= for actionAngleVerticalInverse")
        self._pot= pot
        self._aAV= actionAngleVertical(pot=self._pot)
        # Compute action, frequency, and xmax for each energy
        self._nE= len(Es)
        js= numpy.empty(self._nE)
        Omegas= numpy.empty(self._nE)
        xmaxs= numpy.empty(self._nE)
        self._Es= numpy.sort(numpy.array(Es))
        for ii,E in enumerate(Es):
            tJ,tO= self._aAV.actionsFreqs(0.,\
                     numpy.sqrt(2.*(E-evaluatelinearPotentials(self._pot,0.))))
            js[ii]= tJ
            Omegas[ii]= tO
            xmaxs[ii]=\
               self._aAV.calcxmax(0.,numpy.sqrt(2.*(\
                             E-evaluatelinearPotentials(self._pot,0.))),
                           E=E)
        self._js= js
        self._Omegas= Omegas
        self._xmaxs= xmaxs
        # Set harmonic-oscillator frequencies == frequencies
        self._OmegaHO= Omegas
        # The following work properly for arrays of omega
        self._hoaa= actionAngleHarmonic(omega=self._OmegaHO)
        self._hoaainv= actionAngleHarmonicInverse(omega=self._OmegaHO)
        # Now map all tori
        self._nta= nta
        self._thetaa= numpy.linspace(0.,2.*numpy.pi*(1.-1./nta),nta)
        self._maxiter= maxiter
        self._angle_tol= angle_tol
        self._bisect= bisect
        self._xgrid= self._create_xgrid()
        self._ja= _ja(self._xgrid,self._Egrid,self._pot,self._omegagrid)
        self._djadj= _djadj(self._xgrid,self._Egrid,self._pot,self._omegagrid)
        # Store mean(ja) as probably a better approx. of j
        self._js_orig= copy.copy(self._js)
        self._js= numpy.mean(self._ja,axis=1)
        # Compute Fourier expansions
        self._nforSn= numpy.arange(self._ja.shape[1]//2+1)
        self._nSn= numpy.real(numpy.fft.rfft(self._ja
                                             -numpy.atleast_2d(self._js).T,
                                             axis=1))[:,1:]/self._ja.shape[1]
        self._dSndJ= numpy.real(numpy.fft.rfft(self._djadj-1.,axis=1))[:,1:]\
                          /self._ja.shape[1]
        # Interpolation of small, noisy coeffs doesn't work, so set to zero
        if setup_interp:
            self._nSn[numpy.fabs(self._nSn) < 1e-16]= 0.
            self._dSndJ[numpy.fabs(self._dSndJ) < 1e-15]= 0.
        self._dSndJ/= numpy.atleast_2d(self._nforSn)[:,1:]
        self._nforSn= self._nforSn[1:]
        self._js[self._Es < 1e-10]= 0.
        # Should use sqrt(2nd deriv. pot), but currently not implemented for 1D
        if self._nE > 1:
            self._OmegaHO[self._Es < 1e-10]= self._OmegaHO[1]
            self._Omegas[self._Es < 1e-10]= self._Omegas[1]
        self._nSn[self._Es < 1e-10]= 0.
        self._dSndJ[self._Es < 1e-10]= 0.
        # Setup interpolation if requested
        if setup_interp:
            self._interp= True
            self._setup_interp()
        else:
            self._interp= False
        return None

    def _create_xgrid(self):
        # Find x grid for regular grid in auxiliary angle (thetaa)
        # in practice only need to map 0 < thetaa < pi/2  to +x with +v bc symm
        # To efficiently start the search, we first compute thetaa for a dense
        # grid in x (at +v)
        xgrid= numpy.linspace(-1.,1.,2*self._nta)
        xs= xgrid*numpy.atleast_2d(self._xmaxs).T
        xta= _anglea(xs,numpy.tile(self._Es,(xs.shape[1],1)).T,
                     self._pot,numpy.tile(self._hoaa._omega,(xs.shape[1],1)).T)
        xta[numpy.isnan(xta)]= 0. # Zero energy orbit -> NaN
        # Now use Newton-Raphson to iterate to a regular grid
        cindx= numpy.nanargmin(numpy.fabs(\
                (xta-numpy.rollaxis(numpy.atleast_3d(self._thetaa),1)
                 +numpy.pi) % (2.*numpy.pi)-numpy.pi),axis=2)
        xgrid= xgrid[cindx].T*numpy.atleast_2d(self._xmaxs).T
        Egrid= numpy.tile(self._Es,(self._nta,1)).T
        omegagrid= numpy.tile(self._hoaa._omega,(self._nta,1)).T
        xmaxgrid= numpy.tile(self._xmaxs,(self._nta,1)).T
        ta= _anglea(xgrid,Egrid,self._pot,omegagrid)
        mta= numpy.tile(self._thetaa,(len(self._Es),1))
        # Now iterate
        cntr= 0
        unconv= numpy.ones(xgrid.shape,dtype='bool')
        # We'll fill in the -v part using the +v, also remove the endpoints
        unconv[:,self._nta//4:3*self._nta//4+1]= False
        dta= (ta[unconv]-mta[unconv]+numpy.pi) % (2.*numpy.pi)-numpy.pi
        unconv[unconv]= numpy.fabs(dta) > self._angle_tol
        # Don't allow too big steps
        maxdx= numpy.tile(self._xmaxs/float(self._nta),(self._nta,1)).T
        while not self._bisect:
            dtadx= _danglea(xgrid[unconv],Egrid[unconv],
                            self._pot,omegagrid[unconv])
            dta= (ta[unconv]-mta[unconv]+numpy.pi) % (2.*numpy.pi)-numpy.pi
            dx= -dta/dtadx
            dx[numpy.fabs(dx) > maxdx[unconv]]=\
                (numpy.sign(dx)*maxdx[unconv])[numpy.fabs(dx) > maxdx[unconv]]
            xgrid[unconv]+= dx
            xgrid[unconv*(xgrid > xmaxgrid)]=\
                xmaxgrid[unconv*(xgrid > xmaxgrid)]
            xgrid[unconv*(xgrid < -xmaxgrid)]=\
                xmaxgrid[unconv*(xgrid < -xmaxgrid)]
            unconv[unconv]= numpy.fabs(dta) > self._angle_tol
            newta= _anglea(xgrid[unconv],Egrid[unconv],
                           self._pot,omegagrid[unconv])
            ta[unconv]= newta
            cntr+= 1
            if numpy.sum(unconv) == 0:
                break
            if cntr > self._maxiter:
                warnings.warn(\
                    "Torus mapping with Newton-Raphson did not converge in {} iterations, falling back onto simple bisection (increase maxiter to try harder with Newton-Raphson)"\
                        .format(self._maxiter),galpyWarning)
                break
        if self._bisect or cntr > self._maxiter:
            # Reset cntr
            cntr= 0
            # Start from nearest guess from below
            new_xgrid= numpy.linspace(-1.,1.,2*self._nta)
            da=(xta-numpy.rollaxis(numpy.atleast_3d(self._thetaa),1)+numpy.pi)\
                % (2.*numpy.pi) - numpy.pi
            da[da >= 0.]= -numpy.nanmax(numpy.fabs(da))-0.1
            cindx= numpy.nanargmax(da,axis=2)
            tryx_min= (new_xgrid[cindx].T
                         *numpy.atleast_2d(self._xmaxs).T)[unconv]
            dx= 2./(2.*self._nta-1)*xmaxgrid # delta of initial x grid above
            while True:
                dx*= 0.5
                xgrid[unconv]= tryx_min+dx[unconv]
                newta= (_anglea(xgrid[unconv],Egrid[unconv],
                                self._pot,omegagrid[unconv])+2.*numpy.pi) \
                                % (2.*numpy.pi)
                dta= (newta-mta[unconv]+numpy.pi) % (2.*numpy.pi)-numpy.pi
                tryx_min[newta < mta[unconv]]=\
                    xgrid[unconv][newta < mta[unconv]]
                unconv[unconv]= numpy.fabs(dta) > self._angle_tol
                tryx_min= tryx_min[numpy.fabs(dta) > self._angle_tol]
                cntr+= 1
                if numpy.sum(unconv) == 0:
                    break
                if cntr > self._maxiter:
                    warnings.warn(\
                        "Torus mapping with bisection did not converge in {} iterations"\
                            .format(self._maxiter)
                        +" for energies:"+""\
                  .join(' {:g}'.format(k) for k in sorted(set(Egrid[unconv]))),
                    galpyWarning)
                    break
        xgrid[:,self._nta//4+1:self._nta//2+1]= xgrid[:,:self._nta//4][:,::-1]
        xgrid[:,self._nta//2+1:3*self._nta//4+1]=\
            xgrid[:,3*self._nta//4:][:,::-1]
        ta[:,self._nta//4+1:3*self._nta//4]= \
            _anglea(xgrid[:,self._nta//4+1:3*self._nta//4],
                    Egrid[:,self._nta//4+1:3*self._nta//4],
                    self._pot,
                    omegagrid[:,self._nta//4+1:3*self._nta//4],
                    vsign=-1.)
        self._dta= (ta-mta+numpy.pi) % (2.*numpy.pi)-numpy.pi
        self._mta= mta
        # Store these, they are useful (obv. arbitrary to return xgrid 
        # and not just store it...)
        self._Egrid= Egrid
        self._omegagrid= omegagrid
        self._xmaxgrid= xmaxgrid
        return xgrid

    def check_convergence(self,E,symm=True):
        # First find the torus for this energy
        indx= numpy.nanargmin(numpy.fabs(E-self._Es))
        if numpy.fabs(E-self._Es[indx]) > 1e-10:
            raise ValueError('Given energy not found; please specify an energy used in the initialization of the instance')
        gs= gridspec.GridSpec(2,3,height_ratios=[4,1])
        # mapping of thetaa --> x
        pyplot.subplot(gs[0])
        bovy_plot.bovy_plot(self._thetaa,self._xgrid[indx],
                            color='k',
                            ylabel=r'$x(\theta^A)$',
                            gcf=True)
        pyplot.gca().xaxis.set_major_formatter(NullFormatter())
        pyplot.subplot(gs[3])
        negv= (self._thetaa > numpy.pi/2.)*(self._thetaa < 3.*numpy.pi/2.)
        thetaa_out= numpy.empty_like(self._thetaa)
        thetaa_out[True^negv]= _anglea(self._xgrid[indx][True^negv],
                                       E,self._pot,
                                       self._OmegaHO[indx],vsign=1.)
        thetaa_out[negv]= _anglea(self._xgrid[indx][negv],
                                  E,self._pot,
                                  self._OmegaHO[indx],vsign=-1.)
        bovy_plot.bovy_plot(self._thetaa,
                            ((thetaa_out-self._thetaa+numpy.pi) \
                                 % (2.*numpy.pi))-numpy.pi,
                            color='k',
                            gcf=True,
                            xlabel=r'$\theta^A$',
                            ylabel=r'$\theta^A[x(\theta^A)]-\theta^A$')
        # Recovery of the nSn from J^A(theta^A) behavior
        pyplot.subplot(gs[1])
        bovy_plot.bovy_plot(self._thetaa,self._ja[indx],
                            color='k',
                            ylabel=r'$J^A(\theta^A),J$',gcf=True)
        pyplot.axhline(self._js[indx],color='k',ls='--')
        pyplot.gca().xaxis.set_major_formatter(NullFormatter())
        pyplot.subplot(gs[4])
        bovy_plot.bovy_plot(self._thetaa,
                            numpy.array([self._js[indx]
                                         +2.*numpy.sum(self._nSn[indx]\
                                                   *numpy.cos(self._nforSn*x)) 
                                         for x in self._thetaa])\
                                /self._ja[indx]-1.,
                            color='k',
                            xlabel=r'$\theta^A$',
                            ylabel=r'$\delta J^A/J^A$',gcf=True)
        # Recovery of the dSndJ from dJ^A/dJ(theta^A) behavior
        pyplot.subplot(gs[2])
        bovy_plot.bovy_plot(self._thetaa,self._djadj[indx],
                            color='k',
                            ylabel=r'$\mathrm{d}J^A/\mathrm{d}J(\theta^A)$',
                            gcf=True)
        pyplot.axhline(1.,color='k',ls='--')
        pyplot.gca().xaxis.set_major_formatter(NullFormatter())
        pyplot.subplot(gs[5])
        bovy_plot.bovy_plot(self._thetaa,
                            numpy.array([1.+2.*numpy.sum(self._nforSn\
                                                   *self._dSndJ[indx]\
                                                   *numpy.cos(self._nforSn*x)) 
                                         for x in self._thetaa])\
                                -self._djadj[indx],
                            color='k',
                            xlabel=r'$\theta^A$',
                            ylabel=r'$\delta \mathrm{d}J^A/\mathrm{d}J(\theta^A)$',
                            gcf=True)
        pyplot.tight_layout()
        return None

    ################### FUNCTIONS FOR INTERPOLATION BETWEEN TORI###############
    def _setup_interp(self):
        self._Emin= self._Es[0]
        self._Emax= self._Es[-1]
        self._nnSn= self._nSn.shape[1] # won't be confusing...
        self._nSnFiltered= ndimage.spline_filter(self._nSn,order=3)
        self._dSndJFiltered= ndimage.spline_filter(self._dSndJ,order=3)
        self.J= interpolate.InterpolatedUnivariateSpline(self._Es,self._js,k=3)
        self.E= interpolate.InterpolatedUnivariateSpline(self._js,self._Es,k=3)
        self.OmegaHO= interpolate.InterpolatedUnivariateSpline(self._Es,
                                                               self._OmegaHO,
                                                               k=3)
        self.Omega= interpolate.InterpolatedUnivariateSpline(self._Es,
                                                             self._Omegas,
                                                             k=3)
        return None

    def _coords_for_map_coords(self,E):
        coords= numpy.empty((2,self._nnSn*len(E)))
        coords[0]= numpy.tile((E-self._Emin)/(self._Emax-self._Emin)\
                                  *(self._nE-1.),
                              (self._nnSn,1)).T.flatten()
        coords[1]= numpy.tile(self._nforSn-1,(len(E),1)).flatten()
        return coords

    def nSn(self,E):
        if not self._interp:
            raise RuntimeError("To evaluate nSn, interpolation must be activated at instantiation using setup_interp=True")
        evalE= numpy.atleast_1d(E)
        indxc= (evalE >= self._Emin)*(evalE <= self._Emax)
        coords= self._coords_for_map_coords(evalE[indxc])
        out= numpy.empty((len(evalE),self._nnSn))
        out[indxc]= numpy.reshape(ndimage.interpolation.map_coordinates(\
                self._nSnFiltered,coords,order=3,prefilter=False),
                                  (numpy.sum(indxc),self._nnSn))
        out[True^indxc]= numpy.nan
        return out

    def dSndJ(self,E):
        if not self._interp:
            raise RuntimeError("To evaluate dnSndJ, interpolation must be activated at instantiation using setup_interp=True")
        evalE= numpy.atleast_1d(E)
        indxc= (evalE >= self._Emin)*(evalE <= self._Emax)
        coords= self._coords_for_map_coords(evalE[indxc])
        out= numpy.empty((len(evalE),coords.shape[1]//len(evalE)))
        out[indxc]= numpy.reshape(ndimage.interpolation.map_coordinates(\
                self._dSndJFiltered,coords,order=3,prefilter=False),
                                  (len(evalE),coords.shape[1]//len(evalE)))
        out[True^indxc]= numpy.nan
        return out

    def check_interp(self,E,symm=True):
        truthaAV= actionAngleVerticalInverse(pot=self._pot,Es=[E],
                                             nta=self._nta,setup_interp=False)
        # Check whether S_n is matched
        pyplot.subplot(2,3,1)
        y= numpy.fabs(self.nSn(E)[0,symm::symm+1])
        ymin= numpy.amax([numpy.amin(y[numpy.isfinite(y)]),1e-17])
        ymax= numpy.amax(y[numpy.isfinite(y)])
        bovy_plot.bovy_plot(numpy.fabs(self._nforSn[symm::symm+1]),
                            y,yrange=[ymin,ymax],
                            gcf=True,semilogy=True,
                            xrange=[0.,self._nforSn[-1]],
                            label=r'$\mathrm{Interpolation}$',
                            xlabel=r'$n$',ylabel=r'$|nS_n|$')
        bovy_plot.bovy_plot(self._nforSn[symm::symm+1],
                            truthaAV._nSn[0,symm::symm+1],overplot=True,
                            label=r'$\mathrm{Direct}$')
        pyplot.legend(fontsize=17.)
        pyplot.subplot(2,3,4)
        y= ((self.nSn(E)[0]-truthaAV._nSn[0])\
                                 /truthaAV._nSn[0])[symm::symm+1]
        ymin= numpy.amin(y[numpy.isfinite(y)])
        ymax= numpy.amax(y[numpy.isfinite(y)])
        bovy_plot.bovy_plot(self._nforSn[symm::symm+1],
                            y,yrange=[ymin,ymax],
                            xrange=[0.,self._nforSn[-1]],
                            gcf=True,
                            xlabel=r'$n$',
                            ylabel=r'$S_{n,\mathrm{interp}}/S_{n,\mathrm{direct}}-1$')
        # Check whether d S_n / d J is matched
        pyplot.subplot(2,3,2)
        y= numpy.fabs(self.dSndJ(E)[0,symm::symm+1])
        ymin= numpy.amax([numpy.amin(y[numpy.isfinite(y)]),1e-18])
        ymax= numpy.amax(y[numpy.isfinite(y)])
        bovy_plot.bovy_plot(numpy.fabs(self._nforSn[symm::symm+1]),
                            y,yrange=[ymin,ymax],
                            xrange=[0.,self._nforSn[-1]],
                            gcf=True,semilogy=True,
                            label=r'$\mathrm{Interpolation}$',
                            xlabel=r'$n$',
                            ylabel=r'$|\mathrm{d}S_n/\mathrm{d}J|$')
        bovy_plot.bovy_plot(self._nforSn[symm::symm+1],
                            truthaAV._dSndJ[0,symm::symm+1],overplot=True,
                            label=r'$\mathrm{Direct}$')
        pyplot.legend(fontsize=17.)
        pyplot.subplot(2,3,5)
        y= ((self.dSndJ(E)[0]-truthaAV._dSndJ[0])\
                                 /truthaAV._dSndJ[0])[symm::symm+1]
        ymin= numpy.amin(y[numpy.isfinite(y)])
        ymax= numpy.amax(y[numpy.isfinite(y)])
        bovy_plot.bovy_plot(self._nforSn[symm::symm+1],
                            y,yrange=[ymin,ymax],
                            xrange=[0.,self._nforSn[-1]],
                            gcf=True,
                            xlabel=r'$n$',
                            ylabel=r'$(\mathrm{d}S_n/\mathrm{d}J)_{\mathrm{interp}}/(\mathrm{d}S_n/\mathrm{d}J)_{\mathrm{direct}}-1$')
        # Check energy along the torus
        pyplot.subplot(2,3,3)
        ta= numpy.linspace(0.,2.*numpy.pi,1001)
        x,v= truthaAV(truthaAV._js,ta)
        Edirect= v**2./2.+evaluatelinearPotentials(self._pot,x)
        x,v= self(self.J(E),ta)
        Einterp= v**2./2.+evaluatelinearPotentials(self._pot,x)
        ymin, ymax= numpy.amin([Edirect,Einterp]),numpy.amax([Edirect,Einterp])
        
        bovy_plot.bovy_plot(ta,Einterp,
                            xrange=[0.,2.*numpy.pi],
                            yrange=[ymin-(ymax-ymin)*2.,ymax+(ymax-ymin)*2.],
                            gcf=True,
                            label=r'$\mathrm{Interpolation}$',
                            xlabel=r'$\theta$',
                            ylabel=r'$E$')
        bovy_plot.bovy_plot(ta,Edirect,overplot=True,
                            label=r'$\mathrm{Direct}$')
        pyplot.legend(fontsize=17.)
        pyplot.subplot(2,3,6)
        bovy_plot.bovy_plot(ta,Einterp/Edirect-1.,
                            xrange=[0.,2.*numpy.pi],
                            gcf=True,
                            label=r'$\mathrm{Interpolation}$',
                            xlabel=r'$\theta$',
                            ylabel=r'$E_{\mathrm{interp}}/E_{\mathrm{direct}}-1$')
        pyplot.tight_layout()
        return None

    def _evaluate(self,j,angle,**kwargs):
        """
        NAME:

           __call__

        PURPOSE:

           evaluate the phase-space coordinates (x,v) for a number of angles on a single torus

        INPUT:

           j - action (scalar)

           angle - angle (array [N])

        OUTPUT:

           [x,vx]

        HISTORY:

           2018-04-08 - Written - Bovy (UofT)

        """
        return self._xvFreqs(j,angle,**kwargs)[:2]
        
    def _xvFreqs(self,j,angle,**kwargs):
        """
        NAME:

           xvFreqs

        PURPOSE:

           evaluate the phase-space coordinates (x,v) for a number of angles on a single torus as well as the frequency

        INPUT:

           j - action (scalar)

           angle - angle (array [N])

        OUTPUT:

           ([x,vx],Omega)

        HISTORY:

           2018-04-15 - Written - Bovy (UofT)

        """
        # Find torus
        if not self._interp:
            indx= numpy.nanargmin(numpy.fabs(j-self._js))
            if numpy.fabs(j-self._js[indx]) > 1e-10:
                raise ValueError('Given action/energy not found, to use interpolation, initialize with setup_interp=True')
            tnSn= self._nSn[indx]
            tdSndJ= self._dSndJ[indx]
            tOmegaHO= self._OmegaHO[indx]
            tOmega= self._Omegas[indx]
        else:
            tE= self.E(j)
            tnSn= self.nSn(tE)[0]
            tdSndJ= self.dSndJ(tE)[0]
            tOmegaHO= self.OmegaHO(tE)
            tOmega= self.Omega(tE)
        # First we need to solve for anglea
        angle= numpy.atleast_1d(angle)
        anglea= copy.copy(angle)
        # Now iterate Newton's method
        cntr= 0
        unconv= numpy.ones(len(angle),dtype='bool')
        ta= anglea\
            +2.*numpy.sum(tdSndJ
                  *numpy.sin(self._nforSn*numpy.atleast_2d(anglea).T),axis=1)
        dta= (ta-angle+numpy.pi) % (2.*numpy.pi)-numpy.pi
        unconv[unconv]= numpy.fabs(dta) > self._angle_tol
        # Don't allow too big steps
        maxda= 2.*numpy.pi/101
        while not self._bisect:
            danglea= 1.+2.*numpy.sum(\
                self._nforSn*tdSndJ
                *numpy.cos(self._nforSn*numpy.atleast_2d(anglea[unconv]).T),
                axis=1)
            dta= (ta[unconv]-angle[unconv]+numpy.pi) % (2.*numpy.pi)-numpy.pi
            da= -dta/danglea
            da[numpy.fabs(da) > maxda]= \
                (numpy.sign(da)*maxda)[numpy.fabs(da) > maxda]
            anglea[unconv]+= da
            unconv[unconv]= numpy.fabs(dta) > self._angle_tol
            newta= anglea[unconv]\
                +2.*numpy.sum(tdSndJ
                   *numpy.sin(self._nforSn*numpy.atleast_2d(anglea[unconv]).T),
                              axis=1)
            ta[unconv]= newta
            cntr+= 1
            if numpy.sum(unconv) == 0:
                break
            if cntr > self._maxiter:
                warnings.warn(\
                    "Angle mapping with Newton-Raphson did not converge in {} iterations, falling back onto simple bisection (increase maxiter to try harder with Newton-Raphson)"\
                        .format(self._maxiter),galpyWarning)
                break
        # Fallback onto simple bisection in case of non-convergence
        if self._bisect or cntr > self._maxiter:
            # Reset cntr
            cntr= 0
            trya_min= numpy.zeros(numpy.sum(unconv))
            da= 2.*numpy.pi
            while True:
                da*= 0.5
                anglea[unconv]= trya_min+da
                newta= (anglea[unconv]\
                    +2.*numpy.sum(tdSndJ
                   *numpy.sin(self._nforSn*numpy.atleast_2d(anglea[unconv]).T),
                              axis=1)+2.*numpy.pi) % (2.*numpy.pi)
                dta= (newta-angle[unconv]+numpy.pi) % (2.*numpy.pi)-numpy.pi
                trya_min[newta < angle[unconv]]=\
                    anglea[unconv][newta < angle[unconv]]
                unconv[unconv]= numpy.fabs(dta) > self._angle_tol
                trya_min= trya_min[numpy.fabs(dta) > self._angle_tol]
                cntr+= 1
                if numpy.sum(unconv) == 0:
                    break
                if cntr > self._maxiter:
                    warnings.warn(\
                        "Angle mapping with bisection did not converge in {} iterations"\
                            .format(self._maxiter)
                        +" for angles:"+""\
                  .join(' {:g}'.format(k) for k in sorted(set(angle[unconv]))),
                    galpyWarning)
                    break
        # Then compute the auxiliary action
        ja= j+2.*numpy.sum(tnSn
                           *numpy.cos(self._nforSn*numpy.atleast_2d(anglea).T),
                           axis=1)
        hoaainv= actionAngleHarmonicInverse(omega=tOmegaHO)
        return (*hoaainv(ja,anglea),tOmega)
        
    def _Freqs(self,j,**kwargs):
        """
        NAME:

           Freqs

        PURPOSE:

           return the frequency corresponding to a torus

        INPUT:

           j - action (scalar)

        OUTPUT:

           (Omega)

        HISTORY:

           2018-04-08 - Written - Bovy (UofT)

        """
        # Find torus
        if not self._interp:
            indx= numpy.nanargmin(numpy.fabs(j-self._js))
            if numpy.fabs(j-self._js[indx]) > 1e-10:
                raise ValueError('Given action/energy not found, to use interpolation, initialize with setup_interp=True')
            tOmega= self._Omegas[indx]
        else:
            tE= self.E(j)
            tOmega= self.Omega(tE)
        return tOmega

def _anglea(x,E,pot,omega,vsign=1.):
    """
    NAME:
       _anglea
    PURPOSE:
       Compute the auxiliary angle in the harmonic-oscillator for a grid in x and E
    INPUT:
       x - position
       E - Energy
       pot - the potential
       omega - harmonic-oscillator frequencies
    OUTPUT:
       auxiliary angles
    HISTORY:
       2018-04-13 - Written - Bovy (UofT)
    """
    # Compute v
    v2= 2.*(E-evaluatelinearPotentials(pot,x))
    v2[v2 < 0.]= 0.
    return numpy.arctan2(omega*x,vsign*numpy.sqrt(v2))

def _danglea(x,E,pot,omega,vsign=1.):
    """
    NAME:
       _danglea
    PURPOSE:
       Compute the derivative of the auxiliary angle in the harmonic-oscillator for a grid in x and E at constant E
    INPUT:
       x - position
       E - Energy
       pot - the potential
       omega - harmonic-oscillator frequencies
    OUTPUT:
       d auxiliary angles / d x (2D array)
    HISTORY:
       2018-04-13 - Written - Bovy (UofT)
    """
    # Compute v
    v2= 2.*(E-evaluatelinearPotentials(pot,x))
    v2[v2 < 1e-10]= 2.*(E[v2<1e-10]
                        -evaluatelinearPotentials(pot,x[v2<1e-10]*(1.-1e-10)))
    anglea= numpy.arctan2(omega*x,vsign*numpy.sqrt(v2))
    return omega*numpy.cos(anglea)**2.*v2**-1.5\
        *(v2-x*evaluatelinearForces(pot,x))

def _ja(x,E,pot,omega,vsign=1.):
    """
    NAME:
       _ja
    PURPOSE:
       Compute the auxiliary action in the harmonic-oscillator for a grid in x and E
    INPUT:
       x - position
       E - Energy
       pot - the potential
       omega - harmonic-oscillator frequencies
    OUTPUT:
       auxiliary actions
    HISTORY:
       2018-04-14 - Written - Bovy (UofT)
    """
    return (E-evaluatelinearPotentials(pot,x))/omega+omega*x**2./2.

def _djadj(x,E,pot,omega,vsign=1.):
    """
    NAME:
       _djaj
    PURPOSE:
       Compute the derivative of the auxiliary action in the harmonic-oscillator wrt the action for a grid in x and E
    INPUT:
       x - position
       E - Energy
       pot - the potential
       omega - harmonic-oscillator frequencies
    OUTPUT:
       d(auxiliary actions)/d(action)
    HISTORY:
       2018-04-14 - Written - Bovy (UofT)
    """
    return 1.+(evaluatelinearForces(pot,x)+omega**2.*x)*x/(2.*(E-evaluatelinearPotentials(pot,x))-x*evaluatelinearForces(pot,x))

