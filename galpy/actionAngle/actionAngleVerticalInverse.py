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
from numpy.polynomial import polynomial, chebyshev
from scipy import interpolate,ndimage, optimize
from galpy.potential import evaluatelinearPotentials, \
    evaluatelinearForces
from galpy.util import bovy_plot, galpyWarning
from matplotlib import pyplot, gridspec, cm
from matplotlib.ticker import NullFormatter
from .actionAngleHarmonic import actionAngleHarmonic
from .actionAngleHarmonicInverse import actionAngleHarmonicInverse
from .actionAngleVertical import actionAngleVertical
from .actionAngleInverse import actionAngleInverse
class actionAngleVerticalInverse(actionAngleInverse):
    """Inverse action-angle formalism for one dimensional systems"""
    def __init__(self,pot=None,Es=[0.1,0.3],nta=128,setup_interp=False,
                 use_pointtransform=False,pt_deg=7,pt_nxa=301,
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
        #print("FIXING OMEGA")
        #Omegas= numpy.array([3.37426323, 2.34042556,  0.86533923])
        self._Omegas= Omegas
        self._xmaxs= xmaxs
        # Set harmonic-oscillator frequencies == frequencies
        self._OmegaHO= Omegas
        # The following work properly for arrays of omega
        self._hoaa= actionAngleHarmonic(omega=self._OmegaHO)
        self._hoaainv= actionAngleHarmonicInverse(omega=self._OmegaHO)
        if use_pointtransform and pt_deg > 1:
            self._setup_pointtransform(pt_deg-(1-pt_deg%2),pt_nxa) # make odd
            #print("FIXING PT TRANSFORM")
            """
            self._pt_coeffs[1]= numpy.array([  0.00000000e+00,   9.37029560e-01,   0.00000000e+00,
                                               8.11773692e-02,   0.00000000e+00,  -1.83556394e-02,
                                               0.00000000e+00,   1.48710173e-04])
            self._pt_deriv_coeffs[1]= numpy.array([ 0.93702956,  0.        ,  0.24353211,  0.        , -0.0917782 ,
        0.        ,  0.00104097])
            self._pt_deriv2_coeffs[1]= numpy.array([ 0.        ,  0.48706422,  0.        , -0.36711279,  0.        ,
                    0.00624583])
            #self._pt_xmaxs= numpy.array([ 0.13092613,0.55076953,4.29009437])

            tmp= numpy.tile(numpy.array([ 0.13040087,  0.53628544,  4.12768566]),(nta,1)).T
            tmp2= numpy.tile(numpy.array([ 0.13092613,0.55076953,4.29009437]),(nta,1)).T
            #self._xmaxs= numpy.array([ 0.13092613,0.55076953,4.29009437])
            """
        else:
            # Setup identity point transformation
            self._pt_xmaxs= self._xmaxs
            self._pt_coeffs= numpy.zeros((self._nE,2))
            self._pt_coeffs[:,1]= 1.
            self._pt_deriv_coeffs= numpy.ones((self._nE,1))
            self._pt_deriv2_coeffs= numpy.zeros((self._nE,1))
        # Now map all tori
        self._nta= nta
        self._thetaa= numpy.linspace(0.,2.*numpy.pi*(1.-1./nta),nta)
        self._maxiter= maxiter
        self._angle_tol= angle_tol
        self._bisect= bisect
        self._xgrid= self._create_xgrid()
        self._ja= _ja(self._xgrid,self._Egrid,self._pot,self._omegagrid,
                      self._ptcoeffsgrid,self._ptderivcoeffsgrid,
                      self._xmaxgrid,self._ptxmaxgrid)
        self._djadj= _djadj(self._xgrid,self._Egrid,self._pot,self._omegagrid,
                            self._ptcoeffsgrid,self._ptderivcoeffsgrid,
                            self._ptderiv2coeffsgrid,
                            self._xmaxgrid,self._ptxmaxgrid)
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

    def _setup_pointtransform(self,pt_deg,pt_nxa):
        # Setup a point transformation for each torus
        xamesh= numpy.linspace(-1.,1.,pt_nxa)
        self._pt_coeffs= numpy.empty((self._nE,pt_deg+1))
        self._pt_deriv_coeffs= numpy.empty((self._nE,pt_deg))
        self._pt_deriv2_coeffs= numpy.empty((self._nE,pt_deg-1))
        self._pt_xmaxs= numpy.sqrt(2.*self._js/self._OmegaHO)
        for ii in range(self._nE):
            Ea= self._js[ii]*self._OmegaHO[ii]
            # Function to optimize with least squares: p-p
            def opt_func(coeffs):
                # constraints: symmetric, maps [-1,1] --> [-1,1]
                ccoeffs= numpy.zeros(pt_deg+1)
                ccoeffs[1]= 1.
                ccoeffs[3::2]= coeffs
                ccoeffs/= chebyshev.chebval(1,ccoeffs)
                pt= chebyshev.Chebyshev(ccoeffs)
                xmesh= pt(xamesh)*self._xmaxs[ii]
                # Compute v from (E,xmesh)
                v2mesh= 2.*(self._Es[ii]\
                                -evaluatelinearPotentials(self._pot,
                                                          xmesh))
                v2mesh[v2mesh < 0.]= 0.
                vmesh= numpy.sqrt(v2mesh)
                va2mesh= 2.*(Ea-self._OmegaHO[ii]**2.\
                                 *(xamesh*self._pt_xmaxs[ii])**2./2.)
                va2mesh[va2mesh < 0.]= 0.
                vamesh= numpy.sqrt(va2mesh)
                vtildemesh= vamesh/pt.deriv()(xamesh)/self._xmaxs[ii]*self._pt_xmaxs[ii]
                return vmesh-vtildemesh
            if ii == 0:
                # Start from identity mapping
                start_coeffs= [0.]
                start_coeffs.extend([0. for jj in range((pt_deg+1)//2-2)])
            else:
                # Start from previous best fit
                start_coeffs= coeffs[3::2]/coeffs[1]
            coeffs= optimize.leastsq(opt_func,start_coeffs)[0]
            # Extract full Chebyshev parameters from constrained optimization
            ccoeffs= numpy.zeros(pt_deg+1)
            ccoeffs[1]= 1.
            ccoeffs[3::2]= coeffs
            ccoeffs/= chebyshev.chebval(1,ccoeffs)# map exact [-1,1] --> [-1,1]
            coeffs= ccoeffs
            # Store point transformation as simple polynomial
            self._pt_coeffs[ii]= chebyshev.cheb2poly(coeffs)
            self._pt_deriv_coeffs[ii]= polynomial.polyder(self._pt_coeffs[ii],
                                                          m=1)
            self._pt_deriv2_coeffs[ii]= polynomial.polyder(self._pt_coeffs[ii],
                                                           m=2)

            """
            from matplotlib import pyplot
            pt= chebyshev.Chebyshev(coeffs)
            print("ROOTS")
            print(chebyshev.chebroots(chebyshev.chebder(coeffs)))
            xmesh= pt(xamesh)*self._xmaxs[ii]
            # Compute v from (E,xmesh)
            v2mesh= 2.*(self._Es[ii]\
                            -evaluatelinearPotentials(self._pot,
                                                      xmesh))
            v2mesh[v2mesh < 0.]= 0.
            vmesh= numpy.sqrt(v2mesh)
            va2mesh= 2.*(Ea-self._OmegaHO[ii]**2.\
                             *(xamesh*self._pt_xmaxs[ii])**2./2.)
            va2mesh[va2mesh < 0.]= 0.
            vamesh= numpy.sqrt(va2mesh)
            vtildemesh= vamesh/pt.deriv()(xamesh)/self._xmaxs[ii]*self._pt_xmaxs[ii]
            pyplot.plot(xamesh,vmesh)
            pyplot.plot(xamesh,vamesh)
            pyplot.plot(xamesh,vtildemesh)
            """
        return None

    def _create_xgrid(self):
        # Find x grid for regular grid in auxiliary angle (thetaa)
        # in practice only need to map 0 < thetaa < pi/2  to +x with +v bc symm
        # To efficiently start the search, we first compute thetaa for a dense
        # grid in x (at +v)
        xgrid= numpy.linspace(-1.,1.,2*self._nta)
        xs= xgrid*numpy.atleast_2d(self._pt_xmaxs).T
        xta= _anglea(xs,numpy.tile(self._Es,(xs.shape[1],1)).T,
                     self._pot,numpy.tile(self._hoaa._omega,(xs.shape[1],1)).T,
                     numpy.rollaxis(numpy.tile(self._pt_coeffs,
                                               (xs.shape[1],1,1)),1),
                     numpy.rollaxis(numpy.tile(self._pt_deriv_coeffs,
                                               (xs.shape[1],1,1)),1),
                     numpy.tile(self._xmaxs,(xs.shape[1],1)).T,
                     numpy.tile(self._pt_xmaxs,(xs.shape[1],1)).T)
        xta[numpy.isnan(xta)]= 0. # Zero energy orbit -> NaN
        # Now use Newton-Raphson to iterate to a regular grid
        cindx= numpy.nanargmin(numpy.fabs(\
                (xta-numpy.rollaxis(numpy.atleast_3d(self._thetaa),1)
                 +numpy.pi) % (2.*numpy.pi)-numpy.pi),axis=2)
        xgrid= xgrid[cindx].T*numpy.atleast_2d(self._pt_xmaxs).T
        Egrid= numpy.tile(self._Es,(self._nta,1)).T
        omegagrid= numpy.tile(self._hoaa._omega,(self._nta,1)).T
        xmaxgrid= numpy.tile(self._xmaxs,(self._nta,1)).T
        ptxmaxgrid= numpy.tile(self._pt_xmaxs,(self._nta,1)).T
        ptcoeffsgrid= numpy.rollaxis(numpy.tile(self._pt_coeffs,
                                                (self._nta,1,1)),1)
        ptderivcoeffsgrid= numpy.rollaxis(numpy.tile(self._pt_deriv_coeffs,
                                                     (self._nta,1,1)),1)
        ptderiv2coeffsgrid= numpy.rollaxis(numpy.tile(self._pt_deriv2_coeffs,
                                                      (self._nta,1,1)),1)
        ta= _anglea(xgrid,Egrid,self._pot,omegagrid,ptcoeffsgrid,
                    ptderivcoeffsgrid,xmaxgrid,ptxmaxgrid)
        mta= numpy.tile(self._thetaa,(len(self._Es),1))
        # Now iterate
        cntr= 0
        unconv= numpy.ones(xgrid.shape,dtype='bool')
        # We'll fill in the -v part using the +v, also remove the endpoints
        unconv[:,self._nta//4:3*self._nta//4+1]= False
        dta= (ta[unconv]-mta[unconv]+numpy.pi) % (2.*numpy.pi)-numpy.pi
        unconv[unconv]= numpy.fabs(dta) > self._angle_tol
        # Don't allow too big steps
        maxdx= numpy.tile(self._pt_xmaxs/float(self._nta),(self._nta,1)).T
        while not self._bisect:
            dtadx= _danglea(xgrid[unconv],Egrid[unconv],
                            self._pot,omegagrid[unconv],
                            ptcoeffsgrid[unconv],
                            ptderivcoeffsgrid[unconv],
                            ptderiv2coeffsgrid[unconv],
                            xmaxgrid[unconv],ptxmaxgrid[unconv])
            dta= (ta[unconv]-mta[unconv]+numpy.pi) % (2.*numpy.pi)-numpy.pi
            dx= -dta/dtadx
            dx[numpy.fabs(dx) > maxdx[unconv]]=\
                (numpy.sign(dx)*maxdx[unconv])[numpy.fabs(dx) > maxdx[unconv]]
            xgrid[unconv]+= dx
            xgrid[unconv*(xgrid > ptxmaxgrid)]=\
                ptxmaxgrid[unconv*(xgrid > ptxmaxgrid)]
            xgrid[unconv*(xgrid < -ptxmaxgrid)]=\
                ptxmaxgrid[unconv*(xgrid < -ptxmaxgrid)]
            unconv[unconv]= numpy.fabs(dta) > self._angle_tol
            newta= _anglea(xgrid[unconv],Egrid[unconv],
                           self._pot,omegagrid[unconv],
                           ptcoeffsgrid[unconv],
                           ptderivcoeffsgrid[unconv],
                           xmaxgrid[unconv],ptxmaxgrid[unconv])
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
                         *numpy.atleast_2d(self._pt_xmaxs).T)[unconv]
            dx= 2./(2.*self._nta-1)*ptxmaxgrid # delta of initial x grid above
            while True:
                dx*= 0.5
                xgrid[unconv]= tryx_min+dx[unconv]
                newta= (_anglea(xgrid[unconv],Egrid[unconv],
                                self._pot,omegagrid[unconv],
                                ptcoeffsgrid[unconv],
                                ptderivcoeffsgrid[unconv],
                                xmaxgrid[unconv],ptxmaxgrid[unconv])\
                            +2.*numpy.pi) \
                            % (2.*numpy.pi)
                ta[unconv]= newta
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
                    ptcoeffsgrid[:,self._nta//4+1:3*self._nta//4],
                    ptderivcoeffsgrid[:,self._nta//4+1:3*self._nta//4],
                    xmaxgrid[:,self._nta//4+1:3*self._nta//4],
                    ptxmaxgrid[:,self._nta//4+1:3*self._nta//4],
                    vsign=-1.)
        self._dta= (ta-mta+numpy.pi) % (2.*numpy.pi)-numpy.pi
        self._mta= mta
        # Store these, they are useful (obv. arbitrary to return xgrid 
        # and not just store it...)
        self._Egrid= Egrid
        self._omegagrid= omegagrid
        self._ptcoeffsgrid= ptcoeffsgrid
        self._ptderivcoeffsgrid= ptderivcoeffsgrid
        self._ptderiv2coeffsgrid= ptderiv2coeffsgrid
        self._ptxmaxgrid= ptxmaxgrid
        self._xmaxgrid= xmaxgrid
        return xgrid

    def plot_convergence(self,E):
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
                                       self._OmegaHO[indx],
                                       self._pt_coeffs[indx],
                                       self._pt_deriv_coeffs[indx],
                                       self._xmaxs[indx],
                                       self._pt_xmaxs[indx],
                                       vsign=1.)
        thetaa_out[negv]= _anglea(self._xgrid[indx][negv],
                                  E,self._pot,
                                  self._OmegaHO[indx],
                                  self._pt_coeffs[indx],
                                  self._pt_deriv_coeffs[indx],
                                  self._xmaxs[indx],
                                  self._pt_xmaxs[indx],
                                  vsign=-1.)
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

    def plot_power(self,Es,symm=True):
        Es= numpy.sort(numpy.atleast_1d(Es))
        minn_for_cmap= 4
        if len(Es) < minn_for_cmap:
            gs= gridspec.GridSpec(1,2)
        else:
            outer= gridspec.GridSpec(1,2,width_ratios=[2.,0.05],wspace=0.05)
            gs= gridspec.GridSpecFromSubplotSpec(1,2,subplot_spec=outer[0],
                                                 wspace=0.35)
        overplot= False
        for E in Es:
            # First find the torus for this energy
            indx= numpy.nanargmin(numpy.fabs(E-self._Es))
            if numpy.fabs(E-self._Es[indx]) > 1e-10:
                raise ValueError('Given energy not found; please specify an energy used in the initialization of the instance')
            # n S_n
            y= numpy.fabs(self._nSn[indx,symm::symm+1])
            if len(Es) > 1 and E == Es[0]:
                y4minmax= numpy.fabs(self._nSn[:,symm::symm+1])
                ymin= numpy.amax([numpy.amin(y4minmax[numpy.isfinite(y4minmax)]),
                                  1e-17])
                ymax= numpy.amax(y4minmax[numpy.isfinite(y4minmax)])
            elif len(Es) == 1:
                ymin= numpy.amax([numpy.amin(y[numpy.isfinite(y)]),1e-17])
                ymax= numpy.amax(y[numpy.isfinite(y)])
            if len(Es) < minn_for_cmap:
                label= r'$E = {:g}$'.format(E)
                color= None
            else:
                label= None
                color= cm.plasma((E-Es[0])/(Es[-1]-Es[0]))
            pyplot.subplot(gs[0])
            bovy_plot.bovy_plot(numpy.fabs(self._nforSn[symm::symm+1]),
                                y,yrange=[ymin,ymax],
                                gcf=True,semilogy=True,overplot=overplot,
                                xrange=[0.,self._nforSn[-1]],
                                label=label,color=color,
                                xlabel=r'$n$',ylabel=r'$|nS_n|$')
            # d S_n / d J
            y= numpy.fabs(self._dSndJ[indx,symm::symm+1])
            if len(Es) > 1 and E == Es[0]:
                y4minmax= numpy.fabs(self._dSndJ[:,symm::symm+1])
                ymin= numpy.amax([numpy.amin(y4minmax[numpy.isfinite(y4minmax)]),
                                  1e-17])
                ymax= numpy.amax(y4minmax[numpy.isfinite(y4minmax)])
            elif len(Es) == 1:
                ymin= numpy.amax([numpy.amin(y[numpy.isfinite(y)]),1e-17])
                ymax= numpy.amax(y[numpy.isfinite(y)])
            if len(Es) < minn_for_cmap:
                label= r'$E = {:g}$'.format(E)
                color= None
            else:
                label= None
                color= cm.plasma((E-Es[0])/(Es[-1]-Es[0]))
            pyplot.subplot(gs[1])
            bovy_plot.bovy_plot(numpy.fabs(self._nforSn[symm::symm+1]),
                                y,yrange=[ymin,ymax],
                                gcf=True,semilogy=True,overplot=overplot,
                                xrange=[0.,self._nforSn[-1]],
                                label=label,color=color,
                                xlabel=r'$n$',
                                ylabel=r'$|\mathrm{d}S_n/\mathrm{d}J|$')
            overplot= True
        if len(Es) < minn_for_cmap:
            pyplot.subplot(gs[0])
            pyplot.legend(fontsize=17.)
            pyplot.subplot(gs[1])
            pyplot.legend(fontsize=17.)
            pyplot.tight_layout()
        else:
            pyplot.subplot(outer[1])
            sm= pyplot.cm.ScalarMappable(cmap=cm.plasma,
                                         norm=pyplot.Normalize(vmin=Es[0],
                                                               vmax=Es[-1]))
            sm._A = []
            cbar= pyplot.colorbar(sm,cax=pyplot.gca(),use_gridspec=True,
                                  format=r'$%g$')
            cbar.set_label(r'$E$')
            outer.tight_layout(pyplot.gcf())
        return None        

    def plot_orbit(self,E):
        ta= numpy.linspace(0.,2.*numpy.pi,1001)
        if not self._interp:
            # First find the torus for this energy
            indx= numpy.nanargmin(numpy.fabs(E-self._Es))
            if numpy.fabs(E-self._Es[indx]) > 1e-10:
                raise ValueError('Given energy not found; please specify an energy used in the initialization of the instance')
            tJ= self._js[indx]
        else:
            tJ= self.J(E)
        x,v= self(tJ,ta)
        # First plot orbit in x,v
        pyplot.subplot(1,2,1)
        bovy_plot.bovy_plot(x,v,xlabel=r'$x$',ylabel=r'$v$',gcf=True,
                            xrange=[1.1*numpy.amin(x),1.1*numpy.amax(x)],
                            yrange=[1.1*numpy.amin(v),1.1*numpy.amax(v)])
        # Then plot energy
        pyplot.subplot(1,2,2)
        Eorbit= (v**2./2.+evaluatelinearPotentials(self._pot,x))/E-1.
        ymin, ymax= numpy.amin(Eorbit),numpy.amax(Eorbit)
        bovy_plot.bovy_plot(ta,Eorbit,
                            xrange=[0.,2.*numpy.pi],
                            yrange=[ymin-(ymax-ymin)*3.,ymax+(ymax-ymin)*3.],
                            gcf=True,
                            xlabel=r'$\theta$',
                            ylabel=r'$E/E_{\mathrm{true}}-1$')
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

    def plot_interp(self,E,symm=True):
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

     <   PURPOSE:

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
            txmax= self._xmaxs[indx]
            tptxmax= self._pt_xmaxs[indx]
            tptcoeffs= self._pt_coeffs[indx]
            tptderivcoeffs= self._pt_deriv_coeffs[indx]
        else:
            tE= self.E(j)
            tnSn= self.nSn(tE)[0]
            tdSndJ= self.dSndJ(tE)[0]
            tOmegaHO= self.OmegaHO(tE)
            tOmega= self.Omega(tE)
        # First we need to solve for a<nglea
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
        xa,va= hoaainv(ja,anglea)
        x= txmax*polynomial.polyval((xa/tptxmax).T,tptcoeffs.T,tensor=False).T
        v= va*tptxmax/txmax/polynomial.polyval((xa/tptxmax).T,tptderivcoeffs.T,
                                               tensor=False).T
        return (x,v,tOmega)
        
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
        # Find t<orus
        if not self._interp:
            indx= numpy.nanargmin(numpy.fabs(j-self._js))
            if numpy.fabs(j-self._js[indx]) > 1e-10:
                raise ValueError('Given action/energy not found, to use interpolation, initialize with setup_interp=True')
            tOmega= self._Omegas[indx]
        else:
            tE= self.E(j)
            tOmega= self.Omega(tE)
        return tOmega

def _anglea(x,E,pot,omega,ptcoeffs,ptderivcoeffs,xmax,ptxmax,vsign=1.):
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
       ptcoeffs - coefficients of the polynomial point transformation
       ptderivcoeffs - coefficients of the derivative of the polynomial point transformation
       xmax - xmax of the true torus
       ptxmax - xmax of the point-transformed torus
    OUTPUT:
       auxiliary angles
    HISTORY:
       2018-04-13 - Written - Bovy (UofT)
       2018-11-19 - Added point transformation - Bovy (UofT)
    """
    # Compute v
    v2= 2.*(E-evaluatelinearPotentials(pot,
                                       xmax*polynomial.polyval((x/ptxmax).T,
                                                               ptcoeffs.T,
                                                               tensor=False).T))
    v2[v2 < 0.]= 0.
    return numpy.arctan2(omega*x,xmax/ptxmax*polynomial.polyval((x/ptxmax).T,
                                                              ptderivcoeffs.T,
                                                                tensor=False).T\
                             *vsign*numpy.sqrt(v2))

def _danglea(xa,E,pot,omega,ptcoeffs,ptderivcoeffs,ptderiv2coeffs,
             xmax,ptxmax,vsign=1.):
    """
    NAME:
       _danglea
    PURPOSE:
       Compute the derivative of the auxiliary angle in the harmonic-oscillator for a grid in x and E at constant E
    INPUT:
       xa - position
       E - Energy
       pot - the potential
       omega - harmonic-oscillator frequencies
       ptcoeffs - coefficients of the polynomial point transformation
       ptderivcoeffs - coefficients of the derivative of the polynomial point transformation
       ptderiv2coeffs - coefficients of the second derivative of the polynomial point transformation
       xmax - xmax of the true torus
       ptxmax - xmax of the point-transformed torus
    OUTPUT:
       d auxiliary angles / d x (2D array)
    HISTORY:
       2018-04-13 - Written - Bovy (UofT)
       2018-11-22 - Added point transformation - Bovy (UofT)
    """
    # Compute v
    x= xmax*polynomial.polyval((xa/ptxmax).T,ptcoeffs.T,tensor=False).T
    v2= 2.*(E-evaluatelinearPotentials(pot,x))
    v2[v2 < 1e-10]= 2.*(E[v2 < 1e-10]-evaluatelinearPotentials(pot,
     xmax[v2 < 1e-10]*polynomial.polyval((xa[v2 < 1e-10]/ptxmax[v2 < 1e-10]).T,
                                                        ptcoeffs[v2 < 1e-10].T,
                                                              tensor=False).T))
    piprime= xmax/ptxmax*polynomial.polyval((xa/ptxmax).T,ptderivcoeffs.T,
                                            tensor=False).T
    anglea= numpy.arctan2(omega*xa,piprime*vsign*numpy.sqrt(v2))
    return omega*numpy.cos(anglea)**2.*v2**-1.5/piprime\
        *(v2*(1.
           -xa*xmax/ptxmax**2./piprime\
                  *polynomial.polyval((xa/ptxmax).T,ptderiv2coeffs.T,
                                      tensor=False).T)
          -xa*evaluatelinearForces(pot,x)*piprime)

def _ja(x,E,pot,omega,ptcoeffs,ptderivcoeffs,xmax,ptxmax,vsign=1.):
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
    v2over2= (E-evaluatelinearPotentials(pot,
                                         xmax*polynomial.polyval((x/ptxmax).T,
                                                                ptcoeffs.T,
                                                                tensor=False).T))
    v2over2[v2over2 < 0.]= 0.
    return ((xmax/ptxmax*polynomial.polyval((x/ptxmax).T,
                                            ptderivcoeffs.T,
                                            tensor=False).T)**2.*v2over2/omega\
                +omega*x**2./2.)

def _djadj(xa,E,pot,omega,ptcoeffs,ptderivcoeffs,ptderiv2coeffs,
           xmax,ptxmax,vsign=1.):
    """
    NAME:
       _djaj
    PURPOSE:
       Compute the derivative of the auxiliary action in the harmonic-oscillator wrt the action for a grid in x and E
    INPUT:
       xa - position
       E - Energy
       pot - the potential
       omega - harmonic-oscillator frequencies
    OUTPUT:
       d(auxiliary actions)/d(action)
    HISTORY:
       2018-04-14 - Written - Bovy (UofT)
    """
    x= xmax*polynomial.polyval((xa/ptxmax).T,ptcoeffs.T,tensor=False).T
    v2= 2.*(E-evaluatelinearPotentials(pot,x))
    piprime= xmax/ptxmax*polynomial.polyval((xa/ptxmax).T,ptderivcoeffs.T,
                                            tensor=False).T
    piprime2= xmax/ptxmax**2.\
        *polynomial.polyval((xa/ptxmax).T,ptderiv2coeffs.T,tensor=False).T
    dxAdE= xa/(v2*(1.-piprime2/piprime*xa)
              -xa*evaluatelinearForces(pot,x)*piprime)
    return (piprime**2.
            +(piprime**3.*evaluatelinearForces(pot,x)
              +omega**2.*xa+piprime*piprime2*v2)*dxAdE)
              
