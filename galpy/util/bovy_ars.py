#############################################################################
#Copyright (c) 2011, Jo Bovy
#All rights reserved.
#
#Redistribution and use in source and binary forms, with or without 
#modification, are permitted provided that the following conditions are met:
#
#   Redistributions of source code must retain the above copyright notice, 
#      this list of conditions and the following disclaimer.
#   Redistributions in binary form must reproduce the above copyright notice, 
#      this list of conditions and the following disclaimer in the 
#      documentation and/or other materials provided with the distribution.
#   The name of the author may not be used to endorse or promote products 
#      derived from this software without specific prior written permission.
#
#THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
#"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
#LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
#A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
#HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
#INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
#BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
#OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED
#AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
#LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY
#WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
#POSSIBILITY OF SUCH DAMAGE.
#############################################################################
import scipy as sc
import scipy.stats as stats
import math as m

#TO DO:
#Throw errors in the sample_hull routine

def bovy_ars(domain,isDomainFinite,abcissae,hx,hpx,nsamples=1,
             hxparams=(),maxn=100):
    """bovy_ars: Implementation of the Adaptive-Rejection Sampling
    algorithm by Gilks & Wild (1992): Adaptive Rejection Sampling
    for Gibbs Sampling, Applied Statistics, 41, 337
    Based on Wild & Gilks (1993), Algorithm AS 287: Adaptive Rejection
    Sampling from Log-concave Density Functions, Applied Statistics, 42, 701

    Input:

       domain          - [.,.] upper and lower limit to the domain

       isDomainFinite  - [.,.] is there a lower/upper limit to the domain?

       abcissae        - initial list of abcissae (must lie on either side of the peak in hx if the domain is unbounded

       hx              - function that evaluates h(x) = ln g(x)

       hpx             - function that evaluates hp(x) =  d h(x) / d x

       nsamples        - (optional) number of desired samples (default=1)

       hxparams        - (optional) a tuple of parameters for h(x) and h'(x)

       maxn            - (optional) maximum number of updates to the hull (default=100)
                         
    Output:

       list with nsamples of samples from exp(h(x))

    External dependencies:

       math
       scipy
       scipy.stats

    History:
       2009-05-21 - Written - Bovy (NYU)
    """
    #First set-up the upper and lower hulls
    hull=setup_hull(domain,isDomainFinite,abcissae,hx,hpx,hxparams)
    #Then start  sampling: call sampleone repeatedly
    out= []
    nupdates= 0
    for ii in range(int(nsamples)):
        thissample, hull, nupdates= sampleone(hull,hx,hpx,domain,isDomainFinite,maxn,nupdates,hxparams)
        out.append(thissample)
    return out

def setup_hull(domain,isDomainFinite,abcissae,hx,hpx,hxparams):
    """setup_hull: set up the upper and lower hull and everything that
    comes with that

    Input:
       domain          - [.,.] upper and lower limit to the domain
       isDomainFinite  - [.,.] is there a lower/upper limit to the domain?
       abcissae        - initial list of abcissae (must lie on either side
                         of the peak in hx if the domain is unbounded
       hx              - function that evaluates h(x)
       hpx             - function that evaluates hp(x)
       hxparams        - tuple of parameters for h(x) and h'(x)

    Output:
       list with:
       [0]= c_u
       [1]= xs
       [2]= h(xs)
       [3]= hp(xs)
       [4]= zs
       [5]= s_cum
       [6]= hu(zi)

    History:
       2009-05-21 - Written - Bovy (NYU)
    """
    nx= len(abcissae)
    #Create the output arrays
    xs= sc.zeros(nx)
    hxs= sc.zeros(nx)
    hpxs= sc.zeros(nx)
    zs= sc.zeros(nx-1)
    scum= sc.zeros(nx-1)
    hus= sc.zeros(nx-1)
    #Function evaluations
    xs= sc.sort(abcissae)
    for ii in range(nx):
        hxs[ii]= hx(xs[ii],hxparams)
        hpxs[ii]= hpx(xs[ii],hxparams)
    #THERE IS NO CHECKING HERE TO SEE WHETHER IN THE INFINITE DOMAIN CASE
    #WE HAVE ABCISSAE ON BOTH SIDES OF THE PEAK
    #zi
    for jj in range(nx-1):
        zs[jj]= (hxs[jj+1]-hxs[jj]-xs[jj+1]*hpxs[jj+1]+xs[jj]*hpxs[jj])/(
            hpxs[jj]-hpxs[jj+1])
    #hu
    for jj in range(nx-1):
        hus[jj]= hpxs[jj]*(zs[jj]-xs[jj])+hxs[jj]
    #Calculate cu and scum
    if isDomainFinite[0]:
        scum[0]= 1./hpxs[0]*(m.exp(hus[0])-m.exp(
            hpxs[0]*(domain[0]-xs[0])+hxs[0]))
    else:
        scum[0]= 1./hpxs[0]*m.exp(hus[0])
    if nx > 2:
        for jj in range(nx-2):
            if hpxs[jj+1] == 0.:
                scum[jj+1]= (zs[jj+1]-zs[jj])*m.exp(hxs[jj+1])
            else:
                scum[jj+1]=1./hpxs[jj+1]*(m.exp(hus[jj+1])-m.exp(hus[jj]))
    if isDomainFinite[1]:
        cu=1./hpxs[nx-1]*(m.exp(hpxs[nx-1]*(
            domain[1]-xs[nx-1])+hxs[nx-1]) - m.exp(hus[nx-2]))
    else:
        cu=- 1./hpxs[nx-1]*m.exp(hus[nx-2])
    cu= cu+sc.sum(scum)
    scum= sc.cumsum(scum)/cu
    out=[]
    out.append(cu)
    out.append(xs)
    out.append(hxs)
    out.append(hpxs)
    out.append(zs)
    out.append(scum)
    out.append(hus)

    return out

def sampleone(hull,hx,hpx,domain,isDomainFinite,maxn,nupdates,hxparams):
    """sampleone: sample one point by ars

    Input:
       hull            - the hull (see doc of setup_hull for definition)
       hx              - function that evaluates h(x)
       hpx             - function that evaluates hp(x)
       domain          - [.,.] upper and lower limit to the domain
       isDomainFinite  - [.,.] is there a lower/upper limit to the domain?
       maxn            - maximum number of updates to the hull
       nupdates        - number of updates to the hull that have occured
       hxparams        - tuple of parameters for h(x) and h'(x)

    Output:
       a sample
       a new hull
       nupdates

    History:
       2009-05-21 - Written - Bovy (NYU)
    """
    thishull= hull
    noSampleYet= True
    while noSampleYet:
        #Sample a candidate from the upper hull
        candidate= sample_hull(thishull,domain,isDomainFinite)
        thishux, thishlx= evaluate_hull(candidate,thishull)
        u= stats.uniform.rvs()
        if u < m.exp(thishlx-thishux):
            thissample= candidate
            noSampleYet= False
        else:
            thishx= hx(candidate,hxparams)
            if u < m.exp(thishx-thishux):
                thissample= candidate
                noSampleYet= False
            if nupdates < maxn:
                thishpx= hpx(candidate,hxparams)
                thishull= update_hull(thishull,candidate,thishx,thishpx,
                                      domain,isDomainFinite)
                nupdates= nupdates+1
    return thissample, thishull, nupdates

def sample_hull(hull,domain,isDomainFinite):
    """sample_hull: Sample the upper hull

    Input:
       hull       - hull structure (see setup_hull for a definition of this)
       domain          - [.,.] upper and lower limit to the domain
       isDomainFinite  - [.,.] is there a lower/upper limit to the domain?

    Output:
       a sample from the hull

    History:
       2009-05-21 - Written - Bovy
    """
    u= stats.uniform.rvs()
    #Find largest zs[jj] such that scum[jj] < u
    #The first bin is a special case
    if hull[5][0] >= u:
        if hull[3][0] == 0:
            if isDomainFinite[0]:
                thissample= domain[0]+u/hull[5][0]*(hull[4][0]-domain[0])
            else:
                thissample= 100000000 #Throw some kind of error
        else:
            thissample= hull[4][0]+1./hull[3][0]*m.log(1.-hull[3][0]*hull[0]*(hull[5][0]-u)/m.exp(hull[6][0]))
    else:
        if len(hull[5]) == 1:
            indx= 0
        else:
            indx= 1
            while indx < len(hull[5]) and hull[5][indx] < u:
                indx= indx+1
            indx= indx-1
        if m.fabs(hull[3][indx+1]) == 0:
            if indx != (len(hull[5])-1):
                thissample= hull[4][indx]+(u-hull[5][indx])/(hull[5][indx+1]-hull[5][indx])*(hull[4][indx+1]-hull[4][indx])
            else:
                if isDomainFinite[1]:
                    thissample= hull[4][indx]+(u-hull[5][indx])/(1.-hull[5][indx])*(domain[1]-hull[4][indx])
                else:
                    thissample= 100000 #Throw some kind of error
        else:
            thissample= hull[4][indx]+1./hull[3][indx+1]*m.log(1.+hull[3][indx+1]*hull[0]*(u-hull[5][indx])/m.exp(hull[6][indx]))
    return thissample

def evaluate_hull(x,hull):
    """evaluate_hull: evaluate h_u(x) and (optional) h_l(x)

    Input:
       x     - abcissa
       hull  - the hull (see setup_hull for a definition)

    Output:
      hu(x) (optional), hl(x)

    History:
       2009-05-21 - Written - Bovy (NYU)
    """
    #Find in which [z_{i-1},z_i] interval x lies
    if x < hull[4][0]:
        #x lies in the first interval
        hux= hull[3][0]*(x-hull[1][0])+hull[2][0]
        indx= 0
    else:
        if len(hull[5]) == 1:
            #There are only two intervals
            indx= 1
        else:
            indx= 1
            while indx < len(hull[4]) and hull[4][indx] < x:
                indx= indx+1
            indx= indx-1
        hux= hull[3][indx]*(x-hull[1][indx])+hull[2][indx]
    #Now evaluate hlx
    neginf= sc.finfo(sc.dtype(sc.float64)).min
    if x < hull[1][0] or x > hull[1][-1]:
        hlx= neginf
    else:
        if indx == 0:
            hlx= ((hull[1][1]-x)*hull[2][0]+(x-hull[1][0])*hull[2][1])/(hull[1][1]-hull[1][0])
        elif indx == len(hull[4]):
            hlx= ((hull[1][-1]-x)*hull[2][-2]+(x-hull[1][-2])*hull[2][-1])/(hull[1][-1]-hull[1][-2])
        elif x < hull[1][indx+1]:
            hlx= ((hull[1][indx+1]-x)*hull[2][indx]+(x-hull[1][indx])*hull[2][indx+1])/(hull[1][indx+1]-hull[1][indx])
        else:
            hlx= ((hull[1][indx+2]-x)*hull[2][indx+1]+(x-hull[1][indx+1])*hull[2][indx+2])/(hull[1][indx+2]-hull[1][indx+1])
    return hux, hlx
    
def update_hull(hull,newx,newhx,newhpx,domain,isDomainFinite):
    """update_hull: update the hull with a new function evaluation

    Input:
       hull            - the current hull (see setup_hull for a definition)
       newx            - a new abcissa
       newhx           - h(newx)
       newhpx          - hp(newx)
       domain          - [.,.] upper and lower limit to the domain
       isDomainFinite  - [.,.] is there a lower/upper limit to the domain?

    Output:
       newhull

    History:
       2009-05-21 - Written - Bovy (NYU)
    """
    #BOVY: Perhaps add a check that newx is sufficiently far from any existing point
    #Find where newx fits in with the other xs
    if newx > hull[1][-1]:
        newxs= sc.append(hull[1],newx)
        newhxs= sc.append(hull[2],newhx)
        newhpxs= sc.append(hull[3],newhpx)
        #new z
        newz= ( newhx - hull[2][-1] - newx*newhpx + hull[1][-1]*hull[3][-1])/( hull[3][-1] - newhpx)
        newzs= sc.append(hull[4],newz)
        #New hu
        newhu= hull[3][-1]*(newz-hull[1][-1]) + hull[2][-1]
        newhus= sc.append(hull[6],newhu)
    else:
        indx= 0
        while newx > hull[1][indx]:
            indx=indx+1
        newxs= sc.insert(hull[1],indx,newx)
        newhxs= sc.insert(hull[2],indx,newhx)
        newhpxs= sc.insert(hull[3],indx,newhpx)
        #Replace old z with new zs
        if newx < hull[1][0]:
            newz= (hull[2][0]-newhx-hull[1][0]*hull[3][0]+newx*newhpx)/(newhpx-hull[3][0])
            newzs= sc.insert(hull[4],0,newz)
            #Also add the new hu
            newhu= newhpx*(newz-newx)+newhx
            newhus= sc.insert(hull[6],0,newhu)
        else:
            newz1= (newhx-hull[2][indx-1] - newx*newhpx+hull[1][indx-1]*hull[3][indx-1])/(hull[3][indx-1]-newhpx)
            newz2= (hull[2][indx]-newhx - hull[1][indx]*hull[3][indx]+newx*newhpx)/(newhpx-hull[3][indx])
            #Insert newz1 and replace z_old
            newzs= sc.insert(hull[4],indx-1,newz1)
            newzs[indx]= newz2
            #Update the hus
            newhu1= hull[3][indx-1]*(newz1-hull[1][indx-1])+hull[2][indx-1]
            newhu2= newhpx*(newz2-newx)+newhx
            newhus= sc.insert(hull[6],indx-1,newhu1)
            newhus[indx]= newhu2
    #Recalculate the cumulative sum
    nx= len(newxs)
    newscum= sc.zeros(nx-1)
    if isDomainFinite[0]:
        newscum[0]= 1./newhpxs[0]*(m.exp(newhus[0])-m.exp(
            newhpxs[0]*(domain[0]-newxs[0])+newhxs[0]))
    else:
        newscum[0]= 1./newhpxs[0]*m.exp(newhus[0])
    if nx > 2:
        for jj in range(nx-2):
            if newhpxs[jj+1] == 0.:
                newscum[jj+1]= (newzs[jj+1]-newzs[jj])*m.exp(newhxs[jj+1])
            else:
                newscum[jj+1]=1./newhpxs[jj+1]*(m.exp(newhus[jj+1])-m.exp(newhus[jj]))
    if isDomainFinite[1]:
        newcu=1./newhpxs[nx-1]*(m.exp(newhpxs[nx-1]*(
            domain[1]-newxs[nx-1])+newhxs[nx-1]) - m.exp(newhus[nx-2]))
    else:
        newcu=- 1./newhpxs[nx-1]*m.exp(newhus[nx-2])
    newcu= newcu+sc.sum(newscum)
    newscum= sc.cumsum(newscum)/newcu
    newhull=[]
    newhull.append(newcu)
    newhull.append(newxs)
    newhull.append(newhxs)
    newhull.append(newhpxs)
    newhull.append(newzs)
    newhull.append(newscum)
    newhull.append(newhus)
    return newhull

