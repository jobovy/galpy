import re
import sys
import os, os.path
import math as m
import numpy as nu
import csv
import cPickle as pickle
import galpy.util.bovy_plot as plot
from galpy.potential import MiyamotoNagaiPotential, HernquistPotential, NFWPotential, LogarithmicHaloPotential, PowerSphericalPotential
from galpy.orbit import Orbit
from galpy.actionAngle import actionAngleFlat, actionAnglePower
_degtorad= nu.pi/180.
def hms_to_rad(ra):
    spl= re.split(r' ',ra)
    return (float(spl[0])*15.+float(spl[1])*0.25+
            float(spl[1])*0.25/60.)*_degtorad
def dms_to_rad(dec):
    spl= re.split(r' ',dec)
    return (float(spl[0])+float(spl[1])/60.+float(spl[2])/60./60.)*_degtorad
def read_float(f):
    if f == '':
        return -9999
    else:
        return float(f)
def calcj(rotcurve):
    if rotcurve == 'flat':
        savefilename= 'myjs.sav'
    elif rotcurve == 'power':
        savefilename= 'myjs_power.sav'
    if os.path.exists(savefilename):
        savefile= open(savefilename,'rb')
        myjr= pickle.load(savefile)
        myjp= pickle.load(savefile)
        mywr= pickle.load(savefile)
        mywp= pickle.load(savefile)
        mye= pickle.load(savefile)
        myzmax= pickle.load(savefile)
        e= pickle.load(savefile)
        zmax= pickle.load(savefile)
        savefile.close()
    else:
        dialect= csv.excel
        dialect.skipinitialspace=True
        reader= csv.reader(open('../data/gcs.tsv','r'),delimiter='|',dialect=dialect)
        vxvs= []
        es= []
        zmaxs= []
        for row in reader:
            if row[0][0] == '#':
                continue
            thisra= row[0]
            thisdec= row[1]
            thisd= read_float(row[2])/1000.
            if thisd > 0.2: continue
            thisu= read_float(row[3])
            thisv= read_float(row[4])
            thisw= read_float(row[5])
            thise= read_float(row[6])
            thiszmax= read_float(row[7])
            if thisd == -9999 or thisu == -9999 or thisv == -9999 or thisw == -9999:
                continue
            vxvs.append([hms_to_rad(thisra),dms_to_rad(thisdec),
                         thisd,thisu,thisv,thisw])
            es.append(thise)
            zmaxs.append(thiszmax)
        vxvv= nu.array(vxvs)
        e= nu.array(es)
        zmax= nu.array(zmaxs)

        #Define potential
        lp= LogarithmicHaloPotential(normalize=1.)
        pp= PowerSphericalPotential(normalize=1.,alpha=-2.)
        mp= MiyamotoNagaiPotential(a=0.5,b=0.0375,amp=1.,normalize=.6)
        np= NFWPotential(a=4.5,normalize=.35)
        hp= HernquistPotential(a=0.6/8,normalize=0.05)
        ts= nu.linspace(0.,20.,10000)

        myjr= nu.zeros(len(e))
        myjp= nu.zeros(len(e))
        mywr= nu.zeros(len(e))
        mywp= nu.zeros(len(e))
        mye= nu.zeros(len(e))
        myzmax= nu.zeros(len(e))
        for ii in range(len(e)):
           #Integrate the orbit
            o= Orbit(vxvv[ii,:],radec=True,uvw=True,vo=220.,ro=8.)
            #o.integrate(ts,[mp,np,hp])
            if rotcurve == 'flat':
                o.integrate(ts,lp)
                mye[ii]= o.e()
                myzmax[ii]= o.zmax()*8.
                print e[ii], mye[ii], zmax[ii], myzmax[ii]
                o= o.toPlanar()
                myjr[ii]= o.jr(lp)[0]
            else:
                o= o.toPlanar()
                myjr[ii]= o.jr(pp)[0]
            myjp[ii]= o.jp()[0]
            mywr[ii]= o.wr()[0]
            mywp[ii]= o.wp()

        #Save
        savefile= open(savefilename,'wb')
        pickle.dump(myjr,savefile)
        pickle.dump(myjp,savefile)
        pickle.dump(mywr,savefile)
        pickle.dump(mywp,savefile)
        pickle.dump(mye,savefile)
        pickle.dump(myzmax,savefile)
        pickle.dump(e,savefile)
        pickle.dump(zmax,savefile)
        savefile.close()

    #plot
    if rotcurve == 'flat':
        plot.bovy_print()
        plot.bovy_plot(nu.array([0.,1.]),nu.array([0.,1.]),'k-',
                       xlabel=r'$\mathrm{Holmberg\ et\ al.}\ e$',
                       ylabel=r'$\mathrm{galpy}\ e$')
        plot.bovy_plot(e,mye,'k,',overplot=True)
        plot.bovy_end_print('myee.png')
        
        plot.bovy_print()
        plot.bovy_plot(nu.array([0.,2.5]),
                       nu.array([0.,2.5]),'k-',
                       xlabel=r'$\mathrm{Holmberg\ et\ al.}\ z_{\mathrm{max}}$',
                   ylabel=r'$\mathrm{galpy}\ z_{\mathrm{max}}$')
        plot.bovy_plot(zmax,myzmax,'k,',overplot=True)
        plot.bovy_end_print('myzmaxzmax.png')

    plot.bovy_print()
    plot.bovy_plot(myjp,myjr/2./nu.pi,'k,',
                   xlabel=r'$J_{\phi}$',
                   ylabel=r'$J_R/2\pi$',
                   xrange=[0.7,1.3],
                   yrange=[0.,0.05])
    if rotcurve == 'flat':
        plot.bovy_end_print('jrjp.png')
    else:
        plot.bovy_end_print('jrjp_power.png')

if __name__ == '__main__':
    if len(sys.argv) > 1:
        calcj(sys.argv[1])
    else:
        calcj('flat')
