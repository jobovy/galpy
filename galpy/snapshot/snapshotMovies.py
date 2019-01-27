#Functions to turn snapshots into movies
from __future__ import print_function
import re
import os, os.path
import tempfile
import shutil
import subprocess
import math as m
import galpy.util.bovy_plot as bovy_plot
from Snapshot import *
def snapshotToMovie(snap,filename,*args,**kwargs):
    """
    NAME:

       snapshotToMovie

    PURPOSE:

       turn a list of snapshots into a movie

    INPUT:

       snap - the snapshots (list)

       filename - name of the file to save the movie to

       framerate= in fps

       bitrate= ?
       
       thumbnail=False : create thumbnail image (filename-extension+.jpg)

       thumbsize= size of thumbnail
       
       +Snapshot.plot args and kwargs

    OUTPUT:
    
       movie is saved to file

    DEPENDENCIES:

       this procedure uses ffmpeg and convert

    BUGS:
    
       matplotlib's 'Agg' backend has a memory leak that prevents it from 
       creating hundred's of figures. It is recommended to call
       
       import matplotlib
       matplotlib.use('PDF')

       at the beginning of the movie creating script as the PDF backend does 
       not have the same memory leak.

    HISTORY:

       2011-02-06 - Written - Bovy (NYU)

    """
    if kwargs.has_key('tmpdir'):
        tmpdir= kwargs['tmpdir']
        kwargs.pop('tmpdir')
    else:
        tmpdir= '/tmp'
    if kwargs.has_key('framerate'):
        framerate= kwargs['framerate']
        kwargs.pop('framerate')
    else:
        framerate= 25
    if kwargs.has_key('bitrate'):
        bitrate= kwargs['bitrate']
        kwargs.pop('bitrate')
    else:
        bitrate= 1000
    if kwargs.has_key('thumbnail') and kwargs['thumbnail']:
        thumbnail= True
        kwargs.pop('thumbnail')
    elif kwargs.has_key('thumbnail'):
        kwargs.pop('thumbnail')
        thumbnail= False
    else:
        thumbnail= False
    if kwargs.has_key('thumbsize'):
        thumbsize= kwargs['thumbsize']
    else:
        thumbsize= 300
    #Create all of the files
    tempdir= tempfile.mkdtemp(dir=tmpdir) #Temporary directory
    tmpfiles= []
    nsnap= len(snap)
    file_length= int(m.ceil(m.log10(nsnap)))
    #Determine good xrange BOVY TO DO
    if not kwargs.has_key('xrange'):
        pass
    if not kwargs.has_key('yrange'):
        pass    
    for ii in range(nsnap):
        tmpfiles.append(os.path.join(tempdir,
                                     str(ii).zfill(file_length)))
        bovy_plot.bovy_print()
        snap[ii].plot(*args,**kwargs)
        bovy_plot.bovy_end_print(tmpfiles[ii]+'.pdf')
        #Convert to jpeg
        try:
            subprocess.check_call(['convert',
                                   tmpfiles[ii]+'.pdf',
                                   tmpfiles[ii]+'.jpg'])
        except subprocess.CalledProcessError:
            print("'convert' failed")
            raise subprocess.CalledProcessError
    #turn them into a movie
    try:
        subprocess.check_call(['ffmpeg',
                               '-r',str(framerate),
                               '-b', str(bitrate),
                              '-i',
                              os.path.join(tempdir,
                                           '%'+'0%id.jpg' % file_length),
                              '-y',
                              filename])
        if thumbnail:
            thumbnameTemp= re.split(r'\.',filename)
            thumbnameTemp= thumbnameTemp[0:len(thumbnameTemp)-1]
            thumbname= ''
            for t in thumbnameTemp:
                thumbname+= t
            thumbname+= '.jpg'
            subprocess.check_call(['ffmpeg',
                                   '-itsoffset','-4','-y',
                                   '-i',filename,
                                   '-vcodec',
                                   'mjpeg',
                                   '-vframes','1',
                                   '-an',
                                   '-f',
                                   'rawvideo',
                                   '-s', '%ix%i' % (thumbsize,thumbsize),
                                   thumbname])
    except subprocess.CalledProcessError:
        print("'ffmpeg' failed")
        _cleanupMovieTempdir(tempdir)
        raise subprocess.CalledProcessError
    finally:
        _cleanupMovieTempdir(tempdir)

def _cleanupMovieTempdir(tempdir):
    shutil.rmtree(tempdir)
