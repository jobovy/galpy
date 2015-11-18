###############################################################################
# nemo_util.py: some utilities for handling NEMO snapshots
###############################################################################
import os
import numpy
import tempfile
import subprocess
def read(filename,ext=None):
    """
    NAME:
       read
    PURPOSE:
       read a NEMO snapshot file
    INPUT:
       filename - name of the file
       ext= if set, 'nemo' for NEMO binary format, otherwise assumed ASCII; if not set, gleaned from extension
    OUTPUT:
       snapshots [nbody,ndim,nt]
    HISTORY:
       2015-11-18 - Written - Bovy (UofT)
    """
    if ext is None and filename.split('.')[-1] == 'nemo':
        ext= 'nemo'
    # Convert to ASCII if necessary
    if ext.lower() == 'nemo':
        file_handle, asciifilename= tempfile.mkstemp()
        stderr= open('/dev/null','w')
        try:
            subprocess.check_call(['s2a',filename],stdout=file_handle,
                                  stderr=stderr)
        except subprocess.CalledProcessError:
            file_handle.close()
            os.remove(asciifilename)
        else:
            file_handle.close()
        finally:
            stderr.close()
    else:
        asciifilename= filename
    # Now read
    out= numpy.loadtxt(asciifilename,comment='#')
    # Get the number of snapshots
    nt= (_wc(asciifilename)-numpy.prod(out.shape))//13 # 13 comments/snapshot
    out= numpy.reshape(out,(nt,out.shape[0]//nt,out.shape[1]))
    return out

def _wc(filename):
    try:
        return subprocess.check_output(['wc','-l',filename])
    except subprocess.CalledProcessError:
        return numpy.nan
