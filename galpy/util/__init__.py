import os
import shutil
import warnings
import copy
import tempfile
import pickle
import numpy
class galpyWarning(Warning):
    pass
old_showwarning= copy.copy(warnings.showwarning)
def _warning(
    message,
    category = galpyWarning,
    filename = '',
    lineno = -1):
    if issubclass(category,galpyWarning):
        print("galpyWarning: "+str(message))
    else:
        old_showwarning(message,category,filename,lineno)
warnings.showwarning = _warning
def save_pickles(savefilename,*args):
    """
    NAME:
       save_pickles
    PURPOSE:
       relatively safely save things to a pickle
    INPUT:
       savefilename - name of the file to save to; actual save operation will be performed on a temporary file and then that file will be shell mv'ed to savefilename
       +things to pickle (as many as you want!)
    OUTPUT:
       none
    HISTORY:
       2010-? - Written - Bovy (NYU)
       2011-08-23 - generalized and added to galpy.util - Bovy (NYU)
    """
    saving= True
    interrupted= False
    file, tmp_savefilename= tempfile.mkstemp() #savefilename+'.tmp'
    os.close(file) #Easier this way
    while saving:
        try:
            savefile= open(tmp_savefilename,'wb')
            file_open= True
            for f in args:
                pickle.dump(f,savefile)
            savefile.close()
            file_open= False
            shutil.move(tmp_savefilename,savefilename)
            saving= False
            if interrupted:
                raise KeyboardInterrupt
        except KeyboardInterrupt:
            if not saving:
                raise
            print "KeyboardInterrupt ignored while saving pickle ..."
            interrupted= True
        finally:
            if file_open:
                savefile.close()

def logsumexp(arr,axis=0):
    """Faster logsumexp?"""
    minarr= numpy.amax(arr,axis=axis)
    if axis == 1:
        minarr= numpy.reshape(minarr,(arr.shape[0],1))
    if axis == 0:
        minminarr= numpy.tile(minarr,(arr.shape[0],1))
    elif axis == 1:
        minminarr= numpy.tile(minarr,(1,arr.shape[1]))
    elif axis == None:
        minminarr= numpy.tile(minarr,arr.shape)
    else:
        raise NotImplementedError("'galpy.util.logsumexp' not implemented for axis > 2")
    if axis == 1:
        minarr= numpy.reshape(minarr,(arr.shape[0]))
    return minarr+numpy.log(numpy.sum(numpy.exp(arr-minminarr),axis=axis))

def stable_cholesky(x,tiny):
    """
    NAME:
       stable_cholesky
    PURPOSE:
       Stable version of the cholesky decomposition
    INPUT:
       x - (sc.array) positive definite matrix
       tiny - (double) tiny number to add to the covariance matrix to make the decomposition stable
    OUTPUT:
       L - (matrix) lower matrix
    REVISION HISTORY:
       2009-09-25 - Written - Bovy (NYU)
    """
    thisx= x+tiny*numpy.eye(x.shape[0])
    L= numpy.linalg.cholesky(thisx)
    return L
