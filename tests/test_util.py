#Test the functions in galpy/util/__init__.py
from __future__ import print_function, division
import numpy

def test_save_pickles():
    import os
    import tempfile
    import pickle
    from galpy.util import save_pickles
    savethis= numpy.linspace(0.,100.,1001)
    savefile, tmp_savefilename= tempfile.mkstemp()
    try:
        os.close(savefile) #Easier this way 
        save_pickles(tmp_savefilename,savethis)
        savefile= open(tmp_savefilename,'rb')
        restorethis= pickle.load(savefile)
        savefile.close()
        assert numpy.all(numpy.fabs(restorethis-savethis) < 10.**-10.), 'save_pickles did not work as expected'
    finally:
        os.remove(tmp_savefilename)
    #Also test the handling of KeyboardInterrupt
    try:
        save_pickles(tmp_savefilename,savethis,testKeyboardInterrupt=True)
    except KeyboardInterrupt:
        pass
    else:
        raise AssertionError('save_pickles with testKeyboardInterrupt=True did not raise KeyboardInterrupt')
    savefile= open(tmp_savefilename,'rb')
    restorethis= pickle.load(savefile)
    savefile.close()
    assert numpy.all(numpy.fabs(restorethis-savethis) < 10.**-10.), 'save_pickles did not work as expected when KeyboardInterrupted'
    if os.path.exists(tmp_savefilename): os.remove(tmp_savefilename)   
    return None

def test_logsumexp():
    from galpy.util import logsumexp
    sumthis= numpy.array([[0.,1.]])
    sum= numpy.log(numpy.exp(0.)+numpy.exp(1.))
    assert numpy.all(numpy.fabs(logsumexp(sumthis,axis=0)-sumthis) < 10.**-10.), 'galpy.util.logsumexp did not work as expected'
    assert numpy.fabs(logsumexp(sumthis,axis=1)-sum) < 10.**-10., 'galpy.util.logsumexp did not work as expected'
    assert numpy.fabs(logsumexp(sumthis,axis=None)-sum) < 10.**-10., 'galpy.util.logsumexp did not work as expected'
    return None

def test_fast_cholesky_invert():
    from galpy.util import fast_cholesky_invert
    matrix= numpy.array([[2.,1.],[1.,4.]])
    invmatrix= fast_cholesky_invert(matrix)
    unit= numpy.dot(invmatrix,matrix)
    assert numpy.all(numpy.fabs(numpy.diag(unit)-1.) < 10.**-8.), 'fast_cholesky_invert did not work as expected'
    assert numpy.fabs(unit[0,1]-0.) < 10.**-8., 'fast_cholesky_invert did not work as expected'
    assert numpy.fabs(unit[1,0]-0.) < 10.**-8., 'fast_cholesky_invert did not work as expected'
    #Check the other way around
    unit= numpy.dot(matrix,invmatrix)
    assert numpy.all(numpy.fabs(numpy.diag(unit)-1.) < 10.**-8.), 'fast_cholesky_invert did not work as expected'
    assert numpy.fabs(unit[0,1]-0.) < 10.**-8., 'fast_cholesky_invert did not work as expected'
    assert numpy.fabs(unit[1,0]-0.) < 10.**-8., 'fast_cholesky_invert did not work as expected'
    #Also check determinant
    invmatrix, logdet= fast_cholesky_invert(matrix,logdet=True)
    assert numpy.fabs(logdet-numpy.log(7.)) < 10.**-8., "fast_cholesky_invert's determinant did not work as expected"
    return None

def test_quadpack():
    from galpy.util.quadpack import dblquad
    int= dblquad(lambda y,x: 4.*x*y,0.,1.,lambda z: 0.,lambda z: 1.)
    assert numpy.fabs(int[0]-1.) < int[1], 'galpy.util.quadpack.dblquad did not work as expected'
    return None
