#Test the functions in galpy/util/__init__.py
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
    return None
