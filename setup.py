from setuptools import setup
from distutils.core import Extension
import subprocess
import os, os.path
import glob

longDescription= ""

#Orbit integration C extension
orbit_int_c_src= ['galpy/util/bovy_symplecticode.c','galpy/util/bovy_rk.c']
orbit_int_c_src.extend(glob.glob('galpy/potential_src/potential_c_ext/*.c'))
orbit_int_c_src.extend(glob.glob('galpy/orbit_src/orbit_c_ext/*.c'))

orbit_int_c= Extension('galpy_integrate_c',
                       sources=orbit_int_c_src,
                       libraries=['m'],
                       include_dirs=['galpy/util',
                                     'galpy/potential_src/potential_c_ext'])
ext_modules=[orbit_int_c]

#actionAngle C extension
actionAngle_c_src= glob.glob('galpy/actionAngle_src/actionAngle_c_ext/*.c')
actionAngle_c_src.extend(glob.glob('galpy/potential_src/potential_c_ext/*.c'))

#Installation of this extension using the GSL may (silently) fail, if the GSL
#is built for the wrong architecture, on Mac you can install the GSL correctly
#using
#brew install gsl --universal
actionAngle_c= Extension('galpy_actionAngle_c',
                         sources=actionAngle_c_src,
                         libraries=['m','gsl','gslcblas'],
                         include_dirs=['galpy/actionAngle_src/actionAngle_c_ext',
                                       'galpy/potential_src/potential_c_ext'])
#code to check the GSL version
cmd= ['gsl-config',
      '--version']
try:
    gsl_version= subprocess.check_output(cmd)
except (OSError,subprocess.CalledProcessError):
    raise
    pass
else:
    gsl_version= gsl_version.split('.')
    if float(gsl_version[0]) >= 1. and float(gsl_version[1]) > 14.:
        ext_modules.append(actionAngle_c)

setup(name='galpy',
      version='1.',
      description='Galactic Dynamics in python',
      author='Jo Bovy',
      author_email='bovy@ias.edu',
      license='New BSD',
      long_description=longDescription,
      url='https://github.com/jobovy/galpy',
      package_dir = {'galpy/': ''},
      packages=['galpy','galpy/orbit_src','galpy/potential_src',
                'galpy/df_src','galpy/util','galpy/snapshot_src',
                'galpy/actionAngle_src'],
      package_data={'galpy/df_src':['data/*.sav']},
#      dependency_links = ['https://github.com/dfm/MarkovPy/tarball/master#egg=MarkovPy'],
      install_requires=['numpy','scipy','matplotlib'],
      ext_modules=ext_modules
      )
