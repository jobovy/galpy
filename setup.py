from setuptools import setup
from distutils.core import Extension
import sys
import subprocess
import glob

with open('README.md') as file:
    long_description = file.read()

#code to check the GSL version
cmd= ['gsl-config',
      '--version']
try:
    if sys.version_info < (2,7): #subprocess.check_output does not exist
        gsl_version= subprocess.Popen(cmd,
                                      stdout=subprocess.PIPE).communicate()[0]
    else:
        gsl_version= subprocess.check_output(cmd)
except (OSError,subprocess.CalledProcessError):
    gsl_version= ['0','0']
else:
    gsl_version= gsl_version.split('.')
#HACK for testing
#gsl_version= ['0','0']

#Orbit integration C extension
orbit_int_c_src= ['galpy/util/bovy_symplecticode.c','galpy/util/bovy_rk.c']
orbit_int_c_src.extend(glob.glob('galpy/potential_src/potential_c_ext/*.c'))
orbit_int_c_src.extend(glob.glob('galpy/orbit_src/orbit_c_ext/*.c'))
orbit_int_c_src.extend(glob.glob('galpy/util/interp_2d/*.c'))

orbit_libraries=['m']
if float(gsl_version[0]) >= 1.:
    orbit_libraries.extend(['gsl','gslcblas'])
orbit_int_c= Extension('galpy_integrate_c',
                       sources=orbit_int_c_src,
                       libraries=orbit_libraries,
                       include_dirs=['galpy/util',
                                     'galpy/util/interp_2d',
                                     'galpy/potential_src/potential_c_ext'])
ext_modules=[]
if float(gsl_version[0]) >= 1.:
    ext_modules.append(orbit_int_c)

#actionAngle C extension
actionAngle_c_src= glob.glob('galpy/actionAngle_src/actionAngle_c_ext/*.c')
actionAngle_c_src.extend(glob.glob('galpy/potential_src/potential_c_ext/*.c'))
actionAngle_c_src.extend(glob.glob('galpy/util/interp_2d/*.c'))

#Installation of this extension using the GSL may (silently) fail, if the GSL
#is built for the wrong architecture, on Mac you can install the GSL correctly
#using
#brew install gsl --universal
actionAngle_c= Extension('galpy_actionAngle_c',
                         sources=actionAngle_c_src,
                         libraries=['m','gsl','gslcblas','gomp'],
                         include_dirs=['galpy/actionAngle_src/actionAngle_c_ext',
                                       'galpy/util/interp_2d',
                                       'galpy/potential_src/potential_c_ext'],
                         extra_compile_args=["-fopenmp"])
if float(gsl_version[0]) >= 1. and float(gsl_version[1]) > 14.:
    ext_modules.append(actionAngle_c)
    
#interppotential C extension
interppotential_c_src= glob.glob('galpy/potential_src/potential_c_ext/*.c')
interppotential_c_src.extend(glob.glob('galpy/potential_src/interppotential_c_ext/*.c'))
interppotential_c_src.extend(['galpy/util/bovy_symplecticode.c','galpy/util/bovy_rk.c'])
interppotential_c_src.append('galpy/actionAngle_src/actionAngle_c_ext/actionAngle.c')
interppotential_c_src.append('galpy/orbit_src/orbit_c_ext/integrateFullOrbit.c')
interppotential_c_src.extend(glob.glob('galpy/util/interp_2d/*.c'))

interppotential_c= Extension('galpy_interppotential_c',
                         sources=interppotential_c_src,
                         libraries=['m','gsl','gslcblas','gomp'],
                         include_dirs=['galpy/potential_src/potential_c_ext',
                                       'galpy/util/interp_2d',
                                       'galpy/util/',
                                       'galpy/actionAngle_src/actionAngle_c_ext',
                                       'galpy/orbit_src/orbit_c_ext',
                                       'galpy/potential_src/interppotential_c_ext'],
                         extra_compile_args=["-fopenmp"])
if float(gsl_version[0]) >= 1. and float(gsl_version[1]) > 14.:
    ext_modules.append(interppotential_c)

setup(name='galpy',
      version='0.1',
      description='Galactic Dynamics in python',
      author='Jo Bovy',
      author_email='bovy@ias.edu',
      license='New BSD',
      long_description=long_description,
      url='http://github.com/jobovy/galpy',
      package_dir = {'galpy/': ''},
      packages=['galpy','galpy/orbit_src','galpy/potential_src',
                'galpy/df_src','galpy/util','galpy/snapshot_src',
                'galpy/actionAngle_src'],
      package_data={'galpy/df_src':['data/*.sav'],
                    "": ["README.md","README.dev","LICENSE","AUTHORS.rst"]},
      include_package_data=True,
      install_requires=['numpy','scipy','matplotlib','nose'],
      ext_modules=ext_modules,
      classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
        "Programming Language :: C",
        "Programming Language :: Python",
        "Topic :: Scientific/Engineering :: Astronomy",
        "Topic :: Scientific/Engineering :: Physics"]
      )
