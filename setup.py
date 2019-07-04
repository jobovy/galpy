from setuptools import setup
from distutils.core import Extension
import sys
import distutils.sysconfig as sysconfig
import distutils.ccompiler
from distutils.errors import DistutilsPlatformError
import os, os.path
import platform
import subprocess
import glob
PY3= sys.version > '3'
WIN32= platform.system() == 'Windows'
no_compiler = False  # Flag for cases where we are sure there is no compiler exists in user's system

long_description= ''
previous_line= ''
with open('README.rst') as dfile:
    for line in dfile:
        if not 'image' in line and not 'target' in line \
                and not 'DETAILED' in line and not '**master**' in line \
                and not '**development' in line \
                and not 'DETAILED' in  previous_line:
            long_description+= line
        previous_line= line

# Parse options; current options
# --no-openmp: compile without OpenMP support
# --coverage: compile with gcov support
# --single_ext: compile all of the C code into a single extension (just for testing, do not use this)
# --orbit_ext: just compile the orbit extension (for use w/ --coverage)
# --actionAngle_ext: just compile the actionAngle extension (for use w/ --coverage)
# --interppotential_ext: just compile the interppotential extension (for use w/ --coverage)

pot_libraries = ['m','gsl','gslcblas','gomp']

if WIN32:  # windows does not need 'gomp' whether compiled with OpenMP or not
    pot_libraries.remove('gomp')

#Option to forego OpenMP
try:
    openmp_pos = sys.argv.index('--no-openmp')
except ValueError:
    extra_compile_args = ["-fopenmp" if not WIN32 else "/openmp"]
else:
    del sys.argv[openmp_pos]
    extra_compile_args= ["-DNO_OMP"]
    if not WIN32:  # Because windows guarantee do not have 'gomp' in the list
        pot_libraries.remove('gomp')

#Option to track coverage
try:
    coverage_pos = sys.argv.index('--coverage')
except ValueError:
    extra_link_args= []
else:
    del sys.argv[coverage_pos]
    extra_compile_args.extend(["-O0","--coverage"])
    extra_link_args= ["--coverage"]

#Option to compile everything into a single extension
try:
    single_ext_pos = sys.argv.index('--single_ext')
except ValueError:
    single_ext= False
else:
    del sys.argv[single_ext_pos]
    single_ext= True

#Option to just compile the orbit extension
try:
    orbit_ext_pos = sys.argv.index('--orbit_ext')
except ValueError:
    orbit_ext= False
else:
    del sys.argv[orbit_ext_pos]
    orbit_ext= True

#Option to just compile the actionAngle extension
try:
    actionAngle_ext_pos = sys.argv.index('--actionAngle_ext')
except ValueError:
    actionAngle_ext= False
else:
    del sys.argv[actionAngle_ext_pos]
    actionAngle_ext= True

#Option to just compile the interppotential extension
try:
    interppotential_ext_pos = sys.argv.index('--interppotential_ext')
except ValueError:
    interppotential_ext= False
else:
    del sys.argv[interppotential_ext_pos]
    interppotential_ext= True

#code to check the GSL version; list cmd w/ shell=True only works on Windows 
# (https://docs.python.org/3/library/subprocess.html#converting-argument-sequence)
cmd= ['gsl-config',
      '--version']
try:
    if sys.version_info < (2,7): #subprocess.check_output does not exist
        gsl_version= subprocess.Popen(cmd,shell=sys.platform.startswith('win'),
                                      stdout=subprocess.PIPE).communicate()[0]
    else:
        gsl_version= subprocess.check_output(cmd,shell=sys.platform.startswith('win'))
except (OSError,subprocess.CalledProcessError):
    gsl_version= ['0','0']
else:
    if PY3:
        gsl_version= gsl_version.decode('utf-8')
    gsl_version= gsl_version.split('.')
extra_compile_args.append("-D GSL_MAJOR_VERSION=%s" % (gsl_version[0]))

#HACK for testing
#gsl_version= ['0','0']

# MSVC: inline does not exist (not C99!); default = not necessarily actual, but will have to do for now...
if distutils.ccompiler.get_default_compiler().lower() == 'msvc':
    extra_compile_args.append("-Dinline=__inline")
    # only msvc compiler can be tested with initialize(), msvc is a default on windows
    # check for 'msvc' not WIN32, user can use other compiler like 'mingw32', in such case compiler exists for them
    try:
        test_compiler = distutils.ccompiler.new_compiler()
        test_compiler.initialize()  # try to initialize a test compiler to see if compiler presented
    except DistutilsPlatformError:  # this error will be raised if no compiler in the system
        no_compiler = True

# To properly export GSL symbols on Windows, need to defined GSL_DLL and WIN32
if WIN32:
    extra_compile_args.append("-DGSL_DLL")
    extra_compile_args.append("-DWIN32")

#Orbit integration C extension
orbit_int_c_src= ['galpy/util/bovy_symplecticode.c','galpy/util/bovy_rk.c']
orbit_int_c_src.extend(glob.glob('galpy/potential/potential_c_ext/*.c'))
orbit_int_c_src.extend(glob.glob('galpy/orbit/orbit_c_ext/*.c'))
orbit_int_c_src.extend(glob.glob('galpy/util/interp_2d/*.c'))

orbit_libraries=['m']
if float(gsl_version[0]) >= 1.:
    orbit_libraries.extend(['gsl','gslcblas'])

# On Windows it's unnecessary and erroneous to include m
if WIN32:
    orbit_libraries.remove('m')
    pot_libraries.remove('m')

orbit_include_dirs= ['galpy/util',
                     'galpy/util/interp_2d',
                     'galpy/orbit/orbit_c_ext',
                     'galpy/potential/potential_c_ext']

#actionAngleTorus C extension (files here, so we can compile a single extension if so desidered)
actionAngleTorus_c_src= \
    glob.glob('galpy/actionAngle/actionAngleTorus_c_ext/*.cc')
actionAngleTorus_c_src.extend(\
    glob.glob('galpy/actionAngle/actionAngleTorus_c_ext/torus/src/*.cc'))
actionAngleTorus_c_src.extend(\
    ['galpy/actionAngle/actionAngleTorus_c_ext/torus/src/utils/CHB.cc',
     'galpy/actionAngle/actionAngleTorus_c_ext/torus/src/utils/Err.cc',
     'galpy/actionAngle/actionAngleTorus_c_ext/torus/src/utils/Compress.cc',
     'galpy/actionAngle/actionAngleTorus_c_ext/torus/src/utils/Numerics.cc',
     'galpy/actionAngle/actionAngleTorus_c_ext/torus/src/utils/PJMNum.cc'])
actionAngleTorus_c_src.extend(\
    glob.glob('galpy/potential/potential_c_ext/*.c'))
actionAngleTorus_c_src.extend(\
    glob.glob('galpy/orbit/orbit_c_ext/integrateFullOrbit.c'))
actionAngleTorus_c_src.extend(glob.glob('galpy/util/interp_2d/*.c'))
actionAngleTorus_c_src.extend(glob.glob('galpy/util/*.c'))

actionAngleTorus_include_dirs= \
    ['galpy/actionAngle/actionAngleTorus_c_ext',
     'galpy/actionAngle/actionAngleTorus_c_ext/torus/src',
     'galpy/actionAngle/actionAngleTorus_c_ext/torus/src/utils',
     'galpy/actionAngle/actionAngle_c_ext',
     'galpy/util/interp_2d',
     'galpy/util',
     'galpy/orbit/orbit_c_ext',
     'galpy/potential/potential_c_ext']

if single_ext: #add the code and libraries for the other extensions
    #src
    orbit_int_c_src.extend(glob.glob('galpy/actionAngle/actionAngle_c_ext/*.c'))
    orbit_int_c_src.extend(glob.glob('galpy/potential/interppotential_c_ext/*.c'))
    if os.path.exists('galpy/actionAngle/actionAngleTorus_c_ext/torus/src'):
        # Add Torus code
        orbit_int_c_src.extend(actionAngleTorus_c_src)
        orbit_int_c_src= list(set(orbit_int_c_src))
    #libraries
    for lib in pot_libraries:
        if not lib in orbit_libraries:
            orbit_libraries.append(lib)
    #includes
    orbit_include_dirs.extend(['galpy/actionAngle/actionAngle_c_ext',
                               'galpy/util/interp_2d',
                               'galpy/orbit/orbit_c_ext',
                               'galpy/potential/potential_c_ext'])
    orbit_include_dirs.extend(['galpy/potential/potential_c_ext',
                               'galpy/util/interp_2d',
                               'galpy/util/',
                               'galpy/actionAngle/actionAngle_c_ext',
                               'galpy/orbit/orbit_c_ext',
                               'galpy/potential/interppotential_c_ext'])
    # Add Torus code
    orbit_include_dirs.extend(actionAngleTorus_include_dirs)
    orbit_include_dirs= list(set(orbit_include_dirs))
    
orbit_int_c= Extension('galpy_integrate_c',
                       sources=orbit_int_c_src,
                       libraries=orbit_libraries,
                       include_dirs=orbit_include_dirs,
                       extra_compile_args=extra_compile_args,
                       extra_link_args=extra_link_args)
ext_modules=[]
if float(gsl_version[0]) >= 1. and \
        not actionAngle_ext and not interppotential_ext:
    orbit_int_c_incl= True
    ext_modules.append(orbit_int_c)
else:
    orbit_int_c_incl= False

#actionAngle C extension
actionAngle_c_src= glob.glob('galpy/actionAngle/actionAngle_c_ext/*.c')
actionAngle_c_src.extend(glob.glob('galpy/potential/potential_c_ext/*.c'))
actionAngle_c_src.extend(glob.glob('galpy/util/interp_2d/*.c'))
actionAngle_c_src.extend(['galpy/util/bovy_symplecticode.c','galpy/util/bovy_rk.c'])
actionAngle_c_src.append('galpy/orbit/orbit_c_ext/integrateFullOrbit.c')
actionAngle_include_dirs= ['galpy/actionAngle/actionAngle_c_ext',
                           'galpy/orbit/orbit_c_ext',
                           'galpy/util/',
                           'galpy/util/interp_2d',
                           'galpy/potential/potential_c_ext']

#Installation of this extension using the GSL may (silently) fail, if the GSL
#is built for the wrong architecture, on Mac you can install the GSL correctly
#using
#brew install gsl --universal
actionAngle_c= Extension('galpy_actionAngle_c',
                         sources=actionAngle_c_src,
                         libraries=pot_libraries,
                         include_dirs=actionAngle_include_dirs,
                         extra_compile_args=extra_compile_args,
                         extra_link_args=extra_link_args)
if float(gsl_version[0]) >= 1. \
        and (float(gsl_version[0]) >= 2. or float(gsl_version[1]) >= 14.) and \
        not orbit_ext and not interppotential_ext and not single_ext:
    actionAngle_c_incl= True
    ext_modules.append(actionAngle_c)
else:
    actionAngle_c_incl= False
    
#interppotential C extension
interppotential_c_src= glob.glob('galpy/potential/potential_c_ext/*.c')
interppotential_c_src.extend(glob.glob('galpy/potential/interppotential_c_ext/*.c'))
interppotential_c_src.extend(['galpy/util/bovy_symplecticode.c','galpy/util/bovy_rk.c'])
interppotential_c_src.append('galpy/orbit/orbit_c_ext/integrateFullOrbit.c')
interppotential_c_src.extend(glob.glob('galpy/util/interp_2d/*.c'))

interppotential_include_dirs= ['galpy/potential/potential_c_ext',
                               'galpy/util/interp_2d',
                               'galpy/util/',
                               'galpy/actionAngle/actionAngle_c_ext',
                               'galpy/orbit/orbit_c_ext',
                               'galpy/potential/interppotential_c_ext']

interppotential_c= Extension('galpy_interppotential_c',
                             sources=interppotential_c_src,
                             libraries=pot_libraries,
                             include_dirs=interppotential_include_dirs,
                             extra_compile_args=extra_compile_args,
                             extra_link_args=extra_link_args)
if float(gsl_version[0]) >= 1. \
        and (float(gsl_version[0]) >= 2. or float(gsl_version[1]) >= 14.) \
        and not orbit_ext and not actionAngle_ext and not single_ext:
    interppotential_c_incl= True
    ext_modules.append(interppotential_c)
else:
    interppotential_c_incl= False

# Add the actionAngleTorus extension (src and include specified above)
#Installation of this extension using the GSL may (silently) fail, if the GSL
#is built for the wrong architecture, on Mac you can install the GSL correctly
#using
#brew install gsl --universal
actionAngleTorus_c= Extension('galpy_actionAngleTorus_c',
                              sources=actionAngleTorus_c_src,
                              libraries=pot_libraries,
                              include_dirs=actionAngleTorus_include_dirs,
                              extra_compile_args=extra_compile_args,
                              extra_link_args=extra_link_args)
if float(gsl_version[0]) >= 1. \
        and (float(gsl_version[0]) >= 2. or float(gsl_version[1]) >= 14.) and \
        os.path.exists('galpy/actionAngle/actionAngleTorus_c_ext/torus/src') and \
        not orbit_ext and not interppotential_ext and not single_ext:
    actionAngleTorus_c_incl= True
    ext_modules.append(actionAngleTorus_c)
else:
    actionAngleTorus_c_incl= False
    
setup(name='galpy',
      version='1.4.1',
      description='Galactic Dynamics in python',
      author='Jo Bovy',
      author_email='bovy@astro.utoronto.ca',
      license='New BSD',
      long_description=long_description,
      url='http://github.com/jobovy/galpy',
      package_dir = {'galpy/': ''},
      packages=['galpy','galpy/orbit','galpy/potential',
                'galpy/df','galpy/util','galpy/snapshot',
                'galpy/actionAngle'],
      package_data={'galpy/df':['data/*.sav'],
                    "": ["README.rst","README.dev","LICENSE","AUTHORS.rst"]},
      include_package_data=True,
      install_requires=['numpy>=1.7','scipy','matplotlib','pytest','six'],
      ext_modules=ext_modules if not no_compiler else None,
      classifiers=[
        "Development Status :: 6 - Mature",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
        "Programming Language :: C",
        "Programming Language :: Python :: 2.7",
        "Programming Language :: Python :: 3.3",
        "Programming Language :: Python :: 3.4",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Topic :: Scientific/Engineering :: Astronomy",
        "Topic :: Scientific/Engineering :: Physics"]
      )

def print_gsl_message(num_messages=1):
    if num_messages > 1:
        this_str= 'these installations'
    else:
        this_str= 'this installation'
    print('If you believe that %s should have worked, make sure\n(1) that the GSL include/ directory can be found by the compiler (you might have to edit CFLAGS for this: export CFLAGS="$CFLAGS -I/path/to/gsl/include/", or equivalent for C-type shells; replace /path/to/gsl/include/ with the actual path to the include directory),\n(2) that the GSL library can be found by the linker (you might have to edit LDFLAGS for this: export LDFLAGS="$LDFLAGS -L/path/to/gsl/lib/", or equivalent for C-type shells; replace /path/to/gsl/lib/ with the actual path to the lib directory),\n(3) and that `gsl-config --version` returns the correct version' % this_str)

num_gsl_warn= 0
if not orbit_int_c_incl:
    num_gsl_warn+= 1
    print('\033[91;1m'+'WARNING: orbit-integration C library not installed because your GSL version < 1'+'\033[0m')

if not actionAngle_c_incl and not single_ext:
    num_gsl_warn+= 1
    print('\033[91;1m'+'WARNING: action-angle C library not installed because your GSL version < 1.14'+'\033[0m')
if not interppotential_c_incl and not single_ext:
    num_gsl_warn+= 1
    print('\033[91;1m'+'WARNING: Potential-interpolation C library not installed because your GSL version < 1.14'+'\033[0m')
if not actionAngleTorus_c_incl and not single_ext:
    num_gsl_warn+= 1
    print('\033[91;1m'+'WARNING: action-angle-torus C library not installed because your GSL version < 1.14 or because you did not first download the torus code as explained in the installation guide in the html documentation'+'\033[0m')

if num_gsl_warn > 0:
    print_gsl_message(num_messages=num_gsl_warn)
    print('\033[1m'+'These warning messages about the C code do not mean that the python package was not installed successfully'+'\033[0m')
print('\033[1m'+'Finished installing galpy'+'\033[0m')
print('You can run the test suite using `pytest -v tests/` to check the installation (but note that the test suite currently takes about 50 minutes to run)')

#if single_ext, symlink the other (non-compiled) extensions to galpy_integrate_c.so (use EXT_SUFFIX for python3 compatibility)
if PY3:
    _ext_suffix= sysconfig.get_config_var('EXT_SUFFIX')
else:
    _ext_suffix= '.so'
if single_ext:
    if not os.path.exists('galpy_actionAngle_c%s' % _ext_suffix):
        os.symlink('galpy_integrate_c%s' % _ext_suffix,
                   'galpy_actionAngle_c%s' % _ext_suffix)
    if not os.path.exists('galpy_interppotential_c%s' % _ext_suffix):
        os.symlink('galpy_integrate_c%s' % _ext_suffix,
                   'galpy_interppotential_c%s' % _ext_suffix)
    if not os.path.exists('galpy_actionAngleTorus_c%s' % _ext_suffix) \
            and os.path.exists('galpy/actionAngle/actionAngleTorus_c_ext/torus/src'):
        os.symlink('galpy_integrate_c%s' % _ext_suffix,
                   'galpy_actionAngleTorus_c%s' % _ext_suffix)
