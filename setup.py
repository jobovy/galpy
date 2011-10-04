from setuptools import setup
from distutils.core import Extension
import os, os.path
import glob

longDescription= ""

orbit_int_c_src= ['galpy/util/bovy_symplecticode.c']
orbit_int_c_src.extend(glob.glob('galpy/potential_src/potential_c_ext/*.c'))
orbit_int_c_src.extend(glob.glob('galpy/orbit_src/orbit_c_ext/*.c'))

orbit_int_c= Extension('galpy_integrate_c',
                       sources=orbit_int_c_src,
                       libraries=['m'],
                       include_dirs=['galpy/util',
                                     'galpy/potential_src/potential_c_ext'])

setup(name='galpy',
      version='1.',
      description='Galactic Dynamics in python',
      author='Jo Bovy',
      author_email='bovy@ias.edu',
      license='New BSD',
      long_description=longDescription,
      url='https://github.com/jobovy/galpy',
      package_dir = {'galpy/': ''},
      packages=['galpy'],
#      dependency_links = ['https://github.com/dfm/MarkovPy/tarball/master#egg=MarkovPy'],
      install_requires=['numpy','scipy','matplotlib'],
      ext_modules=[orbit_int_c]
      )
