from setuptools import setup #, Extension
import os, os.path
import re

longDescription= ""


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
      install_requires=['numpy','scipy','matplotlib']
      )
