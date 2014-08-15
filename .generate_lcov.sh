#!/bin/bash
# rm old build
rm -rf build/
rm -vf galpy_integrate_c.so galpy_actionAngle_c.so galpy_interppotential_c.so
# Test the extensions
$PYTHON setup.py build_ext --coverage --single_ext --inplace
$PYTHON setup.py develop --coverage --single_ext --prefix=~/local
/usr/local/python-2.7.3/bin/nosetests -v -w nose/ -e plotting
lcov --capture --directory build/temp.linux-x86_64-2.7/galpy/ --output-file coverage.info
# Generate HTML
genhtml coverage.info --output-directory ~/public_html/galpy_lcov/
rm -vf coverage.info
