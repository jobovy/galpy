#!/bin/bash
# rm old build
rm -rf build/
rm -vf galpy_integrate_c.so galpy_actionAngle_c.so galpy_interppotential_c.so
# Test the orbit extension
python setup.py build_ext --coverage --orbit_ext --inplace
python setup.py develop --coverage --orbit_ext
lcov --capture --directory build/temp.linux-x86_64-2.7/galpy/ --output-file coverage_orbit_initial.info --initial
/usr/local/python-2.7.3/bin/nosetests -v -w nose/ -e test_plotting
lcov --capture --directory build/temp.linux-x86_64-2.7/galpy/ --output-file coverage_orbit.info
rm -rf build/
rm -vf galpy_integrate_c.so galpy_actionAngle_c.so galpy_interppotential_c.so
# Test the actionAngle extension
python setup.py build_ext --coverage --actionAngle_ext --inplace
python setup.py develop --coverage --actionAngle_ext
lcov --capture --directory build/temp.linux-x86_64-2.7/galpy/ --output-file coverage_actionAngle_initial.info --initial
/usr/local/python-2.7.3/bin/nosetests -v -w nose/ -e test_plotting
lcov --capture --directory build/temp.linux-x86_64-2.7/galpy/ --output-file coverage_actionAngle.info
rm -rf build/
rm -vf galpy_integrate_c.so galpy_actionAngle_c.so galpy_interppotential_c.so
# Test the interppotential extension
python setup.py build_ext --coverage --interppotential_ext --inplace
python setup.py develop --coverage --interppotential_ext
lcov --capture --directory build/temp.linux-x86_64-2.7/galpy/ --output-file coverage_interppotential_initial.info --initial
/usr/local/python-2.7.3/bin/nosetests -v -w nose/ -e test_plotting
lcov --capture --directory build/temp.linux-x86_64-2.7/galpy/ --output-file coverage_interppotential.info
# Combine
lcov --add-tracefile coverage_orbit.info --output-file dummy.info
if [ "$?" -ne "0" ]; then
    cp coverage_orbit_initial.info coverage_orbit.info
fi
lcov --add-tracefile coverage_actionAngle.info --output-file dummy.info
if [ "$?" -ne "0" ]; then
    cp coverage_actionAngle_initial.info coverage_actionAngle.info
fi
lcov --add-tracefile coverage_interppotential.info --output-file dummy.info
if [ "$?" -ne "0" ]; then
    cp coverage_interppotential_initial.info coverage_interppotential.info
fi
lcov --add-tracefile coverage_orbit.info --add-tracefile coverage_actionAngle.info --add-tracefile coverage_interppotential.info -output-file coverage.info
# Generate HTML
genhtml coverage.info --output-directory ~/public_html/galpy_lcov/
rm -vf dummy.info
rm -vf coverage_orbit.info coverage_actionAngle.info coverage_interppotential.info
rm -vf coverage_orbit_initial.info coverage_actionAngle_initial.info coverage_interppotential_initial.info
rm -vf coverage.info