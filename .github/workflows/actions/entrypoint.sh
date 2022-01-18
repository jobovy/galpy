#!/bin/sh

yum -y install gsl-devel

PYTHON_VERSIONS=("cp35-cp35m" "cp36-cp36m" "cp37-cp37m" "cp38-cp38" "cp39-cp39" "cp310-cp310")

for PYTHON_VERSION in ${PYTHON_VERSIONS[@]}; do
    /opt/python/${PYTHON_VERSION}/bin/pip install --upgrade pip
    /opt/python/${PYTHON_VERSION}/bin/pip install -U wheel auditwheel
    /opt/python/${PYTHON_VERSION}/bin/python setup.py bdist_wheel -d wheelhouse
done

for whl in /github/workspace/wheelhouse/*.whl; do
    auditwheel repair $whl
    rm $whl
done
