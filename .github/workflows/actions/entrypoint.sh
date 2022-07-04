#!/bin/sh

yum -y install gsl-devel

PYTHON_VERSIONS=("cp37-cp37m" "cp38-cp38" "cp39-cp39" "cp310-cp310")

for PYTHON_VERSION in ${PYTHON_VERSIONS[@]}; do
    /opt/python/${PYTHON_VERSION}/bin/pip install --upgrade pip
    /opt/python/${PYTHON_VERSION}/bin/pip install -U build auditwheel
    /opt/python/${PYTHON_VERSION}/bin/python -m build --wheel --outdir wheelhouse
done

for whl in /github/workspace/wheelhouse/*.whl; do
    auditwheel repair $whl
    rm $whl
done
