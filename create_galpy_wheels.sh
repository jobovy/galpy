#!/bin/bash
# Simple script to create galpy wheels for a bunch of different
# Python version locally; useful for creating ARM64 wheels, because
# not possible with CIs currently
source ~/.bash_profile
PYTHON_VERSIONS=("3.8" "3.9" "3.10")
GALPY_VERSION=1.8.0

rm -rf galpy-wheels-output
mkdir galpy-wheels-output
# Loop over the entire thing to make sure nothing gets reused
for PYTHON_VERSION in "${PYTHON_VERSIONS[@]}"; do
    git clone https://github.com/jobovy/galpy.git galpy-wheels
    cd galpy-wheels
    git checkout v$GALPY_VERSION
    mkdir wheelhouse
    conda activate base
    conda create -y --name galpywheels"$PYTHON_VERSION" python="$PYTHON_VERSION"
    conda env update --name galpywheels"$PYTHON_VERSION" --file .github/conda-build-environment-macos-latest.yml --prune
    conda activate galpywheels"$PYTHON_VERSION"
    pip install build
    CFLAGS="$CFLAGS -I$CONDA_PREFIX/include"
    LDFLAGS="$LDFLAGS -L$CONDA_PREFIX/lib"
    LD_LIBRARY_PATH="$LD_LIBRARY_PATH -L$CONDA_PREFIX/lib"
    python -m build --wheel --outdir wheelhouse
    if [[ "$OSTYPE" == "darwin"* ]]; then
        python -m pip install delocate
        delocate-wheel -v wheelhouse/*
    fi
    mv wheelhouse/* ../galpy-wheels-output
    conda activate base
    conda remove -y --name galpywheels"$PYTHON_VERSION" --all
    cd ../
    rm -rf galpy-wheels
done
