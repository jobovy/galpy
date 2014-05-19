#!/bin/bash
if [[ "$CC" == "clang" ]]; then
    pip install cpp-coveralls
    cpp-coveralls
else
    pip install coveralls
    coveralls
fi
