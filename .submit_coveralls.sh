#!/bin/bash
if [[ "$CC" == "clang" ]]; then
    pip install cpp-coveralls
    cpp-coveralls
else
    coveralls
fi
