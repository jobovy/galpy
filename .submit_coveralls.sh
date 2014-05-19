#!/bin/bash
if [[ "$CC" == "gcc" ]]; then
    cpp-coveralls
else
    coveralls
fi
