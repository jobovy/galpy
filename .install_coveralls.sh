#~/bin/bash
if [[ "$CC" == "gcc" ]]; then
    pip install cpp-coveralls
else
    pip install coveralls
fi
