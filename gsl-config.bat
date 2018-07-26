@echo off
rem simple Windows version of gsl-config for conda installs
set ROOT=%CONDA_PREFIX%\Library

set XCFLAGS=-I"%ROOT%\include"
set XLIBS1=-L"%ROOT%\lib" -lgsl -lgslcblas
set XLIBS2=-L"%ROOT%\lib" -lgsl
set XVERSION=2.3
set XPREFIX=%ROOT%

for %%p in (%*) do (
    if x%%p == x--cflags echo %XCFLAGS%
    if x%%p == x--libs echo %XLIBS1%
    if x%%p == x--libs-without-cblas echo %XLIBS2%
    if x%%p == x--version echo %XVERSION%
    if x%%p == x--prefix echo %XPREFIX%
)
