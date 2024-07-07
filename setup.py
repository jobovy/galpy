import glob
import os
import os.path
import platform
import subprocess
import sys
import sysconfig

from setuptools import Extension, find_namespace_packages, setup
from setuptools.command.build_ext import build_ext
from setuptools.errors import PlatformError

PY3 = sys.version > "3"
WIN32 = platform.system() == "Windows"
no_compiler = False  # Flag for cases where we are sure there is no compiler exists in user's system

long_description = ""
previous_line = ""
with open("README.md") as dfile:
    for line in dfile:
        if (
            not "image" in line
            and not "target" in line
            and not "DETAILED" in line
            and not "**main**" in line
            and not "**development" in line
            and not "DETAILED" in previous_line
        ):
            long_description += line
        previous_line = line

# Parse options; current options
# --no-openmp: compile without OpenMP support
# --coverage: compile with gcov support
# --compiler= set the compiler by hand
# --single_ext: compile all of the C code into a single extension (just for testing, do not use this)

galpy_c_libraries = ["m", "gsl", "gslcblas", "gomp"]

if WIN32:
    # On Windows it's unnecessary and erroneous to include m
    galpy_c_libraries.remove("m")
    # Windows does not need 'gomp' whether compiled with OpenMP or not
    galpy_c_libraries.remove("gomp")

# Option to forego OpenMP
try:
    openmp_pos = sys.argv.index("--no-openmp")
except ValueError:
    if "PYODIDE" in os.environ:
        extra_compile_args = ["-DNO_OMP"]
        galpy_c_libraries.remove("gomp")
    else:
        extra_compile_args = ["-fopenmp" if not WIN32 else "/openmp"]
else:
    del sys.argv[openmp_pos]
    extra_compile_args = ["-DNO_OMP"]
    if not WIN32:  # Because windows guarantee do not have 'gomp' in the list
        galpy_c_libraries.remove("gomp")

# Option to track coverage
try:
    coverage_pos = sys.argv.index("--coverage")
except ValueError:
    extra_link_args = []
else:
    del sys.argv[coverage_pos]
    extra_compile_args.extend(["-O0", "--coverage", "-D USING_COVERAGE"])
    extra_link_args = ["--coverage"]

# Option to compile everything into a single extension
try:
    single_ext_pos = sys.argv.index("--single_ext")
except ValueError:
    single_ext = False
else:
    del sys.argv[single_ext_pos]
    single_ext = True

# Option to not compile any extension
try:
    no_ext_pos = sys.argv.index("--no_ext")
except ValueError:
    no_ext = False
else:
    del sys.argv[no_ext_pos]
    no_ext = True

# code to check the GSL version; list cmd w/ shell=True only works on Windows
# (https://docs.python.org/3/library/subprocess.html#converting-argument-sequence)
cmd = ["gsl-config", "--version"]
try:
    if sys.version_info < (2, 7):  # subprocess.check_output does not exist
        gsl_version = subprocess.Popen(
            cmd, shell=sys.platform.startswith("win"), stdout=subprocess.PIPE
        ).communicate()[0]
    else:
        gsl_version = subprocess.check_output(cmd, shell=sys.platform.startswith("win"))
except (OSError, subprocess.CalledProcessError):
    if "PYODIDE" in os.environ:
        gsl_version = ["2", "7"]
    else:
        gsl_version = ["0", "0"]
else:
    if PY3:
        gsl_version = gsl_version.decode("utf-8")
    gsl_version = gsl_version.split(".")
extra_compile_args.append("-D GSL_MAJOR_VERSION=%s" % (gsl_version[0]))

# HACK for testing
# gsl_version= ['0','0']

# To properly export GSL symbols on Windows, need to defined GSL_DLL and WIN32
if WIN32:
    extra_compile_args.append("-DGSL_DLL")
    extra_compile_args.append("-DWIN32")

# main C extension
galpy_c_src = [
    "galpy/util/bovy_symplecticode.c",
    "galpy/util/bovy_rk.c",
    "galpy/util/leung_dop853.c",
    "galpy/util/bovy_coords.c",
]
galpy_c_src.extend(glob.glob("galpy/potential/potential_c_ext/*.c"))
galpy_c_src.extend(glob.glob("galpy/potential/interppotential_c_ext/*.c"))
galpy_c_src.extend(glob.glob("galpy/util/interp_2d/*.c"))
galpy_c_src.extend(glob.glob("galpy/orbit/orbit_c_ext/*.c"))
galpy_c_src.extend(glob.glob("galpy/actionAngle/actionAngle_c_ext/*.c"))

galpy_c_include_dirs = [
    "galpy/util",
    "galpy/util/interp_2d",
    "galpy/potential/potential_c_ext",
    "galpy/potential/interppotential_c_ext",
    "galpy/orbit/orbit_c_ext",
    "galpy/actionAngle/actionAngle_c_ext",
]

# actionAngleTorus C extension (files here, so we can compile a single extension if so desidered)
actionAngleTorus_c_src = glob.glob("galpy/actionAngle/actionAngleTorus_c_ext/*.cc")
actionAngleTorus_c_src.extend(
    glob.glob("galpy/actionAngle/actionAngleTorus_c_ext/torus/src/*.cc")
)
actionAngleTorus_c_src.extend(
    [
        "galpy/actionAngle/actionAngleTorus_c_ext/torus/src/utils/CHB.cc",
        "galpy/actionAngle/actionAngleTorus_c_ext/torus/src/utils/Err.cc",
        "galpy/actionAngle/actionAngleTorus_c_ext/torus/src/utils/Compress.cc",
        "galpy/actionAngle/actionAngleTorus_c_ext/torus/src/utils/Numerics.cc",
        "galpy/actionAngle/actionAngleTorus_c_ext/torus/src/utils/PJMNum.cc",
    ]
)
actionAngleTorus_c_src.extend(glob.glob("galpy/potential/potential_c_ext/*.c"))
actionAngleTorus_c_src.extend(glob.glob("galpy/orbit/orbit_c_ext/integrateFullOrbit.c"))
actionAngleTorus_c_src.extend(glob.glob("galpy/util/interp_2d/*.c"))
actionAngleTorus_c_src.extend(glob.glob("galpy/util/*.c"))

actionAngleTorus_include_dirs = [
    "galpy/actionAngle/actionAngleTorus_c_ext",
    "galpy/actionAngle/actionAngleTorus_c_ext/torus/src",
    "galpy/actionAngle/actionAngleTorus_c_ext/torus/src/utils",
    "galpy/actionAngle/actionAngle_c_ext",
    "galpy/util/interp_2d",
    "galpy/util",
    "galpy/orbit/orbit_c_ext",
    "galpy/potential/potential_c_ext",
]

if single_ext:  # add the code and libraries for the actionAngleTorus extensions
    if os.path.exists("galpy/actionAngle/actionAngleTorus_c_ext/torus/src"):
        galpy_c_src.extend(actionAngleTorus_c_src)
        galpy_c_src = list(set(galpy_c_src))
        galpy_c_include_dirs.extend(actionAngleTorus_include_dirs)
        galpy_c_include_dirs = list(set(galpy_c_include_dirs))

# Installation of this extension using the GSL may (silently) fail, if the GSL
# is built for the wrong architecture, on Mac you can install the GSL correctly
# using
# brew install gsl --universal
galpy_c = Extension(
    "libgalpy",
    sources=galpy_c_src,
    libraries=galpy_c_libraries,
    include_dirs=galpy_c_include_dirs,
    extra_compile_args=extra_compile_args,
    extra_link_args=extra_link_args,
)
ext_modules = []
if float(gsl_version[0]) >= 1.0 and (
    float(gsl_version[0]) >= 2.0 or float(gsl_version[1]) >= 14.0
):
    galpy_c_incl = True
    ext_modules.append(galpy_c)
else:
    galpy_c_incl = False

# Add the actionAngleTorus extension (src and include specified above)
actionAngleTorus_c = Extension(
    "libgalpy_actionAngleTorus",
    sources=actionAngleTorus_c_src,
    libraries=galpy_c_libraries,
    include_dirs=actionAngleTorus_include_dirs,
    extra_compile_args=extra_compile_args,
    extra_link_args=extra_link_args,
)
if (
    float(gsl_version[0]) >= 1.0
    and (float(gsl_version[0]) >= 2.0 or float(gsl_version[1]) >= 14.0)
    and os.path.exists("galpy/actionAngle/actionAngleTorus_c_ext/torus/src")
    and not single_ext
):
    actionAngleTorus_c_incl = True
    ext_modules.append(actionAngleTorus_c)
else:
    actionAngleTorus_c_incl = False


# Test whether compiler allows for the -fopenmp flag and all other flags
# to guard against compilation errors
# (https://stackoverflow.com/a/54518348)
def compiler_has_flag(compiler, flagname):
    """Test whether a given compiler supports a given option"""
    import tempfile

    from setuptools.errors import CompileError

    with tempfile.NamedTemporaryFile("w", suffix=".cpp") as f:
        f.write("int main (int argc, char **argv) { return 0; }")
        try:
            compiler.compile([f.name], extra_postargs=[flagname])
        except CompileError:
            return False
    return True


# Test whether the compiler is clang, allowing for the fact that it's name might be gcc...
def compiler_is_clang(compiler):
    # Test whether the compiler is clang by running the compiler with the --version flag and checking whether the output contains "clang"
    try:
        output = subprocess.check_output(
            [compiler.compiler[0], "--version"], stderr=subprocess.STDOUT
        )
    except (OSError, subprocess.CalledProcessError):
        return False
    return b"clang" in output


# Now need to subclass BuildExt to access the compiler used and check flags
class BuildExt(build_ext):
    def build_extensions(self):
        ct = self.compiler.compiler_type
        if ct == "unix":
            for ext in self.extensions:
                # only add flags which pass the flag_filter
                extra_compile_args = []
                libraries = ext.libraries
                for flag in set(ext.extra_compile_args):
                    if compiler_has_flag(self.compiler, flag):
                        extra_compile_args.append(flag)
                    elif compiler_is_clang(self.compiler) and flag == "-fopenmp":
                        # clang does not support -fopenmp, but does support -Xclang -fopenmp
                        extra_compile_args.append("-Xclang")
                        extra_compile_args.append("-fopenmp")
                        # Also adjust libraries as needed
                        if "gomp" in libraries:
                            libraries.remove("gomp")
                        if "omp" not in libraries:
                            libraries.append("omp")
                    elif flag == "-fopenmp" and "gomp" in libraries:
                        libraries.remove("gomp")
                ext.extra_compile_args = extra_compile_args
                ext.libraries = libraries
        build_ext.build_extensions(self)


setup(
    cmdclass=dict(build_ext=BuildExt),  # this to allow compiler check above
    name="galpy",
    version="1.10.0",
    description="Galactic Dynamics in python",
    author="Jo Bovy",
    author_email="bovy@astro.utoronto.ca",
    license="New BSD",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="http://github.com/jobovy/galpy",
    packages=find_namespace_packages(where=".", include=["galpy*"]),
    package_data={
        "galpy/orbit": ["named_objects.json"],
        "galpy/df": ["data/*.sav"],
        "": ["README.md", "README.dev", "LICENSE", "AUTHORS.rst"],
    },
    include_package_data=True,
    python_requires=">=3.8",
    install_requires=["packaging", "numpy>=1.7", "scipy", "matplotlib"],
    extras_require={
        "docs": ["sphinxext-opengraph", "sphinx-design", "markupsafe==2.0.1"]
    },
    ext_modules=ext_modules if not no_compiler and not no_ext else None,
    classifiers=[
        "Development Status :: 6 - Mature",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
        "Programming Language :: C",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Environment :: WebAssembly :: Emscripten",
        "Topic :: Scientific/Engineering :: Astronomy",
        "Topic :: Scientific/Engineering :: Physics",
    ],
)


def print_gsl_message(num_messages=1):
    if num_messages > 1:
        this_str = "these installations"
    else:
        this_str = "this installation"
    print(
        'If you believe that %s should have worked, make sure\n(1) that the GSL include/ directory can be found by the compiler (you might have to edit CFLAGS for this: export CFLAGS="$CFLAGS -I/path/to/gsl/include/", or equivalent for C-type shells; replace /path/to/gsl/include/ with the actual path to the include directory),\n(2) that the GSL library can be found by the linker (you might have to edit LDFLAGS for this: export LDFLAGS="$LDFLAGS -L/path/to/gsl/lib/", or equivalent for C-type shells; replace /path/to/gsl/lib/ with the actual path to the lib directory),\n(3) and that `gsl-config --version` returns the correct version'
        % this_str
    )


num_gsl_warn = 0
if not galpy_c_incl:
    num_gsl_warn += 1
    print(
        "\033[91;1m"
        + "WARNING: galpy C library not installed because your GSL version < 1.14"
        + "\033[0m"
    )

if not actionAngleTorus_c_incl and not single_ext:
    num_gsl_warn += 1
    print(
        "\033[91;1m"
        + "WARNING: galpy action-angle-torus C library not installed because your GSL version < 1.14 or because you did not first download the torus code as explained in the installation guide in the html documentation"
        + "\033[0m"
    )

if num_gsl_warn > 0:
    print_gsl_message(num_messages=num_gsl_warn)
    print(
        "\033[1m"
        + "These warning messages about the C code do not mean that the python package was not installed successfully"
        + "\033[0m"
    )
print("\033[1m" + "Finished installing galpy" + "\033[0m")
print(
    "You can run the test suite using `pytest -v tests/` to check the installation (but note that the test suite currently takes about 50 minutes to run)"
)

# if single_ext, symlink the other (non-compiled) extensions to libgalpy.so (use EXT_SUFFIX for python3 compatibility)
if PY3:
    _ext_suffix = sysconfig.get_config_var("EXT_SUFFIX")
else:
    _ext_suffix = ".so"
if single_ext:
    if not os.path.exists(
        "libgalpy_actionAngleTorus%s" % _ext_suffix
    ) and os.path.exists("galpy/actionAngle/actionAngleTorus_c_ext/torus/src"):
        os.symlink(
            "libgalpy%s" % _ext_suffix, "libgalpy_actionAngleTorus%s" % _ext_suffix
        )
