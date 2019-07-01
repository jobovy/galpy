#############################################################################
# Setup Intel Compiler
# Please notice it only supports Visual Studio 2015 or later
#############################################################################
import platform

from distutils import ccompiler
from distutils.ccompiler import *
if platform.system() == 'Windows':  # to prevent linux import error
    from distutils._msvccompiler import _find_exe
    from distutils._msvccompiler import MSVCCompiler
else:
    raise OSError("You are not using Windows, so you should not import this file")


class Intel64CompilerW(MSVCCompiler):
    """
    A modified Intel compiler compatible with an MSVC-built Python.
    Only need to define the Windows one because Linux one is imported from distutils
    """
    compiler_type = 'intel64w'
    compiler_cxx = 'icl'

    def __init__(self, verbose=0, dry_run=0, force=0):
        MSVCCompiler.__init__(self, verbose, dry_run, force)

    def initialize(self, plat_name=None):
        MSVCCompiler.initialize(self)
        self.cc = _find_exe("icl.exe")
        self.lib = _find_exe("xilib.exe")
        self.linker = _find_exe("xilink.exe")
        self.compile_options = ['/nologo', '/O3', '/MD', '/W3',
                                '/Qstd=c99']
        self.compile_options_debug = ['/nologo', '/Od', '/MDd', '/W3',
                                      '/Qstd=c99', '/Z7', '/D_DEBUG']


compiler_class['intel64w'] = ('__intelcompiler', 'Intel64CompilerW',
                              "Intel C Compiler for 64-bit applications on Windows")

ccompiler._default_compilers += ('nt', 'intel64w')

_distutils_new_compiler = new_compiler


def new_compiler(plat=os.name,
                 compiler=None,
                 verbose=0,
                 dry_run=0,
                 force=0):
    """
    make new compiler and pass it to distutils

    dont delete verbose even not using it, distutils expects it
    """
    try:
        if compiler is None:
            compiler = get_default_compiler(plat)
        (module_name, class_name, long_description) = compiler_class[compiler]
    except KeyError:
        msg = "don't know how to compile C/C++ code on platform '%s'" % plat
        if compiler is not None:
            msg = msg + " with '%s' compiler" % compiler
        raise DistutilsPlatformError(msg)
    try:
        __import__ (module_name)
    except ImportError:
        # go back to default compiler
        __import__("distutils." + module_name)
        module_name = "distutils." + module_name
    module = sys.modules[module_name]
    klass = vars(module)[class_name]
    # dryrun compiler
    compiler = klass(None, dry_run, force)
    return compiler


ccompiler.new_compiler = new_compiler
