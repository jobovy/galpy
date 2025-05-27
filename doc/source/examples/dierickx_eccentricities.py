import os, os.path
import subprocess
from astropy.io import fits, ascii
from astropy import units
import numpy as np
from optparse import OptionParser
import sys
import tempfile
from ftplib import FTP
import shutil
import pickle
from tqdm import tqdm
from galpy.potential import LogarithmicHaloPotential
from galpy.potential import evaluatePotentials as evalPot
from galpy.orbit import Orbit
from galpy.actionAngle import estimateDeltaStaeckel, actionAngleStaeckel, UnboundError
from galpy.util import coords
import matplotlib.pyplot as plt
import numpy

_ERASESTR = (
    "                                                                                "
)


def calc_eccentricity(args, options):
    table = os.path.join(args[0], "table2.dat")
    readme = os.path.join(args[0], "ReadMe")
    dierickx = ascii.read(table, readme=readme)
    vxvv = np.dstack(
        [
            dierickx["RAdeg"],
            dierickx["DEdeg"],
            dierickx["Dist"] / 1e3,
            dierickx["pmRA"],
            dierickx["pmDE"],
            dierickx["HRV"],
        ]
    )[0]
    ro, vo, zo = 8.0, 220.0, 0.025
    orbits = Orbit(vxvv, radec=True, ro=ro, vo=vo, zo=zo, solarmotion="hogg")
    # do the integration and individual analytic estimate for each object
    lp = LogarithmicHaloPotential(normalize=1.0)
    e_ana = orbits.e(analytic=True, pot=lp, delta=1e-6)
    ts = np.linspace(0.0, 20.0, 10000)
    orbits.integrate(ts, lp)
    e_int = orbits.e()
    # Now plot everything
    fig = plt.figure()
    plt.scatter(e_int, e_ana, s=1, color="Black", lw=0.0)
    plt.xlabel(r"$\mathrm{galpy\ integrated}\ e$")
    plt.ylabel(r"$\mathrm{galpy\ analytic}\ e$")
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.0)
    fig.tight_layout()
    plt.savefig(
        os.path.join(args[0], "dierickx-integratedeanalytice.png"),
        format="png",
        dpi=200,
    )
    fig = plt.figure()
    plt.hist(e_int, bins=30)
    plt.xlim(0.0, 1.0)
    plt.xlabel(r"$\mathrm{galpy}\ e$")
    fig.tight_layout()
    plt.savefig(
        os.path.join(args[0], "dierickx-integratedehist.png"), format="png", dpi=200
    )
    fig = plt.figure()
    plt.scatter(dierickx["e"], e_int, s=1, color="Black", lw=0.0)
    plt.xlabel(r"$\mathrm{Dierickx\ et\ al.}\ e$")
    plt.ylabel(r"$\mathrm{galpy\ integrated}\ e$")
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.0)
    fig.tight_layout()
    plt.savefig(
        os.path.join(args[0], "dierickx-integratedee.png"), format="png", dpi=200
    )
    fig = plt.figure()
    plt.scatter(dierickx["e"], e_ana, s=1, color="Black", lw=0.0)
    plt.xlabel(r"$\mathrm{Dierickx\ et\ al.}\ e$")
    plt.ylabel(r"$\mathrm{galpy\ estimated}\ e$")
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.0)
    fig.tight_layout()
    plt.savefig(os.path.join(args[0], "dierickx-analyticee.png"), format="png", dpi=200)
    arr = numpy.recarray(
        len(e_ana), dtype=[("analytic_e", float), ("integrated_e", float)]
    )
    arr["analytic_e"] = e_ana
    arr["integrated_e"] = e_int
    with open(os.path.join(args[0], "eccentricities.dat"), "w") as file:
        pickle.dump(arr, file)
        file.close()


def get_table(args, options):
    cat = "J/ApJ/725/L186/"
    tab2name = "table2.dat.gz"
    tab2readme = "ReadMe"
    out = args[0]
    ensure_dir(os.path.join(out, tab2name))
    vizier(
        cat,
        os.path.join(out, tab2name),
        os.path.join(out, tab2readme),
        catalogname=tab2name,
        readmename=tab2readme,
    )
    subprocess.call(["gunzip", os.path.join(out, tab2name)])


def vizier(cat, filePath, ReadMePath, catalogname="catalog.dat", readmename="ReadMe"):
    """
    NAME:
       vizier
    PURPOSE:
       download a catalog and its associated ReadMe from Vizier
    INPUT:
       cat - name of the catalog (e.g., 'III/272' for RAVE, or J/A+A/... for journal-specific catalogs)
       filePath - path of the file where you want to store the catalog (note: you need to keep the name of the file the same as the catalogname to be able to read the file with astropy.io.ascii)
       ReadMePath - path of the file where you want to store the ReadMe file
       catalogname= (catalog.dat) name of the catalog on the Vizier server
       readmename= (ReadMe) name of the ReadMe file on the Vizier server
    OUTPUT:
       (nothing, just downloads)
    HISTORY:
       2016-09-12 - Written - Bovy (UofT)
    """
    _download_file_vizier(cat, filePath, catalogname=catalogname)
    _download_file_vizier(cat, ReadMePath, catalogname=readmename)
    return None


def _download_file_vizier(cat, filePath, catalogname="catalog.dat"):
    """
    Stolen from Jo Bovy's gaia_tools package!
    """
    sys.stdout.write("\r" + "Downloading file %s ...\r" % (os.path.basename(filePath)))
    sys.stdout.flush()
    try:
        # make all intermediate directories
        os.makedirs(os.path.dirname(filePath))
    except OSError:
        pass
    # Safe way of downloading
    downloading = True
    interrupted = False
    file, tmp_savefilename = tempfile.mkstemp()
    os.close(file)  # Easier this way
    ntries = 1
    while downloading:
        try:
            ftp = FTP("cdsarc.u-strasbg.fr")
            ftp.login("anonymous", "test")
            ftp.cwd(os.path.join("pub", "cats", cat))
            with open(tmp_savefilename, "wb") as savefile:
                ftp.retrbinary("RETR %s" % catalogname, savefile.write)
            shutil.move(tmp_savefilename, filePath)
            downloading = False
            if interrupted:
                raise KeyboardInterrupt
        except:
            raise
            if not downloading:  # Assume KeyboardInterrupt
                raise
            elif ntries > _MAX_NTRIES:
                raise IOError(
                    "File %s does not appear to exist on the server ..."
                    % (os.path.basename(filePath))
                )
        finally:
            if os.path.exists(tmp_savefilename):
                os.remove(tmp_savefilename)
        ntries += 1
    sys.stdout.write("\r" + _ERASESTR + "\r")
    sys.stdout.flush()
    return None


def ensure_dir(f):
    """Ensure a a file exists and if not make the relevant path"""
    d = os.path.dirname(f)
    if not os.path.exists(d):
        os.makedirs(d)


def get_options():
    # no options yet - probably none needed?
    usage = "usage: %prog [options] <outpath>"
    parser = OptionParser(usage=usage)
    return parser


if __name__ == "__main__":
    parser = get_options()
    options, args = parser.parse_args()
    get_table(args, options)
    calc_eccentricity(args, options)
