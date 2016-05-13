import os, os.path
try:
    import configparser
except:
    from six.moves import configparser
_APY_LOADED= True
try:
    from astropy import units
except ImportError:
    _APY_LOADED= False
# The default configuration
default_configuration= {'astropy-units':'False',
                        'astropy-coords':'True',
                        'ro':'8.',
                        'vo':'220.'}
default_filename= os.path.join(os.path.expanduser('~'),'.galpyrc')
def write_default(filename):
    writeconfig= configparser.ConfigParser()
    # Write different sections
    writeconfig.add_section('normalization')
    writeconfig.set('normalization','ro',
                    default_configuration['ro'])
    writeconfig.set('normalization','vo',
                    default_configuration['vo'])
    writeconfig.add_section('astropy')
    writeconfig.set('astropy','astropy-units',
                    default_configuration['astropy-units'])
    writeconfig.set('astropy','astropy-coords',
                    default_configuration['astropy-coords'])
    with open(filename,'w') as configfile:
        writeconfig.write(configfile)
    return None

# Read the configuration file
__config__= configparser.ConfigParser(default_configuration)
if __config__.read([default_filename,'.galpyrc']) == []:
    write_default(default_filename)
    __config__.read(default_filename)

# Set configuration variables on the fly
def set_ro(ro):
    """
    NAME:
       set_ro
    PURPOSE:
       set the global configuration value of ro (distance scale)
    INPUT:
       ro - scale in kpc or astropy Quantity
    OUTPUT:
       (none)
    HISTORY:
       2016-01-05 - Written - Bovy (UofT)
    """
    if _APY_LOADED and isinstance(ro,units.Quantity):
        ro= ro.to(units.kpc).value
    __config__.set('normalization','ro',str(ro))

def set_vo(vo):
    """
    NAME:
       set_vo
    PURPOSE:
       set the global configuration value of vo (velocity scale)
    INPUT:
       vo - scale in km/s or astropy Quantity
    OUTPUT:
       (none)
    HISTORY:
       2016-01-05 - Written - Bovy (UofT)
    """
    if _APY_LOADED and isinstance(vo,units.Quantity):
        vo= vo.to(units.km/units.s).value
    __config__.set('normalization','vo',str(vo))
