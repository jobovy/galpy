import os, os.path
try:
    import configparser
except: # pragma: no cover
    from six.moves import configparser
_APY_LOADED= True
try:
    from astropy import units
except ImportError:
    _APY_LOADED= False
# The default configuration
default_configuration= {'normalization': {'ro':'8.',
                                         'vo':'220.'},
                        'astropy': {'astropy-units':'False',
                                    'astropy-coords':'True'},
                        'plot': {'seaborn-bovy-defaults':'False'},
                        'warnings': {'verbose':'False'}}
default_filename= os.path.join(os.path.expanduser('~'),'.galpyrc')
def check_config(configuration):
    # Check that the configuration is a valid galpy configuration
    for sec_key in default_configuration.keys():
        if not configuration.has_section(sec_key):
            return False
        for key in default_configuration[sec_key]:
            if not configuration.has_option(sec_key,key):
                return False
    return True
def write_config(filename,configuration=None):
    # Writes default if configuration is None
    writeconfig= configparser.ConfigParser()
    # Write different sections
    for sec_key in default_configuration.keys():
        writeconfig.add_section(sec_key)
        for key in default_configuration[sec_key]:
            if configuration is None \
                    or not configuration.has_section(sec_key) \
                    or not configuration.has_option(sec_key,key):
                writeconfig.set(sec_key,key,
                                default_configuration[sec_key][key])
            else:
                writeconfig.set(sec_key,key,configuration.get(sec_key,key))
    with open(filename,'w') as configfile:
        writeconfig.write(configfile)
    return None

# Read the configuration file
__config__= configparser.ConfigParser()
cfilename= __config__.read('.galpyrc')
if not cfilename:
    cfilename= __config__.read(default_filename)
    if not cfilename:
        write_config(default_filename)
        __config__.read(default_filename)
if not check_config(__config__):
    write_config(cfilename[-1],__config__)
    __config__.read(cfilename[-1])

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
