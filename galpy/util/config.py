import os, os.path
try:
    import configparser
except:
    from six.moves import configparser
# The default configuration
default_configuration= {'astropy-units':'False'}
default_filename= os.path.join(os.path.expanduser('~'),'.galpyrc')
def write_default(filename):
    writeconfig= configparser.ConfigParser()
    # Write different sections
    writeconfig.add_section('astropy')
    writeconfig.set('astropy','astropy-units',
                    default_configuration['astropy-units'])
    with open(filename,'wb') as configfile:
        writeconfig.write(configfile)
    return None

# Read the configuration file
__config__= configparser.ConfigParser(default_configuration)
if __config__.read([default_filename,'.galpyrc']) == []:
    write_default(default_filename)
    __config__.read(default_filename)
