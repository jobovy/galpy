try:
    import pynbody
    _PYNBODYENABLED= True
except ImportError:
    _PYNBODYENABLED= False
class GadgetSnapshot(object):
    """Snapshot coming out of gadget"""
    def __init__(self,*args,**kwargs):
        """
        NAME:
           __init__
        PURPOSE:
           initialize a Gadget snapshot object
        INPUT:
           Initialize using:
              1) filename
        OUTPUT:
        HISTORY:
           2011-08-15 - Started - Bovy
        """
        if not _PYNBODYENABLED:
            raise ImportError("pynbody could not be loaded to read the gadget snapshot")
        filename= args[0]
        self._sim = pynbody.load(filename)
        self._sim.physical_units()
        self._star= self._sim.star
        return None

    def __getattr__(self,i):
        if i == 'x':
            return self.pos[0]
        elif i =='pos':
            return self._star['pos']

    def plot(self,*args,**kwargs):
        """
        NAME:
           plot
        PURPOSE:
           plot the snapshot
        INPUT:
        
        OUTPUT:
        HISTORY:
           2011-08-15 - Started - Bovy (NYU)
        """
        labeldict= {'t':r'$t$','R':r'$R$','vR':r'$v_R$','vT':r'$v_T$',
                    'z':r'$z$','vz':r'$v_z$','phi':r'$\phi$',
                    'x':r'$x$','y':r'$y$','vx':r'$v_x$','vy':r'$v_y$'}
        #Defaults
        if not kwargs.has_key('d1') and not kwargs.has_key('d2'):
            if len(self.orbits[0].vxvv) == 3:
                d1= 'R'
                d2= 'vR'
            elif len(self.orbits[0].vxvv) == 4:
                d1= 'x'
                d2= 'y'
            elif len(self.orbits[0].vxvv) == 2:
                d1= 'x'
                d2= 'vx'
            elif len(self.orbits[0].vxvv) == 5 \
                    or len(self.orbits[0].vxvv) == 6:
                d1= 'R'
                d2= 'z'
        elif not kwargs.has_key('d1'):
            d2= kwargs['d2']
            kwargs.pop('d2')
            d1= 't'
        elif not kwargs.has_key('d2'):
            d1= kwargs['d1']
            kwargs.pop('d1')
            d2= 't'
        else:
            d1= kwargs['d1']
            kwargs.pop('d1')
            d2= kwargs['d2']
            kwargs.pop('d2')

