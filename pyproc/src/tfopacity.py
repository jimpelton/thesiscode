
import numpy as np

class TFOpacity:
    def __init(self)__:
        # x values of opacity knots
        self.xp = None
        # opacity values of knots corresponding to values in xp
        self.yp = None

    def open(self, fname):
        '''
        open a transfer function, expects two columns of floats, with the
        first column increasing.
        '''
        # open file with numpy, default delim is whitespace.
       self.xp = np.loadtxt(fname, dtype=np.float64, usecols=0)
       self.yp = np.loadtxt(fname, dtype=np.float64, usecols=1)

    def interp(self, x) -> np.float64:
        return np.interp(x, self.xp, self.yp)



