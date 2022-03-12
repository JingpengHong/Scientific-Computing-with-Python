"""
Defintions for problem 0
"""

import numpy as np
import scipy.integrate
from scipy.integrate import DenseOutput
from scipy.interpolate import interp1d

class ForwardEuler(scipy.integrate.OdeSolver):
    
    def __init__(self, fun, t0, y0, t_bound, vectorized, h=None):
        super().__init__(fun, t0, y0, t_bound, vectorized)
        
        if h is None:
            self.h = (t_bound-t0)/100
        else:
            self.h = h
            
    def _step_impl(self):
        
        t = self.t
        y = self.y
        h = self.h
        
        t_new = t + h
        y_new = y + h * self.fun(t_new, y)
        
        self.t = t_new
        self.y = y_new
        
        return True, None
            
    def _dense_output_impl(self):
        
        return ForwardEulerOutput(self.t, self.y)
  

class ForwardEulerOutput(DenseOutput):
    """
    Interpolate ForwardEuler output

    """
    def __init__(self, ts, ys):

        """
        store ts and ys computed in forward Euler method

        These will be used for evaluation
        """
        super(ForwardEulerOutput, self).__init__(np.min(ts), np.max(ts))
        self.interp = interp1d(ts, ys, kind='linear', copy=True)


    def _call_impl(self, t):
        """
        Evaluate on a range of values
        """
        return self.interp(t)
