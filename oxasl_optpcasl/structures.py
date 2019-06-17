"""
Basic data structures used in optpcasl
"""
import numpy as np

class ASLParams:
    """
    Define the ASL parameters (as per Buxton et al. MRM 1998)
    """
    def __init__(self, filename, f, **kwargs):
        self.filename, self.f = filename, f
        self.bat = kwargs.get("bat", 1.0)
        self.m0b = kwargs.get("m0b", 1.0)
        self.t1b = kwargs.get("t1b", 1.65)
        self.t1t = kwargs.get("t1t", 1.445)
        self.lam = kwargs.get("lam", 0.9)
        self.alpha = kwargs.get("alpha", 0.85)
        self.tau = kwargs.get("tau", 1.4)
        self.noise = kwargs.get("noise", 0.002)

class BATDist:
    """
    ATT (BAT) distribution

    The 'taper' parameter (seconds) causes the weighting to decay from 1 to 0.5
    at the beginning and end of the distribution
    """
    def __init__(self, start, end, step, taper=0):
        self.start, self.end, self.step, self.taper = start, end, step, taper
        self.dist = np.arange(start, end+step, step)
        self.exclude_taper = np.arange(start+taper, end-taper+step, step)
        self.weight = np.concatenate((
            np.linspace(0.5, 1.0, np.floor(taper / step)),
            np.ones(int(np.ceil((end - start - 2*taper) / step))),
            np.linspace(1.0, 0.5, np.floor(taper / step)),
        ))
        self.length = len(self.dist)

    def __str__(self):
        return "BAT distribution: %i values between %.2fs and %fs (weight taper=%.2fs)" % (self.length, self.start, self.end, self.taper)

class Scan:
    """
    Details of the desired scan to optimize for
    """
    def __init__(self, duration, npld, slices=1, slicedt=0.0, readout=0.5):
        self.duration, self.npld, self.slices, self.slicedt, self.readout = duration, npld, slices, slicedt, readout

    def __str__(self):
      if self.slices > 1:
          return "%is 2D scan with %i slices (time per slice=%.5fs) and readout time %.2fs" % (self.duration, self.slices, self.slicedt, self.readout)
      else:
          return "%is 3D scan with readout time %fs" % (self.duration, self.readout)

class Limits:
    """
    PLD limits and step size to search over
    """
    def __init__(self, lb, ub, step):
        self.lb, self.ub, self.step = lb, ub, step

    def __str__(self):
        return "PLDs between %.2fs and %.2fs in steps of %.5fs" % (self.lb, self.ub, self.step)
