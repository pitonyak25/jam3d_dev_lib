#!/usr/bin/env python
import sys
import os
import numpy as np
import time
from tools.config import conf
from qcdlib.interpolator import INTERPOLATOR
from qcdlib.core import CORE
from qcdlib import core

class FF(CORE):
    """
    FF for positive pi,k. Use charge conj to get negative FF
    """

    def __init__(self,hadron,shape='nderiv'):
        self.shape = shape
        self.hadron=hadron
        self.aux = conf['aux']
        self.set_default_params()
        self.setup()

    def set_default_params(self):

        # free parameters
        self.shape1 = np.ones((11, 10))
        self.shape2 = np.ones((11, 10))
        self._widths1_fav    = 0.12
        self._widths1_ufav  = 0.12
        self._widths2_fav    = 0
        self._widths2_ufav  = 0

        # internal parameters
        self.widths1 = np.ones(11)
        self.widths2 = np.ones(11)

    def setup(self):
        # 1,  2,  3,  4,  5,  6,  7,  8,  9, 10
        # u, ub,  d, db,  s, sb,  c, cb,  b, bb
        if self.hadron=='pi':
            for i in range(1, 11):
                if   i == 1 or i==4: self.widths1[i] = self._widths1_fav
                else:                  self.widths1[i] = self._widths1_ufav
                if   i == 1 or i==4: self.widths2[i] = self._widths2_fav
                else:                  self.widths2[i] = self._widths2_ufav
        elif self.hadron=='k':
            for i in range(1, 11):
                if   i == 1 or i==6: self.widths1[i] = self._widths1_fav
                else:                  self.widths1[i] = self._widths1_ufav
                if   i == 1 or i==6: self.widths2[i] = self._widths2_fav
                else:                  self.widths2[i] = self._widths2_ufav
        elif self.hadron=='h':
            for i in range(1, 11):
                if   i == 1 or i==4 or i==6: self.widths1[i] = self._widths1_fav
                else:                  self.widths1[i] = self._widths1_ufav
                if   i == 1 or i==4 or i==6: self.widths2[i] = self._widths2_fav
                else:                  self.widths2[i] = self._widths2_ufav

    def get_C(self, z, Q2):
        if self.shape=='nderiv': return core.get_collinear(z,Q2,self.shape1,self.shape2)
        elif self.shape=='deriv': return core.get_dcollinear(z,Q2,self.shape1,self.shape2)

    #def get_dC(self, z, Q2):
    #    return self.get_dcollinear(z,Q2)

    def get_state(self):
        return (self.widths1,self.widths2,self.shape1, self.shape2)

    def set_state(self, state):
        self.widths1 = state[0]
        self.widths2 = state[1]
        self.shape1  = state[2]
        self.shape2  = state[3]

if __name__ == '__main__':

    from qcdlib.aux import AUX
    conf['aux']    = AUX()

    conf['ffpi'] = FF('pi')
    conf['ffk']  = FF('k')

    z = 0.15
    Q2 = 2.4
    print(conf['ffpi'].get_C(z, Q2))
    print(conf['ffk'].get_C(z, Q2))
