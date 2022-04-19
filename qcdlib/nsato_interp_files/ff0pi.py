import sys,os
import numpy as np
from numba import jit
from qcdlib import interp
from qcdlib import constants as const
from qcdlib.core import CORE
from tools.config import conf

#--Collinear parts
L =open(os.environ['JAM3D']+'/qcdlib/tables/dsspipLO_0000.dat').readlines()
xx=np.array([float(val) for val in L[3].split()])
QQ=np.array([float(val) for val in L[4].split()])
iflav=[int(val) for val in L[5].split()]
L=[l.split() for l in L[6:] if '---' not in l]
data=[[float(val) for val in l] for l in L]
data=np.transpose(data)

nx=len(xx)
nQ=len(QQ)
nf=len(iflav)

table=np.zeros((len(iflav),nx,nQ))
for i in range(nf): 
    cnt=-1
    for ix in range(nx):
        for iQ in range(nQ):
            cnt+=1
            table[i,ix,iQ]=data[i][cnt]

@jit(nopython=True)
def get_f(x,Q2):
    #iflav=[-5, -4, -3, -2, -1, 1, 2, 3, 4, 5, 21]
    #        0   1   2   3   4  5  6  7  8, 9, 10
    #self.IFLAV=[21,2,-2,1,-1,3,-3,4,-4,5,-5]
    iflav=[10,6,3,5,4,7,2,8,1,9,0]
    return np.array([interp.intepolate2d(xx,QQ, table[_,:,:],x,np.sqrt(Q2),10,10)/x for _ in iflav])

#--TMD parts
# data=np.load('jam20_params.npy',allow_pickle=True).item(0)        
# widths1 = data['ffpi']['widths1']
# widths2 = data['ffpi']['widths2']

# @jit(nopython=True)
# def get_widths(Q2):
#     s=get_s(Q2)
#     return np.abs(widths1+s*widths2)

class FF(CORE):
    """
    upol FF for positive pi,k. Use charge conj to get negative FF
    """

    def __init__(self,hadron):
        self.hadron=hadron
        self.aux = conf['aux']
        self.set_default_params()
        self.setup()

    def set_default_params(self):

        # free parameters
        self._widths1_fav   = 0.12
        self._widths1_ufav  = 0.15
        self._widths2_fav   = 0
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
                    
    def get_C(self,x, Q2):
        return get_f(x,Q2)

    def get_state(self):
        return self.widths1,self.widths2

    def set_state(self, state):
        self.widths1 = state[0]
        self.widths2 = state[1]

if __name__ == '__main__':

    from qcdlib.aux import AUX
    conf['aux']    = AUX()

    conf['ffpi'] = FF('pi')
    conf['ffk']  = FF('k')
    conf['ffh']  = FF('h')

    z = 0.15
    Q2 = 2.4
    print(conf['ffpi'].get_C(z, Q2))
    print(conf['ffk'].get_C(z, Q2))
    print(conf['ffh'].get_C(z, Q2))