#!/usr/bin/env python
import sys
import os
import numpy as np
from mpmath import fp, mp
from scipy.integrate import quad
import pandas as pd
import time
from tools.residuals import _RESIDUALS
from obslib.AN_pp.reader import READER
from obslib.AN_pp import AN_theory0 as AN_theory
from qcdlib.aux import AUX
from qcdlib.alphaS import ALPHAS
from tools.config import conf
from scipy.interpolate import griddata


class RESIDUALS(_RESIDUALS):

    def __init__(self):
        self.reaction = 'AN'
        self.tabs = conf['AN tabs']
        self.data={}
        for had in ['pi+','pi-','pi0']: 
            self.data[had] = {}
            self.data[had]['200']={}
        self.data['pi0']['500']={}
        self.setup()

    def _get_theory(self, entry):
        
        k, i = entry
        xF = self.tabs[k]['xF'][i]
        pT = self.tabs[k]['pT'][i]
        rs = self.tabs[k]['rs'][i]
        target = self.tabs[k]['target'][i]
        hadron = self.tabs[k]['hadron'][i]
        obs = self.tabs[k]['obs'][i].strip()
        col = self.tabs[k]['col'][i].strip().upper()

#         if obs == 'AN':
#             sigST = AN_theory.get_sigST(
#                 xF, pT, rs, target, hadron, mode='gauss', nx=10, nz=10)
#             sig = AN_theory.get_sig(
#                 xF, pT, rs, target, hadron, mode='gauss', nx=10, nz=10)
#             thy = sigST / sig

#             #print hadron,xF,thy
        
        if obs == 'AN':
            thy = self.get_asym(xF,pT,rs,hadron)

        return thy
    
    def _update(self,rootS,had):

        grid = []
        
        if rootS=='200':
            
            rs=200
                
            if had=='pi+' or had=='pi-':

#                 pTmin = 1 
#                 pTmax = 2.5

#                 PT = np.linspace(pTmin,pTmax,2)
#                 for i in range(len(PT)):
#                     if i==0: 
#                         XF = np.linspace(0.15,0.3,2)
#                     if i==1: 
#                         XF = np.linspace(0.15,0.3,2)
#                     for xF in XF:
#                         grid.append([xF,PT[i]])
            
                XF = np.array([0.2375,0.2875, \
                               0.1625,0.2375,0.2875])
                PT = np.array([1.1,1.55, \
                               1.0,1.75,2.4]) 


            elif had=='pi0':

#                 pTmin = 1 
#                 pTmax = 4

#                 PT = np.linspace(pTmin,pTmax,3)
#                 for i in range(len(PT)):
#                     if i==0: 
#                         XF = np.linspace(-0.65,-0.2,3)
#                         XF = np.append(XF,np.linspace(0.15,0.75,4))
#                     if i==1: 
#                         XF = np.linspace(-0.65,-0.2,3)
#                         XF = np.append(XF,np.linspace(0.15,0.75,4))
#                     if i==2: 
#                         XF = np.linspace(-0.65,-0.2,3)
#                         XF = np.append(XF,np.linspace(0.15,0.75,4))
#                     for xF in XF:
#                         grid.append([xF,PT[i]])

                XF = np.array([0.18,0.36,0.59, \
                               0.201,0.32,0.499,-0.201,-0.32,-0.499, \
                               0.415669,0.565395,0.714242, \
                               0.237,0.371,0.598,-0.237,-0.371,-0.598, \
                               0.28,0.28,0.28,0.32,0.32,0.32,0.37,0.37,0.37,0.43,0.43,0.43, \
                               0.5,0.5,0.5,0.6,0.6,0.6, \
                               0.206614,0.35273,0.503471,0.610336, \
                               0.207031,0.207031,0.207031,0.264528,0.264528,0.264528, \
                               0.324816,0.324816,0.324816])
                PT = np.array([1.0,1.5,2.4, \
                               1.56,2.39,3.66,1.56,2.39,3.66, \
                               2.1,2.85,3.61, \
                               1.19,1.79,2.78,1.19,1.79,2.78, \
                               1.0,1.6118,2.5249,1.0571,1.8728,2.8514,1.2912,2.0779,3.1814,1.5516,1.9819,3.2374,\
                               1.7029,2.5861,3.3166,2.2962,2.871,3.5122, \
                               2.29382,3.02115,2.80169,3.14834, \
                               1.78135,2.67572,3.06124,1.74844,2.74796,3.22564,  \
                               1.76185,3.37753,4.18825])

        elif rootS=='500':
            
            rs=500
            
#             pTmin = 2 
#             pTmax = 4.5
            
#             PT = np.linspace(pTmin,pTmax,3)
#             for i in range(len(PT)):
#                 if i==0: 
#                     XF = np.linspace(0.15,0.35,3)
#                 if i==1: 
#                     XF = np.linspace(0.15,0.35,3)
#                 if i==2: 
#                     XF = np.linspace(0.15,0.35,3)
#                 for xF in XF:
#                     grid.append([xF,PT[i]])

            XF = np.array([0.1621476,0.221754,0.3239232, \
                           0.220524,0.220524,0.220524,0.267502,0.267502,0.326862,0.326862]) 
            PT = np.array([3.25302,3.63569,4.40979, \
                           2.28197,3.39622,4.41765,2.78545,4.36804,3.60701,4.39947])
            
            
#         self.XF  = np.array(grid).T[0]
#         self.PT = np.array(grid).T[1]
        
        self.XF  = XF
        self.PT = PT

        self.data[had][rootS]['XF']  = self.XF
        self.data[had][rootS]['PT']  = self.PT
        
        #print(had,rootS,self.data[had][rootS]['XF'])
        
#         if rootS=='200': had_arr=['pi+','pi-','pi0']
#         elif rootS=='500': had_arr=['pi0']

#         for had in had_arr: self.data[had]={}

        #--look at observables and targets to calculate only necessary structure functions
#         tabs = self.tabs
#         for tab in tabs:
#             had = tabs[tab]['hadron'][0]
#             self.data[had] = np.zeros(self.XF.size)

        n=len(self.XF)

#         target='p'
#         for hadron in had_arr: 
#             for i in range(n):
#                 sigST[i] = AN_theory.get_sigST(self.XF[i], self.PT[i], rs, target, hadron, mode='gauss', nx=10, nz=10)
#                 sig[i] = AN_theory.get_sig(self.XF[i], self.PT[i], rs, target, hadron, mode='gauss', nx=10, nz=10)
#                 thy[i] = sigST[i] / sig[i]

#             self.data[had][rootS]=np.array([thy[i] for i in range(n)])

        target='p'
        sigST=np.zeros(n)
        sig=np.zeros(n)
        thy=np.zeros(n)
        #print(self.XF,self.PT)
        for i in range(n):
            sigST[i] = AN_theory.get_sigST(self.XF[i], self.PT[i], rs, target, had, mode='gauss', nx=10, nz=10)
            sig[i] = AN_theory.get_sig(self.XF[i], self.PT[i], rs, target, had, mode='gauss', nx=10, nz=10)
            thy[i] = sigST[i] / sig[i]

        self.data[had][rootS]['thy']=np.array([thy[i] for i in range(n)])
        
        #print(had,rootS,self.data[had][rootS]['thy'])
            

    def get_asym(self,xF,pT,rs,had):
        if rs==200: rootS='200'
        elif rs==500: rootS='500'
        XF = self.data[had][rootS]['XF']
        PT = self.data[had][rootS]['PT']
        thy= self.data[had][rootS]['thy']

        return griddata((XF,PT),thy,(xF,pT),fill_value=0, method='cubic')

    def gen_report(self, verb=1, level=1):
        """
        verb = 0: Do not print on screen. Only return list of strings
        verb = 1: print on screen the report
        level= 0: only the total chi2s
        level= 1: include point by point
        """

        L = []

        L.append('reaction: %s' % self.reaction)

        L.append('%7s %10s %10s %10s %10s %5s %10s %10s %10s %10s' % (
            'idx', 'tar', 'had', 'col', 'obs', 'npts', 'chi2', 'chi2/npts','rchi2', 'nchi2'))
        for k in self.tabs:
            #print k,len(self.tabs[k]['value'])
            if self.tabs[k]['value'].size == 0:
                continue
            res = self._get_residuals(k)
            rres = self._get_rres(k)
            nres = self._get_nres(k)

            chi2 = np.sum(res**2)
            rchi2 = np.sum(rres**2)
            nchi2 = nres**2
            tar = self.tabs[k]['target'][0]
            col = self.tabs[k]['col'][0].split()[0]
            obs = self.tabs[k]['obs'][0].split()[0]
            had = self.tabs[k]['hadron'][0].split()[0]
            npts = res.size
            L.append('%7d %10s %10s %10s %10s %5d %10.2f %10.2f %10.2f %10.2f' %
                     (k, tar, had, col, obs, npts, chi2, chi2/npts, rchi2, nchi2))

        if level == 1:
            L.append('-' * 100)

            msg = 'idx=%7d,  '
            msg += 'col=%7s,  '
            msg += 'tar=%7s,  '
            msg += 'had=%7s,  '
            msg += 'obs=%7s,  '
            if 'dependence' in self.tabs[k]:
                msg += 'dep=%7s,  '
            msg += 'xF=%10.3e,  '
            msg += 'pT=%10.3e,  '
            msg += 'rs=%10.3e,  '
            msg += 'exp=%10.3e,  '
            msg += 'alpha=%10.3e,  '
            msg += 'thy=%10.3e,  '
            if 'dthy' in self.tabs[k]:
                msg += 'dthy=%10.3e,  '
            msg += 'shift=%10.3e,  '
            msg += 'chi2=%10.3f  '

            for k in self.tabs:
                if len(self.tabs[k]['value']) == 0:
                    continue
                for i in range(len(self.tabs[k]['value'])):
                    row = [k]
                    row.append(self.tabs[k]['col'][i])
                    row.append(self.tabs[k]['target'][i])
                    row.append(self.tabs[k]['hadron'][i])
                    row.append(self.tabs[k]['obs'][i])
                    if 'dependence' in self.tabs[k]:
                        row.append(self.tabs[k]['dependence'][i].strip())
                    row.append(self.tabs[k]['xF'][i])
                    row.append(self.tabs[k]['pT'][i])
                    row.append(self.tabs[k]['rs'][i])
                    row.append(self.tabs[k]['value'][i])
                    row.append(self.tabs[k]['alpha'][i])
                    row.append(self.tabs[k]['thy'][i])
                    if 'dthy' in self.tabs[k]:
                        row.append(self.tabs[k]['dthy'][i])
                    row.append(self.tabs[k]['shift'][i])
                    # row.append(self.tabs[k]['residuals'][i])
                    # row.append(self.tabs[k]['r-residuals'][i])
                    res = self.tabs[k]['residuals'][i]
                    if res < 0:
                        chi2 = -res**2
                    else:
                        chi2 = res**2
                    row.append(chi2)
                    row = tuple(row)
                    L.append(msg % row)

        if verb == 0:
            return L
        elif verb == 1:
            for l in L:
                print(l)
            return L


if __name__ == '__main__':

    from qcdlib.interpolator import INTERPOLATOR
    from qcdlib import pdf0
    from qcdlib import ff0
    from qcdlib import pdf1_test as pdf1
    from qcdlib import ff1_test as ff1
    from qcdlib.aux import AUX
    from reader import READER
    from obslib.AN_pp import AN_theory0 as AN_theory

    conf['aux']    = AUX()

    conf['pdf']          = pdf0.PDF()
    conf['transversity'] = pdf1.PDF()
    conf['sivers']       = pdf1.PDF()
    conf['dsivers']       = pdf1.PDF('deriv')
    conf['ffpi']         = ff0.FF('pi')
    conf['ffk']          = ff0.FF('k')
    conf['collinspi']    = ff1.FF('pi')
    conf['dcollinspi']    = ff1.FF('pi','deriv')
    conf['collinsk']     = ff1.FF('k')
    conf['Htildepi']     = ff1.FF('pi')


    conf['datasets']={}
    conf['datasets']['AN']={}
    conf['datasets']['AN']['xlsx']={}
    conf['datasets']['AN']['xlsx'][1000]='AN_pp/expdata/1000.xlsx' # BRAHMS pim 2.3
    conf['datasets']['AN']['xlsx'][1001]='AN_pp/expdata/1001.xlsx' # BRAHMS pim 4
    conf['datasets']['AN']['xlsx'][1002]='AN_pp/expdata/1002.xlsx' # BRAHMS pip 2.3
    conf['datasets']['AN']['xlsx'][1003]='AN_pp/expdata/1003.xlsx' # BRAHMS pip 4
    conf['datasets']['AN']['xlsx'][2000]='AN_pp/expdata/2000.xlsx' # STAR piz 04
    conf['datasets']['AN']['xlsx'][2001]='AN_pp/expdata/2001.xlsx' # STAR piz 3.3
    conf['datasets']['AN']['xlsx'][2002]='AN_pp/expdata/2002.xlsx' # STAR piz 3.68
    conf['datasets']['AN']['xlsx'][2003]='AN_pp/expdata/2003.xlsx' # STAR piz 3.7
    conf['datasets']['AN']['xlsx'][2004]='AN_pp/expdata/2004.xlsx' # STAR piz 2008 |pT
    conf['datasets']['AN']['xlsx'][3000]='AN_pp/expdata/3000.xlsx' # STAR piz 2020 |xF 200 GeV
    conf['datasets']['AN']['xlsx'][3001]='AN_pp/expdata/3001.xlsx' # STAR piz 2020 |xF 500 GeV
    conf['datasets']['AN']['xlsx'][3002]='AN_pp/expdata/3002.xlsx' # STAR piz 2020 |pT 200 GeV
    conf['datasets']['AN']['xlsx'][3003]='AN_pp/expdata/3003.xlsx' # STAR piz 2020 |pT 500 GeV

    conf['datasets']['AN']['norm']={}
    for k in conf['datasets']['AN']['xlsx']: conf['datasets']['AN']['norm'][k]={'value':1,'fixed':True,'min':0,'max':1}
    conf['datasets']['AN']['filters']={}

    conf['AN tabs'] = READER().load_data_sets('AN')

    conf['residuals'] = RESIDUALS()
    
    conf['residuals']._update('200','pi+')
    conf['residuals']._update('200','pi-')
    conf['residuals']._update('200','pi0')
    conf['residuals']._update('500','pi0')
    
    print(conf['residuals'].get_residuals())

