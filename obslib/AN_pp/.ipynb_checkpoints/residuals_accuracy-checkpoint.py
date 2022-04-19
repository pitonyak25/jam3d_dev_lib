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


class RESIDUALS(_RESIDUALS):

    def __init__(self):
        self.reaction = 'AN'
        self.tabs = conf['AN tabs']
        self.setup()

    def _get_theory(self,entry,nx,nz):
        k, i = entry
        #print(entry)
        xF = self.tabs[k]['xF'][i]
        pT = self.tabs[k]['pT'][i]
        rs = self.tabs[k]['rs'][i]
        target = self.tabs[k]['target'][i]
        hadron = self.tabs[k]['hadron'][i]
        obs = self.tabs[k]['obs'][i].strip()
        col = self.tabs[k]['col'][i].strip().upper()

        if obs == 'AN':
            sigST = AN_theory.get_sigST(
                xF, pT, rs, target, hadron, mode='gauss', nx=nx, nz=nz)
            sig = AN_theory.get_sig(
                xF, pT, rs, target, hadron, mode='gauss', nx=nx, nz=nz)
            thy = sigST / sig

#             print(k,hadron,xF)
#             print('exp_rel=%f'%np.abs(100*(self.tabs[k]['alpha'][i]/self.tabs[k]['value'][i])))

        exp_rel = np.abs(100*(self.tabs[k]['alpha'][i]/self.tabs[k]['value'][i]))
        
        return (k,hadron,xF,thy,exp_rel)

        #return thy

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
    conf['datasets']['AN']['xlsx'][3000]='AN_pp/expdata/3000.xlsx' # STAR piz 2020 |xF
    conf['datasets']['AN']['xlsx'][3001]='AN_pp/expdata/3001.xlsx' # STAR piz 2020 |xF
    conf['datasets']['AN']['xlsx'][3002]='AN_pp/expdata/3002.xlsx' # STAR piz 2020 |pT
    conf['datasets']['AN']['xlsx'][3003]='AN_pp/expdata/3003.xlsx' # STAR piz 2020 |pT

    conf['datasets']['AN']['norm']={}
    for k in conf['datasets']['AN']['xlsx']: conf['datasets']['AN']['norm'][k]={'value':1,'fixed':True,'min':0,'max':1}
    conf['datasets']['AN']['filters']={}

    conf['AN tabs'] = READER().load_data_sets('AN')

    conf['residuals'] = RESIDUALS()
    
    for k in [1000,1001,1002,1003,2000,2001,2002,2003,2004,3000,3001,3002,3003]:
        for i in range(len(conf['AN tabs'][k]['xF'])):
            thy1 = conf['residuals']._get_theory((k,i),8,5)
            thy2 = conf['residuals']._get_theory((k,i),100,100)
            #print('thy_rel=%f'%np.abs((thy10-thy25)/thy25*100))
            
            thy_rel=(thy1[3]-thy2[3])/thy2[3]*100
            
            if np.abs(thy_rel) > thy1[4]: print(thy1[0],thy1[1],thy1[2],thy_rel,thy1[4])
            #print(thy1[0],thy1[1],thy1[2],'thy_rel=%.2f'%thy_rel,'exp_rel=%.2f'%thy1[4])
    
    #print(conf['residuals'].get_residuals())
