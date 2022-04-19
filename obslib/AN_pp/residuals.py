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
from obslib.AN_pp import AN_theory0_hadinjet as AN_theory_hj
from qcdlib.aux import AUX
from qcdlib.alphaS import ALPHAS
from tools.config import conf
from scipy.integrate import fixed_quad


class RESIDUALS(_RESIDUALS):

    def __init__(self):
        self.reaction = 'AN'
        self.tabs = conf['AN tabs']
        self.storage={}
        self.setup()

    def _get_theory(self, entry):
        
        k, i = entry
        obs = self.tabs[k]['obs'][i].strip()
        col = self.tabs[k]['col'][i].strip().upper()
        target = self.tabs[k]['target'][i]
        hadron = self.tabs[k]['hadron'][i]

        if obs == 'AN':
            
            xF = self.tabs[k]['xF'][i]
            pT = self.tabs[k]['pT'][i]
            rs = self.tabs[k]['rs'][i]
            
            sigST = AN_theory.get_sigST(xF, pT, rs, target, hadron, mode='gauss', nx=10, nz=10)
            
            if conf['unpol PDF FF fixed']==True and (xF,pT,rs,target,hadron) not in self.storage:
                self.storage[(xF,pT,rs,target,hadron)] = AN_theory.get_sig(xF,pT,rs,target,hadron,mode='gauss', nx=10, nz=10)
            elif conf['unpol PDF FF fixed']==False:
                self.storage[(xF,pT,rs,target,hadron)] = AN_theory.get_sig(xF,pT,rs,target,hadron,mode='gauss', nx=10, nz=10)
            
            sig = self.storage[(xF,pT,rs,target,hadron)]
            
            thy = sigST / sig

            
        elif obs == 'AUTCollins':
            
            try: pT = self.tabs[k]['pT'][i]
            except (ValueError,KeyError):
                pT = None
                pTmin = self.tabs[k]['pTmin'][i]
                pTmax = self.tabs[k]['pTmax'][i]
                
            try: eta = self.tabs[k]['eta'][i]
            except (ValueError,KeyError):
                eta = None
                etamin = self.tabs[k]['etamin'][i]
                etamax = self.tabs[k]['etamax'][i]
        
            jperp = self.tabs[k]['jperp'][i]
            z = self.tabs[k]['z'][i]
            rs = self.tabs[k]['rs'][i]
            
            if pT == None:
                dsigST = np.vectorize(lambda pT: AN_theory_hj.get_sigST(z,jperp,pT,eta,rs,target,hadron,mode='gauss',nx=10))
                sigST = fixed_quad(dsigST,pTmin,pTmax,n=10)[0]
            
            elif eta == None:
                dsigST = np.vectorize(lambda eta: AN_theory_hj.get_sigST(z,jperp,pT,eta,rs,target,hadron,mode='gauss',nx=10))
                sigST = fixed_quad(dsigST,etamin,etamax,n=10)[0]
            
            if conf['unpol PDF FF fixed']==True and (z,jperp,target,hadron) not in self.storage:
                
                if pT == None:
                    dsig = np.vectorize(lambda pT: AN_theory_hj.get_sig(z,jperp,pT,eta,rs,target,hadron,mode='gauss',nx=10))
                    sig = fixed_quad(dsig,pTmin,pTmax,n=10)[0]
                    self.storage[(z,jperp,rs,target,hadron)] = sig
                
                elif eta == None:
                    dsig = np.vectorize(lambda eta: AN_theory_hj.get_sig(z,jperp,pT,eta,rs,target,hadron,mode='gauss',nx=10))
                    sig = fixed_quad(dsig,etamin,etamax,n=10)[0]
                    self.storage[(z,jperp,rs,target,hadron)] = sig
                    
            elif conf['unpol PDF FF fixed']==False:
                if pT == None:
                    dsig = np.vectorize(lambda pT: AN_theory_hj.get_sig(z,jperp,pT,eta,rs,target,hadron,mode='gauss',nx=10))
                    sig = fixed_quad(dsig,pTmin,pTmax,n=10)[0]
                    self.storage[(z,jperp,rs,target,hadron)] = sig
                
                elif eta == None:
                    dsig = np.vectorize(lambda eta: AN_theory_hj.get_sig(z,jperp,pT,eta,rs,target,hadron,mode='gauss',nx=10))
                    sig = fixed_quad(dsig,etamin,etamax,n=10)[0]
                    self.storage[(z,jperp,rs,target,hadron)] = sig
            
            sig = self.storage[(z,jperp,rs,target,hadron)]
            
            thy = sigST / sig
        
        return thy

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
    for k in conf['datasets']['AN']['xlsx']: conf['datasets']['AN']['norm'][k]={'value':1,'fixed':True,'min':0.8,'max':1.2}
    conf['datasets']['AN']['filters']={}

    conf['AN tabs'] = READER().load_data_sets('AN')

    conf['residuals'] = RESIDUALS()
    
    conf['unpol PDF FF fixed']=True
    
    for i in range(10):
        start = time.time()
        print(i,conf['residuals'].get_residuals())
        #conf['residuals'].get_residuals()
        end = time.time()
        print(i,'time=', (end - start))
