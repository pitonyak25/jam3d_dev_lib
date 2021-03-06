from numpy.random import choice, randn
import numpy as np
from .multiproc import MULTIPROC
from .config import conf
from .tools import save
import copy


class dep_RESIDUALS:

    # basic routimes to prepare the data sets

    def percent_to_absolute(self):
        for k in self.tabs:
            ucorr = [x for x in self.tabs[k] if '_u' in x and '%' in x]
            corr = [x for x in self.tabs[k] if '_c' in x and '%' in x]
            if len(ucorr) != 0:
                for name in ucorr:
                    mod_name = name.replace('%', '')
                    self.tabs[k][mod_name] = self.tabs[k]['value'] * \
                        self.tabs[k][name] / 100.0
            if len(corr) != 0:
                for name in corr:
                    mod_name = name.replace('%', '')
                    self.tabs[k][mod_name] = self.tabs[k]['value'] * \
                        self.tabs[k][name] / 100.0

    def add_columns(self):
        for k in self.tabs:
            npts = len(self.tabs[k]['value'])
            self.tabs[k]['thy'] = np.zeros(npts)
            self.tabs[k]['N'] = np.zeros(npts)
            self.tabs[k]['residuals'] = np.zeros(npts)
            self.tabs[k]['r-residuals'] = np.zeros(npts)
            self.tabs[k]['Shift'] = np.zeros(npts)

    def get_alpha(self):
        for k in self.tabs:
            npts = len(self.tabs[k]['value'])
            alpha2 = np.zeros(npts)
            ucorr = [x for x in self.tabs[k] if '_u' in x and '%' not in x]
            for kk in ucorr:
                alpha2 += self.tabs[k][kk]**2
            self.tabs[k]['alpha'] = alpha2**0.5
            if 'fattening' in conf:
                self.tabs[k]['alpha'] *= conf['fattening']

    def retrieve_norm_uncertainty(self):

        for k in self.tabs:
            if k not in conf['datasets'][self.reaction]['norm']:
                continue
            norm = [x for x in self.tabs[k]
                    if '_c' in x and 'norm' in x and '%' not in x]
            if len(norm) > 1:
                msg = 'ERR: more than one normalization found at %s %d' % (
                    self.reaction, k)
                raise ValueError(msg)
            elif len(norm) == 1:
                #print '%d has norm uncertainty'%k
                dN = self.tabs[k][norm[0]][0] / self.tabs[k]['value'][0]
                conf['datasets'][self.reaction]['norm'][k]['dN'] = dN
            elif len(norm) == 0:
                dN = 1e-10  # conf['default dN']
                conf['datasets'][self.reaction]['norm'][k]['dN'] = dN

    def setup_rparams(self):
        if 'rparams' not in conf:
            conf['rparams'] = {}
        if self.reaction not in conf['rparams']:
            conf['rparams'][self.reaction] = {}
        for k in self.tabs:
            if k not in conf['rparams'][self.reaction]:
                conf['rparams'][self.reaction][k] = {}
            corr = [x for x in self.tabs[k] if '_c' in x and '%' not in x]
            for c in corr:
                conf['rparams'][self.reaction][k][c] = {
                    'value': 0.0, 'registered': False}

    def prepare_multiprocess(self):
        data = []
        for k in self.tabs:
            for i in range(len(self.tabs[k]['value'])):
                data.append([k, i])
        if 'ncpus' in conf:
            ncpus = conf['ncpus']
        else:
            ncpus = 1
        self.mproc = MULTIPROC(ncpus, self._get_theory, data)

    # routines for IMC analysis

    def select_training_sets(self, tab):
        for k in tab:
            key = self.tabs[k].leys()[0]
            npts = len(self.tabs[k][key])
            tab[k]['iT'] = np.zeros(npts)
            if npts > 5:
                nptsT = int(conf['training frac'] * npts)
                iT = choice(npts, nptsT, replace=False)
                for i in iT:
                    tab[k]['iT'][i] = 1

    def resample(self):
        self.tabs = copy.deepcopy(self.original)
        for k in self.tabs:
            npts = len(self.tabs[k]['value'])
            self.tabs[k]['value'] += randn(npts) * self.tabs[k]['alpha']

    def setup_imc(self):
        # only useful for IMC
        if 'cross-validation' in conf:
            if conf['cross-validation']:
                self.select_training_sets()
        if 'bootstrap' in conf:
            if conf['bootstrap']:
                self.resample()

    # master setup

    def setup(self):
        self.percent_to_absolute()
        self.add_columns()
        self.get_alpha()
        self.retrieve_norm_uncertainty()
        self.setup_rparams()
        self.prepare_multiprocess()
        self.original = copy.deepcopy(self.tabs)
        # self.setup_imc()

    # residuals

    def _get_residuals(self, k):
        npts = len(self.tabs[k]['value'])
        exp = self.tabs[k]['value']
        if k in conf['datasets'][self.reaction]['norm']:
            norm = conf['datasets'][self.reaction]['norm'][k]['value']
        else:
            norm = 1.0
        thy = self.tabs[k]['thy'] / norm
        alpha = self.tabs[k]['alpha']
        corr = [x for x in self.tabs[k]
                if '_c' in x and '%' not in x and 'norm' not in x]
        N = np.ones(exp.size)
        ncorr = len(corr)
        if ncorr == 0:
            self.tabs[k]['N'] = N
            self.tabs[k]['residuals'] = (exp - thy) / alpha
            self.tabs[k]['shift'] = np.zeros(exp.size)
        else:
            beta = [self.tabs[k][c] * (thy / (exp + 1e-100)) for c in corr]
            A = np.diag(np.diag(np.ones((ncorr, ncorr)))) + \
                np.einsum('ki,li,i->kl', beta, beta, 1 / alpha**2)
            B = np.einsum('ki,i,i->k', beta, exp - thy, 1 / alpha**2)
            try:
                r = np.einsum('kl,l->k', np.linalg.inv(A), B)
            except:
                r = np.zeros(len(beta))
            shift = np.einsum('k,ki->i', r, beta)
            for i in range(ncorr):
                conf['rparams'][self.reaction][k][corr[i]]['value'] = r[i]
            self.tabs[k]['N'] = N
            self.tabs[k]['residuals'] = (exp - shift - thy) / alpha
            self.tabs[k]['shift'] = shift
        return self.tabs[k]['residuals']

    def _get_residuals_simple(self, k):
        exp = self.tabs[k]['value']
        thy = self.tabs[k]['thy']
        alpha = self.tabs[k]['alpha']
        self.tabs[k]['N'] = np.ones(exp.size)
        self.tabs[k]['residuals'] = (exp - thy) / alpha
        self.tabs[k]['shift'] = np.zeros(exp.size)
        return self.tabs[k]['residuals']

    def _get_rres(self, k):
        rres = []
        rparams = conf['rparams'][self.reaction][k]
        for c in rparams:
            rres.append(rparams[c]['value'])
        return np.array(rres)

    def _get_nres(self, k):
        if k not in conf['datasets'][self.reaction]['norm']:
            return 0
        if conf['datasets'][self.reaction]['norm'][k]['fixed']:
            return 0
        norm = conf['datasets'][self.reaction]['norm'][k]
        return (norm['value'] - 1) / norm['dN']

    def get_theory(self):
        output = self.mproc.run()
        THY = []
        for entry in output:
            k, i, thy = entry
            self.tabs[k]['thy'][i] = thy

    def get_npts(self):
        npts = 0
        for k in self.tabs:
            npts += len(self.tabs[k]['value'])
        return npts

    def get_residuals(self, calc=True, simple=False):
        res, rres, nres = [], [], []
        if calc:
            self.get_theory()
        for k in self.tabs:
            if simple:
                res = np.append(res, self._get_residuals_simple(k))
            else:
                res = np.append(res, self._get_residuals(k))
            rres = np.append(rres, self._get_rres(k))
            nres = np.append(nres, self._get_nres(k))
        return res, rres, nres

    # other functions

    def ___gen_report(self, verb=1, level=1):
        """
        verb = 0: Do not print on screen. Only return list of strings
        verv = 1: print on screen the report
        level= 0: only the total chi2s
        level= 1: include point by point
        """

        L = []

        L.append(self.reaction)

        for k in self.tabs:
            res = self._get_residuals(k)
            rres = self._get_rres(k)
            nres = self._get_nres(k)

            chi2 = np.sum(res**2)
            rchi2 = np.sum(rres**2)
            nchi2 = nres**2
            tar = self.tabs[k]['target'].values[0]
            col = self.tabs[k]['col'].values[0].split()[0]
            npts = res.size
            L.append('%7d %10s %10s %5d %10.2f %10.2f %10.2f' %
                     (k, tar, col, npts, chi2, rchi2, nchi2))

        if level == 1:
            L.append('-' * 100)
            for k in conf['sidistab']:
                if len(conf['sidistab'][k].index) == 0:
                    continue
                for i in range(len(conf['sidistab'][k].index)):
                    x = conf['sidistab'][k]['x'].values[i]
                    obs = conf['sidistab'][k]['obs'].values[i]
                    Q2 = conf['sidistab'][k]['Q2'].values[i]
                    res = conf['sidistab'][k]['residuals'].values[i]
                    thy = conf['sidistab'][k]['thy'].values[i]
                    exp = conf['sidistab'][k]['value'].values[i]
                    alpha = conf['sidistab'][k]['alpha'].values[i]
                    rres = conf['sidistab'][k]['r-residuals'].values[i]
                    col = conf['sidistab'][k]['col'].values[i]
                    shift = conf['sidistab'][k].Shift.values[i]
                    if res < 0:
                        chi2 = -res**2
                    else:
                        chi2 = res**2
                    msg = '%7s %7s x=%10.3e Q2=%10.3e exp=%10.3e alpha=%10.3e thy=%10.3e shift=%10.3e chi2=%10.3f'
                    L.append(msg %
                             (col, obs, x, Q2, exp, alpha, thy, shift, chi2))

        if verb == 0:
            return L
        elif verb == 1:
            for l in L:
                print(l)

    def ___save_results(self, path):
        save(self.tabs, path)

class _RESIDUALS:

    #--basic routimes to prepare the data sets
  
    def percent_to_absolute(self):

        for k in self.tabs:
            ucorr = [x for x in self.tabs[k] if '_u' in x and '%' in x]
            corr  = [x for x in self.tabs[k] if '_c' in x and '%' in x]

            if  len(ucorr)!=0:
                for name in ucorr:
                    mod_name=name.replace('%','')
                    self.tabs[k][mod_name]=self.tabs[k]['value'] * self.tabs[k][name]/100.0

            if  len(corr)!=0:
                for name in corr:
                    mod_name=name.replace('%','')
                    self.tabs[k][mod_name]=self.tabs[k]['value'] * self.tabs[k][name]/100.0
  
    def add_columns(self):

        for k in self.tabs:
            npts=len(self.tabs[k]['value'])
            self.tabs[k]['thy']=np.zeros(npts)
            self.tabs[k]['N']=np.zeros(npts)
            self.tabs[k]['residuals']=np.zeros(npts)
            self.tabs[k]['r-residuals']=np.zeros(npts)
            self.tabs[k]['Shift']=np.zeros(npts)
  
    def get_alpha(self):

        for k in self.tabs:
            npts=len(self.tabs[k]['value'])
            alpha2=np.zeros(npts) 
            ucorr = [x for x in self.tabs[k] if '_u' in x and '%' not in x]
            for kk in ucorr: alpha2+=self.tabs[k][kk]**2
            self.tabs[k]['alpha']=alpha2**0.5
  
    def retrieve_norm_uncertainty(self):
  
        for k in self.tabs:

            if k not in conf['datasets'][self.reaction]['norm']: continue

            norm  = [x for x in self.tabs[k] if '_c' in x and 'norm' in x and '%' not in x]

            if  len(norm)>1:
                msg='ERR: more than one normalization found at %s %d'%(self.reaction,k)
                raise ValueError(msg)

            elif len(norm)==1:
                print('%d has norm uncertainty'%k)
                dN=self.tabs[k][norm[0]][0]/self.tabs[k]['value'][0]
                conf['datasets'][self.reaction]['norm'][k]['dN']=dN

            elif len(norm)==0:
                if 'dN' not in conf['datasets'][self.reaction]['norm'][k]:
                    conf['datasets'][self.reaction]['norm'][k]['dN']=0.01
  
    def setup_rparams(self):

        if  'rparams' not in conf: 
            conf['rparams']={}

        if  self.reaction not in conf['rparams']: 
            conf['rparams'][self.reaction]={}

        for k in self.tabs:

            if  k not in conf['rparams'][self.reaction]:
                conf['rparams'][self.reaction][k]={}

            corr = [x for x in self.tabs[k] if '_c' in x and '%' not in x and 'norm' not in x]
            for c in corr: 
                conf['rparams'][self.reaction][k][c]={'value':0.0,'registered':False}
  
    def resample(self):

        self.tabs=copy.deepcopy(self.original)

        for k in self.tabs:
            npts=len(self.tabs[k]['value'])
            self.tabs[k]['value']+=randn(npts)*self.tabs[k]['alpha']
  
    #--master setup
  
    def setup(self):
      self.percent_to_absolute()
      self.add_columns()
      self.get_alpha()
      self.retrieve_norm_uncertainty()
      self.setup_rparams()
      self.setup_requests()
      self.original=copy.deepcopy(self.tabs)
      if 'bootstrap' in conf and conf['bootstrap']: self.resample()
  
    #--residuals
  
    def _get_residuals(self,k):

        npts=len(self.tabs[k]['value'])
        exp=self.tabs[k]['value']

        if  k in conf['datasets'][self.reaction]['norm']:
            norm=conf['datasets'][self.reaction]['norm'][k]['value']
        else:
            norm=1.0

        thy=self.tabs[k]['thy']/norm
        alpha=self.tabs[k]['alpha']
        corr = [x for x in self.tabs[k] if '_c' in x and '%' not in x and 'norm' not in x]
        N=np.ones(exp.size)
        ncorr=len(corr)

        if  ncorr==0:

            self.tabs[k]['N']=N
            self.tabs[k]['residuals']=(exp-thy)/alpha
            self.tabs[k]['shift']=np.zeros(exp.size)
            self.tabs[k]['prediction']=thy

        else:

            beta=[self.tabs[k][c] * (thy/(exp+1e-100)) for c in corr]
            A=np.diag(np.diag(np.ones((ncorr,ncorr)))) + np.einsum('ki,li,i->kl',beta,beta,1/alpha**2)
            B=np.einsum('ki,i,i->k',beta,exp-thy,1/alpha**2)
            try:
                r=np.einsum('kl,l->k',np.linalg.inv(A),B)
            except:
                r=np.zeros(len(beta))
            shift=np.einsum('k,ki->i',r,beta)
            for i in range(ncorr):
                conf['rparams'][self.reaction][k][corr[i]]['value']=r[i]
            self.tabs[k]['N']=N
            self.tabs[k]['residuals']=(exp-shift-thy)/alpha
            self.tabs[k]['shift']=shift
            self.tabs[k]['prediction']=shift+thy

        return self.tabs[k]['residuals']
  
    def _get_rres(self,k):
        rres=[]
        rparams=conf['rparams'][self.reaction][k]
        for c in rparams:
            rres.append(rparams[c]['value'])
        return np.array(rres)
  
    def _get_nres(self,k):

        if  k not in conf['datasets'][self.reaction]['norm']:
            return 0

        elif  conf['datasets'][self.reaction]['norm'][k]['fixed']:
            return 0

        else:
            norm=conf['datasets'][self.reaction]['norm'][k]
            return (norm['value']-1)/norm['dN']
  
    def get_residuals(self,calc=True):

        res,rres,nres=[],[],[]

        if calc: self.get_theory()

        for k in self.tabs: 
            res=np.append(res,self._get_residuals(k))
            rres=np.append(rres,self._get_rres(k))
            nres=np.append(nres,self._get_nres(k))
        return res,rres,nres
  
    #--mis functions 
 
    def get_npts(self):
        npts=0
        for k in self.tabs: 
            npts+=len(self.tabs[k]['value'])
        return npts

    def get_chi2(self):

      data={self.reaction:{}}

      for idx in self.tabs:
          if len(self.tabs[idx])==0: continue 
          res=self.tabs[idx]['residuals']
          npts=res.size
          chi2=np.sum(res**2)
          data[self.reaction][idx]={'npts':npts,'chi2':chi2}

      return data

    #--theory calculation  
  
    def setup_requests(self):
        requests=[]
        for idx in self.tabs:
            for irow in range(len(self.tabs[idx]['value'])):
                request={}
                request['reaction'] = self.reaction
                request['dataset']  = idx
                request['irow']     = irow
                request['thy']      = 0.0
                requests.append(request)
        self.requests=requests

    def process_request(self,request):
        entry=(request['dataset'],request['irow'])
        request['thy']=self._get_theory(entry)

    def update_tabs_local(self):

        for i  in range(len(self.requests)):
            idx  = self.requests[i]['dataset']
            irow = self.requests[i]['irow']
            self.tabs[idx]['thy'][irow]=self.requests[i]['thy']

    def get_theory(self):

        for i in range(len(self.requests)):
            self.process_request(self.requests[i])

        self.update_tabs_local()

    #--update tabs using external data

    def update_tabs_external(self,request):
        idx  = request['dataset']
        irow = request['irow']
        self.tabs[idx]['thy'][irow]=request['thy']



