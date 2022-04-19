#!/usr/bin/env python
import sys,os
import numpy as np
import time
import copy

#--from scipy
from scipy.optimize  import minimize,leastsq
from scipy.optimize  import least_squares

#--from tools
from tools.tools     import checkdir,save,load
import tools.config
from tools.config    import load_config, conf
from tools.inputmod  import INPUTMOD
from tools.randomstr import id_generator

#--from fitlib
from fitlib.resman import RESMAN

class MAXLIKE:

    def __init__(self,inputfile,nworkers=2,verbose=False,msrhook=None,prior=None,seed=None):

        self.nworkers=nworkers
        self.inputfile=inputfile
        self.verbose=verbose
        self.msrhook=msrhook
        self.prior=prior
        self.seed=seed

    def set_counters(self):
        self.chi2tot=1e1000
        self.dchi2=0
        self.t0 = time.time()
        self.cnt=0

    def print_status(self,res,rres,nres):

        #--update status parameters
        shifts=self.parman.shifts
        etime = (time.time()-self.t0)/60
        npts=res.size
        chi2=np.sum(res**2)
        rchi2=np.sum(rres**2)
        nchi2=np.sum(nres**2)
        chi2tot=chi2+rchi2+nchi2
        dchi2=chi2tot-self.chi2tot
        if  shifts>2:
            if  chi2tot<self.chi2tot:
                self.dchi2=self.chi2tot-chi2tot
                self.chi2tot=chi2tot

        #--build header
        status=[]
        status.append('JAM FITTER')
        status.append('count = %d'%self.cnt)
        status.append('elapsed time(mins)=%f'%etime)
        status.append('shifts  = %d'%shifts)
        status.append('npts    = %d'%npts)
        status.append('chi2    = %f'%chi2)
        status.append('rchi2   = %f'%rchi2)
        status.append('nchi2   = %f'%nchi2)
        status.append('chi2tot = %f'%(chi2tot))
        status.append('dchi2(iter)  = %f'%self.dchi2)
        status.append('dchi2(local) = %f'%dchi2)

        #--special output for pdfs
        #if 'pdf'  in conf['params']:
        #    for _ in conf['pdf'].sr:
        #        status.append('pdf %s:%f'%(_,conf['pdf'].sr[_]))

        #    #status.append('proton uvsr = %f'%conf['pdf'].sr['uvsr'])
        #    #status.append('proton dvsr = %f'%conf['pdf'].sr['dvsr'])
        #    #status.append('proton msr  = %f'%conf['pdf'].sr['msr'])
        #    #if 'svsr' in conf['pdf'].sr: status.append('proton svsr = %f'%conf['pdf'].sr['svsr'])

        #--report from resman
        status.append('')
        status.extend(self.resman.gen_report())

        #--report from parman
        parstatus = self.parman.gen_report()

        #--print into screen
        nstatus=len(status)
        nparstatus=len(parstatus)
        os.system('clear')
        for i in range(max([nstatus,nparstatus])):
            data=[]
            if i<nstatus: data.append(status[i])
            else: data.append('')
            if i<nparstatus: data.append(parstatus[i])
            else: data.append('')
            print('%-120s  | %s'%tuple(data))
        return status,parstatus

    def get_residuals(self,par):
        res,rres,nres=self.resman.get_residuals(par)
        self.cnt+=1
        if  self.cnt%conf['verbose']==0:
            self.print_status(res,rres,nres)
        if len(rres)!=0: res=np.append(res,rres)
        if len(nres)!=0: res=np.append(res,nres)
        return res

    def checklimits(self):

        for k in conf['params']:
            for kk in conf['params'][k]:
                if conf['params'][k][kk]['fixed']!=False: continue
                p=conf['params'][k][kk]['value']
                pmin=conf['params'][k][kk]['min']
                pmax=conf['params'][k][kk]['max']
                if  p<pmin or p>pmax:
                    print('%s-%s out of limits. '%(k,kk))
                    sys.exit()

        for k in conf['datasets']:
            for kk in conf['datasets'][k]['norm']:
                p=conf['datasets'][k]['norm'][kk]['value']
                pmin=conf['datasets'][k]['norm'][kk]['min']
                pmax=conf['datasets'][k]['norm'][kk]['max']
                if  p<pmin or p>pmax:
                    print('%s-%s out of limits. '%(k,kk))
                    sys.exit()

    def get_conf(self,_conf,step):

        conf=copy.deepcopy(_conf)

        #--remove pdf/ff that is not in the step
        distributions=conf['params'].keys()  #--pdf,ppdf,ffpion,ffkaon,...
        for dist in list(distributions):
            if  dist in step['active distributions']:
                print('active',dist)
                continue
            elif 'passive distributions' in step and dist in step['passive distributions']:
                print('passive',dist)
                continue
            else:
                print('remove',dist)
                if '%s parametrization'%dist in conf:
                    del conf['%s parametrization'%dist]
                del conf['params'][dist]

        #--set fixed==True for passive distributions
        if 'passive distributions' in step:
            for dist in step['passive distributions']:
                for par in conf['params'][dist]:

                    if conf['params'][dist][par]['fixed']==False:
                        conf['params'][dist][par]['fixed']=True

                    #--set prior parameters values for passive distributions
                    for istep in step['dep']:
                        prior_order=self.order[istep]
                        prior_params=self.params[istep]
                        for i in range(len(prior_order)):
                            _,_dist,_par = prior_order[i]
                            if  dist==_dist and par==_par:
                                conf['params'][dist][par]['value']=prior_params[i]

        #--another version for fixed parameters
        if 'fix parameters' in step:
            for dist in step['fix parameters']:
                for par in step['fix parameters'][dist]:
                    conf['params'][dist][par]['fixed']=True
                    #--set prior parameters values for passive distributions
                    for istep in step['dep']:
                        prior_order=self.order[istep]
                        prior_params=self.params[istep]
                        for i in range(len(prior_order)):
                            _,_dist,_par = prior_order[i]
                            if  dist==_dist and par==_par:
                                conf['params'][dist][par]['value']=prior_params[i]

        #--remove datasets not in the step
        datasets=list(conf['datasets']) #--idis,dy,....
        for dataset in datasets:
            if  dataset in step['datasets']:

                #--remove entry from xlsx
                xlsx=list(conf['datasets'][dataset]['xlsx'])
                for idx in xlsx:
                    if  idx in step['datasets'][dataset]:
                        continue
                    else:
                        del conf['datasets'][dataset]['xlsx'][idx]

                #--remove entry from norm
                norm=list(conf['datasets'][dataset]['norm'])
                for idx in norm:
                    if  idx in step['datasets'][dataset]:
                        continue
                    else:
                        del conf['datasets'][dataset]['norm'][idx]
            else:
                del conf['datasets'][dataset]

        return conf

    def get_bounds(self):
        order=self.parman.order

        bounds_min=[]
        bounds_max=[]
        for entry in order:
            i,k,kk=entry
            if  i==1:
                p=conf['params'][k][kk]['value']
                pmin=conf['params'][k][kk]['min']
                pmax=conf['params'][k][kk]['max']
                if p<pmin or p>pmax:
                    msg='%s/%s outsize the limits %f %f %f'%(k,kk,p,pmin,pmax)
                    raise ValueError(msg)
                bounds_min.append(conf['params'][k][kk]['min'])
                bounds_max.append(conf['params'][k][kk]['max'])
            elif i==2:
                p=conf['datasets'][k]['norm'][kk]['value']
                pmin=conf['datasets'][k]['norm'][kk]['min']
                pmax=conf['datasets'][k]['norm'][kk]['max']
                if p<pmin or p>pmax:
                    msg='%s/%s outsize the limits %f %f %f'%(k,kk,p,pmin,pmax)
                    raise ValueError(msg)
                bounds_min.append(conf['datasets'][k]['norm'][kk]['min'])
                bounds_max.append(conf['datasets'][k]['norm'][kk]['max'])

        return (bounds_min, bounds_max)

    def get_guess(self,dep=[]):

        order=self.parman.order
        if  conf['flat par']:
            if self.seed!=None: np.random.seed(12345)
            guess=self.parman.gen_flat()
        else:
            guess=self.parman.par

        #--retrieve priors from previous steps according to dep
        if len(dep)>0:

            for istep in dep:
                prior_order=self.order[istep]
                prior_params=self.params[istep]
                #--compare and match prior order to current order
                for i in range(len(prior_order)):
                    for j in range(len(order)):
                        if  order[j]==prior_order[i]:
                            guess[j]=prior_params[i]

        return guess

    def gen_summary(self,step,par):

        res,rres,nres    = self.resman.get_residuals(par)
        status,parstatus = self.print_status(res,rres,nres)
        status.extend(parstatus)
        status=[l+'\n' for l in status]
        fname='step-%d.summary'%step
        F=open(fname,'w')
        F.writelines(status)
        F.close()

    def get_par(self,step,dep=[]):

        #--current value must be within min and max
        self.checklimits()

        #--initialize resman
        self.resman=RESMAN(self.nworkers)
        self.parman=self.resman.parman

        #--setups
        guess=self.get_guess(dep)
        bounds=self.get_bounds()
        self.set_counters()
        self.parman.set_new_params(guess,initial=True)

        #--run fit
        fit = least_squares(self.get_residuals, guess,bounds=bounds,method='trf',ftol=conf['ftol'])

        #--generate summary. It will update system with the final results
        self.gen_summary(step,fit.x)

        #--close resman
        self.resman.shutdown()

        return fit.x

    def gen_output(self,istep):
        """
        modification of the input is done using a dedicated scrip
        at tools/inputmod.py
        """
        inputmod=INPUTMOD(self.inputfile)

        for kind in conf['params']:
            for par in conf['params'][kind]:
                value=conf['params'][kind][par]['value']
                inputmod.mod_par(kind,par,'value',value)

        for reaction in conf['datasets']:
            for idx in conf['datasets'][reaction]['norm']:
                value=conf['datasets'][reaction]['norm'][idx]['value']
                inputmod.mod_norm(reaction,idx,'value',value)

        fname='output-%d.py'%istep
        inputmod.gen_input(fname)

    def run(self):

        global conf

        load_config(self.inputfile)

        #--modify confs
        if  'hooks' not in conf:
            conf['hooks']={}
        if  self.msrhook!=None:
            conf['hooks']['msr']=self.msrhook
        if  self.verbose==False:
            conf['verbose']=1
        else:
            conf['verbose']=self.verbose

        #--backup conf(after mods)
        conf_bkp=copy.deepcopy(conf)

        #--decide if the input should be modified
        output=True
        if 'bootstrap' in conf and conf['bootstrap']==True: output=False

        if 'steps' in conf:
            isteps=sorted(conf['steps'])

            self.order={}
            self.params={}
            chi2={}

            if self.prior!=None:
                prior=np.load(self.prior, allow_pickle = True, encoding = 'latin1').item()() #load(self.prior)
                self.order=prior['order']
                self.params=prior['params']
                chi2=prior['chi2']
                prior_steps=sorted(self.order.keys())
                isteps=[i for i in isteps if i not in prior_steps]

            for i in isteps:
                step=conf['steps'][i]
                conf.update(copy.deepcopy(conf_bkp))
                conf.update(self.get_conf(conf_bkp,step))

                print()
                msg='--step %d: '%i
                msg+='npQCD objects ='
                for _ in conf['params'].keys():   msg+=_+' '
                msg+='datasets ='
                for _ in conf['datasets'].keys(): msg+=_+' '
                print(msg)

                par=self.get_par(i,step['dep'])
                self.order[i]=self.parman.order[:]
                self.params[i]=par[:]
                chi2[i]=self.resman.get_chi2()
                if output: self.gen_output(i)

            #--store the results from steps
            data={'order':self.order,'params':self.params,'chi2':chi2}
            if  self.prior==None:
                fname='%s.msr'%id_generator(size=12)
            else:
                fname=self.prior.split('/')[-1]
            save(data,fname)

            #--run hooks
            if 'msr' in conf['hooks']: #--ms==multi steps
                cmd=conf['hooks']['msr'].replace('<<fname>>',fname)
                os.system(cmd)
        else:
            par=self.get_par(0)
            if output: self.gen_output(0)


        #--run remining hooks if available
        for _ in conf['hooks']:
            if _=='msr': continue
            os.system(conf['hooks'][_])

if __name__=='__main__':

    MAXLIKE('input_evo.py',2,msrhook='mv <<fname>> step2',prior='step1/Y2RM020ZW0NB.msr',seed=12345).run()
    #MAXLIKE('input_evo.py',2,msrhook='mv <<fname>> step3',prior='step2/Y2RM020ZW0NB.msr',seed=12345).run()
    #MAXLIKE('input_evo.py',2,msrhook='mv <<fname>> step4',prior='step3/Y2RM020ZW0NB.msr',seed=12345).run()
    #MAXLIKE('input_evo.py',2,msrhook='mv <<fname>> step5',prior='step4/Y2RM020ZW0NB.msr',seed=12345).run()
    #MAXLIKE('input_evo.py',2,msrhook='mv <<fname>> step6',prior='step5/Y2RM020ZW0NB.msr',seed=12345).run()
    #MAXLIKE('input_evo.py',2,msrhook='mv <<fname>> step7',prior='step6/Y2RM020ZW0NB.msr',seed=12345).run()
    #MAXLIKE('input_evo.py',2,msrhook='mv <<fname>> step8',prior='step7/Y2RM020ZW0NB.msr',seed=12345).run()
