conf={}
#Convert to input file for Jam3d
############################################################################
#conf['method']='cov'
#conf['kappa']=1.1
#conf['tol']=10e-4
#conf['num points'] = 19*2
#conf['burn size']  = 100
#conf['sample size']= 10000

############################################################################
# resouce allocation
conf['ncpus']=1


############################################################################
# paths to external

#conf['path2CJ'] ='../external/CJLIB'
#conf['path2LSS']='../external/LSSLIB'
#conf['path2DSS']='../external/DSSLIB'

############################################################################
# params
conf['shape']=0

conf['params']={}

conf['params']['pdf']={}
conf['params']['pdf']['widths1_uv']  ={'value':    5.89295e-01,'min': 0,'max':1,'fixed':True}
conf['params']['pdf']['widths2_uv']  ={'value':    0.00000e+00,'min':-1,'max':1,'fixed':True}
conf['params']['pdf']['widths1_dv']  ={'value':    5.89295e-01,'min': 0,'max':1,'fixed':'widths1_uv'}
conf['params']['pdf']['widths2_dv']  ={'value':    0.00000e+00,'min':-1,'max':1,'fixed':'widths2_uv'}
conf['params']['pdf']['widths1_sea'] ={'value':    6.33443e-01,'min': 0,'max':1,'fixed':True}
conf['params']['pdf']['widths2_sea'] ={'value':    0.00000e+00,'min':-1,'max':1,'fixed':'widths2_uv'}


conf['params']['ffpi']={}
conf['params']['ffpi']['widths1_fav']   ={'value':    1.15921e-01,'min': 0,'max':1,'fixed':True}
conf['params']['ffpi']['widths2_fav']   ={'value':    0.00000e+00,'min':-1,'max':1,'fixed':True}
conf['params']['ffpi']['widths1_ufav'] ={'value':    1.39782e-01,'min': 0,'max':1,'fixed':True}
conf['params']['ffpi']['widths2_ufav'] ={'value':    0.00000e+00,'min':-1,'max':1,'fixed':'widths2_fav'}
conf['params']['ffk']={}
conf['params']['ffk']['widths1_fav']    ={'value':    1.31266e-01,'min': 0,'max':1,'fixed':True}
conf['params']['ffk']['widths2_fav']    ={'value':    0.00000e+00,'min':-1,'max':1,'fixed':True}
conf['params']['ffk']['widths1_ufav']  ={'value':    1.85599e-01,'min': 0,'max':1,'fixed':True}
conf['params']['ffk']['widths2_ufav']  ={'value':    0.00000e+00,'min':-1,'max':1,'fixed':'widths2_fav'}

#Sivers function
conf['params']['sivers']={}
conf['params']['sivers']['widths1_uv']  ={'value':    2.01072e-01, 'min':0, 'max':1, 'fixed': False}
conf['params']['sivers']['widths1_dv']  ={'value':    2.01072e-01, 'min':0, 'max':1, 'fixed': 'widths1_uv'}
conf['params']['sivers']['widths1_sea'] ={'value':    3.88407e-01, 'min':0, 'max':1, 'fixed': True}
conf['params']['sivers']['widths2_uv']  ={'value':    0.00000e+00, 'min':0, 'max':1, 'fixed': True}
conf['params']['sivers']['widths2_dv']  ={'value':    0.00000e+00, 'min':0, 'max':1, 'fixed': 'widths2_uv'}
conf['params']['sivers']['widths2_sea'] ={'value':    0.00000e+00, 'min':0, 'max':1, 'fixed': 'widths2_uv'}

conf['params']['sivers']['u N0 1'] ={'value':   -1.82741e-01, 'min': -10.0, 'max': 10.0, 'fixed': False}
conf['params']['sivers']['u N1 1'] ={'value':    0.00000e+00, 'min': -10.0, 'max': 1.0, 'fixed': True}
conf['params']['sivers']['u a0 1'] ={'value':   -9.26734e-02, 'min':-2.0, 'max':10.0, 'fixed': False}
conf['params']['sivers']['u a1 1'] ={'value':    0.00000e+00, 'min':-2.0, 'max':10.0, 'fixed': True}
conf['params']['sivers']['u b0 1'] ={'value':    4.42200e+00, 'min': 0.0, 'max':10.0, 'fixed': False}
conf['params']['sivers']['u b1 1'] ={'value':    0.00000e+00, 'min': 0.0, 'max':10.0, 'fixed': True}
conf['params']['sivers']['u c0 1'] ={'value':    0.00000e+00, 'min':-1.0, 'max': 1.0, 'fixed': True}
conf['params']['sivers']['u c1 1'] ={'value':    0.00000e+00, 'min':-1.0, 'max': 1.0, 'fixed': True}
conf['params']['sivers']['u d0 1'] ={'value':    0.00000e+00, 'min':-1.0, 'max': 1.0, 'fixed': True}
conf['params']['sivers']['u d1 1'] ={'value':    0.00000e+00, 'min':-1.0, 'max': 1.0, 'fixed': True}
conf['params']['sivers']['u N0 2'] ={'value':    0.00000e+00, 'min': 0.0, 'max': 1.0, 'fixed': True}
conf['params']['sivers']['u N1 2'] ={'value':    0.00000e+00, 'min': 0.0, 'max': 1.0, 'fixed': True}
conf['params']['sivers']['u a0 2'] ={'value':    0.00000e+00, 'min':-2.0, 'max':10.0, 'fixed': True}
conf['params']['sivers']['u a1 2'] ={'value':    0.00000e+00, 'min':-2.0, 'max':10.0, 'fixed': True}
conf['params']['sivers']['u b0 2'] ={'value':    0.00000e+00, 'min': 0.0, 'max':10.0, 'fixed': True}
conf['params']['sivers']['u b1 2'] ={'value':    0.00000e+00, 'min': 0.0, 'max':10.0, 'fixed': True}
conf['params']['sivers']['u c0 2'] ={'value':    0.00000e+00, 'min':-1.0, 'max': 1.0, 'fixed': True}
conf['params']['sivers']['u c1 2'] ={'value':    0.00000e+00, 'min':-1.0, 'max': 1.0, 'fixed': True}
conf['params']['sivers']['u d0 2'] ={'value':    0.00000e+00, 'min':-1.0, 'max': 1.0, 'fixed': True}
conf['params']['sivers']['u d1 2'] ={'value':    0.00000e+00, 'min':-1.0, 'max': 1.0, 'fixed': True}


conf['params']['sivers']['d N0 1'] ={'value':    5.85923e-02, 'min': -20.0, 'max': 10.0, 'fixed': False}
conf['params']['sivers']['d N1 1'] ={'value':    0.00000e+00, 'min': 0.0, 'max': 1.0, 'fixed': 'u N1 1'}
conf['params']['sivers']['d a0 1'] ={'value':   -5.87940e-01, 'min':-2.0, 'max':10.0, 'fixed': False}
conf['params']['sivers']['d a1 1'] ={'value':    0.00000e+00, 'min':-2.0, 'max':10.0, 'fixed': 'u a1 1'}
conf['params']['sivers']['d b0 1'] ={'value':    3.41349e+00, 'min': 0.0, 'max':10.0, 'fixed': False}
conf['params']['sivers']['d b1 1'] ={'value':    0.00000e+00, 'min': 0.0, 'max':10.0, 'fixed': 'u b1 1'}
conf['params']['sivers']['d c0 1'] ={'value':    0.00000e+00, 'min':-1.0, 'max': 1.0, 'fixed': True}
conf['params']['sivers']['d c1 1'] ={'value':    0.00000e+00, 'min':-1.0, 'max': 1.0, 'fixed': True}
conf['params']['sivers']['d d0 1'] ={'value':    0.00000e+00, 'min':-1.0, 'max': 1.0, 'fixed': True}
conf['params']['sivers']['d d1 1'] ={'value':    0.00000e+00, 'min':-1.0, 'max': 1.0, 'fixed': True}
conf['params']['sivers']['d N0 2'] ={'value':    0.00000e+00, 'min': 0.0, 'max': 1.0, 'fixed': True}
conf['params']['sivers']['d N1 2'] ={'value':    0.00000e+00, 'min': 0.0, 'max': 1.0, 'fixed': True}
conf['params']['sivers']['d a0 2'] ={'value':    0.00000e+00, 'min':-2.0, 'max':10.0, 'fixed': True}
conf['params']['sivers']['d a1 2'] ={'value':    0.00000e+00, 'min':-2.0, 'max':10.0, 'fixed': True}
conf['params']['sivers']['d b0 2'] ={'value':    0.00000e+00, 'min': 0.0, 'max':10.0, 'fixed': True}
conf['params']['sivers']['d b1 2'] ={'value':    0.00000e+00, 'min': 0.0, 'max':10.0, 'fixed': True}
conf['params']['sivers']['d c0 2'] ={'value':    0.00000e+00, 'min':-1.0, 'max': 1.0, 'fixed': True}
conf['params']['sivers']['d c1 2'] ={'value':    0.00000e+00, 'min':-1.0, 'max': 1.0, 'fixed': True}
conf['params']['sivers']['d d0 2'] ={'value':    0.00000e+00, 'min':-1.0, 'max': 1.0, 'fixed': True}
conf['params']['sivers']['d d1 2'] ={'value':    0.00000e+00, 'min':-1.0, 'max': 1.0, 'fixed': True}


conf['params']['sivers']['s N0 1'] ={'value':    0.00000e+00, 'min': -10.0, 'max': 10.0, 'fixed': True}
conf['params']['sivers']['s N1 1'] ={'value':    0.00000e+00, 'min': 0.0, 'max': 1.0, 'fixed': True}
conf['params']['sivers']['s a0 1'] ={'value':    0.00000e+00, 'min':-2.0, 'max':10.0, 'fixed': True}
conf['params']['sivers']['s a1 1'] ={'value':    0.00000e+00, 'min':-2.0, 'max':10.0, 'fixed': True}
conf['params']['sivers']['s b0 1'] ={'value':    0.00000e+00, 'min': 0.0, 'max':10.0, 'fixed': True}
conf['params']['sivers']['s b1 1'] ={'value':    0.00000e+00, 'min': 0.0, 'max':10.0, 'fixed': True}
conf['params']['sivers']['s c0 1'] ={'value':    0.00000e+00, 'min':-1.0, 'max': 1.0, 'fixed': True}
conf['params']['sivers']['s c1 1'] ={'value':    0.00000e+00, 'min':-1.0, 'max': 1.0, 'fixed': True}
conf['params']['sivers']['s d0 1'] ={'value':    0.00000e+00, 'min':-1.0, 'max': 1.0, 'fixed': True}
conf['params']['sivers']['s d1 1'] ={'value':    0.00000e+00, 'min':-1.0, 'max': 1.0, 'fixed': True}
conf['params']['sivers']['s N0 2'] ={'value':    0.00000e+00, 'min': 0.0, 'max': 1.0, 'fixed': True}
conf['params']['sivers']['s N1 2'] ={'value':    0.00000e+00, 'min': 0.0, 'max': 1.0, 'fixed': True}
conf['params']['sivers']['s a0 2'] ={'value':    0.00000e+00, 'min':-2.0, 'max':10.0, 'fixed': True}
conf['params']['sivers']['s a1 2'] ={'value':    0.00000e+00, 'min':-2.0, 'max':10.0, 'fixed': True}
conf['params']['sivers']['s b0 2'] ={'value':    0.00000e+00, 'min': 0.0, 'max':10.0, 'fixed': True}
conf['params']['sivers']['s b1 2'] ={'value':    0.00000e+00, 'min': 0.0, 'max':10.0, 'fixed': True}
conf['params']['sivers']['s c0 2'] ={'value':    0.00000e+00, 'min':-1.0, 'max': 1.0, 'fixed': True}
conf['params']['sivers']['s c1 2'] ={'value':    0.00000e+00, 'min':-1.0, 'max': 1.0, 'fixed': True}
conf['params']['sivers']['s d0 2'] ={'value':    0.00000e+00, 'min':-1.0, 'max': 1.0, 'fixed': True}
conf['params']['sivers']['s d1 2'] ={'value':    0.00000e+00, 'min':-1.0, 'max': 1.0, 'fixed': True}


conf['params']['sivers']['ub N0 1'] ={'value':    0.00000e+00, 'min': 0.0, 'max': 10 , 'fixed': 's N0 1'}
conf['params']['sivers']['ub N1 1'] ={'value':    0.00000e+00, 'min': 0.0, 'max': 1.0, 'fixed': 's N1 1'}
conf['params']['sivers']['ub a0 1'] ={'value':    0.00000e+00, 'min':-2.0, 'max':10.0, 'fixed': 's a0 1'}
conf['params']['sivers']['ub a1 1'] ={'value':    0.00000e+00, 'min':-2.0, 'max':10.0, 'fixed': 's a1 1'}
conf['params']['sivers']['ub b0 1'] ={'value':    0.00000e+00, 'min': 0.0, 'max':10.0, 'fixed': 's b0 1'}
conf['params']['sivers']['ub b1 1'] ={'value':    0.00000e+00, 'min': 0.0, 'max':10.0, 'fixed': 's b1 1'}
conf['params']['sivers']['ub c0 1'] ={'value':    0.00000e+00, 'min':-1.0, 'max': 1.0, 'fixed': 's c0 1'}
conf['params']['sivers']['ub c1 1'] ={'value':    0.00000e+00, 'min':-1.0, 'max': 1.0, 'fixed': 's c1 1'}
conf['params']['sivers']['ub d0 1'] ={'value':    0.00000e+00, 'min':-1.0, 'max': 1.0, 'fixed': 's d0 1'}
conf['params']['sivers']['ub d1 1'] ={'value':    0.00000e+00, 'min':-1.0, 'max': 1.0, 'fixed': 's d1 1'}
conf['params']['sivers']['ub N0 2'] ={'value':    0.00000e+00, 'min': 0.0, 'max': 1.0, 'fixed': 's N0 2'}
conf['params']['sivers']['ub N1 2'] ={'value':    0.00000e+00, 'min': 0.0, 'max': 1.0, 'fixed': 's N1 2'}
conf['params']['sivers']['ub a0 2'] ={'value':    0.00000e+00, 'min':-2.0, 'max':10.0, 'fixed': 's a0 2'}
conf['params']['sivers']['ub a1 2'] ={'value':    0.00000e+00, 'min':-2.0, 'max':10.0, 'fixed': 's a1 2'}
conf['params']['sivers']['ub b0 2'] ={'value':    0.00000e+00, 'min': 0.0, 'max':10.0, 'fixed': 's b0 2'}
conf['params']['sivers']['ub b1 2'] ={'value':    0.00000e+00, 'min': 0.0, 'max':10.0, 'fixed': 's b1 2'}
conf['params']['sivers']['ub c0 2'] ={'value':    0.00000e+00, 'min':-1.0, 'max': 1.0, 'fixed': 's c0 2'}
conf['params']['sivers']['ub c1 2'] ={'value':    0.00000e+00, 'min':-1.0, 'max': 1.0, 'fixed': 's c1 2'}
conf['params']['sivers']['ub d0 2'] ={'value':    0.00000e+00, 'min':-1.0, 'max': 1.0, 'fixed': 's d0 2'}
conf['params']['sivers']['ub d1 2'] ={'value':    0.00000e+00, 'min':-1.0, 'max': 1.0, 'fixed': 's d1 2'}

conf['params']['sivers']['db N0 1'] ={'value':    0.00000e+00, 'min': 0.0, 'max': 10 , 'fixed': 's N0 1'}
conf['params']['sivers']['db N1 1'] ={'value':    0.00000e+00, 'min': 0.0, 'max': 1.0, 'fixed': 's N1 1'}
conf['params']['sivers']['db a0 1'] ={'value':    0.00000e+00, 'min':-2.0, 'max':10.0, 'fixed': 's a0 1'}
conf['params']['sivers']['db a1 1'] ={'value':    0.00000e+00, 'min':-2.0, 'max':10.0, 'fixed': 's a1 1'}
conf['params']['sivers']['db b0 1'] ={'value':    0.00000e+00, 'min': 0.0, 'max':10.0, 'fixed': 's b0 1'}
conf['params']['sivers']['db b1 1'] ={'value':    0.00000e+00, 'min': 0.0, 'max':10.0, 'fixed': 's b1 1'}
conf['params']['sivers']['db c0 1'] ={'value':    0.00000e+00, 'min':-1.0, 'max': 1.0, 'fixed': 's c0 1'}
conf['params']['sivers']['db c1 1'] ={'value':    0.00000e+00, 'min':-1.0, 'max': 1.0, 'fixed': 's c1 1'}
conf['params']['sivers']['db d0 1'] ={'value':    0.00000e+00, 'min':-1.0, 'max': 1.0, 'fixed': 's d0 1'}
conf['params']['sivers']['db d1 1'] ={'value':    0.00000e+00, 'min':-1.0, 'max': 1.0, 'fixed': 's d1 1'}
conf['params']['sivers']['db N0 2'] ={'value':    0.00000e+00, 'min': 0.0, 'max': 1.0, 'fixed': 's N0 2'}
conf['params']['sivers']['db N1 2'] ={'value':    0.00000e+00, 'min': 0.0, 'max': 1.0, 'fixed': 's N1 2'}
conf['params']['sivers']['db a0 2'] ={'value':    0.00000e+00, 'min':-2.0, 'max':10.0, 'fixed': 's a0 2'}
conf['params']['sivers']['db a1 2'] ={'value':    0.00000e+00, 'min':-2.0, 'max':10.0, 'fixed': 's a1 2'}
conf['params']['sivers']['db b0 2'] ={'value':    0.00000e+00, 'min': 0.0, 'max':10.0, 'fixed': 's b0 2'}
conf['params']['sivers']['db b1 2'] ={'value':    0.00000e+00, 'min': 0.0, 'max':10.0, 'fixed': 's b1 2'}
conf['params']['sivers']['db c0 2'] ={'value':    0.00000e+00, 'min':-1.0, 'max': 1.0, 'fixed': 's c0 2'}
conf['params']['sivers']['db c1 2'] ={'value':    0.00000e+00, 'min':-1.0, 'max': 1.0, 'fixed': 's c1 2'}
conf['params']['sivers']['db d0 2'] ={'value':    0.00000e+00, 'min':-1.0, 'max': 1.0, 'fixed': 's d0 2'}
conf['params']['sivers']['db d1 2'] ={'value':    0.00000e+00, 'min':-1.0, 'max': 1.0, 'fixed': 's d1 2'}

conf['params']['sivers']['sb N0 1'] ={'value':    0.00000e+00, 'min': 0.0, 'max': 10 , 'fixed': 's N0 1'}
conf['params']['sivers']['sb N1 1'] ={'value':    0.00000e+00, 'min': 0.0, 'max': 1.0, 'fixed': 's N1 1'}
conf['params']['sivers']['sb a0 1'] ={'value':    0.00000e+00, 'min':-2.0, 'max':10.0, 'fixed': 's a0 1'}
conf['params']['sivers']['sb a1 1'] ={'value':    0.00000e+00, 'min':-2.0, 'max':10.0, 'fixed': 's a1 1'}
conf['params']['sivers']['sb b0 1'] ={'value':    0.00000e+00, 'min': 0.0, 'max':10.0, 'fixed': 's b0 1'}
conf['params']['sivers']['sb b1 1'] ={'value':    0.00000e+00, 'min': 0.0, 'max':10.0, 'fixed': 's b1 1'}
conf['params']['sivers']['sb c0 1'] ={'value':    0.00000e+00, 'min':-1.0, 'max': 1.0, 'fixed': 's c0 1'}
conf['params']['sivers']['sb c1 1'] ={'value':    0.00000e+00, 'min':-1.0, 'max': 1.0, 'fixed': 's c1 1'}
conf['params']['sivers']['sb d0 1'] ={'value':    0.00000e+00, 'min':-1.0, 'max': 1.0, 'fixed': 's d0 1'}
conf['params']['sivers']['sb d1 1'] ={'value':    0.00000e+00, 'min':-1.0, 'max': 1.0, 'fixed': 's d1 1'}
conf['params']['sivers']['sb N0 2'] ={'value':    0.00000e+00, 'min': 0.0, 'max': 1.0, 'fixed': 's N0 2'}
conf['params']['sivers']['sb N1 2'] ={'value':    0.00000e+00, 'min': 0.0, 'max': 1.0, 'fixed': 's N1 2'}
conf['params']['sivers']['sb a0 2'] ={'value':    0.00000e+00, 'min':-2.0, 'max':10.0, 'fixed': 's a0 2'}
conf['params']['sivers']['sb a1 2'] ={'value':    0.00000e+00, 'min':-2.0, 'max':10.0, 'fixed': 's a1 2'}
conf['params']['sivers']['sb b0 2'] ={'value':    0.00000e+00, 'min': 0.0, 'max':10.0, 'fixed': 's b0 2'}
conf['params']['sivers']['sb b1 2'] ={'value':    0.00000e+00, 'min': 0.0, 'max':10.0, 'fixed': 's b1 2'}
conf['params']['sivers']['sb c0 2'] ={'value':    0.00000e+00, 'min':-1.0, 'max': 1.0, 'fixed': 's c0 2'}
conf['params']['sivers']['sb c1 2'] ={'value':    0.00000e+00, 'min':-1.0, 'max': 1.0, 'fixed': 's c1 2'}
conf['params']['sivers']['sb d0 2'] ={'value':    0.00000e+00, 'min':-1.0, 'max': 1.0, 'fixed': 's d0 2'}
conf['params']['sivers']['sb d1 2'] ={'value':    0.00000e+00, 'min':-1.0, 'max': 1.0, 'fixed': 's d1 2'}
############################################################################
# set data sets

conf['datasets']={}

# SIDIS

conf['datasets']['sidis']={}

conf['datasets']['sidis']['norm']={}
conf['datasets']['sidis']['xlsx']={}

conf['datasets']['sidis']['xlsx'][2000]='sidis/expdata/2000.xlsx' # proton   | pi+    | AUTsivers  | hermes     | PT
conf['datasets']['sidis']['xlsx'][2001]='sidis/expdata/2001.xlsx' # proton   | pi+    | AUTsivers  | hermes     | x
conf['datasets']['sidis']['xlsx'][2002]='sidis/expdata/2002.xlsx' # proton   | pi+    | AUTsivers  | hermes     | z
conf['datasets']['sidis']['xlsx'][2003]='sidis/expdata/2003.xlsx' # proton   | pi-    | AUTsivers  | hermes     | PT
conf['datasets']['sidis']['xlsx'][2004]='sidis/expdata/2004.xlsx' # proton   | pi-    | AUTsivers  | hermes     | x
conf['datasets']['sidis']['xlsx'][2005]='sidis/expdata/2005.xlsx' # proton   | pi-    | AUTsivers  | hermes     | z
conf['datasets']['sidis']['xlsx'][2006]='sidis/expdata/2006.xlsx' # proton   | pi0    | AUTsivers  | hermes     | PT
conf['datasets']['sidis']['xlsx'][2007]='sidis/expdata/2007.xlsx' # proton   | pi0    | AUTsivers  | hermes     | x
conf['datasets']['sidis']['xlsx'][2008]='sidis/expdata/2008.xlsx' # proton   | pi0    | AUTsivers  | hermes     | z

conf['datasets']['sidis']['xlsx'][2020]='sidis/expdata/2020.xlsx' # proton | h+    | AUTsivers  | compass    | PT
conf['datasets']['sidis']['xlsx'][2021]='sidis/expdata/2021.xlsx' # proton | h+    | AUTsivers  | compass    | x
conf['datasets']['sidis']['xlsx'][2022]='sidis/expdata/2022.xlsx' # proton | h+    | AUTsivers  | compass    | z
conf['datasets']['sidis']['xlsx'][2023]='sidis/expdata/2023.xlsx' # proton | h-    | AUTsivers  | compass    | PT
conf['datasets']['sidis']['xlsx'][2024]='sidis/expdata/2024.xlsx' # proton | h-    | AUTsivers  | compass    | x
conf['datasets']['sidis']['xlsx'][2025]='sidis/expdata/2025.xlsx' # proton | h-    | AUTsivers  | compass    | z
conf['datasets']['sidis']['xlsx'][2026]='sidis/expdata/2026.xlsx' # deuteron | pi+    | AUTsivers  | compass    | PT
conf['datasets']['sidis']['xlsx'][2027]='sidis/expdata/2027.xlsx' # deuteron | pi+    | AUTsivers  | compass    | x
conf['datasets']['sidis']['xlsx'][2028]='sidis/expdata/2028.xlsx' # deuteron | pi+    | AUTsivers  | compass    | z
conf['datasets']['sidis']['xlsx'][2029]='sidis/expdata/2029.xlsx' # deuteron | pi-    | AUTsivers  | compass    | PT
conf['datasets']['sidis']['xlsx'][2030]='sidis/expdata/2030.xlsx' # deuteron | pi-    | AUTsivers  | compass    | x
conf['datasets']['sidis']['xlsx'][2031]='sidis/expdata/2031.xlsx' # deuteron | pi-    | AUTsivers  | compass    | z

conf['datasets']['sidis']['xlsx'][2046]='sidis/expdata/2046.xlsx' # proton | h-    | AUTsivers  | compass    | PT
conf['datasets']['sidis']['xlsx'][2047]='sidis/expdata/2047.xlsx' # proton | h-    | AUTsivers  | compass    | x
conf['datasets']['sidis']['xlsx'][2048]='sidis/expdata/2048.xlsx' # proton | h-    | AUTsivers  | compass    | z
conf['datasets']['sidis']['xlsx'][2049]='sidis/expdata/2049.xlsx' # proton | h+    | AUTsivers  | compass    | PT
conf['datasets']['sidis']['xlsx'][2050]='sidis/expdata/2050.xlsx' # proton | h+    | AUTsivers  | compass    | x
conf['datasets']['sidis']['xlsx'][2051]='sidis/expdata/2051.xlsx' # proton | h+    | AUTsivers  | compass    | z


for k in conf['datasets']['sidis']['xlsx']: conf['datasets']['sidis']['norm'][k]={'value':1,'fixed':True,'min':0,'max':1}

conf['datasets']['sidis']['filters']={}
#conf['datasets']['sidis']['filters'][0]={}
#conf['datasets']['sidis']['filters'][0]['idx']=[2000,2001,2002,2003,2004,2005,2026,2027,2028,2029,2030,2031]
conf['datasets']['sidis']['filters']=["z>0.2 and z<0.6 and Q2>1.69 and pT>0.2 and pT<0.9"]



#conf["datasets"]["sidis"]["xlsx"][2015]="../database/sidis/expdata/2015.xlsx" # neutron  | pi+    | AUTsivers  | jlab       | x
#conf["datasets"]["sidis"]["xlsx"][2016]="../database/sidis/expdata/2016.xlsx" # neutron  | pi-    | AUTsivers  | jlab       | x

#conf["datasets"]["sidis"]["xlsx"][2006]="../database/sidis/expdata/2006.xlsx" # proton   | pi0    | AUTsivers  | hermes     | PT
#conf["datasets"]["sidis"]["xlsx"][2007]="../database/sidis/expdata/2007.xlsx" # proton   | pi0    | AUTsivers  | hermes     | x
#conf["datasets"]["sidis"]["xlsx"][2008]="../database/sidis/expdata/2008.xlsx" # proton   | pi0    | AUTsivers  | hermes     | z

#conf["datasets"]["sidis"]["xlsx"][2009]="../database/sidis/expdata/2009.xlsx" # proton   | k+     | AUTsivers  | hermes     | PT
#conf["datasets"]["sidis"]["xlsx"][2010]="../database/sidis/expdata/2010.xlsx" # proton   | k+     | AUTsivers  | hermes     | x
#conf["datasets"]["sidis"]["xlsx"][2011]="../database/sidis/expdata/2011.xlsx" # proton   | k+     | AUTsivers  | hermes     | z
#conf["datasets"]["sidis"]["xlsx"][2012]="../database/sidis/expdata/2012.xlsx" # proton   | k-     | AUTsivers  | hermes     | PT
#conf["datasets"]["sidis"]["xlsx"][2013]="../database/sidis/expdata/2013.xlsx" # proton   | k-     | AUTsivers  | hermes     | x
#conf["datasets"]["sidis"]["xlsx"][2014]="../database/sidis/expdata/2014.xlsx" # proton   | k-     | AUTsivers  | hermes     | z
#conf["datasets"]["sidis"]["xlsx"][2017]="../database/sidis/expdata/2017.xlsx" # proton   | k0     | AUTsivers  | compass    | PT
#conf["datasets"]["sidis"]["xlsx"][2018]="../database/sidis/expdata/2018.xlsx" # proton   | k0     | AUTsivers  | compass    | x
#conf["datasets"]["sidis"]["xlsx"][2019]="../database/sidis/expdata/2019.xlsx" # proton   | k0     | AUTsivers  | compass    | z
#conf["datasets"]["sidis"]["xlsx"][2032]="../database/sidis/expdata/2032.xlsx" # deuteron | k+     | AUTsivers  | compass    | PT
#conf["datasets"]["sidis"]["xlsx"][2033]="../database/sidis/expdata/2033.xlsx" # deuteron | k+     | AUTsivers  | compass    | x
#conf["datasets"]["sidis"]["xlsx"][2034]="../database/sidis/expdata/2034.xlsx" # deuteron | k+     | AUTsivers  | compass    | z
#conf["datasets"]["sidis"]["xlsx"][2035]="../database/sidis/expdata/2035.xlsx" # deuteron | k-     | AUTsivers  | compass    | PT
#conf["datasets"]["sidis"]["xlsx"][2036]="../database/sidis/expdata/2036.xlsx" # deuteron | k-     | AUTsivers  | compass    | x
#conf["datasets"]["sidis"]["xlsx"][2037]="../database/sidis/expdata/2037.xlsx" # deuteron | k-     | AUTsivers  | compass    | z
#conf["datasets"]["sidis"]["xlsx"][2038]="../database/sidis/expdata/2038.xlsx" # neutron  | k+     | AUTsivers  | jlab       | x
#conf["datasets"]["sidis"]["xlsx"][2039]="../database/sidis/expdata/2039.xlsx" # neutron  | k-     | AUTsivers  | jlab       | x

#conf["datasets"]["sidis"]["xlsx"][2500]="../database/sidis/expdata/2500.xlsx" # neutron  | pi+    | AUTsivers  | solid      | x
#conf["datasets"]["sidis"]["xlsx"][2501]="../database/sidis/expdata/2501.xlsx" # neutron  | pi-    | AUTsivers  | solid      | x
#conf["datasets"]["sidis"]["xlsx"][2502]="../database/sidis/expdata/2502.xlsx" # proton   | pi+    | AUTsivers  | solid      | x
#conf["datasets"]["sidis"]["xlsx"][2503]="../database/sidis/expdata/2503.xlsx" # proton   | pi-    | AUTsivers  | solid      | x
#conf["datasets"]["sidis"]["xlsx"][2504]="../database/sidis/expdata/2504.xlsx" # proton   | pi+    | AUTsivers  | clas12     | x
#conf["datasets"]["sidis"]["xlsx"][2505]="../database/sidis/expdata/2505.xlsx" # proton   | pi-    | AUTsivers  | clas12     | x
#conf["datasets"]["sidis"]["xlsx"][2506]="../database/sidis/expdata/2506.xlsx" # neutron  | pi+    | AUTsivers  | sbs        | x
#conf["datasets"]["sidis"]["xlsx"][2507]="../database/sidis/expdata/2507.xlsx" # neutron  | pi-    | AUTsivers  | sbs        | x
#conf["datasets"]["sidis"]["xlsx"][2508]="../database/sidis/expdata/2508.xlsx" # neutron  | pi+    | AUTsivers  | solid stat | x
#conf["datasets"]["sidis"]["xlsx"][2509]="../database/sidis/expdata/2509.xlsx" # neutron  | pi-    | AUTsivers  | solid stat | x
#conf["datasets"]["sidis"]["xlsx"][2510]="../database/sidis/expdata/2510.xlsx" # proton   | pi+    | AUTsivers  | solid stat | x
#conf["datasets"]["sidis"]["xlsx"][2511]="../database/sidis/expdata/2511.xlsx" # proton   | pi-    | AUTsivers  | solid stat | x
