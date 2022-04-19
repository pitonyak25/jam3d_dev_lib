#!/usr/bin/env python2
# -*- coding: utf-8 -*-
import sys
import os
import numpy as np
import pandas as pd
import math
import time
from tools.tools import load_config
from qcdlib.aux import AUX
from tools.config import conf
from scipy.integrate import quad, dblquad, fixed_quad

#AN_theory.py - program to calculate A_N in pp -> hX

flavor = ['g','u','ub','d','db','s','sb']
    #  0  1  2  3  4  5  6
target = ['p']
hadron = ['pi+','pi-','pi0']

flavdict = {'g': 0, 'u': 1, 'ub': 2,'d': 3, 'db': 4, 's': 5, 'sb': 6}

# Common color factors and fractions
c = {'r3': 1. / 3., 'r4': 0.25, 'r6': 1. / 6., 'r8': 0.125,
         'r9': 1. / 9., 'r18': 1. / 18., 'r24': 1. / 24., 'r27': 1. / 27.}

m = {}
Hupol = {}
f = {}
ft = {}
d = {}
h = {}
H1p = {}
H = {}
F1p = {}
HTffa = np.zeros(13)
HTffb = np.zeros(13)
Hxxpz = np.zeros((13, 7))

if 'basis' not in conf:
    conf['basis'] = 'default'

def get_f(x, Q2): # Collinear unpolarized PDF
  return conf['pdf'].get_C(x, Q2)

def get_ft(x, Q2): # Collinear unpolarized PDF
  return conf['pdf'].get_C(x, Q2)

def get_d(z, Q2, had): # Collinear unpolarized FF
  if 'pi' in had:
      return conf['ffpi'].get_C(z, Q2)
  elif 'k' in had:
      return conf['ffk'].get_C(z, Q2)

def get_h(x, Q2): # Collinear transversity
  return conf['transversity'].get_C(x, Q2)

# (H_1^{\perp(1)}(z) - z*dH_1^{\perp(1)}(z)/dz)
def get_H1p(z, Q2, had):
  if 'pi' in had:
      return conf['collinspi'].get_C(z, Q2) - z * conf['collinspi'].get_dC(z, Q2)
  elif 'k' in had:
      return conf['collinsk'].get_C(z, Q2) - z * conf['collinsk'].get_dC(z, Q2)

def get_H(z, Q2, had): # -2*z*H_1^{\perp(1)}(z)+\tilde{H}(z)
  if 'pi' in had:
      return -2. * z * conf['collinspi'].get_C(z,Q2) + conf['Htildepi'].get_C(z, Q2)
  elif 'k' in had:
      return -2. * z * conf['collinsk'].get_C(z,Q2) + conf['Htildek'].get_C(z, Q2)


# (f_1T^{\perp(1)}(x) - x*df_1T^{\perp(1)}(x)/dx) Check this for correct Sivers implementation
def get_f1Tp( x, Q2):
    return conf['sivers'].get_C(x, Q2) - x * conf['sivers'].get_dC(x, Q2)

def get_mandelstam(s, t, u):
# Convenient combinations of the partonic Mandelstam variables
   m['s2'] = s * s
   m['s3'] = s**3.
   m['t2'] = t * t
   m['t3'] = t**3.
   m['u2'] = u * u
   m['u3'] = u**3.
   m['ostu'] = 1. / (s * t * u)
   m['os'] = 1. / s
   m['ot'] = 1. / t
   m['ou'] = 1. / u
   m['st'] = s / t
   m['su'] = s / u
   m['ts'] = t / s
   m['tu'] = t / u
   m['us'] = u / s
   m['ut'] = u / t
   m['st2'] = s**2. / t**2.
   m['su2'] = s**2. / u**2.
   m['ts2'] = t**2. / s**2.
   m['tu2'] = t**2. / u**2.
   m['us2'] = u**2. / s**2.
   m['ut2'] = u**2. / t**2.
   m['os2'] = 1. / s**2.
   m['ot2'] = 1. / t**2.
   m['ou2'] = 1. / u**2.
   m['os3'] = 1. / s**3.
   m['ot3'] = 1. / t**3.
   m['ou3'] = 1. / u**3.
   return m

def get_Hupol(m):
  # Hard parts for the unpolarized cross section
   Hupol[1] = 4. * (m['st2'] + m['ut2']) * c['r9']
   Hupol[2] = 4. * (m['su2'] + m['tu2']) * c['r9']
   Hupol[3] = -8. * m['st'] * m['su'] * c['r27']
   Hupol[4] = 4. * (m['ts2'] + m['us2']) * c['r9']
   Hupol[5] = 4. * (m['st2'] + m['ut2']) * c['r9']
   Hupol[6] = -8. * m['ut'] * m['us'] * c['r27']
   Hupol[7] = 4. * (m['ts2'] + m['us2']) * c['r9']
   Hupol[8] = 4. * (m['su2'] + m['tu2']) * c['r9']
   Hupol[9] = -8. * m['tu'] * m['ts'] * c['r27']
   Hupol[10] = 32. * c['r27'] * \
    (m['tu'] + m['ut']) * \
    (1. - 9. * m['ts'] * m['us'] * c['r4'])
   Hupol[11] = 4. * c['r9'] * (- m['su'] - m['us']) * (
    1. - 9. * m['st'] * m['ut'] * c['r4'])
   Hupol[12] = 4. * c['r9'] * (- m['st'] - m['ts']) * (
    1. - 9. * m['su'] * m['tu'] * c['r4'])
   Hupol[13] = (m['tu'] + m['ut']) * c['r6'] - \
    3. * (m['ts2'] + m['us2']) * c['r8']
   Hupol[14] = 4.5 * (3. - m['ts'] * m['us'] -
              m['st'] * m['ut'] - m['su'] * m['tu'])
   return Hupol

def get_HTffa(m, s, t, u):
  # Hard parts for the transversely polarized fragmentation term
   HTffa[0] = 0
   HTffa[1] = - c['r9'] * m['ot'] + c['r8'] * \
    s * (u - s) * m['ot3'] - m['st2'] * m['ou']
   HTffa[2] = c['r27'] * s * (t - u) * m['ot2'] * m['ou'] + \
     c['r9'] * s * (u - 2. * t) * m['ot3'] + s * m['ot2']
   HTffa[3] = c['r27'] * s * m['ot2'] + c['r9'] * \
    s * (t - s) * m['ot3'] - c['r3'] * m['ot']
   HTffa[4] = c['r27'] * s * m['ot'] * \
     m['ou'] - c['r3'] * m['ot']
   HTffa[5] = c['r9'] * s * (u - 2. * t) * m['ot3'] + s * m['ot2']
   HTffa[6] = c['r9'] * s * (t - s) * m['ot3']
   HTffa[7] = HTffa[1]
   HTffa[8] = HTffa[2]
   HTffa[9] = HTffa[3]
   HTffa[10] = HTffa[4]
   HTffa[11] = HTffa[5]
   HTffa[12] = HTffa[6]
   return HTffa

def get_HTffb(m, s, t, u):
  # Hard parts for the transversely polarized fragmentation term
   HTffb[0] = 0
   HTffb[1] = c['r8'] * s * (u - s) * m['ot3'] + 0.5 * c['r9'] * (s - u) * m['ot'] * \
     m['ou'] + 0.5 * (s - u) * (m['t2'] - 2. * t *
                 u - 2. * m['u2']) * m['ot3'] * m['ou']
   HTffb[2] = c['r27'] * 0.5 * s * (t - 3. * u) * m['ot2'] * m['ou'] - s * u * m['ot3'] + c['r9'] * s * (
    2. * u - t) * m['ot3'] - c['r3'] * 0.5 * m['s2'] * m['ot2'] * m['ou']
   HTffb[3] = c['r27'] * 0.5 * (3. * s - t) * m['ot2'] + m['s2'] * m['ot3'] + \
     c['r9'] * s * (t - 2. * s) * m['ot3'] + \
     c['r3'] * 0.5 * u * m['ot2']
   HTffb[4] = 10. * c['r27'] * 0.5 * (s - u) * m['ot'] * m['ou']
   HTffb[5] = c['r9'] * s * \
    (2. * u - t) * m['ot3'] - s * u * m['ot3']
   HTffb[6] = c['r9'] * s * \
    (t - 2. * s) * m['ot3'] + m['s2'] * m['ot3']
   HTffb[7] = HTffb[1]
   HTffb[8] = HTffb[2]
   HTffb[9] = HTffb[3]
   HTffb[10] = HTffb[4]
   HTffb[11] = HTffb[5]
   HTffb[12] = HTffb[6]
   return HTffb

def get_Hxxpz(z, Q2, had, m, s, t, u):
  HTffa = get_HTffa(m, s, t, u)
  HTffb = get_HTffb(m, s, t, u)

  H1p = get_H1p(z, Q2, 'pi+')
  H = get_H(z, Q2, 'pi+')

  if had.endswith('-'):
      H1p = conf['aux'].charge_conj(H1p)
      H = conf['aux'].charge_conj(H)

  elif had.endswith('0'):
      H1pp=H1p
      H1pm=conf['aux'].charge_conj(H1p)
      Hp=H
      Hm=conf['aux'].charge_conj(H)

      H1p=0.5*(H1pp+H1pm)
      #print z,2.0*z**2*0.135*H1pp,2.0*z**2*0.135*H1pm,2.0*z**2*0.135*H1p
      #sys.exit()
      H=0.5*(Hp+Hm)

  Hxxpz = np.einsum('i,j->ij', HTffa, H1p) + np.einsum('i,j->ij', HTffb, H) / z
  return Hxxpz

#  @profile
# Calculation of the unpolarized cross section
def get_dsig(x, z, xF, pT, rs, tar, had):

  Mh = {}
  Mh['pi+'] = conf['aux'].Mpi
  Mh['pi-'] = conf['aux'].Mpi
  Mh['pi0'] = conf['aux'].Mpi
  Mh['k+'] = conf['aux'].Mk
  Mh['k-'] = conf['aux'].Mk


  if pT > 1.:
    Q = pT
  else:
    Q = 1.

  Q2 = Q * Q

  xT = 2. * pT / rs
  xT2 = xT * xT
  xF2 = xF * xF

  # Mandelstam variables at the hadron level
  ss = rs * rs
  tt = -0.5 * ss * (np.sqrt(xF2 + xT2) - xF)
  uu = -0.5 * ss * (np.sqrt(xF2 + xT2) + xF)

  oz = 1. / z

  xp = -x * tt / (z * x * ss + uu)

  # Mandelstam variables at the parton level
  s = x * xp * ss
  t = x * tt * oz
  u = xp * uu * oz

  # Prefactor
  denfac = 1. / ((z * z * x * ss + uu * z) * x * xp)

  m=get_mandelstam(s, t, u)
  get_Hupol(m)

  Hupol1 = Hupol[1]
  Hupol2 = Hupol[2]
  Hupol3 = Hupol[3]
  Hupol4 = Hupol[4]
  Hupol5 = Hupol[5]
  Hupol6 = Hupol[6]
  Hupol7 = Hupol[7]
  Hupol8 = Hupol[8]
  Hupol9 = Hupol[9]
  Hupol10 = Hupol[10]
  Hupol11 = Hupol[11]
  Hupol12 = Hupol[12]
  Hupol13 = Hupol[13]
  Hupol14 = Hupol[14]

  # Get arrays of the nonperturbative functions
  f = get_f(x, Q2)
  ft = get_ft(xp, Q2)
  d = get_d(z, Q2, 'pi+')

  if had.endswith('-'):
      d = conf['aux'].charge_conj(d)

  elif had.endswith('0'):
      dp=d
      dm=conf['aux'].charge_conj(d)
      d=0.5*(dp+dm)
      #print z,z*dp,z*dm,z*d

  fg = f[0]
  fu = f[1]
  fub = f[2]
  fd = f[3]
  fdb = f[4]
  fs = f[5]
  fsb = f[6]

  ftg = ft[0]
  ftu = ft[1]
  ftub = ft[2]
  ftd = ft[3]
  ftdb = ft[4]
  fts = ft[5]
  ftsb = ft[6]

  dg = d[0]
  du = d[1]
  dub = d[2]
  dd = d[3]
  ddb = d[4]
  ds = d[5]
  dsb = d[6]

  upol = 0

  upol += (fu * du + fd * dd + fs * ds) * (ftu + ftd + fts) * Hupol1
  + (fu + fd + fs) * (ftu * du + ftd * dd + fts * ds) * Hupol2
  + (fu * ftu * du + fd * ftd * dd + fs * fts * ds) * Hupol3

  upol += (fub * dub + fdb * ddb + fsb * dsb) * (ftub + ftdb + ftsb) * Hupol1
  + (fub + fdb + fsb) * (ftub * dub + ftdb * ddb + ftsb * dsb) * Hupol2
  + (fub * ftub * dub + fdb * ftdb * ddb + fsb * ftsb * dsb) * Hupol3

  upol += (fu * du + fd * dd + fs * ds) * (ftub + ftdb + ftsb) * Hupol5
  + (fu * ftub + fd * ftdb + fs * ftsb) * (du + dd + ds) * Hupol4
  + (fu * ftub * du + fd * ftdb * dd + fs * ftsb * ds) * Hupol6

  upol += (fub * dub + fdb * ddb + fsb * dsb) * (ftu + ftd + fts) * Hupol5
  + (fub * ftu + fdb * ftd + fsb * fts) * (dub + ddb + dsb) * Hupol4
  + (fub * ftu * dub + fdb * ftd * ddb + fsb * fts * dsb) * Hupol6

  upol += (fu + fd + fs) * (ftub * dub + ftdb * ddb + ftsb * dsb) * Hupol8
  + (fu * ftub + fd * ftdb + fs * ftsb) * (dub + ddb + dsb) * Hupol7
  + (fu * ftub * dub + fd * ftdb * ddb + fs * ftsb * dsb) * Hupol9

  upol += (fub + fdb + fsb) * (ftu * du + ftd * dd + fts * ds) * Hupol8
  + (fub * ftu + fdb * ftd + fsb * fts) * (du + dd + ds) * Hupol7
  + (fub * ftu * du + fdb * ftd * dd + fsb * fts * ds) * Hupol9

  upol += (fu * ftub + fd * ftdb + fs * ftsb) * dg * Hupol10

  upol += (fub * ftu + fdb * ftd + fsb * fts) * dg * Hupol10

  upol += (fu * du + fd * dd + fs * ds) * ftg * Hupol11

  upol += (fub * dub + fdb * ddb + fsb * dsb) * ftg * Hupol11

  upol += (fu + fd + fs) * ftg * dg * Hupol12

  upol += (fub + fdb + fsb) * ftg * dg * Hupol12

  upol += fg * ftg * (du + dd + ds) * Hupol13

  upol += fg * ftg * (dub + ddb + dsb) * Hupol13

  upol += fg * ftg * dg * Hupol14

  upol += fg * (ftu * du + ftd * dd + fts * ds) * Hupol12

  upol += fg * (ftub * dub + ftdb * ddb + ftsb * dsb) * Hupol12

  upol += fg * (ftu + ftd + fts) * dg * Hupol11

  upol += fg * (ftub + ftdb + ftsb) * dg * Hupol11

  return denfac * upol

# @profile calculation of QS correction
def get_dsigP(self, x, z, xF, pT, rs, tar, had):
    if pT > 1.:
        Q = pT
    else:
        Q = 1.

    Q2 = Q * Q

    xT = 2. * pT / rs
    xT2 = xT * xT
    xF2 = xF * xF

    # Mandelstam variables at the hadron level
    ss = rs * rs
    tt = -0.5 * ss * (np.sqrt(xF2 + xT2) - xF)
    uu = -0.5 * ss * (np.sqrt(xF2 + xT2) + xF)

    oz = 1. / z

    xp = -x * tt / (z * x * ss + uu)

    # Mandelstam variables at the parton level
    s = x * xp * ss
    t = x * tt * oz
    u = xp * uu * oz

    m=get_mandelstam(s, t, u)

    # Prefactor
    denfac = 1. / ((z * z * x * ss + uu * z) * x * xp)

    #Mandelstam combinations
    fsi = 1. + m['ut']
    sig1 = -m['ou']*(2.+fsi)*(m['st2']+m['ut2'])*c['r18']

    sig2 = -m['ou']*(2.-7.*fsi)*(m['su2']+m['tu2'])*c['r18']
    sig3 = -m['ou']*(-10.-fsi)*m['st']*m['su']*c['r27']

    sig4 = -m['ou']*(7.+fsi)*(m['st2']+m['ut2'])*c['r18']
    sig5 = -m['ou']*(-1. -7.*fsi)*(m['us2']+m['ts2'])*c['r18']

    sig6 = -m['ou']*(-1.-fsi)*m['us']*m['ut']*c['r27']

    sig7 = -m['ou']*(7.-2.*fsi)*(m['su2']+m['tu2'])*c['r18']
    sig8 = -m['ou']*(-1. -2.*fsi)*(m['us2']+m['ts2'])*c['r18']
    sig9 = -m['ou']*(-1. -fsi)*m['ts']*m['tu']*c['r27']

    sig10 = -m['ou']*c['r6']*c['r9']*(m['tu']+m['ut'])*(1.+18.*m['ts']*m['us'])-m['ou']*fsi*c['r6']* \
    (m['tu']+m['ut'])*(1.-9.*(m['us'])*(m['us']))

    sig11 = -m['ou']*c['r4']*c['r4']*(m['su']+m['us'])*(1.-9.*m['ut']*m['ut'])-m['ou']*fsi*c['r8']* \
    c['r18']*(m['su']+m['us'])*(1.+18.*m['st']*m['ut'])

    sig12 = -m['ou']*c['r4']*c['r4']*(m['ts']+m['st'])*(1.-9.*m['tu']*m['tu']) + m['ou']*fsi*c['r4']* \
    c['r4']*(m['ts']+m['st'])*(1.-9.*m['su']*m['su'])

    #Arrays of pdfs and ffs
    f = get_f(x, Q2, tar)
    ft = get_ft(xp, Q2, tar)
    d = get_d(z, Q2, had)

    fg = f[0]
    fu = f[1]
    fub = f[2]
    fd = f[3]
    fdb = f[4]
    fs = f[5]
    fsb = f[6]

    ftg = ft[0]
    ftu = ft[1]
    ftub = ft[2]
    ftd = ft[3]
    ftdb = ft[4]
    fts = ft[5]
    ftsb = ft[6]

    dg = d[0]
    du = d[1]
    dub = d[2]
    dd = d[3]
    ddb = d[4]
    ds = d[5]
    dsb = d[6]

    F1p = get_F1p(x, Q2)

    gsgp = -2./np.pi * F1p[0]
    usgp = -2./np.pi * F1p[1]
    ubsgp = -2./np.pi * F1p[2]
    dsgp = -2./np.pi * F1p[3]
    dbsgp = -2./np.pi * F1p[4]
    ssgp = -2./np.pi * F1p[5]
    sbsgp = -2./np.pi * F1p[6]

    sigma1 = (usgp*du +dsgp*dd +ssgp*ds)*(fu+fd+fs)*sig1 + (usgp + dsgp +ssgp)*(fu*du+fd*dd+fs*ds)*sig2+(usgp*fu*du+dsgp*fd*dd+ssgp*fs*ds)*sig3

    sigma1b = (ubsgp*dub + dbsgp* ddb + sbsgp*dsb)*(fub+fdb+fsb)*sig1 +(ubsgp+dbsgp+sbsgp)*(fub*dub+fdb*ddb+fsb*dsb)*sig2+(ubsgp*fub*dub+dbsgp*fdb*ddb+sbsgp*fsb*dsb)*sig3

    sigma2 = (usgp*du + dsgp*dd+ ssgp*ds)*(fub + fdb +fsb)*sig4 + (usgp*fub+dsgp*dd+ssgp*ds)*(du+dd+ds)*sig5+(usgp*fub*du+dsgp*fdb*dd+ssgp*fsb*ds)*sig6

    sigma2b = (ubsgp*dub + dbsgp*ddb+sbsgp*dsb)*(fu+fd+fs)*sig4 + (ubsgp*fu + dbsgp*fd + sbsgp*fs)*(dub + ddb +dsb)*sig5+(ubsgp*fu*dub+dbsgp*fd*ddb+sbsgp*fs*dsb)*sig6

    sigma3 = (usgp +dsgp +ssgp)*(fub*dub + fdb*ddb + fsb*dsb)*sig7 + (usgp*fub + dsgp*fdb+ssgp*fsb)*(dub + ddb + dsb)*sig8 + (usgp*fub*dub+dsgp*fdb*ddb+ssgp*fsb*dsb)*sig9

    sigma3b = (ubsgp +dsgp + sbsgp)*(fu*du+fdb*ddb+fsb*dsb)*sig7 + (ubsgp*fu+dbsgp*fd+sbsgp*fs)*(du+dd+ds)*sig8 +(ubsgp*fu*du+dbsgp*fd*dd+sbsgp*fs*ds)*sig9

    sigma4 = (usgp*fub+dsgp*fdb +ssgp*fsb)*dg*sig10

    sigma4b = (ubsgp*fu + dbsgp*fd + sbsgp*fs)*dg*sig10

    sigma5 = (usgp*du + dsgp*dd + ssgp*ds)*fg*sig11

    sigma5b = (ubsgp*dub + dbsgp*ddb +sbsgp*dsb)*fg*sig11

    sigma6 = (usgp + dsgp +ssgp) *fg *dg*sig12

    sigma6b = (ubsgp+dbsgp+sbsgp)*fg*dg*sig12

    sgpcs= sigma1 + sigma2 + sigma3 + sigma4 + sigma5 + sigma6 + sigma1b + sigma2b + sigma3b + sigma4b + sigma5b + sigma6b

    return denfac*oz*sgpcs





#  @profile
# Calculation of the fragmentation term in the transversely polarized cross section
def get_dsigST(x, z, xF, pT, rs, tar, had):

  Mh = {}
  Mh['pi+'] = conf['aux'].Mpi
  Mh['pi-'] = conf['aux'].Mpi
  Mh['pi0'] = conf['aux'].Mpi
  Mh['k+'] = conf['aux'].Mk
  Mh['k-'] = conf['aux'].Mk

  Mh = Mh[had]

  if pT > 1.:
    Q = pT
  else:
    Q = 1.

  Q2 = Q * Q

  xT = 2. * pT / rs
  xT2 = xT * xT
  xF2 = xF * xF

  # Mandelstam variables at the hadron level
  ss = rs * rs
  tt = -0.5 * ss * (np.sqrt(xF2 + xT2) - xF)
  uu = -0.5 * ss * (np.sqrt(xF2 + xT2) + xF)

  oz = 1. / z

  xp = -x * tt / (z * x * ss + uu)

  # Mandelstam variables at the parton level
  s = x * xp * ss
  t = x * tt * oz
  u = xp * uu * oz

  # Prefactor
  numfac = oz * (1. / ((z * z * x * ss + uu * z) * x * xp))

  m=get_mandelstam(s, t, u)

  # Get arrays of the nonperturbative functions
  ft = get_ft(xp, Q2)
  h = get_h(x, Q2)
  Hxxpz = get_Hxxpz(z, Q2, had, m, s, t, u)

  hg = h[0]
  hu = h[1]
  hub = h[2]
  hd = h[3]
  hdb = h[4]
  hs = h[5]
  hsb = h[6]

  ftg = ft[0]
  ftu = ft[1]
  ftub = ft[2]
  ftd = ft[3]
  ftdb = ft[4]
  fts = ft[5]
  ftsb = ft[6]

  Hxxpz1u = Hxxpz[1][1]
  Hxxpz1d = Hxxpz[1][3]
  Hxxpz1s = Hxxpz[1][5]
  Hxxpz2u = Hxxpz[2][1]
  Hxxpz2d = Hxxpz[2][3]
  Hxxpz2s = Hxxpz[2][5]
  Hxxpz3u = Hxxpz[3][1]
  Hxxpz3d = Hxxpz[3][3]
  Hxxpz3s = Hxxpz[3][5]
  Hxxpz4u = Hxxpz[4][1]
  Hxxpz4d = Hxxpz[4][3]
  Hxxpz4s = Hxxpz[4][5]
  Hxxpz5u = Hxxpz[5][1]
  Hxxpz5d = Hxxpz[5][3]
  Hxxpz5s = Hxxpz[5][5]
  Hxxpz6u = Hxxpz[6][1]
  Hxxpz6d = Hxxpz[6][3]
  Hxxpz6s = Hxxpz[6][5]

  Hxxpz7ub = Hxxpz[7][2]
  Hxxpz7db = Hxxpz[7][4]
  Hxxpz7sb = Hxxpz[7][6]
  Hxxpz8ub = Hxxpz[8][2]
  Hxxpz8db = Hxxpz[8][4]
  Hxxpz8sb = Hxxpz[8][6]
  Hxxpz9ub = Hxxpz[9][2]
  Hxxpz9db = Hxxpz[9][4]
  Hxxpz9sb = Hxxpz[9][6]
  Hxxpz10ub = Hxxpz[10][2]
  Hxxpz10db = Hxxpz[10][4]
  Hxxpz10sb = Hxxpz[10][6]
  Hxxpz11ub = Hxxpz[11][2]
  Hxxpz11db = Hxxpz[11][4]
  Hxxpz11sb = Hxxpz[11][6]
  Hxxpz12ub = Hxxpz[12][2]
  Hxxpz12db = Hxxpz[12][4]
  Hxxpz12sb = Hxxpz[12][6]

  ffcs = 0

  ffcs += ftg * (hu * Hxxpz1u + hd * Hxxpz1d + hs * Hxxpz1s)

  ffcs += (hu * ftu * Hxxpz2u) + (hd * ftd * Hxxpz2d) + (hs * fts * Hxxpz2s)

  ffcs += (hu * ftub * Hxxpz3u) + (hd * ftdb * Hxxpz3d) + (hs * ftsb * Hxxpz3s)

  ffcs += (hub * ftu * Hxxpz4u) + (hdb * ftd * Hxxpz4d) + (hsb * fts * Hxxpz4s)

  ffcs += ftu * (hd * Hxxpz5d + hs * Hxxpz5s) + ftd * (hu * Hxxpz5u + hs * Hxxpz5s) + fts * (hu * Hxxpz5u + hd * Hxxpz5d)

  ffcs += ftub * (hd * Hxxpz6d + hs * Hxxpz6s) + ftdb * (hu * Hxxpz6u + hs * Hxxpz6s) + ftsb * (hu * Hxxpz6u + hd * Hxxpz6d)

  ffcs += ftg * (hub * Hxxpz7ub + hdb * Hxxpz7db + hsb * Hxxpz7sb)

  ffcs += (hub * ftub * Hxxpz8ub) + (hdb * ftdb * Hxxpz8db) + (hsb * ftsb * Hxxpz8sb)

  ffcs += (hub * ftu * Hxxpz9ub) + (hdb * ftd * Hxxpz9db) + (hsb * fts * Hxxpz9sb)

  ffcs += (hu * ftub * Hxxpz10ub) + (hd * ftdb * Hxxpz10db) + (hs * ftsb * Hxxpz10sb)

  ffcs += ftub * (hdb * Hxxpz11db + hsb * Hxxpz11sb) + ftdb * (hub * Hxxpz11ub + hsb * Hxxpz11sb) + fts * (hub * Hxxpz11ub + hdb * Hxxpz11db)

  ffcs += ftu * (hdb * Hxxpz12db + hsb * Hxxpz12sb) + ftd * (hub * Hxxpz12ub + hsb * Hxxpz12sb) + fts * (hub * Hxxpz12ub + hdb * Hxxpz12db)

  ffcs = 2. * Mh * pT * numfac * ffcs

  return ffcs

def get_sig(xF, pT, rs, tar, had, mode='gauss', nx=10, nz=10):
    xT = 2. * pT / rs
    xF2 = xF * xF
    xT2 = xT * xT

    # Mandelstam variables at the hadron level
    ss = rs * rs
    tt = -0.5 * ss * (np.sqrt(xF2 + xT2) - xF)
    uu = -0.5 * ss * (np.sqrt(xF2 + xT2) + xF)

    # Lower limits of the z and x integrations
    zmin = np.sqrt(xF2 + xT2)

    def xmin(z): return -uu / (z * ss + tt)

    if mode == 'gauss':
        dsigdzdx = np.vectorize(
            lambda x, z: get_dsig(x, z, xF, pT, rs, tar, had))
        dsigdz = np.vectorize(lambda z: fixed_quad(
            lambda x: dsigdzdx(x, z), xmin(z), 1, n=nx)[0])
        sig = fixed_quad(dsigdz, zmin, 1, n=nz)[0]
    elif mode == 'quad':
        sig = dblquad(lambda x, z: get_dsig(
            x, z, xF, pT, rs, tar, had), zmin, 1., xmin, lambda x: 1.)[0]
    return sig

def get_sigP(xF, pT, rs, tar, had, mode='gauss', nx=100, nz=100):
    xT = 2. * pT / rs
    xF2 = xF * xF
    xT2 = xT * xT

    # Mandelstam variables at the hadron level
    ss = rs * rs
    tt = -0.5 * ss * (np.sqrt(xF2 + xT2) - xF)
    uu = -0.5 * ss * (np.sqrt(xF2 + xT2) + xF)

    # Lower limits of the z and x integrations
    zmin = np.sqrt(xF2 + xT2)

    def xmin(z): return -uu / (z * ss + tt)

    if mode == 'gauss':
        dsigdzdx = np.vectorize(
            lambda x, z: get_dsigP( x, z, xF, pT, rs, tar, had))
        dsigdz = np.vectorize(lambda z: fixed_quad(
            lambda x: dsigdzdx(x, z), xmin(z), 1, n=nx)[0])
        sig = fixed_quad(dsigdz, zmin, 1, n=nz)[0]
    elif mode == 'quad':
        sig = dblquad(lambda x, z: get_dsigP(
            x, z, xF, pT, rs, tar, had), zmin, 1., xmin, lambda x: 1.)[0]
    return sig

def get_sigST(xF, pT, rs, tar, had, mode='gauss', nx=10, nz=10):
    xT = 2. * pT / rs
    xF2 = xF * xF
    xT2 = xT * xT

    # Mandelstam variables at the hadron level
    ss = rs * rs
    tt = -0.5 * ss * (np.sqrt(xF2 + xT2) - xF)
    uu = -0.5 * ss * (np.sqrt(xF2 + xT2) + xF)

    # Lower limits of the z and x integrations
    zmin = np.sqrt(xF2 + xT2)

    def xmin(z): return -uu / (z * ss + tt)

    if mode == 'gauss':
      dsigdzdx = np.vectorize(
        lambda x, z: get_dsigST(x, z, xF, pT, rs, tar, had))
      dsigdz = np.vectorize(lambda z: fixed_quad(
        lambda x: dsigdzdx(x, z), xmin(z), 1, n=nx)[0])
      sig = fixed_quad(dsigdz, zmin, 1, n=nz)[0]
    elif mode == 'quad':
      sig = dblquad(lambda x, z: get_dsigST(
        x, z, xF, pT, rs, tar, had), zmin, 1., xmin, lambda x: 1.)[0]
    return sig


if __name__ == '__main__':

    from qcdlib.ff0 import FF as FF0
    from qcdlib.ff1 import FF as FF1
    from qcdlib.pdf0 import PDF as PDF0
    from qcdlib.pdf1 import PDF as PDF1
    conf['aux']= AUX()
    conf['pdf']=PDF0()
    conf['collinspi']=FF1('pi')
    conf['collinsk']=FF1('k')
    conf['Htildepi']=FF1('pi')
    conf['Htildek']=FF1('k')
    conf['transversity']=PDF1()
    conf['ffpi']=FF0('pi')
    conf['ffk']=FF0('k')
    conf['sivers']=PDF1()

    rs = 200.
    tar = 'p'
    had = 'pi-'
    pT = 1.10
    xF = 0.2375
    xT = 2. * pT / rs
    xF2 = xF * xF
    xT2 = xT * xT

    # Mandelstam variables at the hadron level
    ss = rs * rs
    tt = -0.5 * ss * (np.sqrt(xF2 + xT2) - xF)
    uu = -0.5 * ss * (np.sqrt(xF2 + xT2) + xF)

    # Lower limits of the z and x integrations
    zmin = np.sqrt(xF2 + xT2)

    def xmin(z): return -uu / (z * ss + tt)

    def test():
        den = get_sig(xF, pT, rs, tar, had, mode='gauss', nx=100, nz=100)
        num = get_sigST(xF, pT, rs, tar, had,
                              mode='gauss', nx=100, nz=100)
        num2 = get_sigP(xF, pT, rs, tar, had,
                              mode='gauss', nx=100, nz=100)

        AN = (num + num2) / den
        print AN

#  from timeit import Timer
#  t = Timer("test()", "from __main__ import test")
#  print 't elapsed ',t.timeit(number=1)

#  def test2():
#    den = anthy.get_dsig(0.3,0.6,xF,pT,rs,tar,had)
#    num = anthy.get_dsigST(0.3,0.6,xF,pT,rs,tar,had)
#
#    print den,num
#
#  from timeit import Timer
#  t = Timer("test2()", "from __main__ import test2")
#  print 't elapsed ',t.timeit(number=1)

#  start = time.time()
#  print anthy.get_dsig(0.3,0.6,xF,pT,rs,tar,had)
#  print anthy.get_dsigST(0.3,0.6,xF,pT,rs,tar,had)
#  end = time.time()
#  print 'time=',(end-start)

    start = time.time()
    test()
    end = time.time()
    print 'time=', (end - start)

    # Integration of the numerator from xmin to 1 and from zmin to 1 (the values for xmin and zmin are above)

    # Integration of the denominator from xmin to 1 and from zmin to 1 (the values for xmin and zmin are above)
    #den = dblquad(lambda x,z: ANTHEORY().get_dsig(x,z,xF,pT,rs,tar,had),zmin,1.,xmin,lambda x: 1.)

    #AN = num[0]/den[0]
    # print(AN)
