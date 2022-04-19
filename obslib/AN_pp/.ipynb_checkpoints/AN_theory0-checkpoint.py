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
from numba import jit
from qcdlib import core,pdf0,ff0,pdf1,ff1


#AN_theory0.py - program to calculate A_N in pp -> hX
#This includes both the fragmentation term and the QS term (see 1701.09170)

#flavor = ['g','u','ub','d','db','s','sb']
         #  0   1   2    3    4   5    6

# Common color factors and fractions
# c = {'r3': 1. / 3., 'r4': 0.25, 'r6': 1. / 6., 'r8': 0.125,
#         'r9': 1. / 9., 'r18': 1. / 18., 'r24': 1. / 24., 'r27': 1. / 27.}


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

def get_H1p(z, Q2, had): #(H_1^{\perp(1)}(z) - z*dH_1^{\perp(1)}(z)/dz)
    if 'pi' in had:
        return conf['collinspi'].get_C(z, Q2) - z * conf['dcollinspi'].get_C(z, Q2)
    elif 'k' in had:
        return conf['collinsk'].get_C(z, Q2) - z * conf['dcollinsk'].get_C(z, Q2)

def get_H(z, Q2, had): # -2*z*H_1^{\perp(1)}(z)+\tilde{H}(z)
    if 'pi' in had:
        return -2. * z * conf['collinspi'].get_C(z,Q2) + conf['Htildepi'].get_C(z, Q2)
    elif 'k' in had:
        return -2. * z * conf['collinsk'].get_C(z,Q2) + conf['Htildek'].get_C(z, Q2)

def get_f1Tp(x, Q2): # (f_1T^{\perp(1)}(x) - x*df_1T^{\perp(1)}(x)/dx)
    
    return conf['sivers'].get_C(x, Q2) - x * conf['dsivers'].get_C(x, Q2)

# def get_mandelstam(s, t, u):
# # Convenient combinations of the partonic Mandelstam variables
#     m = {}
#     m['s2'] = s * s
#     m['s3'] = s**3.
#     m['t2'] = t * t
#     m['t3'] = t**3.
#     m['u2'] = u * u
#     m['u3'] = u**3.
#     m['ostu'] = 1. / (s * t * u)
#     m['os'] = 1. / s
#     m['ot'] = 1. / t
#     m['ou'] = 1. / u
#     m['st'] = s / t
#     m['su'] = s / u
#     m['ts'] = t / s
#     m['tu'] = t / u
#     m['us'] = u / s
#     m['ut'] = u / t
#     m['st2'] = s**2. / t**2.
#     m['su2'] = s**2. / u**2.
#     m['ts2'] = t**2. / s**2.
#     m['tu2'] = t**2. / u**2.
#     m['us2'] = u**2. / s**2.
#     m['ut2'] = u**2. / t**2.
#     m['os2'] = 1. / s**2.
#     m['ot2'] = 1. / t**2.
#     m['ou2'] = 1. / u**2.
#     m['os3'] = 1. / s**3.
#     m['ot3'] = 1. / t**3.
#     m['ou3'] = 1. / u**3.
    
#     return m

@jit(nopython=True)
def get_Hupol(s,t,u):
  # Hard parts for the unpolarized cross section
    Hupol = np.zeros(15)
    Hupol[1] = 4. * (s**2. / t**2. + u**2. / t**2.) * 1/9
    Hupol[2] = 4. * (s**2. / u**2. + t**2. / u**2.) * 1/9
    Hupol[3] = -8. * s/t * s/u * 1/27
    Hupol[4] = 4. * (t**2. / s**2. + u**2. / s**2.) * 1/9
    Hupol[5] = 4. * (s**2. / t**2. + u**2. / t**2.) * 1/9
    Hupol[6] = -8. * u/t * u/s * 1/27
    Hupol[7] = 4. * (t**2. / s**2. + u**2. / s**2.) * 1/9
    Hupol[8] = 4. * (s**2. / u**2. + t**2. / u**2.) * 1/9
    Hupol[9] = -8. * t/u * t/s * 1/27
    Hupol[10] = 32. * 1/27 * (t/u + u/t) * (1. - 9. * t/s * u/s * 1/4)
    Hupol[11] = 4. * 1/9 * (- s/u - u/s) * (1. - 9. * s/t * u/t * 1/4)
    Hupol[12] = 4. * 1/9 * (- s/t - t/s) * (1. - 9. * s/u * t/u * 1/4)
    Hupol[13] = (t/u + u/t) * 1/6 - 3. * (t**2. / s**2. + u**2. / s**2.) * 1/8
    Hupol[14] = 4.5 * (3. - t/s * u/s - s/t * u/t - s/u * t/u)

    return Hupol

@jit(nopython=True)
def get_HTffa(s, t, u):
  # Hard parts for the transversely polarized fragmentation term
    HTffa = np.zeros(13)
    HTffa[0] = 0
    HTffa[1] = - 1/9 * 1. / t + 1/8 * s * (u - s) * 1. / t**3. - s**2. / t**2. * 1. / u
    HTffa[2] = 1/27 * s * (t - u) * 1. / t**2. * 1. / u + 1/9 * s * (u - 2. * t) * 1. / t**3. + s * 1. / t**2.
    HTffa[3] = 1/27 * s * 1. / t**2. + 1/9 * s * (t - s) * 1. / t**3. - 1/3 * 1. / t
    HTffa[4] = 1/27 * s * 1. / t * 1. / u - 1/3 * 1. / t
    HTffa[5] = 1/9 * s * (u - 2. * t) * 1. / t**3. + s * 1. / t**2.
    HTffa[6] = 1/9 * s * (t - s) * 1. / t**3.
    HTffa[7] = HTffa[1]
    HTffa[8] = HTffa[2]
    HTffa[9] = HTffa[3]
    HTffa[10] = HTffa[4]
    HTffa[11] = HTffa[5]
    HTffa[12] = HTffa[6]
    
    return HTffa

@jit(nopython=True)
def get_HTffb(s, t, u):
  # Hard parts for the transversely polarized fragmentation term
    HTffb = np.zeros(13)
    HTffb[0] = 0
    HTffb[1] = 1/8 * s * (u - s) * 1. / t**3. + 0.5 * 1/9 * (s - u) * 1. / t * 1. / u + 0.5 * (s - u) * (t*t - 2. * t * u - 2. * u*u) * 1. / t**3. * 1. / u
    HTffb[2] = 1/27 * 0.5 * s * (t - 3. * u) * 1. / t**2. * 1. / u - s * u * 1. / t**3. + 1/9 * s * (2. * u - t) * 1. / t**3. - 1/3 * 0.5 * s*s * 1. / t**2. * 1. / u
    HTffb[3] = 1/27 * 0.5 * (3. * s - t) * 1. / t**2. + s*s * 1. / t**3. + 1/9 * s * (t - 2. * s) * 1. / t**3. + 1/3 * 0.5 * u * 1. / t**2.
    HTffb[4] = 10. * 1/27 * 0.5 * (s - u) * 1. / t * 1. / u
    HTffb[5] = 1/9 * s * (2. * u - t) * 1. / t**3. - s * u * 1. / t**3.
    HTffb[6] = 1/9 * s * (t - 2. * s) * 1. / t**3. + s*s * 1. / t**3.
    HTffb[7] = HTffb[1]
    HTffb[8] = HTffb[2]
    HTffb[9] = HTffb[3]
    HTffb[10] = HTffb[4]
    HTffb[11] = HTffb[5]
    HTffb[12] = HTffb[6]
    return HTffb

@jit(nopython=True)
def get_Hxxpz(z, Q2, had, s, t, u, HTffa, HTffb, H1p, H):

    if had=='jet':
      #These are dummy formulas so we don't get a runtime warning
      #We set the frag cross section to zero for jet
        H1p=np.ones(11)
        H=np.ones(11)

    if had.endswith('-'):
        H1p = core.charge_conj(H1p)
        H = core.charge_conj(H)

    elif had.endswith('0'):
        H1pp=H1p
        H1pm=core.charge_conj(H1p)
        Hp=H
        Hm=core.charge_conj(H)

        H1p=0.5*(H1pp+H1pm)
      #print z,2.0*z**2*0.135*H1pp,2.0*z**2*0.135*H1pm,2.0*z**2*0.135*H1p
      #sys.exit()
        H=0.5*(Hp+Hm)

    #print had,z,H1p[1],H1p[3]

    return HTffa,H1p,HTffb,H

@jit(nopython=True)
def get_HQS(s,t,u): #Note there is a 1/u in each hard factor compared to KQVY 2006
    HQS = np.zeros(13)
    
    fsi = 1. + u/t
 
    HQS[1] = -1./u*(2.+fsi)*(s**2. / t**2.+u**2. / t**2.)*1/18

    HQS[2] = -1./u*(2.-7.*fsi)*(s**2. / u**2.+t**2. / u**2.)*1/18
    HQS[3] = -1./u*(-10.-fsi)*s/t*s/u*1/27

    HQS[4] = -1./u*(7.+fsi)*(s**2. / t**2.+u**2. / t**2.)*1/18
    HQS[5] = -1./u*(-1. -7.*fsi)*(u**2. / s**2.+t**2. / s**2.)*1/18

    HQS[6] = -1./u*(-1.-fsi)*u/s*u/t*1/27

    HQS[7] = -1./u*(7.-2.*fsi)*(s**2. / u**2.+t**2. / u**2.)*1/18
    HQS[8] = -1./u*(-1. -2.*fsi)*(u**2. / s**2.+t**2. / s**2.)*1/18
    HQS[9] = -1./u*(-1. -fsi)*t/s*t/u*1/27

    HQS[10] = -1./u*1/6*1/9*(t/u+u/t)*(1.+18.*t/s*u/s)-1. / u*fsi*1/6*(t/u+u/t)*(1.-9.*(u/s)*(u/s))

    HQS[11] = -1./u*1/4*1/4*(s/u+u/s)*(1.-9.*u/t*u/t)-1. / u*fsi*1/8*1/18*(s/u+u/s)*(1.+18.*s/t*u/t)

    HQS[12] = -1./u*1/4*1/4*(t/s+s/t)*(1.-9.*t/u*t/u) + 1. / u*fsi*1/4*1/4*(t/s+s/t)*(1.-9.*s/u*s/u)

    return HQS

#  @profile
# Calculation of the unpolarized cross section
@jit(nopython=True)
def get_var(x, z, xF, pT, rs):

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
    numfac = oz * (1. / ((z * z * x * ss + uu * z) * x * xp))

    #m=get_mandelstam(s, t, u)
#     Hupol=get_Hupol(s,t,u)

#     Hupol1 = Hupol[1]
#     Hupol2 = Hupol[2]
#     Hupol3 = Hupol[3]
#     Hupol4 = Hupol[4]
#     Hupol5 = Hupol[5]
#     Hupol6 = Hupol[6]
#     Hupol7 = Hupol[7]
#     Hupol8 = Hupol[8]
#     Hupol9 = Hupol[9]
#     Hupol10 = Hupol[10]
#     Hupol11 = Hupol[11]
#     Hupol12 = Hupol[12]
#     Hupol13 = Hupol[13]
#     Hupol14 = Hupol[14]
    
    return denfac,numfac,s,t,u,Q2,xp

@jit(nopython=True)
def _get_dsig(x, z, xF, pT, rs, tar, had, f, ft, d): 
  # Get arrays of the nonperturbative functions
  #print x,xp
    
    var=get_var(x, z, xF, pT, rs)
    denfac=var[0]
    s,t,u=var[2],var[3],var[4]
    Q2,xp=var[5],var[6]
    
    Hupol=get_Hupol(s,t,u)

    if had=='jet': d=np.ones(11)

    if had.endswith('-'):
        d = core.charge_conj(d)

    elif had.endswith('0'):
        dp=d
        dm=core.charge_conj(d)
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

    upol += (fu * du + fd * dd + fs * ds) * (ftu + ftd + fts) * Hupol[1] \
    + (fu + fd + fs) * (ftu * du + ftd * dd + fts * ds) * Hupol[2] \
    + (fu * ftu * du + fd * ftd * dd + fs * fts * ds) * Hupol[3] 

    upol += (fub * dub + fdb * ddb + fsb * dsb) * (ftub + ftdb + ftsb) * Hupol[1] \
    + (fub + fdb + fsb) * (ftub * dub + ftdb * ddb + ftsb * dsb) * Hupol[2] \
    + (fub * ftub * dub + fdb * ftdb * ddb + fsb * ftsb * dsb) * Hupol[3]  

    upol += (fu * du + fd * dd + fs * ds) * (ftub + ftdb + ftsb) * Hupol[5] \
    + (fu * ftub + fd * ftdb + fs * ftsb) * (du + dd + ds) * Hupol[4] \
    + (fu * ftub * du + fd * ftdb * dd + fs * ftsb * ds) * Hupol[6] 

    upol += (fub * dub + fdb * ddb + fsb * dsb) * (ftu + ftd + fts) * Hupol[5] \
    + (fub * ftu + fdb * ftd + fsb * fts) * (dub + ddb + dsb) * Hupol[4] \
    + (fub * ftu * dub + fdb * ftd * ddb + fsb * fts * dsb) * Hupol[6] 

    upol += (fu + fd + fs) * (ftub * dub + ftdb * ddb + ftsb * dsb) * Hupol[8] \
    + (fu * ftub + fd * ftdb + fs * ftsb) * (dub + ddb + dsb) * Hupol[7] \
    + (fu * ftub * dub + fd * ftdb * ddb + fs * ftsb * dsb) * Hupol[9] 

    upol += (fub + fdb + fsb) * (ftu * du + ftd * dd + fts * ds) * Hupol[8] \
    + (fub * ftu + fdb * ftd + fsb * fts) * (du + dd + ds) * Hupol[7] \
    + (fub * ftu * du + fdb * ftd * dd + fsb * fts * ds) * Hupol[9] 

    upol += (fu * ftub + fd * ftdb + fs * ftsb) * dg * Hupol[10]

    upol += (fub * ftu + fdb * ftd + fsb * fts) * dg * Hupol[10]

    upol += (fu * du + fd * dd + fs * ds) * ftg * Hupol[11]

    upol += (fub * dub + fdb * ddb + fsb * dsb) * ftg * Hupol[11]

    upol += (fu + fd + fs) * ftg * dg * Hupol[12]

    upol += (fub + fdb + fsb) * ftg * dg * Hupol[12]

    upol += fg * ftg * (du + dd + ds) * Hupol[13]

    upol += fg * ftg * (dub + ddb + dsb) * Hupol[13]

    upol += fg * ftg * dg * Hupol[14]

    upol += fg * (ftu * du + ftd * dd + fts * ds) * Hupol[12]

    upol += fg * (ftub * dub + ftdb * ddb + ftsb * dsb) * Hupol[12]

    upol += fg * (ftu + ftd + fts) * dg * Hupol[11]

    upol += fg * (ftub + ftdb + ftsb) * dg * Hupol[11]

    return denfac * upol

#  @profile
# Calculation of the fragmentation term in the transversely polarized cross section
@jit(nopython=True)
def _get_dsigST(x, z, xF, pT, rs, tar, had, f, ft, h, d, f1Tp, Hxxpz):  
    
    var=get_var(x, z, xF, pT, rs)
    numfac=var[1]
    s,t,u=var[2],var[3],var[4]
    Q2,xp=var[5],var[6]

    #m=get_mandelstam(s, t, u)
    
    M=0.93891897
    
    if had.startswith('pi'): Mh=0.134
    elif had.startswith('k'): Mh = 0.497
    elif had=='jet': Mh=1 #This is a dummy formula so we don't get a runtime error

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

    fg = f[0]
    fu = f[1]
    fub = f[2]
    fd = f[3]
    fdb = f[4]
    fs = f[5]
    fsb = f[6]
    
    if had=='jet': d=np.ones(11)
        
    if had.endswith('-'):
        d = core.charge_conj(d)
    elif had.endswith('0'):
        dp=d
        dm=core.charge_conj(d)
        d=0.5*(dp+dm)

    dg = d[0]
    du = d[1]
    dub = d[2]
    dd = d[3]
    ddb = d[4]
    ds = d[5]
    dsb = d[6]

    uQS = f1Tp[1]
    ubQS = f1Tp[2]
    dQS = f1Tp[3]
    dbQS = f1Tp[4]
    sQS = f1Tp[5]
    sbQS = f1Tp[6]

  #Fragmentation term

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

    ffcs += ftub * (hdb * Hxxpz11db + hsb * Hxxpz11sb) + ftdb * (hub * Hxxpz11ub + hsb * Hxxpz11sb) + ftsb * (hub * Hxxpz11ub + hdb * Hxxpz11db)

    ffcs += ftu * (hdb * Hxxpz12db + hsb * Hxxpz12sb) + ftd * (hub * Hxxpz12ub + hsb * Hxxpz12sb) + fts * (hub * Hxxpz12ub + hdb * Hxxpz12db)

    ffcs = 2. * Mh * pT * numfac * ffcs

    if had=='jet': ffcs=0.0


  #Qiu-Sterman term

    HQS = get_HQS(s,t,u)
    sig1 = HQS[1]
    sig2 = HQS[2]
    sig3 = HQS[3]
    sig4 = HQS[4]
    sig5 = HQS[5]
    sig6 = HQS[6]
    sig7 = HQS[7]
    sig8 = HQS[8]
    sig9 = HQS[9]
    sig10 = HQS[10]
    sig11 = HQS[11]
    sig12 = HQS[12]

    QScs = 0

    QScs += (uQS*du +dQS*dd +sQS*ds)*(ftu+ftd+fts)*sig1 + (uQS + dQS +sQS)*(ftu*du+ftd*dd+fts*ds)*sig2 + (uQS*ftu*du+dQS*ftd*dd+sQS*fts*ds)*sig3

    QScs += (ubQS*dub + dbQS* ddb + sbQS*dsb)*(ftub+ftdb+ftsb)*sig1 + (ubQS+dbQS+sbQS)*(ftub*dub+ftdb*ddb+ftsb*dsb)*sig2 + (ubQS*ftub*dub+dbQS*ftdb*ddb+sbQS*ftsb*dsb)*sig3

    QScs += (uQS*du + dQS*dd+ sQS*ds)*(ftub + ftdb +ftsb)*sig4 + (uQS*ftub+dQS*ftdb+sQS*ftsb)*(du+dd+ds)*sig5 + (uQS*ftub*du+dQS*ftdb*dd+sQS*ftsb*ds)*sig6

    QScs += (ubQS*dub + dbQS*ddb+sbQS*dsb)*(ftu+ftd+fts)*sig4 + (ubQS*ftu + dbQS*ftd + sbQS*fts)*(dub + ddb +dsb)*sig5 + (ubQS*ftu*dub+dbQS*ftd*ddb+sbQS*fts*dsb)*sig6

    QScs += (uQS +dQS +sQS)*(ftub*dub + ftdb*ddb + ftsb*dsb)*sig7 + (uQS*ftub + dQS*ftdb+sQS*ftsb)*(dub + ddb + dsb)*sig8 + (uQS*ftub*dub+dQS*ftdb*ddb+sQS*ftsb*dsb)*sig9

    QScs += (ubQS +dbQS + sbQS)*(ftu*du+ftd*dd+fts*ds)*sig7 + (ubQS*ftu+dbQS*ftd+sbQS*fts)*(du+dd+ds)*sig8 +(ubQS*ftu*du+dbQS*ftd*dd+sbQS*fts*ds)*sig9

    QScs += (uQS*ftub+dQS*ftdb +sQS*ftsb)*dg*sig10

    QScs += (ubQS*ftu + dbQS*ftd + sbQS*fts)*dg*sig10

    QScs += (uQS*du + dQS*dd + sQS*ds)*ftg*sig11

    QScs += (ubQS*dub + dbQS*ddb +sbQS*dsb)*ftg*sig11

    QScs += (uQS + dQS +sQS) *ftg *dg*sig12

    QScs += (ubQS+dbQS+sbQS)*ftg*dg*sig12

    QScs= 2. * pT * M * numfac * QScs #Note there is a 1/u in the hard factors

    return ffcs + QScs
    #return ffcs
    #return  QScs
    
def get_dsig(x, z, xF, pT, rs, tar, had):

    var=get_var(x, z, xF, pT, rs)
    s,t,u=var[2],var[3],var[4]
    Q2,xp=var[5],var[6]

  # Get arrays of the nonperturbative functions
    f=np.zeros(11)
    ft=np.zeros(11)
    d=np.zeros(11)
    
    f = get_f(x, Q2)
    ft = get_ft(xp, Q2)
    d = get_d(z, Q2, 'pi+')
    
    return _get_dsig(x, z, xF, pT, rs, tar, had, f, ft, d)

def get_dsigST(x, z, xF, pT, rs, tar, had):

    var=get_var(x, z, xF, pT, rs)
    s,t,u=var[2],var[3],var[4]
    Q2,xp=var[5],var[6]

  # Get arrays of the nonperturbative functions
    f=np.zeros(11)
    ft=np.zeros(11)
    h=np.zeros(11)
    f1Tp=np.zeros(11)
    Hxxpz = np.zeros((13, 7))
    d=np.zeros(11)
    
    HTffa = get_HTffa(s, t, u)
    HTffb = get_HTffb(s, t, u)
    
    if had == 'jet': H1p = np.ones(11)
    else: H1p=get_H1p(z, Q2, 'pi+')
    H = get_H(z, Q2, 'pi+')

    f = get_f(x, Q2)
    ft = get_ft(xp, Q2)
    h = get_h(x, Q2)
    f1Tp = get_f1Tp(x, Q2)
    d = get_d(z, Q2, 'pi+')
    Hxxpz = get_Hxxpz(z, Q2, had, s, t, u, HTffa, HTffb, H1p, H)
    
    Hxxpz = np.einsum('i,j->ij', Hxxpz[0], Hxxpz[1]) + np.einsum('i,j->ij', Hxxpz[2], Hxxpz[3]) / z
    
    return _get_dsigST(x, z, xF, pT, rs, tar, had, f, ft, h, d, f1Tp, Hxxpz)

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

    if had=='jet':
        z = 1.
        xmin = -uu / (ss + tt)
        dsig = np.vectorize(lambda x: get_dsig(x, z, xF, pT, rs, tar, had))
        sig = fixed_quad(dsig, xmin, 1, n=nx)[0]
    else:
        if mode == 'gauss':
            dsigdzdx = np.vectorize(lambda x, z: get_dsig(x, z, xF, pT, rs, tar, had))
            dsigdz = np.vectorize(lambda z: fixed_quad(lambda x: dsigdzdx(x, z), xmin(z), 1., n=nx)[0])
            sig = fixed_quad(dsigdz, zmin, 1., n=nz)[0]
        elif mode == 'quad':
            epsrel=1e-03
            sig = dblquad(lambda x, z: get_dsig(x, z, xF, pT, rs, tar, had), zmin, 1., xmin, lambda x: 1.,epsrel=epsrel)[0]
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

    if had=='jet':
        z = 1.
        xmin = -uu / (ss + tt) 
        dsig=np.vectorize(lambda x: get_dsigST(x, z, xF, pT, rs, tar, had))
        sig = fixed_quad(dsig, xmin, 1, n=nx)[0]
    else:
        if mode == 'gauss':
            dsigdzdx = np.vectorize(lambda x, z: get_dsigST(x, z, xF, pT, rs, tar, had))
            dsigdz = np.vectorize(lambda z: fixed_quad(lambda x: dsigdzdx(x, z), xmin(z), 1., n=nx)[0])
            sig = fixed_quad(dsigdz, zmin, 1., n=nz)[0]
        elif mode == 'quad':
            epsrel=1e-03
            sig = dblquad(lambda x, z: get_dsigST(x, z, xF, pT, rs, tar, had), zmin, 1., xmin, lambda x: 1.,epsrel=epsrel)[0]

    #print had,sig
    return sig


if __name__ == '__main__':


    from qcdlib.ff0 import FF as FF0
    from qcdlib.ff1_test import FF as FF1
    from qcdlib.pdf0 import PDF as PDF0
    from qcdlib.pdf1_test import PDF as PDF1
    conf['aux']= AUX()
    conf['pdf']=PDF0()
    conf['collinspi']=FF1('pi')
    conf['collinsk']=FF1('k')
    conf['dcollinspi']=FF1('pi','deriv')
    conf['dcollinsk']=FF1('k','deriv')
    conf['Htildepi']=FF1('pi')
    conf['Htildek']=FF1('k')
    conf['transversity']=PDF1()
    conf['sivers']=PDF1()
    conf['dsivers']=PDF1('deriv')
    conf['ffpi']=FF0('pi')
    conf['ffk']=FF0('k')

    rs = 200.
    tar = 'p'
    #had = 'pi+'
    had='pi-'
    pT = 2.0
    xF = 0.2

    def test():
        for pT in [1.0, 2.0, 3.0, 4.0, 5.0]:
            for xF in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]:
                den = get_sig(xF, pT, rs, tar, had, mode='gauss', nx=10, nz=10)
                num = get_sigST(xF, pT, rs, tar, had,
                       mode='gauss', nx=10, nz=10)

                AN = num / den
                print(xF, pT, AN)

#  test()

# from timeit import Timer
# t = Timer("test()", "from __main__ import test")
# print 't elapsed ',t.timeit(number=1)

# def test2():
#  den = anthy.get_dsig(0.3,0.6,xF,pT,rs,tar,had)
#  num = anthy.get_dsigST(0.3,0.6,xF,pT,rs,tar,had)
#
#  print den,num
#
# from timeit import Timer
# t = Timer("test2()", "from __main__ import test2")
# print 't elapsed ',t.timeit(number=1)

# start = time.time()
# print anthy.get_dsig(0.3,0.6,xF,pT,rs,tar,had)
# print anthy.get_dsigST(0.3,0.6,xF,pT,rs,tar,had)
# end = time.time()
# print 'time=',(end-start)

    for k in range(2):
        start = time.time()
        test()
        end = time.time()
        print('time=', (end - start))

  # Integration of the numerator from xmin to 1 and from zmin to 1 (the values for xmin and zmin are above)

  # Integration of the denominator from xmin to 1 and from zmin to 1 (the values for xmin and zmin are above)
  #den = dblquad(lambda x,z: ANTHEORY().get_dsig(x,z,xF,pT,rs,tar,had),zmin,1.,xmin,lambda x: 1.)

  #AN = num[0]/den[0]
  # print(AN)
