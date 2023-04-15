#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# :Name:     mcjtplotsv2.py
# :Purpose:  Final plots
# :Author:   Ricardo Chavez -- Mar 19th, 2019 -- Cambridge, England
# :Modified:
#------------------------------------------------------------------------------
from astropy.table import Table
from astropy.io import fits
from astropy.io import ascii
from astropy.visualization import hist
from astropy.cosmology import FlatwCDM
from astropy.cosmology import Flatw0waCDM
from astropy.cosmology import w0waCDM
import scipy.optimize as op
import scipy.stats as st
import scipy as sp
import numpy as np
from astropy import constants as const
from numba import jit, double

from pymultinest.solve import solve
import pymultinest
import json

import corner
# from chainconsumer import ChainConsumer
from getdist import plots, MCSamples, densities
# from hubblediagram import hubblediagram
import datetime

# import emcee

# from gfmcv2 import gfmcv2

import time
import glob
# import pyfits

import matplotlib.pyplot as plt

c = const.c.to('km/s').value
k_b = const.k_B.to('J/K').value
m_p = const.m_p.to('kg').value

EBVG = 0.184*1.2
EBVGErr = 0.048

EBVC = 0.1212*1.2
EBVCErr = 0.037

LSL = 1.84#1.84
x0 = 1.79
x0Err = 0.08


def lnprior1(cube):
    cube[0] = cube[0]*0.5 + 0.5
    cube[1] = cube[1]
    cube[2] = cube[2]*2.0 - 2.0
    return cube


def lnlike1(cube):
    h0, Om, w0 = cube[0], cube[1], cube[2]
    theta = [h0, Om, w0]
    xsq = chsq1(theta)[0]
    return -0.5*xsq


def chsq1(theta):
    h0, Om, w0 = theta

    wb = 0.02225
    Or = 4.153e-5 * h0**(-2)
    Ok = 0.0
    w1 = 0.0
    Ob = wb * h0**(-2.0)
    if Ob > Om: return (9e9, 9e9)

    #SNIa
    x, z, xerr, zerr, cov = Mdata
    MB = 0.0
    cosmo = FlatwCDM(H0=100, Om0=Om, w0=w0)

    DL = (cosmo.luminosity_distance(z).value*100.0)/c
    Mum = 5.0*np.log10(DL)

    Mu = x - MB

    Rs = (Mu - Mum)
    Ws = 1.0/(xerr**2)

    xsqA = np.sum(Rs**2*Ws)
    xsqB = np.sum(Rs*Ws)
    xsqC = np.sum(Ws)

    xsqs = xsqA - xsqB**2/xsqC
    xsqs2 = xsqs - np.sum(np.log(Ws))

    #CMB
    cmb_data = np.array([[301.77, 0.09], [1.7482, 0.0048], [0.02226, 0.00016]])
    rev_C_cmb = np.array([[ 1.47529862e+02, -9.50243997e+02,  6.75330855e+03],
       [-9.49518029e+02,  8.87656028e+04,  1.66515286e+06],
       [ 6.78491359e+03,  1.66491606e+06,  7.46953427e+07]])

    vCMBth = [la(h0, Om, w0, w1, wb), Rz(h0, Om, w0, w1, wb), wb]
    Rc = vCMBth - cmb_data[:,0]
    xsqc = np.dot(np.dot(np.transpose(Rc), rev_C_cmb), Rc)

    #BAO
    zs = zst(h0, Om, wb)
    rd = rs(zs, h0, Om, w0, w1, wb)

    lm = np.zeros(10)
    ix0 = [0, 1, 2, 3]
    for i in ix0:
        zi = bao_data[i,0]
        rdf = bao_data[i,3]
        Dv = DV(zi, h0, Om, w0, w1)
        lm[i] = (Dv*rdf)/rd

    ix1 = [4, 5] #, 6, 7, 8]
    for i in ix1:
        zi = bao_data[i,0]
        Da = DA(zi, h0, Om, w0, w1)
        lm[i] = Da/rd

    ix2 = [9]
    zi = bao_data[ix2,0]
    Dm = DM(zi, h0, Om, w0, w1)
    lm[ix2] = Dm/rd

    ixf = np.concatenate((ix0, ix1))
    Rb = lm[ixf] - bao_data[ixf,1]

    xsqb = np.dot(np.dot(np.transpose(Rb), rev_C_bao[0:6, 0:6]), Rb)
    #--------------------------------------------------------------------------
    xsqT = xsqs + xsqc + xsqb
    xsqT2 = xsqs2 + xsqc + xsqb

    RT = np.concatenate((Rs, Rc, Rb))
    return (xsqT, RT)


def lnprior2(cube):
    cube[0] = cube[0]*0.5 + 0.5
    cube[1] = cube[1]
    cube[2] = cube[2]*2 - 2
    cube[3] = cube[3]*6.0 - 4.0
    return cube


def lnlike2(cube):
    h0, Om, w0, w1 = cube[0], cube[1], cube[2], cube[3]
    theta = h0, Om, w0, w1
    xsq = chsq2(theta)
    return -0.5*xsq


def chsq2(theta):
    h0, Om, w0, w1  = theta

    wb = 0.02225
    Or = 4.153e-5 * h0**(-2)
    Ok = 0.0
    Ob = wb * h0**(-2.0)
    if Ob > Om: return 9e9

    #HII
    x, y, z, xerr, yerr, zerr, Rxy = Mdata
    cosmo = Flatw0waCDM(H0=100, Om0=Om, w0=w0, wa=w1)

    ixG = np.where(z>10)
    ixH = np.where(z<10)

    Mum = z*0.0
    MumErr = z*0.0

    Mum[ixG] = z[ixG]
    MumErr[ixG] = zerr[ixG]

    Mum[ixH] = 5*np.log10((cosmo.luminosity_distance(z[ixH]).value*100)/c) + 25
    MumErr[ixH] = (5.0/np.log(10))*(zerr[ixH]/z[ixH])

    Mu = 2.5*(beta*x + alpha) - 2.5*y - 100.195
    MuErr = 2.5*np.sqrt((yerr)**2 + beta**2*(xerr)**2
            + betaErr**2*x**2
            + alphaErr**2
            # - 2.0*beta*Rxy*xerr*yerr
    )

    R = (Mu - Mum)
    W = 1.0/(MuErr**2 + MumErr**2)

    xsqA = np.sum(R**2*W)
    xsqB = np.sum(R*W)
    xsqC = np.sum(W)

    xsqh = xsqA - xsqB**2/xsqC
    xsqh2 = xsqh - np.sum(np.log(W))

    #CMB
    cmb_data = np.array([[301.77, 0.09], [1.7482, 0.0048], [0.02226, 0.00016]])
    rev_C_cmb = np.array([[ 1.47529862e+02, -9.50243997e+02,  6.75330855e+03],
       [-9.49518029e+02,  8.87656028e+04,  1.66515286e+06],
       [ 6.78491359e+03,  1.66491606e+06,  7.46953427e+07]])

    vCMBth = [la(h0, Om, w0, w1, wb), Rz(h0, Om, w0, w1, wb), wb]
    res = vCMBth - cmb_data[:,0]
    xsqc = np.dot(np.dot(np.transpose(res), rev_C_cmb), res)

    #BAO
    zs = zst(h0, Om, wb)
    rd = rs(zs, h0, Om, w0, w1, wb)

    lm = np.zeros(10)

    ix0 = [0, 1, 2, 3]
    for i in ix0:
        zi = bao_data[i,0]
        rdf = bao_data[i,3]
        Dv = DV(zi, h0, Om, w0, w1)
        lm[i] = (Dv*rdf)/rd

    ix1 = [4, 5] #, 6, 7, 8]
    for i in ix1:
        zi = bao_data[i,0]
        Da = DA(zi, h0, Om, w0, w1)
        lm[i] = Da/rd

    ix2 = [9]
    zi = bao_data[ix2,0]
    Dm = DM(zi, h0, Om, w0, w1)
    lm[ix2] = Dm/rd

    ixf = np.concatenate((ix0, ix1))
    R = lm[ixf] - bao_data[ixf,1]
    W = 1.0/(bao_data[ixf,2]**2)

    xsqb = np.dot(np.dot(R, rev_C_bao[0:6, 0:6]), np.transpose(R))
    #--------------------------------------------------------------------------
    xsqT = xsqh + xsqc + xsqb
    xsqT2 = xsqh2 + xsqc + xsqb

    if fr == 2:
        return xsqT2
    else:
        return xsqT


def lnprior21(cube):
    cube[0] = cube[0]*2 - 2
    cube[1] = cube[1]*6.0 - 4.0
    return cube


def lnlike21(cube):
    w0, w1 = cube[0], cube[1]
    theta = w0, w1
    xsq = chsq21(theta)
    return -0.5*xsq


def chsq21(theta):
    w0, w1  = theta

    h0 = 0.681
    Om = 0.309
    wb = 0.02225
    Or = 4.153e-5 * h0**(-2)
    Ok = 0.0
    Ob = wb * h0**(-2.0)

    #HII
    x, y, z, xerr, yerr, zerr, Rxy = Mdata
    cosmo = Flatw0waCDM(H0=100, Om0=Om, w0=w0, wa=w1)

    ixG = np.where(z>10)
    ixH = np.where(z<10)

    Mum = z*0.0
    MumErr = z*0.0

    Mum[ixG] = z[ixG]
    MumErr[ixG] = zerr[ixG]

    Mum[ixH] = 5*np.log10((cosmo.luminosity_distance(z[ixH]).value*100)/c) + 25
    MumErr[ixH] = (5.0/np.log(10))*(zerr[ixH]/z[ixH])

    Mu = 2.5*(beta*x + alpha) - 2.5*y - 100.195
    MuErr = 2.5*np.sqrt((yerr)**2 + beta**2*(xerr)**2
            + betaErr**2*x**2
            + alphaErr**2
            # - 2.0*beta*Rxy*xerr*yerr
    )

    R = (Mu - Mum)
    W = 1.0/(MuErr**2 + MumErr**2)

    xsqA = np.sum(R**2*W)
    xsqB = np.sum(R*W)
    xsqC = np.sum(W)

    xsqh = xsqA - xsqB**2/xsqC
    xsqh2 = xsqh - np.sum(np.log(W))

    #CMB
    cmb_data = np.array([[301.77, 0.09], [1.7482, 0.0048], [0.02226, 0.00016]])
    rev_C_cmb = np.array([[ 1.47529862e+02, -9.50243997e+02,  6.75330855e+03],
       [-9.49518029e+02,  8.87656028e+04,  1.66515286e+06],
       [ 6.78491359e+03,  1.66491606e+06,  7.46953427e+07]])

    vCMBth = [la(h0, Om, w0, w1, wb), Rz(h0, Om, w0, w1, wb), wb]
    res = vCMBth - cmb_data[:,0]
    xsqc = np.dot(np.dot(np.transpose(res), rev_C_cmb), res)

    #BAO
    zs = zst(h0, Om, wb)
    rd = rs(zs, h0, Om, w0, w1, wb)

    lm = np.zeros(10)

    ix0 = [0, 1, 2, 3]
    for i in ix0:
        zi = bao_data[i,0]
        rdf = bao_data[i,3]
        Dv = DV(zi, h0, Om, w0, w1)
        lm[i] = (Dv*rdf)/rd

    ix1 = [4, 5] #, 6, 7, 8]
    for i in ix1:
        zi = bao_data[i,0]
        Da = DA(zi, h0, Om, w0, w1)
        lm[i] = Da/rd

    ix2 = [9]
    zi = bao_data[ix2,0]
    Dm = DM(zi, h0, Om, w0, w1)
    lm[ix2] = Dm/rd

    ixf = np.concatenate((ix0, ix1))
    R = lm[ixf] - bao_data[ixf,1]
    W = 1.0/(bao_data[ixf,2]**2)

    xsqb = np.dot(np.dot(R, rev_C_bao[0:6, 0:6]), np.transpose(R))
    #--------------------------------------------------------------------------
    xsqT = xsqh + xsqc + xsqb
    xsqT2 = xsqh2 + xsqc + xsqb

    if fr == 2:
        return xsqT2
    else:
        return xsqT


def mcjtplotsv2(ve, dpath, cpath, clc = 0 , opt = 0, spl = 0, zps = 0, prs = 1
        , drd = 0, obs = 0, nwk = 1000, sps = 10000, tds = 6, vbs = 0, fr0 = 1
        , a = 0, aErr = 0, b = 0, bErr = 0):

    print('+++++++++++++++++++++++++++++++++++++++++++')
    print('mcjtplotsv2: ' + str(datetime.datetime.now()))
    print('+++++++++++++++++++++++++++++++++++++++++++')

    #============= Parameters ======================================
    fr = fr0
    fg0 = clc
    fg1 = opt
    fsim = 1

    #============= Main Body =======================================
    print(datetime.datetime.now())
    start = datetime.datetime.now()
    # Cases
    if fg1 == 1:
        # SNIa+CMB+BAO
        print('SNIa+CMB+BAO')
        fend = 'mcscbv6_'+str(ve)+'_'+ str(0)+'_'+ str(fg1) + '_stat'
        prefix = cpath + fend
        print(fend)

        parameters = [r"h_0", r"\Omega_m", r"w_0"]
        n_params = len(parameters)

        result = solve(LogLikelihood=lnlike1, Prior=lnprior1,
            n_dims=n_params, outputfiles_basename=prefix, verbose=vbs,
            n_live_points = sps, resume = 1, init_MPI = False,
            sampling_efficiency = 0.8)

        gdsJoint = MCSamples(samples=result['samples'], names = parameters
            , labels = parameters, name_tag = 'SNIa+CMB+BAO'
            , ranges={"w_0":[-2.0, 0.0]}
                    )

        # HIIG+CMB+BAO
        print('HIIG+CMB+BAO')
        fend ='mchcbv8_'+str(ve)+'_'+str(spl)+str(zps)+str(fg1)+str(drd)+str(obs)+str(fr0)
        prefix = cpath + fend
        print(fend)
        n_params = len(parameters)

        result = solve(LogLikelihood=lnlike1, Prior=lnprior1,
            n_dims=n_params, outputfiles_basename=prefix, verbose=vbs,
            n_live_points = sps, resume = 1, init_MPI = False,
            sampling_efficiency = 0.8)

        gdsT = MCSamples(samples=result['samples'], names = parameters
            , labels = parameters, name_tag = 'HII+CMB+BAO'
            , ranges={"w_0":[-2.0, 0.0]}
            )

        #Joint
        print('Joint')
        parameters = ["\Omega_m", "w_0"]
        path = dpath+'results/JTOmW0'+ str(fg1) +'.pdf'
        g = plots.getSubplotPlotter()
        g.settings.axes_fontsize = 8
        g.settings.num_plot_contours = 2
        g.settings.legend_fontsize = 10
        g.triangle_plot([gdsT, gdsJoint], parameters,
            shaded=False)
        g.export(path)
    elif fg1 == 2:
        # HIIG+CMB+BAO
        print('HIIG+CMB+BAO')
        fend ='mchcbv8_'+str(ve)+'_'+str(spl)+str(zps)+str(fg1)+str(drd)+str(obs)+str(fr0)
        prefix = cpath + fend
        print(fend)
        parameters = [r"h_0", r"\Omega_m", r"w_0", r"w_a"]
        n_params = len(parameters)

        result = solve(LogLikelihood=lnlike21, Prior=lnprior21,
            n_dims=n_params, outputfiles_basename=prefix, verbose=vbs,
            n_live_points = sps, resume = 1, init_MPI = False,
            sampling_efficiency = 0.8)

        gdsT = MCSamples(samples=result['samples'], names = parameters
            , labels = parameters, name_tag = 'HII+CMB+BAO'
            , ranges={"w_a":[-1.5, 1.5]
                    , "w_0":[-1.2, -0.74]}
            )

        # BAO+CMB+SNIa
        print('SNIa+CMB+BAO')
        fend ='mcscbv6_'+str(ve)+'_'+ str(0) + '_' + str(fg1) + '_stat'
        prefix = cpath + fend
        print(fend)
        parameters = [r"h_0", r"\Omega_m", r"w_0", r"w_a"]
        n_params = len(parameters)

        result = solve(LogLikelihood=lnlike2, Prior=lnprior2,
            n_dims=n_params, outputfiles_basename=prefix, verbose=vbs,
            n_live_points=sps, resume = 1, init_MPI = False,
            sampling_efficiency = 0.8)

        gdsJoint = MCSamples(samples=result['samples'], names = parameters
            , labels = parameters, name_tag = 'SNIa+CMB+BAO'
            , ranges={"w_a":[-1.5, 1.5]
                    , "w_0":[-1.2, -0.74]}
                    )
                    # , ranges={"w_a":[-3.0, 2.0]
                    # , "w_0":[-1.2, -0.74]}
        #Joint
        print('Joint')
        parameters = ["w_a", "w_0"]
        path = dpath+'results/JTw0wa'+ str(fg1) +'.pdf'
        g = plots.getSubplotPlotter()
        g.settings.axes_fontsize = 8
        g.settings.num_plot_contours = 2
        g.settings.legend_fontsize = 10
        g.triangle_plot([gdsT, gdsJoint], parameters,
            shaded=False)
        g.export(path)
    elif fg1 == 21:
        # HIIG+CMB+BAO
        print('HIIG+CMB+BAO')
        fend ='mchcbv8_'+str(ve)+'_'+str(spl)+str(zps)+str(2)+str(drd)+str(obs)+str(fr0)
        prefix = cpath + fend
        print(fend)

        parameters = [r"w_0", r"w_a"]
        n_params = len(parameters)

        result = solve(LogLikelihood=lnlike21, Prior=lnprior21,
            n_dims=n_params, outputfiles_basename=prefix, verbose=vbs,
            n_live_points = sps, resume = 1, init_MPI = False,
            sampling_efficiency = 0.8)

        gdsT = MCSamples(samples=result['samples'], names = parameters
            , labels = parameters, name_tag = 'HII+CMB+BAO'
            , ranges={"w_a":[-2.0, 1.0]
                    , "w_0":[-1.3, -0.5]}
            )

        # BAO+CMB+SNIa
        print('SNIa+CMB+BAO')
        fend = 'mcscbv6_'+str(ve)+'_'+ str(0)+'_'+ str(2) + '_stat'
        prefix = cpath + fend
        print(fend)

        parameters = [r"w_0", r"w_a"]
        n_params = len(parameters)

        result = solve(LogLikelihood=lnlike21, Prior=lnprior21,
            n_dims=n_params, outputfiles_basename=prefix, verbose=vbs,
            n_live_points = sps, resume = 1, init_MPI = False,
            sampling_efficiency = 0.8)

        gdsJoint = MCSamples(samples=result['samples'], names = parameters
            , labels = parameters, name_tag = 'SNIa+CMB+BAO'
            , ranges={"w_a":[-2.0, 1.0]
                    , "w_0":[-1.3, -0.5]}
                    )

        #Joint
        print('Joint')
        parameters = ["w_a", "w_0"]
        path = dpath+'results/JTw0wa'+ str(fg1) +'.pdf'
        g = plots.getSubplotPlotter()
        g.settings.axes_fontsize = 8
        g.settings.num_plot_contours = 2
        g.triangle_plot([gdsT, gdsJoint], parameters,
            shaded=False)
        g.export(path)

    print(datetime.datetime.now())
    return
