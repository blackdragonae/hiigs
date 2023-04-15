#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# :Name:     mch2gcmbbaov8.py
# :Purpose:  MCMC Joint HII/CMB/BAO
# :Author:   Ricardo Chavez -- Apr 14th, 2020 -- Cambridge, England
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

LSL = 1.83#1.84
x0 = 1.79
x0Err = 0.08


def zst(h0, Om, wb):
    g1 = (0.0783*(wb)**(-0.238))/(1.0 + 39.5*(wb)**(0.763))
    g2 = 0.560/(1.0 + 21.1*(wb)**(1.81))
    Zst = 1048.0*(1.0 + 0.00124*(wb)**(-0.738))*(1.0 + g1*(Om*h0**2)**(g2))
    return Zst


def Rz(h0, Om, w0, w1, wb):
    Ob = wb*(h0)**(-2.0)
    cosmo = Flatw0waCDM(H0=100.0, Om0=Om, w0=w0, wa=w1, Ob0=Ob)

    Zst = zst(h0, Om, wb)
    I = (cosmo.comoving_distance(Zst).value*100)/c
    Rz = np.sqrt(Om)*I
    return Rz


def intrs(x, h0, Om, w0, w1, wb):
    Ok = 0.0
    Or = 4.153e-5 * h0**(-2.0)
    Ow = 1.0 - Om - Or - Ok
    Ob = wb * h0**(-2.0)
    Og = 2.469e-5 * h0**(-2.0)
    cs = 1.0/np.sqrt(3.0 + 3.0*((3.0*Ob)/(4.0*Og))*x)
    f = cs/(x**2*np.sqrt(Or*x**(-4.0) + Om*x**(-3.0) + Ok*x**(-2.0) + Ow*x**(-3.0*(1.0 + w0 + w1 ))*np.exp(-3.0 * w1 * (1.0 - x))))
    return f


def rs(z, h0, Om, w0, w1, wb):
    Ho = h0*100.0
    Hd = c/Ho
    a = 1.0/(1.0 + z)
    I  = sp.integrate.quad(intrs, 0.0, a, args=(h0, Om, w0, w1, wb))
    frs = Hd*I[0]
    return frs


def la(h0, Om, w0, w1, wb):
    Ob = wb*(h0)**(-2.0)
    cosmo = Flatw0waCDM(H0=h0*100.0, Om0=Om, w0=w0, wa=w1, Ob0=Ob)

    Zst = zst(h0, Om, wb)
    Da = cosmo.angular_diameter_distance(Zst).value
    Rs = rs(Zst, h0, Om, w0, w1, wb)
    La = (1.0 + Zst)*((np.pi*Da)/Rs)
    return La


def DM(z, h0, Om, w0, w1):
    cosmo = Flatw0waCDM(H0=h0*100.0, Om0=Om, w0=w0, wa=w1)
    dM = (cosmo.luminosity_distance(z).value)/(1.0 + z)
    return dM


def DA(z, h0, Om, w0, w1):
    cosmo = Flatw0waCDM(H0=h0*100.0, Om0=Om, w0=w0, wa=w1)
    dA =  (cosmo.luminosity_distance(z).value)/(1.0 + z)**2
    return dA


def DV(z, h0, Om, w0, w1):
    cosmo = Flatw0waCDM(H0=h0*100.0, Om0=Om, w0=w0, wa=w1)
    Hz = cosmo.efunc(z)*h0*100.0
    dV = ((c*DA(z, h0, Om, w0, w1)**2*z*(1.0 + z)**2)/Hz)**(1./3.)
    return dV


def ploter(dpath, fend, result, parameters, parametersc, tag, ctag, nobs):
    # x, y, z, xerr, yerr, zerr, Rxy, Muo, MuoErr, vTg = Mdata
    fchsq = 'chsq' + str(ctag)
    n_params = len(parameters)
    print(fchsq, nobs)

    # Plot getdist
    path = dpath+'results/GDtr'+fend+'.pdf'
    GDsamples = MCSamples(samples=result['samples'], names=parameters,
                          labels=parameters, name_tag=tag, 
                          ranges={r"\Omega_m":[0.0, None]})

    g = plots.getSubplotPlotter()
    g.settings.num_plot_contours = 2
    g.triangle_plot(GDsamples, filled=True, title_limit=1)
    g.export(path)

    t = GDsamples.getTable(limit=1).tableTex()
    theta = GDsamples.getMeans()

    chsq = globals()[fchsq](theta)[0]
    res = globals()[fchsq](theta)[1]

    # Cov Matix and FoM
    # pars=[r"\Omega_m", r"w_0"]
    FoM = 1./np.sqrt(np.linalg.det(GDsamples.cov()))
    print('FoM = ', FoM)

    # Plot Corner
    path = dpath+'results/tr'+fend+'.pdf'
    fig = corner.corner(result['samples'],
                        labels=parametersc,
                        plot_contours='true',
                        levels=1.0 - np.exp(
                            -0.5 * np.arange(1.0, 3.1, 1.0) ** 2),
                        smooth=1.0,
                        bins=100,
                        color='black',
                        show_titles=1)
    plt.savefig(path)

    # Print parameter values
    print('parameter values:')
    for name, col in zip(parameters, result['samples'].transpose()):
        print('%15s : %.3f +- %.3f' % (name, col.mean(), col.std()))

    print(t)

    dof = nobs - n_params

    print('Chsq min:')
    print(theta)
    print(chsq)
    print(nobs)
    print(chsq/dof)

    # Test
    DMu = np.mean(res)
    print(len(res))
    nres = (res - DMu)/np.std(res)
    if nobs > 3:
        path = dpath+'results/Hnres'+fend+'.pdf'
        fig = plt.subplots(1)
        hist(nres, bins='knuth', histtype='stepfilled', alpha=0.2)
        plt.savefig(path)

    ksD, ksP = st.kstest(nres, 'norm')
    print(ksD, ksP)
    print(np.std(res))

    # Print to log file
    f = open(dpath+'results/GDlog'+fend+'.txt', 'w')
    f.write(t)
    f.write('--------------------------------------\n')
    f.write(str(chsq)+'\n')
    f.write(str(chsq/dof)+'\n')
    f.write(str(ksD)+','+str(ksP)+'\n')
    f.write(str(np.std(res))+'\n')
    f.write(str(nobs)+'\n')
    f.write(str(FoM)+'\n')
    f.write(str(start)+'\n')
    f.write(str(datetime.datetime.now()))
    f.close
    return GDsamples


def lnpriorh1(cube):
    cube[0] = cube[0]
    cube[1] = cube[1]*2 - 2
    return cube


def lnlikeh1(cube):
    Om, w0 = cube[0], cube[1]
    theta = Om, w0
    xsq = chsqh1(theta)[0]
    return -0.5*xsq


def chsqh1(theta):
    Om, w0 = theta
    x, y, z, xerr, yerr, zerr, Rxy, Muo, MuoErr, vTg = Mdata

    cosmo = FlatwCDM(H0=100, Om0=Om, w0=w0)
    #------------------------------------------------------------------------------
    ixG = np.where(z>10)
    ixH = np.where(z<10)

    Mum = z*0.0
    MumErr = z*0.0

    Mum[ixG] = z[ixG]
    MumErr[ixG] = zerr[ixG]

    Mum[ixH] = 5*np.log10((cosmo.luminosity_distance(z[ixH]).value*100)/c) + 25
    MumErr[ixH] = (5.0/np.log(10))*(zerr[ixH]/z[ixH])

    Mu = Muo
    MuErr = MuoErr

    R = (Mu - Mum)
    W = 1.0/(MuErr**2 + MumErr**2)

    xsqA = np.sum(R**2*W)
    xsqB = np.sum(R*W)
    xsqC = np.sum(W)

    xsq = xsqA - xsqB**2/xsqC
    xsq2 = xsq - np.sum(np.log(W))

    if fr == 2:
        return (xsq2, R, Mum, MuErr)
    else:
        return (xsq, R, Mum, MuErr)


def caseh1(cpath, dpath, fend, vbs, sps, prs, ctag):
    x, y, z, xerr, yerr, zerr, Rxy, Muo, MuoErr, vTg = Mdata
    parameters = [r"\Omega_m", r"w_0"]
    parametersc = [r"$\Omega_m$", r"$w_0$"]
    n_params = len(parameters)

    prefix = cpath + fend
    print(prefix)

    result = solve(LogLikelihood=lnlikeh1, Prior=lnpriorh1,
                   n_dims=n_params, outputfiles_basename=prefix, verbose=vbs,
                   n_live_points=sps, resume=prs, init_MPI=False,
                   sampling_efficiency=0.8)

    tag = 'HIIG (181)'
    GDsamples = ploter(dpath, fend, result, parameters, parametersc, tag, ctag, len(x))
    return GDsamples


def lnpriorc1(cube):
    cube[0] = cube[0]*0.5 + 0.5
    cube[1] = cube[1]
    cube[2] = cube[2]*2.0 - 2.0
    return cube


def lnlikec1(cube):
    h0, Om, w0 = cube[0], cube[1], cube[2]
    theta = [h0, Om, w0]
    xsq = chsqc1(theta)[0]
    return -0.5*xsq


def chsqc1(theta):
    h0, Om, w0 = theta

    wb = 0.02225
    Or = 4.153e-5 * h0**(-2)
    Ok = 0.0
    w1 = 0.0
    Ob = wb * h0**(-2.0)

    if Ob > Om: return (9e9, 9e9)
    #---------------------------------------------------------------------------
    cmb_data = np.array([[301.77, 0.09], [1.7482, 0.0048], [0.02226, 0.00016]])
    rev_C_cmb = np.array([[ 1.47529862e+02, -9.50243997e+02,  6.75330855e+03],
       [-9.49518029e+02,  8.87656028e+04,  1.66515286e+06],
       [ 6.78491359e+03,  1.66491606e+06,  7.46953427e+07]])
    #---------------------------------------------------------------------------
    #Modeling
    vCMBth = [la(h0, Om, w0, w1, wb), Rz(h0, Om, w0, w1, wb), wb]
    R = vCMBth - cmb_data[:,0]

    #ChiSq
    xsq = np.dot(np.dot(np.transpose(R), rev_C_cmb), R)
    return (xsq, R)


def casec1(cpath, dpath, fend, vbs, sps, prs, ctag):
    parameters = [r"h_0", r"\Omega_m", r"w_0"]
    parametersc = [r"$h_0$", r"$\Omega_m$", r"$w_0$"]
    n_params = len(parameters)

    prefix = cpath + fend
    print(prefix)

    result = solve(LogLikelihood=lnlikec1, Prior=lnpriorc1,
                   n_dims=n_params, outputfiles_basename=prefix, verbose=vbs,
                   n_live_points=sps, resume=prs, init_MPI=False,
                   sampling_efficiency=0.8)

    tag = 'CMB (3)'
    GDsamples = ploter(dpath, fend, result, parameters, parametersc, tag, ctag, 3)
    return GDsamples


def lnpriorb1(cube):
    cube[0] = cube[0]
    cube[1] = cube[1]*2 - 2
    return cube


def lnlikeb1(cube):
    Om, w0 = cube[0], cube[1]
    theta = [Om, w0]
    xsq = chsqb1(theta)[0]
    return -0.5*xsq


def chsqb1(theta):
    Om, w0 = theta
    
    h0 = 0.6774
    w1 = 0.0
    wb = 0.02225
    zs = zst(h0, Om, wb)
    rd = rs(zs, h0, Om, w0, w1, wb)
    #--------------------------------------------------------------------------
    lm = np.zeros(10)
    # print lm
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

    xsq = np.dot(np.dot(R, rev_C_bao[0:6, 0:6]), np.transpose(R))
    return (xsq, R)


def caseb1(cpath, dpath, fend, vbs, sps, prs, ctag):
    parameters = [r"\Omega_m", r"w_0"]
    parametersc = [r"$\Omega_m$", r"$w_0$"]
    n_params = len(parameters)

    prefix = cpath + fend
    print(prefix)

    result = solve(LogLikelihood=lnlikeb1, Prior=lnpriorb1,
                   n_dims=n_params, outputfiles_basename=prefix, verbose=vbs,
                   n_live_points=sps, resume=prs, init_MPI=False,
                   sampling_efficiency=0.8)

    tag = 'BAO (6)'
    GDsamples = ploter(dpath, fend, result, parameters, parametersc, tag, ctag, 6)
    return GDsamples


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

    #HII
    x, y, z, xerr, yerr, zerr, Rxy, Muo, MuoErr, vTg = Mdata
    cosmo = FlatwCDM(H0=100, Om0=Om, w0=w0)

    ixG = np.where(z>10)
    ixH = np.where(z<10)

    Mum = z*0.0
    MumErr = z*0.0

    Mum[ixG] = z[ixG]
    MumErr[ixG] = zerr[ixG]

    Mum[ixH] = 5*np.log10((cosmo.luminosity_distance(z[ixH]).value*100)/c) + 25
    MumErr[ixH] = (5.0/np.log(10))*(zerr[ixH]/z[ixH])

    Mu = Muo
    MuErr = MuoErr

    Rh = (Mu - Mum)
    Wh = 1.0/(MuErr**2 + MumErr**2)

    xsqA = np.sum(Rh**2*Wh)
    xsqB = np.sum(Rh*Wh)
    xsqC = np.sum(Wh)

    xsqh = xsqA - xsqB**2/xsqC
    xsqh2 = xsqh - np.sum(np.log(Wh))

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
    xsqT = xsqh + xsqc + xsqb
    xsqT2 = xsqh2 + xsqc + xsqb

    RT = np.concatenate((Rh, Rc, Rb))

    if fr == 2:
        return (xsqT2, RT)
    else:
        return (xsqT, RT)


def case1(cpath, dpath, fend, vbs, sps, prs, ctag, tag):
    parameters = [r"h_0", r"\Omega_m", r"w_0"]
    parametersc = [r"$h_0$", r"$\Omega_m$", r"$w_0$"]
    n_params = len(parameters)

    prefix = cpath + fend
    print(prefix)

    result = solve(LogLikelihood=lnlike1, Prior=lnprior1,
                   n_dims=n_params, outputfiles_basename=prefix, verbose=vbs,
                   n_live_points=sps, resume=prs, init_MPI=False,
                   sampling_efficiency=0.8)

    GDsamples = ploter(dpath, fend, result, parameters, parametersc, tag, ctag, 198)
    return GDsamples


def lnprior2(cube):
    cube[0] = cube[0]*0.5 + 0.5
    cube[1] = cube[1]
    cube[2] = cube[2]*2 - 2
    cube[3] = cube[3]*6.0 - 4.0
    return cube


def lnlike2(cube):
    h0, Om, w0, w1 = cube[0], cube[1], cube[2], cube[3]
    theta = h0, Om, w0, w1
    xsq = chsq2(theta)[0]
    return -0.5*xsq


def chsq2(theta):
    h0, Om, w0, w1  = theta

    wb = 0.02225
    Or = 4.153e-5 * h0**(-2)
    Ok = 0.0
    Ob = wb * h0**(-2.0)
    if Ob > Om: return (9e9, 9e9)

    #HII
    x, y, z, xerr, yerr, zerr, Rxy, Muo, MuoErr, vTg = Mdata
    cosmo = Flatw0waCDM(H0=100, Om0=Om, w0=w0, wa=w1)

    ixG = np.where(z>10)
    ixH = np.where(z<10)

    Mum = z*0.0
    MumErr = z*0.0

    Mum[ixG] = z[ixG]
    MumErr[ixG] = zerr[ixG]

    Mum[ixH] = 5*np.log10((cosmo.luminosity_distance(z[ixH]).value*100)/c) + 25
    MumErr[ixH] = (5.0/np.log(10))*(zerr[ixH]/z[ixH])

    Mu = Muo
    MuErr = MuoErr

    Rh = (Mu - Mum)
    Wh = 1.0/(MuErr**2 + MumErr**2)

    xsqA = np.sum(Rh**2*Wh)
    xsqB = np.sum(Rh*Wh)
    xsqC = np.sum(Wh)

    xsqh = xsqA - xsqB**2/xsqC
    xsqh2 = xsqh - np.sum(np.log(Wh))

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
    zi = bao_data[ix2, 0]
    Dm = DM(zi, h0, Om, w0, w1)
    lm[ix2] = Dm/rd

    ixf = np.concatenate((ix0, ix1))
    Rb = lm[ixf] - bao_data[ixf,1]

    xsqb = np.dot(np.dot(np.transpose(Rb), rev_C_bao[0:6, 0:6]), Rb)
    #--------------------------------------------------------------------------
    xsqT = xsqh + xsqc + xsqb
    xsqT2 = xsqh2 + xsqc + xsqb

    RT = np.concatenate((Rh, Rc, Rb))

    if fr == 2:
        return (xsqT2, RT)
    else:
        return (xsqT, RT)


def case2(cpath, dpath, fend, vbs, sps, prs, ctag, tag):
    parameters = [r"h_0", r"\Omega_m", r"w_0", r"w_a"]
    parametersc = [r"$h_0$", r"$\Omega_m$", r"$w_0$", r"$w_a$"]
    n_params = len(parameters)

    prefix = cpath + fend
    print(prefix)

    result = solve(LogLikelihood=lnlike2, Prior=lnprior2,
                   n_dims=n_params, outputfiles_basename=prefix, verbose=vbs,
                   n_live_points=sps, resume=prs, init_MPI=False,
                   sampling_efficiency=0.8)

    GDsamples = ploter(dpath, fend, result, parameters, parametersc, tag, ctag, 190)
    return GDsamples


def union2020(ve, dpath, drd, obs):
    Tpath = dpath+'indat/Union2020v16.dat'
    data = Table.read(Tpath, format='ascii', comment='#')

    vix = data['col1']

    vx = data['col2']
    vy = data['col3']
    vz = data['col4']

    vxErr = data['col5']
    vyErr = data['col6']
    vzErr = data['col7']

    vsp = data['col8']

    rXY = np.corrcoef(vx, vy)
    vRxy = vx*0.0 + rXY[0,1]

    ix = np.where((vx - vxErr) <= LSL)

    return (vix[ix], vx[ix], vy[ix], vz[ix], vxErr[ix], vyErr[ix], vzErr[ix],
            vRxy[ix], vsp[ix])


def mch2gcmbbaov8(ve, dpath, cpath, clc = 0 , opt = 0, spl = 0, zps = 0, prs = 1
        , drd = 0, obs = 0, nwk = 1000, sps = 10000, tds = 6, vbs = 0, fr0 = 1
        , a = 0, aErr = 0, b = 0, bErr = 0):

    print('+++++++++++++++++++++++++++++++++++++++++++')
    print('mch2gcmbbaov8: ' + str(datetime.datetime.now()))
    print('+++++++++++++++++++++++++++++++++++++++++++')

    #============= Parameters ======================================
    #Fixed beta
    global Mdata
    global bao_data
    global rev_C_bao

    global alpha
    global alphaErr
    global beta
    global betaErr

    global fr
    global x0
    global x0Err
    global start
    global zs

    zs = zps

    alpha = a
    alphaErr = aErr
    beta = b
    betaErr = bErr

    print(alpha, beta)

    x0 = x0
    x0Err = x0Err

    fr = fr0
    fg0 = clc
    fg1 = opt
    fsim = 1

    #============= Main Body =======================================
    if spl == 2:
        # Union2020v4 data
        vTg, vX, vY, vZ, vXErr, vYErr, vZErr, vRxy, vsp = union2020(ve, dpath, drd, obs)
        if (zps == 1 or zps == 2 or zps == 3):
            ix = np.where(vZ < 10.)
            vTg = vTg[ix]
            vX = vX[ix]
            vY = vY[ix]
            vZ = vZ[ix]
            vXErr = vXErr[ix]
            vYErr = vYErr[ix]
            vZErr = vZErr[ix]
            vRxy = vRxy[ix]
    else:
        print('spl option error, must be an integer in the range [0:2]')
        return

    Muo = 2.5*(beta*vX + alpha) - 2.5*vY - 100.195
    MuoErr = 2.5*np.sqrt((vYErr)**2 + beta**2*(vXErr)**2
                 + betaErr**2*vX**2
                 + alphaErr**2
                 # - 2.0*beta*vRxy*vXErr*vYErr
                 # + 0.22**2
                 )

    Mdata = [vX, vY, vZ, vXErr, vYErr, vZErr, vRxy, Muo, MuoErr, vTg]

    bao_data = np.array([[0.122, 539.0, 17.0, 147.5], [0.44, 1716.40, 83.00, 148.6], [0.60, 2220.80, 101.00, 148.6], [0.73, 2516.00, 86.00, 148.6], [0.32, 6.67, 0.13, 148.11], [0.57, 9.52, 0.19, 148.11], [1.19, 12.6621, 0.9876, 147.78], [1.50, 12.4349, 1.0429, 147.78], [1.83, 13.1305, 1.0465, 147.78], [2.40, 36.0, 1.2, -1.0]])

    rev_C_bao = np.array([[ 3.46020761e-03,  0.00000000e+00,  0.00000000e+00,
         0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
         0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
         0.00000000e+00],
       [ 0.00000000e+00,  1.70712004e-04, -7.18471550e-05,
        -1.11633210e-04,  0.00000000e+00,  0.00000000e+00,
         0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
         0.00000000e+00],
       [ 0.00000000e+00, -7.18471550e-05,  1.65283175e-04,
         4.69828510e-05,  0.00000000e+00,  0.00000000e+00,
         0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
         0.00000000e+00],
       [ 0.00000000e+00, -1.11633210e-04,  4.69828510e-05,
         2.17800000e-04,  0.00000000e+00,  0.00000000e+00,
         0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
         0.00000000e+00],
       [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
         0.00000000e+00,  5.91715976e+01,  0.00000000e+00,
         0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
         0.00000000e+00],
       [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
         0.00000000e+00,  0.00000000e+00,  2.77008310e+01,
         0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
         0.00000000e+00],
       [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
         0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
         1.23421173e+00, -5.15450600e-01,  2.75529061e-01,
         0.00000000e+00],
       [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
         0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        -5.15450600e-01,  1.29862650e+00, -5.35046490e-01,
         0.00000000e+00],
       [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
         0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
         2.75529061e-01, -5.35046490e-01,  1.13742521e+00,
         0.00000000e+00],
       [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
         0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
         0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
         6.94444444e-01]])

    print(datetime.datetime.now())
    start = datetime.datetime.now()
    # Cases
    if fg1 == 1:
        # HIIG
        print('HIIG')
        fend = 'mchv69_152_1151212'
        ctag = 'h' + str(fg1)
        print(fend, ctag)
        Hsamples = caseh1(cpath, dpath, fend, vbs, sps, 1, ctag)

        #CMB
        print('CMB')
        fend = 'mccv10_149_'+str(fg1)
        ctag = 'c' + str(fg1)
        print(fend, ctag)
        Csamples = casec1(cpath, dpath, fend, vbs, sps, 1, ctag)

        # BAO
        print('BAO')
        fend = 'mcbv6_149_'+str(fg1)
        ctag = 'b' + str(fg1)
        print(fend, ctag)
        Bsamples = caseb1(cpath, dpath, fend, vbs, sps, 1, ctag)

        # HIIG+CMB+BAO
        print('HIIG+CMB+BAO')
        fend ='mchcbv8_'+str(ve)+'_'+str(spl)+str(zps)+str(fg1)+str(drd)+str(obs)+str(fr0)
        ctag = str(fg1)
        print(fend, ctag)
        tag = 'HIIG+CMB+BAO (190)'
        Jsamples = case1(cpath, dpath, fend, vbs, sps, prs, ctag, tag)

        # Joint Plot 
        tag0 = 'HIIG+CMB+BAO (190)'
        GD0 = Jsamples
        t0 = GD0.getTable(limit=1).tableTex()
        FoM0 = 1./np.sqrt(np.linalg.det(GD0.cov()))

        tag1 = 'HIIG (181)'
        GD1 = Hsamples
        t1 = GD1.getTable(limit=1).tableTex()
        FoM1 = 1./np.sqrt(np.linalg.det(GD1.cov()))

        tag2 = 'CMB (3)'
        GD2 = Csamples
        t2 = GD2.getTable(limit=1).tableTex()
        FoM2 = 1./np.sqrt(np.linalg.det(GD2.cov()))

        tag3 = 'BAO (6)'
        GD3 = Bsamples
        t3 = GD3.getTable(limit=1).tableTex()
        FoM3 = 1./np.sqrt(np.linalg.det(GD3.cov()))

        print(t0)
        print(FoM0)
        print(t1)
        print(FoM1)
        print(t2)
        print(FoM2)
        print(t3)
        print(FoM3)

        # Joint
        print('Joint')
        path = dpath+'results/JHCBOmW0v_' + str(ve) + '.pdf'
        parameters = [r"\Omega_m", r"w_0"]
        g = plots.getSubplotPlotter()
        g.settings.figure_legend_frame = False
        g.settings.legend_fontsize = 10
        g.settings.alpha_filled_add=0.3
        g.triangle_plot([GD0, GD1, GD2, GD3], parameters,
                filled=False, title_limit=1,
                contour_colors=['black','red','blue','green'],
                param_limits={r"\alpha":[32.8, 33.7]
                                , r"\beta":[4.7, 5.35]
                                , r"h_0":[0.6, 0.85]
                                , r"\Omega_m":[0.0, 0.52]
                                , r"w_0":[-2.1, -0.2]}
                )
        g.export(path)

        # Joint SNIa+CMB+BAO
        print('SNIa+CMB+BAO')
        fend = 'mcscbv6_152_1_1_stat'
        ctag = str(fg1)
        print(fend, ctag)
        tag = 'SNIa+CMB+BAO (1057)'
        JSsamples = case1(cpath, dpath, fend, vbs, sps, 1, ctag, tag)
        GD4 = JSsamples
        t4 = GD4.getTable(limit=1).tableTex()
        FoM4 = 1./np.sqrt(np.linalg.det(GD4.cov()))

        print(t0)
        print(FoM0)
        print(t4)
        print(FoM4)

        #Joint OmW0
        print('Joint OmW0')
        path = dpath+'results/JHHSSOmW0v_' + str(ve) + '.pdf'
        parameters = [r"\Omega_m", r"w_0"]
        g = plots.getSubplotPlotter()
        g.settings.figure_legend_frame = False
        g.settings.legend_fontsize = 10
        g.settings.alpha_filled_add=0.3
        g.triangle_plot([GD0, GD4], parameters,
                filled=False,
                contour_colors=['black','red'],
                param_limits={r"\Omega_m":[0.25, 0.35]
                                , r"w_0":[-1.2, -0.74]}
                )
        g.export(path)
    elif fg1 == 2:
        # HIIG+CMB+BAO
        print('HIIG+CMB+BAO (190)')
        fend ='mchcbv8_'+str(ve)+'_'+str(spl)+str(zps)+str(fg1)+str(drd)+str(obs)+str(fr0)
        ctag = str(fg1)
        print(fend, ctag)
        tag = 'HIIG+CMB+BAO (190)'
        GD0 = case2(cpath, dpath, fend, vbs, sps, prs, ctag, tag)
        t0 = GD0.getTable(limit=1).tableTex()
        FoM0 = 1./np.sqrt(np.linalg.det(GD0.cov()))

        # Joint SNIa+CMB+BAO
        print('SNIa+CMB+BAO (1057)')
        fend = 'mcscbv6_152_1_2_stat'
        ctag = str(fg1)
        print(fend, ctag)
        tag = 'SNIa+CMB+BAO (1057)'
        GD1 = case2(cpath, dpath, fend, vbs, sps, 1, ctag, tag)
        t1 = GD1.getTable(limit=1).tableTex()
        FoM1 = 1./np.sqrt(np.linalg.det(GD1.cov()))

        print(t0)
        print(FoM0)
        print(t1)
        print(FoM1)

        #Joint W0Wa
        print('Joint W0Wa')
        path = dpath+'results/JHHSSW0Wav_' + str(ve) + '.pdf'
        parameters = [r"w_a", r"w_0"]
        g = plots.getSubplotPlotter()
        g.settings.figure_legend_frame = False
        g.settings.legend_fontsize = 10
        g.settings.alpha_filled_add=0.3
        g.triangle_plot([GD0, GD1], parameters,
                filled=False,
                contour_colors=['black','red'],
                param_limits={r"\Omega_m":[0.25, 0.35]
                                , r"w_0":[-1.2, -0.74]
                                , r"w_a":[-1.5, 1.5]}
                )
        g.export(path)

    print(datetime.datetime.now())
    return
