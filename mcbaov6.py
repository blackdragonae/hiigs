#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# :Name:     mcbaov6.py
# :Purpose:  MCMC for HII galaxies
# :Author:   Ricardo Chavez -- Apr 13th, 2020 -- Morelia, Mexico
# :Modified: Testing old features with new likelihood
#------------------------------------------------------------------------------
from astropy.table import Table
from astropy.visualization import hist
import numpy as np
from astropy import constants as const
from astropy.io import ascii
import scipy.optimize as op
import scipy.stats as st
import corner
import scipy  as sp
from getdist import plots, MCSamples
import sys
# from hubblediagram import hubblediagram
import datetime
import glob
import json
import pymultinest
from pymultinest.solve import solve
from astropy.cosmology import Flatw0waCDM

import time

import matplotlib.pyplot as plt

c = const.c.to('km/s').value
k_b = const.k_B.to('J/K').value
m_p = const.m_p.to('kg').value


def zst(h0, Om, wb):
    g1 = (0.0783*(wb)**(-0.238))/(1.0 + 39.5*(wb)**(0.763))
    g2 = 0.560/(1.0 + 21.1*(wb)**(1.81))
    Zst = 1048.0*(1.0 + 0.00124*(wb)**(-0.738))*(1.0 + g1*(Om*h0**2)**(g2))
    return Zst


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


def lnprior0(cube):
    cube[0] = cube[0]
    return cube


def lnlike0(cube):
    Om = cube[0]
    theta = Om
    xsq = chsq0(theta)
    return -0.5*xsq


def chsq0(theta):
    Om = theta

    h0 = 0.6774
    # h0 = 0.7
    w0 = -1.0
    w1 = 0.0
    wb = 0.02225

    zs = zst(h0, Om, wb)
    rd = rs(zs, h0, Om, w0, w1, wb)
    #---------------------------------------------------------------------------
    lm = np.zeros(10)

    ix0 = [0, 1, 2, 3]
    for i in ix0:
        zi = bao_data[i,0]
        rdf = bao_data[i,3]
        Dv = DV(zi, h0, Om, w0, w1)
        lm[i] = (Dv*rdf)/rd

    ix1 = [4, 5, 6, 7, 8]
    for i in ix1:
        zi = bao_data[i,0]
        Da = DA(zi, h0, Om, w0, w1)
        lm[i] = Da/rd

    ix2 = [9]
    zi = bao_data[ix2,0]
    Dm = DM(zi, h0, Om, w0, w1)
    lm[ix2] = Dm/rd

    # print lm[ix2], bao_data[ix2,1]
    # print lm[ix2] - bao_data[ix2,1]
    # print bao_data[ix2,2]

    ixf = np.concatenate((ix0, ix1, ix2))
    R = lm[ixf] - bao_data[ixf,1]
    W = 1.0/(bao_data[ixf,2]**2)

    # xsq  = np.sum(R**2*W)
    xsq = np.dot(np.dot(R, rev_C_bao), np.transpose(R))
    # xsq = np.dot(np.dot(R, rev_C_bao[0:9, 0:9]), np.transpose(R))
    return xsq


def lnprior1(cube):
    cube[1] = cube[1]*2 - 2
    return cube


def lnlike1(cube):
    Om, w0 = cube[0], cube[1]
    theta = [Om, w0]
    xsq = chsq1(theta)
    return -0.5*xsq


def chsq1(theta):
    Om, w0 = theta

    h0 = 0.6774
    w1 = 0.0
    wb = 0.02225
    zs = zst(h0, Om, wb)
    rd = rs(zs, h0, Om, w0, w1, wb)
    #---------------------------------------------------------------------------
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

    # xsq  = np.sum(R**2*W)
    # xsq = np.dot(np.dot(R, rev_C_bao), np.transpose(R))
    xsq = np.dot(np.dot(R, rev_C_bao[0:6, 0:6]), np.transpose(R))
    return xsq


def lnprior2(cube):
    cube[0] = cube[0]*0.5 + 0.5
    cube[1] = cube[1]
    cube[2] = cube[2]*2.0 - 2.0
    return cube


def lnlike2(cube):
    h0, Om, w0 = cube[0], cube[1], cube[2]
    theta = [h0, Om, w0]
    xsq = chsq2(theta)
    return -0.5*xsq


def chsq2(theta):
    h0, Om, w0 = theta

    w1 = 0.0
    wb = 0.02225
    zs = zst(h0, Om, wb)
    rd = rs(zs, h0, Om, w0, w1, wb)
    #---------------------------------------------------------------------------
    lm = np.zeros(10)

    ix0 = [0, 1, 2, 3]
    for i in ix0:
        zi = bao_data[i,0]
        rdf = bao_data[i,3]
        Dv = DV(zi, h0, Om, w0, w1)
        lm[i] = (Dv*rdf)/rd

    ix1 = ix1 = [4, 5] #, 6, 7, 8] [4, 5, 6, 7, 8]
    for i in ix1:
        zi = bao_data[i,0]
        Da = DA(zi, h0, Om, w0, w1)
        lm[i] = Da/rd

    ix2 = [9]
    zi = bao_data[ix2,0]
    Dm = DM(zi, h0, Om, w0, w1)
    lm[ix2] = Dm/rd

    # ixf = np.concatenate((ix0, ix1, ix2))
    ixf = np.concatenate((ix0, ix1))

    # R = lm - bao_data[:,1]
    R = lm[ixf] - bao_data[ixf,1]
    # W = 1.0/(bao_data[:,2]**2)
    W = 1.0/(bao_data[ixf,2]**2)

    # xsq  = np.sum(R**2*W)
    xsq = np.dot(np.dot(R, rev_C_bao[0:6, 0:6]), np.transpose(R))
    return xsq


def mcbaov6(ve, dpath, cpath, clc = 0 , opt = 0, sps = 1000, prs = 0):

    print('+++++++++++++++++++++++++++++++++++++++++++')
    print('mcbaov6: ')
    print('+++++++++++++++++++++++++++++++++++++++++++')

    #===============================================================
    #============= Parameters ======================================
    #===============================================================
    #Fixed beta

    fg0 = clc
    fg1 = opt
    fsim = 1

    global bao_data
    global rev_C_bao

    fend ='mcbv6_'+str(ve)+'_'+str(fg1)
    print(fend)
    #============= Main Body =======================================
    vf = np.zeros(10)

    bao_data = np.array([[0.122, 539.0, 17.0, 147.5], [0.44, 1716.40, 83.00, 148.6], [0.60, 2220.80, 101.00, 148.6], [0.73, 2516.00, 86.00, 148.6], [0.32, 6.67, 0.13, 148.11], [0.57, 9.52, 0.19, 148.11], [1.19, 12.6621, 0.9876, 147.78], [1.50, 12.4349, 1.0429, 147.78], [1.83, 13.1305, 1.0465, 147.78], [2.40, 36.0, 1.2, -1.0]])

# [2.40, 36.0, 1.2, -1.0]

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
    if fg1 == 0:
        parameters = ["\Omega_m"]
        n_params = len(parameters)

        prefix = cpath + fend

        result = solve(LogLikelihood=lnlike0, Prior=lnprior0,
            n_dims=n_params, outputfiles_basename=prefix, verbose=False,
            n_live_points = sps, resume = prs, init_MPI = False,
            sampling_efficiency = 0.8)

        print('parameter values:')
        theta = np.zeros(n_params)
        vMean = np.zeros(n_params)
        vStd = np.zeros(n_params)
        i = 0
        for name, col in zip(parameters, result['samples'].transpose()):
            print('%15s : %.3f +- %.3f' % (name, col.mean(), col.std()))
            vMean[i] = col.mean()
            vStd[i] = col.std()
            theta[i] = col.mean()
            i+=1

        print('Chsq min:')
        chsq = chsq0(theta)
        dof = 10 - n_params

        print(chsq0(theta))
        print('10')
        print(chsq0(theta)/(10 - n_params))

        #Plot getdist
        path = dpath+'results/GDtr'+fend+'.pdf'
        GDsamples = MCSamples(samples=result['samples'], names = parameters
            , labels = parameters, name_tag = 'BAO'
            # , ranges={"\Omega_m":[0.0, 0.6]
            #         , "w_0":[-2.0, 0.0]}
                    )
        GDsamples.updateSettings({'contours': [0.68, 0.95, 0.99]})

        g = plots.getSubplotPlotter()
        g.settings.num_plot_contours = 3
        g.triangle_plot(GDsamples, filled=True)
        g.export(path)

        f = open(dpath+'results/GDlog'+fend+'.txt', 'w')
        t = GDsamples.getTable(limit = 1).tableTex()
        f.write(t)
        f.write('--------------------------------------\n')
        f.write(str(chsq)+'\n')
        f.write(str(chsq/dof)+'\n')
        f.close

        # Plot Corner
        path = dpath+'results/tr'+fend+'.pdf'
        fig = corner.corner(result['samples'],
                        labels=["$\Omega_m$", "$w_0$"],
                        # truths= vMean,
                        # quantiles = [0.16, 0.84],
ix1                        # range = [(0.0, 0.6), (-2.0, 0.0)],
                        plot_contours = 'true',
                        levels = 1.0 - np.exp(
                           -0.5 * np.arange(1.0, 3.1, 1.0) ** 2),
                        smooth = 1.0,
                        bins = 100,
                        color = 'black',
                        show_titles = 1)
        plt.savefig(path)
    elif fg1 == 1:
        parameters = ["\Omega_m", "w_0"]
        n_params = len(parameters)

        prefix = cpath + fend

        result = solve(LogLikelihood=lnlike1, Prior=lnprior1,
            n_dims=n_params, outputfiles_basename=prefix, verbose=False,
            n_live_points = sps, resume = prs, init_MPI = False,
            sampling_efficiency = 0.8)

        print('parameter values:')
        theta = np.zeros(n_params)
        vMean = np.zeros(n_params)
        vStd = np.zeros(n_params)
        i = 0
        for name, col in zip(parameters, result['samples'].transpose()):
            print('%15s : %.3f +- %.3f' % (name, col.mean(), col.std()))
            vMean[i] = col.mean()
            vStd[i] = col.std()
            theta[i] = col.mean()
            i+=1

        print('Chsq min:')
        print(chsq1(theta))
        print('6')
        print(chsq1(theta)/(6 - n_params))

        #Plot getdist
        path = dpath+'results/GDtr'+fend+'.pdf'
        GDsamples = MCSamples(samples=result['samples'], names = parameters
            , labels = parameters, name_tag = 'BAO'
            , ranges={"\Omega_m":[0.0, None]}
            #         , "w_0":[-2.0, 0.0]}
                    )
        GDsamples.updateSettings({'contours': [0.68, 0.95]})

        g = plots.getSubplotPlotter()
        g.settings.num_plot_contours = 2
        g.triangle_plot(GDsamples, filled=True)
        g.export(path)

        f = open(dpath+'results/GDlog'+fend+'.txt', 'w')
        t = GDsamples.getTable(limit = 1).tableTex()
        f.write(t)
        f.close

        # Cov Matix and FoM
        FoM = 1./np.sqrt(np.linalg.det(GDsamples.cov()))
        print('FoM = ', FoM)

        # Plot Corner
        path = dpath+'results/tr'+fend+'.pdf'
        fig = corner.corner(result['samples'],
                        labels=["$\Omega_m$", "$w_0$"],
                        # truths= vMean,
                        # quantiles = [0.16, 0.84],
                        # range = [(0.0, 0.6), (-2.0, 0.0)],
                        plot_contours = 'true',
                        levels = 1.0 - np.exp(
                           -0.5 * np.arange(1.0, 3.1, 1.0) ** 2),
                        smooth = 1.0,
                        bins = 100,
                        color = 'black',
                        show_titles = 1)
        plt.savefig(path)
    elif fg1 == 2:
        parameters = ["h_0", "\Omega_m", "w_0"]
        n_params = len(parameters)

        prefix = cpath + fend

        result = solve(LogLikelihood=lnlike2, Prior=lnprior2,
            n_dims=n_params, outputfiles_basename=prefix, verbose=False,
            n_live_points = sps, resume = prs, init_MPI = False,
            sampling_efficiency = 0.8)

        print('parameter values:')
        theta = np.zeros(n_params)
        vMean = np.zeros(n_params)
        vStd = np.zeros(n_params)
        i = 0
        for name, col in zip(parameters, result['samples'].transpose()):
            print('%15s : %.3f +- %.3f' % (name, col.mean(), col.std()))
            vMean[i] = col.mean()
            vStd[i] = col.std()
            theta[i] = col.mean()
            i+=1

        print('Chsq min:')
        print(chsq2(theta))
        print('10')
        print(chsq2(theta)/(10 - n_params))

        #Plot getdist
        path = dpath+'results/GDtr'+fend+'.pdf'
        GDsamples = MCSamples(samples=result['samples'], names = parameters
            , labels = parameters, name_tag = 'BAO'
            # , ranges={"\Omega_m":[0.0, 0.6]
            #         , "w_0":[-2.0, 0.0]}
                    )
        GDsamples.updateSettings({'contours': [0.68, 0.95, 0.99]})

        g = plots.getSubplotPlotter()
        g.settings.num_plot_contours = 3
        g.triangle_plot(GDsamples, filled=True)
        g.export(path)

        f = open(dpath+'results/GDlog'+fend+'.txt', 'w')
        t = GDsamples.getTable(limit = 1).tableTex()
        f.write(t)
        f.close

        # Plot Corner
        path = dpath+'results/tr'+fend+'.pdf'
        fig = corner.corner(result['samples'],
                        # labels=["$\Omega_m$", "$w_0$"],
                        # truths= vMean,
                        # quantiles = [0.16, 0.84],
                        # range = [(0.0, 0.6), (-2.0, 0.0)],
                        plot_contours = 'true',
                        levels = 1.0 - np.exp(
                           -0.5 * np.arange(1.0, 3.1, 1.0) ** 2),
                        smooth = 1.0,
                        bins = 100,
                        color = 'black',
                        show_titles = 1)
        plt.savefig(path)

    print(datetime.datetime.now())
    return
