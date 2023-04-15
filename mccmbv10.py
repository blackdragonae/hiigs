#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# :Name:     mccmbv10.py
# :Purpose:  MCMC for CMB
# :Author:   Ricardo Chavez -- Apr 13th, 2020 -- Morelia, Michoacan
# :Modified:
#------------------------------------------------------------------------------
from astropy.table import Table
from astropy.visualization import hist
import numpy as np
import scipy  as sp
from astropy import constants as const
from astropy.io import ascii
from astropy.cosmology import Flatw0waCDM
import json
import corner
from getdist import plots, MCSamples
import pymultinest
from pymultinest.solve import solve
import sys
import time
import matplotlib.pyplot as plt
import datetime
import glob

c = const.c.to('km/s').value
k_b = const.k_B.to('J/K').value
m_p = const.m_p.to('kg').value


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


def lnprior1(cube):
    cube[1] = cube[1]*2 - 2
    cube[2] = cube[2]*0.05
    return cube


def lnlike1(cube):
    Om, w0, wb = cube[0], cube[1], cube[2]
    theta = [Om, w0, wb]
    xsq = chsq1(theta)
    return -0.5*xsq


def chsq1(theta):
    Om, w0, wb = theta

    h0 = 0.6774
    Or = 4.153e-5 * h0**(-2)
    Ok = 0.0
    w1 = 0.0
    Ob = wb * h0**(-2.0)

    if Ob > Om: return 9e9
    #---------------------------------------------------------------------------
    cmb_data = np.array([[301.77, 0.09], [1.7482, 0.0048], [0.02226, 0.00016]])
    rev_C_cmb = np.array([[ 1.47529862e+02, -9.50243997e+02,  6.75330855e+03],
       [-9.49518029e+02,  8.87656028e+04,  1.66515286e+06],
       [ 6.78491359e+03,  1.66491606e+06,  7.46953427e+07]])
    #---------------------------------------------------------------------------
    #Modeling
    vCMBth = [la(h0, Om, w0, w1, wb), Rz(h0, Om, w0, w1, wb), wb]
    res = vCMBth - cmb_data[:,0]

    #ChiSq
    xsq = np.dot(np.dot(np.transpose(res), rev_C_cmb), res)

    return xsq


def lnprior2(cube):
    cube[0] = cube[0]*0.5 + 0.5
    cube[1] = cube[1]
    cube[2] = cube[2]*2 - 2
    cube[3] = cube[3]*0.05
    return cube


def lnlike2(cube):
    h0, Om, w0, wb = cube[0], cube[1], cube[2], cube[3]
    theta = [h0, Om, w0, wb]
    xsq = chsq2(theta)
    return -0.5*xsq


def chsq2(theta):
    h0, Om, w0, wb = theta

    w1 = 0.0
    Ob = wb * h0**(-2.0)

    if Ob > Om: return 9e9
    #---------------------------------------------------------------------------
    cmb_data = np.array([[301.77, 0.09], [1.7482, 0.0048], [0.02226, 0.00016]])
    rev_C_cmb = np.array([[ 1.47529862e+02, -9.50243997e+02,  6.75330855e+03],
       [-9.49518029e+02,  8.87656028e+04,  1.66515286e+06],
       [ 6.78491359e+03,  1.66491606e+06,  7.46953427e+07]])
    #---------------------------------------------------------------------------
    #Modeling
    vCMBth = [la(h0, Om, w0, w1, wb), Rz(h0, Om, w0, w1, wb), wb]
    res = vCMBth - cmb_data[:,0]

    #ChiSq
    xsq = np.dot(np.dot(np.transpose(res), rev_C_cmb), res)

    return xsq


def mccmbv10(ve, dpath, cpath, clc = 0 , opt = 0, sps = 1000, prs = 0):

    print('+++++++++++++++++++++++++++++++++++++++++++')
    print('mccmbv10: ')
    print('+++++++++++++++++++++++++++++++++++++++++++')

    #===============================================================
    #============= Parameters ======================================
    #===============================================================
    #Fixed beta

    fg0 = clc
    fg1 = opt

    fend ='mccv10_'+str(ve)+'_'+str(fg1)
    print(fend)
    #============= Main Body =======================================
    print(datetime.datetime.now())
    start = datetime.datetime.now()

    # Cases
    if fg1 == 1:
        parameters = ["\Omega_m", "w_0", "w_b"]
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
            theta[i] = col.mean()
            i+=1

        print('Chsq min:')
        # theta = [0.00, -0.46]
        print(chsq1(theta))
        print('1')

        #Plot getdist
        path = dpath+'results/GDtr'+fend+'.pdf'
        GDsamples = MCSamples(samples=result['samples'], names = parameters
            , labels = parameters, name_tag = 'CMB'
            # , ranges={"\Omega_m":[0.0, 0.6]
            #         , "w_0":[-2.0, 0.0]}
                    )
        GDsamples.updateSettings({'contours': [0.68, 0.95, 0.99]})

        g = plots.getSubplotPlotter()
        g.settings.num_plot_contours = 2
        g.triangle_plot(GDsamples, filled=True)
        g.export(path)

        f = open(dpath+'results/GDlog'+fend+'.txt', 'w')
        t = GDsamples.getTable(limit = 1).tableTex()
        f.write(t)
        f.close

        # Cov Matix and FoM
        pars=[r"\Omega_m", r"w_0"]
        FoM = 1./np.sqrt(np.linalg.det(GDsamples.cov(pars)))
        print('FoM = ', FoM)

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
                        color = 'black'
                        # show_titles = 1
                        )
        plt.savefig(path)
    elif fg1 == 2:
        parameters = ["h_0", "\Omega_m", "w_0", "w_b"]
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
            theta[i] = col.mean()
            i+=1

        print('Chsq min:')
        print(chsq2(theta))
        print('3')
        print(chsq2(theta)/(3 - n_params))

        #Plot getdist
        path = dpath+'results/GDtr'+fend+'.pdf'
        GDsamples = MCSamples(samples=result['samples'], names = parameters
            , labels = parameters, name_tag = 'CMB'
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
                        plot_contours = 'true',
                        levels = 1.0 - np.exp(
                           -0.5 * np.arange(1.0, 3.1, 1.0) ** 2),
                        smooth = 1.0,
                        bins = 100,
                        show_titles = 1
                        )
        plt.savefig(path)


    print(datetime.datetime.now())
    return
