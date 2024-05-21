#  ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#  :Name:     mch2gv79.py
#  :Purpose:  MCMC for HII galaxies KMOS sample
#  :Author:   Ricardo Chavez -- Jun 8th, 2020 -- Morelia, Mexico
#  :Notes: testing samples, sampling_efficiency: 0.8, subsample testing
#  ------------------------------------------------------------------------------
from astropy.table import Table
from astropy.visualization import hist
from astropy.cosmology import FlatwCDM
from astropy.cosmology import Flatw0waCDM
from astropy.cosmology import w0waCDM
from astropy.cosmology import LambdaCDM
from astropy.cosmology import FlatLambdaCDM
from astropy.cosmology import z_at_value
from matplotlib.ticker import FormatStrFormatter
import matplotlib.mlab as mlab
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import astropy.units as u
import scipy.stats as st
import numpy as np
from astropy import constants as const
from pymultinest.solve import solve
import corner
from getdist import plots, MCSamples
import datetime
import matplotlib.pyplot as plt
from matplotlib import gridspec
import pickle
import pandas as pd

c = const.c.to('km/s').value
k_b = const.k_B.to('erg/K').value
m_p = const.m_p.to('g').value

EBVG = 0.184*1.2
EBVGErr = 0.048

EBVC = 0.1212*1.2
EBVCErr = 0.037

LSL = 1.83  # 1.84
x0 = 1.79
x0Err = 0.08


def lsprior(cube):
    cube[0] = cube[0]*2. + 32.5
    cube[1] = cube[1]*1. + 4.5
    return cube


def lslike(cube):
    alpha, beta = cube[0], cube[1]
    theta = alpha, beta
    xsq = lschsq(theta)[0]
    return -0.5*xsq


def lschsq(theta):
    alpha, beta = theta
    x, y, xerr, yerr = lsdata

    ym = beta*x + alpha
    ymerr = np.sqrt(beta**2*(xerr)**2
            # - 2.0*beta*Rxy*xerr*yerr
            # + 0.22**2
            )

    R = (y - ym)
    W = 1.0/(yerr**2 + ymerr**2)

    xsq = np.sum(R**2*W)
    xsq2 = xsq - np.sum(np.log(W))

    if lsfr == 2:
        return (xsq2, R, ym, ymerr)
    else:
        return (xsq, R, ym, ymerr)


def lsfit(dpath, cpath, fend, prs, x, xerr, y, yerr):
    global lsdata
    global lsfr
    lsfr = 1

    lsdata = x, y, xerr, yerr
    parameters = [r"\alpha",r"\beta"]
    parametersc = [r"$\alpha$",r"$\beta$"]
    n_params = len(parameters)

    fend = fend + '_LS_'
    prefix = cpath + fend

    sps = 10000
    vbs = 1

    result = solve(LogLikelihood=lslike, Prior=lsprior,
                   n_dims=n_params, outputfiles_basename=prefix, verbose=vbs,
                   n_live_points=sps, resume=prs, init_MPI=False,
                   sampling_efficiency=0.8)

    tag = 'LS'
    # ploter(dpath, fend, result, parameters, parametersc, tag, ctag)

    path = dpath+'results/GDtr'+fend+'.pdf'
    GDsamples = MCSamples(samples=result['samples'], names=parameters,
                          labels=parameters, name_tag=tag #,
                          # ranges={r"\Omega_m":[0.0, None]}
                          )

    g = plots.getSubplotPlotter() #width_inch=4
    g.settings.num_plot_contours = 2
    g.settings.title_limit_fontsize = 14
    g.triangle_plot(GDsamples, filled=True, title_limit=1,
            contour_colors=['crimson'],
            param_limits={r"\alpha":[32.8, 33.7]
                        # , r"\beta":[4.7, 5.35]
                        # , r"h":[0.6, 0.85]
                        # , r"\Omega_m":[0.0, 0.52]
                        # , r"w_0":[-2.1, -0.2]
                        }
            )
    g.export(path)

    t = GDsamples.getTable(limit=1).tableTex()
    theta = GDsamples.getMeans()

    # Print parameter values
    print('parameter values:')
    for name, col in zip(parameters, result['samples'].transpose()):
        print('%15s : %.3f +- %.3f' % (name, col.mean(), col.std()))

    print(t)

    fy = theta[1]*x + theta[0]

    return fy


def ploter(dpath, cpath, fend, prs, result, parameters, parametersc, tag, ctag):
    x, y, z, xerr, yerr, zerr, Rxy, Muo, MuoErr, vTg = Mdata
    fchsq = 'chsq' + str(ctag)
    n_params = len(parameters)
    nobs = len(x)
    print(fchsq, nobs)

    # Plot getdist
    path = dpath+'results/GDtr'+fend+'.pdf'
    GDsamples = MCSamples(samples=result['samples'], names=parameters
                          , labels=parameters, name_tag=tag
                          , ranges={r"\Omega_m":[0.0, None],
                                  r"\Omega_{\Lambda}":[0.0, None]}
                          )

    g = plots.getSubplotPlotter(width_inch=10.5) #
    g.settings.num_plot_contours = 2
    g.settings.title_limit_fontsize = 16
    g.settings.axes_fontsize = 16
    g.settings.legend_fontsize = 16
    g.settings.lab_fontsize = 16
    g.triangle_plot(GDsamples, filled=True, title_limit=1,
            contour_colors=['crimson'],
            param_limits={r"\alpha":[32.8, 33.8]
                               , r"\beta":[4.7, 5.4]
                               , r"M_B":[-19.35, -19.15]
                               , r"h":[0.6, 0.85]
                               , r"\Omega_m":[0.0, 0.6]
                               , r"\Omega_{\Lambda}":[0.0, 1.0]
                               , r"w_0":[-2.0, 0.0]
                               , r"w_a":[-1.0, 1.0]
                               , r"\omega_b":[0.020, 0.0238]
                            }
            )
    g.export(path)

    t = GDsamples.getTable(limit=1).tableTex()
    theta = GDsamples.getMeans()
    # dic = GDsamples.getParamBestFitDict()

    print('parameter values:')
    print(theta)
    # print(dic)

    chsq = globals()[fchsq](theta)[0]
    res = globals()[fchsq](theta)[1]

    # Cov Matix and FoM
    cpars=[r"\Omega_m", r"w_0"]
    cpars=[r"h"]
    # FoM = 1./np.sqrt(np.linalg.det(GDsamples.cov(pars=cpars)))
    FoM = 1./np.sqrt(np.linalg.det(GDsamples.cov()))
    print('FoM = ', FoM)
    print(GDsamples.cov())

    # Plot Corner
    path = dpath+'results/tr'+fend+'.pdf'
    fig = corner.corner(result['samples'],
                        labels=parametersc,
                        plot_contours='true',
                        levels=1.0 - np.exp(
                            -0.5 * np.arange(1.0, 3.1, 1.0) ** 2),
                        # smooth=1.0,
                        bins=100,
                        color='black',
                        show_titles=1)
    plt.savefig(path)

    # Print parameter values
    print('parameter values:')
    for name, col in zip(parameters, result['samples'].transpose()):
        print('%15s : %.3f +- %.3f' % (name, col.mean(), col.std()))

    print(t)

    # print(GDsamples.getBestFit(max_posterior=True))
    # print(GDsamples.getBestFit(max_posterior=False))

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
    ixn = np.where(abs(nres) > 3.)
    print('N3=', vTg[ixn])

    path = dpath+'results/Hnres'+fend+'.pdf'
    fig = plt.subplots(1)
    hist(nres, bins='knuth', histtype='stepfilled',
         alpha=0.2)
    plt.savefig(path)

    ksD, ksP = st.kstest(nres, 'norm')

    print(ksD, ksP)
    print(np.std(res))

    # GDsamples.getGelmanRubin()

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
    f.close()

    # print(getMargeStats(include_bestfit=True))

    # L-Sigma
    # Params
    # dmu = 1
    #
    # if dmu == 1:
    #     Nms = 1000
    #     ah = np.random.choice(result['samples'][:, 2], Nms, replace=False)
    #     aOm = np.random.choice(result['samples'][:, 3], Nms, replace=False)
    #     aw0 = np.random.choice(result['samples'][:, 4], Nms, replace=False)
    #
    #     print(np.mean(aw0), np.std(aw0), len(aw0))
    #
    #     path = dpath+'results/HPP'+fend+'.pdf'
    #     fig = plt.subplots(1)
    #     hist(aw0, bins='knuth', histtype='stepfilled', alpha=0.2)
    #     plt.savefig(path)
    #
    #     ixG = np.where(z>10.0)
    #     ixH = np.where(z<10.0)
    #
    #     Mum = z*0.0
    #     MumErr = z*0.0
    #
    #     Mum[ixG] = z[ixG]
    #     MumErr[ixG] = zerr[ixG]
    #
    #     print(len(ixH[0]))
    #     for i in ixH[0]:
    #         print(i, z[i])
    #         aMum = np.zeros(Nms)
    #         aMumErr = np.zeros(Nms)
    #
    #         for j in range(len(ah)):
    #             cosmo = FlatwCDM(H0=ah[j]*100.0, Om0=aOm[j], w0=aw0[j])
    #             aMum[j] = 5.0*np.log10(cosmo.luminosity_distance(z[i]).value) + 25.0
    #
    #         Mum[i] = np.mean(aMum)
    #         MumErr[i] = np.std(aMum)
    #
    #     vMu = Mum
    #     vMuErr = MumErr
    #
    #     with open(dpath+'indat/distmod.pickle', 'wb') as handle:
    #         print('saving ...')
    #         pickle.dump([vMu, vMuErr], handle, protocol=pickle.HIGHEST_PROTOCOL)
    #
    # else:
    #     with open(dpath+'indat/distmod.pickle', 'rb') as handle:
    #         vMu, vMuErr = pickle.load(handle)
    #
    # path = dpath+'results/LS'+fend+'.pdf'

    #*******************************************************************
    path = dpath+'results/LS'+fend+'.pdf'
    vMu = globals()[fchsq](theta)[2]
    vMuErr = globals()[fchsq](theta)[3]
    xp = x
    xsig = xerr
    yp = y + 0.4*(vMu + DMu) + 40.08
    # yp = y + 0.4*(vMu) + 40.08
    ysig = np.sqrt(yerr**2 + 0.4**2*vMuErr**2)
    
    # if ((zs == 1) or (zs == 6)):
    #     fy = beta*x + alpha
    # elif zs == 2:
    #     fy = theta[0]*x
    # elif zs == 3:
    #     fy = 5.0*x
    # else:
    #     fy = theta[1]*x + theta[0]
    
    # fy = theta[1]*x + theta[0]
    fy = lsfit(dpath, cpath, fend, prs, xp, xsig, yp, ysig)
    lsres = yp - fy

    # Test
    DL = np.mean(lsres)
    print(len(lsres))
    nresL = (lsres - DL)/np.std(lsres)

    ksDL, ksPL = st.kstest(nresL, 'norm')

    print('L-Sigma residuals test:')
    print(ksDL, ksPL)
    print(np.std(lsres))
    
    frms = np.std(lsres)
    nob = len(x)
    
    xmin = min(xp)- 0.1
    xmax = max(xp)+ 0.1
    ymin = min(yp)- 0.3
    ymax = max(yp)+ 0.6
    
    xl = "$\log \sigma (\mathrm{H}\\beta)$ $\mathrm{(km\ s^{-1})}$"
    yl = "$\log L(\mathrm{H}\\beta)$ $\mathrm{(erg\ s^{-1})}$"
    
    # ixz = np.where((z > 0.2) & (z < 10))
    # ix0 = np.arange(len(x))
    # ixn = np.where(ix0 > 188)
    
    plt.figure(figsize=(6, 6))
    # gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1])
    # ax1 = plt.subplot(gs[0])
    ax1 = plt.subplot(111)
    plt.errorbar(xp, yp, xerr=xsig, yerr=ysig, fmt='o', markersize=3.0, lw=0.6, capsize=0.0, alpha=0.5)
    # plt.errorbar(xp[ixn], yp[ixn], xerr=xsig[ixn], yerr=ysig[ixn], fmt='ro', markersize=0.5)
    # plt.plot(xp[ixn], yp[ixn], 'r*', markersize=0.4)
    plt.plot(xp, fy, lw=2, c='red')
    
    plt.annotate("$\log L(H"+ r"\beta"+") = (4.99 \pm 0.11) \log \sigma  + (33.29 \pm 0.14)$", xy=(xmin+0.05, ymax-0.3),
                 xytext=(xmin+0.05, ymax-0.3), fontsize=14)
    
    plt.annotate("$\sigma_{\log L} = $"+str("%5.3f" % frms)
                 + "$, N = $"+str("%5.0f" % nob), xy=(xmin+0.05, ymax-0.6),
                 xytext=(xmin+0.05, ymax-0.6), fontsize=14)
    
    for i in range(len(xp)):
        # plt.annotate(vTg[i], xy=(xp[i], yp[i]), fontsize=2)
        if vTg[i] == 6:
            plt.errorbar(xp[i], yp[i], xerr=xsig[i], yerr=ysig[i], fmt='og', markersize=6.0, lw=0.6, capsize=0.0, alpha=0.8)
        if vTg[i] == 7:
            plt.errorbar(xp[i], yp[i], xerr=xsig[i], yerr=ysig[i], fmt='or', markersize=6.0, lw=0.6, capsize=0.0, alpha=0.8)
        if vTg[i] == 0:
            plt.errorbar(xp[i], yp[i], xerr=xsig[i], yerr=ysig[i], fmt='om', markersize=3.0, lw=0.6, capsize=0.0, alpha=0.5)

    
    plt.axis([xmin, xmax, ymin, ymax])
    plt.xlabel(xl, fontsize=18)
    plt.ylabel(yl, fontsize=18)

    axins = inset_axes(ax1, width="30%", height="50%",
                   bbox_to_anchor=(.67, .1, .97, .6),
                   bbox_transform=ax1.transAxes, loc=3)
    
    axins.hist(nresL, bins=8, histtype='stepfilled', alpha=0.2, density=True)
    (mu, sigma) = st.norm.fit(nresL)
    print(mu, sigma)
    xn = np.linspace(-3.0, 3.0, 100)
    yn = st.norm.pdf(xn, mu, sigma)
    axins.plot(xn, yn, 'r', linewidth=2)
    axins.tick_params(axis='both', which='major', labelsize=12)
    axins.set_xlabel(r'$\Delta \log L/ \sigma_{\log L}$', fontsize=14)
    axins.set_ylabel(r'$pdf$', fontsize=14)
    
    # plt.setp(ax1.get_xticklabels(), visible=False)
    
    # ax2 = plt.subplot(gs[1], sharex=ax1)
    # res = yp - fy
    # plt.errorbar(xp, res, yerr=ysig, fmt='o', markersize=1.5, lw=0.5, capsize=0.0)
    # plt.plot(xp, fy-fy, lw=2, c='red')

    # for i in range(len(xp)):
    #     # plt.annotate(vTg[i], xy=(xp[i], yp[i]), fontsize=2)
    #     if vTg[i] == 6:
    #         plt.errorbar(xp[i], res[i], yerr=ysig[i], fmt='og', markersize=5.0, lw=1.0, capsize=0.0)
    #     if vTg[i] == 7:
    #         plt.errorbar(xp[i], res[i], yerr=ysig[i], fmt='or', markersize=5.0, lw=1.0, capsize=0.0)
    
    # plt.xlabel(xl, fontsize=18)
    # plt.ylabel(r'$\Delta log L$', fontsize=18)
    
    plt.savefig(path)

    #Hubble diagram
    ixG = np.where(z>10.0)
    ixH = np.where(z<10.0)
    zG = (theta[2]*100.0)*10**(0.2*(z[ixG] - 25.))/c
    z[ixG] = zG

    si = np.argsort(z)
    zs = z[si]
    Tgs = vTg[si]

    vMuo =  2.5*(theta[1]*x + theta[0]) - 2.5*y - 100.195
    vMuoErr = 2.5*np.sqrt((yerr)**2 + theta[1]**2*(xerr)**2)
    vMuos = vMuo[si]
    vMuoErrs = vMuoErr[si]

    vMus  = vMu[si]

    ix = np.where(zs < 10.0)
    ixi = np.where(zs < 0.15)

    print('NHDD=', len(zs[ix]))

    zm = np.linspace(0.0001, 8.0, 10000)
    cosmo = FlatwCDM(H0=theta[2]*100.0, Om0=theta[3], w0=theta[4], Tcmb0=2.725)
    cosmoU = FlatwCDM(H0=(theta[2]+ 0.039)*100.0, Om0=theta[3]+(0.12), w0=theta[4]+(0.52), Tcmb0=2.725)
    cosmoD = FlatwCDM(H0=(theta[2]- 0.039)*100.0, Om0=theta[3]-(0.069), w0=theta[4]-(0.29), Tcmb0=2.725)
    # cosmo = FlatwCDM(H0=theta[2]*100.0, Om0=theta[3], w0=-1.0, Tcmb0=2.725)
    # cosmo = FlatwCDM(H0=theta[2]*100.0, Om0=0.3, w0=-1.0, Tcmb0=2.725)
    cosmoL = FlatLambdaCDM(H0=theta[2]*100.0, Om0=1.0, Tcmb0=2.725)
    cosmoN = FlatwCDM(H0=67.37, Om0=0.3147, w0=-1.0, Tcmb0=2.725) # Planck 2018
    # cosmo = LambdaCDM(H0=theta[2]*100.0, Om0=theta[3], Ode0=theta[4])
    # cosmo = w0waCDM(H0=theta[2]*100.0, Om0=theta[3], Ode0=theta[4], w0=theta[5], wa=0.0, Tcmb0= 2.725)
    Mumz = cosmo.distmod(zm).value
    MumzU = cosmoU.distmod(zm).value
    MumzD = cosmoD.distmod(zm).value
    MumL = cosmoL.distmod(zm).value
    MumN = cosmoN.distmod(zm).value

    ages = np.array([10, 6, 4, 2, 1.5, 1, 0.8, 0.66])*u.Gyr
    ageticks = [z_at_value(cosmo.age, age) for age in ages]

    path = dpath+'results/Hubble'+fend+'.pdf'
    plt.figure(figsize=(8, 6))
    # gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1])
    # ax1 = plt.subplot(gs[0])
    ax1 = plt.subplot(111)

    plt.errorbar(zs[ix], vMuos[ix], yerr=vMuoErrs[ix], fmt='o', markersize=3.0, lw=0.6, capsize=0.0, alpha=0.5)
    plt.plot(zm, Mumz, lw=1.0, linestyle='-', alpha=1.0, c='black', label=r'$h, \Omega_m, w_0 = 0.731, 0.302, -1.01 $')
    # plt.fill_between(zm, MumzD, MumzU, color='red', alpha=0.5)
    plt.plot(zm, MumL, lw=1.0, c='black', linestyle='--', alpha=0.5, label=r'$h, \Omega_m = 0.731, 1.0 $')
    # plt.plot(zm, MumN, lw=2, c='blue', linestyle='-', alpha=0.5, label=r'Planck 2018')
    for i in range(len(zs[ix])):
        # plt.annotate(vTg[i], xy=(xp[i], yp[i]), fontsize=2)
        if Tgs[ix][i] == 6:
            plt.errorbar(zs[ix][i], vMuos[ix][i], yerr=vMuoErrs[ix][i], fmt='og', markersize=6.0, lw=0.6, capsize=0.0, alpha=0.8)
        if Tgs[ix][i] == 7:
            plt.errorbar(zs[ix][i], vMuos[ix][i], yerr=vMuoErrs[ix][i], fmt='or', markersize=6.0, lw=0.6, capsize=0.0, alpha=0.8)
        if Tgs[ix][i] == 0:
            plt.errorbar(zs[ix][i], vMuos[ix][i], yerr=vMuoErrs[ix][i], fmt='om', markersize=3.0, lw=0.6, capsize=0.0, alpha=0.5)
    
    # MCMC params
    nt = 400
    hs = np.random.uniform(theta[2]-(0.039), theta[2]+(0.039), nt)
    Oms = np.random.uniform(theta[3]-(0.069), theta[3]+(0.12), nt)
    ws = np.random.uniform(theta[4]-(0.29), theta[4]+(0.52), nt)

    for i in range(nt):
        # cosmos = FlatwCDM(H0=hs[i]*100.0, Om0=Oms[i], w0=ws[i], Tcmb0=2.725)
        cosmos = FlatwCDM(H0=hs[i]*100.0, Om0=Oms[i], w0=ws[i], Tcmb0=2.725)
        plt.plot(zm, cosmos.distmod(zm).value, lw=0.015, c='red', linestyle=':', alpha=0.015)
    
    plt.ylabel(r'$\mu$', fontsize=18)
    plt.xlabel(r'$z$', fontsize=18)
    plt.yticks(fontsize=16)
    plt.xticks(fontsize=16)
    plt.legend(loc='upper left', fontsize=10)

    ax3 = ax1.twiny()
    ax3.set_xticks(ageticks)
    ax3.set_xticklabels(['{:g}'.format(age) for age in ages.value])
    ax3.set_xlabel('Age of the Universe (Gyr)', fontsize=16)
    ax3.tick_params(axis='both', which='major', labelsize=16)

    zmin, zmax = 0.0, 8.0
    ax1.set_xlim(zmin, zmax)
    ax1.set_ylim(20.0, 56.0)
    ax3.set_xlim(zmin, zmax)

    axins = inset_axes(ax1, width="75%", height="68%",
                   bbox_to_anchor=(.1, .1, .65, .7),
                   bbox_transform=ax1.transAxes, loc=3)
    
    axins.errorbar(zs[ixi], vMuos[ixi], yerr=vMuoErrs[ixi], fmt='o', markersize=3.0, lw=0.6, capsize=0.0, alpha=0.5)
    axins.plot(zm, Mumz, lw=1.0, c='black')
    # axins.fill_between(zm, MumzD, MumzU, color='red', alpha=0.3)
    axins.plot(zm, MumL, lw=1.0, c='black', linestyle='--', alpha=0.5)
    # axins.plot(zm, MumN, lw=2, c='blue', linestyle='-', alpha=0.5)
    for i in range(len(zs[ixi])):
        if Tgs[ixi][i] == 0:
            plt.errorbar(zs[ixi][i], vMuos[ixi][i], yerr=vMuoErrs[ixi][i], fmt='om', markersize=3.0, lw=0.6, capsize=0.0, alpha=0.5)
    
    for i in range(nt):
        cosmos = FlatwCDM(H0=hs[i]*100.0, Om0=Oms[i], w0=ws[i], Tcmb0=2.725)
        axins.plot(zm, cosmos.distmod(zm).value, lw=0.015, c='red', linestyle=':', alpha=0.015)

    axins.set_xlim(0.0, 0.15)
    axins.set_ylim(20.0, 42.0)
    axins.tick_params(axis='both', which='major', labelsize=12)
    axins.set_xlabel(r'$z$', fontsize=14)
    axins.set_ylabel(r'$\mu$', fontsize=14)

    axins2 = inset_axes(ax1, width="25%", height="68%",
                   bbox_to_anchor=(.7, .1, .95, .7),
                   bbox_transform=ax1.transAxes, loc=3)
    n, bins, patches = axins2.hist(nres, bins=12, histtype='stepfilled', alpha=0.2, density=True)
    (mu, sigma) = st.norm.fit(nres)
    print(mu, sigma)
    xn = np.linspace(-3.0, 3.0, 100)
    yn = st.norm.pdf(xn, mu, sigma)
    axins2.plot(xn, yn, 'r', linewidth=2)
    axins2.tick_params(axis='both', which='major', labelsize=12)
    axins2.set_xlabel(r'$\Delta \mu/ \sigma_{\mu}$', fontsize=14)
    axins2.set_ylabel(r'$pdf$', fontsize=14)
    

    # plt.setp(ax1.get_xticklabels(), visible=False)
    
    # ax2 = plt.subplot(gs[1], sharex=ax1)
    # ax2.set_xlim(zmin, zmax)

    # res = vMuos[ix] - vMus[ix]
    # plt.errorbar(zs[ix], res, yerr=vMuoErrs[ix], fmt='o', markersize=3.0, lw=0.6, capsize=0.0, alpha=0.5)
    # plt.plot(zm, Mumz-Mumz, lw=2, c='red')
    # plt.plot(zm, MumL-Mumz, lw=2, c='black')
    # # plt.plot(zm, MumN-Mumz, lw=2, c='blue')
    # for i in range(len(zs[ix])):
    #     # plt.annotate(vTg[i], xy=(xp[i], yp[i]), fontsize=2)
    #     if Tgs[ix][i] == 6:
    #         plt.errorbar(zs[ix][i], res[i], yerr=vMuoErrs[ix][i], fmt='og', markersize=6.0, lw=0.6, capsize=0.0, alpha=0.8)
    #     if Tgs[ix][i] == 7:
    #         plt.errorbar(zs[ix][i], res[i], yerr=vMuoErrs[ix][i], fmt='or', markersize=6.0, lw=0.6, capsize=0.0, alpha=0.8)
    

    # plt.ylabel(r'$\Delta \mu$', fontsize=18)
    # plt.xlabel(r'$z$', fontsize=18)

    # plt.xticks(fontsize=16)
    # plt.yticks(fontsize=16)

    plt.tight_layout()
   
    plt.savefig(path)

    return


def lnprior0(cube):
    cube[0] = cube[0]*4. + 30.
    cube[1] = cube[1]*4. + 3.
    return cube


def lnlike0(cube):
    alpha, beta = cube[0], cube[1]
    theta = alpha, beta
    xsq = chsq0(theta)[0]
    return -0.5*xsq


def chsq0(theta):
    alpha, beta = theta
    x, y, z, xerr, yerr, zerr, Rxy, Muo, MuoErr, vTg = Mdata

    Mum = z
    MumErr = zerr

    Mu = 2.5*(beta*x + alpha) - 2.5*y - 100.195
    MuErr = 2.5*np.sqrt((yerr)**2 + beta**2*(xerr)**2
            # - 2.0*beta*Rxy*xerr*yerr
            # + 0.22**2
    )

    R = (Mu - Mum)
    W = 1.0/(MuErr**2 + MumErr**2)

    xsq = np.sum(R**2*W)
    xsq2 = xsq - np.sum(np.log(W))

    if fr == 2:
        return (xsq2, R, Mum, MuErr)
    else:
        return (xsq, R, Mum, MuErr)


def case0(cpath, dpath, fend, vbs, sps, prs, ctag):
    x, y, z, xerr, yerr, zerr, Rxy, Muo, MuoErr, vTg = Mdata
    parameters = [r"\alpha", r"\beta"]
    parametersc = [r"$\alpha$", r"$\beta$"]
    n_params = len(parameters)

    prefix = cpath + fend

    result = solve(LogLikelihood=lnlike0, Prior=lnprior0,
                   n_dims=n_params, outputfiles_basename=prefix, verbose=vbs,
                   n_live_points=sps, resume=prs, init_MPI=False,
                   sampling_efficiency=0.8)

    tag = 'HIIGx'
    ploter(dpath, cpath, fend, result, parameters, parametersc, tag, ctag)
    return


def lnprior100(cube):
    cube[0] = cube[0]*4. + 30.
    cube[1] = cube[1]*4. + 3.
    return cube


def lnlike100(cube):
    alpha, beta = cube[0], cube[1]
    theta = alpha, beta
    xsq = chsq100(theta)[0]
    return -0.5*xsq


def chsq100(theta):
    alpha, beta = theta
    x, y, z, xerr, yerr, zerr, Rxy, Muo, MuoErr, vTg = Mdata

    lgL = y + 0.4*z + 40.078
    lgLErr = np.sqrt(yerr**2 + 0.4**2*zerr**2)

    lgLm = beta*x + alpha
    lgLmErr = np.sqrt(beta**2*xerr**2)

    R = (lgL - lgLm)
    W = 1.0/(lgLErr**2 + lgLmErr**2 + 0.20**2)

    xsq = np.sum(R**2*W)
    xsq2 = xsq - np.sum(np.log(W))

    if fr == 2:
        return (xsq2, R, lgLm, lgLErr)
    else:
        return (xsq, R, lgLm, lgLErr)


def case100(cpath, dpath, fend, vbs, sps, prs, ctag):
    x, y, z, xerr, yerr, zerr, Rxy, Muo, MuoErr, vTg = Mdata
    parameters = [r"\alpha", r"\beta"]
    parametersc = [r"$\alpha$", r"$\beta$"]
    n_params = len(parameters)

    prefix = cpath + fend

    result = solve(LogLikelihood=lnlike100, Prior=lnprior100,
                   n_dims=n_params, outputfiles_basename=prefix, verbose=vbs,
                   n_live_points=sps, resume=prs, init_MPI=False,
                   sampling_efficiency=0.8)

    tag = 'HIIGx'
    ploter(dpath, cpath, fend, result, parameters, parametersc, tag, ctag)
    return


def lnprior1(cube):
    cube[0] = cube[0]*4. + 3.
    return cube


def lnlike1(cube):
    beta = cube[0]
    theta = beta
    xsq = chsq1(theta)[0]
    return -0.5*xsq


def chsq1(theta):
    beta = theta
    x, y, z, xerr, yerr, zerr, Rxy, Muo, MuoErr, vTg = Mdata

    apha = 0.0
    w0 = -1
    Om = 0.3
    cosmo = FlatwCDM(H0=100.0, Om0=Om, w0=w0)
    # -------------------------------------------------------------------------
    ixG = np.where(z > 10.0)
    ixH = np.where(z < 10.0)

    Mum = z*0.0
    MumErr = z*0.0

    Mum[ixG] = z[ixG]
    MumErr[ixG] = zerr[ixG]

    Mum[ixH] = 5.0*np.log10(cosmo.luminosity_distance(z[ixH]).value) + 25.0
    MumErr[ixH] = (5.0/np.log(10.0))*(zerr[ixH]/z[ixH])

    Mu = 2.5*(beta*x + alpha) - 2.5*y - 100.195
    MuErr = 2.5*np.sqrt((yerr)**2 + beta**2*(xerr)**2
            # - 2.0*beta*Rxy*xerr*yerr
            #+ 0.22**2
    )

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


def case1(cpath, dpath, fend, vbs, sps, prs, ctag):
    x, y, z, xerr, yerr, zerr, Rxy, Muo, MuoErr, vTg = Mdata
    parameters = [r"\beta"]
    parametersc = [r"$\beta$"]
    n_params = len(parameters)

    prefix = cpath + fend

    result = solve(LogLikelihood=lnlike1, Prior=lnprior1,
                   n_dims=n_params, outputfiles_basename=prefix, verbose=vbs,
                   n_live_points=sps, resume=prs, init_MPI=False,
                   sampling_efficiency=0.8)

    tag = 'HIIGx'
    ploter(dpath, cpath, fend, result, parameters, parametersc, tag, ctag)
    return


def lnprior3(cube):
    cube[0] = cube[0]*2. + 32.5
    cube[1] = cube[1]*1. + 4.5
    cube[2] = cube[2]
    return cube


def lnlike3(cube):
    alpha, beta, h0 = cube[0], cube[1], cube[2]
    theta = alpha, beta, h0
    xsq = chsq3(theta)[0]
    return -0.5*xsq


def chsq3(theta):
    alpha, beta, h0 = theta
    x, y, z, xerr, yerr, zerr, Rxy, Muo, MuoErr, vTg = Mdata
    x, y, z, xerr, yerr, zerr, zerrU, zerrD, Rxy, Muo, MuoErr, vTg = MdataA

    w0 = -1.0
    Om = 0.33
    cosmo = FlatwCDM(H0=h0*100.0, Om0=Om, w0=w0)
    # -------------------------------------------------------------------------
    ixG = np.where(z > 10.0)
    ixH = np.where(z < 10.0)

    Mum = z*0.0
    MumErr = z*0.0
    MumErr1 = z*0.0
    MumErr2 = z*0.0

    Mum[ixG] = z[ixG]
    MumErr[ixG] = zerr[ixG]

    Mum[ixH] = 5.0*np.log10(cosmo.luminosity_distance(z[ixH]).value) + 25.0
    MumErr[ixH] = (5.0/np.log(10.0))*(zerr[ixH]/z[ixH])

    Mu = 2.5*(beta*x + alpha) - 2.5*y - 100.195
    if fg0 == 0:
        MuErr = 2.5*np.sqrt((yerr)**2 + beta**2*(xerr)**2
            # - 2.0*beta*Rxy*xerr*yerr
            #+ 0.22**2
            )
    elif fg0 == 1:
        MuErr = 2.5*np.sqrt((yerr)**2 + beta**2*(xerr)**2
            # - 2.0*beta*Rxy*xerr*yerr
            + (0.053/2.5)**2
            # + 0.22**2
            )

    if fa == 1:
        MumErr1 = 2*zerrU*zerrD/(zerrU + zerrD)
        MumErr2 = (zerrU - zerrD)/(zerrU + zerrD)
        MumErr[ixG] = MumErr1[ixG] + MumErr2[ixG]*(Mu[ixG] - Mum[ixG])

    R = (Mu - Mum)
    W = 1.0/(MuErr**2 + MumErr**2)

    xsq = np.sum(R**2*W)
    xsq2 = xsq - np.sum(np.log(W))

    if fr == 2:
        return (xsq2, R, Mum, MumErr)
    else:
        return (xsq, R, Mum, MumErr)


def case3(cpath, dpath, fend, vbs, sps, prs, ctag):
    x, y, z, xerr, yerr, zerr, Rxy, Muo, MuoErr, vTg = Mdata
    parameters = [r"\alpha", r"\beta", r"h"]
    parametersc = [r"$\alpha$", r"$\beta$", r"$h$"]
    n_params = len(parameters)

    prefix = cpath + fend

    result = solve(LogLikelihood=lnlike3, Prior=lnprior3,
                   n_dims=n_params, outputfiles_basename=prefix, verbose=vbs,
                   n_live_points=sps, resume=prs, init_MPI=False,
                   sampling_efficiency=0.8)

    tag = 'HIIGx'
    ploter(dpath, cpath, fend, prs, result, parameters, parametersc, tag, ctag)
    return


def lnprior30(cube):
    cube[0] = cube[0]
    return cube


def lnlike30(cube):
    h0 = cube[0]
    theta = h0
    xsq = chsq30(theta)[0]
    return -0.5*xsq


def chsq30(theta):
    h0 = np.asscalar(theta)
    x, y, z, xerr, yerr, zerr, Rxy, Muo, MuoErr, vTg = Mdata

    w0 = -1
    Om = 0.3
    q0 = -0.55
    cosmo = FlatwCDM(H0=h0*100.0, Om0=Om, w0=w0)
    # -------------------------------------------------------------------------
    ixG = np.where(z > 10.0)
    ixH = np.where(z < 10.0)

    Mum = z*0.0
    MumErr = z*0.0

    Mum[ixG] = z[ixG]
    MumErr[ixG] = zerr[ixG]

    Mum[ixH] = 5.0*np.log10(cosmo.luminosity_distance(z[ixH]).value) + 25.0
    # Mum[ixH] = 5.0*np.log10((c*z[ixH])/(h0*100.)) + 25.0    
    # Mum[ixH] = 5.0*np.log10((c*(z[ixH] + 0.5*(1 - q0)*z[ixH]**2))/(h0*100.)) + 25.0  
    MumErr[ixH] = (5.0/np.log(10.0))*(zerr[ixH]/z[ixH])

    Mu = 2.5*(beta*x + alpha) - 2.5*y - 100.195
    if fg0 == 0:
        MuErr = 2.5*np.sqrt((yerr)**2 + beta**2*(xerr)**2
            + x**2*(betaErr)**2 + alphaErr**2
            # - 2.0*beta*Rxy*xerr*yerr
            #+ 0.22**2
            )
    elif fg0 == 1:
        MuErr = 2.5*np.sqrt((yerr)**2 + beta**2*(xerr)**2
            + x**2*(betaErr)**2 + alphaErr**2
            # - 2.0*beta*Rxy*xerr*yerr
            + (0.053/2.5)**2
            #+ 0.22**2
            )

    R = (Mu - Mum)
    W = 1.0/(MuErr**2 + MumErr**2)

    xsq = np.sum(R**2*W)
    xsq2 = xsq - np.sum(np.log(W))

    if fr == 2:
        return (xsq2, R, Mum, MuErr)
    else:
        return (xsq, R, Mum, MuErr)


def case30(cpath, dpath, fend, vbs, sps, prs, ctag):
    x, y, z, xerr, yerr, zerr, Rxy, Muo, MuoErr, vTg = Mdata
    parameters = [r"h"]
    parametersc = [r"$h$"]
    n_params = len(parameters)

    prefix = cpath + fend

    result = solve(LogLikelihood=lnlike30, Prior=lnprior30,
                   n_dims=n_params, outputfiles_basename=prefix, verbose=vbs,
                   n_live_points=sps, resume=prs, init_MPI=False,
                   sampling_efficiency=0.8)

    tag = 'HIIGx'
    ploter(dpath, cpath, fend, result, parameters, parametersc, tag, ctag)
    return


def lnprior31(cube):
    cube[0] = cube[0]*2. + 32.5
    cube[1] = cube[1]*1. + 4.5
    cube[2] = cube[2]
    cube[3] = cube[3]*1. - 1.0
    return cube


def lnlike31(cube):
    alpha, beta, h0, q0 = cube[0], cube[1], cube[2], cube[3]
    theta = alpha, beta, h0, q0
    xsq = chsq31(theta)[0]
    return -0.5*xsq


def chsq31(theta):
    alpha, beta, h0, q0 = theta
    x, y, z, xerr, yerr, zerr, Rxy, Muo, MuoErr, vTg = Mdata

    j0 = 1.0
    #---------------------------------------------------------------------------
    ixG = np.where(z>10)
    ixH = np.where(z<10)

    Mum = z*0.0
    MumErr = z*0.0

    Mum[ixG] = z[ixG]
    MumErr[ixG] = zerr[ixG]

    # l = z[ixH]/(1.0 + z[ixH])
    # Mum[ixH] = 5.0*np.log10((c/(h0*100.0))*(l + (1.0/2.0)*(3.0 - q0)*l**2)) + 25.0

    Mum[ixH] = 5.0*np.log10(((c*z[ixH])/(h0*100.0))*(1.0 + (1.0/2.0)*(1.0 - q0)*z[ixH] -(1.0/6.0)*(1.0 - q0 - 3.0*q0**2 + j0)*z[ixH]**2)) + 25.0
    # MumErr[ixH] = 2.0*(5.0/np.log(10.0))*(zerr[ixH]/z[ixH])*(((q0 - 1.0)*z[ixH] - 1.0)/((q0 - 1.0)*z[ixH] - 2.0))
    MumErr[ixH] = (5.0/np.log(10.0))*(zerr[ixH]/z[ixH])

    Mu = 2.5*(beta*x + alpha) - 2.5*y - 100.195
    if fg0 == 0:
        MuErr = 2.5*np.sqrt((yerr)**2 + beta**2*(xerr)**2
            # + x**2*(betaErr)**2 + alphaErr**2
            # - 2.0*beta*Rxy*xerr*yerr
            #+ 0.22**2
            )
    elif fg0 == 1:
        MuErr = 2.5*np.sqrt((yerr)**2 + beta**2*(xerr)**2
            # + x**2*(betaErr)**2 + alphaErr**2
            # - 2.0*beta*Rxy*xerr*yerr
            + (0.053/2.5)**2
            #+ 0.22**2
            )

    R = (Mu - Mum)
    W = 1.0/(MuErr**2 + MumErr**2)

    xsq = np.sum(R**2*W)
    xsq2 = xsq - np.sum(np.log(W))

    if fr == 2:
        return (xsq2, R, Mum, MuErr)
    else:
        return (xsq, R, Mum, MuErr)


def case31(cpath, dpath, fend, vbs, sps, prs, ctag):
    x, y, z, xerr, yerr, zerr, Rxy, Muo, MuoErr, vTg = Mdata
    parameters = [r"\alpha", r"\beta", r"h", r"q_0"]
    parametersc = [r"$\alpha$", r"$\beta$", r"$h$", r"$q_0$"]
    n_params = len(parameters)

    prefix = cpath + fend

    result = solve(LogLikelihood=lnlike31, Prior=lnprior31,
                   n_dims=n_params, outputfiles_basename=prefix, verbose=vbs,
                   n_live_points=sps, resume=prs, init_MPI=False,
                   sampling_efficiency=0.8)

    tag = 'HIIGx'
    
    ploter(dpath, cpath, fend, prs, result, parameters, parametersc, tag, ctag)
    return


def lnprior4(cube):
    cube[0] = cube[0]*2. + 32.5
    cube[1] = cube[1]*1. + 4.5
    cube[2] = cube[2]*0.4 + 0.5
    cube[3] = cube[3]
    return cube


def lnlike4(cube):
    alpha, beta, h0, Om = cube[0], cube[1], cube[2], cube[3]
    theta = alpha, beta, h0, Om
    xsq = chsq4(theta)[0]
    return -0.5*xsq


def chsq4(theta):
    alpha, beta, h0, Om = theta
    x, y, z, xerr, yerr, zerr, Rxy, Muo, MuoErr, vTg = Mdata

    w0 = -1
    cosmo = FlatwCDM(H0=h0*100.0, Om0=Om, w0=w0)
    # -------------------------------------------------------------------------
    ixG = np.where(z > 10.0)
    ixH = np.where(z < 10.0)

    Mum = z*0.0
    MumErr = z*0.0

    Mum[ixG] = z[ixG]
    MumErr[ixG] = zerr[ixG]

    Mum[ixH] = 5.0*np.log10(cosmo.luminosity_distance(z[ixH]).value) + 25.0
    MumErr[ixH] = (5.0/np.log(10.0))*(zerr[ixH]/z[ixH])

    Mu = 2.5*(beta*x + alpha) - 2.5*y - 100.195
    if fg0 == 0:
        MuErr = 2.5*np.sqrt((yerr)**2 + beta**2*(xerr)**2
            # - 2.0*beta*Rxy*xerr*yerr
            #+ 0.22**2
            )
    elif fg0 == 1:
        MuErr = 2.5*np.sqrt((yerr)**2 + beta**2*(xerr)**2
            # - 2.0*beta*Rxy*xerr*yerr
            + (0.053/2.5)**2
            #+ 0.22**2
            )

    R = (Mu - Mum)
    W = 1.0/(MuErr**2 + MumErr**2)

    xsq = np.sum(R**2*W)
    xsq2 = xsq - np.sum(np.log(W))

    if fr == 2:
        return (xsq2, R, Mum, MuErr)
    else:
        return (xsq, R, Mum, MuErr)


def case4(cpath, dpath, fend, vbs, sps, prs, ctag):
    x, y, z, xerr, yerr, zerr, Rxy, Muo, MuoErr, vTg = Mdata
    parameters = [r"\alpha", r"\beta", r"h", r"\Omega_m"]
    parametersc = [r"$\alpha$", r"$\beta$", r"h", r"$\Omega_m$"]
    n_params = len(parameters)

    prefix = cpath + fend

    result = solve(LogLikelihood=lnlike4, Prior=lnprior4,
                   n_dims=n_params, outputfiles_basename=prefix, verbose=vbs,
                   n_live_points=sps, resume=prs, init_MPI=False,
                   sampling_efficiency=0.8)

    tag = 'HIIGx'
    ploter(dpath, cpath, fend, prs, result, parameters, parametersc, tag, ctag)
    return


def lnprior41(cube):
    cube[0] = cube[0]
    return cube


def lnlike41(cube):
    Om = cube[0]
    theta = Om
    xsq = chsq41(theta)[0]
    return -0.5*xsq


def chsq41(theta):
    Om = theta
    x, y, z, xerr, yerr, zerr, Rxy, Muo, MuoErr, vTg = Mdata

    alpha = 0.0
    alphaErr = 0.0

    w0 = -1
    cosmo = FlatwCDM(H0=100.0, Om0=Om, w0=w0)
    # -------------------------------------------------------------------------
    ixG = np.where(z > 10.0)
    ixH = np.where(z < 10.0)

    Mum = z*0.0
    MumErr = z*0.0

    Mum[ixG] = z[ixG]
    MumErr[ixG] = zerr[ixG]

    Mum[ixH] = 5.*np.log10((cosmo.luminosity_distance(z[ixH]).value*100.)/c) + 25.
    MumErr[ixH] = (5.0/np.log(10.0))*(zerr[ixH]/z[ixH])

    Mu = 2.5*(beta*x + alpha) - 2.5*y - 100.195
    if fg0 == 0:
        MuErr = 2.5*np.sqrt((yerr)**2 + beta**2*(xerr)**2
                 + betaErr**2*x**2
                 + alphaErr**2
                # - 2.0*beta*Rxy*xerr*yerr
                #+ 0.22**2
            )
    elif fg0 == 1:
        MuErr = 2.5*np.sqrt((yerr)**2 + beta**2*(xerr)**2
                + betaErr**2*x**2
                + alphaErr**2
                # - 2.0*beta*Rxy*xerr*yerr
                + (0.053/2.5)**2
                #+ 0.22**2
            )

    R = (Mu - Mum)
    W = 1.0/(MuErr**2 + MumErr**2)

    xsqA = np.sum(R**2*W)
    xsqB = np.sum(R*W)
    xsqC = np.sum(W)

    xsq = xsqA - xsqB**2/xsqC
    xsq2 = xsq - np.sum(np.log(W))

    if fr == 2:
        return (xsq2, R, Mum, MumErr)
    else:
        return (xsq, R, Mum, MumErr)


def case41(cpath, dpath, fend, vbs, sps, prs, ctag):
    x, y, z, xerr, yerr, zerr, Rxy, Muo, MuoErr, vTg = Mdata
    parameters = [r"\Omega_m"]
    parametersc = [r"$\Omega_m$"]
    n_params = len(parameters)

    prefix = cpath + fend

    result = solve(LogLikelihood=lnlike41, Prior=lnprior41,
                   n_dims=n_params, outputfiles_basename=prefix, verbose=vbs,
                   n_live_points=sps, resume=prs, init_MPI=False,
                   sampling_efficiency=0.8)

    tag = 'HIIGx'
    ploter(dpath, cpath, fend, prs, result, parameters, parametersc, tag, ctag)
    return


def lnprior42(cube):
    cube[0] = cube[0]*0.5 + 0.5
    cube[1] = cube[1]
    return cube


def lnlike42(cube):
    h0, Om = cube[0], cube[1]
    theta = h0, Om
    xsq = chsq42(theta)[0]
    return -0.5*xsq


def chsq42(theta):
    h0, Om = theta
    x, y, z, xerr, yerr, zerr, Rxy, Muo, MuoErr, vTg = Mdata

    w0 = -1
    cosmo = FlatwCDM(H0=h0*100.0, Om0=Om, w0=w0)
    # -------------------------------------------------------------------------
    ixG = np.where(z > 10.0)
    ixH = np.where(z < 10.0)

    Mum = z*0.0
    MumErr = z*0.0

    Mum[ixG] = z[ixG]
    MumErr[ixG] = zerr[ixG]

    Mum[ixH] = 5.0*np.log10(cosmo.luminosity_distance(z[ixH]).value) + 25.0
    MumErr[ixH] = (5.0/np.log(10.0))*(zerr[ixH]/z[ixH])

    Mu = Muo
    MuErr = MuoErr

    R = (Mu - Mum)
    W = 1.0/(MuErr**2 + MumErr**2)

    xsq = np.sum(R**2*W)
    xsq2 = xsq - np.sum(np.log(W))

    if fr == 2:
        return (xsq2, R, Mum, MuErr)
    else:
        return (xsq, R, Mum, MuErr)


def case42(cpath, dpath, fend, vbs, sps, prs, ctag):
    x, y, z, xerr, yerr, zerr, Rxy, Muo, MuoErr, vTg = Mdata
    parameters = [r"h", r"\Omega_m"]
    parametersc = [r"h", r"$\Omega_m$"]
    n_params = len(parameters)

    prefix = cpath + fend

    result = solve(LogLikelihood=lnlike42, Prior=lnprior42,
                   n_dims=n_params, outputfiles_basename=prefix, verbose=vbs,
                   n_live_points=sps, resume=prs, init_MPI=False,
                   sampling_efficiency=0.8)

    tag = 'HIIGx'
    ploter(dpath, cpath, fend, prs, result, parameters, parametersc, tag, ctag)
    return


def lnprior5(cube):
    cube[0] = cube[0]*2. + 32.5
    cube[1] = cube[1]*1. + 4.5
    cube[2] = cube[2]*0.4 + 0.5
    cube[3] = cube[3]
    cube[4] = cube[4]*2. - 2.
    return cube


def lnlike5(cube):
    alpha, beta, h0, Om, w0 = cube[0], cube[1], cube[2], cube[3], cube[4]
    theta = alpha, beta, h0, Om, w0
    xsq = chsq5(theta)[0]
    return -0.5*xsq


def chsq5(theta):
    alpha, beta, h0, Om, w0 = theta
    x, y, z, xerr, yerr, zerr, Rxy, Muo, MuoErr, vTg = Mdata

    cosmo = FlatwCDM(H0=h0*100.0, Om0=Om, w0=w0, Tcmb0= 2.725)
    #---------------------------------------------------------------------------
    ixG = np.where(z>10.0)
    ixH = np.where(z<10.0)

    Mum = z*0.0
    MumErr = z*0.0
    # MumErru = z*0.0
    # MumErrd = z*0.0

    Mum[ixG] = z[ixG]
    MumErr[ixG] = zerr[ixG]

    Mum[ixH] = cosmo.distmod(z[ixH]).value
    # MumErr[ixH] = cosmo.distmod(z[ixH] + zerr[ixH]).value - Mum[ixH]
    # MumErrd[ixH] = Mum[ixH] - cosmo.distmod(z[ixH] - zerr[ixH]).value
    MumErr[ixH] = (5.0/np.log(10.0))*(zerr[ixH]/z[ixH])

    # print(MumErru - MumErr, MumErr - MumErrd, MumErru - MumErrd)

    Mu = 2.5*(beta*x + alpha) - 2.5*y - 100.195
    if fg0 == 0:
        MuErr = 2.5*np.sqrt((yerr)**2 + beta**2*(xerr)**2
            # - 2.0*beta*Rxy*xerr*yerr
            #+ 0.22**2
            )
    elif fg0 == 1:
        MuErr = 2.5*np.sqrt((yerr)**2 + beta**2*(xerr)**2
            # - 2.0*beta*Rxy*xerr*yerr
            + (0.053/2.5)**2
            #+ 0.22**2
            )

    R = (Mu - Mum)
    W = 1.0/(MuErr**2 + MumErr**2)

    xsq = np.sum(R**2*W)
    xsq2 = xsq - np.sum(np.log(W))

    if fr == 2:
        return (xsq2, R, Mum, MumErr)
    else:
        return (xsq, R, Mum, MumErr)


def case5(cpath, dpath, fend, vbs, sps, prs, ctag):
    x, y, z, xerr, yerr, zerr, Rxy, Muo, MuoErr, vTg = Mdata

    parameters = [r"\alpha",r"\beta", r"h", r"\Omega_m", r"w_0"]
    parametersc = [r"$\alpha$",r"$\beta$", r"$h$", r"$\Omega_m$", r"$w_0$"]
    n_params = len(parameters)

    prefix = cpath + fend

    result = solve(LogLikelihood=lnlike5, Prior=lnprior5,
                   n_dims=n_params, outputfiles_basename=prefix, verbose=vbs,
                   n_live_points=sps, resume=prs, init_MPI=False,
                   sampling_efficiency=0.8)

    tag = 'HIIGx'
    ploter(dpath, cpath, fend, prs, result, parameters, parametersc, tag, ctag)
    return


def lnprior51(cube):
    cube[0] = cube[0]
    cube[1] = cube[1]*2. - 2.
    return cube


def lnlike51(cube):
    Om, w0 = cube[0], cube[1]
    theta = Om, w0
    xsq = chsq51(theta)[0]
    return -0.5*xsq


def chsq51(theta):
    Om, w0 = theta
    x, y, z, xerr, yerr, zerr, Rxy, Muo, MuoErr, vTg = Mdata

    cosmo = FlatwCDM(H0=100.0, Om0=Om, w0=w0)
    # -------------------------------------------------------------------------
    ixG = np.where(z > 10.0)
    ixH = np.where(z < 10.0)

    Mum = z*0.0
    MumErr = z*0.0

    Mum[ixG] = z[ixG]
    MumErr[ixG] = zerr[ixG]

    Mum[ixH] = 5.*np.log10((cosmo.luminosity_distance(z[ixH]).value*100.)/c) + 25.
    MumErr[ixH] = (5.0/np.log(10.0))*(zerr[ixH]/z[ixH])

    Mu = 2.5*(beta*x + alpha) - 2.5*y - 100.195
    if fg0 == 0:
        MuErr = 2.5*np.sqrt((yerr)**2 + beta**2*(xerr)**2
                 + betaErr**2*x**2
                 + alphaErr**2
                # - 2.0*beta*Rxy*xerr*yerr
                #+ 0.22**2
            )
    elif fg0 == 1:
        MuErr = 2.5*np.sqrt((yerr)**2 + beta**2*(xerr)**2
                + betaErr**2*x**2
                + alphaErr**2
                # - 2.0*beta*Rxy*xerr*yerr
                + (0.053/2.5)**2
                #+ 0.22**2
            )

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


def case51(cpath, dpath, fend, vbs, sps, prs, ctag):
    x, y, z, xerr, yerr, zerr, Rxy, Muo, MuoErr, vTg = Mdata
    parameters = [r"\Omega_m", r"w_0"]
    parametersc = [r"$\Omega_m$", r"$w_0$"]
    n_params = len(parameters)

    prefix = cpath + fend

    result = solve(LogLikelihood=lnlike51, Prior=lnprior51,
                   n_dims=n_params, outputfiles_basename=prefix, verbose=vbs,
                   n_live_points=sps, resume=prs, init_MPI=False,
                   sampling_efficiency=0.8)

    tag = 'HIIGx'
    ploter(dpath, cpath, fend, prs, result, parameters, parametersc, tag, ctag)
    return


def lnprior52(cube):
    cube[0] = cube[0]*4. + 3.
    cube[1] = cube[1]
    cube[2] = cube[2]*2. - 2.
    return cube


def lnlike52(cube):
    beta, Om, w0 = cube[0], cube[1], cube[2]
    theta = beta, Om, w0
    xsq = chsq52(theta)[0]
    return -0.5*xsq


def chsq52(theta):
    beta, Om, w0 = theta
    x, y, z, xerr, yerr, zerr, Rxy, Muo, MuoErr, vTg = Mdata

    alpha = 0.0
    h = 100.0
    cosmo = FlatwCDM(H0=h, Om0=Om, w0=w0)
    # ---------------------------------------------------------------------------
    # ixG = np.where(z > 10.0)
    # ixH = np.where(z < 10.0)

    # Mum = z*0.0
    # MumErr = z*0.0

    # vZ = 73.2*10**(0.2*(z[ixG] - 25.))/c

    # Mum[ixG] = 5.*np.log10((cosmo.luminosity_distance(vZ).value*h)/c) + 25.
    # MumErr[ixG] = zerr[ixG]

    # Mum[ixH] = 5.*np.log10((cosmo.luminosity_distance(z[ixH]).value*h)/c) + 25.
    # MumErr[ixH] = (5.0/np.log(10.0))*(zerr[ixH]/z[ixH])

    Mum = 5.*np.log10((cosmo.luminosity_distance(z).value*h)/c) + 25.
    MumErr = (5.0/np.log(10.0))*(zerr/z)

    Mu = 2.5*(beta*x + alpha) - 2.5*y - 100.195
    MuErr = 2.5*np.sqrt((yerr)**2 + beta**2*(xerr)**2
            # - 2.0*beta*Rxy*xerr*yerr
            #+ 0.22**2
    )

    R = (Mu - Mum)
    W = 1.0/(MuErr**2 + MumErr**2)

    xsqA = np.sum(R**2*W)
    xsqB = np.sum(R*W)
    xsqC = np.sum(W)

    xsq = xsqA - xsqB**2/xsqC
    xsq2 = xsq - np.sum(np.log(W))

    if fr == 2:
        return (xsq2, R, Mum, MumErr)
    else:
        return (xsq, R, Mum, MumErr)


def case52(cpath, dpath, fend, vbs, sps, prs, ctag):
    x, y, z, xerr, yerr, zerr, Rxy, Muo, MuoErr, vTg = Mdata
    parameters = [r"\beta", r"\Omega_m", r"w_0"]
    parametersc = [r"$\beta$", r"$\Omega_m$", r"$w_0$"]
    n_params = len(parameters)

    prefix = cpath + fend

    result = solve(LogLikelihood=lnlike52, Prior=lnprior52,
                   n_dims=n_params, outputfiles_basename=prefix, verbose=vbs,
                   n_live_points=sps, resume=prs, init_MPI=False,
                   sampling_efficiency=0.8)

    tag = 'HIIGx'
    ploter(dpath, cpath, fend, result, parameters, parametersc, tag, ctag)
    return


def lnprior53(cube):
    cube[0] = cube[0]
    cube[1] = cube[1]*2. - 2.
    return cube


def lnlike53(cube):
    Om, w0 = cube[0], cube[1]
    theta = Om, w0
    xsq = chsq53(theta)[0]
    return -0.5*xsq


def chsq53(theta):
    Om, w0 = theta
    x, y, z, xerr, yerr, zerr, Rxy, Muo, MuoErr, vTg = Mdata

    alpha = 0.0
    beta = 5.0
    cosmo = FlatwCDM(H0=100.0, Om0=Om, w0=w0)
    # ---------------------------------------------------------------------------
    ixG = np.where(z > 10.0)
    ixH = np.where(z < 10.0)

    Mum = z*0.0
    MumErr = z*0.0

    Mum[ixG] = z[ixG]
    MumErr[ixG] = zerr[ixG]

    Mum[ixH] = 5.*np.log10((cosmo.luminosity_distance(z[ixH]).value*100.)/c) + 25.
    MumErr[ixH] = (5.0/np.log(10.0))*(zerr[ixH]/z[ixH])

    Mu = 2.5*(beta*x + alpha) - 2.5*y - 100.195
    MuErr = 2.5*np.sqrt((yerr)**2 + beta**2*(xerr)**2
            # - 2.0*beta*Rxy*xerr*yerr
            #+ 0.22**2
    )

    R = (Mu - Mum)
    W = 1.0/(MuErr**2 + MumErr**2)

    xsqA = np.sum(R**2*W)
    xsqB = np.sum(R*W)
    xsqC = np.sum(W)

    xsq = xsqA - xsqB**2/xsqC
    xsq2 = xsq - np.sum(np.log(W))

    if fr == 2:
        return (xsq2, R, Mum, MumErr)
    else:
        return (xsq, R, Mum, MumErr)


def case53(cpath, dpath, fend, vbs, sps, prs, ctag):
    x, y, z, xerr, yerr, zerr, Rxy, Muo, MuoErr, vTg = Mdata
    parameters = [r"\Omega_m", r"w_0"]
    parametersc = [r"$\Omega_m$", r"$w_0$"]
    n_params = len(parameters)

    prefix = cpath + fend

    result = solve(LogLikelihood=lnlike53, Prior=lnprior53,
                   n_dims=n_params, outputfiles_basename=prefix, verbose=vbs,
                   n_live_points=sps, resume=prs, init_MPI=False,
                   sampling_efficiency=0.8)

    tag = 'HIIGx'
    ploter(dpath, fend, result, parameters, parametersc, tag, ctag)
    return


def lnprior54(cube):
    cube[0] = cube[0]*0.5 + 0.5
    cube[1] = cube[1]
    cube[2] = cube[2]*2.0 - 2.0
    return cube


def lnlike54(cube):
    h0, Om, w0 = cube[0], cube[1], cube[2]
    theta = h0, Om, w0
    xsq = chsq54(theta)[0]
    return -0.5*xsq


def chsq54(theta):
    h0, Om, w0 = theta
    x, y, z, xerr, yerr, zerr, Rxy, Muo, MuoErr, vTg = Mdata

    cosmo = FlatwCDM(H0=h0*100.0, Om0=Om, w0=w0)
    # -------------------------------------------------------------------------
    ixG = np.where(z > 10.0)
    ixH = np.where(z < 10.0)

    Mum = z*0.0
    MumErr = z*0.0

    Mum[ixG] = z[ixG]
    MumErr[ixG] = zerr[ixG]

    Mum[ixH] = 5.0*np.log10(cosmo.luminosity_distance(z[ixH]).value) + 25.0
    MumErr[ixH] = (5.0/np.log(10.0))*(zerr[ixH]/z[ixH])

    Mu = Muo
    MuErr = MuoErr

    R = (Mu - Mum)
    W = 1.0/(MuErr**2 + MumErr**2)

    xsq = np.sum(R**2*W)
    xsq2 = xsq - np.sum(np.log(W))

    if fr == 2:
        return (xsq2, R, Mum, MuErr)
    else:
        return (xsq, R, Mum, MuErr)


def case54(cpath, dpath, fend, vbs, sps, prs, ctag):
    x, y, z, xerr, yerr, zerr, Rxy, Muo, MuoErr, vTg = Mdata
    parameters = [r"h", r"\Omega_m", r"w_0"]
    parametersc = [r"h", r"$\Omega_m$", r"$w_0$"]
    n_params = len(parameters)

    prefix = cpath + fend

    result = solve(LogLikelihood=lnlike54, Prior=lnprior54,
                   n_dims=n_params, outputfiles_basename=prefix, verbose=vbs,
                   n_live_points=sps, resume=prs, init_MPI=False,
                   sampling_efficiency=0.8)

    tag = 'HIIG'
    ploter(dpath, cpath, fend, prs, result, parameters, parametersc, tag, ctag)
    return


def lnprior6(cube):
    cube[0] = cube[0]*10 + 25
    cube[1] = cube[1]*8
    cube[2] = cube[2] + 0.3
    cube[3] = cube[3]
    cube[4] = cube[4]*2 - 2
    cube[5] = cube[5]*3 - 2
    return cube


def lnlike6(cube):
    alpha, beta, h0, Om, w0, w1 = cube[0], cube[1], cube[2], cube[3], cube[4], cube[5]
    theta = alpha, beta, h0, Om, w0, w1
    xsq = chsq6(theta)[0]
    return -0.5*xsq


def chsq6(theta):
    alpha, beta, h0, Om, w0, w1 = theta
    x, y, z, xerr, yerr, zerr, Rxy, Muo, MuoErr, vTg = Mdata

    cosmo = Flatw0waCDM(H0=h0*100.0, Om0=Om, w0=w0, wa=w1)
    #---------------------------------------------------------------------------
    ixG = np.where(z>10.0)
    ixH = np.where(z<10.0)

    Mum = z*0.0
    MumErr = z*0.0

    Mum[ixG] = z[ixG]
    MumErr[ixG] = zerr[ixG]

    Mum[ixH] = 5.0*np.log10(cosmo.luminosity_distance(z[ixH]).value) + 25.0
    MumErr[ixH] = (5.0/np.log(10.0))*(zerr[ixH]/z[ixH])

    Mu = 2.5*(beta*x + alpha) - 2.5*y - 100.195
    if fg0 == 0:
        MuErr = 2.5*np.sqrt((yerr)**2 + beta**2*(xerr)**2
            # - 2.0*beta*Rxy*xerr*yerr
            #+ 0.22**2
        )
    elif fg0 == 1:
        MuErr = 2.5*np.sqrt((yerr)**2 + beta**2*(xerr)**2
            # - 2.0*beta*Rxy*xerr*yerr
            + (0.053/2.5)**2
            #+ 0.22**2
        )
        
    R = (Mu - Mum)
    W = 1.0/(MuErr**2 + MumErr**2)

    xsq = np.sum(R**2*W)
    xsq2 = xsq - np.sum(np.log(W))

    if fr == 2:
        return (xsq2, R, Mum, MumErr)
    else:
        return (xsq, R, Mum, MumErr)


def case6(cpath, dpath, fend, vbs, sps, prs, ctag):
    x, y, z, xerr, yerr, zerr, Rxy, Muo, MuoErr, vTg = Mdata
    parameters = [r"\alpha",r"\beta", r"h", r"\Omega_m", r"w_0", r"w_a"]
    parametersc = [r"$\alpha$",r"$\beta$", r"$h$", r"$\Omega_m$", r"$w_0$", r"$w_a$"]
    n_params = len(parameters)

    prefix = cpath + fend

    result = solve(LogLikelihood=lnlike6, Prior=lnprior6,
                   n_dims=n_params, outputfiles_basename=prefix, verbose=vbs,
                   n_live_points=sps, resume=prs, init_MPI=False,
                   sampling_efficiency=0.8)

    tag = 'HIIGx'
    ploter(dpath, cpath, fend, prs, result, parameters, parametersc, tag, ctag)
    return


def lnprior61(cube):
    cube[0] = cube[0]
    cube[1] = cube[1]*2 - 2
    cube[2] = cube[2]*3 - 2
    return cube


def lnlike61(cube):
    Om, w0, w1 = cube[0], cube[1], cube[2]
    theta = Om, w0, w1
    xsq = chsq61(theta)[0]
    return -0.5*xsq


def chsq61(theta):
    Om, w0, w1 = theta
    x, y, z, xerr, yerr, zerr, Rxy, Muo, MuoErr, vTg = Mdata

    cosmo = Flatw0waCDM(H0=100.0, Om0=Om, w0=w0, wa=w1)
    # -------------------------------------------------------------------------
    ixG = np.where(z > 10.0)
    ixH = np.where(z < 10.0)

    Mum = z*0.0
    MumErr = z*0.0

    Mum[ixG] = z[ixG]
    MumErr[ixG] = zerr[ixG]

    Mum[ixH] = 5.*np.log10((cosmo.luminosity_distance(z[ixH]).value*100.)/c) + 25.
    MumErr[ixH] = (5.0/np.log(10.0))*(zerr[ixH]/z[ixH])

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


def case61(cpath, dpath, fend, vbs, sps, prs, ctag):
    x, y, z, xerr, yerr, zerr, Rxy, Muo, MuoErr, vTg = Mdata
    parameters = [r"\Omega_m", r"w_0", r"w_a"]
    n_params = len(parameters)

    prefix = cpath + fend

    result = solve(LogLikelihood=lnlike61, Prior=lnprior61,
                   n_dims=n_params, outputfiles_basename=prefix, verbose=vbs,
                   n_live_points=sps, resume=prs, init_MPI=False,
                   sampling_efficiency=0.8)

    tag = 'HIIGx'
    ploter(dpath, fend, result, parameters, tag, ctag)
    return


def lnprior7(cube):
    cube[0] = cube[0]*0.5 + 0.5
    cube[1] = cube[1]
    cube[2] = cube[2]
    return cube


def lnlike7(cube):
    h0, Om, Ol = cube[0], cube[1], cube[2]
    theta = h0, Om, Ol
    xsq = chsq7(theta)[0]
    return -0.5*xsq


def chsq7(theta):
    h0, Om, Ol = theta
    x, y, z, xerr, yerr, zerr, Rxy, Muo, MuoErr, vTg = Mdata

    cosmo = LambdaCDM(H0=h0*100.0, Om0=Om, Ode0=Ol)
    #---------------------------------------------------------------------------
    ixG = np.where(z>10.0)
    ixH = np.where(z<10.0)

    Mum = z*0.0
    MumErr = z*0.0

    Mum[ixG] = z[ixG]
    MumErr[ixG] = zerr[ixG]

    Mum[ixH] = 5.0*np.log10(cosmo.luminosity_distance(z[ixH]).value) + 25.0
    MumErr[ixH] = (5.0/np.log(10.0))*(zerr[ixH]/z[ixH])

    Mu = 2.5*(beta*x + alpha) - 2.5*y - 100.195
    MuErr = 2.5*np.sqrt((yerr)**2 + beta**2*(xerr)**2
            # - 2.0*beta*Rxy*xerr*yerr
            #+ 0.22**2
    )

    R = (Mu - Mum)
    W = 1.0/(MuErr**2 + MumErr**2)

    xsq = np.sum(R**2*W)
    xsq2 = xsq - np.sum(np.log(W))

    if fr == 2:
        return (xsq2, R, Mum, MumErr)
    else:
        return (xsq, R, Mum, MumErr)


def case7(cpath, dpath, fend, vbs, sps, prs, ctag):
    x, y, z, xerr, yerr, zerr, Rxy, Muo, MuoErr, vTg = Mdata
    parameters = [r"h", r"\Omega_m", r"\Omega_{\Lambda}"]
    parametersc = [r"$h$", r"$\Omega_m$", r"$\Omega_{\Lambda}$"]
    n_params = len(parameters)

    prefix = cpath + fend

    result = solve(LogLikelihood=lnlike7, Prior=lnprior7,
                   n_dims=n_params, outputfiles_basename=prefix, verbose=vbs,
                   n_live_points=sps, resume=prs, init_MPI=False,
                   sampling_efficiency=0.8)

    tag = 'HIIGx'
    ploter(dpath, fend, result, parameters, parametersc, tag, ctag)
    return


def lnprior71(cube):
    cube[0] = cube[0]
    cube[1] = cube[1]
    return cube


def lnlike71(cube):
    Om, Ol = cube[0], cube[1]
    theta = Om, Ol
    xsq = chsq71(theta)[0]
    return -0.5*xsq


def chsq71(theta):
    Om, Ol = theta
    x, y, z, xerr, yerr, zerr, Rxy, Muo, MuoErr, vTg = Mdata

    cosmo = LambdaCDM(H0=100.0, Om0=Om, Ode0=Ol)
    #---------------------------------------------------------------------------
    ixG = np.where(z > 10.0)
    ixH = np.where(z < 10.0)

    Mum = z*0.0
    MumErr = z*0.0

    Mum[ixG] = z[ixG]
    MumErr[ixG] = zerr[ixG]

    Mum[ixH] = 5.*np.log10((cosmo.luminosity_distance(z[ixH]).value*100.)/c) + 25.
    MumErr[ixH] = (5.0/np.log(10.0))*(zerr[ixH]/z[ixH])

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


def case71(cpath, dpath, fend, vbs, sps, prs, ctag):
    x, y, z, xerr, yerr, zerr, Rxy, Muo, MuoErr, vTg = Mdata
    parameters = [r"\Omega_m", r"\Omega_{\Lambda}"]
    parametersc = [r"$\Omega_m$", r"$\Omega_{\Lambda}$"]
    n_params = len(parameters)

    prefix = cpath + fend

    result = solve(LogLikelihood=lnlike71, Prior=lnprior71,
                   n_dims=n_params, outputfiles_basename=prefix, verbose=vbs,
                   n_live_points=sps, resume=prs, init_MPI=False,
                   sampling_efficiency=0.8)

    tag = 'HIIGx'
    ploter(dpath, fend, result, parameters, parametersc, tag, ctag)
    return


def lnprior72(cube):
    cube[0] = cube[0]*2. + 32.5
    cube[1] = cube[1]*1. + 4.5
    cube[2] = cube[2]*0.5 + 0.5
    cube[3] = cube[3]
    cube[4] = cube[4]
    return cube


def lnlike72(cube):
    alpha, beta, h0, Om, Ol = cube[0], cube[1], cube[2], cube[3], cube[4]
    theta = alpha, beta, h0, Om, Ol
    xsq = chsq72(theta)[0]
    return -0.5*xsq


def chsq72(theta):
    alpha, beta, h0, Om, Ol = theta
    x, y, z, xerr, yerr, zerr, Rxy, Muo, MuoErr, vTg = Mdata

    cosmo = LambdaCDM(H0=h0*100.0, Om0=Om, Ode0=Ol, Tcmb0= 2.725)
    #---------------------------------------------------------------------------
    ixG = np.where(z>10.0)
    ixH = np.where(z<10.0)

    Mum = z*0.0
    MumErr = z*0.0

    Mum[ixG] = z[ixG]
    MumErr[ixG] = zerr[ixG]

    Mum[ixH] = 5.0*np.log10(cosmo.luminosity_distance(z[ixH]).value) + 25.0
    MumErr[ixH] = (5.0/np.log(10.0))*(zerr[ixH]/z[ixH])

    Mu = 2.5*(beta*x + alpha) - 2.5*y - 100.195
    MuErr = 2.5*np.sqrt((yerr)**2 + beta**2*(xerr)**2
            # - 2.0*beta*Rxy*xerr*yerr
            #+ 0.22**2
    )

    R = (Mu - Mum)
    W = 1.0/(MuErr**2 + MumErr**2)

    xsq = np.sum(R**2*W)
    xsq2 = xsq - np.sum(np.log(W))

    if fr == 2:
        return (xsq2, R, Mum, MumErr)
    else:
        return (xsq, R, Mum, MumErr)


def case72(cpath, dpath, fend, vbs, sps, prs, ctag):
    x, y, z, xerr, yerr, zerr, Rxy, Muo, MuoErr, vTg = Mdata
    parameters = [r"\alpha", r"\beta", r"h", r"\Omega_m", r"\Omega_{\Lambda}"]
    parametersc = [r"$\alpha$", r"$\beta$", r"$h$", r"$\Omega_m$",
            r"$\Omega_{\Lambda}$"]
    n_params = len(parameters)

    prefix = cpath + fend

    result = solve(LogLikelihood=lnlike72, Prior=lnprior72,
                   n_dims=n_params, outputfiles_basename=prefix, verbose=vbs,
                   n_live_points=sps, resume=prs, init_MPI=False,
                   sampling_efficiency=0.8)

    tag = 'HIIGx'
    ploter(dpath, cpath, fend, result, parameters, parametersc, tag, ctag)
    return


def lnprior73(cube):
    cube[0] = cube[0]*2. + 32.5
    cube[1] = cube[1]*1. + 4.5
    cube[2] = cube[2]*0.5 + 0.5
    cube[3] = cube[3]
    cube[4] = cube[4]
    cube[5] = cube[5]*3 - 2
    return cube


def lnlike73(cube):
    alpha, beta, h0, Om, Ol, w0 = cube[0], cube[1], cube[2], cube[3], cube[4], cube[5]
    theta = alpha, beta, h0, Om, Ol, w0
    xsq = chsq73(theta)[0]
    return -0.5*xsq


def chsq73(theta):
    alpha, beta, h0, Om, Ol, w0 = theta
    x, y, z, xerr, yerr, zerr, Rxy, Muo, MuoErr, vTg = Mdata

    cosmo = w0waCDM(H0=h0*100.0, Om0=Om, Ode0=Ol, w0=w0, wa=0.0, Tcmb0= 2.725)
    #---------------------------------------------------------------------------
    ixG = np.where(z>10.0)
    ixH = np.where(z<10.0)

    Mum = z*0.0
    MumErr = z*0.0

    Mum[ixG] = z[ixG]
    MumErr[ixG] = zerr[ixG]

    Mum[ixH] = 5.0*np.log10(cosmo.luminosity_distance(z[ixH]).value) + 25.0
    MumErr[ixH] = (5.0/np.log(10.0))*(zerr[ixH]/z[ixH])

    Mu = 2.5*(beta*x + alpha) - 2.5*y - 100.195
    MuErr = 2.5*np.sqrt((yerr)**2 + beta**2*(xerr)**2
            # - 2.0*beta*Rxy*xerr*yerr
            #+ 0.22**2
    )

    R = (Mu - Mum)
    W = 1.0/(MuErr**2 + MumErr**2)

    xsq = np.sum(R**2*W)
    xsq2 = xsq - np.sum(np.log(W))

    if fr == 2:
        return (xsq2, R, Mum, MumErr)
    else:
        return (xsq, R, Mum, MumErr)


def case73(cpath, dpath, fend, vbs, sps, prs, ctag):
    x, y, z, xerr, yerr, zerr, Rxy, Muo, MuoErr, vTg = Mdata
    parameters = [r"\alpha", r"\beta", r"h", r"\Omega_m", r"\Omega_{\Lambda}", r"w_0"]
    parametersc = [r"$\alpha$", r"$\beta$", r"$h$", r"$\Omega_m$",
            r"$\Omega_{\Lambda}$", r"$w_0$"]
    n_params = len(parameters)

    prefix = cpath + fend

    result = solve(LogLikelihood=lnlike73, Prior=lnprior73,
                   n_dims=n_params, outputfiles_basename=prefix, verbose=vbs,
                   n_live_points=sps, resume=prs, init_MPI=False,
                   sampling_efficiency=0.8)

    tag = 'HIIGx'
    ploter(dpath, cpath, fend, result, parameters, parametersc, tag, ctag)
    return


def hd(cpath, dpath, fend, h0=0.7, Om=0.3, Ol=0.7, w0=-1.0, w1=0.0):
    x, y, z, xerr, yerr, zerr, Rxy, Muo, MuoErr, vTg = Mdata

    cosmo = FlatwCDM(H0=h0*100.0, Om0=Om, w0=w0)

    ixG = np.where(z>10.0)
    ixH = np.where(z<10.0)

    Mum = z*0.0
    MumErr = z*0.0

    Mum[ixG] = z[ixG]
    MumErr[ixG] = zerr[ixG]

    Mum[ixH] = 5.0*np.log10(cosmo.luminosity_distance(z[ixH]).value) + 25.0
    MumErr[ixH] = (5.0/np.log(10.0))*(zerr[ixH]/z[ixH])

    return


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
    # ix = np.where((vx + vxErr) <= LSL)
    # ix = np.where((vx - vxErr) <= 1.8)

    return (vix[ix], vx[ix], vy[ix], vz[ix], vxErr[ix], vyErr[ix], vzErr[ix],
            vRxy[ix], vsp[ix])


def gehrz(ve, dpath, drd):
    #H0 = h0*100
    H0 = 73.2
    H0Err = 3.1

    # H0 = 69.1
    # H0Err = 3.0

    # Reading GEH2R Data
    TpathLS = dpath+'indat/tablaregiones2018.dat' #gehr2017n2.csv #gehr2017n2t22.csv'
    # TpathLS = dpath+'indat/gehr2017n2.csv'
    tGHN = Table.read(TpathLS, format='ascii')

    vANX = tGHN['logSsi']
    vANXErr = tGHN['erlogSsi']

    # vANX = tGHN['LogSigma']
    # vANXErr = tGHN['ERLogSigma']

    if drd == 0:
        vANY = np.log10(tGHN['FSC'])
        vANYErr = (1.0/np.log(10))*(tGHN['erFSC']/tGHN['FSC'])
        # vANY = np.log10(tGHN['FsinCG'])
        # vANYErr = (1.0/np.log(10))*(tGHN['ERRFsinCG']/tGHN['FsinCG'])
    elif drd == 1:
        vANY = np.log10(tGHN['FCC'])
        vANYErr = (1.0/np.log(10))*(tGHN['erFCC']/tGHN['FCC'])
    elif drd == 2:
        vANY = np.log10(tGHN['FCG'])
        vANYErr = (1.0/np.log(10))*(tGHN['erFCG']/tGHN['FCG'])

    vANZ = H0*10**(0.2*(tGHN['disM'] - 25.0))/c
    vANZErr = np.sqrt((vANZ/H0)**2*H0Err**2+(0.2*np.log(10)*vANZ)**2*tGHN['erdiM']**2)/c

    vANW = vANX*0.0
    vANWErr = vANXErr*0.0

    rXY = np.corrcoef(vANX, vANY)
    vANRxy = vANX*0.0 + rXY[0,1]

    return vANX, vANXErr, vANY, vANYErr, vANZ, vANZErr, vANW, vANWErr, vANRxy


def gehr(ve, dpath, drd):
    # Reading GEH2R Data
    #TpathLS = dpath+'indat/gehr2017n.csv'
    # TpathLS = dpath+'indat/GEHRv10.dat'
    # TpathLS = dpath+'indat/tablaregiones2018.dat'
    # TpathLS = dpath+'indat/tablaregiones2018_t_r.dat'
    TpathLS = dpath+'indat/tablaregiones2018_cm_t.dat'
    # TpathLS = dpath+'indat/tablaregiones2018_con_cm_r.dat'
    tGHN = Table.read(TpathLS, format='ascii')

    print(tGHN.colnames)

    vANX = tGHN['logSsi']   
    vANXErr = tGHN['erlogSsi']

    if drd == 0:
        vANY = np.log10(tGHN['FSC'])
        vANYErr = (1.0/np.log(10))*(tGHN['erFSC']/tGHN['FSC'])
        # vANY = np.log10(tGHN['FsinCG'])
        # vANYErr = (1.0/np.log(10))*(tGHN['ERRFsinCG']/tGHN['FsinCG'])
    elif drd == 1:
        vANY = np.log10(tGHN['FCC'])
        vANYErr = (1.0/np.log(10))*(tGHN['erFCC']/tGHN['FCC'])
    elif drd == 2:
        vANY = np.log10(tGHN['FCG'])
        vANYErr = (1.0/np.log(10))*(tGHN['erFCG']/tGHN['FCG'])

    vANZ = tGHN['mu']
    vANZErr = tGHN['std_final']
    vANZbrErr = tGHN['std_br']

    # print(vANZErr - vANZbrErr)

    vANZUErr = tGHN['conf_lvl_max']
    vANZDErr = tGHN['conf_lvl_min']

    vANW = np.log10(tGHN['EW'])
    vANWErr = (1.0/np.log(10))*(tGHN['erEWC']/tGHN['EW'])

    # vANR = tGHN['Ru']
    # vANRErr = tGHN['Ruerr']

    # ixS3 = np.where(vANR < 0.0)
    # vANZ[ixS3] = -99

    rXY = np.corrcoef(vANX, vANY)
    vANRxy = vANX*0.0 + rXY[0,1]

    return vANX, vANXErr, vANY, vANYErr, vANZ, vANZErr, vANZUErr, vANZDErr, vANW, vANWErr, vANRxy


def lh2g(ve, dpath, drd):
    # TpathH2gEc = dpath+'indat/HIIGxCh14v0817.dat'
    # TpathH2gEc = dpath+'indat/HIIGxv10.dat'
    TpathH2gEc = dpath+'indat/tablagalaxias2018.dat'

    tHII = Table.read(TpathH2gEc, format='ascii')

    vBX = tHII['logSsi']
    vBXErr = tHII['erlogSsi']

    # print np.mean(tHII['AVC'])
    # # print np.std(tHII['AVG'])
    # print np.mean(tHII['eraAVC'])

    # print np.mean((np.log10(tHII['FCG']) - np.log10(tHII['FSC']))/(0.4*3.33249))
    # print np.std((np.log10(tHII['FCG']) - np.log10(tHII['FSC']))/(0.4*3.33249))
    # print np.mean(tHII['erFCG']/tHII['FCG'])
    # print np.mean(tHII['erFSC']/tHII['FSC'])

    if drd == 0:
        vBY = np.log10(tHII['FSC'])
        vBYErr = (1.0/np.log(10)) * (tHII['erFSC']/tHII['FSC'])
    elif drd == 1:
        vBY = np.log10(tHII['FCC'])
        vBYErr = (1.0/np.log(10)) * (tHII['erFCC']/tHII['FCC'])
    elif drd == 2:
        vBY = np.log10(tHII['FCG'])
        vBYErr = (1.0/np.log(10)) * (tHII['erFCG']/tHII['FCG'])

    vBZ = tHII['z']
    vBZErr = tHII['z']*1e-2 #tHII['erZ']
    vBZErr = np.sqrt(vBZErr**2 + 0.0005**2)

    vBW = np.log10(tHII['EWCh14'])
    vBWErr = (1.0/np.log(10)) * (tHII['erEWCh14']/tHII['EWCh14'])

    # vBR = tHII['Ru']
    # vBRErr = tHII['RuErr']

    vBR = vBW*0.0
    vBRErr = vBWErr*0.0

    ixS = np.where(vBX < 0.0)
    vBZ[ixS] = -99

    # ixS2 = np.where((vBX - vBXErr) > 1.8) #1.84, 1.8
    ixS2 = np.where((vBX - vBXErr) >= LSL)
    vBZ[ixS2] = -99

    # ixS3 = np.where(vBR < 0.0)
    # vBZ[ixS3] = -99

    rXY = np.corrcoef(vBX, vBY)
    vBRxy = vBX*0.0 + rXY[0,1]

    return vBX, vBXErr, vBY, vBYErr, vBZ, vBZErr, vBW, vBWErr, vBR, vBRErr, vBRxy


def hzh2g(ve, dpath, drd, obs):
    # Reading HZ Data
    TpathLS = dpath+'indat/hzh2g.txt'
    tHzH2G = Table.read(TpathLS, format='ascii')

    index = tHzH2G['col1']

    vHX = tHzH2G['col3']
    vHXErr = tHzH2G['col4']

    vY = np.log10(tHzH2G['col5'])
    vYErr = (1.0/np.log(10)) * (tHzH2G['col6']/tHzH2G['col5'])

    vEBV = tHzH2G['col7']
    vEBVErr = tHzH2G['col8']

    # ixo = np.where(vEBV == 1000)
    # ixn = np.where(index >= 1000)

    # vEBV[ixn] = np.mean(vEBV[ixo])
    # vEBVErr[ixn] = np.mean(vEBVErr[ixo])

    # print np.mean(vEBV)
    # print np.median(vEBV)
    # print np.std(vEBV)
    # print np.mean(vEBVErr)
    # print np.median(vEBVErr)

    if drd == 0:
        vHY = vY
        vHYErr = vYErr
    elif drd == 1:
        vHY = vY + 0.4*4.598*EBVC
        vHYErr = np.sqrt(vYErr**2 + (0.4*4.598)**2*EBVCErr**2)
    elif drd == 2:
        vHY = vY + 0.4*3.33249*EBVG
        vHYErr = np.sqrt(vYErr**2 + (0.4*3.33249)**2*EBVGErr**2)
    else:
        print('error')
        # vHY = vY + 0.4*3.33249*0.19
        # vHYErr = np.sqrt(vYErr**2 + (0.4*3.33249)**2*0.10**2)
        # vHY = vY + 0.4*3.33249*vEBV
        # Calzetti
        # vHY = vY + 0.4*4.598*vEBV
        # vHYErr = np.sqrt(vYErr**2 + (0.4*4.598)**2*vEBVErr**2)
        # vHY = vY + 0.4*4.598*0.15
        # vHYErr = np.sqrt(vYErr**2 + (0.4*4.598)**2*0.06**2)

        # vHYErr = vHY*0.2

    # print 'EBV'
    # print np.mean(vEBV)
    # print np.mean(vEBVErr)

    vHZ = tHzH2G['col2']
    vHZErr = tHzH2G['col2']*0.01 #OJO

    ixS2 = np.where((vHX - vHXErr) >= LSL) #1.84
    index[ixS2] = -1

    ixS3 = np.where(index >= 1000)
    index[ixS3] = 0

    if obs == 0:
        ixr = np.where(index >= 0)
    elif obs == 1:
        ixr = np.where(index >= 1)
    elif obs == 2:
        ixr = np.where((index >= 1)&(index < 200))
    elif obs == 3:
        ixr = np.where((index >= 0)&(index < 200))
    else:
        ixr = np.where(index >= 200)

    rXY = np.corrcoef(vHX, vHY)
    vHRxy = vHX*0.0 + rXY[0,1]

    return vHX[ixr], vHXErr[ixr], vHY[ixr], vHYErr[ixr], vHZ[ixr], vHZErr[ixr], vHRxy[ixr]


def hzmfh2g(ve, dpath, drd, obs):
    if drd == 0:
        print('error')
    elif drd == 1:
        Tpath = dpath+'indat/Union2018Cv2.txt'
        tHzH2Gc = Table.read(Tpath, format='ascii')

        vID = tHzH2Gc['col1']
        vHX = tHzH2Gc['col2']
        vHY = tHzH2Gc['col3']
        vHZ = tHzH2Gc['col4']
        vHXErr = tHzH2Gc['col5']
        vHYErr = tHzH2Gc['col6']
        vHZErr = tHzH2Gc['col7']
    elif drd == 2:
        Tpath = dpath+'indat/Union2018Gv5.txt'
        tHzH2Gg = Table.read(Tpath, format='ascii')

        vID = tHzH2Gg['col1']
        vHX = tHzH2Gg['col2']
        vHY = tHzH2Gg['col3']
        vHZ = tHzH2Gg['col4']
        vHXErr = tHzH2Gg['col5']
        vHYErr = tHzH2Gg['col6']
        vHZErr = tHzH2Gg['col7']
    else:
        print('error')

    ixi = np.where(vID >= 139)
    vHX = vHX[ixi]
    vHY = vHY[ixi]
    vHZ = vHZ[ixi]
    vHXErr = vHXErr[ixi]
    vHYErr = vHYErr[ixi]
    vHZErr = vHZErr[ixi]

    ixr = np.where((vHX - vHXErr) <= LSL) #test 1.84

    rXY = np.corrcoef(vHX, vHY)
    vHRxy = vHX*0.0 + rXY[0,1]

    return vHX[ixr], vHXErr[ixr], vHY[ixr], vHYErr[ixr], vHZ[ixr], vHZErr[ixr], vHRxy[ixr]


def hzh2g2020(ve, dpath, drd, obs):
    Tpath = dpath+'indat/hzh2g2020v2.dat'
    data = Table.read(Tpath, format='ascii', comment='#')

    vsp = data['flag']

    vix = data['ID']
    vName = data['Target']

    sigO = data['sigma_obs']
    sigOErr = data['err_sigma_obs']

    sigI = data['sigma_RES_INST']
    sigIErr = data['err_sigma_RES_INST']

    sigfs = data['sigma_fs']
    sigth = data['sigma_th']

    ixmf = np.where(vsp == 4)
    sigIErr[ixmf] = 22.6816746*0.03 # MOSFIE inst. broad. uncertainty fixed at 3%

    ixs = np.where(sigIErr > 0.)

    sigC = np.sqrt(sigO[ixs]**2 - sigI[ixs]**2 - sigfs[ixs]**2 - sigth[ixs]**2)
    sigCErr = np.sqrt((sigO[ixs]/sigC)**2*sigOErr[ixs]**2
                      + (sigI[ixs]/sigC)**2*sigIErr[ixs]**2)

    lsig = np.log10(sigC)
    lsigErr = 1./(np.log(10.))*(sigCErr/sigC)

    ixr = np.where((lsig - lsigErr) <= LSL)

    vixf = vix[ixs][ixr]
    vNamef = vName[ixs][ixr]

    KX = lsig[ixr]
    KXErr = lsigErr[ixr]

    vfxo = data['FHx_obs'][ixs][ixr]*1.e-17
    vfxoErr = data['err_FHx_obs'][ixs][ixr]*1.e-17

    vfg = data['note'][ixs][ixr]
    vk = data['Kapha_Hx'][ixs][ixr]
    vAv = data['Av'][ixs][ixr]
    vAvErr = data['err_Av'][ixs][ixr]
    Rv = 2.77

    vfc = vfxo*0.0
    vfcErr = vfxo*0.0
    for i in range(len(vfc)):
        if vfg[i] == 2:
            vfc[i] = vfxo[i]*10.**((0.4*vAv[i]*vk[i])/(Rv))
            vfcErr[i] = vfc[i]*np.sqrt((vfxoErr[i]/vfxo[i])**2
                                       + ((0.4*vk[i]*np.log(10.))/Rv)**2*vAvErr[i]**2)
        else:
            vfc[i] = (vfxo[i]*10.**((0.4*vAv[i]*vk[i])/(Rv)))/2.86
            vfcErr[i] = vfc[i]*np.sqrt((vfxoErr[i]/vfxo[i])**2
                                       + ((0.4*vk[i]*np.log(10.))/Rv)**2*vAvErr[i]**2)

    KY = np.log10(vfc)
    KYErr = (1./np.log(10.))*(vfcErr/vfc)

    KZ = data['z'][ixs][ixr]
    KZErr = data['err_z'][ixs][ixr]

    vspf = data['flag'][ixs][ixr]

    rXY = np.corrcoef(KX, KY)
    vKRxy = KX*0.0 + rXY[0,1]

    # lsigt = data['logsigma_corr'][ixs][ixr]
    # lsigtErr = data['logerr_sigma_corr'][ixs][ixr]
    # lfxc = data['logFHb_corr'][ixs][ixr]
    # lfxcErr = data['logerr_FHb_corr'][ixs][ixr]
    # print(vfc)

    # for i in range(len(KX)-1):
        # print(vixf[i], KZ[i], KZErr[i], KX[i], KXErr[i], vKRxy[i], vspf[i])
    # print(len(KX))

    return KX, KXErr, KY, KYErr, KZ, KZErr, vKRxy, vspf


def hzLlerena(ve, dpath, drd, obs):
    Tpath = dpath+'indat/hzLlerena.csv'
    data = Table.read(Tpath, format='csv', comment='#')
    dX = data.to_pandas()

    Tpathf = dpath+'indat/Llerena2023FluxesClean.csv'
    fdata = Table.read(Tpathf, format='csv', comment='#')
    dY = fdata.to_pandas()

    Tpathz = dpath+'indat/Llerena2023Redshifts.tex'
    zdata = Table.read(Tpathz)
    dZ = zdata.to_pandas()

    # print(dX.head())
    # print(dY.head())
    # print(dZ.head())

    # print('LLerena Dataset')
    df = merged_df = dX.merge(dY, on='ID').merge(dZ, on='ID')

    # print(df.head())
    # print(df.columns)

    sigO = df['w_O3_single'].values
    sigOErr = df['w_O3_single_err'].values 

    # sigO = df['w_O3_N'].values
    # sigOErr = df['w_O3_N_err'].values

    # sigC = np.sqrt(sigO**2 + 2.91**2)
    # sigCErr = np.sqrt((sigO/sigC)**2*sigOErr**2 + (2.91/sigC)**2*0.31**2)

    sigC = sigO + 2.91 
    sigCErr = np.sqrt(sigOErr**2 + 0.31**2)

    vLX = np.log10(sigC)
    vLXErr = 1./(np.log(10.))*(sigCErr/sigC)

    fHb = df['Hb'].values
    fHbErr = df['HbErr'].values

    fHa = df['Ha'].values
    fHaErr = df['HaErr'].values

    ixf = np.where(fHbErr > 0.0)

    vID = df['ID'].values[ixf]

    vLX = vLX[ixf]
    vLXErr = vLXErr[ixf]

    fHa = fHa[ixf]
    fHaErr = fHaErr[ixf]

    fHb = fHb[ixf]
    fHbErr = fHbErr[ixf]

    vY = np.log10(fHb*1.e-18)
    vYErr = (1./np.log(10.))*(fHbErr/fHb)

    if drd == 0:
        vHY = vY
        vHYErr = vYErr
    elif drd == 1:
        vHY = vY + 0.4*4.598*EBVC
        vHYErr = np.sqrt(vYErr**2 + (0.4*4.598)**2*EBVCErr**2)
    elif drd == 2:
        kHa = 2.22
        kHb = 3.33249
        Rv = 2.77 #2.74+-0.13 # 2.77

        vHY = vY*0.0
        vHYErr = vYErr*0.0

        EBV = vY*0.0
        EBVErr = vY*0.0

        ixC = np.where(fHaErr > 0.0)
        EBVc = (np.log10(2.86) - np.log10(fHa[ixC]/fHb[ixC]))/(0.4*(kHa - kHb))
        EBVcErr = np.sqrt((1./(0.4*(kHa - kHb)*np.log(10.))**2)*((1./fHa[ixC])**2*fHaErr[ixC]**2
                                                                    + (1./fHb[ixC])**2*fHbErr[ixC]**2))
        
        EBV[ixC] = EBVc
        EBVErr[ixC] = EBVcErr

        EBVm = np.mean(EBVc)
        EBVErrm = np.std(EBVcErr)

        print('EBV mean LLerena')
        print(EBVm, EBVErrm)

        ixE = np.where(fHaErr < 0.0)
        EBV[ixE] = EBVm
        EBVErr[ixE] = EBVErrm
            
        vHY = vY + 0.4*3.33249*EBV
        vHYErr = np.sqrt(vYErr**2 + (0.4*3.33249)**2*EBVErr**2)
    else:
        print('error')
    
    vLY = vHY
    vLYErr = vHYErr

    vLZ = df['z'].values[ixf]
    vLZErr = vLZ*0.0001

    vEWHb = df['EWHb'].values[ixf]

    ixW = np.where(vEWHb >= 49.)
    vID = vID[ixW]
    vEWHb = vEWHb[ixW]
    vLX = vLX[ixW]
    vLXErr = vLXErr[ixW]
    vLY = vLY[ixW]
    vLYErr = vLYErr[ixW]
    vLZ = vLZ[ixW]
    vLZErr = vLZErr[ixW]
    EBV = EBV[ixW]
    EBVErr = EBVErr[ixW]    

    rXY = np.corrcoef(vLX, vLY)
    vLRxy = vLX*0.0 + rXY[0,1]

    ixr = np.where((vLX - vLXErr) <= 1.85)
    vID = vID[ixr]
    vEWHb = vEWHb[ixr]
    vLX = vLX[ixr]
    vLXErr = vLXErr[ixr]
    vLY = vLY[ixr]
    vLYErr = vLYErr[ixr]
    vLZ = vLZ[ixr]
    vLZErr = vLZErr[ixr]
    vLRxy = vLRxy[ixr]
    vEBV = EBV[ixr]
    vEBVErr = EBVErr[ixr]

    print('Data')
    for i in range(len(vID)):
        print(vID[i], vEWHb[i], vLX[i], vLXErr[i], vLY[i], vLYErr[i], vLZ[i], vLZErr[i], vEBV[i], vEBVErr[i]) 

    return vLX, vLXErr, vLY, vLYErr, vLZ, vLZErr, vLRxy 


def hzJWST(ve, dpath, drd, obs):
    tc = 1

    index = np.array([16745, 10016374, 19606, 22251, 47100])

    # Correcting for Thermal broadening
    if tc == 1:
        sigH = np.array([55.0, 62.0])
        sigHErr = np.array([2.0, 2.0])

        sigO = np.array([39.1, 39.0, 71.0])
        sigOErr = np.array([1.8, 1.0, 4.0])

        T = 15000.0 #np.array([15000.0, 15000.0])
        TErr = 100.0 #np.array([100.0, 100.0])

        sigthH = sigH*0.0 + np.sqrt((k_b * T)/m_p) * 1.e-5
        sigthHErr = sigH*0.0 + (sigthH/2)*(TErr/T)

        sigthO = sigO*0.0 + np.sqrt((k_b * T)/(m_p*16.0)) * 1.e-5
        sigthOErr = sigO*0.0 + (sigthO/2)*(TErr/T)

        sigHC = np.sqrt(sigH**2 - sigthH**2)
        sigHCErr = np.sqrt((sigH/sigHC)**2*sigHErr**2 + (sigthH/sigHC)**2*sigthHErr**2)

        sigOC = np.sqrt(sigO**2 - sigthO**2)
        sigOCErr = np.sqrt((sigO/sigOC)**2*sigOErr**2 + (sigthO/sigOC)**2*sigthOErr**2)

        # sigOOC = np.sqrt(sigOC**2 + 2.91**2)
        # sigOOCErr = np.sqrt((sigOC/sigOOC)**2*sigOCErr**2 + (2.91/sigOOC)**2*0.31**2)

        sigOOC = sigOC + 2.91
        sigOOCErr = np.sqrt(sigOCErr**2 + 0.31**2)

        # sigOOC = sigOC
        # sigOOCErr = sigOCErr

        sigC = np.concatenate((sigHC, sigOOC))
        sigCErr = np.concatenate((sigHCErr, sigOOCErr))

        vJX = np.log10(sigC)
        vJXErr = 1./(np.log(10.))*(sigCErr/sigC)

        # print(vJX)
        # print(vJXErr)
        # vJXErr = vJXErr#*3.2
    else:
        vJX = np.array([1.74, 1.79])
        vJXErr = np.array([0.01579, 0.014])

    # Fluxes and Errors
    FHa = np.array([207.72, 179.18, 189.51, 337.58, 440.0*0.63])
    FHaErr = np.array([6.91, 8.07, 16.15, 9.13, 30.0*0.63])

    FHb = np.array([40.86, 71.71, 61.46, 109.33, 440.0/4.5])
    FHbErr = np.array([6.19, 9.86, 10.89, 8.48, 30.0/4.5])

    vY = np.log10(FHb*1.e-20)
    vYErr = (1./np.log(10.))*(FHbErr/FHb)

    if drd == 0:
        vHY = vY
        vHYErr = vYErr
    elif drd == 1:
        vHY = vY + 0.4*4.598*EBVC
        vHYErr = np.sqrt(vYErr**2 + (0.4*4.598)**2*EBVCErr**2)
    elif drd == 2:
        kHa = 2.22
        kHb = 3.33249
        Rv = 2.77 #2.74+-0.13 # 2.77

        EBV = (np.log10(2.86) - np.log10(FHa/FHb))/(0.4*(kHa - kHb))
        EBVErr = np.sqrt((1./(0.4*(kHa - kHb)*np.log(10.))**2)*((1./FHa)**2*FHaErr**2
                                                        + (1./FHb)**2*FHbErr**2))
        EBV[np.where(EBV < 0)] = 0. 
        EBV[4] = 0.8/Rv 
        EBVErr[4] = (1./Rv)*0.8

        print(EBV)
        print(EBVErr)

        vHY = vY + 0.4*3.33249*EBV
        vHYErr = np.sqrt(vYErr**2 + (0.4*3.33249)**2*EBVErr**2)
    else:
        print('error')
    
    vJY = vHY #- np.log10(2.86)
    vJYErr = vHYErr #*3.2

    vJZ = np.array([5.56616, 5.50411, 5.88979, 5.79912, 7.43173])
    vJZErr = np.array([0.00011, 0.00007, 0.00008, 0.00007, 0.00015])

    rXY = np.corrcoef(vJX, vJY)
    vJRxy = vJX*0.0 + rXY[0,1]

    print(vJX - vJXErr)
    ixr = np.where((vJX - vJXErr) <= 1.85) #test 1.84 LSL

    return vJX[ixr], vJXErr[ixr], vJY[ixr], vJYErr[ixr], vJZ[ixr], vJZErr[ixr], vJRxy[ixr]


def mch2gv79(ve, dpath, cpath, clc=0, opt=0, spl=0, zps=0, prs=1, drd=0,
             obs=0, nwk=1000, sps=10000, tds=6, vbs=0, fr0=1, fra=0,
             a=0, aErr=0, b=0, bErr=0):
    print('+++++++++++++++++++++++++++++++++++++++++++')
    print('mch2gv79: ' + str(datetime.datetime.now()))
    print('+++++++++++++++++++++++++++++++++++++++++++')

    # ============= Parameters ======================================
    global Mdata
    global MdataA
    # global lsdata
    global alpha
    global alphaErr
    global beta
    global betaErr
    global fr
    global fa
    global x0
    global x0Err
    global start
    global zs
    global fg0

    zs = zps

    alpha = a
    alphaErr = aErr
    beta = b
    betaErr = bErr

    print(alpha, beta)

    x0 = x0
    x0Err = x0Err

    fr = fr0
    fa = fra
    fg0 = clc
    fg1 = opt
    fsim = 1

    if fg0 == 0:
        fend = 'mchv79_' + str(ve) + '_' + str(spl) + str(zps) + str(fg1) + \
            str(drd) + str(obs) + str(fr0) + str(fra) + '_GM19_stat'
    elif fg0 == 1:
        fend = 'mchv79_' + str(ve) + '_' + str(spl) + str(zps) + str(fg1) + \
            str(drd) + str(obs) + str(fr0) + str(fra) + '_GM19_sys'

    print(fend)
    print(datetime.datetime.now())
    start = datetime.datetime.now()
    # ============= Main Body =======================================
    if spl == 1:
        # Union2020
        # 0: GEHRs, 1:local HIIG, 2: Literatura, 3: XShooter, 4: MOSFIRE, 5: KMOS
        vTg, vX, vY, vZ, vXErr, vYErr, vZErr, vRxy, vsp = union2020(ve, dpath, drd, obs)

        # JWST Data
        vJX, vJXErr, vJY, vJYErr, vJZ, vJZErr, vJRxy = hzJWST(ve, dpath, drd, obs)
        ixJs = vJX*0 + 6

        # LLerana Data
        vLX, vLXErr, vLY, vLYErr, vLZ, vLZErr, vLRxy = hzLlerena(ve, dpath, drd, obs)
        ixLs = vLX*0 + 7

        if (zps == 0):
            ix = np.where(vX > 0)
        elif (zps == 1 or zps == 2 or zps == 3):
            ix = np.where(vsp != 0)
        elif (zps == 6):
            ix = np.where((vsp != 0) & (vsp != 2))
        elif (zps == 7):
            ix = np.where(vsp != 1)
        elif (zps == 8):
            ix = np.where((vsp != 0) & (vsp != 1))
        elif (zps == 10):
            ix = np.where(vsp == 0)
        elif (zps == 11):
            ix = np.where(vsp == 1)
        elif (zps == 12):
            ix = np.where((vsp == 0) | (vsp == 1))
        elif (zps == 22):
            ix = np.where(vsp != 2)
        elif (zps == 23):
            ix = np.where((vsp == 0) | (vsp == 1) | (vsp == 5))
        elif (zps == 24):
            ix = np.where(vsp != 5)
        elif (zps == 25):
            ix = np.where((vsp != 5) & (vsp != 2))
        elif (zps == 26):
            ix = np.where((vZ <= 0.1) | (vZ > 5.0))
        elif (zps == 30):
            ixU = np.where(vX > 0)
            ixJ = np.where(vJZ > 0)
            print(len(vX[ixU]), len(vJX[ixJ]))

            vX = np.concatenate((vX[ixU], vJX[ixJ]))
            vXErr = np.concatenate((vXErr[ixU], vJXErr[ixJ]))

            vY = np.concatenate((vY[ixU], vJY[ixJ]))
            vYErr = np.concatenate((vYErr[ixU], vJYErr[ixJ]))

            vZ = np.concatenate((vZ[ixU], vJZ[ixJ]))
            vZErr = np.concatenate((vZErr[ixU], vJZErr[ixJ]))

            # vZUErr = np.concatenate((vzpZUErr[ixAZ], vZErr[ixU]))
            # vZDErr = np.concatenate((vzpZDErr[ixAZ], vZErr[ixU]))

            vRxy = np.concatenate((vRxy[ixU], vJRxy[ixJ]))

            vsp = np.concatenate((vsp[ixU], ixJs[ixJ]))
            vTg = vsp

            ix = np.where(vX > 0)
        elif (zps == 31):
            ixU = np.where(vsp != 0)
            ixJ = np.where(vJZ > 0)
            print(len(vX[ixU]), len(vJX[ixJ]))

            vX = np.concatenate((vX[ixU], vJX[ixJ]))
            vXErr = np.concatenate((vXErr[ixU], vJXErr[ixJ]))

            vY = np.concatenate((vY[ixU], vJY[ixJ]))
            vYErr = np.concatenate((vYErr[ixU], vJYErr[ixJ]))

            vZ = np.concatenate((vZ[ixU], vJZ[ixJ]))
            vZErr = np.concatenate((vZErr[ixU], vJZErr[ixJ]))

            # vZUErr = np.concatenate((vzpZUErr[ixAZ], vZErr[ixU]))
            # vZDErr = np.concatenate((vzpZDErr[ixAZ], vZErr[ixU]))

            vRxy = np.concatenate((vRxy[ixU], vJRxy[ixJ]))

            vsp = np.concatenate((vsp[ixU], ixJs[ixJ]))
            vTg = vsp

            ix = np.where(vX > 0)
        elif (zps == 40):
            ixU = np.where(vX > 0)
            ixJ = np.where(vJZ > 0)
            ixL = np.where(vLZ > 0)
            print(len(vX[ixU]), len(vJX[ixJ]), len(vLX[ixL]))

            vX = np.concatenate((vX[ixU], vJX[ixJ], vLX[ixL]))
            vXErr = np.concatenate((vXErr[ixU], vJXErr[ixJ], vLXErr[ixL]))

            vY = np.concatenate((vY[ixU], vJY[ixJ], vLY[ixL]))
            vYErr = np.concatenate((vYErr[ixU], vJYErr[ixJ], vLYErr[ixL]))

            vZ = np.concatenate((vZ[ixU], vJZ[ixJ], vLZ[ixL]))
            vZErr = np.concatenate((vZErr[ixU], vJZErr[ixJ], vLZErr[ixL]))

            # vZUErr = np.concatenate((vzpZUErr[ixAZ], vZErr[ixU]))
            # vZDErr = np.concatenate((vzpZDErr[ixAZ], vZErr[ixU]))

            vRxy = np.concatenate((vRxy[ixU], vJRxy[ixJ], vLRxy[ixL]))

            vsp = np.concatenate((vsp[ixU], ixJs[ixJ], ixLs[ixL]))
            vTg = vsp

            ix = np.where(vX > 0)
        elif (zps == 41):
            ixU = np.where(vsp != 0)
            ixJ = np.where(vJZ > 0)
            ixL = np.where(vLZ > 0)
            print(len(vX[ixU]), len(vJX[ixJ]), len(vLX[ixL]))

            vX = np.concatenate((vX[ixU], vJX[ixJ], vLX[ixL]))
            vXErr = np.concatenate((vXErr[ixU], vJXErr[ixJ], vLXErr[ixL]))

            vY = np.concatenate((vY[ixU], vJY[ixJ], vLY[ixL]))
            vYErr = np.concatenate((vYErr[ixU], vJYErr[ixJ], vLYErr[ixL]))

            vZ = np.concatenate((vZ[ixU], vJZ[ixJ], vLZ[ixL]))
            vZErr = np.concatenate((vZErr[ixU], vJZErr[ixJ], vLZErr[ixL]))

            # vZUErr = np.concatenate((vzpZUErr[ixAZ], vZErr[ixU]))
            # vZDErr = np.concatenate((vzpZDErr[ixAZ], vZErr[ixU]))

            vRxy = np.concatenate((vRxy[ixU], vJRxy[ixJ], vLRxy[ixL]))

            vsp = np.concatenate((vsp[ixU], ixJs[ixJ], ixLs[ixL]))
            vTg = vsp

            ix = np.where(vX > 0)
        elif (zps == 42):
            # OJO corregir
            ixU = np.where((vsp == 0) | (vsp == 1))
            ixJ = np.where(vJZ > 0)
            ixL = np.where(vLZ > 0)
            print(len(vX[ixU]), len(vJX[ixJ]), len(vLX[ixL]))

            vX = np.concatenate((vX[ixU], vJX[ixJ], vLX[ixL]))
            vXErr = np.concatenate((vXErr[ixU], vJXErr[ixJ], vLXErr[ixL]))

            vY = np.concatenate((vY[ixU], vJY[ixJ], vLY[ixL]))
            vYErr = np.concatenate((vYErr[ixU], vJYErr[ixJ], vLYErr[ixL]))

            vZ = np.concatenate((vZ[ixU], vJZ[ixJ], vLZ[ixL]))
            vZErr = np.concatenate((vZErr[ixU], vJZErr[ixJ], vLZErr[ixL]))

            # vZUErr = np.concatenate((vzpZUErr[ixAZ], vZErr[ixU]))
            # vZDErr = np.concatenate((vzpZDErr[ixAZ], vZErr[ixU]))

            vRxy = np.concatenate((vRxy[ixU], vJRxy[ixJ], vLRxy[ixL]))

            vsp = np.concatenate((vsp[ixU], ixJs[ixJ], ixLs[ixL]))
            vTg = vsp

            ix = np.where(vX > 0)
        elif (zps == 50):
            ixU = np.where(vX > 0)
            ixL = np.where(vLZ > 0)
            print(len(vX[ixU]), len(vLX[ixL]))

            vX = np.concatenate((vX[ixU], vLX[ixL]))
            vXErr = np.concatenate((vXErr[ixU], vLXErr[ixL]))

            vY = np.concatenate((vY[ixU], vLY[ixL]))
            vYErr = np.concatenate((vYErr[ixU], vLYErr[ixL]))

            vZ = np.concatenate((vZ[ixU], vLZ[ixL]))
            vZErr = np.concatenate((vZErr[ixU], vLZErr[ixL]))

            # vZUErr = np.concatenate((vzpZUErr[ixAZ], vZErr[ixU]))
            # vZDErr = np.concatenate((vzpZDErr[ixAZ], vZErr[ixU]))

            vRxy = np.concatenate((vRxy[ixU], vLRxy[ixL]))

            vsp = np.concatenate((vsp[ixU], ixLs[ixL]))
            vTg = vsp

            ix = np.where(vX > 0)
        else:
            print('zps option error')
    elif spl == 2:
        # GEHR Data zero point
        vzpX, vzpXErr, vzpY, vzpYErr, vzpZ, vzpZErr, vzpW, vzpWErr, vzpRxy = \
                gehr(ve, dpath, drd)
        ixzp = vzpX*0 + 0
        # GEHR Data
        vANX, vANXErr, vANY, vANYErr, vANZ, vANZErr, vANW, vANWErr, vANRxy = \
                gehrz(ve, dpath, drd)
        ixAN = vANX*0 + 0
        # Local HIIgx Data
        vBX, vBXErr, vBY, vBYErr, vBZ, vBZErr, vBW, vBWErr, vBR, vBRErr, vBRxy = \
                lh2g(ve, dpath, drd)
        ixB = vBX*0 + 1
        # Reading HZ Datatest
        vHX, vHXErr, vHY, vHYErr, vHZ, vHZErr, vHRxy = hzh2g(ve, dpath, drd, obs)
        ixH = vHX*0 + 2

        # Reading HZ Data Mosfire
        vHMX, vHMXErr, vHMY, vHMYErr, vHMZ, vHMZErr, vHMRxy = \
                hzmfh2g(ve, dpath, drd, obs)
        ixHM = vHMX*0 + 3

        zlim = 0.0
        ixAZ = np.where(vANZ >= zlim)
        ixBZ = np.where(vBZ <= 0.2)
        ixHMZ = np.where(vHMZ <= 3.0)

        # Joining samples
        if zps == 1:
            print(len(vANX[ixAZ]), len(vBX[ixBZ]), len(vHX), len(vHMX))

            vX = np.concatenate((vANX[ixAZ], vBX[ixBZ], vHX, vHMX))
            vXErr = np.concatenate((vANXErr[ixAZ], vBXErr[ixBZ], vHXErr, vHMXErr))

            vY = np.concatenate((vANY[ixAZ], vBY[ixBZ], vHY, vHMY))
            vYErr = np.concatenate((vANYErr[ixAZ], vBYErr[ixBZ], vHYErr, vHMYErr))

            vZ = np.concatenate((vANZ[ixAZ], vBZ[ixBZ], vHZ, vHMZ))
            vZErr = np.concatenate((vANZErr[ixAZ], vBZErr[ixBZ], vHZErr, vHMZErr))

            vRxy = np.concatenate((vANRxy[ixAZ], vBRxy[ixBZ], vHRxy, vHMRxy))

            vsp = np.concatenate((ixAN[ixAZ], ixB[ixBZ], ixH, ixHM))
            vTg = vsp

            ix = np.where(vX > 0)
        elif zps == 2:
            print(len(vBX[ixBZ]), len(vHX), len(vHMX))

            vX = np.concatenate((vBX[ixBZ], vHX, vHMX))
            vXErr = np.concatenate((vBXErr[ixBZ], vHXErr, vHMXErr))

            vY = np.concatenate((vBY[ixBZ], vHY, vHMY))
            vYErr = np.concatenate((vBYErr[ixBZ], vHYErr, vHMYErr))

            vZ = np.concatenate((vBZ[ixBZ], vHZ, vHMZ))
            vZErr = np.concatenate((vBZErr[ixBZ], vHZErr, vHMZErr))

            vRxy = np.concatenate((vBRxy[ixBZ], vHRxy, vHMRxy))

            vsp = np.concatenate((ixAN[ixAZ], ixB[ixBZ], ixH, ixHM))
            vTg = vsp

            ix = np.where(vX > 0)
        elif zps == 5:
            print(len(vzpX), len(vBX[ixBZ]), len(vHX), len(vHMX[ixHMZ]))

            vX = np.concatenate((vzpX, vBX[ixBZ], vHX, vHMX[ixHMZ]))
            vXErr = np.concatenate((vzpXErr, vBXErr[ixBZ], vHXErr, vHMXErr[ixHMZ]))

            vY = np.concatenate((vzpY, vBY[ixBZ], vHY, vHMY[ixHMZ]))
            vYErr = np.concatenate((vzpYErr, vBYErr[ixBZ], vHYErr, vHMYErr[ixHMZ]))

            vZ = np.concatenate((vzpZ, vBZ[ixBZ], vHZ, vHMZ[ixHMZ]))
            vZErr = np.concatenate((vzpZErr, vBZErr[ixBZ], vHZErr, vHMZErr[ixHMZ]))

            vRxy = np.concatenate((vzpRxy, vBRxy[ixBZ], vHRxy, vHMRxy[ixHMZ]))

            vsp = np.concatenate((ixzp[ixAZ], ixB[ixBZ], ixH, ixHM))
            vTg = vsp

            ix = np.where(vX > 0)
    elif spl == 3:
        # GEHR Data zero point
        vzpX, vzpXErr, vzpY, vzpYErr, vzpZ, vzpZErr, vzpW, vzpWErr, vzpRxy = \
                gehr(ve, dpath, drd)
        ixzp = vzpX*0 + 0
        # GEHR Data
        vANX, vANXErr, vANY, vANYErr, vANZ, vANZErr, vANW, vANWErr, vANRxy = \
                gehrz(ve, dpath, drd)
        ixAN = vANX*0 + 0
        # Local HIIgx Data
        vBX, vBXErr, vBY, vBYErr, vBZ, vBZErr, vBW, vBWErr, vBR, vBRErr, vBRxy = \
                lh2g(ve, dpath, drd)
        ixB = vBX*0 + 1
        # Reading HZ Datatest
        vHX, vHXErr, vHY, vHYErr, vHZ, vHZErr, vHRxy = hzh2g(ve, dpath, drd, obs)
        ixH = vHX*0 + 2

        # Reading HZ Data Mosfire
        vHMX, vHMXErr, vHMY, vHMYErr, vHMZ, vHMZErr, vHMRxy = \
                hzmfh2g(ve, dpath, drd, obs)
        ixHM = vHMX*0 + 3

        # Reading KMOS (hzh2g2020v2) data
        vKX, vKXErr, vKY, vKYErr, vKZ, vKZErr, vKRxy, vKTg = \
                hzh2g2020(ve, dpath, drd, obs)

        zlim = 0.0
        ixAZ = np.where(vANZ >= zlim)
        ixBZ = np.where(vBZ <= 0.2)
        ixHMZ = np.where(vHMZ <= 3.0)

        # Joining samples
        if zps == 1:
            print(len(vzpZ[ixAZ]), len(vBX[ixBZ]), len(vHX), len(vKX))

            vX = np.concatenate((vzpX[ixAZ], vBX[ixBZ], vHX, vKX))
            vXErr = np.concatenate((vzpXErr[ixAZ], vBXErr[ixBZ], vHXErr, vKXErr))

            vY = np.concatenate((vzpY[ixAZ], vBY[ixBZ], vHY, vKY))
            vYErr = np.concatenate((vzpYErr[ixAZ], vBYErr[ixBZ], vHYErr, vKYErr))

            vZ = np.concatenate((vzpZ[ixAZ], vBZ[ixBZ], vHZ, vKZ))
            vZErr = np.concatenate((vzpZErr[ixAZ], vBZErr[ixBZ], vHZErr, vKZErr))

            vRxy = np.concatenate((vzpRxy[ixAZ], vBRxy[ixBZ], vHRxy, vKRxy))

            vsp = np.concatenate((ixzp[ixAZ], ixB[ixBZ], ixH, vKTg))
            vTg = vsp

            ix = np.where(vX > 0)
        elif zps == 2:
            print(len(vzpZ[ixAZ]), len(vBX[ixBZ]), len(vKX))

            vX = np.concatenate((vzpX[ixAZ], vBX[ixBZ], vKX))
            vXErr = np.concatenate((vzpXErr[ixAZ], vBXErr[ixBZ], vKXErr))

            vY = np.concatenate((vzpY[ixAZ], vBY[ixBZ], vKY))
            vYErr = np.concatenate((vzpYErr[ixAZ], vBYErr[ixBZ], vKYErr))

            vZ = np.concatenate((vzpZ[ixAZ], vBZ[ixBZ], vKZ))
            vZErr = np.concatenate((vzpZErr[ixAZ], vBZErr[ixBZ], vKZErr))

            vRxy = np.concatenate((vzpRxy[ixAZ], vBRxy[ixBZ], vKRxy))

            vsp = np.concatenate((ixzp[ixAZ], ixB[ixBZ], vKTg))
            vTg = vsp

            ix = np.where(vX > 0)
    elif spl == 4:
        print('spl = 4')
        # Union2020
        # 0: GEHRs, 1:local HIIG, 2: Literatura, 3: XShooter, 4: MOSFIRE, 5: KMOS
        vTg, vX, vY, vZ, vXErr, vYErr, vZErr, vRxy, vsp = union2020(ve, dpath, drd, obs)
        
        # New 2023 GEHR Data zero point
        vzpX, vzpXErr, vzpY, vzpYErr, vzpZ, vzpZErr, vzpZUErr, vzpZDErr, vzpW, vzpWErr, vzpRxy = \
                gehr(ve, dpath, drd)
        ixzp = vzpX*0 + 0

        # JWST Data
        vJX, vJXErr, vJY, vJYErr, vJZ, vJZErr, vJRxy = hzJWST(ve, dpath, drd, obs)
        ixJs = vJX*0 + 6

        # LLerana Data
        vLX, vLXErr, vLY, vLYErr, vLZ, vLZErr, vLRxy = hzLlerena(ve, dpath, drd, obs)
        ixLs = vLX*0 + 7

        if (zps == 0):
            ixAZ = np.where(vzpZ > 0)
            ixU = np.where(vsp != 0)
            ixJ = np.where(vJZ > 0)
            print(len(vzpX[ixAZ]), len(vX[ixU]), len(vJX[ixJ]))

            vX = np.concatenate((vzpX[ixAZ], vX[ixU], vJX[ixJ]))
            vXErr = np.concatenate((vzpXErr[ixAZ], vXErr[ixU], vJXErr[ixJ]))

            vY = np.concatenate((vzpY[ixAZ], vY[ixU], vJY[ixJ]))
            vYErr = np.concatenate((vzpYErr[ixAZ], vYErr[ixU], vJYErr[ixJ]))

            vZ = np.concatenate((vzpZ[ixAZ], vZ[ixU], vJZ[ixJ]))
            vZErr = np.concatenate((vzpZErr[ixAZ], vZErr[ixU], vJZErr[ixJ]))

            # vZUErr = np.concatenate((vzpZUErr[ixAZ], vZErr[ixU]))
            # vZDErr = np.concatenate((vzpZDErr[ixAZ], vZErr[ixU]))

            vRxy = np.concatenate((vzpRxy[ixAZ], vRxy[ixU], vJRxy[ixJ]))

            vsp = np.concatenate((ixzp[ixAZ], vsp[ixU], ixJs[ixJ]))
            vTg = vsp

            ix = np.where(vX > 0)
        elif (zps == 1):
            ixAZ = np.where(vzpX > 0)
            ixU = np.where(vsp == 1)
            print(len(vzpX[ixAZ]), len(vX[ixU]))

            vX = np.concatenate((vzpX[ixAZ], vX[ixU]))
            vXErr = np.concatenate((vzpXErr[ixAZ], vXErr[ixU]))

            vY = np.concatenate((vzpY[ixAZ], vY[ixU]))
            vYErr = np.concatenate((vzpYErr[ixAZ], vYErr[ixU]))

            vZ = np.concatenate((vzpZ[ixAZ], vZ[ixU]))
            vZErr = np.concatenate((vzpZErr[ixAZ], vZErr[ixU]))

            vRxy = np.concatenate((vzpRxy[ixAZ], vRxy[ixU]))

            vsp = np.concatenate((ixzp[ixAZ], vsp[ixU]))
            vTg = vsp

            ix = np.where(vX > 0)
        elif (zps == 2):
            ixAZ = np.where(vzpX > 0)
            ixU = np.where((vsp != 0) & (vsp != 2))
            print(len(vzpX[ixAZ]), len(vX[ixU]))

            vX = np.concatenate((vzpX[ixAZ], vX[ixU]))
            vXErr = np.concatenate((vzpXErr[ixAZ], vXErr[ixU]))

            vY = np.concatenate((vzpY[ixAZ], vY[ixU]))
            vYErr = np.concatenate((vzpYErr[ixAZ], vYErr[ixU]))

            vZ = np.concatenate((vzpZ[ixAZ], vZ[ixU]))
            vZErr = np.concatenate((vzpZErr[ixAZ], vZErr[ixU]))

            vRxy = np.concatenate((vzpRxy[ixAZ], vRxy[ixU]))

            vsp = np.concatenate((ixzp[ixAZ], vsp[ixU]))
            vTg = vsp

            ix = np.where(vX > 0)
        elif (zps == 3):
            ixAZ = np.where(vzpX > 0)
            ixU = np.where((vsp != 0) & (vsp != 5))
            print(len(vzpX[ixAZ]), len(vX[ixU]))

            vX = np.concatenate((vzpX[ixAZ], vX[ixU]))
            vXErr = np.concatenate((vzpXErr[ixAZ], vXErr[ixU]))

            vY = np.concatenate((vzpY[ixAZ], vY[ixU]))
            vYErr = np.concatenate((vzpYErr[ixAZ], vYErr[ixU]))

            vZ = np.concatenate((vzpZ[ixAZ], vZ[ixU]))
            vZErr = np.concatenate((vzpZErr[ixAZ], vZErr[ixU]))

            vRxy = np.concatenate((vzpRxy[ixAZ], vRxy[ixU]))

            vsp = np.concatenate((ixzp[ixAZ], vsp[ixU]))
            vTg = vsp

            ix = np.where(vX > 0)
        elif (zps == 4):
            ixAZ = np.where(vzpZ > 0)
            ixU = np.where(vsp != 0)
            print(len(vzpX[ixAZ]), len(vX[ixU]))

            vX = np.concatenate((vzpX[ixAZ], vX[ixU]))
            vXErr = np.concatenate((vzpXErr[ixAZ], vXErr[ixU]))

            vY = np.concatenate((vzpY[ixAZ], vY[ixU]))
            vYErr = np.concatenate((vzpYErr[ixAZ], vYErr[ixU]))

            vZ = np.concatenate((vzpZ[ixAZ], vZ[ixU]))
            vZErr = np.concatenate((vzpZErr[ixAZ], vZErr[ixU]))

            # vZUErr = np.concatenate((vzpZUErr[ixAZ], vZErr[ixU]))
            # vZDErr = np.concatenate((vzpZDErr[ixAZ], vZErr[ixU]))

            vRxy = np.concatenate((vzpRxy[ixAZ], vRxy[ixU]))

            vsp = np.concatenate((ixzp[ixAZ], vsp[ixU]))
            vTg = vsp

            ix = np.where(vX > 0)
        elif (zps == 5):
            ixAZ = np.where(vzpZ > 0)
            ixU = np.where(vsp != 0)
            ixJ = np.where(vJZ > 0)
            ixL = np.where(vLZ > 0)
            print(len(vzpX[ixAZ]), len(vX[ixU]), len(vJX[ixJ]), len(vLX[ixL]))

            vX = np.concatenate((vzpX[ixAZ], vX[ixU], vJX[ixJ], vLX[ixL]))
            vXErr = np.concatenate((vzpXErr[ixAZ], vXErr[ixU], vJXErr[ixJ], vLXErr[ixL]))

            vY = np.concatenate((vzpY[ixAZ], vY[ixU], vJY[ixJ], vLY[ixL]))
            vYErr = np.concatenate((vzpYErr[ixAZ], vYErr[ixU], vJYErr[ixJ], vLYErr[ixL]))

            vZ = np.concatenate((vzpZ[ixAZ], vZ[ixU], vJZ[ixJ], vLZ[ixL]))
            vZErr = np.concatenate((vzpZErr[ixAZ], vZErr[ixU], vJZErr[ixJ], vLZErr[ixL]))

            # vZUErr = np.concatenate((vzpZUErr[ixAZ], vZErr[ixU]))
            # vZDErr = np.concatenate((vzpZDErr[ixAZ], vZErr[ixU]))

            vRxy = np.concatenate((vzpRxy[ixAZ], vRxy[ixU], vJRxy[ixJ], vLRxy[ixL]))

            vsp = np.concatenate((ixzp[ixAZ], vsp[ixU], ixJs[ixJ], ixLs[ixL]))
            vTg = vsp

            ix = np.where(vX > 0)
    else:
        print('spl option error')
        return

    # return

    vTg = vTg[ix]
    vX = vX[ix]
    vY = vY[ix]
    vZ = vZ[ix]
    vXErr = vXErr[ix]
    vYErr = vYErr[ix]
    vZUErr = vZErr[ix]
    vZDErr = vZErr[ix]
    vZErr = vZErr[ix]
    # vZUErr = vZUErr[ix]
    # vZDErr = vZDErr[ix]
    vRxy = vRxy[ix]
    vsp = vsp[ix]

    Muo = 2.5*(beta*vX + alpha) - 2.5*vY - 100.195
    MuoErr = 2.5*np.sqrt((vYErr)**2 + beta**2*(vXErr)**2
                 + betaErr**2*vX**2
                 + alphaErr**2
                 # - 2.0*beta*vRxy*vXErr*vYErr
                 # + 0.22**2
                 )
    # n = 100000
    # aalpha = np.random.normal(alpha, alphaErr, n)
    # abeta  = np.random.normal(beta, betaErr, n)
    # vMuo = vX*0
    # vMuoErr = vX*0
    # for i in range(0, len(vX)):
        # aX = np.random.normal(vX[i], vXErr[i], n)
        # aY = np.random.normal(vY[i], vYErr[i], n)
        # # S = 0.22/beta
        # S = 0.22/5.0
        # aYN = np.random.normal(0.0, S, n)
        # aXN = np.random.normal(0.0, S, n)
        # aMuo = 2.5*(abeta*(aX+aXN) + aalpha - aY) - 100.195
        # vMuo[i] = np.mean(aMuo)
        # vMuoErr[i] = np.std(aMuo)
        # # print np.mean(aMuo), Muo[i]
        # # print np.std(aMuo), MuoErr[i], np.std(aMuo) - MuoErr[i]

    # ixf = np.where((vXErr <= 0.1) & (vYErr <= 0.2))
    # print(len(vX[ixf]))
    # Mdata = [vX[ixf], vY[ixf], vZ[ixf], vXErr[ixf], vYErr[ixf], vZErr[ixf], vRxy[ixf], Muo[ixf], MuoErr[ixf]]

    Mdata = np.array([vX, vY, vZ, vXErr, vYErr, vZErr, vRxy, Muo, MuoErr, vTg])
    MdataA = np.array([vX, vY, vZ, vXErr, vYErr, vZErr, vZUErr, vZDErr, vRxy, Muo, MuoErr, vTg])

    # df = pd.DataFrame(np.transpose(Mdata))
    # print(df)
    # dfs = df.sort_values(by=[2], ascending=True)
    # print(dfs)

    # path = dpath+'results/Union2020Sv1.csv'
    # dfs.to_csv(path, columns = [9, 2, 0, 3, 1, 4], index=False)

    # return 

    for i in range(0, len(vX)):
        print(vTg[i], ',', vZ[i],',', vZErr[i],',', vX[i],',', vXErr[i],',', vY[i],',', vYErr[i])#,',', vsp[i])

    print(len(vTg))
    # hd(cpath, dpath, fend)
    # return

    path = dpath+'results/HLSig'+fend+'.pdf'
    fig = plt.subplots(1)
    hist(vX, bins='knuth', histtype='stepfilled',
        alpha=0.2)
    plt.xlabel("$log \sigma$", fontsize=18)
    plt.ylabel("$n$", fontsize=18)
    plt.savefig(path)

    path = dpath+'results/HLESig'+fend+'.pdf'
    fig = plt.subplots(1)
    hist(vXErr, bins='knuth', histtype='stepfilled',
        alpha=0.2)
    plt.xlabel("$\epsilon log \sigma$", fontsize=18)
    plt.ylabel("$n$", fontsize=18)
    plt.savefig(path)

    path = dpath+'results/HLfx'+fend+'.pdf'
    fig = plt.subplots(1)
    hist(vY, bins='knuth', histtype='stepfilled',
        alpha=0.2)
    plt.xlabel("$log f$", fontsize=18)
    plt.ylabel("$n$", fontsize=18)
    plt.savefig(path)

    path = dpath+'results/HLEfx'+fend+'.pdf'
    fig = plt.subplots(1)
    hist(vYErr, bins='knuth', histtype='stepfilled',
        alpha=0.2)
    plt.xlabel("$\epsilon log f$", fontsize=18)
    plt.ylabel("$n$", fontsize=18)
    plt.savefig(path)


    # ixZ = np.where(vZ <= 5.0)
    ixZ = np.where(vZ > 0.0)
    path = dpath+'results/HZ'+fend+'.pdf'
    fig = plt.subplots(1)
    hist(vZ[ixZ], bins='knuth', histtype='stepfilled',
        alpha=0.2)
    plt.xlabel("$z$", fontsize=18)
    plt.ylabel("$n(z)$", fontsize=18)
    plt.savefig(path)
    plt.clf()

    # return

    ctag = str(fg1)
    # Cases
    if fg1 == 0:
        case0(cpath, dpath, fend, vbs, sps, prs, ctag)
    elif fg1 == 100:
        case100(cpath, dpath, fend, vbs, sps, prs, ctag)
    elif fg1 == 1:
        case1(cpath, dpath, fend, vbs, sps, prs, ctag)
    elif fg1 == 3:
        case3(cpath, dpath, fend, vbs, sps, prs, ctag)
    elif fg1 == 30:
        case30(cpath, dpath, fend, vbs, sps, prs, ctag)
    elif fg1 == 31:
        case31(cpath, dpath, fend, vbs, sps, prs, ctag)
    elif fg1 == 4:
        case4(cpath, dpath, fend, vbs, sps, prs, ctag)
    elif fg1 == 41:
        case41(cpath, dpath, fend, vbs, sps, prs, ctag)
    elif fg1 == 42:
        case42(cpath, dpath, fend, vbs, sps, prs, ctag)
    elif fg1 == 5:
        case5(cpath, dpath, fend, vbs, sps, prs, ctag)
    elif fg1 == 51:
        case51(cpath, dpath, fend, vbs, sps, prs, ctag)
    elif fg1 == 52:
        case52(cpath, dpath, fend, vbs, sps, prs, ctag)
    elif fg1 == 53:
        case53(cpath, dpath, fend, vbs, sps, prs, ctag)
    elif fg1 == 54:
        case54(cpath, dpath, fend, vbs, sps, prs, ctag)
    elif fg1 == 6:
        case6(cpath, dpath, fend, vbs, sps, prs, ctag)
    elif fg1 == 61:
        case61(cpath, dpath, fend, vbs, sps, prs, ctag)
    elif fg1 == 7:
        case7(cpath, dpath, fend, vbs, sps, prs, ctag)
    elif fg1 == 71:
        case71(cpath, dpath, fend, vbs, sps, prs, ctag)
    elif fg1 == 72:
        case72(cpath, dpath, fend, vbs, sps, prs, ctag)
    elif fg1 == 73:
        case73(cpath, dpath, fend, vbs, sps, prs, ctag)

    print(datetime.datetime.now())
    return
