#  ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#  :Name:     mcsnev11.py
#  :Purpose:  MCMC for SNe Ia Pantheon compilation
#  :Author:   Ricardo Chavez -- Apr 10th, 2020 -- Morelia, Mich
#  :Modified:
#  ------------------------------------------------------------------------------
import numpy as np
from pymultinest.solve import solve
from astropy.visualization import hist
from astropy import constants as const
from astropy.table import Table
from astropy.cosmology import FlatwCDM
from astropy.cosmology import Flatw0waCDM
import matplotlib.pyplot as plt
import corner
from getdist import plots, MCSamples
import scipy.stats as st
import datetime
# import pyfits
import glob

c = const.c.to('km/s').value


def ploter(dpath, fend, result, parameters, parametersc, tag, ctag):
    x, z, xerr, zerr, cov = Mdata
    fchsq = 'chsq' + str(ctag)
    n_params = len(parameters)
    nobs = len(x)
    print(fchsq, nobs)

    # Plot getdist
    path = dpath+'results/GDtr'+fend+'.pdf'
    GDsamples = MCSamples(samples=result['samples'], names=parameters,
                          labels=parameters, name_tag=tag)
    # GDsamples.updateSettings({'contours': [0.68, 0.95, 0.99]})

    g = plots.getSubplotPlotter()
    g.settings.num_plot_contours = 2
    g.triangle_plot(GDsamples, filled=True, title_limit=1)
    g.export(path)

    t = GDsamples.getTable(limit=1).tableTex()
    theta = GDsamples.getMeans()

    chsq = globals()[fchsq](theta)[0]
    res = globals()[fchsq](theta)[1]
    
    # Cov Matix and FoM
    FoM = 1./np.sqrt(np.linalg.det(GDsamples.cov()))
    print('FoM = ', FoM)
    
    # Plot Corner
    path = dpath+'results/tr'+fend+'.pdf'
    fig = corner.corner(result['samples'],
                        labels=parametersc,
                        # truths= vMean,
                        # quantiles = [0.16, 0.84],
                        # range = [(0.1, 0.6), (-2.0, -0.5)],
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

    path = dpath+'results/Hnres'+fend+'.pdf'
    fig = plt.subplots(1)
    hist(nres, bins='knuth', histtype='stepfilled',
         alpha=0.2)
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
    return


def lnprior1(cube):
    cube[0] = cube[0]
    cube[1] = cube[1]*2. - 2.
    return cube


def lnlike1(cube):
    Om, w0 = cube[0], cube[1]
    theta = [Om, w0]
    xsq = chsq1(theta)[0]
    return -0.5*xsq


def chsq1(theta):
    Om, w0 = theta
    x, z, xerr, zerr, cov = Mdata

    MB = 0.0

    cosmo = FlatwCDM(H0=100.0, Om0=Om, w0=w0)
    #--------------------------------------------------------------------------
    DL = (cosmo.luminosity_distance(z).value*100.0)/c
    Mum = 5.0*np.log10(DL)

    Mu = x - MB

    R = (Mu - Mum)
    # resI = resD*0.0+1.0
    W = 1.0/(xerr**2)

    # Cmu = np.asarray(cov)*0.0
    # Cmu[np.diag_indices_from(Cmu)] += xerr**2
    # ICmu = np.linalg.inv(Cmu)

    # xsqA = np.dot(np.dot(np.transpose(resD), ICmu), resD)
    # xsqB = np.dot(np.dot(np.transpose(np.sqrt(resD)), ICmu), np.sqrt(resD))
    # xsqC = np.dot(np.dot(np.transpose(resI), ICmu), resI)
    # xsq = xsqA - xsqB**2/xsqC + np.log(xsqC/(2.0*np.pi))

    xsqA = np.sum(R**2*W)
    xsqB = np.sum(R*W)
    xsqC = np.sum(W)

    xsq = xsqA - xsqB**2/xsqC
    xsq2 = xsq - np.sum(np.log(W))

    return (xsq, R)


def case1(cpath, dpath, fend, vbs, sps, prs, ctag):
    parameters = [r"\Omega_m", r"w_0"]
    parametersc = [r"$\Omega_m$", r"$w_0$"]
    n_params = len(parameters)

    prefix = cpath + fend

    result = solve(LogLikelihood=lnlike1, Prior=lnprior1,
        n_dims=n_params, outputfiles_basename=prefix, verbose=vbs,
        n_live_points = sps, resume = prs, init_MPI = False,
        sampling_efficiency = 0.8)

    tag = 'SNIa'
    ploter(dpath, fend, result, parameters, parametersc, tag, ctag)
    return


def lnprior2(cube):
    cube[0] = cube[0]
    cube[1] = cube[1]*2. - 2.
    cube[2] = cube[2]*3. - 2.
    return cube


def lnlike2(cube):
    Om, w0, w1 = cube[0], cube[1], cube[2]
    theta = [Om, w0, w1]
    xsq = chsq2(theta)[0]
    return -0.5*xsq


def chsq2(theta):
    Om, w0, w1 = theta
    x, z, xerr, zerr, cov = Mdata

    MB = 0.0

    cosmo = Flatw0waCDM(H0=100.0, Om0=Om, w0=w0, wa=w1)
    #--------------------------------------------------------------------------
    DL = (cosmo.luminosity_distance(z).value*100.0)/c
    Mum = 5.0*np.log10(DL)

    Mu = x - MB

    R = (Mu - Mum)
    # resI = resD*0.0+1.0
    W = 1.0/(xerr**2)

    # Cmu = np.asarray(cov)*0.0
    # Cmu[np.diag_indices_from(Cmu)] += xerr**2
    # ICmu = np.linalg.inv(Cmu)

    # xsqA = np.dot(np.dot(np.transpose(resD), ICmu), resD)
    # xsqB = np.dot(np.dot(np.transpose(np.sqrt(resD)), ICmu), np.sqrt(resD))
    # xsqC = np.dot(np.dot(np.transpose(resI), ICmu), resI)
    # xsq = xsqA - xsqB**2/xsqC + np.log(xsqC/(2.0*np.pi))

    xsqA = np.sum(R**2*W)
    xsqB = np.sum(R*W)
    xsqC = np.sum(W)

    xsq = xsqA - xsqB**2/xsqC
    xsq2 = xsq - np.sum(np.log(W))

    return (xsq, R)


def case2(cpath, dpath, fend, vbs, sps, prs, ctag):
    parameters = [r"\Omega_m", r"w_0", r"w_a"]
    parametersc = [r"$\Omega_m$", r"$w_0$", r"$w_a$"]
    n_params = len(parameters)

    prefix = cpath + fend

    result = solve(LogLikelihood=lnlike2, Prior=lnprior2,
        n_dims=n_params, outputfiles_basename=prefix, verbose=vbs,
        n_live_points = sps, resume = prs, init_MPI = False,
        sampling_efficiency = 0.8)

    tag = 'SNIa'
    ploter(dpath, fend, result, parameters, parametersc, tag, ctag)
    return


def snpthb(ve, dpath):
    # Reading SNe Data
    Tpath = dpath+'indat/pantheonbin.txt'
    t = Table.read(Tpath, format='ascii', comment ='#')

    ix = t['col1']

    x = t['col5']
    xErr = t['col6']

    z = t['col3']
    zErr = t['col4']

    TCpath = dpath+'indat/pantheoncovbin.txt'
    files = open(TCpath)
    lines = files.readlines()
    files.close()

    data = []
    for line in lines:
        if line.startswith('#'): continue
        c=line.rstrip().replace('INDEF','Nan').split()
        data.append([float(x) for x in c])

    array = []
    cnt = 0
    line = []
    for i in range(0, len(data)):
        cnt += 1
        # print(cnt)
        line.append(data[i][0])
        if cnt % 40 == 0:
            cnt = 0
            if len(line) > 0:
                array.append(line)
            line = []

    # print(array)

    # fpath=dpath + 'results/covbin.pdf'
    # fig, ax1 = plt.subplots(1,1)
    # imgplot = plt.imshow(array, cmap=plt.cm.viridis, vmin=-0.001, vmax=0.001)
    # ax1.set_xticklabels(['', 0.01,'',0.1,'',.50,'',1.0,'',2.0])
    # ax1.set_yticklabels(['', 0.01,'',0.1,'',.50,'',1.0,'',2.0])
    # ax1.set_xlabel('z')
    # ax1.set_ylabel('z')
    # plt.colorbar()
    # plt.savefig(fpath)
    return x, xErr, z, zErr, array


def snpth(ve, dpath):
    # Reading SNe Data
    Tpath = dpath+'indat/pantheon.txt'
    t = Table.read(Tpath, format='ascii', comment ='#')

    ix = t['name']

    x = t['mb']
    xErr = t['dmb']

    z = t['zcmb']
    zErr = t['dz']

    TCpath = dpath+'indat/pantheoncov.txt'
    files = open(TCpath)
    lines = files.readlines()
    files.close()

    data = []
    for line in lines:
        if line.startswith('#'): continue
        c=line.rstrip().replace('INDEF','Nan').split()
        data.append([float(x) for x in c])

    array = []
    cnt = 0
    line = []
    for i in range(0, len(data)):
        cnt += 1
        # print(cnt)
        line.append(data[i][0])
        if cnt % 1048 == 0:
            cnt = 0
            if len(line) > 0:
                array.append(line)
            line = []

    # print(array)

    # fpath=dpath + 'results/covbin.pdf'
    # fig, ax1 = plt.subplots(1,1)
    # imgplot = plt.imshow(array, cmap=plt.cm.viridis, vmin=-0.001, vmax=0.001)
    # ax1.set_xticklabels(['', 0.01,'',0.1,'',.50,'',1.0,'',2.0])
    # ax1.set_yticklabels(['', 0.01,'',0.1,'',.50,'',1.0,'',2.0])
    # ax1.set_xlabel('z')
    # ax1.set_ylabel('z')
    # plt.colorbar()
    # plt.savefig(fpath)
    return x, xErr, z, zErr, array


def mcsnev11(ve, dpath, cpath, clc=0, opt=0, sps=900, prs=0, vbs=0):
    print('+++++++++++++++++++++++++++++++++++++++++++')
    print('mcsnev11: ')
    print('+++++++++++++++++++++++++++++++++++++++++++')

    #============= Parameters ======================================
    global Mdata
    global start

    fg0 = clc
    fg1 = opt

    fend ='mcsv11_'+str(ve)+'_'+ str(fg0)+'_'+ str(fg1) + '_stat'
    print(fend)
    print(datetime.datetime.now())
    start = datetime.datetime.now()
    #===============================================================
    if fg0 == 0:
        x, xErr, z, zErr, cov = snpthb(ve, dpath)
    elif fg0 == 1:
        x, xErr, z, zErr, cov = snpth(ve, dpath)

    Mdata = x, z, xErr, zErr, cov

    fpath=dpath + 'results/cov.pdf'
    fig, ax1 = plt.subplots(1,1)
    imgplot = plt.imshow(cov, cmap=plt.cm.viridis)#, vmin=-0.001, vmax=0.001)
    # ax1.set_xticklabels(['', 0.01,'',0.1,'',.50,'',1.0,'',2.0])
    # ax1.set_yticklabels(['', 0.01,'',0.1,'',.50,'',1.0,'',2.0])
    ax1.set_xlabel('z')
    ax1.set_ylabel('z')
    plt.colorbar()
    plt.savefig(fpath)

    exit() 

    ctag = str(fg1)
    # Cases
    if fg1 == 0:
        case0(cpath, dpath, fend, vbs, sps, prs, ctag)
    elif fg1 == 1:
        case1(cpath, dpath, fend, vbs, sps, prs, ctag)
    elif fg1 == 11:
        case11(cpath, dpath, fend, vbs, sps, prs, ctag)
    elif fg1 == 2:
        case2(cpath, dpath, fend, vbs, sps, prs, ctag)
    elif fg1 == 21:
        case21(cpath, dpath, fend, vbs, sps, prs, ctag)
    elif fg1 == 22:
        case22(cpath, dpath, fend, vbs, sps, prs, ctag)
    elif fg1 == 3:
        case3(cpath, dpath, fend, vbs, sps, prs, ctag)
    elif fg1 == 31:
        case31(cpath, dpath, fend, vbs, sps, prs, ctag)
    elif fg1 == 32:
        case32(cpath, dpath, fend, vbs, sps, prs, ctag)
    elif fg1 == 33:
        case33(cpath, dpath, fend, vbs, sps, prs, ctag)

    print(datetime.datetime.now())
    return
