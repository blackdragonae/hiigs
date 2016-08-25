#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# :Name:     mcsnexp.py                  
# :Purpose:  MCMC for SNe Ia JLA compilation     
# :Author:   Ricardo Chavez -- April 12th, 2016 -- Cambridge, England  
# :Modified:                                           
#------------------------------------------------------------------------------ 
import numpy as np
from cosmoc import distL
from astropy import constants as const

def lnlike(theta, x, y, w, m, z, xerr, yerr, werr, merr, zerr):
    alpha, beta, MB1, DM = theta
    
    if (0.0 <= alpha <= 1.0 and 0.0 <= beta <= 6.0 and -25.0 <= MB1 <= -15.0
      and -1.0 <= DM <= 0.0):
        h0 = 0.7
        Or = 4.153e-5 * h0**(-2)
        Om = 0.289
        Ok = 0.0
        w0 = -1.0
        w1 = 0.0
        
        Mum = z*0.0
        MB = z*0.0
        for k in range(0, len(z)):
            Mum[k] = 5*np.log10(distL(z[k], h0, Or, Om, Ok, w0, w1)) + 25
            if m[k] < 10:
                MB[k] = MB1
            else:
                MB[k] = MB1 + DM
                
        Mu = x - (MB - alpha*y + beta*w)
        
        inv_sigma2 = 1.0/(xerr**2 + alpha**2*yerr**2 + beta**2*werr**2)
    
        res = (Mu - Mum)
        xsq = np.sum((Mu-Mum)**2*inv_sigma2)
        llq = -0.5*(xsq)
        
        llq2 = -0.5*(np.sum((Mu-Mum)**2*inv_sigma2 - np.log(inv_sigma2)))
        
        print theta, xsq
        
        return llq2, xsq, res
    else:
        return -np.inf, np.inf, np.inf   
def lnprior(theta):
    alpha, beta, MB1, DM = theta
    if (0.0 <= alpha <= 1.0 and 0.0 <= beta <= 6.0 and -25.0 <= MB1 <= -15.0
      and -1.0 <= DM <= 0.0):
        return 0.0
    return -np.inf
def lnprob(theta, x, y, w, m, z, xerr, yerr, werr, merr, zerr):
    lp = lnprior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(theta, x, y, w, m, z, xerr, yerr, werr, merr, zerr)[0]
#------------------------------------------------------------------------------
def lnlike1(theta, x, y, w, m, z, xerr, yerr, werr, merr, zerr):
    alpha, beta, MB1, DM, Om = theta
    
    if (0.0 <= alpha <= 1.0 and 0.0 <= beta <= 6.0 and -25.0 <= MB1 <= -15.0
      and -1.0 <= DM <= 0.0 and 0.0 <= Om <= 1.0):
        c = const.c.to('km/s').value
        h0 = 0.7
        Or = 4.153e-5 * h0**(-2)
        Ok = 0.0
        w0 = -1.0
        w1 = 0.0
    
        Mum = z*0.0
        MB = z*0.0
        for k in range(0, len(z)):
            Mum[k] = 5*np.log10(distL(z[k], h0, Or, Om, Ok, w0, w1)) + 25
            if m[k] < 10:
                MB[k] = MB1
            else:
                MB[k] = MB1 + DM
                
        Mu = x - (MB - alpha*y + beta*w)
        
        inv_sigma2 = 1.0/(xerr**2 + alpha**2*yerr**2 + beta**2*werr**2)
        
        res = (Mu - Mum)
        xsq = np.sum((Mu-Mum)**2*inv_sigma2)
        llq = -0.5*(xsq)
        
        print theta, xsq
        
        llq2 = -0.5*(np.sum((Mu-Mum)**2*inv_sigma2 - np.log(inv_sigma2)))
        
        return llq2, xsq, res
    else:
        return -np.inf, np.inf, np.inf
def lnprior1(theta):
    alpha, beta, MB1, DM, Om = theta
    if (0.0 <= alpha <= 1.0 and 0.0 <= beta <= 6.0 and -25.0 <= MB1 <= -15.0
      and -1.0 <= DM <= 0.0 and 0.0 <= Om <= 1.0):
        return 0.0
    return -np.inf
def lnprob1(theta, x, y, w, m, z, xerr, yerr, werr, merr, zerr):
    lp = lnprior1(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike1(theta, x, y, w, m, z, xerr, yerr, werr, merr, zerr)[0]
#------------------------------------------------------------------------------
def lnlike2(theta, x, y, z, xerr, yerr, zerr):
    alpha, beta, h0, Om = theta
    
    if (0.0 <= beta <= 10.0 and 20.0 <= alpha <= 40.0 and 0.5 <= h0 <= 1.0 
       and 0.0 <= Om <= 1.0):
        c = const.c.to('km/s').value
        Or = 4.153e-5 * h0**(-2)
        Ok = 0.0
        w0 = -1.0
        w1 = 0.0
    
        Mum = z*0.0
        MumErr = z*0.0
        for k in range(0, len(z)):
            if z[k] > 10:
                Mum[k] = z[k]
                MumErr[k] = zerr[k]
            else:
                Mum[k] = 5*np.log10(distL(z[k], h0, Or, Om, Ok, w0, w1)) + 25
                MumErr[k] = (5.0/np.log(10)) * (zerr[k]/z[k])
                
        Mu = 2.5*(beta*x + alpha) - 2.5*y - 100.195
        MuErr = 2.5*np.sqrt(yerr**2 + beta**2*xerr**2)
    
        inv_sigma2 = 1.0/(MuErr**2 + MumErr**2)
    
        xsq = np.sum((Mu-Mum)**2*inv_sigma2)
        print theta, xsq
        return -0.5*(xsq)
    else:
        return -np.inf

def lnprior2(theta):
    alpha, beta, h0, Om = theta
    if (0.0 <= beta <= 10.0 and 20.0 <= alpha <= 40.0 and 0.5 <= h0 <= 1.0 
       and 0.0 <= Om <= 1.0):
        return 0.0
    return -np.inf

def lnprob2(theta, x, y, z, xerr, yerr, zerr):
    lp = lnprior2(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike2(theta, x, y, z, xerr, yerr, zerr)
#------------------------------------------------------------------------------
def lnlike3(theta, x, y, z, xerr, yerr, zerr):
    alpha, beta, h0, Om, w0 = theta
    
    if (0.0 <= beta <= 10.0 and 20.0 <= alpha <= 40.0 and 0.5 <= h0 <= 1.0 
       and 0.0 <= Om <= 1.0 and -2.0 <= w0 <= 0.0):
        c = const.c.to('km/s').value
        Or = 4.153e-5 * h0**(-2)
        Ok = 0.0
        w1 = 0.0
    
        Mum = z*0.0
        MumErr = z*0.0
        for k in range(0, len(z)):
            if z[k] > 10:
                Mum[k] = z[k]
                MumErr[k] = zerr[k]
            else:
                Mum[k] = 5*np.log10(distL(z[k], h0, Or, Om, Ok, w0, w1)) + 25
                MumErr[k] = (5.0/np.log(10)) * (zerr[k]/z[k])
                
        Mu = 2.5*(beta*x + alpha) - 2.5*y - 100.195
        MuErr = 2.5*np.sqrt(yerr**2 + beta**2*xerr**2)
    
        inv_sigma2 = 1.0/(MuErr**2 + MumErr**2)
    
        xsq = np.sum((Mu-Mum)**2*inv_sigma2)
        print theta, xsq
        return -0.5*(xsq)
    else:
        return -np.inf

def lnprior3(theta):
    alpha, beta, h0, Om, w0 = theta
    if (0.0 <= beta <= 10.0 and 20.0 <= alpha <= 40.0 and 0.5 <= h0 <= 1.0 
       and 0.0 <= Om <= 1.0 and -2.0 <= w0 <= 0.0):
        return 0.0
    return -np.inf

def lnprob3(theta, x, y, z, xerr, yerr, zerr):
    lp = lnprior3(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike3(theta, x, y, z, xerr, yerr, zerr)
#------------------------------------------------------------------------------
def lnlike4(theta, x, y, z, xerr, yerr, zerr):
    alpha, beta, h0, Om, w0, w1 = theta
    
    if (0.0 <= beta <= 10.0 and 20.0 <= alpha <= 40.0 and 0.5 <= h0 <= 1.0 
       and 0.0 <= Om <= 1.0 and -2.0 <= w0 <= 0.0  and -1.0 <= w1 <= 1.0):
        c = const.c.to('km/s').value
        Or = 4.153e-5 * h0**(-2)
        Ok = 0.0
    
        Mum = z*0.0
        MumErr = z*0.0
        for k in range(0, len(z)):
            if z[k] > 10:
                Mum[k] = z[k]
                MumErr[k] = zerr[k]
            else:
                Mum[k] = 5*np.log10(distL(z[k], h0, Or, Om, Ok, w0, w1)) + 25
                MumErr[k] = (5.0/np.log(10)) * (zerr[k]/z[k])
                
        Mu = 2.5*(beta*x + alpha) - 2.5*y - 100.195
        MuErr = 2.5*np.sqrt(yerr**2 + beta**2*xerr**2)
    
        inv_sigma2 = 1.0/(MuErr**2 + MumErr**2)
    
        xsq = np.sum((Mu-Mum)**2*inv_sigma2)
        print theta, xsq
        return -0.5*(xsq)
    else:
        return -np.inf

def lnprior4(theta):
    alpha, beta, h0, Om, w0, w1 = theta
    if (0.0 <= beta <= 10.0 and 20.0 <= alpha <= 40.0 and 0.5 <= h0 <= 1.0 
       and 0.0 <= Om <= 1.0 and  -2.0 <= w0 <= 0.0 and -1.0 <= w1 <= 1.0):
        return 0.0
    return -np.inf

def lnprob4(theta, x, y, z, xerr, yerr, zerr):
    lp = lnprior4(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike4(theta, x, y, z, xerr, yerr, zerr)
#------------------------------------------------------------------------------
def mcsnexp(ve, dpath, clc = 0 , opt = 0):
    from astropy.table import Table
    import astropy.units as u
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import scipy.optimize as op
    import corner
    import emcee
    import scipy.stats as st
    
    import sys
    from emcee.utils import MPIPool
    
    from hubblediagram import hubblediagram
    
    print '+++++++++++++++++++++++++++++++++++++++++++'
    print 'MCSNEXP: '
    print '+++++++++++++++++++++++++++++++++++++++++++'

    #===============================================================       
    #============= Parameters ======================================
    #===============================================================  
    fg0 = clc
    fg1 = opt
    
    nwk = 300
    sps = 900
    tds = 4
        
    fend ='_mcsnexp'+ str(fg1)#+'z3'
    print fend
    
    #===============================================================     
    #============= Main Body =======================================   
    #===============================================================
    # Reading SNE JLA data  
    TpathJLA = dpath+'indat/JLAm.csv'
    tJLA = Table.read(TpathJLA, format='csv', delimiter = ';', comment ='#')
    
    x = tJLA['mb']
    xErr = tJLA['e_mb']
    
    y = tJLA['x1']
    yErr = tJLA['e_x1']
    
    w = tJLA['c']
    wErr = tJLA['e_c']
    
    m = tJLA['logMst']
    mErr = tJLA['e_logMst']
    
    z = tJLA['zcmb']
    zErr = z*0.0
    
    # First guess
    alpha_i = 0.10
    beta_i = 3.0
    MB1_i = -20.0
    DM_i = 0.0
    h0_i = 0.6
    Om_i = 0.2
    Ok_i = 0.05
    w0_i = -1.0
    w1_i = 0.0
    
    # Cases
    if fg1 == 0:
        if fg0 == 1:
            # Maximum likelihood estimation 
            nll = lambda *args: -lnlike(*args)[0]
            result = op.minimize(nll, [alpha_i, beta_i, MB1_i, DM_i], 
               args=(x, y, w, m, z, xErr, yErr, wErr, mErr, zErr), tol = 1e-3)
            
            # MCMC
            ndim, nwalkers = 4, nwk
            pos = [result["x"] + 1e-5*np.random.randn(ndim) 
               for i in range(nwalkers)]
        
            sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, 
                    args=(x, y, w, m, z, xErr, yErr, wErr, mErr, zErr), 
                    threads = tds)
                    
            pos, prob, state = sampler.run_mcmc(pos, 100)
            sampler.reset()
            
            sampler.run_mcmc(pos, sps, rstate0=state)  

            samples = sampler.chain[:, 50:, :].reshape((-1, ndim))
            
            np.save(dpath+'temp/MCMCSNE'+fend+'.npy', samples)
        else:
            samples = np.load(dpath+'temp/MCMCSNE'+fend+'.npy')
        
        alpha_mcmc, beta_mcmc, MB1_mcmc, DM_mcmc = map(
                                 lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
                                 zip(*np.percentile(samples, [16, 50, 84],
                                                    axis=0)))
        
        # Results                                            
        print alpha_mcmc
        print beta_mcmc
        print MB1_mcmc
        print DM_mcmc
        print len(x)
        
        chsq = lnlike([alpha_mcmc[0], beta_mcmc[0], MB1_mcmc[0], 
           DM_mcmc[0]], 
           x, y, w, m, z, xErr, yErr, wErr, mErr, zErr)[1]
        
        res = lnlike([alpha_mcmc[0], beta_mcmc[0], MB1_mcmc[0], 
           DM_mcmc[0]], 
           x, y, w, m, z, xErr, yErr, wErr, mErr, zErr)[2]
        
        print chsq   
        print chsq/(len(x) - 5)
        
        #Test
        nres = res/np.std(res)
        
        path = dpath+'results/Hnres'+fend+'.png'
        plt.hist(nres, 20)
        plt.savefig(path)
        
        ksD, ksP =  st.kstest(nres, 'norm')
        
        print ksD, ksP
        print np.std(res)
        
        #Log
        f = open(dpath+'results/log'+fend+'.dat', 'w')
        f.write('log'+fend+'.dat\n')
        f.write(str(alpha_mcmc)+'\n')
        f.write(str(beta_mcmc)+'\n')
        f.write(str(MB1_mcmc)+'\n')
        f.write(str(DM_mcmc)+'\n')
        f.write(str(chsq)+'\n')
        f.write(str(len(x))+'\n')
        f.write(str(chsq/(len(x) - 5))+'\n')
        f.write(str(ksD)+','+str(ksP)+'\n')
        f.write(str(np.std(res))+'\n')
        f.close
    
        #Plot
        path = dpath+'results/tr'+fend+'.png'
        fig = corner.corner(samples,
                        labels=["$\\alpha$", "$\\beta$","$M_B^1$", 
                           "$\Delta_M$" ],
                        truths=[alpha_mcmc[0],  beta_mcmc[0], 
                           MB1_mcmc[0], DM_mcmc[0]], 
                        quantiles = [0.16, 0.84],
                        range = [(0.11, 0.18), (2.7,3.7), (-19.12, -18.96), 
                           (-0.12, 0.0)],
                        plot_contours = 'true', 
                        levels = 1.0 - np.exp(
                           -0.5 * np.arange(1.0, 3.1, 1.0) ** 2),
                        smooth = 1.0, 
                        bins = 100,
                        color = 'black')
        plt.savefig(path)
        
    elif fg1 == 1:
        if fg0 == 1:
            # Bayesian Analysis 
            ndim, nwalkers = 5, nwk
            pos = [[alpha_i, beta_i, MB1_i, DM_i, Om_i] + 1e-5*np.random.randn(ndim) 
            for i in range(nwalkers)]
            
            # Initialize the sampler with the chosen specs.
            sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob1, 
            args = (x, y, w, m, z, xErr, yErr, wErr, mErr, zErr), 
               threads = tds)
            
            # Run 100 steps as a burn-in.
            pos, prob, state = sampler.run_mcmc(pos, 100)
               
            # Reset the chain to remove the burn-in samples.
            sampler.reset()
            
            # Starting from the final position in the burn-in chain
            sampler.run_mcmc(pos, sps, rstate0=state)
            
            # Formating result
            samples = sampler.chain[:, 50:, :].reshape((-1, ndim))
            
            np.save(dpath+'temp/MCMCSNE'+fend+'.npy', samples)
        else:
            samples = np.load(dpath+'temp/MCMCSNE'+fend+'.npy')      
    
        alpha_mcmc, beta_mcmc, MB1_mcmc, DM_mcmc, Om_mcmc = map(
                                 lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
                                 zip(*np.percentile(samples, [16, 50, 84],
                                                    axis=0)))
    
        print alpha_mcmc
        print beta_mcmc
        print MB1_mcmc
        print DM_mcmc
        print Om_mcmc
        print len(x)
        
        chsq = lnlike1([alpha_mcmc[0], beta_mcmc[0], MB1_mcmc[0], 
           DM_mcmc[0], Om_mcmc[0]], 
           x, y, w, m, z, xErr, yErr, wErr, mErr, zErr)[1]
        
        res = lnlike1([alpha_mcmc[0], beta_mcmc[0], MB1_mcmc[0], 
           DM_mcmc[0], Om_mcmc[0]], 
           x, y, w, m, z, xErr, yErr, wErr, mErr, zErr)[2]
        
        print chsq   
        print chsq/(len(x) - 5)
        
        #Test
        nres = res/np.std(res)
        
        path = dpath+'results/Hnres'+fend+'.png'
        plt.hist(nres, 20)
        plt.savefig(path)
        
        ksD, ksP =  st.kstest(nres, 'norm')
        
        print ksD, ksP
        print np.std(res)
        
        #Log
        f = open(dpath+'results/log'+fend+'.dat', 'w')
        f.write('log'+fend+'.dat\n')
        f.write(str(alpha_mcmc)+'\n')
        f.write(str(beta_mcmc)+'\n')
        f.write(str(MB1_mcmc)+'\n')
        f.write(str(DM_mcmc)+'\n')
        f.write(str(Om_mcmc)+'\n')
        f.write(str(chsq)+'\n')
        f.write(str(len(x))+'\n')
        f.write(str(chsq/(len(x) - 5))+'\n')
        f.write(str(ksD)+','+str(ksP)+'\n')
        f.write(str(np.std(res))+'\n')
        f.close
        
        # Results
        path = dpath+'results/tr'+fend+'.png'
        fig = corner.corner(samples,
                        labels=["$\\alpha$", "$\\beta$","$M_B^1$", 
                           "$\Delta_M$", "$\Omega_m$"],
                        truths=[alpha_mcmc[0], beta_mcmc[0], MB1_mcmc[0], 
                           DM_mcmc[0], Om_mcmc[0]], 
                        quantiles = [0.16, 0.84],
                        range = [(0.10, 0.16), (2.0,3.5), (-19.12, -18.96), 
                           (-0.12, 0.0), (0.1, 0.4)],
                        plot_contours = 'true', 
                        levels = 1.0 - np.exp(
                           -0.5 * np.arange(1.0, 3.1, 1.0) ** 2),
                        smooth = 1.0, 
                        bins = 100,
                        color = 'black')
        plt.savefig(path)

    elif fg1 == 2:
        #Maximum likelihood estimation 1 h0 = free
        nll = lambda *args: -lnlike2(*args)
        result = op.minimize(nll, [alpha_i, beta_i, h0_i, Om_i], 
           args=(vX, vY, vZ, vXErr, vYErr, vZErr), tol = 1e-4)
        
        # Bayesian Analysis h0 = 0.7
        ndim, nwalkers = 4, 800
        pos = [result["x"] + 1e-5*np.random.randn(ndim) 
           for i in range(nwalkers)]
        
        sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob2, 
                    args=(vX, vY, vZ, vXErr, vYErr, vZErr), 
                    threads=10)
        sampler.run_mcmc(pos, 1600)  

        samples = sampler.chain[:, 50:, :].reshape((-1, ndim))
    
        alpha_mcmc, beta_mcmc, h0_mcmc, Om_mcmc = map(
                                 lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
                                 zip(*np.percentile(samples, [16, 50, 84],
                                                    axis=0)))
    
        print alpha_mcmc
        print beta_mcmc
        print h0_mcmc
        print Om_mcmc
        print len(vX)
        
        print -2.0*lnlike2([alpha_mcmc[0], beta_mcmc[0], h0_mcmc[0], 
           Om_mcmc[0]], vX, vY, vZ, vXErr, vYErr, vZErr)
        
        #Log
        f = open(dpath+'results/log'+fend+'.dat', 'w')
        f.write('log'+fend+'.dat\n')
        f.write(str(alpha_mcmc)+'\n')
        f.write(str(beta_mcmc)+'\n')
        f.write(str(h0_mcmc)+'\n')
        f.write(str(Om_mcmc)+'\n')
        f.write(str(-2.0*lnlike2([alpha_mcmc[0], beta_mcmc[0], 
            h0_mcmc[0], Om_mcmc[0]], vX, vY, vZ, vXErr, vYErr, vZErr))+'\n')
        f.write(str(len(vX))+'\n')
        f.close
        
        # Results
        path = dpath+'results/tr'+fend+'.eps'
        fig = corner.corner(samples,
                        labels=["$\\alpha$", "$\\beta$", "$h_0$", 
                           "$\Omega_m$"],
                        truths=[alpha_mcmc[0],  beta_mcmc[0], h0_mcmc[0], 
                           Om_mcmc[0]], 
                        quantiles = [0.16, 0.84],
                        range = [(32.5,34.0), (4.5,5.5), (0.55, 0.95), 
                           (0.0, 0.65)],
                        plot_contours = 'true',
                        levels = 1.0 - np.exp(
                           -0.5 * np.arange(1.0, 3.1, 1.0) ** 2),
                        smooth = 0.6, 
                        bins = 100,
                        color = 'black')
                        
        plt.savefig(path, dpi = 900)
        
    elif fg1 == 3:
        #Maximum likelihood estimation
        nll = lambda *args: -lnlike3(*args)
        result = op.minimize(nll, [alpha_i, beta_i, h0_i, Om_i, w0_i], 
           args=(vX, vY, vZ, vXErr, vYErr, vZErr), tol = 1e-4)
        
        # Bayesian Analysis
        ndim, nwalkers = 5, 1000
        pos = [result["x"] + 1e-5*np.random.randn(ndim) 
           for i in range(nwalkers)]
        
        sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob3, 
                    args=(vX, vY, vZ, vXErr, vYErr, vZErr), 
                    threads=10)
        sampler.run_mcmc(pos, 2000)  

        samples = sampler.chain[:, 50:, :].reshape((-1, ndim))
    
        alpha_mcmc, beta_mcmc, h0_mcmc, Om_mcmc, w0_mcmc = map(
                                 lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
                                 zip(*np.percentile(samples, [16, 50, 84],
                                                    axis=0)))
    
        print alpha_mcmc
        print beta_mcmc
        print h0_mcmc
        print Om_mcmc
        print w0_mcmc
        print len(vX)
        
        print -2.0*lnlike3([alpha_mcmc[0], beta_mcmc[0], h0_mcmc[0], 
           Om_mcmc[0], w0_mcmc[0]], vX, vY, vZ, vXErr, vYErr, vZErr)
        #Log
        f = open(dpath+'results/log'+fend+'.dat', 'w')
        f.write('log'+fend+'.dat\n')
        f.write(str(alpha_mcmc)+'\n')
        f.write(str(beta_mcmc)+'\n')
        f.write(str(h0_mcmc)+'\n')
        f.write(str(Om_mcmc)+'\n')
        f.write(str(w0_mcmc)+'\n')
        f.write(str(-2.0*lnlike3([alpha_mcmc[0], beta_mcmc[0], 
            h0_mcmc[0], Om_mcmc[0], w0_mcmc[0]], vX, vY, vZ, vXErr, 
            vYErr, vZErr))+'\n')
        f.write(str(len(vX))+'\n')
        f.close
        
        # Results
        path = dpath+'results/tr'+fend+'.eps'
        fig = corner.corner(samples,
                        labels=["$\\alpha$", "$\\beta$", "$h_0$", 
                           "$\Omega_m$", "$w_0$"],
                        truths=[alpha_mcmc[0],  beta_mcmc[0], h0_mcmc[0], 
                           Om_mcmc[0], w0_mcmc[0]], 
                        quantiles = [0.16, 0.84],
                        range = [(32.5,34.0), (4.5,5.5), (0.55, 0.95), 
                           (0.0, 0.65), (-2.0, 0.0)],
                        plot_contours = 'true',
                        levels = 1.0 - np.exp(
                           -0.5 * np.arange(1.0, 3.1, 1.0) ** 2),
                        smooth = 0.6, 
                        bins = 100,
                        color = 'black')
        plt.savefig(path, dpi = 900)

    elif fg1 == 4:
        #Maximum likelihood estimation
        nll = lambda *args: -lnlike4(*args)
        result = op.minimize(nll, [alpha_i, beta_i, h0_i, Om_i, w0_i, 
           w1_i], args=(vX, vY, vZ, vXErr, vYErr, vZErr), tol = 1e-4)
        
        # Bayesian Analysis
        ndim, nwalkers = 6, 1000
        pos = [result["x"] + 1e-5*np.random.randn(ndim) 
           for i in range(nwalkers)]
        
        sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob4, 
                    args=(vX, vY, vZ, vXErr, vYErr, vZErr), 
                    threads=10)
        sampler.run_mcmc(pos, 2000)  

        samples = sampler.chain[:, 50:, :].reshape((-1, ndim))
    
        alpha_mcmc, beta_mcmc, h0_mcmc, Om_mcmc, w0_mcmc, w1_mcmc = map(
                                 lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
                                 zip(*np.percentile(samples, [16, 50, 84],
                                                    axis=0)))
    
        print alpha_mcmc
        print beta_mcmc
        print h0_mcmc
        print Om_mcmc
        print w0_mcmc
        print w1_mcmc
        print len(vX)
        
        print -2.0*lnlike4([alpha_mcmc[0], beta_mcmc[0], h0_mcmc[0], 
           Om_mcmc[0], w0_mcmc[0], w1_mcmc[0]], vX, vY, vZ, vXErr, vYErr, 
           vZErr)
        
        #Log
        f = open(dpath+'results/log'+fend+'.dat', 'w')
        f.write('log'+fend+'.dat\n')
        f.write(str(alpha_mcmc)+'\n')
        f.write(str(beta_mcmc)+'\n')
        f.write(str(h0_mcmc)+'\n')
        f.write(str(Om_mcmc)+'\n')
        f.write(str(w0_mcmc)+'\n')
        f.write(str(w1_mcmc)+'\n')
        f.write(str(-2.0*lnlike4([alpha_mcmc[0], beta_mcmc[0], 
          h0_mcmc[0], Om_mcmc[0], w0_mcmc[0], w1_mcmc[0]], vX, vY, vZ, 
          vXErr, vYErr, vZErr))+'\n')
        f.write(str(len(vX))+'\n')
        f.close
    
        # Results
        path = dpath+'results/tr'+fend+'.eps'
        fig = corner.corner(samples,
                        labels=["$\\alpha$", "$\\beta$", "$h_0$", 
                           "$\Omega_m$", "$w_0$", "$w_1$" ],
                        truths=[alpha_mcmc[0],  beta_mcmc[0], h0_mcmc[0], 
                           Om_mcmc[0], w0_mcmc[0], w1_mcmc[0]], 
                        quantiles = [0.16, 0.84], 
                        range = [(32.5,34.0), (4.5,5.5), (0.55, 0.95), 
                           (0.0, 0.65), (-2.0, 0.0), (-0.5, 0.5)],
                        plot_contours = 'true',
                        levels = 1.0 - np.exp(
                           -0.5 * np.arange(1.0, 3.1, 1.0) ** 2),
                        smooth = 0.6, 
                        bins = 100,
                        color = 'black')
                        
        plt.savefig(path, dpi = 200)
        
    return
    
