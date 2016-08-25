#+                                                                              
# :Name:     mcsne.py                                                
# :Purpose:  MCMC for SNe Ia         
# :Author:   Ricardo Chavez                                           
# :Modified:                                           
#- 
import numpy as np
from cosmoc import distL

c = 2.9979e+5
fgS = 0

def lnlike00(theta, x, y, yerr):
    h0, Om = theta
    if 0.0 <= h0 <= 1.5 and 0.0 <= Om <= 1.0:
        Or = 4.153e-5 * h0**(-2)
        Ok = 0.0
        w0 = -1.0
        w1 = 0.0
        vDLdt = x*0.0
        for k in range(0, len(x)): 
            vDLdt[k] = distL(x[k], h0, Or, Om, Ok, w0, w1)
        model = 5.0*np.log10(vDLdt) + 25.0
        inv_sigma2 = 1.0/(yerr**2)
        #inv_CM = np.linalg.inv(yerr)
        vD = (y-model)
        
        xsq = np.sum((y-model)**2*inv_sigma2)
        #xsl = -0.5*(np.sum((y-model)**2*inv_sigma2))
        
        #xsq = np.dot(np.dot(np.transpose(vD), inv_CM), vD)
        xsl = -0.5*(xsq)
        
        print theta, xsq
        return xsl
    else:
        return -np.inf  
def lnprior0(theta):
    h0, Om = theta
    if 0.0 <= h0 <= 1.5 and 0.0 <= Om <= 1.0:
        return 0.0
    return -np.inf
def lnlike0(theta, x, y, yerr):    
    h0, Om = theta
    
    Or = 4.153e-5 * h0**(-2)
    Ok = 0.0
    w0 = -1.0
    w1 = 0.0
    
    vDLdt = x*0.0
    for k in range(0, len(x)): 
        vDLdt[k] = distL(x[k], h0, Or, Om, Ok, w0, w1)
    model = 5.0*np.log10(vDLdt) + 25.0
    inv_sigma2 = 1.0/(yerr**2)
    #inv_CM = np.linalg.inv(yerr)
    vD = (y-model)
    
    xsq = np.sum((y-model)**2*inv_sigma2)
    #xsl = -0.5*(np.sum((y-model)**2*inv_sigma2))
    
    #xsq = np.dot(np.dot(np.transpose(vD), inv_CM), vD)
    xsl = -0.5*(xsq)
        
    print theta, xsq
    return xsl
def lnprob0(theta, x, y, yerr):
    lp = lnprior0(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike0(theta, x, y, yerr)

def lnlike01(theta, x, y, yerr):
    Om, Ok = theta
    if 0.0 <= Om <= 1.0 and 0.0 <= Ok <= 1.0:
        h0 = 0.7
        Or = 4.153e-5 * h0**(-2)
        w0 = -1.0
        w1 = 0.0
        vDLdt = x*0.0
        for k in range(0, len(x)): 
            vDLdt[k] = distL(x[k], h0, Or, Om, Ok, w0, w1)
        model = 5.0*np.log10(vDLdt) + 25.0
        inv_sigma2 = 1.0/(yerr**2)
        
        xsq = np.sum((y-model)**2*inv_sigma2)
        xsl = -0.5*(np.sum((y-model)**2*inv_sigma2))
        
        print theta, xsq
        return xsl
    else:
        return -np.inf
def lnprior1(theta):
    Om, Ok = theta
    if 0.0 <= Om <= 1.0 and 0.0 <= Ok <= 1.0:
        return 0.0
    return -np.inf
def lnlike1(theta, x, y, yerr):    
    Om, Ok = theta
    
    h0 = 0.7
    Or = 4.153e-5 * h0**(-2)
    w0 = -1.0
    w1 = 0.0
    
    vDLdt = x*0.0
    for k in range(0, len(x)): 
        vDLdt[k] = distL(x[k], h0, Or, Om, Ok, w0, w1)
    model = 5.0*np.log10(vDLdt) + 25.0
    inv_sigma2 = 1.0/(yerr**2)
    
    xsq = np.sum((y-model)**2*inv_sigma2)
    xsl = -0.5*(np.sum((y-model)**2*inv_sigma2))
        
    print theta, xsq
    return xsl
def lnprob1(theta, x, y, yerr):
    lp = lnprior1(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike1(theta, x, y, yerr)

def lnlike02(theta, x, y, yerr):
    Om, w0 = theta
    if 0.0 <= Om <= 1.0 and -2.0 <= w0 <= 0.0:
        h0 = 0.7
        Or = 4.153e-5 * h0**(-2)
        Ok = 0.0
        w1 = 0.0
        vDLdt = x*0.0
        for k in range(0, len(x)): 
            vDLdt[k] = distL(x[k], h0, Or, Om, Ok, w0, w1)
        model = 5.0*np.log10(vDLdt) + 25.0
        
        if fgS == 1:
            inv_CM = np.linalg.inv(yerr)
            vD = (y-model)
            xsq = np.dot(np.dot(np.transpose(vD), inv_CM), vD)
        else:
            inv_sigma2 = 1.0/(yerr**2)
            xsq = np.sum((y-model)**2*inv_sigma2)
        
        xsl = -0.5*(xsq)
        
        #print theta, xsq
        return xsl
    else:
        return -np.inf
def lnprior2(theta):
    Om, w0 = theta
    if 0.0 <= Om <= 1.0 and -2.0 <= w0 <= 0.0:
        return 0.0
    return -np.inf
def lnlike2(theta, x, y, yerr):    
    Om, w0 = theta
    
    h0 = 0.7
    Or = 4.153e-5 * h0**(-2)
    Ok = 0.0
    w1 = 0.0
    
    vDLdt = x*0.0
    for k in range(0, len(x)): 
        vDLdt[k] = distL(x[k], h0, Or, Om, Ok, w0, w1)
    model = 5.0*np.log10(vDLdt) + 25.0
    if fgS == 1:
        inv_CM = np.linalg.inv(yerr)
        vD = (y-model)
        xsq = np.dot(np.dot(np.transpose(vD), inv_CM), vD)
    else:
        inv_sigma2 = 1.0/(yerr**2)
        xsq = np.sum((y-model)**2*inv_sigma2)
    
    xsl = -0.5*(xsq)
        
    #print theta, xsq
    return xsl
def lnprob2(theta, x, y, yerr):
    lp = lnprior2(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike2(theta, x, y, yerr)

def lnlike03(theta, x, y, yerr):
    Om, Ok = theta
    if 0.0 <= Om <= 1.0 and 0.0 <= Ok <= 1.0:
        h0 = 0.7
        Or = 4.153e-5 * h0**(-2)
        w0 = -1.0
        w1 = 0.0
        vDLdt = x*0.0
        for k in range(0, len(x)): 
            vDLdt[k] = distL(x[k], h0, Or, Om, Ok, w0, w1)
        model = 5.0*np.log10(vDLdt) + 25.0
        inv_sigma2 = 1.0/(yerr**2)
        
        xsq = np.sum((y-model)**2*inv_sigma2)
        xsl = -0.5*(np.sum((y-model)**2*inv_sigma2))
        
        #print theta, xsq
        return xsl
    else:
        return -np.inf
def lnprior3(theta):
    Om, Ok = theta
    if 0.0 <= Om <= 1.0 and 0.0 <= Ok <= 1.0:
        return 0.0
    return -np.inf
def lnlike3(theta, x, y, yerr):    
    Om, Ok = theta
    
    h0 = 0.7
    Or = 4.153e-5 * h0**(-2)
    w0 = -1.0
    w1 = 0.0
    
    vDLdt = x*0.0
    for k in range(0, len(x)): 
        vDLdt[k] = distL(x[k], h0, Or, Om, Ok, w0, w1)
    model = 5.0*np.log10(vDLdt) + 25.0
    inv_sigma2 = 1.0/(yerr**2)
    
    xsq = np.sum((y-model)**2*inv_sigma2)
    xsl = -0.5*(np.sum((y-model)**2*inv_sigma2))
        
    #print theta, xsq
    return xsl
def lnprob3(theta, x, y, yerr):
    lp = lnprior3(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike3(theta, x, y, yerr)
    
def lnlike05(theta, x, y, yerr):
    h0, Om, w0, w1  = theta
        
    if 0.0 <= h0 <= 1.5  and 0.0 <= Om <= 1.0 and -2.0 <= w0 <= 0.0:
        Or = 4.153e-5 * h0**(-2)
        vDLdt = x*0.0
        for k in range(0, len(x)): 
            vDLdt[k] = distL(x[k], h0, Or, Om, 0.0, w0, w1)
        model = 5.0*np.log10(vDLdt) + 25.0
        inv_sigma2 = 1.0/(yerr**2)
        
        xsq = np.sum((y-model)**2*inv_sigma2)
        xsl = -0.5*(np.sum((y-model)**2*inv_sigma2))
        
        #print theta, xsq
        return xsl
    else:
        return -np.inf
def lnprior5(theta):
    h0, Om, w0, w1  = theta
    if 0.0 <= h0 <= 1.5  and 0.0 <= Om <= 1.0 and -2.0 <= w0 <= 0.0 and -1.0 <= w1 <= 1.0:
        return 0.0
    return -np.inf
def lnlike5(theta, x, y, yerr):    
    h0, Om, w0, w1  = theta
    
    Or = 4.153e-5 * h0**(-2)
    vDLdt = x*0.0
    for k in range(0, len(x)): 
        vDLdt[k] = distL(x[k], h0, Or, Om, 0.0, w0, w1)
    model = 5.0*np.log10(vDLdt) + 25.0
    inv_sigma2 = 1.0/(yerr**2)
    
    xsq = np.sum((y-model)**2*inv_sigma2)
    xsl = -0.5*(np.sum((y-model)**2*inv_sigma2))
        
    #print theta, xsq
    return xsl
def lnprob5(theta, x, y, yerr):
    lp = lnprior5(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike5(theta, x, y, yerr)
    
def mcsne(ve, dpath, fg0, fg1):
    from astropy.table import Table
    import astropy.units as u
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import scipy.optimize as op
    import corner
    import emcee
    
    from hubblediagram import hubblediagram
    
    print '+++++++++++++++++++++++++++++++++++++++++++'
    print 'MCSNE: '
    print '+++++++++++++++++++++++++++++++++++++++++++'

    #=============================================================================
#============= Parameter  ==================================================== #=============================================================================
    c = 2.9979e+5

    fend ='_MCSNe20'
    #============================================================================   #============= Main Body ====================================================
#============================================================================
           
    # Reading Supernovae Ia data                                                     
    TpathLS = dpath+'indat/SCPUnion21.txt'
    tSne = Table.read(TpathLS, format='ascii')
    vZ = tSne['col2']
    vMu = tSne['col3']
    vMuErr = tSne['col4']
    index = range(1, len(vZ))
    
    # Reading Supernovae Ia covariance matrix                                                     
    TpathLS = dpath+'indat/SCPUnion21cm.txt'
    TCM = Table.read(TpathLS, format='ascii')
    
    aCM = TCM.to_pandas()   
            
    # Hubble diagram                                                                 
    path = dpath+'results/HD'+fend+'.eps'
    hubblediagram(path, ve, vZ, vMu, vMuErr, index)
    
    if fgS == 1:
        vMuErr = aCM
    
    # First guess.    
    h0_i = 0.60
    Om_i = 0.25
    Ok_i = 0.0
    w0_i = -1.5
    w1_i = 0.0
    
    if fg1 == 0:
        # Maximum likelihood estimation
        nll = lambda *args: -lnlike00(*args)
        result = op.minimize(nll, [h0_i, Om_i], args=(vZ, vMu, vMuErr))
        h0_ml, Om_ml = result["x"]
        print h0_ml, Om_ml
        print -2.0*lnlike00([h0_ml, Om_ml], vZ, vMu, vMuErr), len(vZ)
            
        # MCMC
        if fg0 == 1: 
            ndim, nwalkers = 2, 100
            pos = [result["x"] + 1e-4*np.random.randn(ndim) for i in range(nwalkers)]
    
            sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob0, args=(vZ, vMu, vMuErr),
                      threads = 4)
            sampler.run_mcmc(pos, 500)
    
            samples = sampler.chain[:, 50:, :].reshape((-1, ndim))
    
            np.save(dpath+'temp/MCMCSne0.npy', samples)
        else:
            samples = np.load(dpath+'temp/MCMCSne0.npy')
        
        h0_mcmc, Om_mcmc = map(lambda v: (v[1], v[2]-v[1], v[2]-v[0]),
                                 zip(*np.percentile(samples, [16, 50, 84],
                                                    axis=0)))
                                                    
        print h0_mcmc
        print Om_mcmc
        print -2.0*lnlike00([h0_mcmc[0], Om_mcmc[0]], vZ, vMu, vMuErr), len(vZ)
    
        # Results
        path = dpath+'results/tr'+fend+'.eps'
        fig = corner.corner(samples,
                            labels=["$h_0$", "$\Omega_m$"],
                            truths=[h0_mcmc[0], Om_mcmc[0]], 
                            quantiles = [0.16, 0.84], 
                            plot_contours = 'true',
                            levels = 1.0 - np.exp(-0.5 * np.arange(1.0, 3.1, 1.0) ** 2),
                            smooth = 1.0, color = 'black')

        plt.savefig(path)
    
    elif fg1 == 1:
        # Maximum likelihood estimation
        nll = lambda *args: -lnlike01(*args)
        result = op.minimize(nll, [Om_i, Ok_i], args=(vZ, vMu, vMuErr))
        Om_ml, Ok_ml = result["x"]
        print Om_ml, Ok_ml
        print -2.0*lnlike01([Om_ml, Ok_ml], vZ, vMu, vMuErr), len(vZ)
            
        # MCMC
        if fg0 == 1: 
            ndim, nwalkers = 2, 100
            pos = [result["x"] + 1e-5*np.random.randn(ndim) for i in range(nwalkers)]
    
            sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob1, args=(vZ, vMu, vMuErr),
                      threads = 4)
            sampler.run_mcmc(pos, 500)
    
            samples = sampler.chain[:, 50:, :].reshape((-1, ndim))
    
            np.save(dpath+'temp/MCMCSne1.npy', samples)
        else:
            samples = np.load(dpath+'temp/MCMCSne1.npy')
        
        Om_mcmc, Ok_mcmc = map(lambda v: (v[1], v[2]-v[1], v[2]-v[0]),
                                 zip(*np.percentile(samples, [16, 50, 84],
                                                    axis=0)))
                                                    
        print Om_mcmc
        print Ok_mcmc
        print -2.0*lnlike01([Om_mcmc[0], Ok_mcmc[0]], vZ, vMu, vMuErr), len(vZ)
    
        # Results
        path = dpath+'results/tr'+fend+'1.eps'
        fig = corner.corner(samples,
                            labels=["$\Omega_m$", "$\Omega_k$"],
                            truths=[Om_mcmc[0], Ok_mcmc[0]], 
                            levels = 1.0 - np.exp(-0.5 * np.arange(1.0, 3.1, 1.0) ** 2),
                            quantiles = [0.16, 0.84], smooth = 1.0)
        plt.savefig(path)
    elif fg1 == 2:
        if fg0 == 1: 
            # Maximum likelihood estimation
            nll = lambda *args: -lnlike02(*args)
            result = op.minimize(nll, [Om_i, w0_i], args=(vZ, vMu, vMuErr))
            Om_ml, w0_ml = result["x"]
            print Om_ml, w0_ml
            print -2.0*lnlike02([Om_ml, w0_ml], vZ, vMu, vMuErr), len(vZ)
        
            # MCMC
            ndim, nwalkers = 2, 700
            pos = [result["x"] + 1e-5*np.random.randn(ndim) for i in range(nwalkers)]
    
            sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob2, args=(vZ, vMu, vMuErr), 
                      threads = 4)
            sampler.run_mcmc(pos, 1400)
    
            samples = sampler.chain[:, 50:, :].reshape((-1, ndim))
    
            np.save(dpath+'temp/MCMCSne2.npy', samples)
        else:
            samples = np.load(dpath+'temp/MCMCSne2.npy')
        
        Om_mcmc, w0_mcmc = map(lambda v: (v[1], v[2]-v[1], v[2]-v[0]),
                                 zip(*np.percentile(samples, [16, 50, 84],
                                                    axis=0)))
        
        # Results                                            
        print Om_mcmc
        print w0_mcmc
        print -2.0*lnlike02([Om_mcmc[0], w0_mcmc[0]], vZ, vMu, vMuErr), len(vZ)
        
        #Log
        f = open(dpath+'results/log'+fend+'.dat', 'w')
        f.write('log'+fend+'.dat\n')
        f.write(str(Om_mcmc)+'\n')
        f.write(str(w0_mcmc)+'\n')
        f.write(str(-2.0*lnlike02([Om_mcmc[0], w0_mcmc[0]], vZ, vMu, vMuErr))+'\n')
        f.write(str(len(vZ))+'\n')
        f.close
        
        #Plot
        path = dpath+'results/tr'+fend+'.png'
        fig = corner.corner(samples,
                            labels=["$\Omega_m$", "$w_0$"],
                            truths=[Om_mcmc[0], w0_mcmc[0]], 
                            quantiles = [0.16, 0.84], 
                            range = [(0.,0.5), (-1.5, 0.0)], 
                            plot_contours = 'true',
                            levels = 1.0 - np.exp(-0.5 * np.arange(1.0, 3.1, 1.0) ** 2),
                            smooth = 0.6, 
                            bins = 100,
                            color = 'black')
        plt.savefig(path, dpi = 500)
        
    elif fg1 == 3:
        # Maximum likelihood estimation
        nll = lambda *args: -lnlike03(*args)
        result = op.minimize(nll, [Om_i, Ok_i], args=(vZ, vMu, vMuErr))
        Om_ml, Ok_ml = result["x"]
        print Om_ml, Ok_ml
        print -2.0*lnlike03([Om_ml, Ok_ml], vZ, vMu, vMuErr), len(vZ)
        
        # MCMC
        if fg0 == 1: 
            ndim, nwalkers = 2, 200
            pos = [result["x"] + 1e-5*np.random.randn(ndim) for i in range(nwalkers)]
    
            sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob3, args=(vZ, vMu, vMuErr), 
                      threads = 4)
            sampler.run_mcmc(pos, 500)
    
            samples = sampler.chain[:, 50:, :].reshape((-1, ndim))
    
            np.save(dpath+'temp/MCMCSne3.npy', samples)
        else:
            samples = np.load(dpath+'temp/MCMCSne3.npy')
        
        Om_mcmc, Ok_mcmc = map(lambda v: (v[1], v[2]-v[1], v[2]-v[0]),
                                 zip(*np.percentile(samples, [16, 50, 84],
                                                    axis=0)))
        
                                                                                                
        print Om_mcmc
        print Ok_mcmc
        print -2.0*lnlike03([Om_mcmc[0], Ok_mcmc[0]], vZ, vMu, vMuErr), len(vZ)
    
        # Results
        path = dpath+'results/tr'+fend+'3.eps'
        fig = corner.corner(samples,
                            labels=["$\Omega_m$", "$\Omega_k$"],
                            truths=[Om_mcmc[0], Ok_mcmc[0]], 
                            #truths=[Om_ml, Ok_ml], 
                            quantiles = [0.16, 0.84], 
                            range = [(0.,0.65), (0.,0.4)],
                            levels = 1.0 - np.exp(-0.5 * np.arange(1.0, 3.1, 1.0) ** 2), 
                            plot_contours = 1, 
                            smooth = 1.0)
        plt.savefig(path)
            
    elif fg1 == 5:
        # Maximum likelihood estimation
        nll = lambda *args: -lnlike0(*args)
        result = op.minimize(nll, [h0_i, Om_i, w0_i, w1_i], args=(vZ, vMu, vMuErr))
        h0_ml, Om_ml, w0_ml, w1_ml = result["x"]
        print h0_ml, Om_ml, w0_ml, w1_ml
        
        # MCMC
        if fg0 == 1: 
            ndim, nwalkers = 4, 100
            pos = [result["x"] + 1e-4*np.random.randn(ndim) for i in range(nwalkers)]
    
            import emcee
            sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(vZ, vMu, vMuErr),
                       threads = 4 )
            sampler.run_mcmc(pos, 500)
    
            samples = sampler.chain[:, 50:, :].reshape((-1, ndim))
    
            np.save(dpath+'temp/MCMCSne.npy', samples)
        else:
            samples = np.load(dpath+'temp/MCMCSne.npy')
        
        h0_mcmc, Om_mcmc, w0_mcmc, w1_mcmc = map(lambda v: (v[1], v[2]-v[1], v[2]-v[0]),
                                 zip(*np.percentile(samples, [16, 50, 84],
                                                    axis=0)))
                                                    
        print h0_mcmc
        print Om_mcmc
        print w0_mcmc    
        print w1_mcmc
    
        x = vZ
        y = vMu
        yerr = vMuErr
    
        Or = 4.153e-5 * h0_mcmc[0]**(-2)
        vDLdt = x*0.0
        for k in range(0, len(x)): 
            vDLdt[k] = distL(x[k], h0_mcmc[0], Or, Om_mcmc[0], 0.0, w0_mcmc[0], w1_mcmc[0])
        model = 5.0*np.log10(vDLdt) + 25.0
        inv_sigma2 = 1.0/(yerr**2)
    
        print np.sum((y-model)**2*inv_sigma2), len(x)
    
        # Results
        path = dpath+'results/triangle'+fend+'.eps'
        fig = corner.corner(samples,
                            labels=["$h_0$", "$\Omega_m$", "$w_0$", "$w_1$"],
                            truths=[h0_mcmc[0], Om_mcmc[0], w0_mcmc[0], w1_mcmc[0]],
                            levels = 1.0 - np.exp(-0.5 * np.arange(1.0, 3.1, 1.0) ** 2), 
                            quantiles = [0.16, 0.84], smooth = 1.0)
        plt.savefig(path)
    else:
        print 'Error in fg1 parameter'
    
    return