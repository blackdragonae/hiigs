#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# :Name:     mcjointplotsv3.py
# :Purpose:  Final plots
# :Author:   Ricardo Chavez -- Aug 5th, 2020 -- Morelia, Mich
# :Modified:
#------------------------------------------------------------------------------
import numpy as np
from pymultinest.analyse import  Analyzer
import corner
from getdist import plots, MCSamples, densities
import datetime
import time
import glob
import matplotlib.pyplot as plt


def mcjointplotsv3(ve, dpath, cpath, opt = 0):

    print('+++++++++++++++++++++++++++++++++++++++++++')
    print('mcjointplotsv3: ' + str(datetime.datetime.now()))
    print('+++++++++++++++++++++++++++++++++++++++++++')

    #============= Parameters ======================================
    fg1 = opt

    #============= Main Body =======================================
    print(datetime.datetime.now())
    start = datetime.datetime.now()
    # Cases
    if fg1 == 0:
        path = dpath+'results/JOmv_' + str(ve) + '.pdf'

        file0 = cpath + 'mchv69_152_1141212'
        file1 = cpath + 'mchv69_152_1641212'
        file2 = cpath + 'mchv69_149_2241212'

        n_params = 1
        parameters = [r"\Omega_m"]

    elif fg1 == 1:
        path = dpath+'results/JOmW0v_' + str(ve) + '.pdf'

        file0 = cpath + 'mchv69_152_1151212'
        file1 = cpath + 'mchv69_152_1651212'
        file2 = cpath + 'mchv69_149_2251212'

        n_params = 2
        parameters = [r"\Omega_m", r"w_0"]

    elif fg1 == 2:
        path = dpath+'results/Jh0Omv_' + str(ve) + '.pdf'

        file0 = cpath + 'mchv69_152_1142212'
        file1 = cpath + 'mchv69_152_1642212'
        file2 = cpath + 'mchv69_149_2242212'

        n_params = 2
        parameters = [r"h", r"\Omega_m"]

    elif fg1 == 3:
        path = dpath+'results/Jh0Omw0v_' + str(ve) + '.pdf'

        file0 = cpath + 'mchv69_149_1154212'
        file1 = cpath + 'mchv69_149_1654212'
        file2 = cpath + 'mchv69_149_2254212'

        n_params = 3
        parameters = [r"h_0", r"\Omega_m", r"w_0"]

    elif fg1 == 4:
        path = dpath+'results/Jabh0Omv_' + str(ve) + '.pdf'

        file0 = cpath + 'mchv69_149_104211'
        file1 = cpath + 'mchv69_149_1224211'
        file2 = cpath + 'mchv69_149_254211'

        n_params = 4
        parameters = [r"\alpha", r"\beta", r"h_0", r"\Omega_m"]

    elif fg1 == 5:
        path = dpath+'results/Jabh0OmW0v_' + str(ve) + '.pdf'

        file0 = cpath + 'mchv69_149_105211'
        file1 = cpath + 'mchv69_149_1225211'
        file2 = cpath + 'mchv69_149_255211'

        n_params = 5
        parameters = [r"\alpha", r"\beta", r"h_0", r"\Omega_m", r"w_0"]




    tag0 = 'Full (181)'
    tag1 = 'Our Data (157)'
    tag2 = 'GM2019 (153)'

    a0 = Analyzer(n_params, outputfiles_basename=file0)
    data0 = a0.get_data()[:,2:]
    weights0 = a0.get_data()[:,0]

    a1 = Analyzer(n_params, outputfiles_basename=file1)
    data1 = a1.get_data()[:,2:]
    weights1 = a1.get_data()[:,0]

    a2 = Analyzer(n_params, outputfiles_basename=file2)
    data2 = a2.get_data()[:,2:]
    weights2 = a2.get_data()[:,0]

    GD0 = MCSamples(samples=data0, names=parameters,
                          labels=parameters, name_tag=tag0,
                          weights=weights0,
                          ranges={r"\Omega_m":[0.0, None]})

    t0 = GD0.getTable(limit=1).tableTex()
    FoM0 = 1./np.sqrt(np.linalg.det(GD0.cov()))

    GD1 = MCSamples(samples=data1, names=parameters,
                          labels=parameters, name_tag=tag1,
                          weights=weights1,
                          ranges={r"\Omega_m":[0.0, None]})

    t1 = GD1.getTable(limit=1).tableTex()
    FoM1 = 1./np.sqrt(np.linalg.det(GD1.cov()))

    GD2 = MCSamples(samples=data2, names=parameters,
                          labels=parameters, name_tag=tag2,
                          weights=weights2,
                          ranges={r"\Omega_m":[0.0, None]})

    t2 = GD2.getTable(limit=1).tableTex()
    FoM2 = 1./np.sqrt(np.linalg.det(GD2.cov()))

    print(t0)
    print(FoM0)
    print(t1)
    print(FoM1)
    print(t2)
    print(FoM2)

    #Joint
    print('Joint')
    g = plots.getSubplotPlotter() # width_inch=4 
    g.settings.figure_legend_frame = False
    g.settings.legend_fontsize = 10
    g.settings.alpha_filled_add=0.3
    g.triangle_plot([GD0, GD1, GD2], parameters, 
                filled=True,
                contour_colors=['red','blue','green'],
                param_limits={r"\alpha":[32.8, 33.7]
                                , r"\beta":[4.7, 5.35]
                                , r"h_0":[0.6, 0.85]
                                , r"\Omega_m":[0.0, 0.52]
                                , r"w_0":[-2.1, -0.2]}
                )

    # plt.hlines(0.995, 0.01, 0.03, colors='red')
    # plt.annotate("$\Omega_m = 0.244^{+0.040}_{-0.049}$", xy=(0.032, 0.98),
            # xytext=(0.032, 0.99), fontsize=8)

    # plt.hlines(0.925, 0.01, 0.03, colors='blue')
    # plt.annotate("$\Omega_m = 0.243^{+0.047}_{-0.057}$", xy=(0.032, 0.92),
            # xytext=(0.032, 0.92), fontsize=8)

    # plt.hlines(0.855, 0.01, 0.03, colors='green')
    # plt.annotate("$\Omega_m = 0.290^{+0.057}_{-0.068}$", xy=(0.032, 0.85),
            # xytext=(0.032, 0.85), fontsize=8)

    g.export(path)
    print(datetime.datetime.now())
    return
