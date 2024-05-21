# -*- coding: utf-8 -*-
# hiigs main module on mraos
# Modified: Tue 30 Apr 2024

__all__ = ['main']
__version__ = "10.0.0"
__author__ = "Ricardo Chavez (rchavez@irya.unam.mx)"
__copyright__ = "Copyright 2024 Ricardo Chavez"
__contributors__ = [
    # Alphabetical by first name.
    'Ricardo Chavez'
]



import sys 
import os
import time 
# import matplotlib
# matplotlib.use('Agg')

# from mch2gv2 import mch2gv2
from mch2gv79 import mch2gv79
# from mcsnev11 import mcsnev11
# from mcbaov6 import mcbaov6
# from mccmbv10 import mccmbv10
# from mch2gcmbbaov8 import mch2gcmbbaov8
# from mcsnecmbbaov6 import mcsnecmbbaov6
# from mcjointplotsv3 import mcjointplotsv3


def main():
    start_time = time.time()
    mcd = os.path.dirname(os.path.abspath(__file__))

    dpath = mcd+'/dat/'
    cpath = dpath + 'resultsMN/'
    # cpath = '/export/data/Chavez/hiigs/resultsMN/'
    # cpath = '/Users/rchavez/h2dat/results/'

    ve = '190'

    if len(sys.argv) > 1:
        nsps = int(sys.argv[1])
    else:
        nsps = 1000

    print('+++++++++++++++++++++++++++++++++++++++++++')
    print('hiigs: '+ve)
    print('+++++++++++++++++++++++++++++++++++++++++++')

    mch2gv79(ve, dpath, cpath, spl=1, zps=40
            , opt=5, clc=0, drd=2, obs=1, prs=0, vbs=1
        #     , a=33.11, aErr=0.145, b=5.05, bErr=0.097 #drd = 1
        #     , a=33.268, aErr=0.083, b=5.022, bErr=0.058 #drd = 2
            , a=0.0, aErr=0.0, b=0.0, bErr=0.0 #Dflt
             , sps=14000, fr0=1, fra=0
            )

    # mch2gv2(ve, dpath, cpath, spl=1, zps=12
    #         , opt=1, clc=1, drd=2, obs=1, prs=0, vbs=1
    #         # # , a=33.11, aErr=0.145, b=5.05, bErr=0.097 #drd = 1
    #     #     , a=33.268, aErr=0.083, b=5.022, bErr=0.058 #drd = 2
    #         , a=0.0, aErr=0.0, b=0.0, bErr=0.0 #Dflt
    #         , sps=10000, fr0=1
    #         )

    # mcsnev11(ve, dpath, cpath, clc=1, opt=2, sps=1000, prs=0, vbs=0)

    # mccmbv10(ve, dpath, cpath, opt = 1, prs = 0, sps = 1000)

    # mcbaov6(ve, dpath, cpath, opt = 1, prs = 0, sps = 1000)

    # mch2gcmbbaov8(ve, dpath, cpath, spl = 2, zps = 1
            # , opt = 2, clc = 1, drd = 2, obs = 1, prs = 1, vbs = 0
            # #, a = 33.11, aErr = 0.145, b = 5.05, bErr = 0.097 #drd = 1
            # , a = 33.268, aErr = 0.083, b = 5.022, bErr = 0.058 #drd = 2
            # #, a = 0.0, aErr = 0.0, b = 0.0, bErr = 0.0 #Dflt
            # , sps = 1000, fr0 = 1
            # )

    # mcsnecmbbaov6(ve, dpath, cpath, clc=0, opt=2, sps=1000, prs=0, vbs=0)

    # mcjtplotsv2(ve, dpath, cpath, spl=2, zps=1
            # , opt=21, clc=1, drd=2, obs=1, prs=1, vbs = 0
            # #, a = 33.11, aErr = 0.145, b = 5.05, bErr = 0.097 #drd = 1
            # , a=33.268, aErr=0.083, b=5.022, bErr=0.058 #drd = 2
            # #, a = 0.0, aErr = 0.0, b = 0.0, bErr = 0.0 #Dflt
            # , sps=1000, fr0=1
            # )

    # mcjointplotsv3(ve, dpath, cpath, opt=0)

    print('The End')
    print("ETime:--- %s seconds ---" % (time.time() - start_time))
    return

if __name__ == "__main__":
    main()
