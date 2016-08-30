# -*- coding: utf-8 -*-
#hiigs main module

__all__ = ["hiigs"]
__version__ = "1.0.0"
__author__ = "Ricardo Chavez (rc681@cam.ac.uk)"
__copyright__ = "Copyright 2016 Ricardo Chavez"
__contributors__ = [
    # Alphabetical by first name.
    'Ricardo Chavez'
]

from mcsnexp import mcsnexp

def hiigs(ve=None, dpath=None):
    import os
    mcd = os.path.dirname(os.path.abspath(__file__))
    
    dpath = mcd+'/hiidat/'
    ve = '1_15'
    
    print '+++++++++++++++++++++++++++++++++++++++++++'
    print 'HIIGS: '
    print '+++++++++++++++++++++++++++++++++++++++++++'
    
    #Calls the subrutine for SNeIa JLA Sample analysis
    mcsnexp(ve, dpath, clc = 1 , opt = 1)
    
    print 'done'
        
    return 
    
if __name__ == "__main__":
    hiigs()