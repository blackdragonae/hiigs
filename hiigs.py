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

def hiigs(ve=None, dpath=None):
    import os
    mcd = os.path.dirname(os.path.abspath(__file__))
    
    dpath = mcd+'/hiidat/'
    
    ve = 0.0
    
    from mcsne import mcsne
    
    mcsne(ve, dpath, 1, 2)
        
    return 
    
if __name__ == "__main__":
    hiigs()