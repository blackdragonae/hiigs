#+                                                                                                          
# :Name:     hubblediagram.py         
# :Purpose:  This code plots a Hubble diagram for a set of data     
# :Author:   Ricardo Chavez 23 February 2012       
# :Modified: Ricardo Chavez 16 February 2016, adapting to python      
#-    
def hubblediagram(path, ve, vZ, vMu, vMuErr, index):
   import numpy as np
   import math as m 
   import matplotlib.pyplot as plt
   
   from cosmoc import distL
       
   print '+++++++++++++++++++++++++++++++++++++++++++'
   print 'HUBBLEDIAGRAM: '
   print '+++++++++++++++++++++++++++++++++++++++++++'

   #====================================================================================       
   #============= Parameters ===========================================================        
   #====================================================================================
   c = 2.9979e+5
   
   #====================================================================================        
   #============= Main Body ============================================================    
   #====================================================================================
   # ;;;;;Calculating Theoretical Mu (LCDM)
   vZfit = np.arange(0.0001, 3.5, 0.0001, dtype=np.float)
   Hp = 70.0
   Om = 0.286
   Ok = 0.0
   h0 = Hp/100.0
   Or = 4.153e-5 * h0**(-2)
   w0 = -1.0
   
   vDLConc = vZfit * 0.0
   
   for i in range(0, len(vZfit)):
       vDLConc[i] = distL(vZfit[i], h0, Or, Om, Ok, w0, 0.0)
          
   vMufit = 5.0*np.log10(vDLConc) + 25.0
   
   
   plt.rc('font', family='serif')
   plt.plot(vZfit, vMufit)
   plt.errorbar(vZ,vMu,yerr=vMuErr, linestyle="None", capsize = 0, ecolor = "red")
   plt.scatter(vZ, vMu, s=8, c='r', marker=".")
   
   
   plt.axis([-0.08, 3.5, 18, 49])
   
   plt.xlabel(r'\textbf{$z$}', fontsize=16)
   plt.ylabel(r'\textbf{$\mu$}', fontsize=16)
   
   plt.savefig(path)      
   return