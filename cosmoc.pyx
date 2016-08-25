#cython: cdivision=True

from cython_gsl cimport *

ctypedef double * double_ptr
ctypedef void * void_ptr

cdef double dint(double x, void * params) nogil:
	cdef double Or, Om, Ok, w0, w1, Ow, f
	Or = (<double_ptr> params)[0]
	Om = (<double_ptr> params)[1]
	Ok = (<double_ptr> params)[2]
	w0 = (<double_ptr> params)[3]
	w1 = (<double_ptr> params)[4]
	Ow = 1.0 - Om - Or - Ok
	f = 1.0/sqrt(Or*(1.0 + x)**4 + Om*(1.0 + x)**3 + Ok*(1.0 + x)**2 + Ow*(1.0 + x)**(3.0*(1.0 + w0 + w1 ))*exp(-3.0 * w1 * (x/(x+1.0))))
	return f


def integral(z, Or, Om, Ok, w0, w1):
	cdef gsl_integration_workspace * w
	cdef double I, error
	w = gsl_integration_workspace_alloc (1000)
		
	cdef gsl_function F
	cdef double params[5]
	params[0] = Or
	params[1] = Om
	params[2] = Ok
	params[3] = w0
	params[4] = w1
	F.function = &dint
	F.params = &params
	
	gsl_integration_qags (&F, 0, z, 0, 1e-9, 1000, w, &I, &error)
	
	gsl_integration_workspace_free (w)
	
	return I
	
def Ez(z, h0, Or, Om, Ok, w0, w1):
	cdef double Ow, Ez
	Ow = 1.0 - Om - Or - Ok
	Ez = sqrt(Or*(1.0 + z)**4 + Om*(1.0 + z)**3 + Ok*(1.0 + z)**2 + Ow*(1.0 + z)**(3.0*(1.0 + w0 + w1 ))*exp(-3.0 * w1 * (z/(z+1.0))))
	return Ez
	
def distP(z, h0, Or, Om, Ok, w0, w1):
	cdef double c, Ho, Hd, Dm
	c = 2.99792458e+5
	Ho = h0*100.0
	Hd = c/Ho
	I  = integral(z, Or, Om, Ok, w0, w1)
	Dm = Hd*I
	return Dm

def distL(z, h0, Or, Om, Ok, w0, w1):
	cdef double Dl
	Dl = distP(z, h0, Or, Om, Ok, w0, w1) * (1.0 + z)
	return Dl
	
def Rz(h0, Or, Om, Ok, w0, w1):
	cdef double c, Ob, g1, g2, Zst, I, Rz
    
	c = 2.99792458e+5
	Ob  = 0.022242*(h0)**(-2)
	g1 = 0.0783*((Ob*h0**2)**(-0.238)/(1.0 + 39.5*(Ob*h0**2)**(0.763)))
	g2 = 0.560/(1.0 + 21.1*(Ob*h0**2)**(1.81))
    
	Zst = 1048.0*(1.0 + 0.001240*(Ob*h0**2)**(-0.738))*(1.0 + g1*(Om*h0**2)**(g2))
    
	I = integral(Zst, Or, Om, Ok, w0, w1)
    
	Rz = sqrt(Om)*I
	return Rz
	
def Az(z, h0, Or, Om, Ok, w0, w1):
	cdef double c, I, Az
	
	c = 2.99792458e+5
	I  = integral(z, Or, Om, Ok, w0, w1)
	Az = (sqrt(Om)/Ez(z, h0, Or, Om, Ok, w0, w1)**(1.0/3.0))*((1.0/z)*I)**(2.0/3.0)
	return Az