##########################################################################
##################### IMPORTING REQUIRED MODULES ####################
 
import numpy
import pylab
from scipy.optimize import leastsq # Levenberg-Marquadt Algorithm #

from scipy.interpolate import interp1d ##Interpolation ##
 
#########################################################################
########################## LOADING DATA ###############################
 
a = numpy.loadtxt('test_1.txt')
x = a[:,0]
y = a[:,1]
print x
print y
 
########################################################################
######################### DEFINING FUNCTIONS ########################
 
def lorentzian(x,p):
    min_step = x[1] - x[0]
    ind1 = (x< p[1])
    x1 = x[ind1]
    ind2 = (x > p[1])
    x2 = x[ind2]
    numerator_left = (p[0]**2 )
    denominator_left = ( x1 - (p[1]) )**2 + p[0]**2
    numerator_right = (p[2]**2 )
    denominator_right = ( x2 - (p[1]) )**2 + p[2]**2
    y_left = p[3]*(numerator_left/denominator_left)
    y_right = p[3]*(numerator_right/denominator_right)
    lin_comb = numpy.hstack((y_left,y_right))
    return lin_comb
 
def residuals(p,y,x):
    err = y - lorentzian(x,p)
    return err
 
###########################################################################
###################### BACKGROUND SUBTRACTION #######################
 
# defining the 'background' part of the spectrum #
ind_bg_low = (x > 0.0) & (x < 470.0)
ind_bg_high = (x > 600.0) & (x < 1000.0)
 
x_bg = numpy.concatenate((x[ind_bg_low],x[ind_bg_high]))
y_bg = numpy.concatenate((y[ind_bg_low],y[ind_bg_high]))
#pylab.plot(x_bg,y_bg)
 
# interpolating the background #
f = interp1d(x_bg,y_bg)
background = f(x)
 
# removing fitted background #
y_bg_corr = y - background
 
########################################################################
########################## FITTING DATA ## ############################
 
# initial values #
p = [5.0,520.0,5.0,12e3] # [hwhm1, peak center, hwhm2, intensity] #
 
# optimization #
pbest = leastsq(residuals,p,args=(y_bg_corr,x),full_output=1)
best_parameters = pbest[0]
 
# fit to data #
fit = lorentzian(x,best_parameters)
 
#######################################################################
############################## PLOTTING ##############################
 
pylab.plot(x,y_bg_corr,'wo')
pylab.plot(x,fit,'r-',lw=2)
pylab.xlabel(r'$\omega$ (cm$^{-1}$)', fontsize=18)
pylab.ylabel('Intensity (a.u.)', fontsize=18)
pylab.show()
