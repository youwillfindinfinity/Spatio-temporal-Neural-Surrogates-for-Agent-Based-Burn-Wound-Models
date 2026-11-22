nx = 500
ny = nx
dx = 1.
total_cytokines = 6
total_celltypes = 10
boundaryat = 50
relaxationmcs = 10000 # changed from 10000
# unit conversion (for square domain)
#input the total size of the grids in cm along x axis
true_size = 5
#input the total time for solving at 1 fipy time stepin minutes (from seconds)(ignore the relaxation mcs)
fipy_duration = int(1) #in hours
s_mcs = 60.
h_mcs = 1/60.
#input the basic mass unit (here it is pg meaning input acceptable only in picograms!)
true_mass = 1
#conversions 
lineconv = true_size/nx #cm per grid
areaconv = true_size**2/nx**2
volumeconv = (true_size**2*1)/(nx**2*1) # unit length along z (2D assumption) ml or cm3 as base
massconv = true_mass
# parameters for il8
Dil8 =  2.09*10**-6*s_mcs/areaconv
muil8 = 0.2*h_mcs
keil8 = 234 * 10**-5 * volumeconv * h_mcs
kndnil8 = 1.46 * 10**-5 * volumeconv * h_mcs
thetanail8 = 3.024 * 10**-5 * volumeconv * h_mcs
# parameters for il1
Dil1 = 3*10**-7*s_mcs/areaconv
muil1 = 0.6 * h_mcs
knail1 = 225 * 10**-5 * volumeconv * h_mcs
# parameters for il6
Dil6 = 8.49*10**-8*s_mcs/areaconv
muil6 = 0.5*h_mcs
km1il6 = 250 * 10**-5 * volumeconv * h_mcs
# parameters for il10
Dil10 = 1.45*10**-8*s_mcs/areaconv
muil10 = 0.5*h_mcs
km2il10 = 45 * 10**-5 * volumeconv * h_mcs
# parameters for tnf
Dtnf = 4.07*10**-9*s_mcs/areaconv
mutnf = 0.5*0.225*h_mcs
knatnf = 250 * 10**-5 * volumeconv * h_mcs
km1tnf = 70 * 10**-5 * volumeconv * h_mcs
# parameters for tgf
Dtgf = 2.6*10**-7*s_mcs/areaconv
mutgf = 0.5 * 1/25 * h_mcs
km2tgf = 280 * 10**-5 * volumeconv * h_mcs
# agent params
# promotion or inhibition saturation concentrations
cil8 = 2*10**-9
cil1 = 5*10**-9
cil6 = 5*10**-9
cil10 = 5*10**-9
ctnf = 5*10**-9
ctgf = 5*10**-9

# cell activation probability
lnril8 = 0.25
lnril6 = 0.25
lnril1 = 0.25
lnrtnf = 0.25
tnril10 = -0.5
lmril6 = 0.5
lmrtnf = 0.5
tmril10 = -0.5
lm1il10 = 1.0
lftgf = 1.0

tranril6 = 1.0
actnr = [[cil8,lnril8],[cil6,lnril6],[cil1,lnril1],[ctnf,lnrtnf],[cil10,tnril10]]
actmr = [[cil6,lmril6],[ctnf,lmrtnf],[cil10,tmril10]]
actm1 = [[cil10,lm1il10]]
actf = [[ctgf,lftgf]]
tranmr = [[cil6, tranril6]]
#there is no cell division
lifespane = 1000000 #give a large value they live long # changed from 1000000
lifespannr = 20
lifespanm = 24
lifespanmr = 1000000 
# lifespanf = 120
lifespanf = 1000000 # changed from 1000000
timeforgrowth = 0.5 # after which a cell can divide, using 3/4 of the total life span here
# curves parameters
# Sigmoid params
sigmoida = 1
sigmoidb = 4
# params for m2lifespan, experiment 6 MM2
vmax = -17.79826
km = -963211.7


#divpre = -1 #cant divide
#divprnr = int((lifespannr - lifespannr*timeforgrowth)*2) # approximately half probability since there are 30 chances to divide from 90 -120 hours
#divprm = int((lifespanm - lifespanm*timeforgrowth)*2)
#divprf = int((lifespanf - lifespanf*timeforgrowth)*2)

divpre = 1 #cant divide
divprnr = -1
divprm = -1
divprf = int((lifespanf - lifespanf*timeforgrowth)*2)
