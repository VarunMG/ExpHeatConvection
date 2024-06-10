import numpy as np
import dedalus.public as d3
import logging
import matplotlib.pyplot as plt
from matplotlib import cm
import os
logger = logging.getLogger(__name__)


#######################################
### initializing and Nu calculation ###
#######################################

def conductionState(alpha_coeff,beta,ell,z):
    return -1*alpha_coeff*ell**2*np.exp(-1*z/ell) + 0.5*beta*z**2 - alpha_coeff*ell*z+ alpha_coeff*ell**2*np.exp(-1/ell) + alpha_coeff*ell - 0.5*beta

def calcNu(b_var,alpha_coeff,beta,ell):
    TAvg = d3.Average(b_var,coords['x']).evaluate()
    TAvgVals = TAvg.allgather_data('g')
    return conductionState(alpha_coeff,beta,ell,0)/TAvgVals[0][0]

##############################
### File Writing Functions ###
##############################

def makedirs(auxDir,dataDir):
    try:
        os.mkdir(auxDir)
        os.mkdir(dataDir)
    except:
        pass

def writeNu(fileName,tVals,NuVals):
    with open(fileName,'wb') as NuData:
        np.save(NuData,tVals)
        np.save(NuData,NuVals)
    return 1

def writeAllVertMeans(fileName,vertMeanData):
    with open(fileName, 'wb') as vertMeanFile:
        np.save(vertMeanFile,vertMeanData)
    return 1

def writeFields(fileName,time,b_var,u_var,v_var):
    b_var.change_scales(1)
    u_var.change_scales(1)
    v_var.change_scales(1)
    with open(fileName,'wb') as fluidData:
        np.save(fluidData,time)
        np.save(fluidData,b_var.allgather_data('g').T)
        np.save(fluidData,u_var.allgather_data('g').T)
        np.save(fluidData,v_var.allgather_data('g').T)
    return 1

#################
### Main Code ###
#################

# Parameters
alpha = np.pi/5
Nx, Nz = 1024, 256
Rayleigh = 1e5
Prandtl = 7
ell = 0.1
beta = 0
stop_sim_time = 200
max_timestep= 0.0001

dealias = 3/2
timestepper = d3.RK443
dtype = np.float64

# initial conditions params
rand_init = True #if True, then random perturbation of initial condition. Otherwise, sinusoidal perturbation

# Data writing parameters

write2Files = True
write_freq = 1000

# Bases
coords = d3.CartesianCoordinates('x', 'z')
dist = d3.Distributor(coords, dtype=dtype)
xbasis = d3.RealFourier(coords['x'], size=Nx, bounds=(-1*np.pi/alpha, np.pi/alpha), dealias=dealias)
zbasis = d3.ChebyshevT(coords['z'], size=Nz, bounds=(0, 1), dealias=dealias)

# Fields
phi = dist.Field(name='phi', bases=(xbasis,zbasis))
u = dist.Field(name='u', bases=(xbasis,zbasis))
v = dist.Field(name='v', bases=(xbasis,zbasis))
b = dist.Field(name='b', bases=(xbasis,zbasis))

Tsource = dist.Field(name='Tsource',bases=zbasis)

tau_v1 = dist.Field(name='tau_v1', bases=xbasis)
tau_v2 = dist.Field(name='tau_v2', bases=xbasis)

tau_phi1 = dist.Field(name='tau_phi1', bases=xbasis)
tau_phi2 = dist.Field(name='tau_phi2', bases=xbasis)

tau_b1 = dist.Field(name='tau_b1', bases=xbasis)
tau_b2 = dist.Field(name='tau_b2', bases=xbasis)

tau_u1 = dist.Field(name='tau_u1', bases=xbasis)
tau_u2 = dist.Field(name='tau_u2', bases=xbasis)

# Substitutions
Pr = Prandtl
Ra = Rayleigh
kappa = 4*(Rayleigh * Prandtl)**(-1/2)
nu = 4*(Rayleigh / Prandtl)**(-1/2)
x, z = dist.local_grids(xbasis, zbasis)
ex, ez = coords.unit_vector_fields(dist)
lift_basis = zbasis.clone_with(a=1/2, b=1/2) # First derivative basis
lift = lambda A, n: d3.Lift(A, lift_basis, n)

grad_v = d3.grad(v) + ez*lift(tau_v1,-1)
grad_phi = d3.grad(phi) + ez*lift(tau_phi1,-1)
grad_b = d3.grad(b) + ez*lift(tau_b1,-1) # First-order reduction
dz = lambda A: d3.Differentiate(A, coords['z'])
dx = lambda A: d3.Differentiate(A, coords['x'])

# Problem
# First-order form: "div(f)" becomes "trace(grad_f)"
# First-order form: "lap(f)" becomes "div(grad_f)"
problem = d3.IVP([phi, u, v, b, tau_v1, tau_v2, tau_phi1, tau_phi2, tau_b1, tau_b2], namespace=locals())
problem.add_equation("div(grad_v) + lift(tau_v2,-1) - phi= 0")
problem.add_equation("dt(phi) - Pr*div(grad_phi)-  Pr*Ra*dx(dx(b)) + lift(tau_phi2,-1) = -dx(u*phi - v*lap(u))  ")
problem.add_equation("dt(b) - div(grad_b) + lift(tau_b2,-1) = -u*dx(b)-v*dz(b)+Tsource")
problem.add_equation("dx(u) + dz(v)+ lift(tau_v1,-1) = 0", condition='nx!=0')
problem.add_equation("u = 0", condition='nx==0')
problem.add_equation("b(z=1) = 0")
problem.add_equation("v(z=1) = 0")
problem.add_equation("dz(b)(z=0) = 0")
problem.add_equation("v(z=0) = 0")
problem.add_equation("dz(v)(z=1) = 0")
problem.add_equation("dz(v)(z=0) = 0")

# Solver
solver = problem.build_solver(timestepper)
solver.stop_sim_time = stop_sim_time

#Exponential heating
alpha_coeff = 1/(ell*(1-np.exp(-1/ell)))
Tsource['g'] = alpha_coeff*np.exp(-z/ell) - beta

# Initial conditions
if rand_init:
    b.fill_random('g', seed=42, distribution='normal', scale=1e-3) # Random noise
    b['g'] *= z*(z-1) #damp noise at walls but maintain the boundary conditions
else:
    b['g'] += 0.05*np.cos((1/2)*np.pi*(x-alpha))*np.cos(np.pi*z/2) #adding a perturbation
b['g'] += conductionState(alpha_coeff,beta,ell,z) # Add conduction state background


# CFL
CFL = d3.CFL(solver, initial_dt=max_timestep, cadence=10, safety=0.5, threshold=0.05,
             max_change=1.5, min_change=0.5, max_dt=max_timestep)
CFL.add_velocity(u*ex+v*ez)


#solver.print_subproblem_ranks(dt=max_timestep)


##volume of box
volume = ((2*np.pi)/alpha)*2

# Flow properties
flow = d3.GlobalFlowProperty(solver, cadence=10)
flow.add_property(b,name='TAvg')
flow.add_property(b*v,name="Nu")

# Main loop
startup_iter = 10
tVals = []
NuVals = []
allVertMeans = []


mainDir = '/scratch/gudibanda/'
genFileName = 'R'+str(Ra)+'Pr'+str(Pr)+'alpha'+str(alpha)+ 'ell'+str(ell)+'beta'+str(beta)+'Nx'+str(Nx)+'Nz'+str(Nz)+'_T' + str(stop_sim_time)
auxDirName = mainDir + genFileName + '_auxData/'
runOutDirName = mainDir + genFileName + '_runOutput/'
makedirs(auxDirName,runOutDirName)

NuFileName =  auxDirName +  genFileName + '_NuData.npy'
fluidDataFileName = runOutDirName + 'fluidData'

try:
    logger.info('Starting main loop')
    while solver.proceed:
        timestep = CFL.compute_timestep()
        solver.step(timestep)
        
        flow_TAvg = flow.volume_integral('TAvg')/volume
        flow_Nu = calcNu(b,alpha_coeff,beta,ell)
        tVals.append(solver.sim_time)
        NuVals.append(flow_Nu)
        if (solver.iteration-1) % 10 == 0:
            logger.info('Iteration=%i, Time=%e, dt=%e, Nu=%0.16f, <T>=%0.16f' %(solver.iteration, solver.sim_time, timestep, flow_Nu, flow_TAvg))
        if write2Files and (solver.iteration-1) % write_freq == 0:
            writeNu(NuFileName,tVals,NuVals)
            fileName = fluidDataFileName + str(round(10000000*solver.sim_time)/10000000) + '.npy'
            write = writeFields(fileName,solver.sim_time,b,u,v)
            if write == 0:
                print('fields are not writing')

except:
    logger.error('Exception raised, triggering end of main loop.')
    raise
finally:
    solver.log_stats()

fileName = fluidDataFileName + str(round(10000*solver.sim_time)/10000) + '.npy'
writeFields(fileName,solver.sim_time,b,u,v)
writeNu(NuFileName,tVals,NuVals)
