from expHeating_helper import *
from steady_state_helper import *

def saveArrs(uArr, vArr, bArr, phiArr,dt,outputFile):
    with open(outputFile,'wb') as outFile:
        np.save(outFile,uArr)
        np.save(outFile,vArr)
        np.save(outFile,bArr)
        np.save(outFile,phiArr)
        np.save(outFile,dt)
    return 0

def longRun(R,Pr,alpha,Nx,Nz,ell,beta,T,timestep, fileName=None):
    testProb = expHeat_Problem(R,Pr,alpha,Nx,Nz,ell,beta,time_step=timestep)
    testProb.initialize()
    testProb.solve_system(T,True,False,True)
    if fileName is None:
        fileName = 'R' + str(R) + 'Pr' + str(Pr) + 'alpha' + str(alpha) + 'ell' + str(ell) + 'beta' + str(beta) + 'Nx' + str(Nx) + 'Nz' + str(Nz) + '_T'+str(T)+'.npy'
    testProb.saveToFile(fileName)

def getSteady(Ra,Pr,alpha,Nx,Nz,ell,beta,T,tol,guessFile,steadyStateFile,dtscale):
    uArr, vArr, bArr, phiArr, dt = open_fields(guessFile)
    starting_SS_state = arrsToStateVec(phiArr, bArr)
    startingGuess = starting_SS_state
    starting_dt = dt*dtscale
    startingProblem = expHeat_Problem(Ra,Pr,alpha,Nx,Nz,ell,beta)
    startingProblem.initialize()
    findSteadyState(startingProblem,startingGuess,T,tol,50,True)
    startingProblem.saveToFile(steadyStateFile)


def branchFollow(Pr,alpha,Ra_start,num_steps,Ra_step, Nx, Nz,ell,beta,startFile, T,tol,dtscale):
    uArr, vArr, bArr, phiArr, dt = open_fields(startFile)
    starting_SS_state = arrsToStateVec(phiArr, bArr)
    startingGuess = starting_SS_state
    starting_dt = dt*dtscale
    print("old dt:", dt)
    print("new dt:", starting_dt)
    RaVals, NuVals = follow_branch(Pr,alpha,Ra_start,num_steps,Ra_step, Nx, Nz,ell,beta, startingGuess, starting_dt, T,tol)
    return RaVals, NuVals

def searchOverAlpha(Ra,Pr,Nx,Nz,ell,beta,starting_alpha,alpha_step,startingFile,tol):
    uArr, vArr, bArr, phiArr, dt = open_fields(startingFile)
    starting_SS_state = arrsToStateVec(phiArr, bArr)
    startingGuess = starting_SS_state
    alphaVals, NuVals = varyAlpha(Ra,Pr,Nx,Nz,ell,beta,starting_alpha,alpha_step,startingGuess,dt,tol)
    return alphaVals, NuVals

def refine(uArr,vArr,bArr,phiArr,alpha,Nx,Nz,scale):
    ##run this using only ONE processor or else will not work
    print(uArr.shape)
    print("--------")
    coords = d3.CartesianCoordinates('x', 'z')
    dist = d3.Distributor(coords, dtype=np.float64)
    xbasis = d3.RealFourier(coords['x'], Nx, bounds=(-np.pi/alpha, np.pi/alpha), dealias=3/2)
    zbasis = d3.Chebyshev(coords['z'], Nz, bounds=(-1, 1), dealias=3/2)
    u = dist.Field(name='u', bases=(xbasis, zbasis))
    v = dist.Field(name='v', bases=(xbasis, zbasis))
    b = dist.Field(name='b', bases=(xbasis, zbasis))
    phi = dist.Field(name='phi', bases=(xbasis, zbasis))
    u.load_from_global_grid_data(uArr)
    v.load_from_global_grid_data(vArr)
    b.load_from_global_grid_data(bArr)
    phi.load_from_global_grid_data(phiArr)
    u.change_scales(scale)
    v.change_scales(scale)
    b.change_scales(scale)
    phi.change_scales(scale)
    newuArr = u['g']
    newvArr = v['g']
    newbArr = b['g']
    newphiArr = phi['g']
    print(newuArr.shape)
    return newuArr, newvArr, newbArr, newphiArr

def refine2(uArr,vArr,bArr,phiArr,alpha,NxOld,NzOld,NxNew,NzNew):
    print(uArr.shape)
    print("---------")
    coords = d3.CartesianCoordinates('x', 'z')
    dist = d3.Distributor(coords, dtype=np.float64)
    xbasis = d3.RealFourier(coords['x'], NxOld, bounds=(-np.pi/alpha, np.pi/alpha), dealias=3/2)
    zbasis = d3.Chebyshev(coords['z'], NzOld, bounds=(-1, 1), dealias=3/2)
    u = dist.Field(name='u', bases=(xbasis, zbasis))
    v = dist.Field(name='v', bases=(xbasis, zbasis))
    b = dist.Field(name='b', bases=(xbasis, zbasis))
    phi = dist.Field(name='phi', bases=(xbasis, zbasis))
    u.load_from_global_grid_data(uArr)
    v.load_from_global_grid_data(vArr)
    b.load_from_global_grid_data(bArr)
    phi.load_from_global_grid_data(phiArr)
    newuArr = np.zeros((NxNew,NzNew))
    newvArr = np.zeros((NxNew,NzNew))
    newbArr = np.zeros((NxNew,NzNew))
    newphiArr = np.zeros((NxNew,NzNew))
    
    xbasisNew = d3.RealFourier(coords['x'], NxNew, bounds=(-np.pi/alpha, np.pi/alpha), dealias=3/2)
    zbasisNew = d3.Chebyshev(coords['z'], NzNew, bounds=(-1, 1), dealias=3/2)
    xValsNew = dist.local_grid(xbasisNew)
    zValsNew = dist.local_grid(zbasisNew)
    for i in range(NxNew):
        for j in range(NzNew):
            pointx = xValsNew[i][0]
            pointz = zValsNew[0][j]
            uPointVal = u(x=pointx,z=pointz)
            uPointVal = uPointVal.evaluate()['g']

            vPointVal = v(x=pointx,z=pointz)
            vPointVal = vPointVal.evaluate()['g']

            bPointVal = b(x=pointx,z=pointz)
            bPointVal = bPointVal.evaluate()['g']
            
            phiPointVal = phi(x=pointx,z=pointz)
            phiPointVal = phiPointVal.evaluate()['g']
            
            newuArr[i][j] = uPointVal
            newvArr[i][j] = vPointVal
            newbArr[i][j] = bPointVal
            newphiArr[i][j] = phiPointVal
    return newuArr,newvArr,newbArr,newphiArr

#expHeat_Problem(Ra,Pr,alpha,Nx,Nz,ell,beta

#########################
### For refining grid ###
#########################
'''
<<<<<<< HEAD
dataFile = 'Ra67151.0Pr7alpha2.56201Nx180Nz130_SS.npy'
newFile = 'Ra67151.0Pr7alpha2.56201Nx210Nz150_SS_refined.npy'
=======
dataFile = 'Ra85704.0Pr7alpha2.56201Nx180Nz130_SS.npy'
newFile = 'Ra85704.0Pr7alpha2.56201Nx210Nz150_SS_refined.npy'
>>>>>>> 51fec7a7cbb36f4bb08c49f671057e07ebaacbc5
NxOld = 180
NzOld = 130
alpha = 2.56201
NxNew = 210
NzNew = 150
uArr, vArr, bArr, phiArr, dt = open_fields(dataFile)
uRef, vRef, bRef, phiRef = refine2(uArr,vArr,bArr,phiArr,alpha,NxOld,NzOld,NxNew,NzNew)
saveArrs(uRef, vRef, bRef, phiRef,dt,newFile)
'''
###########################################
### For refining grid by changing scale ###
###########################################
'''
dataFile = 'Ra4370.0Pr7alpha2.56201Nx100Nz70_SS.npy'
newFile = 'Ra4370.0Pr7alpha2.56201Nx120Nz84_SS_refined.npy'
Nxold = 100
Nzold = 70
alpha = 2.56201
scale = 1.2
uArr, vArr, bArr, phiArr, dt = open_fields(dataFile)
uRef, vRef, bRef, phiRef = refine(uArr, vArr, bArr, phiArr,alpha,Nxold,Nzold,scale)
saveArrs(uRef, vRef, bRef, phiRef,dt,newFile)
'''     

####################
### for long run ###
####################
'''
<<<<<<< HEAD
Ra = 2000
Pr = 7
alpha = 2.56201
Nx = 90
Nz = 60
ell = 0.1
beta = 0
T = 10

longRun(Ra,Pr,alpha,Nx,Nz,ell,beta,T)
=======
R = 3500
Pr = 7
alpha = 2.5183
Nx = 200
Nz = 100
ell = 0.1
beta = 1
T = 50
timestep = 0.1

longRun(R,Pr,alpha,Nx,Nz,ell,beta,T,timestep)
>>>>>>> 51fec7a7cbb36f4bb08c49f671057e07ebaacbc5
'''
#########################################
### For finding a single steady state ###
#########################################
'''
<<<<<<< HEAD
Ra = 30763
Pr = 7
alpha=2.56201
Nx=180
Nz=130
ell = 0.1
beta = 0
T=0.5
dtscale = 1/1.1
guessFile = 'Ra30763.0Pr7alpha2.56201Nx180Nz130_SS_refined.npy'
steadyFile = 'Ra30763.0Pr7alpha2.56201Nx180Nz130_SS.npy'
=======
Ra = 3500
Pr = 7
alpha=2.5183
Nx=200
Nz=100
ell = 0.1
beta = 1
T=1.0
dtscale = 1
guessFile = 'R3500Pr7alpha2.5183ell0.1beta1Nx200Nz100_T50.npy'
steadyFile = 'R3500Pr7alpha2.5183ell0.1beta1Nx200Nz100_SS.npy'
>>>>>>> 51fec7a7cbb36f4bb08c49f671057e07ebaacbc5
getSteady(Ra,Pr,alpha,Nx,Nz,ell,beta,T,1e-6,guessFile, steadyFile,dtscale)
'''
##############################
### For following a branch ###
##############################

Pr = 7
alpha = 2.5183
Ra_start = 37923
num_steps = 10
Ra_step = 1.1
Nx = 200
Nz = 100
ell = 0.1
beta = 1
startFile = 'Ra37923.0Pr7alpha2.5183Nx200Nz100_SS.npy'
T = 1.0
tol = 1e-6
dtscale = 1

RaVals, NuVals = branchFollow(Pr, alpha, Ra_start, num_steps, Ra_step, Nx,Nz,ell,beta,startFile,T,tol,dtscale)

##################
### Vary alpha ###
##################

startingFile = '/grad/gudibanda/expHeating_Convection/Ra10016.0Pr7alpha2.56201Nx120Nz84_SS.npy'
Ra = 10016
Pr = 7
starting_alpha = 2.56201
alpha_step = 0.1
Nx = 120
Nz = 84
beta = 0
ell = 0.1
tol = 1e-6 

searchOverAlpha(Ra,Pr,Nx,Nz,ell,beta,starting_alpha,alpha_step,startingFile,tol)
