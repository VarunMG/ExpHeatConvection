from expHeating_helper import *
from scipy import optimize

######################################
### to load arrays from .npy files ###
######################################

def open_fields(file_name):
    with open(file_name,'rb') as l_File:
        uFromFile = np.load(l_File)
        vFromFile = np.load(l_File)
        bFromFile = np.load(l_File)
        phiFromFile = np.load(l_File)
        dt = np.load(l_File)
    return uFromFile, vFromFile,bFromFile, phiFromFile, dt

#########################
### finding steady state ###
#########################

def probToStateVec(Prob):
    Nx = Prob.Nx
    Nz = Prob.Nz
    
    Prob.phi.change_scales(1)
    Prob.b.change_scales(1)
    bArr = Prob.b.allgather_data('g')
    phiArr = Prob.phi.allgather_data('g')

    X = np.zeros(2*Nx*Nz)
    bArr = bArr.flatten()
    phiArr = phiArr.flatten()
    X[0:Nx*Nz] = phiArr
    X[Nx*Nz:] = bArr
    return X

def arrsToStateVec(phiArr,bArr):
    Nx,Nz = bArr.shape
    X = np.zeros(2*Nx*Nz)
    bArr = bArr.flatten()
    phiArr = phiArr.flatten()
    X[0:Nx*Nz] = phiArr
    X[Nx*Nz:] = bArr
    return X

def stateToArrs(X,Nx,Nz):
    phiArr = X[0:Nx*Nz]
    bArr = X[Nx*Nz:]
    
    #get them into a format we can put into the flow map function
    phiArr = np.reshape(phiArr,(-1,Nz))
    bArr = np.reshape(bArr,(-1,Nz))
    return phiArr, bArr

def Gt(X,T,problem):
    phiArr, bArr = stateToArrs(X,problem.Nx,problem.Nz)
    uArr, vArr = problem.phi_lap.getVel(phiArr)
    problem.time = 0
    problem.phi.load_from_global_grid_data(phiArr)
    problem.u.load_from_global_grid_data(uArr)
    problem.v.load_from_global_grid_data(vArr)
    problem.b.load_from_global_grid_data(bArr)
    problem.solve_system(T)
    Gt_Vec = probToStateVec(problem)
    Gt_Vec = (Gt_Vec - X)/T
    #statusFile = open("optimizationStatus.txt","a")
    #statusFile.write("flow map call \n")
    #statusFile.close()
    logger.info("flow map call")
    logger.info(np.linalg.norm(Gt_Vec))
    return Gt_Vec

#def jac_approx(X,dX,F,T,problem):
#    Nx = problem.Nx
#    Nz = problem.Nz
#    #print(dX)
#    #dX = zeroBoundariesdx(dX,Nx,Nz)
#    mach_eps = np.finfo(float).eps
#    normX = np.linalg.norm(X)
#    normdX = np.linalg.norm(dX)
#    dotprod = np.dot(X,dX)
#    logger.info("norm dx: %f",normdX)
#    eps = (np.sqrt(mach_eps)/(normdX**2))*max(dotprod,normdX)
#    logger.info("1e6*esp: %f", 1e6*eps)
#    #eps = 1e-3
#    GtArr = Gt(X+eps*dX,T,problem).T
#    return (GtArr + F.T)/eps

#def zeroBoundariesdx(dx_vec,Nx, Nz):
#    dxPhi, dxb = stateToArrs(dx_vec,Nx,Nz)
#    #logger.info("dxphi:")
#    #logger.info(dxPhi.shape)
#    #logger.info("dxb:")
#    #logger.info(dxb.shape)
#    dxPhi[:,0] = np.zeros(Nx)
#    dxPhi[:,Nz-1] = np.zeros(Nx)
#    dxb[:,0] = np.zeros(Nx)
#    dxb[:,Nz-1] = np.zeros(Nx)
#    #logger.info("dxphi:")
#    #logger.info(dxPhi)
#    #logger.info("dxb:")
#    #logger.info(dxb)
#    return arrsToStateVec(dxPhi,dxb)


#def findSteadyState2(problem,guess,T,tol,max_iters,empty_arg):
#    Nx = problem.Nx
#    Nz = problem.Nz
#    #Delta_X = np.random.random(len(guess))
#    #Delta_X = zeroBoundariesdx(Delta_X,Nx,Nz)
#    norm_X0 = np.linalg.norm(guess)
#    newtonStep = 0
#    found = False
#    X = guess
#    for i in range(max_iters):
#        logger.info("------------------------")
#        logger.info("At iteration:" + str(i))
#        GT = Gt(X,T,problem)
#        normGT = np.linalg.norm(GT)/norm_X0
#        logger.info("error:")
#        logger.info(normGT)
#        if normGT < tol:
#            logger.info("solution within tolerance")
#            found = True
#            break
#        b = -1*GT
#        A = lambda dX: jac_approx(X,dX,b,T,problem)
#        A_matrix = LinearOperator((2*Nx*Nz,2*Nx*Nz),matvec=A)
#        Delta_X,code =gmres(A_matrix,b,tol=1e-3)#maxiter=500)
#        if code != 0:
#            logger.info('WARNING: GMRES FAILED TO CONVERGE')
#        X = X + Delta_X
        
#    logger.info("steady state found!")
#    phiStead, bStead = stateToArrs(X,Nx,Nz)
#    problem.phi.load_from_global_grid_data(phiStead)
#    problem.b.load_from_global_grid_data(bStead)
#    uStead, vStead = problem.phi_lap.getVel(phiStead)
#    problem.u.load_from_global_grid_data(uStead)
#    problem.v.load_from_global_grid_data(vStead)
#    return X,i


def findSteadyState(problem,guess,T,tol,max_iters,write):
    #problem.dt = 1e-4
    normX0 = np.linalg.norm(guess)
    GT_func = lambda X: Gt(X,T,problem)
    print('about to enter newt krylov')
    #err_func = lambda X: np.linalg.norm(GT_func(X))/normX0
    sol=optimize.newton_krylov(GT_func,guess,verbose=True,maxiter=20)#,tol_norm=err_func)
    return sol
    

#def findSteadyState(problem,guess,T,tol,max_iters,write):
    #problem is an RBC_problem
    #guess is a guess for the state vec
    #T is time we are integrating out to
    #tol is tolerance for Newton method 
    #imax_iters is max Newton iterations that will be done
#    X = guess
#    normX0 = np.linalg.norm(X)
#    err = np.linalg.norm(Gt(X,T,problem))/normX0
#    logger.info("initial error: %f", err)
#    #err = 1e10
#    iters = 0
#
#    Nx = problem.Nx
#    Nz = problem.Nz
#    delta_X = np.zeros(len(X))
#    while err > tol and iters < max_iters:
#        #statusFile = open("optimizationStatus.txt","a")
#        #statusFile.write("------------------ \n")
#        #statusText = 'iteration: ' + str(iters) + '\n'
#        #statusFile.write(statusText)
#        #statusFile.close()
#        logger.info("---------------------")
#        logger.info("iteration:" + str(iters))
#        if write == 'y':
#            print("iter: ",iters)
#            print(X)
#            print("-------------")

#        F = -1*Gt(X,T,problem)
#        A = lambda dX : jac_approx(X,dX,F,T,problem)
#        A_mat = LinearOperator((2*Nx*Nz,2*Nx*Nz),matvec=A)
#        delta_X,code =gmres(A_mat,F,tol=1e-3)#,x0=delta_X)
#        #print(delta_X)
#        if code != 0:
#            raise("gmres did not converge")
#        X= X+delta_X
#        logging.info("Completed iteration: %i", iters)
#        iters += 1
#        err = np.linalg.norm(Gt(X,T,problem))/normX0
#        logger.info("error:" + str(err))
#    #statusFile = open("optimizationStatus.txt","a")
#    #statusFile.write("loop over \n")
#    #statusFile.close()
#    logger.info("loop over")
#    phiStead, bStead = stateToArrs(X,Nx,Nz)
#    problem.phi.load_from_global_grid_data(phiStead)
#    problem.b.load_from_global_grid_data(bStead)
#    uStead, vStead = problem.phi_lap.getVel(phiStead)
#    problem.u.load_from_global_grid_data(uStead)
#    problem.v.load_from_global_grid_data(vStead)
#    return iters

def follow_branch(Pr,alpha,Ra_start,num_steps,Ra_step, Nx, Nz,ell,beta, startingGuess, starting_dt, T,tol):
    #Ra_start is starting Rayleigh number
    #Ra_end is the ending Rayleigh number
    #Ra_step is the step in Ra taken when increasing Ra
    RaVals = np.round(Ra_start*(Ra_step**np.arange(1,num_steps+1)))
    #RaVals = np.arange(Ra_start,Ra_end,Ra_step)
    #steady_states = []
    Nu_Vals = []
    guess = startingGuess
    dt = starting_dt
    filename_end = 'Pr'+str(Pr)+'alpha'+str(alpha)+'ell'+str(ell)+'beta'+str(beta)+'Nx' + str(Nx) + 'Nz' + str(Nz) + '_SS.npy'
    for Ra in RaVals:
        steady = expHeat_Problem(Ra,Pr,alpha,Nx,Nz,ell,beta,time_step=dt)
        steady.initialize()
        steady_vec = findSteadyState(steady, guess, T, tol, 50, False)
        #print('Ra= ',Ra)
        #print('steady state found . Iters = ', iters)
        #steady_states.append(steady)
        Nu = steady.calcNu()
        Nu_Vals.append(Nu)
        logging.info("Ra= %i", Ra)
        #logging.info('Steady State Found. Iters = %i', iters)
        logging.info("Nu= %f", Nu)
        saveFile = 'Ra'+str(Ra)+filename_end
        #saveFile = filename+'Ra'+str(Ra)+'.npy'
        steady.saveToFile(saveFile)
        
        steady.phi.change_scales(1)
        steady.b.change_scales(1)
        steady_b = steady.b.allgather_data('g')
        steady_phi = steady.phi.allgather_data('g')
        guess = arrsToStateVec(steady_phi, steady_b)
        dt = steady.time_step
    print(RaVals)
    print(Nu_Vals)
    return RaVals, Nu_Vals#, steady_states
        

def foundOptimalNu(NuArr):
    if len(NuArr) >= 3 and (NuArr[-2] > NuArr[-3]) and (NuArr[-2] > NuArr[-1]):
        return True
    return False

def findOptimalAlpha(Ra,Pr,Nx,Nz,starting_alpha,alpha_step,startingGuess,dt,tol,outputOpt):
    found = False
    alpha = starting_alpha
    guess = startingGuess
    Nu_Vals = []
    alpha_Vals = []
    max_iters = 50
    iters = 0
    while not found and iters < max_iters:
        print("alpha values checked so far")
        print(alpha_Vals)
        print("Nu vals calculated so far")
        print(Nu_Vals)
        steady = IH_Problem(Ra,Pr,alpha,Nx,Nz,time_step=dt)
        steady.initialize()
        steadystate_iters = findSteadyState(steady, guess, 2, tol, 50, False)
        print('alpha=',alpha)
        print("steady state found. Iters=",steadystate_iters)
        Nu1 = steady.calc_Nu()
        Nu2 = steady.calc_Nu2()
        logger.info("THE DIFFERENTLY COMPUTED NU VALS ARE:")
        logger.info("Nu1= %f",Nu1)
        logger.info("Nu2= %f",Nu2)
        #print("Nu value:",Nu)
        alpha_Vals.append(alpha)
        Nu_Vals.append(Nu2)
        alpha = alpha+alpha_step
        found = foundOptimalNu(Nu_Vals)
        if found:
            print("we found an optimal alpha!")
        else:
            print("not yet found optimal alpha :( ")
        iters+= 1
        steady.b.change_scales(1)
        steady.phi.change_scales(1)
        steady_b = steady.b.allgather_data('g')
        steady_phi = steady.phi.allgather_data('g')
        dt = steady.time_step
        guess = arrsToStateVec(steady_phi, steady_b)
    if found:
        print("found an optimal state!!!")
        optAlpha = np.array([alpha_Vals[-3],alpha_Vals[-2],alpha_Vals[-1]])
        optNu = np.array([Nu_Vals[-3],Nu_Vals[-2],Nu_Vals[-1]])
        quadInterp = np.polyfit(optAlpha,optNu,2)
        alphaMax = -1*quadInterp[1]/(2*quadInterp[0])
        NuMax = quadInterp[0]*alphaMax**2 + quadInterp[1]*alphaMax + quadInterp[2]
        if outputOpt:
            steady = IH_Problem(Ra,Pr,alphaMax,Nx,Nz,time_step=dt)
            steady.initialize()
            steadystate_iters = findSteadyState(steady,guess,2,tol,50,False)
            fileName = 'Ra'+str(Ra)+'Pr'+str(Pr)+'_optimalState.npy'
            steady.saveToFile(fileName)
            print("Calculate Nu:")
            print(steady.calc_Nu())
        return alpha_Vals, Nu_Vals, alphaMax, NuMax
    return alpha_Vals, Nu_Vals, -1, -1
        
def varyAlpha(Ra,Pr,Nx,Nz,ell,beta,starting_alpha,alpha_step,startingGuess,dt,tol):
    alpha = starting_alpha
    guess = startingGuess
    NuVals = []
    alpha_Vals = []
    max_iters = 30
    iters = 0
    while iters <= max_iters:
        logger.info('alpha values checked so far:')
        logger.info(alpha_Vals)
        logger.info('Nu values calculated so far:')
        logger.info(NuVals)

        steady = expHeat_Problem(Ra,Pr,alpha,Nx,Nz,ell,beta,time_step=dt)
        steady.initialize()
        steadystate_iters = findSteadyState(steady,guess,1.0,tol,50,False)
        logger.info('alpha=%0.16f',alpha)
        logger.info('steady state found!. Number of iters=%0.16f',steadystate_iters)
        Nu= steady.calcNu()
        logger.info('Nu=%0.16f',Nu)
        alpha_Vals.append(alpha)
        NuVals.append(Nu)
        alpha = alpha - alpha_step
        
        steady.b.change_scales(1)
        steady.phi.change_scales(1)
        steady_b = steady.b.allgather_data('g')
        steady_phi = steady.phi.allgather_data('g')
        dt = steady.time_step
        guess = arrsToStateVec(steady_phi, steady_b)
    logger.info('------------')
    logger.info('alpha values:')
    logger.info(alpha_Vals)
    logger.info('Nu values:')
    logger.info(NuVals)
    return alpha_Vals,NuVals
        
    
