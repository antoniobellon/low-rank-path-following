import numpy as np 
import scipy 
import time  
import warnings 
import residual
import linearized_kkt as lk 
      
warnings.filterwarnings("ignore")

class _PredictorCorrector:

    def __init__(self, n: int, m: int, rank: int, params: dict, ini_stepsize: float, res_tol: float):

        """ Constructor pre-allocates and pre-computes persistent
            data structures. """ 

        # Storing problem dimensions
        self._n = n
        self._m = m
        self._rank = rank

        # Create all necessary modules and pre-allocate the workspace 
        self._LinearizedKKTsystem = lk._LinearizedKKTsystem(n=n, m=m, rank=rank)
        self._linear_KKT_sol = np.zeros((int(n*rank+m+0.5*rank*(rank-1)),))
        self._candidate_Y = np.zeros((n, rank)) 
        self._candidate_lam = np.zeros((m, )) 
        self._Y = np.zeros((n, rank)) 
        self._X = np.zeros((n, n))
        self._lam = np.zeros((m,))   

        # Storing algorithm parameters 
        self._final_time = float(params["problem"]["final_time"])
        self._initial_time = float(params["problem"]["initial_time"]) 
        self._ini_stepsize = ini_stepsize
        self.times = []

        # Storing algorithm parameters for stepsize tuning 
        self._gamma1 = float(params["problem"]["gamma1"])
        self._gamma2 = float(params["problem"]["gamma2"])
        self._res_tol = res_tol
       
        # Initializing variables for storing solution, residuals and runtime
        self._primal_solutions_list = [] 
        self._LR_residuals = []
        self._LR_runtime = 0.0
       
    def run(self, A: np.ndarray, b: np.ndarray, C: np.ndarray, Y_0: np.ndarray, lam_0: np.ndarray, 
            STEPSIZE_TUNING: bool, PRINT_DATA: bool): 

        """ The actual algorithm is executed. """ 
 
        iteration = 0 
         
        # Get copies of all problem parameters  
        dt = self._ini_stepsize   
        curr_time = self._initial_time
        next_time = self._initial_time + dt
        n, m, rank = self._n, self._m, self._rank

        # Store initial solution as the current iterate
        np.copyto(self._Y, Y_0)
        np.copyto(self._X,np.matmul(Y_0,Y_0.T))
        np.copyto(self._lam, lam_0)  
        self.times.append(curr_time)
        
        # Store initial solution in the solutions array
        self._primal_solutions_list.append(np.array(self._X))

        while curr_time < self._final_time:   
 
            # Compute linearized KKT system
            H = self._LinearizedKKTsystem.computeMatrix(A= A(next_time), C=C(next_time), Y=self._Y, lam=self._lam) 
            k = self._LinearizedKKTsystem.computeRhs(A=A(next_time), b=b(next_time), C=C(next_time), Y=self._Y, X=self._X)   

            # Solve the system and store the candidate solution step, recording the execution time
            start_time = time.time() 
            np.copyto(self._linear_KKT_sol, scipy.linalg.solve(H, k, assume_a='sym'))
            np.copyto(self._candidate_Y, np.reshape(self._linear_KKT_sol[:n*rank],(rank,n)).T)
            self._LR_runtime += time.time() - start_time 
            
            # Store current candidate solution and multipliers
            self._candidate_Y += self._Y  
            np.copyto(self._candidate_lam, self._linear_KKT_sol[n*rank:n*rank+m]) 
            
            # Compute and store the residual 
            res = residual.resid(n=n, m=m, rank=rank, A=A(next_time), b=b(next_time),  C=C(next_time), 
                                            Y=self._candidate_Y, lam=self._candidate_lam)
            self._LR_residuals.append(max(res))

            # Possibly print some data 
            if not PRINT_DATA and iteration%10==0: print("\nITERATION", iteration) 

            if PRINT_DATA: 

                print("\nITERATION", iteration) 
                print("running time t", time.time() - start_time) 
                print("TIME", curr_time) 
                print("TARGET TIME", next_time) 
                print("TIME STEP", dt) 
                print("res VS res_tol:", res, self._res_tol )   
            
            # According to STEPSIZE_TUNING: either reduce dt by a factor_gamma1 if the residual threshold is violated ...
            if STEPSIZE_TUNING and max(res)>self._res_tol: 

                dt *= self._gamma1
                next_time = curr_time + dt  
                print("reducing stepsize...")
                print("res VS res_tol:", res, self._res_tol )    
                    
            # ... or keep a constant stepsize
            else:

                # Update (and store) the solution and the time
                np.copyto(self._Y, self._candidate_Y)
                np.copyto(self._lam, self._candidate_lam) 
                np.copyto(self._X,np.matmul(self._Y,self._Y.T))  
                self._primal_solutions_list.append(np.array(self._X))
                curr_time += dt
                self.times.append(curr_time)
                iteration += 1 
                
                # STEPSIZE_TUNING also indicates whether to optimistically tune the stepsize by a factor_gamma2 or not 
                if STEPSIZE_TUNING:
                    dt = min(self._final_time  - curr_time, self._gamma2 * dt)
                else:
                    dt = min(self._final_time  - curr_time, dt)

                next_time = curr_time+dt 