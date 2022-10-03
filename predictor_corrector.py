import numpy as np 
import scipy
import pickle
import time 
import warnings

import sdp_solver as ip
import residual as residual
import linearized_kkt as lk
from scipy.stats import ortho_group
      
class _PredictorCorrector:

    def __init__(self, n: int, m: int, rank: int, params: dict, ini_stepsize: float, res_tol: float):
        """ Constructor pre-allocates and pre-computes persistent
            data structures. """ 

        self._n = n
        self._m = m
        self._rank = rank

        # Create all necessary modules and pre-allocate the workspace 
         
        self._LinearizedKKTsystem = lk._LinearizedKKTsystem(n=n, m=m, rank=rank)
        self._Y = np.zeros((n, rank)) 
        self._lam = np.zeros((m, ))   
        self._candidate_Y = np.zeros((n, rank))
        self._candidate_lam = np.zeros((m, )) 
        self._linear_KKT_sol = np.zeros((int(n*rank+m+0.5*rank*(rank-1)),))
        self._X = np.zeros((n, n))

        # Parameters 
        self._gamma1 = float(params["problem"]["gamma1"])
        self._gamma2 = float(params["problem"]["gamma2"])
        self._final_time = float(params["problem"]["final_time"])
        self._initial_time = float(params["problem"]["initial_time"])
        self._verbose = int(params["verbose"])
        self._ini_stepsize = ini_stepsize
        self._res_tol = res_tol

        self._LR_runtime = 0.0
        self._SDP_highest_acc_runtime = 0.0 
        self._SDP_high_acc_runtime = 0.0 
        self._SDP_low_acc_runtime = 0.0 

        self._LR_res_list = []
        self._SDP_highest_acc_res_list = []
        self._SDP_high_acc_res_list = []
        self._SDP_low_acc_res_list = [] 

        self._LR_exact_diff = [] 
        self._SDP_high_acc_exact_diff = []
        self._SDP_low_acc_exact_diff = [] 

        self._LR_value_diff= [] 
        self._SDP_high_acc_value_diff = []
        self._SDP_low_acc_value_diff = [] 
        
        self._condition_numbers = []

        self._iterations = 0 

    def run(self, A: np.ndarray, b: np.ndarray, C: np.ndarray, Y_0: np.ndarray, lam_0: np.ndarray, 
            check_residual: bool, use_SDP_solver: bool, print_data: bool, silent: bool): 
        
        if silent:
            warnings.filterwarnings("ignore")
        # Get copies of all problem parameters  
        dt = self._ini_stepsize   
        curr_time = self._initial_time
        next_time = self._initial_time + dt

        n, m, rank = self._n, self._m, self._rank

        np.copyto(self._Y, Y_0)
        np.copyto(self._X,np.matmul(Y_0,Y_0.T))
        np.copyto(self._lam, lam_0) 
        
        reduction_steps = 0
        
        res_0 = residual.resid(n=n, m=m, rank=rank, A=A(curr_time), b=b(curr_time), C=C(curr_time), 
                                                Y=self._Y, lam=self._lam, print_data=print_data)
        if print_data:
            
            f = open(f"results/0.pkl", "wb")  
            pickle.dump([res_0,dt,reduction_steps], f)
            
            f.close() 

        self._LR_res_list.append(res_0)
        self._SDP_high_acc_res_list.append(res_0)
        self._SDP_low_acc_res_list.append(res_0) 

        while curr_time < self._final_time:   
            
            H = self._LinearizedKKTsystem.computeMatrix(A= A(next_time), C=C(next_time), Y=self._Y, lam=self._lam) 
            k = self._LinearizedKKTsystem.computeRhs(A=A(next_time), b= b(next_time), C=C(next_time), Y=self._Y, X=self._X)   
            
            self._condition_numbers.append(np.linalg.cond(H)) 
            start_time = time.time() 
            
            np.copyto(self._linear_KKT_sol, scipy.linalg.solve(H, k, assume_a='sym'))
            np.copyto(self._candidate_Y, np.reshape(self._linear_KKT_sol[:n*rank],(rank,n)).T)
            
            run_time_LR = time.time()-start_time 
            self._LR_runtime += run_time_LR
            
            self._candidate_Y += self._Y
            np.copyto(self._candidate_lam, self._linear_KKT_sol[n*rank:n*rank+m])
             
            res = residual.resid(n=n, m=m, rank=rank, A=A(next_time), b=b(next_time),  C=C(next_time), 
                                            Y=self._candidate_Y, lam=self._candidate_lam, print_data=print_data)
            
            self._LR_res_list.append(res)
            candidate_X = np.matmul(self._candidate_Y,self._candidate_Y.T)

            # if use_SDP_solver:
                
            #     #HIGHEST ACCURACY SOLUTION
            #     run_time_SDP_0, actual_X_0, actual_lam_0 = ip._get_SDP_solution(n, m, A(next_time), b(next_time), C(next_time), TOLERANCE=1.0e-14)
            #     SDP_res_0 = residual.SDP_resid(n=n, m=m, rank=rank, A=A(next_time), b=b(next_time),  C=C(next_time), 
            #                                 X=actual_X_0, lam=actual_lam_0, print_data=print_data)

            #     self._SDP_highest_acc_res_list.append(SDP_res_0)  
            #     self._SDP_highest_acc_runtime += run_time_SDP_0

            #     exact_norm = np.linalg.norm(actual_X_0, 'fro')
            #     exact_value = np.tensordot(C(next_time), actual_X_0, 2)

            #     # HIGH ACCURACY SOLUTION
            #     run_time_SDP_1, actual_X_1, actual_lam_1 = ip._get_SDP_solution(n, m, A(next_time), b(next_time), C(next_time), TOLERANCE=1.0e-10)
            #     SDP_res_1 = residual.SDP_resid(n=n, m=m, rank=rank, A=A(next_time), b=b(next_time),  C=C(next_time), 
            #                                 X=actual_X_1, lam=actual_lam_1, print_data=print_data)

            #     self._SDP_high_acc_runtime += run_time_SDP_1
            #     self._SDP_high_acc_res_list.append(SDP_res_1)  
            #     self._SDP_high_acc_exact_diff.append(np.linalg.norm(actual_X_1-actual_X_0, 'fro')/exact_norm)  
            #     self._SDP_high_acc_value_diff.append(float(np.tensordot(C(next_time), actual_X_1-actual_X_0, 2)/exact_value))

            #     # LOWER ACCURACY SOLUTION
            #     run_time_SDP_2, actual_X_2, actual_lam_2 = ip._get_SDP_solution(n, m, A(next_time), b(next_time), C(next_time), TOLERANCE=1.0e-3)
            #     SDP_res_2 = residual.SDP_resid(n=n, m=m, rank=rank, A=A(next_time), b=b(next_time),  C=C(next_time), 
            #                                 X=actual_X_2, lam=actual_lam_2, print_data=print_data)
                
            #     self._SDP_low_acc_res_list.append(SDP_res_2)  
            #     self._SDP_low_acc_runtime += run_time_SDP_2
            #     self._SDP_low_acc_exact_diff.append(np.linalg.norm(actual_X_2-actual_X_0, 'fro')/exact_norm)  
            #     self._SDP_low_acc_value_diff.append(float(np.tensordot(C(next_time), actual_X_2-actual_X_0, 2)/exact_value))
                
            #     self._LR_exact_diff.append(np.linalg.norm(candidate_X-actual_X_0, 'fro')/exact_norm)  
            #     self._LR_value_diff.append(float(np.tensordot(C(next_time), candidate_X-actual_X_0, 2)/exact_value))

            if print_data: 

                print("\nITERATION", self._iterations) 
                print("running time t", time.time() - start_time) 
                print("TIME", curr_time) 
                print("TARGET TIME", next_time) 
                print("TIME STEP", dt) 
                print("res VS res_tol:", res, self._res_tol ) 

                # print("current Y\n", self._Y) 
                # print("candidate Y\n", self._candidate_Y)  
                # print("candidate X\n", candidate_X)  
                # print("current lam\n", self._lam)
                # print("candidate lam\n", self._candidate_lam)  
                
            elif not silent:

                    print("\nITERATION", self._iterations, end = "\n")  
            
            if check_residual and res>self._res_tol: 

                    dt *= self._gamma1
                    next_time = curr_time + dt 
                    reduction_steps += 1 
                    if print_data:
                        print("reducing stepsize...")
                        print("res =", res)
                        print("VS res tol =", self._res_tol)  
                    
            else:

                # Update the solution
                np.copyto(self._Y, self._candidate_Y)
                np.copyto(self._lam, self._candidate_lam) 
                np.copyto(self._X,np.matmul(self._Y,self._Y.T))  
                
                # Go to the next iteration. 
                # Try to shrink 'delta' a little bit.
                curr_time += dt
                self._iterations += 1
                reduction_steps = 0
                if check_residual:
                    dt = min(self._final_time  - curr_time, self._gamma2 * dt)
                else:
                    dt = min(self._final_time  - curr_time, dt)
                next_time = curr_time+dt 