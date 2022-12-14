from traceback import print_tb
import numpy as np  
import mosek_ipm_solver as mis
import residual
import time

class _IPM_tracker:

    def __init__(self, n: int, m: int):
        """ Constructor pre-allocates and pre-computes persistent
            data structures. """ 

         # Storing algorithm parameters 
        self._m = m
        self._n = n 

        # List of solutions
        self._primal_solutions_list = []

         # variables for runtime
        self._SDP_highest_acc_runtime = 0.0 
        self._SDP_high_acc_runtime = 0.0 
        self._SDP_low_acc_runtime = 0.0 

        # variables for residuals
        self._SDP_highest_acc_res_list = []
        self._SDP_high_acc_res_list = []
        self._SDP_low_acc_res_list = [] 

        # variables for relative distance from solution
        self._LR_exact_diff = [] 
        self._SDP_high_acc_exact_diff = []
        self._SDP_low_acc_exact_diff = [] 

    def run(self, A: np.ndarray, b: np.ndarray, C: np.ndarray, initial_time: float, final_time: float, initial_stepsize: float, PRINT_DATA: bool): 

        iteration = 0
        
        n, m = self._n, self._m 
        dt = initial_stepsize
        next_time = initial_time + dt
        curr_time  = final_time

        run_time_SDP_0, actual_X_0, actual_lam_0 = mis._get_SDP_solution(n, m, A(curr_time), b(curr_time), C(curr_time), TOLERANCE=1.0e-14)
        self._primal_solutions_list.append(actual_X_0)

        start_time =  time.time() 
           
        while curr_time < final_time:   

            try:

                #HIGHEST ACCURACY SOLUTION
                run_time_SDP_0, actual_X_0, actual_lam_0 = mis._get_SDP_solution(n, m, A(next_time), b(next_time), C(next_time), rel_gap_termination_tolerance=1.0e-16)
                self._primal_solutions_list.append(actual_X_0)

                SDP_res_0 = residual.SDP_resid(n=n, m=m, A=A(next_time), b=b(next_time),  C=C(next_time), 
                                            X=actual_X_0, lam=actual_lam_0)
                self._SDP_highest_acc_res_list.append(max(SDP_res_0))  
            
                self._SDP_highest_acc_runtime += run_time_SDP_0
                exact_norm = np.linalg.norm(actual_X_0, 'fro') 

                # HIGH ACCURACY SOLUTION
                run_time_SDP_1, actual_X_1, actual_lam_1 = mis._get_SDP_solution(n, m, A(next_time), b(next_time), C(next_time), rel_gap_termination_tolerance=1.0e-10)
                SDP_res_1 = residual.SDP_resid(n=n, m=m, A=A(next_time), b=b(next_time),  C=C(next_time), 
                                            X=actual_X_1, lam=actual_lam_1)

                self._SDP_high_acc_runtime += run_time_SDP_1
                self._SDP_high_acc_res_list.append(max(SDP_res_1))  
                self._SDP_high_acc_exact_diff.append(np.linalg.norm(actual_X_1-actual_X_0, 'fro')/exact_norm)   

                # LOWER ACCURACY SOLUTION
                run_time_SDP_2, actual_X_2, actual_lam_2 = mis._get_SDP_solution(n, m, A(next_time), b(next_time), C(next_time), rel_gap_termination_tolerance=1.0e-3)
                SDP_res_2 = residual.SDP_resid(n=n, m=m, A=A(next_time), b=b(next_time),  C=C(next_time), 
                                            X=actual_X_2, lam=actual_lam_2)

                self._SDP_low_acc_res_list.append(max(SDP_res_2))
                self._SDP_low_acc_runtime += run_time_SDP_2
                self._SDP_low_acc_exact_diff.append(np.linalg.norm(actual_X_2-actual_X_0, 'fro')/exact_norm)   
            except: 
                self._primal_solutions_list.append(np.zeros((n,n))) 
            curr_time += dt
            next_time = curr_time+dt 

            if not PRINT_DATA and iteration%10==0: print("\nITERATION", iteration) 

            if PRINT_DATA: 
                print("\nITERATION", iteration) 
                print("running time t", time.time() - start_time) 
                print("TIME", curr_time) 
                print("TARGET TIME", next_time) 
                print("TIME STEP", dt) 
            
            iteration += 1