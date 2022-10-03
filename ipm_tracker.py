import numpy as np  
import sdp_solver as ss
import residual as residual
import linearized_kkt as lk 

class _IPM_tracker:

    def __init__(self, n: int, m: int, rank: int, params: dict, ini_stepsize: float):
        """ Constructor pre-allocates and pre-computes persistent
            data structures. """ 

         # Storing algorithm parameters 
        self._m = m
        self._n = n
        self._rank = rank

        self._final_time = float(params["problem"]["final_time"])
        self._initial_time = float(params["problem"]["initial_time"]) 
        self._ini_stepsize = ini_stepsize

        # self._primal_solutions_list = []

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
        
         # variables for relative distance from optimal value
        self._LR_value_diff= [] 
        self._SDP_high_acc_value_diff = []
        self._SDP_low_acc_value_diff = []  


    def run(self, A: np.ndarray, b: np.ndarray, C: np.ndarray): 
        
        n, m, rank = self._n, self._m, self._rank
        dt = self._ini_stepsize    
        next_time = self._initial_time + dt
        curr_time  = self._initial_time 

        while curr_time < self._final_time:   

            #HIGHEST ACCURACY SOLUTION
            run_time_SDP_0, actual_X_0, actual_lam_0 = ss._get_SDP_solution(n, m, A(next_time), b(next_time), C(next_time), TOLERANCE=1.0e-14)
            # self._primal_solutions_list.append(actual_X_0)
            SDP_res_0 = residual.SDP_resid(n=n, m=m, rank=rank, A=A(next_time), b=b(next_time),  C=C(next_time), 
                                        X=actual_X_0, lam=actual_lam_0)

            self._SDP_highest_acc_res_list.append(SDP_res_0)  
            self._SDP_highest_acc_runtime += run_time_SDP_0

            exact_norm = np.linalg.norm(actual_X_0, 'fro')
            exact_value = np.tensordot(C(next_time), actual_X_0, 2)

            # HIGH ACCURACY SOLUTION
            run_time_SDP_1, actual_X_1, actual_lam_1 = ss._get_SDP_solution(n, m, A(next_time), b(next_time), C(next_time), TOLERANCE=1.0e-10)
            SDP_res_1 = residual.SDP_resid(n=n, m=m, rank=rank, A=A(next_time), b=b(next_time),  C=C(next_time), 
                                        X=actual_X_1, lam=actual_lam_1)

            self._SDP_high_acc_runtime += run_time_SDP_1
            self._SDP_high_acc_res_list.append(SDP_res_1)  
            self._SDP_high_acc_exact_diff.append(np.linalg.norm(actual_X_1-actual_X_0, 'fro')/exact_norm)  
            self._SDP_high_acc_value_diff.append(float(np.tensordot(C(next_time), actual_X_1-actual_X_0, 2)/exact_value))

            # LOWER ACCURACY SOLUTION
            run_time_SDP_2, actual_X_2, actual_lam_2 = ss._get_SDP_solution(n, m, A(next_time), b(next_time), C(next_time), TOLERANCE=1.0e-3)
            SDP_res_2 = residual.SDP_resid(n=n, m=m, rank=rank, A=A(next_time), b=b(next_time),  C=C(next_time), 
                                        X=actual_X_2, lam=actual_lam_2)

            self._SDP_low_acc_res_list.append(SDP_res_2)  
            self._SDP_low_acc_runtime += run_time_SDP_2
            self._SDP_low_acc_exact_diff.append(np.linalg.norm(actual_X_2-actual_X_0, 'fro')/exact_norm)  
            self._SDP_low_acc_value_diff.append(float(np.tensordot(C(next_time), actual_X_2-actual_X_0, 2)/exact_value))

            curr_time += dt