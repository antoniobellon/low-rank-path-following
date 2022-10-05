from cgi import print_arguments 
import sdp_solver as ip
import numpy as np 
import scipy
import residual as residual
import pickle
import time
import math 
import warnings
 
 
class _SDPtracking:

    def __init__(self, n: int, m: int, rank: int, params: dict, ini_stepsize: float):
        """ Constructor pre-allocates and pre-computes persistent
            data structures. """ 

        self._n = n
        self._m = m
        self._rank = rank
 
        self._X = np.zeros((n, n))
 
        self._ini_stepsize = ini_stepsize
        self._final_time = float(params["problem"]["final_time"])
        self._initial_time = float(params["problem"]["initial_time"]) 

        self._SDP_highest_acc_runtime = 0.0 
        self._SDP_high_acc_runtime = 0.0 
        self._SDP_low_acc_runtime = 0.0   
 
    def run(self, A: np.ndarray, b: np.ndarray, C: np.ndarray): 
        
        print("solving sdp...")

       
        # Get copies of all problem parameters  
        dt = self._ini_stepsize   
        curr_time = self._initial_time
        next_time = self._initial_time + dt
       
        n, m, rank = self._n, self._m, self._rank
        
 
        iteration = 0   
        
        while curr_time < self._final_time:  
                
                print("current time:", curr_time)
                # HIGHEST ACCURACY SOLUTION
                # run_time_SDP_0, actual_X_0, actual_lam_0 = ip._get_SDP_solution(n, m, A(next_time), b(next_time), C(next_time), TOLERANCE=1.0e-16)
                # self._SDP_highest_acc_runtime += run_time_SDP_0
               
                # HIGH ACCURACY SOLUTION
                run_time_SDP_1, actual_X_1, actual_lam_1 = ip._get_SDP_solution(n, m, A(next_time), b(next_time), C(next_time), TOLERANCE=1.0e-10)
                self._SDP_high_acc_runtime += run_time_SDP_1
      
                # LOWER ACCURACY SOLUTION
                run_time_SDP_2, actual_X_2, actual_lam_2 = ip._get_SDP_solution(n, m, A(next_time), b(next_time), C(next_time), TOLERANCE=1.0e-3)
                self._SDP_low_acc_runtime += run_time_SDP_2 
                
                # Go to the next iteration.  
                curr_time += dt
                iteration += 1 
                dt = min(self._final_time  - curr_time, dt)
                next_time = curr_time+dt 
            
            

 
 
 
 
