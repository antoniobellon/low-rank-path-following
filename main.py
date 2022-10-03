 
import predictor_corrector as pc 
import create_examples_data  as ex 
import sdp_solver as ip 
import parameters as par 
import numpy as np 
import create_examples_data  
from threading import Timer 
import os
import pickle

params = par.getParameters(print_par=False) 
initial_time = float(params["problem"]["initial_time"])

# n, m, A, b, C = ex._choose_example() 

# n = int(input("Enter n:"))
# m = int(input("Enter m:"))
# n, m, A, b, C = ex._create_example(n,m) 

n = int(input("Enter n:"))
problem = create_examples_data._ProblemCreator(n,n)
n, m, A, b, C = problem._create_MaxCut(n) 
f = open('MaxCut experiments/exp_100.pkl', 'rb')
pickled_exp = pickle.load(f) 
C_init=pickled_exp['exp_info']['exp_data']['C_init'][29]
C_pert=pickled_exp['exp_info']['exp_data']['C_pert'][29]
def C(time: np.float):
    return C_init+time*C_pert

Y_0,  rank, lam_0  = ip._get_initial_point(n=n, m=m, A = A(initial_time), b=b(initial_time), C=C(initial_time), TOLERANCE=1.0e-10)
 
predcorr = pc._PredictorCorrector(n=n, m=m, rank=rank, params=params, ini_stepsize = 0.01, res_tol=1e-4) 
predcorr.run(A, b, C, Y_0, lam_0, check_residual=False, use_SDP_solver = True, print_data=True, silent = False)
print(predcorr._LR_value_diff)
f = open('debug.pkl', 'wb')
pickle.dump(predcorr, f, pickle.HIGHEST_PROTOCOL)
f.close
# vis.visualize_sol(params=params)
print("TOTAL runtime", predcorr._LR_runtime)
print("TOTAL SDP runtime", predcorr._SDP_high_acc_runtime)

print("TOTAL residual", np.mean(predcorr._LR_res_list))
print("TOTAL SDP residual", np.mean(predcorr._SDP_high_acc_res_list)) 
