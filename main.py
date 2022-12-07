import predictor_corrector as pc 
import create_examples_data  as ex 
import mosek_ipm_solver as ip 
import experiments.ipm_tracker as it
import parameters as par 
import numpy as np   

params = par.getParameters(print_par=False) 
initial_time = float(params["problem"]["initial_time"])

# n = int(input("Enter n:"))
# m = int(input("Enter m:"))
# problem = ex._ProblemCreator(n,m)
# n, m, A, b, C = problem._create_random_problem(n,m) 

n = int(input("Enter n:"))
problem = ex._ProblemCreator(n,n)
n, m, A, b, C = problem._create_MaxCut(n)  

Y_0, rank, lam_0  = ip._get_initial_point(n=n, m=m, A = A(initial_time), b=b(initial_time), C=C(initial_time), TOLERANCE=1.0e-10)

ini_stepsize = 0.01
res_tol = 1e-4
predcorr = pc._PredictorCorrector(n=n, m=m, rank=rank, params=params, ini_stepsize=ini_stepsize, res_tol=res_tol) 
predcorr.run(A, b, C, Y_0, lam_0, STEPSIZE_TUNING=False, PRINT_DATA=False)
 
ipm_track = it._IPM_tracker(n=n, m=m, params=params, ini_stepsize = ini_stepsize)
ipm_track.run(A, b, C, PRINT_DATA=False)

print("Average LR runtime", predcorr._LR_runtime) 
print("Average LR residual", np.mean(predcorr._LR_residuals)) 
print("Average SDP [low accuracy] runtime", ipm_track._SDP_low_acc_runtime) 
print("Average SDP [low accuracy] residual", np.mean(ipm_track._SDP_low_acc_res_list)) 
print("Average SDP [high accuracy] runtime", ipm_track._SDP_high_acc_runtime) 
print("Average SDP [high accuracy] residual", np.mean(ipm_track._SDP_high_acc_res_list)) 
print("Average SDP [very high accuracy] runtime", ipm_track._SDP_highest_acc_runtime) 
print("Average SDP [very high accuracy] residual", np.mean(ipm_track._SDP_highest_acc_res_list)) 