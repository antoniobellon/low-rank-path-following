import numpy as np   
import create_examples_data  as ex 
import predictor_corrector as pc 
import mosek_ipm_solver as ip  
 
n = int(input("Enter n:"))
m = int(input("Enter m:"))
problem = ex._ProblemCreator()
n, m, A, b, C = problem._create_random_problem(n,m) 
 
initial_time     = 0
final_time       = 1
initial_stepsize = 0.01
res_tol          = 1e-4

Y_0, rank, lam_0  = ip._get_initial_point(n=n, m=m, A = A(initial_time), b=b(initial_time), C=C(initial_time), TOLERANCE=1.0e-10)

predcorr = pc._PredictorCorrector(n=n, m=m, rank=rank) 
predcorr.run(A, b, C, Y_0, lam_0, 
             initial_time     = initial_time,
             final_time       = final_time,
             initial_stepsize = initial_stepsize,
             gamma_1          = 0.5,
             gamma_2          = 1.5,
             res_tol          = res_tol,
             STEPSIZE_TUNING  = True,
             PRINT_DATA       = True)  

print("Average runtime", predcorr._LR_runtime) 
print("Average residual", np.mean(predcorr._LR_residuals))  