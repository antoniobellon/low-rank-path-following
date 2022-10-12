import numpy as np
import time 
import pickle

import create_examples_data  as ex
import predictor_corrector as pc  
import ipm_tracker as it
import parameters as par  
import mosek_ipm_solver as ss 

params = par.getParameters(print_par=True) 

MAX_POWER = 6
INIT_STEP = 0.01
NR_TOLS = 4

doubles = [10**k for k in range(1, MAX_POWER)]
res_tolerances = [10**-k for k in np.linspace(2, 2*NR_TOLS, NR_TOLS)]


exp_dict = {'LR_res'                  : {str(d) : [] for d in doubles}, 
            'SDP_highest_acc_res'     : {str(d) : [] for d in doubles}, 
            'SDP_high_acc_res'        : {str(d) : [] for d in doubles}, 
            'SDP_low_acc_res'         : {str(d) : [] for d in doubles}, 
            'LR_run'                  : {str(d) : [] for d in doubles}, 
            'SDP_highest_acc_run'     : {str(d) : [] for d in doubles}, 
            'SDP_high_acc_run'        : {str(d) : [] for d in doubles}, 
            'SDP_low_acc_run'         : {str(d) : [] for d in doubles}, 
            'LR_exact_diff'           : {str(d) : [] for d in doubles}, 
            'SDP_high_acc_exact_diff' : {str(d) : [] for d in doubles}, 
            'SDP_low_acc_exact_diff'  : {str(d) : [] for d in doubles}, 
            'LR_STEPSIZE_TUNING_run'  : {str(d) : [] for d in res_tolerances}
            } 
 
exp_info = {'exp_data'               : {'A_init' : [], 
                                        'A_pert' : [], 
                                        'b_init' : [],
                                        'b_pert' : [], 
                                        'C_init' : [], 
                                        'C_pert' : []},
            'primal_solution'        : [],
            'dual_solution'          : [],
            'rank'                   : []
            }

def run_experiment(PROBLEM_SIZE, SAMPLE_SIZE): 

    # Check if the file 'MaxCut experiments/exp_%d.pkl'%(PROBLEM_SIZE) exists and ...
    file_there = True
    try:
        f = open('MaxCut experiments/exp_%d.pkl'%(PROBLEM_SIZE)) 
    except IOError:
        file_there = False
    
    # ... if it does not exist, create it from scratch
    if not file_there:
        f = open('MaxCut experiments/exp_%d.pkl'%(PROBLEM_SIZE), 'wb')
        pickle.dump({'exp_dict':exp_dict, 'exp_dict':exp_dict, 'exp_info':exp_info}, f, protocol=pickle.HIGHEST_PROTOCOL)
        f.close()

    # Create list to store the running times of each experiment
    running_times = []

    # Start solving SAMPLE_SIZE number of random examples
    for k in range(SAMPLE_SIZE):

        f = open('MaxCut experiments/exp_%d.pkl'%(PROBLEM_SIZE),'rb')
        pickled_exp = pickle.load(f) 
        f.close()

        # Give a rough estimate of termination time
        if k > 0:
            mean_time = np.mean(running_times)
            time_string = np.str(np.int(np.divmod((SAMPLE_SIZE-k)*mean_time,60)[0]))+"."+ np.str(np.int(np.divmod((SAMPLE_SIZE-k)*mean_time,60)[1]))
            print("expected time to termination:", time_string) 
            print("")
        before = time.time()

        print("    K =", k)
        
        # Create a problem
        problem = ex._ProblemCreator(PROBLEM_SIZE, PROBLEM_SIZE)
        n, m, A, b, C = problem._create_MaxCut(PROBLEM_SIZE) 

        # Find initial points
        initial_time = float(params["problem"]["initial_time"])
        final_time = float(params["problem"]["final_time"])
        Y_0, rank, lam_0  = ss._get_initial_point(n=n, m=m, A = A(initial_time), b=b(initial_time), C=C(initial_time), TOLERANCE=1.0e-14)
        
        # Store problem data
        pickled_exp['exp_info']['exp_data']['C_init'].append(problem._C_init) 
        pickled_exp['exp_info']['exp_data']['C_pert'].append(problem._C_pert) 
        pickled_exp['exp_info']['rank'].append(rank)

        # Start experiments with STEPSIZE_TUNING=FALSE and try eith different stepsize of size 1/sub
        for sub in doubles:
            
            print("    subdivision =", sub)

            # Solve TV-SDP using Low Rank Path-Following
            predcorr = pc._PredictorCorrector(n=n, m=m, rank=rank, params=params, ini_stepsize = 1/sub, res_tol=1e-4) 
            predcorr.run(A, b, C, Y_0, lam_0, STEPSIZE_TUNING=False, PRINT_DATA=False) 

            pickled_exp['exp_dict']['LR_res'][str(sub)].append(np.mean(predcorr._LR_residuals))
            pickled_exp['exp_dict']['LR_run'][str(sub)].append(np.mean(predcorr._LR_runtime))
            pickled_exp['exp_info']['primal_solution'].append(predcorr._primal_solutions_list) 

            ipm_track = it._IPM_tracker(n=n, m=m, rank=rank, params=params, ini_stepsize = 1/sub)
            ipm_track.run(A, b, C, PRINT_DATA=False)

            pickled_exp['exp_dict']['SDP_highest_acc_res'][str(sub)].append(np.mean(ipm_track._SDP_highest_acc_res_list))
            pickled_exp['exp_dict']['SDP_high_acc_res'][str(sub)].append(np.mean(ipm_track._SDP_high_acc_res_list))
            pickled_exp['exp_dict']['SDP_low_acc_res'][str(sub)].append(np.mean(ipm_track._SDP_low_acc_res_list))
            
            pickled_exp['exp_dict']['SDP_highest_acc_run'][str(sub)].append(np.mean(ipm_track._SDP_highest_acc_runtime))
            pickled_exp['exp_dict']['SDP_high_acc_run'][str(sub)].append(np.mean(ipm_track._SDP_high_acc_runtime))
            pickled_exp['exp_dict']['SDP_low_acc_run'][str(sub)].append(np.mean(ipm_track._SDP_low_acc_runtime))

            pickled_exp['exp_dict']['SDP_high_acc_exact_diff'][str(sub)].append(np.mean(ipm_track._SDP_high_acc_exact_diff))
            pickled_exp['exp_dict']['SDP_low_acc_exact_diff'][str(sub)].append(np.mean(ipm_track._SDP_low_acc_exact_diff)) 
            
            LR_exact_diff = []
            for i in range(sub+1):
                actual_X_0 = ipm_track._primal_solutions_list[i]
                exact_norm = np.linalg.norm(actual_X_0, 'fro') 
                LR_exact_diff.append(np.linalg.norm(predcorr._primal_solutions_list[i]-actual_X_0, 'fro')/exact_norm)   
            pickled_exp['exp_dict']['LR_exact_diff'][str(sub)].append(np.mean(LR_exact_diff)) 

        for tol in res_tolerances:
            
            print("residual tolerance =", tol)

            predcorr = pc._PredictorCorrector(n=n, m=m, rank=rank, params=params, ini_stepsize = INIT_STEP, res_tol=tol) 
            predcorr.run(A, b, C, Y_0, lam_0, STEPSIZE_TUNING=True, PRINT_DATA=True) 
           
            pickled_exp['exp_dict']['LR_STEPSIZE_TUNING_run'][str(tol)].append(np.mean(predcorr._LR_runtime))

        running_times.append(time.time() - before)

        f = open('MaxCut experiments/exp_%d.pkl'%(PROBLEM_SIZE), 'wb')
        pickle.dump(pickled_exp, f, pickle.HIGHEST_PROTOCOL)
        f.close

run_experiment(PROBLEM_SIZE=int(input("Enter PROBLEM_SIZE:")), SAMPLE_SIZE=int(input("Enter SAMPLE_SIZE:")))