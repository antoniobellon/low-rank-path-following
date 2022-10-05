import numpy as np
import time 
import pickle

import create_examples_data  as ex
import predictor_corrector as pc  
import parameters as par 
import sdp_tracking as st 
import sdp_solver as ss 

params = par.getParameters(print_par=True) 

MAX_POWER = 7
INIT_STEP = 0.01
NR_TOLS = 4

res_tolerances = [10**-k for k in np.linspace(2, 2*NR_TOLS, NR_TOLS)]
doubles = [10**k for k in range(1, MAX_POWER)]

# sub_dict = {'LR_res'                  : {str(d) : [] for d in doubles}, 
#             'SDP_highest_acc_res'     : {str(d) : [] for d in doubles}, 
#             'SDP_high_acc_res'        : {str(d) : [] for d in doubles}, 
#             'SDP_low_acc_res'         : {str(d) : [] for d in doubles}, 
#             'LR_run'                  : {str(d) : [] for d in doubles}, 
#             'SDP_highest_acc_run'     : {str(d) : [] for d in doubles}, 
#             'SDP_high_acc_run'        : {str(d) : [] for d in doubles}, 
#             'SDP_low_acc_run'         : {str(d) : [] for d in doubles}, 
#             'LR_exact_diff'           : {str(d) : [] for d in doubles}, 
#             'SDP_high_acc_exact_diff' : {str(d) : [] for d in doubles}, 
#             'SDP_low_acc_exact_diff'  : {str(d) : [] for d in doubles}, 
#             } 

# res_dict = {'LR_run'                : {str(d) : [] for d in res_tolerances},
#             'SDP_highest_acc_run'   : {str(d) : [] for d in res_tolerances},
#             'SDP_high_acc_run'      : {str(d) : [] for d in res_tolerances},
#             'SDP_low_acc_run'       : {str(d) : [] for d in res_tolerances}
#             } 

# exp_info = {'exp_data'               : {'A_init' : [], 
#                                         'A_pert' : [], 
#                                         'b_init' : [],
#                                         'b_pert' : [], 
#                                         'C_init' : [], 
#                                         'C_pert' : []},
#             'rank'                   : [],
#             'LR_opt_value'           : {str(d) : [] for d in doubles},
#             'SDP_high_acc_opt_value' : {str(d) : [] for d in doubles},
#             'SDP_low_acc_opt_value'  : {str(d) : [] for d in doubles}
#             }
 
def run_experiment(SAMPLE_SIZE, PROBLEM_SIZE=100): 
   
    # file_there = True

    # try:
    #     f = open('experiments/exp_%d.pkl'%(PROBLEM_SIZE)) 
    # except IOError:
    #     file_there = False
    
    # if not file_there:
    #     f = open('experiments/exp_%d.pkl'%(PROBLEM_SIZE), 'wb')
    #     pickle.dump({'sub_dict':sub_dict, 'res_dict':res_dict, 'exp_info':exp_info}, f, protocol=pickle.HIGHEST_PROTOCOL)
    #     f.close()

    times = []

    for k in range(SAMPLE_SIZE):

        f = open('MaxCut experiments/exp_%d.pkl'%(PROBLEM_SIZE),'rb')
        pickled_exp = pickle.load(f) 
        f.close()

        if k > 0:
            mean_time = np.mean(times)
            time_string = np.str(np.int(np.divmod((SAMPLE_SIZE-k)*mean_time,60)[0]))+"."+ np.str(np.int(np.divmod((SAMPLE_SIZE-k)*mean_time,60)[1]))
            print("expected time to termination:", time_string) 
            print("")
        before = time.time()

        print("    K =", k)
        
        problem = ex._ProblemCreator(PROBLEM_SIZE, PROBLEM_SIZE)
        n, m, A, b, C_unused = problem._create_MaxCut(PROBLEM_SIZE) 

        C_init = pickled_exp['exp_info']['exp_data']['C_init'][k]
        C_pert = pickled_exp['exp_info']['exp_data']['C_pert'][k]

        def C(time: np.float): 
            return C_init+time*C_pert

        # n, m, A, b, C = problem._create_bigeasy(PROBLEM_SIZE) 

        initial_time = float(params["problem"]["initial_time"])
        Y_0, rank, lam_0  = ss._get_initial_point(n=n, m=m, A = A(initial_time), b=b(initial_time), C=C(initial_time), TOLERANCE=1.0e-14)
        
        # pickled_exp['exp_info']['exp_data']['C_init'].append(problem._C_init) 
        # pickled_exp['exp_info']['exp_data']['C_pert'].append(problem._C_pert) 
        # pickled_exp['exp_info']['rank'].append(rank)

        for sub in doubles:
            
            print("    subdivision =", sub)

            predcorr = pc._PredictorCorrector(n=n, m=m, rank=rank, params=params, ini_stepsize = 1/sub, res_tol=1e-4) 
            predcorr.run(A, b, C, Y_0, lam_0, check_residual=False, use_SDP_solver = True, print_data=False, silent=True) 
            
        #     pickled_exp['sub_dict']['LR_res'][str(sub)].append(np.mean(predcorr._LR_res_list))
        #     pickled_exp['sub_dict']['SDP_highest_acc_res'][str(sub)].append(np.mean(predcorr._SDP_highest_acc_res_list))
        #     pickled_exp['sub_dict']['SDP_high_acc_res'][str(sub)].append(np.mean(predcorr._SDP_high_acc_res_list))
        #     pickled_exp['sub_dict']['SDP_low_acc_res'][str(sub)].append(np.mean(predcorr._SDP_low_acc_res_list))

        #     pickled_exp['sub_dict']['LR_run'][str(sub)].append(np.mean(predcorr._LR_runtime))
        #     pickled_exp['sub_dict']['SDP_highest_acc_run'][str(sub)].append(np.mean(predcorr._SDP_highest_acc_runtime))
        #     pickled_exp['sub_dict']['SDP_high_acc_run'][str(sub)].append(np.mean(predcorr._SDP_high_acc_runtime))
        #     pickled_exp['sub_dict']['SDP_low_acc_run'][str(sub)].append(np.mean(predcorr._SDP_low_acc_runtime))

        #     pickled_exp['sub_dict']['LR_exact_diff'][str(sub)].append(np.mean(predcorr._LR_exact_diff)) 
        #     pickled_exp['sub_dict']['SDP_high_acc_exact_diff'][str(sub)].append(np.mean(predcorr._SDP_high_acc_exact_diff))
            # pickled_exp['sub_dict']['SDP_low_acc_exact_diff'][str(sub)].append(np.mean(predcorr._SDP_low_acc_exact_diff))

            pickled_exp['exp_info']['LR_opt_value'][str(sub)].append(np.mean(predcorr._LR_value_diff))
            pickled_exp['exp_info']['SDP_high_acc_opt_value'][str(sub)].append(np.mean(predcorr._SDP_high_acc_value_diff))
            pickled_exp['exp_info']['SDP_low_acc_opt_value'][str(sub)].append(np.mean(predcorr._SDP_low_acc_value_diff))

            # pickled_exp['exp_info']['LR_opt_value'][str(sub)] = []
            # pickled_exp['exp_info']['SDP_high_acc_opt_value'][str(sub)] = []
            # pickled_exp['exp_info']['SDP_low_acc_opt_value'][str(sub)] = []

        # sdptrack = st._SDPtracking(n=n, m=m, rank=rank, params=params, ini_stepsize = INIT_STEP)
        # sdptrack.run(A, b, C)

        # for tol in res_tolerances:
             
        #     print("residual tolerance =", tol)

        #     predcorr = pc._PredictorCorrector(n=n, m=m, rank=rank, params=params, ini_stepsize = INIT_STEP, res_tol=tol) 
        #     predcorr.run(A, b, C, Y_0, lam_0, check_residual=True, use_SDP_solver = False, print_data=False, silent=False) 
        
        #     pickled_exp['res_dict']['LR_run'][str(tol)].append(np.mean(predcorr._LR_runtime)) 
        #     pickled_exp['res_dict']['SDP_high_acc_run'][str(tol)].append(np.mean(sdptrack._SDP_high_acc_runtime))
        #     pickled_exp['res_dict']['SDP_low_acc_run'][str(tol)].append(np.mean(sdptrack._SDP_low_acc_runtime))
 
        times.append(time.time() - before)

        f = open('MaxCut experiments/exp_%d.pkl'%(PROBLEM_SIZE), 'wb')
        pickle.dump(pickled_exp, f, pickle.HIGHEST_PROTOCOL)
        f.close
         
run_experiment(100)