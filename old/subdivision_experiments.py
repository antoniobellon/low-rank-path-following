from dataclasses import asdict
from traceback import print_tb
from operator import add
import matplotlib.pyplot as plt 
import predictor_corrector as pc 
import sdp_tracking as sdp
import plotly.express as px
import create_examples_data  as ex
import sdp_solver as ip 
import parameters as par 
import pandas as pd
import numpy as np
import time
import csv
import os

font = {
        'family': 'serif',
        'color' : 'black',
        'weight': 'normal',
        'size'  :  9
        } 

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.sans-serif": ["Helvetica"]}
    )  

params = par.getParameters(print_par=True) 

MAX_POWER = 9
doubles = [2**k for k in range(1, MAX_POWER)]
 
def start_experiment(PROBLEM_SIZE=int(input("Enter PROBLEM_SIZE:")),SAMPLE_SIZE=int(input("Enter SAMPLE_SIZE:"))):
    PROBLEM_SIZE, SAMPLE_SIZE, res_dict, run_dict, condition_nrs = run_experiment(PROBLEM_SIZE, SAMPLE_SIZE)   
    return PROBLEM_SIZE, SAMPLE_SIZE, res_dict, run_dict, condition_nrs

def run_experiment(PROBLEM_SIZE, SAMPLE_SIZE): 
     
    res_dict = {'LR_res'                : {2**k : [] for k in range(1, MAX_POWER)},
                'LR_res_mean'           : [],
                'LR_res_std'            : [],
                'SDP_high_acc_res'      : {2**k : [] for k in range(1, MAX_POWER)},
                'SDP_high_acc_res_mean' : [],
                'SDP_high_acc_res_std'  : [],
                'SDP_low_acc_res'       : {2**k : [] for k in range(1, MAX_POWER)},
                'SDP_low_acc_res_mean'  : [],
                'SDP_low_acc_res_std'   : []
               } 

    run_dict = {'LR_run'                : {2**k : [] for k in range(1, MAX_POWER)},
                'LR_run_mean'           : [],
                'LR_run_std'            : [],
                'SDP_high_acc_run'      : {2**k : [] for k in range(1, MAX_POWER)},
                'SDP_high_acc_run_mean' : [],
                'SDP_high_acc_run_std'  : [],
                'SDP_low_acc_run'       : {2**k : [] for k in range(1, MAX_POWER)},
                'SDP_low_acc_run_mean'  : [],
                'SDP_low_acc_run_std'   : []
                } 

    condition_nrs = {2**k : [] for k in range(1, MAX_POWER)}
    times = []

    for k in range(SAMPLE_SIZE):
        mean_time = np.mean(times)
        if k > 0:
            time_string = np.str(np.int(np.divmod((SAMPLE_SIZE-k)*mean_time,60)[0]))+"."+ np.str(np.int(np.divmod((SAMPLE_SIZE-k)*mean_time,60)[1]))
            print("expected time to termination:",time_string) 
        before = time.time()

        print("    K =", k)

        n, m, A, b, C = ex._create_MaxCut(PROBLEM_SIZE) 
        initial_time = float(params["problem"]["initial_time"])
        Y_0, rank, lam_0  = ip._get_initial_point(n=n, m=m, A = A(initial_time), b=b(initial_time), C=C(initial_time), TOLERANCE=1.0e-10)

        for subdivision in doubles:
            
            print("    subdivision = ", subdivision)

            predcorr = pc._PredictorCorrector(n=n, m=m, rank=rank, params=params, ini_stepsize = 1/subdivision) 
            predcorr.run(A, b, C, Y_0, lam_0, check_residual=False, use_SDP_solver = True, print_data=False, silent=True) 
            
            res_dict['LR_res'          ][subdivision].append(np.mean(predcorr._LR_res_list))
            res_dict['SDP_high_acc_res'][subdivision].append(np.mean(predcorr._SDP_high_acc_res_list))
            res_dict['SDP_low_acc_res' ][subdivision].append(np.mean(predcorr._SDP_low_acc_res_list))

            run_dict['LR_run'          ][subdivision].append(predcorr._LR_runtime) 
            run_dict['SDP_high_acc_run'][subdivision].append(predcorr._SDP_high_acc_runtime) 
            run_dict['SDP_low_acc_run' ][subdivision].append(predcorr._SDP_low_acc_runtime) 

            condition_nrs[subdivision].append(1/np.mean(predcorr._condition_numbers))

        times.append(time.time()-before)

    LR_res_data = res_dict['LR_res'          ]
    SDP_high_acc_res_data = res_dict['SDP_high_acc_res']
    SDP_low_acc_res_data = res_dict['SDP_low_acc_res']

    for subdivision in doubles:

        res_dict['LR_res_mean'          ].append(np.mean(res_dict['LR_res'][subdivision]))
        res_dict['LR_res_std'           ].append(np.std(res_dict['LR_res'][subdivision]))
        res_dict['SDP_high_acc_res_mean'].append(np.mean(res_dict['SDP_high_acc_res'][subdivision]))
        res_dict['SDP_high_acc_res_std' ].append(np.std(res_dict['SDP_high_acc_res'][subdivision]))
        res_dict['SDP_low_acc_res_mean' ].append(np.mean(res_dict['SDP_low_acc_res'][subdivision]))
        res_dict['SDP_low_acc_res_std'  ].append(np.std(res_dict['SDP_low_acc_res'][subdivision]))

        run_dict['LR_run_mean'          ].append(np.mean(run_dict['LR_run'][subdivision]))
        run_dict['LR_run_std'           ].append(np.std(run_dict['LR_run'][subdivision]))
        run_dict['SDP_high_acc_run_mean'].append(np.mean(run_dict['SDP_high_acc_run'][subdivision]))
        run_dict['SDP_high_acc_run_std' ].append(np.std(run_dict['SDP_high_acc_run'][subdivision]))
        run_dict['SDP_low_acc_run_mean' ].append(np.mean(run_dict['SDP_low_acc_run'][subdivision]))
        run_dict['SDP_low_acc_run_std'  ].append(np.std(run_dict['SDP_low_acc_run'][subdivision]))

    res_dict['LR_res_upper_bound'] = [x + y for (x,y) in zip(res_dict['LR_res_mean'], res_dict['LR_res_std'])]
    res_dict['LR_res_lower_bound'] = [x - y for (x,y) in zip(res_dict['LR_res_mean'], res_dict['LR_res_std'])]
    res_dict['SDP_high_acc_res_upper_bound'] = [x + y for (x,y) in zip(res_dict['SDP_high_acc_res_mean'], res_dict['SDP_high_acc_res_std'])]
    res_dict['SDP_high_acc_res_lower_bound'] = [x - y for (x,y) in zip(res_dict['SDP_high_acc_res_mean'], res_dict['SDP_high_acc_res_std'])]
    res_dict['SDP_low_acc_res_upper_bound'] = [x + y for (x,y) in zip(res_dict['SDP_low_acc_res_mean'], res_dict['SDP_low_acc_res_std'])]
    res_dict['SDP_low_acc_res_lower_bound'] = [x - y for (x,y) in zip(res_dict['SDP_low_acc_res_mean'], res_dict['SDP_low_acc_res_std'])]

    run_dict['LR_run_upper_bound'] = [x + y for (x,y) in zip(run_dict['LR_run_mean'], run_dict['LR_run_std'])]
    run_dict['LR_run_lower_bound'] = [x - y for (x,y) in zip(run_dict['LR_run_mean'], run_dict['LR_run_std'])]
    run_dict['SDP_high_acc_run_upper_bound'] = [x + y for (x,y) in zip(run_dict['SDP_high_acc_run_mean'], run_dict['SDP_high_acc_run_std'])]
    run_dict['SDP_high_acc_run_lower_bound'] = [x - y for (x,y) in zip(run_dict['SDP_high_acc_run_mean'], run_dict['SDP_high_acc_run_std'])]
    run_dict['SDP_low_acc_run_upper_bound'] = [x + y for (x,y) in zip(run_dict['SDP_low_acc_run_mean'], run_dict['SDP_low_acc_run_std'])]
    run_dict['SDP_low_acc_run_lower_bound'] = [x - y for (x,y) in zip(run_dict['SDP_low_acc_run_mean'], run_dict['SDP_low_acc_run_std'])]

    return PROBLEM_SIZE, SAMPLE_SIZE, res_dict, run_dict, condition_nrs

def plot_residual_experiments(PROBLEM_SIZE, SAMPLE_SIZE, res_dict):

    products_list = [[1,2], [4,1],[1,50], [4,2],[1,3], [4,3],[1,-50], [4,4]]

    df = pd.DataFrame (products_list, columns = ['product_name', 'my_name'])
    fig = px.box(df, x='product_name', y="my_name")
    fig.show()  
    ax = plt.subplot()  
  
    print("Plotting data...")

    ax.fill_between(doubles, res_dict['LR_res_upper_bound'], res_dict['LR_res_lower_bound'], color= '#FF3F3F')
    ax.plot(doubles, res_dict['LR_res_mean'],  'r-', label='LR average residual') 

    ax.fill_between(doubles, res_dict['SDP_high_acc_res_upper_bound'], res_dict['SDP_high_acc_res_lower_bound'], color= '#3399FF')
    ax.plot(doubles, res_dict['SDP_high_acc_res_mean'], 'b-', label='SDP average residual with high accuracy') 

    ax.fill_between(doubles, res_dict['SDP_low_acc_res_upper_bound'], res_dict['SDP_low_acc_res_lower_bound'], color= '#6FE73B')
    ax.plot(doubles, res_dict['SDP_low_acc_res_mean'],  'g-', label='SDP average residual with low accuracy') 
    
    my_string ='sample size=%s'%str(SAMPLE_SIZE)
    # ax.text(8, 0.9*np.max(res_dict['LR_res_mean']), my_string, fontdict=font)
    ax.set_xticks(doubles)

    plt.xscale("log",basex=2) 
    plt.title("Accuracy as a function of the timestep (x,y in log scale)")
    legend = ax.legend(loc='upper right', shadow=False)
    plt.show() 

def log_plot_residual_experiments(PROBLEM_SIZE, SAMPLE_SIZE, res_dict):

    ax = plt.subplot()  
  
    print("Plotting data...")

    # ax.fill_between(doubles, res_dict['LR_res_upper_bound'], res_dict['LR_res_lower_bound'], color= '#FF3F3F')
    ax.plot(doubles, res_dict['LR_res_mean'],  'r-', label='LR average residual') 

    # ax.fill_between(doubles, res_dict['SDP_high_acc_res_upper_bound'], res_dict['SDP_high_acc_res_lower_bound'], color= '#3399FF')
    ax.plot(doubles, res_dict['SDP_high_acc_res_mean'], 'b-', label='SDP average residual with high accuracy') 

    # ax.fill_between(doubles, res_dict['SDP_low_acc_res_upper_bound'], res_dict['SDP_low_acc_res_lower_bound'], color= '#6FE73B')
    ax.plot(doubles, res_dict['SDP_low_acc_res_mean'],  'g-', label='SDP average residual with low accuracy') 
    
    my_string ='sample size=%s'%str(SAMPLE_SIZE)
    # ax.text(8, 0.9*np.max(res_dict['LR_res_mean']), my_string, fontdict=font)
    ax.set_xticks(doubles)

    plt.xscale("log",basex=2)
    plt.yscale("log")
    plt.title("Accuracy as a function of the timestep (x,y in log scale)")
    legend = ax.legend(loc='upper right', shadow=False)
    plt.show()

def plot_runtime_experiments(PROBLEM_SIZE, SAMPLE_SIZE, run_dict):

    ax = plt.subplot()  
  
    print("Plotting data...")

    # ax.fill_between(doubles, run_dict['LR_run_upper_bound'], run_dict['LR_run_lower_bound'], color= '#FF3F3F')
    ax.plot(doubles, run_dict['LR_run_mean'],  'r-', label='LR average runtime') 

    # ax.fill_between(doubles, run_dict['SDP_high_acc_run_upper_bound'], run_dict['SDP_high_acc_run_lower_bound'], color= '#3399FF')
    ax.plot(doubles, run_dict['SDP_high_acc_run_mean'], 'b-', label='SDP average runtime with high accuracy') 

    # ax.fill_between(doubles, run_dict['SDP_low_acc_run_upper_bound'], run_dict['SDP_low_acc_run_lower_bound'], color= '#6FE73B')
    ax.plot(doubles, run_dict['SDP_low_acc_run_mean'],  'g-', label='SDP average runtime with low accuracy') 
    
    my_string ='sample size=%s'%str(SAMPLE_SIZE)
    # ax.text(8, 0.9*np.max(run_dict['LR_run_mean']), my_string, fontdict=font)
    ax.set_xticks(doubles)

    plt.xscale("log",basex=2) 
    plt.title("Accuracy as a function of the timestep (x,y in log scale)")
    legend = ax.legend(loc='upper right', shadow=False)
    plt.show()

PROBLEM_SIZE, SAMPLE_SIZE, res_dict, run_dict, condition_nrs = start_experiment()   

with open('subdivision_experiments.csv', 'a', encoding='UTF8', newline='') as f:
    
    writer = csv.writer(f) 
    # write the data
    writer.writerow(['PROBLEM_SIZE=', PROBLEM_SIZE, 'SAMPLE_SIZE=', SAMPLE_SIZE, 'SUBDIVISIONS_NR=', MAX_POWER])

    writer.writerow(['LR residual list']) 
    for subdivision in doubles:
        writer.writerow([subdivision,':',res_dict['LR_res'][subdivision]])

    writer.writerow(['LR residual mean']) 
    writer.writerow(res_dict['LR_res_mean'])

    writer.writerow(['LR residual std']) 
    writer.writerow(res_dict['LR_res_std'])

    writer.writerow(['SDP high accuracy residual list']) 
    for subdivision in doubles:
        writer.writerow([subdivision,':',res_dict['SDP_high_acc_res'][subdivision]]) 

    writer.writerow(['SDP high accuracy residual mean']) 
    writer.writerow(res_dict['SDP_high_acc_res_mean'])

    writer.writerow(['SDP high accuracy residual std']) 
    writer.writerow(res_dict['SDP_high_acc_res_std'])

    writer.writerow(['SDP low accuracy residual list']) 
    for subdivision in doubles:
        writer.writerow([subdivision,':',res_dict['SDP_low_acc_res'][subdivision]]) 

    writer.writerow(['SDP low accuracy residual mean']) 
    writer.writerow(res_dict['SDP_low_acc_res_mean'])

    writer.writerow(['SDP low accuracy residual std']) 
    writer.writerow(res_dict['SDP_low_acc_res_std'])

    writer.writerow(['CONDITION NUMBERS']) 
    for subdivision in doubles:
        writer.writerow([subdivision,':',condition_nrs[subdivision]])
   
    writer.writerow([])
    writer.writerow(['*************************************************' ])
    writer.writerow([])
    f.close() 

log_plot_residual_experiments(PROBLEM_SIZE, SAMPLE_SIZE, res_dict)
plot_residual_experiments(PROBLEM_SIZE, SAMPLE_SIZE, res_dict) 
plot_runtime_experiments(PROBLEM_SIZE, SAMPLE_SIZE, run_dict)

 
 

 
