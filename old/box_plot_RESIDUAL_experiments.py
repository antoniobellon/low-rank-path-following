from dataclasses import asdict
from traceback import print_tb
from operator import add

from sympy import false
 
import predictor_corrector as pc  
import sdp_tracking as st
import create_examples_data  as ex
import sdp_solver as ip 
import parameters as par 

import matplotlib.pyplot as plt  
import plotly.graph_objects as go 
import numpy as np
import time 
 
params = par.getParameters(print_par=True) 
 
INIT_STEP = 0.1
NR_TOLS = 4
res_tolerances = [10**-k for k in np.linspace(2,2*NR_TOLS, NR_TOLS)]
 
def start_experiment(PROBLEM_SIZE=int(input("Enter PROBLEM_SIZE:")),SAMPLE_SIZE=int(input("Enter SAMPLE_SIZE:"))):
    PROBLEM_SIZE, SAMPLE_SIZE, run_dict, condition_nrs = run_experiment(PROBLEM_SIZE, SAMPLE_SIZE)   
    return PROBLEM_SIZE, SAMPLE_SIZE, run_dict 

def run_experiment(PROBLEM_SIZE, SAMPLE_SIZE): 
 
    run_dict = {'LR_run'                : {str(d) : [] for d in res_tolerances},
                'SDP_high_acc_run'      : {str(d) : [] for d in res_tolerances},
                'SDP_low_acc_run'       : {str(d) : [] for d in res_tolerances}
                } 

    condition_nrs = {str(d) : [] for d in res_tolerances}
    times = []

    for k in range(SAMPLE_SIZE):

        if k > 0:
            mean_time = np.mean(times)
            time_string = np.str(np.int(np.divmod((SAMPLE_SIZE-k)*mean_time,60)[0]))+"."+ np.str(np.int(np.divmod((SAMPLE_SIZE-k)*mean_time,60)[1]))
            print("expected time to termination:", time_string) 
        before = time.time()

        print("    K =", k)

        n, m, A, b, C = ex._create_MaxCut(PROBLEM_SIZE) 
        initial_time = float(params["problem"]["initial_time"])
        Y_0, rank, lam_0  = ip._get_initial_point(n=n, m=m, A = A(initial_time), b=b(initial_time), C=C(initial_time), TOLERANCE=1.0e-10)

        sdptrack = st._SDPtracking(n=n, m=m, rank=rank, params=params, ini_stepsize = INIT_STEP)
        sdptrack.run(A, b, C)

        for tol in res_tolerances:
            
            print("residual tolerance =", tol)

            predcorr = pc._PredictorCorrector(n=n, m=m, rank=rank, params=params, ini_stepsize = INIT_STEP, res_tol=tol) 
            predcorr.run(A, b, C, Y_0, lam_0, check_residual=True, use_SDP_solver = False, print_data=False, silent=True) 
           
            run_dict['LR_run'          ][str(tol)].append(predcorr._LR_runtime) 
            run_dict['SDP_high_acc_run'][str(tol)].append(sdptrack._SDP_high_acc_runtime) 
            run_dict['SDP_low_acc_run' ][str(tol)].append(sdptrack._SDP_low_acc_runtime) 

            condition_nrs[str(tol)].append(1/np.mean(predcorr._condition_numbers))

        times.append(time.time() - before)
        
    return PROBLEM_SIZE, SAMPLE_SIZE, run_dict, condition_nrs

def plot_runtime_experiments(PROBLEM_SIZE, SAMPLE_SIZE, run_dict):

    print("Plotting data...")
    
    fig = go.Figure()
     
    LR_runtime = [np.mean(run_dict['LR_run'][tol]) for tol in res_tolerances]
    SDP_high_runtime = [np.mean(run_dict['SDP_high_acc_run'][tol]) for tol in res_tolerances]
    SDP_low_runtime = [np.mean(run_dict['SDP_low_acc_run'][tol]) for tol in res_tolerances]

    fig.add_trace(go.Box(
        y=LR_runtime,
        x=['1e-2','1e-4','1e-6','1e-8'],
        name='LR runtime', 
        marker_color='#3D9970' 
        ))

    fig.add_trace(go.Box(
        y=SDP_high_runtime,
        x=['1e-2','1e-4','1e-6','1e-8'],
        name='SDP high accuracy runtime', 
        marker_color='#FF851B' 
        ))

    fig.add_trace(go.Box(
        y=SDP_low_runtime,
        x=['1e-2','1e-4','1e-6','1e-8'],
        name='SDP low accuracy runtime', 
        marker_color='#FF4136' 
        ))

    title_string="Runtime for n="+"%s"%np.str(PROBLEM_SIZE)+" with "+"%s"%np.str(SAMPLE_SIZE)+" samples"
    fig.update_layout(
        # title=title_string,
        yaxis_title='runtime',
        xaxis_title='residual tolerance', 
        autosize=True,
        font_family="Helvetica", 
        title_font_family="Helvetica",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)')

    fig.update_xaxes(type="category", nticks=NR_TOLS, ticks="outside", mirror=True, showline=True)
    fig.update_yaxes(type="log", ticks="outside", mirror=True, showline=True)
 
    fig.show()

def no_log_plot_runtime_experiments(PROBLEM_SIZE, SAMPLE_SIZE, run_dict):

    print("Plotting data...")
    
    fig = go.Figure()
     
    LR_runtime = [np.mean(run_dict['LR_run'][str(tol)]) for tol in res_tolerances]
    SDP_high_runtime = [np.mean(run_dict['SDP_high_acc_run'][str(tol)]) for tol in res_tolerances]
    SDP_low_runtime = [np.mean(run_dict['SDP_low_acc_run'][str(tol)]) for tol in res_tolerances]

    fig.add_trace(go.Box(
        y=LR_runtime,
        x=['1e-2','1e-4','1e-6','1e-8'],
        name=r'$\text{LR runtime}$', 
        marker_color='#3D9970' 
        ))

    fig.add_trace(go.Box(
        y=SDP_high_runtime,
        x=['1e-2','1e-4','1e-6','1e-8'],
        name=r'$\text{SDP high accuracy runtime}$', 
        marker_color='#FF851B' 
        ))

    fig.add_trace(go.Box(
        y=SDP_low_runtime,
        x=['1e-2','1e-4','1e-6','1e-8'],
        name=r'$\text{SDP low accuracy runtime}$', 
        marker_color='#FF4136'
        ))

    title_string="Runtime for n="+"%s"%np.str(PROBLEM_SIZE)+" with "+"%s"%np.str(SAMPLE_SIZE)+" samples"
    fig.update_layout(
        # title=title_string,,
        width=810,
        height=500,
        yaxis_title= r'$\text{runtime } [s]$',
        xaxis_title=r'$\text{residual tolerance}$',
        autosize=True,
        font_family="Helvetica", 
        title_font_family="Helvetica", 
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)', 
        margin=dict(autoexpand=True)
              
    )

    fig.update_xaxes(
        type="category", 
        nticks=NR_TOLS,
        ticks="outside",
        rangeselector_font=dict(size=18))
 
    fig.show()

PROBLEM_SIZE, SAMPLE_SIZE, run_dict, condition_nrs = start_experiment()   

# plot_runtime_experiments(PROBLEM_SIZE, SAMPLE_SIZE, run_dict)
no_log_plot_runtime_experiments(PROBLEM_SIZE, SAMPLE_SIZE, run_dict)