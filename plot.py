from dataclasses import asdict
from traceback import print_tb
from operator import add

import plotly.graph_objects as go 
import numpy as np 
import pickle

MAX_POWER = 7#7
INIT_STEP = 0.01#0.01
NR_TOLS = 4#4

res_tolerances = [10**-k for k in np.linspace(2,2*NR_TOLS, NR_TOLS)]
doubles = [10**k for k in range(1, MAX_POWER)]

def plot_subdivision_residual(SAMPLE_SIZE, sub_dict):

    print("Plotting data...")
    
    fig = go.Figure()

    subdivisions = ["{:.0e}".format(1/sub) for sub in doubles for x in range(SAMPLE_SIZE)]
    LR_residuals = [x for sub in doubles for x in sub_dict['LR_res'][str(sub)]]
    SDP_highest_residuals = [x for sub in doubles for x in sub_dict['SDP_highest_acc_res'][str(sub)]]
    SDP_high_residuals = [x for sub in doubles for x in sub_dict['SDP_high_acc_res'][str(sub)]]
    SDP_low_residuals = [x for sub in doubles for x in sub_dict['SDP_low_acc_res'][str(sub)]]
     
    fig.add_trace(go.Box(
        y=SDP_low_residuals,
        x=subdivisions,
        name=r'$\text{IPM low accuracy}$', 
        marker_color='#FF851B',
        boxpoints = False
        ))

    fig.add_trace(go.Box(
        y=SDP_high_residuals,
        x=subdivisions,
        name=r'$\text{IPM high accuracy}$', 
        marker_color='#FF4136',
        boxpoints = False
        ))

    fig.add_trace(go.Box(
        y=SDP_highest_residuals,
        x=subdivisions,
        name=r'$\text{IPM very high accuracy}$', 
        marker_color='#800020',
        boxpoints = False
        ))

    fig.add_trace(go.Box(
        y=LR_residuals,
        x=subdivisions,
        name=r'$\text{LR}$', 
        marker_color='#3D9970',
        boxpoints = False
        ))
    fig.update_layout(
        width=810,
        height=500,
        yaxis_title=r'$\text{residuals}$', 
        xaxis_title=r'$\text{stepsize}$', 
        autosize=True,
        font_family="Helvetica", 
        title_font_family="Helvetica", 
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
        )

    fig.update_xaxes(type="category", nticks=MAX_POWER, ticks="outside")
    fig.update_yaxes(type="log", ticks="outside",exponentformat = 'e')
    fig.show()

def plot_subdivision_exact_difference(SAMPLE_SIZE, sub_dict):

    print("Plotting data...")
    
    fig = go.Figure()

    subdivisions = ["{:.0e}".format(1/sub) for sub in doubles for x in range(SAMPLE_SIZE)]
    LR_residuals = [x for sub in doubles for x in sub_dict['LR_exact_diff'][str(sub)]] 
    SDP_high_residuals = [x for sub in doubles for x in sub_dict['SDP_high_acc_exact_diff'][str(sub)]]
    SDP_low_residuals = [x for sub in doubles for x in sub_dict['SDP_low_acc_exact_diff'][str(sub)]]
     
    fig.add_trace(go.Box(
        y=LR_residuals,
        x=subdivisions,
        name=r'$\text{LR}$', 
        marker_color='#3D9970',
        boxpoints = False
        ))

    fig.add_trace(go.Box(
        y=SDP_low_residuals,
        x=subdivisions,
        name=r'$\text{IPM low accuracy}$', 
        marker_color='#FF851B',
        boxpoints = False
        ))

    fig.add_trace(go.Box(
        y=SDP_high_residuals,
        x=subdivisions,
        name=r'$\text{IPM high accuracy}$', 
        marker_color='#FF4136',
        boxpoints = False
        ))
   
    fig.update_layout(
        width=810,
        height=500,
        yaxis_title=r'$\text{relative distance from solution}$', 
        xaxis_title=r'$\text{stepsize}$', 
        autosize=True,
        font_family="Helvetica", 
        title_font_family="Helvetica", 
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)')

    fig.update_xaxes(type="category", nticks=MAX_POWER, ticks="outside")
    fig.update_yaxes(type="log", ticks="outside", exponentformat = 'e')
 
    fig.show()

def plot_subdivision_value_difference(SAMPLE_SIZE, exp_info_dict):

    print("Plotting data...")
    
    fig = go.Figure()

    subdivisions = ["{:.0e}".format(1/sub) for sub in doubles for x in range(SAMPLE_SIZE)]
    LR_val_diff = [x for sub in doubles for x in exp_info_dict['LR_opt_value'][str(sub)]] 
    SDP_high_val_diff = [x for sub in doubles for x in exp_info_dict['SDP_high_acc_opt_value'][str(sub)]]
    SDP_low_val_diff = [x for sub in doubles for x in exp_info_dict['SDP_low_acc_opt_value'][str(sub)]]

    print(len(SDP_low_val_diff))
    print(len(LR_val_diff))
     
    fig.add_trace(go.Box(
        y=LR_val_diff,
        x=subdivisions,
        name=r'$\text{LR}$', 
        marker_color='#3D9970',
        boxpoints = 'all'
        ))

    fig.add_trace(go.Box(
        y=SDP_low_val_diff,
        x=subdivisions,
        name=r'$\text{IPM low accuracy}$', 
        marker_color='#FF851B',
        boxpoints = False
        ))

    fig.add_trace(go.Box(
        y=SDP_high_val_diff,
        x=subdivisions,
        name=r'$\text{IPM high accuracy}$', 
        marker_color='#FF4136',
        boxpoints = False
        ))

    fig.update_layout(
        width=810,
        height=500,
        yaxis_title=r'$\text{optimal value difference}$', 
        xaxis_title=r'$\text{stepsize}$', 
        autosize=True,
        font_family="Helvetica", 
        title_font_family="Helvetica", 
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
        )

    # fig.update_xaxes(type="category", nticks=MAX_POWER, ticks="outside")
    # fig.update_yaxes(type="log", ticks="outside",exponentformat = 'e')
    fig.show()

def plot_subdivision_runtime(SAMPLE_SIZE, sub_dict):

    print("Plotting data...")
    
    fig = go.Figure()
    
    subdivisions = ["{:.0e}".format(1/sub) for sub in doubles for x in range(SAMPLE_SIZE)]
    LR_runtime = [x for sub in doubles for x in sub_dict['LR_run'][str(sub)]]
    SDP_highest_runtime = [x for sub in doubles for x in sub_dict['SDP_highest_acc_run'][str(sub)]]
    SDP_high_runtime = [x for sub in doubles for x in sub_dict['SDP_high_acc_run'][str(sub)]]
    SDP_low_runtime = [x for sub in doubles for x in sub_dict['SDP_low_acc_run'][str(sub)]]
     
    fig.add_trace(go.Box(
        y=LR_runtime,
        x=subdivisions,
        name=r'$\text{LR}$', 
        marker_color='#3D9970' ,
        boxpoints = False
        ))

    # fig.add_trace(go.Box(
    #     y=SDP_highest_runtime,
    #     x=subdivisions,
    #     name=r'$\text{SDP highest accuracy runtime}$', 
    #     marker_color='#800020' ,
    #     boxpoints = False
    #     ))

    fig.add_trace(go.Box(
        y=SDP_low_runtime,
        x=subdivisions,
        name= r'$\text{IPM low accuracy}$', 
        marker_color='#FF851B' ,
        boxpoints = False
        ))

    fig.add_trace(go.Box(
        y=SDP_high_runtime,
        x=subdivisions,
        name=r'$\text{IPM high accuracy}$', 
        marker_color='#FF4136' ,
        boxpoints = False
        ))

    fig.update_layout( 
        width=810,
        height=500,
        yaxis_title= r'$\text{runtime } [s]$',
        xaxis_title=r'$\text{stepsize}$',
        autosize=True,
        font_family="Helvetica", 
        title_font_family="Helvetica", 
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)', 
        margin=dict(autoexpand=True)             
    )

    fig.update_xaxes(type="category", nticks=MAX_POWER, ticks="outside")
    fig.update_yaxes(type="log", ticks="outside", exponentformat = 'e')
 
    fig.show()

def plot_rank(rank_dict):

    print("Plotting data...")  
    
    fig = go.Figure(data=[go.Histogram(x=np.array(np.sort(rank_dict)))]) 
     
    fig.update_layout( 
        width=810,
        height=500,
        yaxis_title= r'$\text{runtime } [s]$',
        xaxis_title=r'$\text{rank}$',
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
    fig.update_yaxes(ticks="outside", exponentformat = 'e')
 
    fig.show()

def plot_residual_runtime(SAMPLE_SIZE, res_dict):

    print("Plotting data...")
    
    fig = go.Figure() 

    
    LR_runtime = [x for tol in res_tolerances for x in res_dict['LR_run'][str(tol)]]
    residuals = ["{:.0e}".format(tol) for tol in res_tolerances for x in res_dict['LR_run'][str(tol)]] 
    # SDP_highest_runtime = [x for tol in res_tolerances for x in res_dict['SDP_highest_acc_run'][str(tol)]]
    SDP_high_runtime = [x for tol in res_tolerances for x in res_dict['SDP_high_acc_run'][str(tol)]]
    SDP_low_runtime = [x for tol in res_tolerances for x in res_dict['SDP_low_acc_run'][str(tol)]]

   
    fig.add_trace(go.Box(
        y=LR_runtime,
        x=residuals,
        name=r'$\text{LR}$', 
        marker_color='#3D9970' ,
        boxpoints = False
        ))

    # fig.add_trace(go.Box(
    #     y=SDP_highest_runtime,
    #     x=residuals,
    #     name=r'$\text{SDP highest accuracy runtime}$', 
    #     marker_color='#800020' ,
    #     boxpoints = False
    #     ))

    fig.add_trace(go.Box(
        y=SDP_low_runtime,
        x=residuals,
        name=r'$\text{IPM low accuracy}$', 
        marker_color='#FF851B',
        boxpoints = False
        )) 

    fig.add_trace(go.Box(
        y=SDP_high_runtime,
        x=residuals,
        name=r'$\text{IPM high accuracy}$', 
        marker_color='#FF4136' ,
        boxpoints = False
        ))
 
    fig.update_layout( 
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
    fig.update_yaxes(ticks="outside", exponentformat = 'e')
 
    fig.show()

 
PROBLEM_SIZE = int(input("Enter PROBLEM_SIZE:"))
f = open('MaxCut experiments old/exp_%d.pkl'%(PROBLEM_SIZE),'rb')
pickled_exp = pickle.load(f) 
f.close() 

sub_dict = pickled_exp['sub_dict']
res_dict = pickled_exp['res_dict']
exp_info_dict = pickled_exp['exp_info'] 
rank_dict = exp_info_dict['rank'] 
 
SAMPLE_SIZE = len(pickled_exp['sub_dict']['LR_res']['10'])
print("SAMPLE_SIZE: ",SAMPLE_SIZE)

plot_subdivision_residual(SAMPLE_SIZE, sub_dict)
plot_subdivision_exact_difference(SAMPLE_SIZE, sub_dict)
plot_subdivision_value_difference(SAMPLE_SIZE, exp_info_dict)
plot_subdivision_runtime(SAMPLE_SIZE, sub_dict) 
plot_residual_runtime(SAMPLE_SIZE, res_dict)
plot_rank(rank_dict) 