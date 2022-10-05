from dataclasses import asdict
from traceback import print_tb
from operator import add
import matplotlib.pyplot as plt 
import predictor_corrector as pc 
import sdp_tracking as sdp
import create_examples_data  as ex
import sdp_solver as ip 
import parameters as par
import old.visualize as vis
import numpy as np
import os
import csv
 
params = par.getParameters(print_par=True) 

def divide(k):
    def divide_by_k(x):
        return x/k
    return divide_by_k

def start_experiment(PROBLEM_SIZE=int(input("Enter PROBLEM_SIZE:")),SAMPLE_SIZE=int(input("Enter SAMPLE_SIZE:"))):
    PROBLEM_SIZE, SAMPLE_SIZE, subdivision_runtime_dict, subdivision_accuracy_dict = run_experiment(PROBLEM_SIZE, SAMPLE_SIZE)   
    return PROBLEM_SIZE, SAMPLE_SIZE, subdivision_runtime_dict, subdivision_accuracy_dict

def run_experiment(PROBLEM_SIZE, SAMPLE_SIZE): 
    
    # dictionary containing as a function of the subdivision cardinality:
    #   1. the subdivision
    #   2. the accuracy list for LR
    #   3. the accuracy list for SDP

    subdivision_accuracy_dict = {2:[np.linspace(0, 1, 3),[],[]], 
                                4:[np.linspace(0, 1, 5),[],[]], 
                                8:[np.linspace(0, 1, 9),[],[]], 
                                16:[np.linspace(0, 1, 17),[],[]], 
                                32:[np.linspace(0, 1, 33),[],[]], 
                                64:[np.linspace(0, 1, 65),[],[]]}

    # dictionary containing as a function of the subdivision cardinality:
    #   1. the subdivision
    #   2. the runtime list for LR
    #   3. the runtime list for SDP

    subdivision_runtime_dict = {2:[np.linspace(0, 1, 3),[],[]], 
                                4:[np.linspace(0, 1, 5),[],[]], 
                                8:[np.linspace(0, 1, 9),[],[]], 
                                16:[np.linspace(0, 1, 17),[],[]], 
                                32:[np.linspace(0, 1, 33),[],[]], 
                                64:[np.linspace(0, 1, 65),[],[]]}

    for k in range(SAMPLE_SIZE):
        print("    k = ", k)

        n, m, A, b, C = ex._create_MaxCut(PROBLEM_SIZE) 

        initial_time = float(params["problem"]["initial_time"])
        
        Y_0, rank, lam_0  = ip._get_initial_point(n=n, m=m, A = A(initial_time), b=b(initial_time), C=C(initial_time))

        for subdivision in [2**i for i in range(1,7)]:

            predcorr = pc._PredictorCorrector(n=n, m=m, rank=rank, params=params, ini_stepsize = 1/subdivision) 
            predcorr.run(A, b, C, Y_0, lam_0, check_residual=False, use_SDP_solver = True, print_data=False, silent=False) 
            if k==0:
                subdivision_accuracy_dict[subdivision][1] = predcorr._LR_accuracy_list
                subdivision_accuracy_dict[subdivision][2] = predcorr._SDP_accuracy_list
                subdivision_runtime_dict[subdivision][1] = predcorr._LR_runtime_list
                subdivision_runtime_dict[subdivision][2] = predcorr._SDP_runtime_list  
            else:
               subdivision_accuracy_dict[subdivision][1] = [sum(x) for x in zip(subdivision_accuracy_dict[subdivision][1], predcorr._LR_accuracy_list)]
               subdivision_accuracy_dict[subdivision][2] = [sum(x) for x in zip(subdivision_accuracy_dict[subdivision][2], predcorr._SDP_accuracy_list)] 
               subdivision_runtime_dict[subdivision][1] = [sum(x) for x in zip(subdivision_runtime_dict[subdivision][1], predcorr._LR_runtime_list)]  
               subdivision_runtime_dict[subdivision][2] = [sum(x) for x in zip(subdivision_runtime_dict[subdivision][2], predcorr._SDP_runtime_list)]  

    for subdivision in [2**i for i in range(1,7)]:
        subdivision_accuracy_dict[subdivision][1] = [x/SAMPLE_SIZE for x in subdivision_accuracy_dict[subdivision][1]]
        subdivision_accuracy_dict[subdivision][2] = [x/SAMPLE_SIZE for x in subdivision_accuracy_dict[subdivision][2]]
        subdivision_runtime_dict[subdivision][1] = [x/SAMPLE_SIZE for x in subdivision_runtime_dict[subdivision][1]]
        subdivision_runtime_dict[subdivision][2] = [x/SAMPLE_SIZE for x in subdivision_runtime_dict[subdivision][2]]
     
    return PROBLEM_SIZE, SAMPLE_SIZE, subdivision_runtime_dict, subdivision_accuracy_dict

def plot_accuracy_experiments(PROBLEM_SIZE, SAMPLE_SIZE, subdivision_runtime_dict, subdivision_accuracy_dict):

    plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.sans-serif": ["Helvetica"]}) 
 
    font = {
            'family': 'serif',
            'color' : 'black',
            'weight': 'normal',
            'size'  :  9
    } 

    fig, ax = plt.subplots() 
    ax.plot(subdivision_accuracy_dict[2][0], subdivision_accuracy_dict[2][1], 'r-',label='2')
    ax.plot(subdivision_accuracy_dict[2][0], subdivision_accuracy_dict[2][2], 'r--')

    ax.plot(subdivision_accuracy_dict[4][0], subdivision_accuracy_dict[4][1], 'b-', label='4')
    ax.plot(subdivision_accuracy_dict[4][0], subdivision_accuracy_dict[4][2], 'b--')

    ax.plot(subdivision_accuracy_dict[8][0], subdivision_accuracy_dict[8][1], 'g-', label='8')
    ax.plot(subdivision_accuracy_dict[8][0], subdivision_accuracy_dict[8][2], 'g--')

    ax.plot(subdivision_accuracy_dict[16][0], subdivision_accuracy_dict[16][1], 'c-', label='16')
    ax.plot(subdivision_accuracy_dict[16][0], subdivision_accuracy_dict[16][2], 'c--')

    ax.plot(subdivision_accuracy_dict[32][0], subdivision_accuracy_dict[32][1], 'm-', label='32')
    ax.plot(subdivision_accuracy_dict[32][0], subdivision_accuracy_dict[32][2], 'm--') 
   
    ax.plot(subdivision_accuracy_dict[64][0], subdivision_accuracy_dict[64][1], 'k-', label='64')
    ax.plot(subdivision_accuracy_dict[64][0], subdivision_accuracy_dict[64][2], 'k--')
    
    plt.axis([0,1,0,1.1*np.max(subdivision_accuracy_dict[2][1])])

    legend = ax.legend(loc='upper right', shadow=False)

    plt.show()

PROBLEM_SIZE, SAMPLE_SIZE, subdivision_runtime_dict, subdivision_accuracy_dict = start_experiment()
 
plot_accuracy_experiments(PROBLEM_SIZE, SAMPLE_SIZE, subdivision_runtime_dict, subdivision_accuracy_dict)

 