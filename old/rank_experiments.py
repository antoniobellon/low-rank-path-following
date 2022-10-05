from dataclasses import asdict
from traceback import print_tb
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

 
dict = {}
params = par.getParameters(print_par=True) 

def start_experiment(PROBLEM_SIZE=int(input("Enter PROBLEM_SIZE:")),SAMPLE_SIZE=int(input("Enter SAMPLE_SIZE:"))):
    return run_rank_experiment(PROBLEM_SIZE, SAMPLE_SIZE)   

def run_rank_experiment(PROBLEM_SIZE, SAMPLE_SIZE): 
    
    LR_runtime = 0.0
    SDP_runtime = 0.0

    exact_rank_lagr_tracking = []
    inexact_rank_lagr_tracking = []

    for k in range(SAMPLE_SIZE):
        print("    k = ", k)

        n, m, A, b, C = ex._create_MaxCut(PROBLEM_SIZE) 

        initial_time = float(params["problem"]["initial_time"])
        
        Y_0, rank, lam_0  = ip._get_initial_point(n=n, m=m, A = A(initial_time), b=b(initial_time), C=C(initial_time))
        zero_col = np.array([np.zeros(n)])
        
        for i in [0,1]: 
            print("Rank vs r factor: ",rank, rank+i)

            if i>0: 
                Y_0 = np.concatenate((Y_0,zero_col.T), axis=1) 

            predcorr = pc._PredictorCorrector(n=n, m=m, rank=np.shape(Y_0)[1], params=params)    
            predcorr.run(A, b, C, Y_0, lam_0, use_SDP_solver = True, print_data=False, silent=False) 

            if i==0:
                exact_rank_lagr_tracking.append(predcorr._lagrangian_dictionary)
                 
            else:
                inexact_rank_lagr_tracking.append(predcorr._lagrangian_dictionary)
                 
            # if (n,rank,np.shape(Y_0)[1]) in dict.keys():
            #     dict[(n,rank,np.shape(Y_0)[1])][0] += 1
            #     dict[(n,rank,np.shape(Y_0)[1])][1] += predcorr._total_runtime
            #     dict[(n,rank,np.shape(Y_0)[1])][2] += predcorr._total_SDP_runtime
            #     dict[(n,rank,np.shape(Y_0)[1])][3] += predcorr._iterations
            # else:
            #     dict[(n,rank,np.shape(Y_0)[1])]=[1,predcorr._total_runtime, predcorr._total_SDP_runtime, predcorr._iterations]
              
    # for k in dict.keys():
    #     iter =  dict[k][0]
    #     dict[k][1] /= iter
    #     dict[k][2] /= iter
    #     dict[k][3] /= iter
    #     print(k,dict[k])

    exact_total_dict = {}
    for dict in exact_rank_lagr_tracking:
        for time in dict.keys():
            if time not in exact_total_dict.keys():
                exact_total_dict[time] = dict[time]
            else:
                exact_total_dict[time] += dict[time]
                exact_total_dict[time] /=2

    inexact_total_dict = {}
    for dict in inexact_rank_lagr_tracking:
        for time in dict.keys():
            if time not in inexact_total_dict.keys():
                inexact_total_dict[time] = dict[time]
            else:
                inexact_total_dict[time] += dict[time]
                inexact_total_dict[time] /=2


    with open('rank_experiments.csv', 'a', encoding='UTF8', newline='') as f:
        
        writer = csv.writer(f) 
        # write the data
        writer.writerow(["SIZE=",PROBLEM_SIZE])
        writer.writerow(["exact rank"]) 
        writer.writerow(exact_total_dict.keys())
        writer.writerow(exact_total_dict.values())
        writer.writerow(["inexact rank"])
        writer.writerow(inexact_total_dict.keys())
        writer.writerow(inexact_total_dict.values())
        f.close()
    return exact_total_dict, inexact_total_dict

exact_total_dict, inexact_total_dict = start_experiment()



plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.sans-serif": ["Helvetica"]}) 

params = par.getParameters(print_par=True) 

SAMPLE_SIZE =   2

MIN_SIZE = 2
MAX_SIZE = 20
SAMPLE_STEP = 2

N = np.arange(MIN_SIZE, MAX_SIZE, SAMPLE_STEP)

font = {
        'family': 'serif',
        'color' : 'black',
        'weight': 'normal',
        'size'  :  9
} 

fig, ax = plt.subplots()

 
exact_total_dict1 = sorted(exact_total_dict)
sorted_exact_total_dict = {key:exact_total_dict[key] for key in exact_total_dict1}
 
# inexact_total_dict1 = sorted(exact_total_dict)
# sorted_inexact_total_dict = {key:inexact_total_dict[key] for key in exact_total_dict1}

# print(sorted_exact_total_dict.keys())
# print(sorted_exact_total_dict.values())
# print(sorted_inexact_total_dict.keys())
# print(sorted_inexact_total_dict.values())

ax.plot(sorted_exact_total_dict.keys(), sorted_exact_total_dict.values(), 'r-', label='LOW RANK lagrangian')
# ax.plot(sorted_inexact_total_dict.keys(), sorted_inexact_total_dict.values(), 'b-', label='r=$r^*+1$')
# ax.text(MIN_SIZE, MAX , 'sample size=%s'%str(SAMPLE_SIZE),fontdict=font)
# plt.axis([0,1,0,200])


# 'Sigmoid(%s)'%(a)
legend = ax.legend(loc='upper left', shadow=False)

# Put a  nicer background color on the legend.
# legend.get_frame().set_facecolor('C0')

plt.show()