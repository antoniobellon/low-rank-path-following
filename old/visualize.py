import pickle
import numpy as np 
import os

np.set_printoptions(edgeitems=30, linewidth=100000
    # ,formatter=dict(float=lambda x: "%.5g" % x)
    )
    
def visualize_sol(params):

    t = float(params["problem"]["initial_time"])

    res_tol = float(params["problem"]["res_tol"])
    
    dirListing = os.listdir("results/")
    file_number = len(dirListing)

    # each file named res contains: 
    # res[0] = res
    # res[1] = dt 
    # res[2] = reduction steps 
    
    print("============================================")
    for i in range(file_number): 
        f = open("results/%d.pkl"%(i),"rb") 
        res = pickle.load(f)
        dt = res[1]
        t += dt
        print("ITERATION", i)
        print("reduction steps: ", res[2])
        print("t = %f\n"%t)  
        print("residual threshold: ", res_tol)
        print("residual: ", res[0])  
        print("============================================")
        f.close()