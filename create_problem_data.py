import scipy as sc
import numpy as np 
from scipy import stats
 
normal_seed = stats.norm(loc=0, scale=1).rvs 

class _ProblemCreator:

    def __init__(self):
        pass

    def _create_random_problem(self, n: int, m: int):

        A_init = np.ndarray((m, n, n)) 
        A_pert = np.ndarray((m, n, n)) 
 
        for i in range(m):
            rand_A_init = sc.sparse.random(n, n, density=0.2, data_rvs=normal_seed).toarray()
            rand_A_pert = sc.sparse.random(n, n, density=0.2, data_rvs=normal_seed).toarray() 
            
            sym_rand_A_init = (rand_A_init + rand_A_init.T)
            sym_rand_A_pert = (rand_A_pert + rand_A_pert.T)

            A_init[i] = sym_rand_A_init
            A_pert[i] = sym_rand_A_pert
        
        A_init = np.array(A_init)
        A_pert = np.array(A_pert)  

        b_init = np.ones(m)  
        b_pert = np.ones(m)  

        C_init = np.identity(n)
        C_pert = np.identity(n)

        def A(time: np.float): 
            return A_init + time*A_pert 
        
        def b(time: np.float):
            return b_init + time*b_pert 
        
        def C(time: np.float):
            return C_init + time*C_pert 

        return A, b, C

    def _create_MaxCut(self, n: int):
        
        A_init = np.ndarray((n, n, n)) 
        
        for i in range(n): 
            constraint_i = np.zeros((n,n))
            constraint_i[i,i] = 1  
            A_init[i]=constraint_i
    
        b_init = np.ones(n) 

        rand_C_init = sc.sparse.random(n, n, density=0.5, data_rvs=normal_seed).toarray()
        NZ = rand_C_init.nonzero() 
        I = NZ[0]
        J = NZ[1]
        nr_NZ = len(I)
        V = np.random.rand(nr_NZ,) 
        rand_C_pert = sc.sparse.coo_matrix((V,(I,J)),shape=(n,n)).toarray()*10
        
        C_init = np.abs(rand_C_init - rand_C_init.T) 
        C_pert = np.abs(rand_C_pert - rand_C_pert.T)    

        def A(time: np.float): 
            return A_init
        
        def b(time: np.float):
            return b_init 

        def C(time: np.float): 
            return C_init + time*C_pert 
       
        return A, b, C 