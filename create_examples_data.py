import scipy as sc
import numpy as np
from scipy import stats
 
normal_seed = stats.norm(10,1) .rvs  
poisson_seed = stats.poisson(100,loc=10).rvs  

class _ProblemCreator:

    def __init__(self,n: int, m:int):
        
        self._A_init = np.ndarray((m, n, n)) 
        self._A_pert = np.ndarray((m, n, n)) 
        self._b_init = np.ndarray((m, )) 
        self._b_pert = np.ndarray((m, )) 
        self._C_init = np.ndarray((n, n)) 
        self._C_pert = np.ndarray((n, n)) 

    def _create_random_problem(self, n: int, m:int):

        A_init = []
        A_pert = []
        
        for i in range(m):
            rand_A_init = sc.sparse.random(n, n, density=0.7, data_rvs=normal_seed).toarray()
            rand_A_pert = sc.sparse.random(n, n, density=0.7, data_rvs=normal_seed).toarray()
            rand_A_pert = np.identity(n)
            sym_rand_A_init = (rand_A_init + rand_A_init.T)
            sym_rand_A_pert = (rand_A_pert + rand_A_pert.T)
            A_init.append(sym_rand_A_init) 
            A_pert.append(sym_rand_A_pert)  
        
        A_init = np.array(A_init)
        A_pert = np.array(A_pert)

        np.copyto(self._A_init, A_init) 
        np.copyto(self._A_pert, A_pert) 

        b_init = np.ones(m)  
        b_pert = np.ones(m)  

        np.copyto(self._b_init, b_init) 
        np.copyto(self._b_pert, b_pert) 

        C_init = np.identity(n)
        C_pert = np.identity(n)

        np.copyto(self._C_init, C_init) 
        np.copyto(self._C_pert, C_pert) 

        def A(time: np.float): 
            return A_init+time*A_pert 
        
        def b(time: np.float):
            return b_init+time*b_pert 
        
        def C(time: np.float):
            return C_init+time*C_pert 

        return n, m, A, b, C

    def _create_bigeasy(self, n: int):
        
        A_init = [] 
         
        constraint_i = np.zeros((n,n))
        constraint_i[0,0] = 1  
        A_init.append(constraint_i)  

        A_init = np.array(A_init) 
        np.copyto(self._A_init, A_init) 
        
        b_pert = np.ones(1)
        np.copyto(self._b_pert, b_pert) 

        C_init = np.eye(n)
        
        np.copyto(self._C_init, C_init)  

        def A(time: np.float): 
            return A_init
        
        def b(time: np.float):
            return time*b_pert+1

        def C(time: np.float): 
            return C_init 

        return n, 1, A, b, C

    def _create_MaxCut(self, n: int):
        
        A_init = [] 
        
        for i in range(n): 
            constraint_i = np.zeros((n,n))
            constraint_i[i,i] = 1  
            A_init.append(constraint_i)  

        A_init = np.array(A_init) 
        np.copyto(self._A_init, A_init) 
        
        b_init = np.ones(n)
        np.copyto(self._b_init, b_init) 

        rand_C_init = sc.sparse.random(n, n, density=0.5, data_rvs=normal_seed).toarray()
        NZ = rand_C_init.nonzero() 
        I = NZ[0]
        J = NZ[1]
        nr_NZ = len(I)
        V = np.random.rand(nr_NZ,) 
        rand_C_pert = sc.sparse.coo_matrix((V,(I,J)),shape=(n,n)).toarray()
        
        C_init = np.abs(rand_C_init - rand_C_init.T) 
        C_pert = np.abs(rand_C_pert - rand_C_pert.T)  
        # print(C_init)
        # print(C_pert)

        np.copyto(self._C_init, C_init) 
        np.copyto(self._C_pert, C_pert) 

        def A(time: np.float): 
            return A_init
        
        def b(time: np.float):
            return b_init 

        def C(time: np.float): 
            return C_init+time*C_pert 

        return n, n, A, b, C

    def _choose_example():

        example_ID = input("Enter example ID:")
        example_name = "example_%s" %example_ID 

        if example_name == "example_1a":
                
            n = 3
            m = 3

            A0 = [[0,.5,0],
                [.5,0,0],
                [0,0,0]]
            A1 = [[0,0,.5],
                [0,0,0],
                [.5,0,0]]
            A2 = [[0,0,0],
                [0,0,.5],
                [0,.5,0]]
            Z = np.zeros((3,3))

            A_init = np.array([A0,A1,A2], dtype=np.float64)
            A_pert = np.array([Z,Z,Z], dtype=np.float64)

            b_init = np.array([1., 1., 1.])  
            b_pert = np.array([1., 1., 1.])  

            C_init = np.identity(3)
            C_pert = np.zeros((3,3))

            def A(time: np.float):
                return A_init+time*A_pert 
            
            def b(time: np.float):
                return b_init+time*b_pert 
            
            def C(time: np.float):
                return C_init+time*C_pert 
    
        
        if example_name == "example_1b":

            n = 3
            m = 3 

            A0 = [[1,1,0],
                [1,0,0],
                [0,0,0]]
            A1 = [[0,0,1],
                [0,0,0],
                [1,0,0]]
            A2 = [[0,0,0],
                [0,0,1],
                [0,1,1]]
            Z = np.zeros((3,3))

            A_init = np.array([A0,A1,A2], dtype=np.float64)
            A_pert = np.array([Z,Z,Z], dtype=np.float64)

            b_init = np.array([1., 1., 1.])  
            b_pert = np.array([1., 1., 1.])  

            C_init = np.identity(3)
            C_pert = np.zeros((3,3))

            def A(time: np.float):
                return A_init+time*A_pert 
            
            def b(time: np.float):
                return b_init+time*b_pert 
            
            def C(time: np.float):
                return C_init+time*C_pert 
    
        
        if example_name == "example_1c":

            n = 3
            m = 4 

            A0 = [[1,0,0],
                [0,0,0],
                [0,0,0]]
            A1 = [[0,0.5,0],
                [0.5,0,0],
                [0,0,0]]
            A2 = [[0,0,0],
                [0,0,0.5],
                [0,0.5,0]]
            A3 = [[0,0,0.5],
                [0,1,0],
                [0.5,0,0]]
            Z = np.zeros((3,3))

            A_init = np.array([A0,A1,A2,A3], dtype=np.float64)
            A_pert = np.array([Z,Z,Z,Z], dtype=np.float64)

            b_init = np.array([1., 0., 0., 0.])  
            b_pert = np.array([ 0., 0., 0., 1.])  

            C_init = np.identity(3)
            C_pert = np.zeros((3,3))

            def A(time: np.float):
                return A_init+time*A_pert 
            
            def b(time: np.float):
                return b_init+time*b_pert 
            
            def C(time: np.float):
                return C_init+time*C_pert 
    
        if example_name == "example_rand1":

            n = 3
            m = 1 

            rand_A0 = sc.sparse.random(3, 3, density=0.5).toarray()
            rand_Z = sc.sparse.random(3, 3, density=0.5).toarray()

            A0 = (rand_A0 + rand_A0.T)
            Z = (rand_Z + rand_Z.T) * 1e-4

            A_init = np.array([A0], dtype=np.float64)
            A_pert = np.array([Z], dtype=np.float64)

            b_init = np.array([1.])  
            b_pert = np.random.randint(100, size=(1,))/100 

            C_init = np.identity(3)
            C_pert = np.zeros((3,3)) 

            def A(time: np.float):
                return A_init+time*A_pert 
            
            def b(time: np.float):
                return b_init+time*b_pert 
            
            def C(time: np.float):
                return C_init+time*C_pert 
        
        if example_name == "example_rand2":

            n = 3
            m = 2 

            rand_A0 = sc.sparse.random(3, 3, density=0.5).toarray()
            rand_A1 = sc.sparse.random(3, 3, density=0.5).toarray()
            rand_Z = sc.sparse.random(3, 3, density=0.5).toarray()

            A0 = (rand_A0 + rand_A0.T)
            A1 = (rand_A1 + rand_A1.T)
            Z = (rand_Z + rand_Z.T) * 1e-5

            A_init = np.array([A0,A1], dtype=np.float64)
            A_pert = np.array([Z,Z], dtype=np.float64)

            b_init = np.array([1., 1.])  
            b_pert = np.random.randint(100, size=(2,))/100 

            C_init = np.identity(3)
            C_pert = np.zeros((3,3))

            def A(time: np.float):
                return A_init+time*A_pert 
            
            def b(time: np.float):
                return b_init+time*b_pert 
            
            def C(time: np.float):
                return C_init+time*C_pert 

        if example_name == "example_rand3":

            n = 3
            m = 3 

            rand_A0 = sc.sparse.random(3, 3, density=0.5).toarray()
            rand_A1 = sc.sparse.random(3, 3, density=0.5).toarray()
            rand_A2 = sc.sparse.random(3, 3, density=0.5).toarray()
            rand_Z = sc.sparse.random(3, 3, density=0.5).toarray()
            A0 = (rand_A0 + rand_A0.T)
            A1 = (rand_A1 + rand_A1.T)
            A2 = (rand_A2 + rand_A2.T)
            Z = (rand_Z + rand_Z.T) * 1e-4

            A_init = np.array([A0,A1,A2], dtype=np.float64)
            A_pert = np.array([Z,Z,Z], dtype=np.float64)

            b_init = np.array([1., 1., 1.])  
            b_pert = np.random.randint(100, size=(3,))/100 

            C_init = np.identity(3)
            C_pert = np.zeros((3,3))

            def A(time: np.float):
                return A_init+time*A_pert 
            
            def b(time: np.float):
                return b_init+time*b_pert 
            
            def C(time: np.float):
                return C_init+time*C_pert 

        if example_name == "example_rand4":

            n = 3
            m = 4 

            rand_A0 = sc.sparse.random(3, 3, density=0.5).toarray()
            rand_A1 = sc.sparse.random(3, 3, density=0.5).toarray()
            rand_A2 = sc.sparse.random(3, 3, density=0.5).toarray()
            rand_A3 = sc.sparse.random(3, 3, density=0.5).toarray()
            rand_Z = sc.sparse.random(3, 3, density=0.5).toarray()
            A0 = (rand_A0 + rand_A0.T)
            A1 = (rand_A1 + rand_A1.T)
            A2 = (rand_A2 + rand_A2.T)
            A3 = (rand_A3 + rand_A3.T)
            Z = (rand_Z + rand_Z.T) * 1e-4

            A_init = np.array([A0,A1,A2,A3], dtype=np.float64)
            A_pert= np.array([Z,Z,Z,Z], dtype=np.float64)

            b_init = np.array([1., 1., 1., 1.])  
            b_pert = np.array([1., 1., 1., 1.])  

            C_init = np.identity(3)
            C_pert = np.zeros((3,3)) 

            def A(time: np.float):
                return A_init+time*A_pert 
            
            def b(time: np.float):
                return b_init+time*b_pert 
            
            def C(time: np.float):
                return C_init+time*C_pert 
    
        return n, m, A, b, C
