import numpy as np  
from qiskit.quantum_info import state_fidelity, random_density_matrix, DensityMatrix


class _QuantumUtils:

    def __init__(self, ro: np.ndarray, N: int, T: int):

        self.ro = ro 
        self.T = T  
        self.N = N
        self.M = 4**self.N
        N = self.N
        M = self.M
        triangular_M = M*(M+1)//2
        self.m = M + triangular_M + 2
        self.n = 2**(N+1) + M + 2
        n = self.n
        m = self.m

        self.measure_basis = _QuantumUtils.create_measure_basis(self)

        self.real_measure_basis = [_QuantumUtils.make_it_real(self,matrix) for matrix in self.measure_basis]
         
        self.probabilities = np.zeros((M), dtype=float)
        for i in range(M):
            self.probabilities[i] = abs(_QuantumUtils.frob_inner_prod(self,self.measure_basis[i], self.ro)) 

        self.observations = np.zeros((M,T+10), dtype=int)
        for i in range(M):
            self.observations[i] = np.random.binomial(1, self.probabilities[i], T+10)

        self.approx_probabilities = np.zeros((M,T+10), dtype=float)
        for i in range(M):
            for t in range(T+10):  
                self.approx_probabilities[i,t] = 1/(t+1)*np.sum(self.observations[i,:t+1])
        
        self.A_list = np.ndarray((m,n,n)) 
        self.b_list = np.zeros(m) 
        self.C_objective = np.zeros((n,n))
        self.C_objective[0,0] = 1  

        # Identity constraints
        constr_index = 0
        for i in range(M):
            for j in range(i,M):
                constraint = np.zeros((n,n),dtype=float)
                if i == j:
                    constraint[i+1,i+1] = 1
                    self.A_list[constr_index] = constraint 
                    self.b_list[constr_index] = 1    
                else:
                    constraint[i+1,j+1] = 0.5
                    constraint[j+1,i+1] = 0.5
                    self.A_list[constr_index] = constraint 
                    self.b_list[constr_index] = 0
                constr_index += 1 

        # Measurements constraints
        for i in range(M):
            constraint = np.zeros((n,n),dtype=float)
            constraint[i+1,M+1] = -1
            constraint[M+1,i+1] = -1 
            constraint[M+2:,M+2:] = self.real_measure_basis[i] 
            self.A_list[i+triangular_M] = constraint 
            self.b_list[i+triangular_M] = 0 

        # Trace constraint
        constraint = np.zeros((n,n),dtype=float)
        constraint[M+2:,M+2: ] = np.eye(2**(N+1),2**(N+1)) 
        self.A_list[M+triangular_M] = constraint 
        self.b_list[M+triangular_M] = 2 

    def frob_inner_prod(self,A,B):

        return(np.trace(np.matmul(A,B)))

    def create_measure_basis(self):

        N = self.N 

        E = np.ndarray((4,2,2), dtype=complex)
        E[0] = np.array([[0.5, 0],[0, 0]], dtype=complex)
        E[1] = np.array([[1/6, np.sqrt(2)/6], [np.sqrt(2)/6, 2/6]], dtype=complex)
        E[2] = np.array([[1/6, np.exp(1j*2*np.pi/3)*(np.sqrt(2)/6)], [np.exp(-1j*2*np.pi/3)*(np.sqrt(2)/6), 2/6]], dtype=complex)
        E[3] = np.array([[1/6,-1/6],[-1/6,1/6]], dtype=complex) 

        measure_basis = E
        size_measure_basis = 4

        if N==1: return measure_basis
        else:
            for i in range(N-1): 
                    kronecker = _QuantumUtils.kronecker(self,measure_basis,E)
                    measure_basis = np.ndarray((size_measure_basis*4,2,2), dtype=complex)
                    measure_basis = kronecker  
            return measure_basis
    
    def kronecker(self, matrix_list_A, matrix_list_B):
        
        kronecker_list= []
        for A in matrix_list_A:
            for B in matrix_list_B:
                kronecker_list.append(np.kron(A, B))
        return kronecker_list

    def make_it_real(self,matrix):

        Re = matrix.real
        Im = matrix.imag 
        return np.block([[Re, -Im], [Im, Re]])

    def extract_solution(self, big_matrix):

        N = self.N
        M = self.M
        X_real = big_matrix[M+2:2**N+M+2, M+2:2**N+M+2]
        X_imag = big_matrix[2**N+M+2:2**(N+1)+M+2, M+2:2**N+M+2]
        return X_real + 1j*X_imag
    
    def interpol_probabilities(self, time: float, interpol_factor=1):

        M = self.M
        interpol = np.zeros((M),dtype=float) 
        floor, remain = divmod(time,interpol_factor) 
        low_interpolator = int(floor*interpol_factor)
        high_interpolator = int((floor+1)*interpol_factor)
        for i in range(M):
            interpol[i] = self.approx_probabilities[i,low_interpolator] 
            +(remain/interpol_factor)*(self.approx_probabilities[i,high_interpolator]-self.approx_probabilities[i,low_interpolator])
        return interpol

    def from_density_to_sdp_solution(self, dens_matrix):

        N = self.N
        M = self.M
        sdp_sol = np.zeros((self.n,self.n), dtype=float)
        sdp_sol[0,0] = 0
        for i in range(M):
            b = _QuantumUtils.frob_inner_prod(self, self.measure_basis[i], dens_matrix) 
            sdp_sol[i+1,i+1] = 1
            sdp_sol[i+1,M+1] = b
            sdp_sol[M+1,i+1] = sdp_sol[i+1,M+1] 
            sdp_sol[M+1,M+1] = self.b_list[-1]
        sdp_sol[M+2: 4**N + M + 2,M+2: 4**N + M + 2] = _QuantumUtils.make_it_real(self, np.array(dens_matrix))
        return sdp_sol
 
    # CONSTRAINTS AND OBJECTIVE DEFINITIONS   
    
    def A(self, time: np.float, exact_data=False):  

        M = self.M
        n = self.n 
        constraint =  np.zeros((n,n),dtype=float)
        constraint[0,0] = 1 
        constraint[M+1,M+1] = -1
        for i in range(M):
            if exact_data:
                constraint[i+1,M+1] = self.probabilities[i]
                constraint[M+1,i+1] = self.probabilities[i]
            else:
                constraint[i+1,M+1] = self.interpol_probabilities(time)[i]
                constraint[M+1,i+1] = self.interpol_probabilities(time)[i]
        self.A_list[-1] = constraint   
        return self.A_list
    
    def b(self, time: np.float, exact_data=False):  

        M = self.M
        if exact_data:
            self.b_list[-1] = np.sum([x**2 for x in self.probabilities])
        else:
            self.b_list[-1] = np.sum([x**2 for x in self.interpol_probabilities(time)])
        return self.b_list  

    def C(self, time: float):
        return self.C_objective

    def print_state_comparison(self, sigma):
        # PRINT DATA 
        print('\nOriginal state: \n', np.array(self.ro))
        print('Estimated state: \n', sigma) 
        print("FIDELITY")
        print(state_fidelity(self.ro, DensityMatrix(sigma)) ) 
        print('\nDISTRIBUTIONS COMPARISION')
        print('Original state      Estimated state') 
        for i in range(self.M): 
            print(self.probabilities[i],abs(_QuantumUtils.frob_inner_prod(self, self.measure_basis[i], sigma)))
        # print('\nOBJECTIVE VALUE')
        # print('Original state      Estimated state') 
        # print(qutils.frob_inner_prod(qutils.C(0),qutils.from_density_to_sdp_solution(ro)),qutils.frob_inner_prod(qutils.C(0),ipm_exact))
        # print('\nCONSTRAINTS RESIDUAL')
        # print('Original state      Estimated state') 
        # count_1, count_2 = 0,0
        # for i in range(m):
        #     count_1 = max(count_1, qutils.frob_inner_prod(qutils.A(0,exact_data=True)[i],qutils.from_density_to_sdp_solution(ro))-qutils.b(0,exact_data=True)[i])
        #     count_2 = max(count_2, qutils.frob_inner_prod(qutils.A(0,exact_data=True)[i],ipm_exact)-qutils.b(0,exact_data=True)[i])
        #     print(qutils.A(0,exact_data=True)[i],qutils.b(0,exact_data=True)[i])
        # print(count_1, count_2)
        # print('\nDISTRIBUTION BLOCK')
        # M = qutils.M
        # print(qutils.from_density_to_sdp_solution(ro)[1:M+2,1:M+2])
        # print(ipm_exact[1:qutils.M+2,1:M+2])
        # print('\nREAL STATE BLOCK')
        # print(qutils.from_density_to_sdp_solution(ro)[M+2: 4**N + M + 2,M+2: 4**N + M + 2])
        # print(ipm_exact[M+2: 4**N + M + 2,M+2: 4**N + M + 2])