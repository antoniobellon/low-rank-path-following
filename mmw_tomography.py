import numpy as np  
import parameters as par 
import quantum_utilities as qu  
from scipy.linalg import expm

class _MMW:

    def __init__(self, qutils: qu._QuantumUtils) -> None:
        
        self.qutils = qutils 
        self.mmw_sols = []

    def grd(self, omega, E, Y):

        return 2 * (np.trace(E * omega) - Y) * E

    def mmw_next_omega(self, 𝜔, E, Y):
        
        η = 0.05
        s = - sum([ _MMW.grd(self, 𝜔[_], E[_], Y[_]) for _ in range(len(𝜔)) ]) * η
        e = expm(s)
        return e/np.trace(e)  

    def run_mmw(self):
        
        qutils = self.qutils
        N = qutils.N
        M = qutils.M
        T = qutils.T

        omega_1 = .5*np.eye(2**N,2**N) 

        observations_list = []
        measurements_list = []

        for i in range(T):
            for j in range(M):
                observations_list.append(qutils.observations[j,i])
                measurements_list.append(qutils.measure_basis[j]) 

        self.mmw_sols.append(omega_1)

        iteration = 0
        for i in range(T):
            if iteration%10==0: print("\nITERATION", iteration)
            iteration += 1
            step = len(self.mmw_sols)
            omega_next = _MMW.mmw_next_omega(self, self.mmw_sols, measurements_list[:step], observations_list[:step])
            self.mmw_sols.append(omega_next)
       
           