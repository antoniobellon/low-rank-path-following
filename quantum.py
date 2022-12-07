import matplotlib.pyplot as plt
import numpy as np 
import predictor_corrector as pc  
import mosek_ipm_solver as ip 
import experiments.ipm_tracker as it 
import parameters as par 
import quantum_utilities as qu
import mmw_tomography as mt
from scipy.linalg import sqrtm

from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
from qiskit import BasicAer
from qiskit.compiler import transpile
from qiskit.quantum_info.operators import Operator, Pauli
from qiskit.quantum_info import state_fidelity, random_density_matrix, DensityMatrix
from qiskit_experiments.library.tomography import basis
from qiskit.extensions import RXGate, XGate, CXGate
from scipy.linalg import expm
 
N = 2                                     # number of qubits 
M = 4**N                                  # number of measurements 
T = 300                                   # number of samples for each measurement 

ro = random_density_matrix(2**N)

qutils = qu._QuantumUtils(ro=ro, N=N, T=T)

n = qutils.n
m = qutils.m

measure_basis = qutils.measure_basis
real_measure_basis = qutils.real_measure_basis
probabilities = qutils.probabilities
approx_probabilities = qutils.approx_probabilities
 
# Solve SDP problem using EXACT distributions 
run_time, ipm_exact, dual_sol = ip._get_SDP_solution(n=n, m=m, A=qutils.A(0, exact_data=True), b=qutils.b(0, exact_data=True), C=qutils.C_objective, TOLERANCE=10e-14)
sdp_exact = qutils.extract_solution(ipm_exact) 
sdp_fidelity_exact = state_fidelity(ro, sdp_exact) 
 
# Solve TV-SDP problem using EMPIRICAL distributions 
ipm_track = it._IPM_tracker(n=n, m=m, initial_time=0, final_time=T, ini_stepsize=1)
ipm_track.run(qutils.A, qutils.b, qutils.C, PRINT_DATA=False)

sdp_approx = []
sdp_fidelity_approx = [] 

for sol in ipm_track._primal_solutions_list:
    sol = qutils.extract_solution(sol)
    sdp_approx.append(sol) 
    sdp_fidelity_approx.append(state_fidelity(ro, sol, False)) 

# Solve Quantum Tomography problem using Matrix Multiplicative Weights (MMW) algorithm
mmw_track = mt._MMW(qutils=qutils)
mmw_track.run_mmw()

mmw_fidelity = []
for sol in mmw_track.mmw_sols:
    mmw_fidelity.append(state_fidelity(ro, sol,False))  

# Print and plot results
qutils.print_state_comparison(sdp_exact)

plt.title("Fidelity of state estimation using different methods\n " + r"$F = \left( \operatorname{Tr} \sqrt{\sqrt{\rho} \omega \sqrt{\rho}} \right)^2$")
plt.xlabel("Measurement step")
plt.ylabel(r"Fidelity, $F$")
plt.axis([0, T, 0, 1.01])
plt.plot([sdp_fidelity_exact for i in range(T)],'r') 
plt.plot(sdp_fidelity_approx,'g')
# plt.plot(predcorr.times,path_following_fidelity,'b')
plt.plot(mmw_fidelity,'y')
plt.show()

"""
#Solve TV-SDP problem using PATH-FOLLOWING 
params = par.getParameters(print_par=False) 
ini_stepsize=0.1
Y_0, rank, lam_0  = ip._get_initial_point(n=n, m=m, A=A(float(params["problem"]["initial_time"])), 
                                                    b=b(float(params["problem"]["initial_time"])), 
                                                    C=C(float(params["problem"]["initial_time"])), 
                                                    TOLERANCE=1.0e-10)
predcorr = pc._PredictorCorrector(n=n, m=m, rank=rank, params=params, ini_stepsize=ini_stepsize, res_tol=1.0e-1) 
predcorr.run(A, b, C, Y_0, lam_0, STEPSIZE_TUNING=False, PRINT_DATA=True)

sols_path_following = []
path_following_fidelity = [] 

for sol in predcorr._primal_solutions_list:
    sol = extract_solution(sol)
    sols_path_following.append(DensityMatrix(sol))   
    path_following_fidelity.append(state_fidelity(ro, sol, False)) 

"""
"""
POSSIBLY INTERESTING STATES

# ro = DensityMatrix([[2/3, 0.1+0.1j],
#                     [0.1-0.1j,1/3]])
# ro = DensityMatrix([[0.5,0.5], 
#                     [0.5,0.5]])

# ro = DensityMatrix([[ 0.37940936+2.59017042e-19j, -0.34319942+5.39230262e-02j],
#                     [-0.34319942-5.39230262e-02j,  0.62059064-2.59017042e-19j]])
# ro = DensityMatrix([[0.74028827-3.40984013e-19j, 0.08853243+1.47399695e-01j],
#                     [0.08853243-1.47399695e-01j, 0.25971173+3.40984013e-19j]])

# THE CHERRY DENSITY MATRIX
# ro = DensityMatrix([[0.43018514+2.29021342e-18j, 0.3169516 -3.55076112e-02j],
#                     [0.3169516 +3.55076112e-02j, 0.56981486-2.29021342e-18j]])
 
# ro = DensityMatrix([[0.63335599+1.02553228e-18j, 0.03510334+3.08507217e-01j],
#                [0.03510334-3.08507217e-01j, 0.36664401-1.02553228e-18j]],
#               dims=(2,))
# ro = DensityMatrix([[ 0.81274231+1.44635475e-18j, -0.15006145-3.03190832e-02j],
#                [-0.15006145+3.03190832e-02j,  0.18725769-1.44635475e-18j]],
#               dims=(2,))

# A DENSITY MATRIX FOR WHICH THE ESTIAMTED STATE IS CLOSE BUT FIDELITY IS SMALL
# ro =  DensityMatrix([[ 0.34834704+5.06147547e-18j, -0.18407055+4.26851419e-01j],
#  [-0.18407055-4.26851419e-01j,  0.65165296-5.06147547e-18j]])
# ro =  DensityMatrix([[ 0.13005021-7.66287415e-19j, -0.08573115+3.21469586e-01j],
#  [-0.08573115-3.21469586e-01j,  0.86994979+7.66287415e-19j]])
# """