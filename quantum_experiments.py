import matplotlib.pyplot as plt
import numpy as np 
import predictor_corrector as pc  
import mosek_ipm_solver as ip 
import experiments.ipm_tracker as it 
import parameters as par 
import quantum_utilities as qu
import mmw_tomography as mt
from scipy.linalg import sqrtm
import plotly.graph_objects as go 

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
T = 209                                   # number of samples for each measurement 
SAMPLE_SIZE = 20

sdp_fidelity_list = np.ndarray((SAMPLE_SIZE,T+1), dtype=float)
mmw_fidelity_list = np.ndarray((SAMPLE_SIZE,T+1), dtype=float)

for sample_index in range(SAMPLE_SIZE):

    ro = random_density_matrix(2**N)
    qutils = qu._QuantumUtils(ro=ro, N=N, T=T)
    n = qutils.n
    m = qutils.m

    measure_basis = qutils.measure_basis
    real_measure_basis = qutils.real_measure_basis
    probabilities = qutils.probabilities
    approx_probabilities = qutils.approx_probabilities
    
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
        mmw_fidelity.append(state_fidelity(ro, sol, False))

    sdp_fidelity_list[sample_index] = sdp_fidelity_approx
    mmw_fidelity_list[sample_index] = mmw_fidelity

 
 
def plot_experiments(sdp_fidelities, mmw_fidelities):

    print("Plotting data...")
    
    fig = go.Figure() 
    
    sdp_fidelities = [fide for sample_index in range(SAMPLE_SIZE) for fide in sdp_fidelity_list[sample_index]]
    mmw_fidelities = [fide for sample_index in range(SAMPLE_SIZE) for fide in mmw_fidelity_list[sample_index]]
    ascisse = [t for y in range(SAMPLE_SIZE) for t in range(T+1)]

   
    fig.add_trace(go.Box(
        y=sdp_fidelities,
        x=ascisse,
        name=r'$\text{SDP fidelity}$', 
        marker_color='#3D9970' ,
        boxpoints = False
        )) 

    fig.add_trace(go.Box(
        y=mmw_fidelities,
        x=ascisse,
        name=r'$\text{MMW fidelity}$', 
        marker_color='#FF4136' ,
        boxpoints = False
        )) 
 
    fig.update_layout( 
        title = r'$\text{Fidelity of state estimation using different methods }$',
        width=810,
        height=500,
        yaxis_title= r'$\text{Fidelity distribution }$',
        xaxis_title=r'$\text{Measurement step } t$',
        autosize=True,
        font_family="Helvetica", 
        title_font_family="Helvetica", 
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)', 
        margin=dict(autoexpand=True)
        )

    fig.update_xaxes(
        type="category", 
        nticks=(T+1)//10,
        ticks="outside",
        rangeselector_font=dict(size=18))
    fig.update_yaxes(ticks="outside", exponentformat = 'e')
 
    fig.show()  
  
plot_experiments(sdp_fidelity_list,mmw_fidelity_list)
 