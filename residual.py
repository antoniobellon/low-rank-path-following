import predictor_corrector as pc 
import numpy as np 

def resid(n: int, m:int, rank:int, A: np.ndarray, b: np.ndarray, C: np.ndarray, Y: np.ndarray,
                 lam: np.ndarray) -> float:
    """
    Computes the residual.
    """ 
    _YY =  np.full((n,n), fill_value=0.0)
    _lamTimesA = np.full((n,n), fill_value=0.0)
    lagran_grad = np.zeros((n,rank)) 
    np.dot(Y, Y.T, out=_YY)

    # Compute Lagrangian residual 
    np.copyto(_lamTimesA,np.tensordot(lam, A, 1)) 
    np.matmul(C-_lamTimesA, Y, out=lagran_grad)  
    resA = np.linalg.norm(lagran_grad.ravel(), np.inf)
     
    # Compute constraints residual 
    constr_err = -b 
    for i in range(m):
        constr_err[i] += np.dot(_YY.ravel(), A[i,:,:].ravel())  
    resB = np.linalg.norm(constr_err, np.inf) 
    res = np.array([resA, resB]) 

    return res

def SDP_resid(n: int, m:int, rank:int, A: np.ndarray, b: np.ndarray, C: np.ndarray, X: np.ndarray,
                 lam: np.ndarray) -> float:
    """
    Computes the residual.
    """  
    _lamTimesA = np.full((n,n), fill_value=0.0)
    lagran_grad = np.zeros((n,n))  

    # Compute Lagrangian residual 
    np.copyto(_lamTimesA,np.tensordot(lam, A, 1)) 
    # print(np.shape(C-_lamTimesA))
    # print(np.shape(X))
    # print(np.shape(lagran_grad))
    np.matmul(C-_lamTimesA, X, out=lagran_grad)  
    resA = np.linalg.norm(lagran_grad.ravel(), np.inf)
     
    # Compute constraints residual 
    constr_err = -b 
    for i in range(m):
        constr_err[i] += np.dot(X.ravel(), A[i,:,:].ravel())  
    resB = np.linalg.norm(constr_err, np.inf) 
    res = np.array([resA, resB]) 

    return res