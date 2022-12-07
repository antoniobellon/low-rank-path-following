import numpy as np  
 
class _LinearizedKKTsystem:

    def __init__(self, n: int, m: int, rank: int) -> None:

        self._n = n
        self._m = m
        self._rank = rank
        self._nvars = n * rank 
        self._rantisymm = int((rank-1)*rank/2)  
        self._gradientGconstraints = np.full((self._nvars,m), fill_value=0.0)
        self._gradientHconstraints = np.full((self._nvars, self._rantisymm), fill_value=0.0)
        self._LinearizedKKTmatrix = np.full((self._nvars+m+ self._rantisymm,self._nvars+m+ self._rantisymm), fill_value=0.0)
        self._LinearizedKKTrhs = np.full((self._nvars+m+ self._rantisymm), fill_value=0.0)

    def computeMatrix(self, A: np.ndarray, C: np.ndarray, Y: np.ndarray, lam: np.ndarray) -> np.ndarray:
         
        np.copyto(self._LinearizedKKTmatrix[:self._nvars,:self._nvars],2*np.kron(np.eye(self._rank), C-np.tensordot(lam, A, 1)))

        for i in range(self._m): 
            np.copyto(self._gradientGconstraints.T[i], -2*(np.ravel(np.matmul(A[i],Y).T))) 

        np.copyto(self._LinearizedKKTmatrix[:self._nvars,self._nvars:self._nvars+self._m],self._gradientGconstraints)
        np.copyto(self._LinearizedKKTmatrix[self._nvars:self._nvars+self._m,:self._nvars],self._gradientGconstraints.T) 
        
        k = 0 
        for i in range(self._rank):
            for j in range(i+1,self._rank):
                part_1 = np.tensordot(np.reshape(Y[:,i], (self._n, 1)),np.eye(1,self._rank,j),1)
                part_2 = np.tensordot(np.reshape(Y[:,j], (self._n, 1)),np.eye(1,self._rank,i),1) 
                np.copyto(self._gradientHconstraints[:,k],((part_2-part_1).T).ravel())
                k += 1 
     
        np.copyto(self._LinearizedKKTmatrix[:self._nvars,self._nvars+self._m:],self._gradientHconstraints )
        np.copyto(self._LinearizedKKTmatrix[self._nvars+self._m:,:self._nvars],self._gradientHconstraints.T )
         
        return self._LinearizedKKTmatrix

    def computeRhs(self,  A: np.ndarray, b: np.ndarray, C: np.ndarray, Y: np.ndarray, X: np.ndarray) -> np.ndarray:

        np.copyto(self._LinearizedKKTrhs[:self._nvars],-2*(np.matmul(C,Y).T).ravel())
        for i in range(self._m):
            self._LinearizedKKTrhs[self._nvars+i]=np.tensordot(A[i],X) - b[i]
        
        return self._LinearizedKKTrhs    