"""
matlib.py

Put any requested function or class definitions in this file.  You can use these in your script.

Please use comments and docstrings to make the file readable.
"""

# HOUSEKEEPING #
import scipy.linalg as la
import numpy as np
from numba import njit

# Problem 0

# Part A
def solve_chol(A, b):
    '''
    solve A * x = b for x
    
    use Cholesky decomposition
    '''
    
    L = la.cholesky(A, lower=True) # cholesky decomposition
    x = la.solve_triangular(
            L.T,
            la.solve_triangular(
                L,
                b,
                lower = True
            ),
            lower = False
        )
    return x

# Part B
def solve_lu(A, b):
    """
    solve A * x = b for x
    
    use LU decomposition
    """
    P, L, U = la.lu(A)
    x = la.solve_triangular(
            U,
            la.solve_triangular(
                L,
                P.T @ b,
                lower=True
            ),
            lower=False
        )
    return x

# Part C
def matrix_pow(A, n):
    '''
    Compute matrix power using eigenvalue decomposition
    
    Assume A is symmetric
    '''
    L, Q = la.eigh(A)
    Ln = L ** n
    An = Q @ np.diag(Ln) @ Q.T
    return An

# Part D

def abs_det(A):
    '''
    Compute the absolute value of the determinant of A 
    Use LU decomposition
    Assume A is a square matrix
    '''
    P, L, U = la.lu(A)
    detU = np.prod(np.diag(U)) # determinant of upper triangular matrix
    
    return np.abs(detU) # abosulte values of determinant of P and L are 1

# Problem 1 Matrix-Matrix Multiplication

# Part A

@njit
def matmul_ijk(B,C):
    p, r1 = B.shape
    r2, q = C.shape
    
    if r1 != r2:
        raise AssertionError("Imcompatible")
        
    r = r1
    A = np.zeros((p, q))
    for i in range(p):
        for j in range(q):
            for k in range(r):
                 A[i,j] = A[i,j] + B[i,k] * C[k,j]
    return A

@njit
def matmul_ikj(B,C):
    p, r1 = B.shape
    r2, q = C.shape
    
    if r1 != r2:
        raise AssertionError("Imcompatible")
        
    r = r1
    A = np.zeros((p, q))
    for i in range(p):
        for k in range(r):
            for j in range(q):
                 A[i,j] = A[i,j] + B[i,k] * C[k,j]
    return A

@njit
def matmul_jik(B,C):
    p, r1 = B.shape
    r2, q = C.shape
    
    if r1 != r2:
        raise AssertionError("Imcompatible")
        
    r = r1
    A = np.zeros((p, q))
    for j in range(q):
        for i in range(p):
            for k in range(r):
                 A[i,j] = A[i,j] + B[i,k] * C[k,j]
    return A

@njit
def matmul_jki(B,C):
    p, r1 = B.shape
    r2, q = C.shape
    
    if r1 != r2:
        raise AssertionError("Imcompatible")
        
    r = r1
    A = np.zeros((p, q))
    for j in range(q):
        for k in range(r):
            for i in range(p):
                 A[i,j] = A[i,j] + B[i,k] * C[k,j]
    return A

@njit
def matmul_kij(B,C):
    p, r1 = B.shape
    r2, q = C.shape
    
    if r1 != r2:
        raise AssertionError("Imcompatible")
        
    r = r1
    A = np.zeros((p, q))
    for k in range(r):
        for i in range(p):
            for j in range(q):
                 A[i,j] = A[i,j] + B[i,k] * C[k,j]
    return A

@njit
def matmul_kji(B,C):
    p, r1 = B.shape
    r2, q = C.shape
    
    if r1 != r2:
        raise AssertionError("Imcompatible")
        
    r = r1
    A = np.zeros((p, q))
    for k in range(r):
        for j in range(q):
            for i in range(p):
                 A[i,j] = A[i,j] + B[i,k] * C[k,j]
    return A

# Part B

@njit
def matmul_blocked(B, C):
	p, r1 = B.shape
	r2, q = C.shape

	if not (p == r1 == r2 == q):
		raise AssertionError("Not square matrices")

	n = p

	if n < 65:
		return matmul_ikj(B, C)

	A = np.zeros((n, n))

	slices = (slice(0, n//2), slice(n//2, n))
	for I in slices:
		for K in slices:
			for J in slices:
				A[I, J] = A[I, J] + matmul_blocked(B[I, K], C[K, J])

	return A

# Part C

@njit
def matmul_strassen(B, C):
	p, r1 = B.shape
	r2, q = C.shape

	if not (p == r1 == r2 == q):
		raise AssertionError("Not square matrices")

	n = p

	if n < 65:
		return matmul_ikj(B, C)

	A = np.zeros((n, n))

	s1 = slice(0, n//2)
	s2 = slice(n//2, n)

	B11, B12, B21, B22 = B[s1,s1], B[s1,s2], B[s2, s1], B[s2, s2]
	C11, C12, C21, C22 = C[s1,s1], C[s1,s2], C[s2, s1], C[s2, s2]

	M1 = matmul_strassen((B11 + B22), (C11 + C22))
	M2 = matmul_strassen((B21 + B22), C11)
	M3 = matmul_strassen(B11, (C12 - C22))
	M4 = matmul_strassen(B22, (C21 - C11))
	M5 = matmul_strassen((B11 + B12), C22)
	M6 = matmul_strassen((B21 - B11), (C11 + C12))
	M7 = matmul_strassen((B12 - B22), (C21 + C22))

	A[s1, s1] = M1 + M4 - M5 + M7
	A[s1, s2] = M3 + M5
	A[s2, s1] = M2 + M4
	A[s2, s2] = M1 - M2 + M3 + M6

	return A

# Problem 2 Markov Chains

# Q1
def markov_matrix(n):
    '''
    generate matrix A for the random walk on the sidewalk of length n
    '''
    
    A = np.zeros((n, n))
    
    for i in range(1, n-1):
        A[i-1,i] = 0.5
        A[i+1,i] = 0.5
    
    # modifications for the endpoints
    A[0,0], A[1,0]         = 0.5, 0.5
    A[n-2,n-1], A[n-1,n-1] = 0.5, 0.5
    
    return A
    






