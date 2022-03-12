"""
life.py

Put any requested function or class definitions in this file.  You can use these in your script.

Please use comments and docstrings to make the file readable.
"""

# Housekeeping
import numpy as np
import scipy.sparse as sparse

# Example from class

def neighbors(i, j, m, n):
    inbrs = [-1, 0, 1]
    if i == 0:
        inbrs = [0, 1]
    if i == m-1:
        inbrs = [-1, 0]
    jnbrs = [-1, 0, 1]
    if j == 0:
        jnbrs = [0, 1]
    if j == n-1:
        jnbrs = [-1, 0]

    for delta_i in inbrs:
        for delta_j in jnbrs:
            if delta_i == delta_j == 0:
                continue
            yield i + delta_i, j + delta_j

def count_alive_neighbors(S):
    m, n = S.shape
    cts = np.zeros(S.shape, dtype=np.int64)
    for i in range(m):
        for j in range(n):
            for i2, j2 in neighbors(i, j, m, n):
                cts[i,j] = cts[i,j] + S[i2, j2]

    return cts

# Part A

def flat_index(S,i,j):
    '''
    Find the index of k of flatted array s
    s[k] = S[i,j]
    '''
    m, n = S.shape
    index = np.array([i,j])
    k = np.ravel_multi_index(index, (m,n))
    
    return k

def coordinate_index(S,k):
    '''
    Find the index of (i,j) of 2-dimensional array S
    S[i,j] = s[k]
    '''
    m, n  = S.shape
    index = np.unravel_index(k,(m,n))
    i = index[0]
    j = index[1]
    
    return i,j

def two_dim(s,m,n):
    '''
    Convert the one-dimensional s to m by n two-domensional S
    '''
    S = np.reshape(s, (m, n))
    
    return S

# Part B

def grid_adjacency(m,n):
    """
    returns the adjacency matrix for an m x n grid
    """
    k = m*n
    A = sparse.dok_matrix((k,k), dtype=np.int64)
    
    for r in range(m):
        for c in range(n):
            
            i =  r*n + c # index in one-dimensional array s
            
            
            if c>0:
                A[i-1,i] = A[i,i-1] = 1 # same row
                
            if r>0:
                A[i-n,i] = A[i,i-n] = 1 # same column
            
            if c<n-1 and r>0:
                A[i-n+1,i] = A[i,i-n+1] = 1 # diagonal neighbors
                
            if c>0 and r>0:
                A[i-n-1,i] = A[i,i-n-1] = 1 # diagonal neighbors
    
    
    return A

# Part C

def count_alive_neighbors_matmul(S, A):
    """
    return counts of alive neighbors in the state array S.

    Uses matrix-vector multiplication on a flattened version of S
    """
    
    s = S.flatten()
    c = A @ s
    
    cts = np.reshape(c, S.shape)
    
    return cts
    
# Part D

def count_alive_neighbors_slice(S):
    """
    return counts of alive neighbors in the state array S.

    Uses slices of cts and S
    """
    cts = np.zeros(S.shape, dtype=np.int64)

    cts[1:, :]  = cts[1:, :]  + S[:-1, :]    # upper
    cts[:-1, :] = cts[:-1, :] + S[1:, :]     # lower
    cts[:, 1:]  = cts[:, 1:]  + S[:, :-1]    # left
    cts[:, :-1] = cts[:, :-1] + S[:, 1:]     # right
    cts[1:, 1:] = cts[1:, 1:] + S[:-1, :-1]  # upper-left
    cts[1:,:-1] = cts[1:,:-1] + S[:-1,1:]    # upper-right
    cts[:-1,1:] = cts[:-1,1:] + S[1:,:-1]    # lower-left
    cts[:-1,:-1]= cts[:-1,:-1]+ S[1:,1:]     # lower-right
    
    return cts
