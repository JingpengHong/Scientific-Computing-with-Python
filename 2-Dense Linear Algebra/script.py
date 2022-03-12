"""
use this file to script the creation of plots, run experiments, print information etc.

Please put in comments and docstrings in to make your code readable
"""
from matlib import *
import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as la
from scipy.linalg import blas
import time 

# Problem 0

# Part B

# compare computing time between Cholesky and LU decomposition
n = np.logspace(1, np.log10(4000), num=10)
n = np.int64(np.round(n))
tchol =list()
tlu = list()

for i in n:
    A = np.random.randn(i, i)
    A = A @ A.T # generate symmetric positive definite (SPD) matrix
    x = np.random.rand(i)
    b = A @ x
    
    # compute time for Cholesky decomposition
    t0chol=time.time()
    x = solve_chol(A,b)
    t1chol=time.time()
    tcholdiff = t1chol-t0chol
    tchol.append(tcholdiff)
    
    # compute time for LU decomposition
    t0lu=time.time()
    x = solve_lu(A,b)
    t1lu=time.time()
    tludiff = t1lu-t0lu
    tlu.append(tludiff)
    
'''
Time comparison
'''

plt.loglog(np.log(n),tchol, label='Cholesky')
plt.loglog(np.log(n),tlu, label='LU')
plt.xlabel("n (in log) ")
plt.xscale('linear')
plt.ylabel("Time")
plt.legend(loc='best')
plt.title('Run times: Cholesky and LU decomposition')
plt.savefig('time_chol_lu.png')
    
# Problem 1

# Part A

n = np.logspace(np.log10(100), np.log10(4000), num=10)
n = np.int64(np.round(n))

tijk =list()
tikj =list()
tjik =list()
tjki =list()
tkij =list()
tkji =list()
tblas = list()
tnp = list()

for i in n:
    
    dtype = np.float32
    B = np.array(np.random.randn(i, i), dtype=dtype)
    C = np.array(np.random.randn(i, i), dtype=dtype)
    
    
    t0=time.time()
    A = matmul_ijk(B,C)
    t1=time.time()
    diff = t1-t0
    tijk.append(diff)
    
    t0=time.time()
    A = matmul_ikj(B,C)
    t1=time.time()
    diff = t1-t0
    tikj.append(diff)
    
    t0=time.time()
    A = matmul_jik(B,C)
    t1=time.time()
    diff = t1-t0
    tjik.append(diff)
    
    t0=time.time()
    A = matmul_jki(B,C)
    t1=time.time()
    diff = t1-t0
    tjki.append(diff)
    
    t0=time.time()
    A = matmul_kij(B,C)
    t1=time.time()
    diff = t1-t0
    tkij.append(diff)
    
    t0=time.time()
    A = matmul_kji(B,C)
    t1=time.time()
    diff = t1-t0
    tkji.append(diff)
    
    t0=time.time()
    A = blas.sgemm(1.0,B,C)
    t1=time.time()
    diff = t1-t0
    tblas.append(diff)
    
    t0=time.time()
    A = np.matmul(B,C)
    t1=time.time()
    diff = t1-t0
    tnp.append(diff)
    
plt.loglog(np.log(n),tijk, label='ijk')
plt.loglog(np.log(n),tikj, label='ikj')
plt.loglog(np.log(n),tjik, label='jik')
plt.loglog(np.log(n),tjki, label='jki')
plt.loglog(np.log(n),tkij, label='kij')
plt.loglog(np.log(n),tkji, label='kji')
plt.loglog(np.log(n),tblas, label='BLAS')
plt.loglog(np.log(n),tnp, label='matmul')

plt.xlabel("n (in log) ")
plt.ylabel("Time")
plt.xscale('linear')
plt.legend(loc='upper left', ncol=2)
plt.title('Run times: Matrix Multiplication')
plt.savefig('time_matmul.png')

# Part B Compare matmul_blocked to matmul_ikj

tmatmulb    = list()
tmatmulikj  = list()

for i in range(6,13):
    
    n = 2 ** i
        
    dtype = np.float32
    B = np.array(np.random.randn(n, n), dtype=dtype)
    C = np.array(np.random.randn(n, n), dtype=dtype)
        
    t0=time.time()
    A = matmul_blocked(B,C)
    t1=time.time()
    diff = t1-t0
    tmatmulb.append(diff)
    
    t0=time.time()
    A = matmul_ikj(B,C)
    t1=time.time()
    diff = t1-t0
    tmatmulikj.append(diff)
    
plt.loglog(2 ** np.arange(6, 13), tmatmulikj, label='matmul_ikj')
plt.loglog(2 ** np.arange(6, 13), tmatmulb, label='matmul_blocked')
plt.xscale('log', basex=2)
plt.xlabel("n")
plt.ylabel("Time")
plt.title("Run times: matmul_ikj and matmul_blocked")
plt.legend(loc='best')
plt.savefig('ikj_blocked.png')

# Part C Compare matmul_strassen to matmul_blocked

tstrassen = list()
tblocked  = list()

for i in range(6,13):
    
    n = 2 ** i
        
    dtype = np.float32
    B = np.array(np.random.randn(n, n), dtype=dtype)
    C = np.array(np.random.randn(n, n), dtype=dtype)
        
    t0=time.time()
    A = matmul_strassen(B,C)
    t1=time.time()
    diff = t1-t0
    tstrassen.append(diff)
    
    t0=time.time()
    A = matmul_blocked(B,C)
    t1=time.time()
    diff = t1-t0
    tblocked.append(diff)
    
plt.loglog(2 ** np.arange(6, 13), tstrassen, label='Strassen Algorithm')
plt.loglog(2 ** np.arange(6, 13), tblocked, label='Blocked Matrix Multiplication')
plt.xscale('log', basex=2)
plt.xlabel("n")
plt.ylabel("Time")
plt.title("Run times: matmul_strassen and matmul_blocked")
plt.legend(loc='best')
plt.savefig('strassen_blocked.png')


# Problem 2 Markov Chains

# Q2 simulation

n = 50
A = markov_matrix(n)

p0 = np.zeros(n)
p0[1] = 1 # initial position
p = list()

for t in (10, 100, 1000, 2000):
    
    At = matrix_pow(A, t) # transition matrix
    p1 = At @ p0
    p.append(p1)

plt.plot(range(50), p[0], label='t=10')
plt.plot(range(50), p[1], label='t=100')
plt.plot(range(50), p[2], label='t=1000')
plt.xlabel("Position")
plt.ylabel("Probability")
plt.title("Position Distribution in Markov Chain")
plt.legend(loc='best')
plt.savefig('markov.png')

# Q3 calculate the eigenvector with largest eigenvalue 

lam, v = la.eigh(A, eigvals=(n-1,n-1)) #calculate the largest eigenpair
v = v / np.sum(v) # normalize the eigenvector 

# the euclidean distance between v and p when t = 1000
dist1 = la.norm(v - p[2])

# the euclidean distance between v and p when t = 2000
dist2 = la.norm(v - p[3]) 


    



