"""
use this file to script the creation of plots, run experiments, print information etc.

Please put in comments and docstrings in to make your code readable
"""

# Problem 0

# Housekeeping #
from life import *
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import scipy.sparse as sparse
import time

# Part C

tcsc = list()
tcsr = list()
tdia = list()

for (m,n) in [(100,100), (1000,1000)]:
    
    # random initialization of the Game of Life
    S = np.random.rand(m, n) < 0.3
    s = S.flatten()

    # adjacency matrix
    A = grid_adjacency(m,n)
    
    # Compressed Sparse Column
    A_csc = sparse.csc_matrix(A)
    t0=time.time()
    c = A_csc @ s
    t1=time.time()
    diff = t1-t0
    tcsc.append(diff)
    
    # Compressed Sparse Row
    A_csr = sparse.csr_matrix(A)
    t0=time.time()
    c = A_csr @ s
    t1=time.time()
    diff = t1-t0
    tcsr.append(diff)
    
    # Sparse Diagonal Matrix
    A_dia = sparse.dia_matrix(A)
    t0=time.time()
    c = A_dia @ s
    t1=time.time()
    diff = t1-t0
    tdia.append(diff)
    
print(tcsc, tcsr, tdia)
    
# Compare the function count_alive_neighbors and function count_alive_neighbors_matmul

t_cts = list()
t_cts_matmul1 = list()
t_cts_matmul2 = list()
t_cts_matmul3 = list()

for (m,n) in [(100,100), (1000,1000)]:
    
    S = np.random.rand(m, n) < 0.3
    A = grid_adjacency(m,n)
    
    # count_alive_neighbors
    t0=time.time()
    cts = count_alive_neighbors(S)
    t1=time.time()
    diff = t1-t0
    t_cts.append(diff)

    # Compressed Sparse Column
    A_csc = sparse.csc_matrix(A)
    t0=time.time()
    cts = count_alive_neighbors_matmul(S, A_csc)
    t1=time.time()
    diff = t1-t0
    t_cts_matmul1.append(diff)
    
    # Compressed Sparse Row
    A_csr = sparse.csr_matrix(A)
    t0=time.time()
    cts = count_alive_neighbors_matmul(S, A_csr)
    t1=time.time()
    diff = t1-t0
    t_cts_matmul2.append(diff)
    
    # Sparse Diagonal Matrix
    A_dia = sparse.dia_matrix(A)
    t0=time.time()
    cts = count_alive_neighbors_matmul(S, A_dia)
    t1=time.time()
    diff = t1-t0
    t_cts_matmul3.append(diff)

print(t_cts, t_cts_matmul1, t_cts_matmul2, t_cts_matmul3)

# Part D
'''
Compare
`count_alive_neighbors `, 
`count_alive_neighbors_matmul` ,
`count_alive_neighbors_slice`
'''
t_cts = list()
t_cts_slice = list()
t_cts_matmul1 = list()

for (m,n) in [(100,100), (1000,1000)]:
    
    S = np.random.rand(m, n) < 0.3
    A = grid_adjacency(m,n)
    
    # count_alive_neighbors
    t0=time.time()
    cts = count_alive_neighbors(S)
    t1=time.time()
    diff = t1-t0
    t_cts.append(diff)
    
    # count_alive_neighbors_slice
    t0=time.time()
    cts = count_alive_neighbors_slice(S)
    t1=time.time()
    diff = t1-t0
    t_cts_slice.append(diff)
    
    # Sparse Diagonal Matrix
    A_csc = sparse.csc_matrix(A)
    t0=time.time()
    cts = count_alive_neighbors_matmul(S, A_csc)
    t1=time.time()
    diff = t1-t0
    t_cts_matmul1.append(diff)
    
print(t_cts, t_cts_slice, t_cts_matmul1)

# Part E

# seed 0 and sparsity 0.1
m, n = 50, 50
np.random.seed(0)
S = np.random.rand(m, n) < 0.1

fig = plt.figure(figsize=(5,5))
fig.set_tight_layout(True)

# Plot an image that persists
im = plt.imshow(S, animated=True)
plt.axis('off') # turn off ticks

def update(*args):

    global S
    
    # Update image to display next step
    cts = count_alive_neighbors_slice(S)
    
    # Game of life update
    S = np.logical_or(
        np.logical_and(cts == 2, S),
        cts == 3
    )
    im.set_array(S)
    return im,

anim = FuncAnimation(fig, update, frames=50, interval=200, blit=True)
anim.save('life_seed0_1.gif', dpi=80, writer='imagemagick')

# seed 0 and sparsity 0.3

m, n = 50, 50
np.random.seed(0)
S = np.random.rand(m, n) < 0.5

fig = plt.figure(figsize=(5,5))
fig.set_tight_layout(True)

# Plot an image that persists
im = plt.imshow(S, animated=True)
plt.axis('off') # turn off ticks

def update(*args):

    global S
    
    # Update image to display next step
    cts = count_alive_neighbors_slice(S)
    
    # Game of life update
    S = np.logical_or(
        np.logical_and(cts == 2, S),
        cts == 3
    )
    im.set_array(S)
    return im,

anim = FuncAnimation(fig, update, frames=50, interval=200, blit=True)
anim.save('life_seed0_3.gif', dpi=80, writer='imagemagick')


# seed 0 and sparsity 0.6
m, n = 50, 50
np.random.seed(100)
S = np.random.rand(m, n) < 0.1

fig = plt.figure(figsize=(5,5))
fig.set_tight_layout(True)

# Plot an image that persists
im = plt.imshow(S, animated=True)
plt.axis('off') # turn off ticks

def update(*args):

    global S
    
    # Update image to display next step
    cts = count_alive_neighbors_slice(S)
    
    # Game of life update
    S = np.logical_or(
        np.logical_and(cts == 2, S),
        cts == 3
    )
    im.set_array(S)
    return im,

anim = FuncAnimation(fig, update, frames=50, interval=200, blit=True)
anim.save('life_seed0_6.gif', dpi=80, writer='imagemagick')
