"""
fibonacci

functions to compute fibonacci numbers

Complete problems 2 and 3 in this file.
"""

import time # to compute runtimes
from tqdm import tqdm # progress bar
import numpy as np
import matplotlib.pyplot as plt

# Question 2
def fibonacci_recursive(n):
     
    # check if n=0 or n=1
    if n in {0, 1}:
        return n
    
    # if n>1, F_n=F_{n-1}+F_{n-2}
    return fibonacci_recursive(n-1)+fibonacci_recursive(n-2)

# Print the first 30 Fibonacci numbers using linear recursion
recursive = [fibonacci_recursive(i) for i in range(30)]
print(f"Linear recursion: {recursive}")


# Question 2
def fibonacci_iter(n):
    
    a, b = 0, 1
    
    while n>0:
        a, b = b, a+b
        n -= 1
    
    return a

# Print the first 30 Fibonacci numbers using iteration
iteration = [fibonacci_iter(i) for i in range(30)]
print(f"Iteration: {iteration}")


# Question 3
def mat_power(x, n):
    """
    computes the power x ** n when x is a vector or matrix

    assume n is a nonegative integer
    """
    def isodd(n):
        """
        returns True if n is odd
        """
        return n & 0x1 == 1

    if n == 1:
        return x
    if n == 0:
        return np.identity(len(x))

    if isodd(n):
        return np.dot(mat_power(np.dot(x, x), n // 2), x)
    else:
        return mat_power(np.dot(x, x), n // 2)
    
def fibonacci_power(n):
    if n == 0:
        return 0
    if n == 1:
        return 1
    
    A   = np.array([[1,1],[1,0]])
    x_1 = np.array([1,0])
    
    x_n = np.dot(mat_power(A, n-1), x_1) # calculate x_n = A^(n-1) x_1
    
    return x_n[0] # F_n is the first element in x_n

# Print the first 30 Fibonacci numbers using using the matrix-vector product
power = [fibonacci_power(i) for i in range(30)]
print(f"Matrix-Vector Product: {power}")


if __name__ == '__main__':
    """
    this section of the code only executes when
    this file is run as a script.
    """
    def get_runtimes(ns, f):
        """
        get runtimes for fibonacci(n)

        e.g.
        trecursive = get_runtimes(range(30), fibonacci_recusive)
        will get the time to compute each fibonacci number up to 29
        using fibonacci_recursive
        """
        ts = []
        for n in tqdm(ns):
            t0 = time.time()
            fn = f(n)
            t1 = time.time()
            ts.append(t1 - t0)

        return ts


    nrecursive = range(35)
    trecursive = get_runtimes(nrecursive, fibonacci_recursive)

    niter = range(10000)
    titer = get_runtimes(niter, fibonacci_iter)

    npower = range(10000)
    tpower = get_runtimes(npower, fibonacci_power)

## write your code for problem 4 below...
def get_runtimes(ns, f):
    """
    get runtimes for fibonacci(n)
    """
    ts = []
    for n in ns:
        t0 = time.time()
        fn = f(n)
        t1 = time.time()
        ts.append(t1 - t0)

    return ts

# the time to compute each fibonacci number using fibonacci_recursive
nrecursive = range(35)
trecursive = get_runtimes(nrecursive, fibonacci_recursive)

# the time to compute each fibonacci number using fibonacci_iter
niter = range(10000)
titer = get_runtimes(niter, fibonacci_iter)

# the time to compute each fibonacci number using fibonacci_power
npower = range(10000)
tpower = get_runtimes(npower, fibonacci_power)

# the plot of the run times
plt.loglog(nrecursive, trecursive, '.', label="Recursion")
plt.loglog(niter, titer, '.', label="Iteration")
plt.loglog(npower, tpower, '.', label="Power")
plt.xlabel("n")
plt.ylabel("Run Times")
plt.title("Run times by algorithms")
plt.legend()
plt.savefig('fibonacci_runtime.png')
    
    
