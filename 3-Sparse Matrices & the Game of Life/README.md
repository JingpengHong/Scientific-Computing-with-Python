# Homework 3 - Sparse Matrices and The Game of Life

This homework is intentionally short so you can get started with the group project.

Please put any written answers in [`answers.md`](answers.md)

Reminder: you can embed images in markdown.  If you are asked to make an image, save it as a `png` file (or `gif` for animation), commit it to git, and embed it in `answers.md`.

Please put functions and class definitions in [`life.py`](life.py).  This file should contain a Python *module*.

Please put your code to generate plots, run experiments etc. in [`script.py`](script.py).  This file is a Python *script*.

You can run
```
conda install --file requirements.txt
```
to install the packages in [`requirements.txt`](requirements.txt)

## Important Information

### Due Date
This assignment is due Friday, October 29 at 12pm (noon) Chicago time.

### Grading Rubric

The following rubric will be used for grading.

|   | Autograder | Correctness | Style | Total |
|:-:|:-:|:-:|:-:|:-:|
| Problem 0 |  |   |  | /50 |
| Part A |    | /7 | /3 | /10 |
| Part B | /4 | /3 | /3 | /10 |
| Part C | /2 | /5 | /3 | /10 |
| Part D | /2 | /5 | /3 | /10 |
| Part E |    | /5 | /5 | /10 |


Correctness will be based on code (i.e. did you provide was was aksed for) and the content of [`answers.md`](answers.md).

To get full points on style you should use comments to explain what you are doing in your code and write docstrings for your functions.  In other words, make your code readable and understandable.

### Autograder

You can run tests for the functions you will write in problems 1-3 using either `unittest` or `pytest` (you may need to `conda install pytest`).  From the command line:
```
python -m unittest test.py
```
or
```
pytest test.py
```
The tests are in [`test.py`](test.py).  You do not need to modify (or understand) this code.

You need to pass all tests to receive full points.

Please enable GitHub Actions on your repository (if it isn't already) - this will cause the autograder to run automatically every time you push a commit to GitHub, and you can get quick feedback.


## Problem 0 - The Game of Life (50 points)

In this problem, you'll implement the game of life in several different ways.  You may wish to look at the [example from class](https://caam37830.github.io/book/09_computing/agent_based_models.html#the-game-of-life).

Recall that we defined a function `count_alive_neighbors`
```python
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
```

The update for the cells in the Game of life can be written:
```python
cts = count_alive_neighbors(S)
# Game of life update
S = np.logical_or(
    np.logical_and(cts == 2, S),
    cts == 3
)
```

Something to keep in mind is that booleans `True` and `False` are treated as the integers 1 and 0 in arithmetic, so `True + True = 2`.

This means that `cts[i,j]` is just a linear combination of elements of `S`.  We would like to express the computation of `cts` in terms of matrix-vector multiplication

### Part A (10 points)

Suppose `S` is a `m` by `n` numpy array.  Then `S.flatten()` is a 1-dimensional numpy array of length `m*n`.  We'll let `s = S.flatten()`.

Read the documentation for `np.ravel_multi_index`.  How can you compute `k` so that `s[k]` is the same entry as `S[i,j]`?

Read the documentation for `np.unravel_index`.  How can you compute `i,j ` so that `S[i,j]` is the same entry as `s[k]`?

Read the documentation for `np.reshape`.  How can you turn the array `s` back into the 2-dimensional array `S`?

### Part B (10 points)

Suppose that `s = S.flatten()`, where `S` is the array encoding the state of cells in the Game of Life.

Let `c` be equivalent to `cts.flatten()`.  We can compute `c = A @ s` where `A` is the *adjacency matrix* of the grid that the states `S` live on.  I.e. `A[i,j] = 1` if `i` is a neighbor of `j`, and `A[i,j] = 0` if `i` and `j` are not neighbors.

Define a function
```python
def grid_adjacency(m,n):
    """
    returns the adjacency matrix for an m x n grid
    """
```
The return type should be a sparse matrix type from `scipy.sparse`.  Give a brief explanation of why you chose the matrix type that you did.

### Part C (10 points)

Convert the matrix you obtain into the following 3 forms:
* `csc_matrix`
* `csr_matrix`
* `dia_matrix`

Test matrix multiplication `c = A @ s` using each of the three forms, where `s` comes from a random initialization of the Game of Life.  Which is fastest on a 100 x 100 grid?  Which is fastest on a 1000 x 1000 grid?

Implement a function
```python
def count_alive_neighbors_matmul(S, A):
    """
    return counts of alive neighbors in the state array S.

    Uses matrix-vector multiplication on a flattened version of S
    """
```

which returns the same output as `count_alive_neighbors`.

Is this function faster or slower than `count_alive_neighbors`?  Does it depend on the sparse matrix type?


### Part D (10 points)

Another way to compute `cts` is to use vectorized numpy operations and slicing.  For example, we can compute the sum of all neighbors directly above using
```python
cts = np.zeros(S.shape)
cts[1:, :] = cts[1:, :] + S[:-1, :]
```

Write a function `count_alive_neighbors_slice`
which has the same inputs and outputs as `count_alive_neighbors`, but which is implemented using slices of `cts` and `S` (there should be 8 slices of each - one for each direction a neighbor can be located).

Is this function faster or slower than `count_alive_neighbors`?  What about `count_alive_neighbors_matmul`?

In terms of memory access of `A` and `cts`, which of the three matrix types in part C would do matrix-vector multiplication in a way most similar to `count_alive_neighbors_slice`?

### Part E (10 points)

Choose your favorite method of counting neighbors.  

Experiment with `np.random.seed` and the sparsity of the initial state of `S` to initialize the game of life.  Create a GIF of your favorite simulation which runs for 50 frames, and embed it in `answers.py`.


## Feedback

If you'd like share how long it took you to complete this assignment, it will help adjust the difficulty for future assignments.  You're welcome to share additional feedback as well.
