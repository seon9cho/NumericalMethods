# solutions.py
# iterative_solvers.py
"""Volume 1: Iterative Solvers.
<Name>
<Class>
<Date>
"""

import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt
from scipy import sparse

# Helper function
def diag_dom(n, num_entries=None):
    """Generate a strictly diagonally dominant (n, n) matrix.

    Parameters:
        n (int): The dimension of the system.
        num_entries (int): The number of nonzero values.
            Defaults to n^(3/2)-n.

    Returns:
        A ((n,n) ndarray): A (n, n) strictly diagonally dominant matrix.
    """
    if num_entries is None:
        num_entries = int(n**1.5) - n
    A = np.zeros((n,n))
    rows = np.random.choice(np.arange(0,n), size=num_entries)
    cols = np.random.choice(np.arange(0,n), size=num_entries)
    data = np.random.randint(-4, 4, size=num_entries)
    for i in range(num_entries):
        A[rows[i], cols[i]] = data[i]
    for i in range(n):
        A[i,i] = np.sum(np.abs(A[i])) + 1
    return A

# Problems 1 and 2
def jacobi_method(A, b, tol=1e-8, maxiters=100, plot=False):
    """Calculate the solution to the system Ax = b via the Jacobi Method.

    Parameters:
        A ((n,n) ndarray): A square matrix.
        b ((n ,) ndarray): A vector of length n.
        tol (float): The convergence tolerance.
        maxiters (int): The maximum number of iterations to perform.
        plot (bool): If True, plot the convergence rate of the algorithm
            (for Problem 2).

    Returns:
        ((n,) ndarray): The solution to system Ax = b.
    """
    n = len(b)
    # Initial guess
    x0 = np.zeros(n)
    d = np.diag(A)
    # Keep track of the errors in case plot=True
    error = []
    for i in range(maxiters):
        # 16.2
        x1 = x0 + (b-A@x0)/d
        error.append(la.norm(x1-x0, ord=np.inf))
        # Stopping criteria
        if la.norm(x1-x0, ord=np.inf) < tol:
            # Plot the error as a function of iterations
            if plot == True:
                plt.semilogy(np.arange(0,i+1), error)
                plt.title("Convergence of Jacobi Method")
                plt.xlabel("Iteration")
                plt.ylabel("Absolute Error of Approximation")
                plt.show()
            return x1
        x0 = x1
    # Plot the error as a function of iterations
    if plot == True:
        plt.semilogy(np.arange(0,maxiters), error)
        plt.title("Convergence of Jacobi Method")
        plt.xlabel("Iteration")
        plt.ylabel("Absolute Error of Approximation")
        plt.show()
    return x1

# Problem 3
def gauss_seidel(A, b, tol=1e-8, maxiters=100, plot=False):
    """Calculate the solution to the system Ax = b via the Gauss-Seidel Method.

    Parameters:
        A ((n, n) ndarray): A square matrix.
        b ((n, ) ndarray): A vector of length n.
        tol (float): The convergence tolerance.
        maxiters (int): The maximum number of iterations to perform.
        plot (bool): If true, plot the convergence rate of the algorithm.

    Returns:
        x ((n,) ndarray): The solution to system Ax = b.
    """
    n = len(b)
    # Initial guess
    x0 = np.zeros(n)
    # Keep track of the errors in case plot=True
    error = []
    for i in range(maxiters):
        # 16.4
        x1 = np.copy(x0)
        for j in range(len(x0)):
            x1[j] += 1/A[j,j]*(b[j] - np.dot(A[j],x1))
        error.append(la.norm(x1-x0, ord=np.inf))
        # Stopping criteria
        if la.norm(x1-x0, ord=np.inf) < tol:
            # Plot the error as a function of iterations
            if plot == True:
                plt.semilogy(np.arange(0,i+1), error)
                plt.title("Convergence of Gauss Seidel Method")
                plt.xlabel("Iteration")
                plt.ylabel("Absolute Error of Approximation")
                plt.show()
            return x1
        x0 = x1
    # Plot the error as a function of iterations
    if plot == True:
        plt.semilogy(np.arange(0,maxiters), error)
        plt.title("Convergence of Gauss Seidel Method")
        plt.xlabel("Iteration")
        plt.ylabel("Absolute Error of Approximation")
        plt.show()
    return x1

# Problem 4
def gauss_seidel_sparse(A, b, tol=1e-8, maxiters=100):
    """Calculate the solution to the sparse system Ax = b via the Gauss-Seidel
    Method.

    Parameters:
        A ((n, n) csr_matrix): A (n, n) sparse CSR matrix.
        b ((n, ) ndarray): A vector of length n.
        tol (float): The convergence tolerance.
        maxiters (int): the maximum number of iterations to perform.

    Returns:
        x ((n,) ndarray): The solution to system Ax = b.
    """
    n = len(b)
    # Initial guess
    x0 = np.zeros(n)
    for i in range(maxiters):
        # 16.4 modified for csr matrix
        x1 = np.copy(x0)
        for j in range(len(x0)):
            rowstart = A.indptr[j]
            rowend = A.indptr[j+1]
            Ajx = A.data[rowstart:rowend] @ x1[A.indices[rowstart:rowend]]
            x1[j] += 1/A[j,j]*(b[j] - Ajx)
        # Stopping criteria
        if la.norm(x1-x0, ord=np.inf) < tol:
            return x1
        x0 = x1
    return x1

# Problem 5
def sor(A, b, omega, tol=1e-8, maxiters=100):
    """Calculate the solution to the system Ax = b via Successive Over-
    Relaxation.

    Parameters:
        A ((n, n) csr_matrix): A (n, n) sparse matrix.
        b ((n, ) Numpy Array): A vector of length n.
        omega (float in [0,1]): The relaxation factor.
        tol (float): The convergence tolerance.
        maxiters (int): The maximum number of iterations to perform.

    Returns:
        ((n,) ndarray): The solution to system Ax = b.
        (int): The number of iterations computed.
    """
    n = len(b)
    # Initial guess
    x0 = np.zeros(n)
    for i in range(maxiters):
        # 16.5
        x1 = np.copy(x0)
        for j in range(len(x0)):
            rowstart = A.indptr[j]
            rowend = A.indptr[j+1]
            Ajx = A.data[rowstart:rowend] @ x1[A.indices[rowstart:rowend]]
            x1[j] += omega/A[j,j]*(b[j] - Ajx)
        # Stopping criteria
        if la.norm(x1-x0, ord=np.inf) < tol:
            return x1, i
        x0 = x1
    return x1, maxiters

# Problem 6
def hot_plate(n, omega, tol=1e-8, maxiters=100, plot=False):
    """Generate the system Au = b and then solve it using sor().
    If show is True, visualize the solution with a heatmap.

    Parameters:
        n (int): Determines the size of A and b.
            A is (n^2, n^2) and b is one-dimensional with n^2 entries.
        omega (float in [0,1]): The relaxation factor.
        tol (float): The iteration tolerance.
        maxiters (int): The maximum number of iterations.
        plot (bool): Whether or not to visualize the solution.

    Returns:
        ((n^2,) ndarray): The 1-D solution vector u of the system Au = b.
        (int): The number of computed iterations in SOR.
    """
    def create_A(n):
        """Let I be the n × n identity matrix, and define
                        [B I        ]        [-4  1            ]
                        [I B I      ]        [ 1 -4  1         ]
                    A = [  I . .    ]    B = [    1  .  .      ],
                        [      . . I]        [          .  .  1]
                        [        I B]        [             1 -4]
        where A is (n**2,n**2) and each block B is (n,n).
        Construct and returns A as a sparse matrix.
    
        Parameters:
            n (int): Dimensions of the sparse matrix B.
    
        Returns:
            A ((n**2,n**2) SciPy sparse matrix)
        """
        Iden = sparse.diags([1], [0], shape=(n,n))
        B = sparse.diags([1,-4,1], [-1,0,1], shape=(n,n))
        A = sparse.lil_matrix((n**2, n**2))
        for i in range(n):
            A[i*n:(i+1)*n,i*n:(i+1)*n] = B
            if i > 0:
                A[(i-1)*n:i*n,i*n:(i+1)*n] = Iden
                A[i*n:(i+1)*n,(i-1)*n:i*n] = Iden
        return A
    A = sparse.csr_matrix(create_A(n))
    c = np.zeros(n)
    c[0] = -100
    c[-1] = -100
    b = np.tile(c, n)
    u, i = sor(A, b, omega, tol=tol, maxiters=maxiters)
    if plot == True:
        plt.pcolormesh(u.reshape((n,n)), cmap="coolwarm")
        plt.show()
    
    return u, i

# Problem 7
def prob7():
    """Run hot_plate() with omega = 1, 1.05, 1.1, ..., 1.9, 1.95, tol=1e-2,
    and maxiters = 1000 with A and b generated with n=20. Plot the iterations
    computed as a function of omega.
    """
    omegas = np.arange(100, 200, 5)/100
    n = 20
    iterations = []
    for omega in omegas:
        u, i = hot_plate(n, omega, tol=1e-2, maxiters=1000)
        iterations.append(i)
    plt.plot(omegas, iterations)
    plt.ylabel("iterations")
    plt.xlabel("omega")
    plt.show()
    return omegas[np.argmin(iterations)]

'''
n = 10
b = np.random.random(n)
A = diag_dom(n)
x = jacobi_method(A, b, plot=True)
print(A@x, b)
print(np.allclose(A@x, b))
A2 = np.random.random((n,n))
x2 = jacobi_method(A2, b, plot=True)
print(np.allclose(A2@x2, b))
x3 = gauss_seidel(A, b, plot=True)
print(A@x3, b)
print(np.allclose(A@x3, b))
A2 = np.random.random((n,n))
x4 = gauss_seidel(A2, b, plot=True)
print(np.allclose(A2@x4, b))
A = sparse.csr_matrix(diag_dom(5000))
b = np.random.random(5000)
x = gauss_seidel_sparse(A, b, maxiters=10000, tol=1e-12)
print(np.allclose(A@x, b))
A = sparse.csr_matrix(diag_dom(5000))
b = np.random.random(5000)
x = sor(A, b, 0.9, maxiters=10000, tol=1e-12)
print(np.allclose(A@x, b))
print(hot_plate(16, 1.9, plot=True, maxiters=1000))
'''
print(prob7())