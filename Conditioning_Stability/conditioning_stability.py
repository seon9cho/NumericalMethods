# condition_stability.py
"""Volume 1: Conditioning and Stability.
<Name>
<Class>
<Date>
"""

import numpy as np
import sympy as sy
import scipy.linalg as la
import matplotlib.pyplot as plt


# Problem 1
def matrix_cond(A):
    """Calculate the condition number of A with respect to the 2-norm."""
    s_vals = la.svdvals(A)
    # if the smallest singular value is 0, then A is singular so the condition number is inf.
    if s_vals[-1] == 0:
        return np.inf
    else:
        return s_vals[0]/s_vals[-1]

# Problem 2
def prob2():
    """Randomly perturb the coefficients of the Wilkinson polynomial by
    replacing each coefficient c_i with c_i*r_i, where r_i is drawn from a
    normal distribution centered at 1 with standard deviation 1e-10.
    Plot the roots of 100 such experiments in a single figure, along with the
    roots of the unperturbed polynomial w(x).

    Returns:
        (float) The average absolute condition number.
        (float) The average relative condition number.
    """
    w_roots = np.arange(1, 21)

    # Get the exact Wilkinson polynomial coefficients using SymPy.
    x, i = sy.symbols('x i')
    w = sy.poly_from_expr(sy.product(x-i, (i, 1, 20)))[0]
    w_coeffs = np.array(w.all_coeffs())
    abs_cond = []
    rel_cond = []
    # First, plot the Wilkinson roots.
    plt.ion()
    plt.scatter(np.real(w_roots), np.imag(w_roots), label="Original")
    for i in range(100):
        # Perturb each of the coefficients
        r = np.random.normal(1, 1e-10, len(w_coeffs))
        new_coeffs = w_coeffs * r
        new_roots = np.sort(np.roots(np.poly1d(new_coeffs)))
        # Plot each of the perturbed results
        if i == 0:
            plt.plot(np.real(new_roots), np.imag(new_roots), ',', c='k', label="Perturbed")
        else:
            plt.plot(np.real(new_roots), np.imag(new_roots), ',', c='k')
        # Store the absolute and relative condition numbers
        k = la.norm(new_roots - w_roots, np.inf) / la.norm(r, np.inf)
        abs_cond.append(k)
        rel_k = k * la.norm(w_coeffs, np.inf) / la.norm(w_roots, np.inf)
        rel_cond.append(rel_k)
    plt.xlabel("Real Axis")
    plt.ylabel("Imaginary Axis")
    plt.legend()
    plt.show()
    # Return the average of the condition numbers
    return np.mean(abs_cond), np.mean(rel_cond)


# Problem 3
def eig_cond(A):
    """Approximate the condition numbers of the eigenvalue problem at A.

    Parameters:
        A ((n,n) ndarray): A square matrix.

    Returns:
        (float) The absolute condition number of the eigenvalue problem at A.
        (float) The relative condition number of the eigenvalue problem at A.
    """
    # Create an n x n perturbation matrix H
    reals = np.random.normal(0, 1e-10, A.shape)
    imags = np.random.normal(0, 1e-10, A.shape)
    H = reals + 1j*imags
    # Calculate the eigenvalues of the original and the perturbed matrix
    A_eigs = la.eigvals(A)
    new_eigs = la.eigvals(A+H)
    # Solve for the condition numbers
    k = la.norm(A_eigs - new_eigs, 2) / la.norm(H, 2)
    rel_k = k * la.norm(A, 2) / la.norm(A_eigs, 2)
    return k, rel_k

# Problem 4
def prob4(domain=[-100, 100, -100, 100], res=50):
    """Create a grid [x_min, x_max] x [y_min, y_max] with the given resolution. For each
    entry (x,y) in the grid, find the relative condition number of the
    eigenvalue problem, using the matrix   [[1, x], [y, 1]]  as the input.
    Use plt.pcolormesh() to plot the condition number over the entire grid.

    Parameters:
        domain ([x_min, x_max, y_min, y_max]):
        res (int): number of points along each edge of the grid.
    """
    # Create the x and y domains depending on the bounds and the resolution
    x_dom = np.linspace(domain[0], domain[1], res)
    y_dom = np.linspace(domain[2], domain[3], res)
    X, Y = np.meshgrid(x_dom, y_dom)
    # Create a matrix that corresponds to each of the grid points
    C = []
    for x in x_dom:
        temp = []
        for y in y_dom:
            M = np.array([[1,x], [y,1]])
            k, r = eig_cond(M)
            temp.append(r)
        C.append(temp)

    # Plot the results in a pcolormesh
    plt.pcolormesh(X, Y, C, cmap='gray_r')
    plt.colorbar()
    plt.show()


# Problem 5
def prob5(n):
    """Approximate the data from "stability_data.npy" on the interval [0,1]
    with a least squares polynomial of degree n. Solve the least squares
    problem using the normal equation and the QR decomposition, then compare
    the two solutions by plotting them together with the data. Return
    the mean squared error of both solutions, ||Ax-b||_2.

    Parameters:
        n (int): The degree of the polynomial to be used in the approximation.

    Returns:
        (float): The forward error using the normal equations.
        (float): The forward error using the QR decomposition.
    """
    # Load the data and create the vandermonde matrix
    xk, yk = np.load("stability_data.npy").T
    A = np.vander(xk, n+1)
    # Solve for the least square solution using the inverse of A.T@A
    c1 = la.inv(A.T @ A) @ A.T @ yk
    Q, R = la.qr(A, mode='economic')
    # Solve for the least square solution using QR decomposition
    c2 = la.solve_triangular(R, Q.T@yk)
    domain = np.linspace(0, 1, 500)
    # Plot the two graphs along with the original data
    plt.plot(xk, yk, 'k*', ms=2, label="Data points")
    plt.plot(domain, np.polyval(c1, domain), label="Normal Equations")
    plt.plot(domain, np.polyval(c2, domain), label="QR Solver")
    plt.legend()
    plt.show()
    # Solve for the condition numbers
    fe1 = la.norm(A@c1 - yk, 2)
    fe2 = la.norm(A@c2 - yk, 2)
    return fe1, fe2

# Problem 6
def prob6():
    """For n = 5, 10, ..., 50, compute the integral I(n) using SymPy (the
    true values) and the subfactorial formula (may or may not be correct).
    Plot the relative forward error of the subfactorial formula for each
    value of n. Use a log scale for the y-axis.
    """
    # Create a list containing the values of n
    N = [i * 5 for i in range(1,11)]
    errors = []
    for n in N:
        # Solve for the integral using sy.integrate()
        x = sy.symbols('x')
        f = x**n * sy.exp(x - 1)
        r1 = float(sy.integrate(f, (x, 0, 1)))
        # Solve for the integral using the equation 10.6
        r2 = float((-1)**n * sy.subfactorial(n)) + (-1)**(n+1) * sy.factorial(n)/np.exp(1)
        ref = abs(r1-r2)/abs(r1)
        errors.append(ref)
    # Plot the resulting errors in a semilogy graph
    plt.semilogy(N, errors)
    plt.xlabel("n")
    plt.ylabel("Reletive forward error")

    '''
    10.6 is a stable way to compute I(n) since when calculated this way, the 
    relative forward error stays at around 1 even for large n.
    '''

