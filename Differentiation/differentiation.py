# solutions.py
"""Volume 1: Differentiation.
<Name>
<Class>
<Date>
"""
import sympy as sy
import numpy as np
import matplotlib.pyplot as plt
import autograd.numpy as anp
from autograd import grad
from autograd import elementwise_grad
import time

# Problem 1
def prob1():
    """Return the derivative of (sin(x) + 1)^sin(cos(x)) using SymPy."""
    x = sy.symbols('x')
    f = (sy.sin(x) + 1)**(sy.sin(sy.cos(x)))
    f_p = sy.diff(f, x)
    return sy.lambdify(x, f_p, "numpy")

# Problem 2
def fdq1(f, x, h=1e-5):
    """Calculate the first order forward difference quotient of f at x."""
    return (f(x + h) - f(x)) / h

def fdq2(f, x, h=1e-5):
    """Calculate the second order forward difference quotient of f at x."""
    return (-3*f(x) + 4*f(x+h) - f(x+2*h))/(2*h)

def bdq1(f, x, h=1e-5):
    """Calculate the first order backward difference quotient of f at x."""
    return (f(x) - f(x-h)) / h

def bdq2(f, x, h=1e-5):
    """Calculate the second order backward difference quotient of f at x."""
    return (3*f(x) - 4*f(x-h) + f(x-2*h))/(2*h)

def cdq2(f, x, h=1e-5):
    """Calculate the second order centered difference quotient of f at x."""
    return (f(x+h) - f(x-h)) / (2*h)

def cdq4(f, x, h=1e-5):
    """Calculate the fourth order centered difference quotient of f at x."""
    return (f(x-2*h) - 8*f(x-h) + 8*f(x+h) - f(x+2*h))/(12*h)

# Problem 3
def prob3(x0):
    """Let f(x) = (sin(x) + 1)^(sin(cos(x))). Use prob1() to calculate the
    exact value of f'(x0). Then use fdq1(), fdq2(), bdq1(), bdq2(), cdq1(),
    and cdq2() to approximate f'(x0) for h=10^-8, 10^-7, ..., 10^-1, 1.
    Track the absolute error for each trial, then plot the absolute error
    against h on a log-log scale.

    Parameters:
        x0 (float): The point where the derivative is being approximated.
    """
    h = np.logspace(-8, 0, 9)
    x = sy.symbols('x')
    f = (sy.sin(x) + 1)**(sy.sin(sy.cos(x)))
    f = sy.lambdify(x, f, "numpy")
    f_p = prob1()

    o1f = []
    o2f = []
    o1b = []
    o2b = []
    o2c = []
    o4c = []
    for hk in h:
        o1f.append(abs(f_p(x0) - fdq1(f, x0, h=hk)))
        o2f.append(abs(f_p(x0) - fdq2(f, x0, h=hk)))
        o1b.append(abs(f_p(x0) - bdq1(f, x0, h=hk)))
        o2b.append(abs(f_p(x0) - bdq2(f, x0, h=hk)))
        o2c.append(abs(f_p(x0) - cdq2(f, x0, h=hk)))
        o4c.append(abs(f_p(x0) - cdq4(f, x0, h=hk)))

    plt.ion()
    plt.loglog(h, o1f, '-o', label="Order 1 Forward")
    plt.loglog(h, o2f, '-o', label="Order 2 Forward")
    plt.loglog(h, o1b, '-o', label="Order 1 Backward")
    plt.loglog(h, o2b, '-o', label="Order 2 Backward")
    plt.loglog(h, o2c, '-o', label="Order 2 Centered")
    plt.loglog(h, o4c, '-o', label="Order 4 Centered")
    plt.legend() 
    plt.xlabel("h")
    plt.ylabel("Absolute Error")
    plt.show()

# Problem 4
def prob4():
    """The radar stations A and B, separated by the distance 500m, track a
    plane C by recording the angles alpha and beta at one-second intervals.
    Your goal, back at air traffic control, is to determine the speed of the
    plane.

    Successive readings for alpha and beta at integer times t=7,8,...,14
    are stored in the file plane.npy. Each row in the array represents a
    different reading; the columns are the observation time t, the angle
    alpha (in degrees), and the angle beta (also in degrees), in that order.
    The Cartesian coordinates of the plane can be calculated from the angles
    alpha and beta as follows.

    x(alpha, beta) = a tan(beta) / (tan(beta) - tan(alpha))
    y(alpha, beta) = (a tan(beta) tan(alpha)) / (tan(beta) - tan(alpha))

    Load the data, convert alpha and beta to radians, then compute the
    coordinates x(t) and y(t) at each given t. Approximate x'(t) and y'(t)
    using a forward difference quotient for t=7, a backward difference
    quotient for t=14, and a centered difference quotient for t=8,9,...,13.
    Return the values of the speed at each t.
    """
    data = np.load('plane.npy')
    a = 500
    t = data[::,0]
    alpha = np.deg2rad(data[::,1])
    beta = np.deg2rad(data[::,2])
    x = a * np.tan(beta) / (np.tan(beta) - np.tan(alpha))
    y = a * np.tan(beta) * np.tan(alpha) / (np.tan(beta) - np.tan(alpha))
    x_p = 1/2 * (x[2:] - x[:-2])
    y_p = 1/2 * (y[2:] - y[:-2])
    return np.sqrt(x_p**2 + y_p**2)

# Problem 5
def jacobian_cdq2(f, x, h=1e-5):
    """Approximate the Jacobian matrix of f:R^n->R^m at x using the second
    order centered difference quotient.

    Parameters:
        f (function): the multidimensional function to differentiate.
            Accepts a NumPy (n,) ndarray and returns an (m,) ndarray.
            For example, f(x,y) = [x+y, xy**2] could be implemented as follows.
            >>> f = lambda x: np.array([x[0] + x[1], x[0] * x[1]**2])
        x ((n,) ndarray): the point in R^n at which to compute the Jacobian.
        h (float): the step size in the finite difference quotient.

    Returns:
        ((m,n) ndarray) the Jacobian matrix of f at x.
    """
    n = len(x)
    iden = np.eye(n)
    J = np.array([(f(x + h*iden[j])-f(x - h*iden[j]))/(2*h) for j in range(n)])
    return J.T

# Problem 6
def cheb_poly(x, n):
    """Compute the nth Chebyshev polynomial at x.

    Parameters:
        x (autograd.ndarray): the points to evaluate T_n(x) at.
        n (int): The degree of the polynomial.
    """
    if n == 0:
        return anp.ones_like(x)

    elif n == 1:
        return x

    else:
        return 2*x*cheb_poly(x, n-1) - cheb_poly(x, n-2)


def prob6():
    """Use Autograd and cheb_poly() to create a function for the derivative
    of the Chebyshev polynomials, and use that function to plot the derivatives
    over the domain [-1,1] for n=0,1,2,3,4.
    """
    domain = np.linspace(-1,1,100)
    T_p = elementwise_grad(cheb_poly)
    plt.ion()
    
    plt.plot(domain, T_p(domain, 0), label="n = 0")
    plt.plot(domain, T_p(domain, 1), label="n = 1")
    plt.plot(domain, T_p(domain, 2), label="n = 2")
    plt.plot(domain, T_p(domain, 3), label="n = 3")
    plt.plot(domain, T_p(domain, 4), label="n = 4")
    plt.legend()
    plt.show()

# Problem 7
def prob7(N=200):
    """Let f(x) = (sin(x) + 1)^sin(cos(x)). Perform the following experiment N
    times:

        1. Choose a random value x0.
        2. Use prob1() to calculate the “exact” value of f′(x0). Time how long
            the entire process takes, including calling prob1() (each
            iteration).
        3. Time how long it takes to get an approximation of f'(x0) using
            cdq4(). Record the absolute error of the approximation.
        4. Time how long it takes to get an approximation of f'(x0) using
            Autograd (calling grad() every time). Record the absolute error of
            the approximation.

    Plot the computation times versus the absolute errors on a log-log plot
    with different colors for SymPy, the difference quotient, and Autograd.
    For SymPy, assume an absolute error of 1e-18.
    """
    f = lambda x: (anp.sin(x) + 1)**(anp.sin(anp.cos(x)))
    T1 = []
    T2 = []
    T3 = []
    E2 = []
    E3 = []
    for i in range(N):
        x = np.random.random() * 10
        t1 = time.time()
        p_sy = prob1()(x)
        t2 = time.time()
        e2 = abs(p_sy - cdq4(f, x))
        t3 = time.time()
        e3 = abs(p_sy - grad(f)(x))
        t4 = time.time()

        T1.append(t2-t1)
        T2.append(t3-t2)
        T3.append(t4-t3)
        E2.append(e2)
        E3.append(e3)

    E1 = np.ones_like(T1) * 1e-18
    plt.ion()
    plt.scatter(T1, E1, alpha=0.3, label="Sympy")
    plt.scatter(T2, E2, alpha=0.3, label="Difference Quotients")
    plt.scatter(T3, E3, alpha=0.3, label="Autograd")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlim(2e-5, 5e-2)
    plt.ylim(5e-19, 1e-10)
    plt.legend()
    plt.show()

def test1():
    x = sy.symbols('x')
    f = (sy.sin(x) + 1)**(sy.sin(sy.cos(x)))
    f = sy.lambdify(x, f, "numpy")
    fp = prob1()
    domain = np.linspace(-np.pi, np.pi, 200)
    plt.ion()
    plt.plot(domain, f(domain))
    plt.plot(domain, fp(domain))
    plt.scatter(domain, cdq4(f, domain), c='r', s=10, alpha=0.5)
    ax = plt.gca()
    ax.spines["bottom"].set_position("zero")
    plt.show()


