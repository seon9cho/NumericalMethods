# solutions.py
"""Volume 2: Polynomial Interpolation.
<Name>
<Class>
<Date>
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as la
from scipy.interpolate import BarycentricInterpolator

# Problem 1
def lagrange(xint, yint, points):
    """Find an interpolating polynomial of lowest degree through the points
    (xint, yint) using the Lagrange method and evaluate that polynomial at
    the specified points.

    Parameters:
        xint ((n,) ndarray): x values to be interpolated.
        yint ((n,) ndarray): y values to be interpolated.
        points((m,) ndarray): x values at which to evaluate the polynomial.

    Returns:
        ((m,) ndarray): The value of the polynomial at the specified points.
    """
    l_basis = []
    for i, x in enumerate(xint):
        denom = np.prod([x - xk for xk in xint if xk != x])
        numer = 1
        for xk in xint:
            if xk != x:
                numer *= (points - xk)
        l_basis.append(numer/denom)
    p = sum([y*l_basis[i] for i,y in enumerate(yint)])

    return p

# Problems 2 and 3
class Barycentric:
    """Class for performing Barycentric Lagrange interpolation.

    Attributes:
        w ((n,) ndarray): Array of Barycentric weights.
        n (int): Number of interpolation points.
        x ((n,) ndarray): x values of interpolating points.
        y ((n,) ndarray): y values of interpolating points.
    """

    def __init__(self, xint, yint):
        """Calculate the Barycentric weights using initial interpolating points.

        Parameters:
            xint ((n,) ndarray): x values of interpolating points.
            yint ((n,) ndarray): y values of interpolating points.
        """
        self.xint = xint
        self.yint = yint
        self.n = len(xint)
        w = np.ones(self.n)
        self.C = (np.max(xint) - np.min(xint)) / 4
        shuffle = np.random.permutation(self.n - 1)
        for j in range(self.n):
            temp = (xint[j] - np.delete(xint,j)) / self.C
            temp = temp[shuffle]
            w[j] /= np.product(temp)
        self.weights = w

    def __call__(self, points):
        """Using the calcuated Barycentric weights, evaluate the interpolating polynomial
        at points.

        Parameters:
            points ((m,) ndarray): Array of points at which to evaluate the polynomial.

        Returns:
            ((m,) ndarray): Array of values where the polynomial has been computed.
        """
        denom = sum([self.weights[i]/(points - self.xint[i]) for i in range(self.n)])
        numer = sum([self.yint[i]*self.weights[i]/(points - self.xint[i]) for i in range(self.n)])
        return numer/denom

    # Problem 3
    def add_weights(self, xint, yint):
        """Update the existing Barycentric weights using newly given interpolating points
        and create new weights equal to the number of new points.

        Parameters:
            xint ((m,) ndarray): x values of new interpolating points.
            yint ((m,) ndarray): y values of new interpolating points.
        """
        for x in xint:
            self.n += 1
            shuffle = np.random.permutation(self.n - 1)
            temp = (x - self.xint) / self.C
            temp = temp[shuffle]
            w = 1 / np.product(temp)
            self.weights /= (self.xint - x)/self.C
            self.weights = np.append(self.weights, w)
            self.xint = np.append(self.xint, x)

        self.yint = np.append(self.yint, yint)


# Problem 4
def prob4():
    """For n = 2^2, 2^3, ..., 2^8, calculate the error of intepolating Runge's
    function on [-1,1] with n points using SciPy's BarycentricInterpolator
    class, once with equally spaced points and once with the Chebyshev
    extremal points. Plot the absolute error of the interpolation with each
    method on a log-log plot.
    """
    domain = np.linspace(-1, 1, 400)
    n_array = [2**i for i in range(2,9)]
    f = lambda x: 1 / (1 + 25*x**2)
    plt.ion()
    poly_error = []
    cheby_error = []
    for n in n_array:
        x = np.linspace(-1, 1, n)
        poly = BarycentricInterpolator(x)
        poly.set_yi(f(x))
        poly_error.append(la.norm(f(domain) - poly(domain), ord=np.inf))
        y = np.array([(1/2)*(2*np.cos(j*np.pi/n)) for j in range(n+1)])
        cheby = BarycentricInterpolator(y)
        cheby.set_yi(f(y))
        cheby_error.append(la.norm(f(domain) - cheby(domain), ord=np.inf))
    plt.loglog(n_array, poly_error, label="equally spaced points", basex=2)
    plt.loglog(n_array, cheby_error, label="Chebyshev extremal points", basex=2)
    plt.legend()
    plt.show()


# Problem 5
def chebyshev_coeffs(f, n):
    """Obtain the Chebyshev coefficients of a polynomial that interpolates
    the function f at n points.

    Parameters:
        f (function): Function to be interpolated.
        n (int): Number of points at which to interpolate.

    Returns:
        coeffs ((n+1,) ndarray): Chebyshev coefficients for the interpolating polynomial.
    """
    extrema = np.cos((np.pi * np.arange(2*n))/n)
    samples = f(extrema)
    coeffs = np.real(np.fft.fft(samples))[:n+1]
    coeffs /= n 
    coeffs[0] /= 2
    coeffs[n] /= 2

    return coeffs

# Problem 6
def prob6(n):
    """Interpolate the air quality data found in airdata.npy using
    Barycentric Lagrange interpolation. Plot the original data and the
    interpolating polynomial.

    Parameters:
        n (int): Number of interpolating points to use.
    """
    data = np.load('airdata.npy')
    fx = lambda a, b, n: .5*(a+b + (b-a) * np.cos(np.arange(n+1) * np.pi/n))
    a, b = 0, 366 - 1/24
    domain = np.linspace(0, b, 8784)
    pts = fx(a, b, n)
    temp = np.abs(pts - domain.reshape(8784, 1))
    temp2 = np.argmin(temp, axis=0)
    poly = Barycentric(domain[temp2], data[temp2])

    plt.ion()
    plt.subplot(121)
    plt.plot(domain, data)
    plt.title("Data")
    plt.subplot(122)
    plt.plot(domain, poly(domain))
    plt.title("Interpolation")
    plt.show()

f = lambda x: 1/(1+25*x**2)

def test2():
    x = np.linspace(-1,1,15)
    y = f(x)
    dom = np.linspace(-1,1,100)
    B = Barycentric(x[::2], y[::2])

    plt.plot(dom, f(dom))
    plt.plot(dom, B(dom))
    plt.ylim((-0.3, 1.2))
    plt.show()

    B.add_weights(x[1::2], y[1::2])

    plt.plot(dom, f(dom))
    plt.plot(dom, B(dom))
    plt.show()
