# montecarlo_integration.py
"""Volume 1: Monte Carlo Integration.
<Name>
<Class>
<Date>
"""
import numpy as np
from scipy import linalg as la
from scipy import stats
import matplotlib.pyplot as plt
import sys

# Problem 1
def ball_volume(n, N=10000):
    """Estimate the volume of the n-dimensional unit ball.

    Parameters:
        n (int): The dimension of the ball. n=2 corresponds to the unit circle,
            n=3 corresponds to the unit sphere, and so on.
        N (int): The number of random points to sample.

    Returns:
        (float): An estimate for the volume of the n-dimensional unit ball.
    """
    # Volume of the n-dimentional hypercube with length 2
    V_s = 2**n
    # Sample points
    points = np.random.uniform(-1, 1, (n,N))
    # Determine the number of points within the circle
    lengths = la.norm(points, axis=0)
    num_within = np.count_nonzero(lengths < 1)
    # Estimate the circle's area
    return V_s * (num_within / N)

# Problem 2
def mc_integrate1d(f, a, b, N=10000):
    """Approximate the integral of f on the interval [a,b].

    Parameters:
        f (function): the function to integrate. Accepts and returns scalars.
        a (float): the lower bound of interval of integration.
        b (float): the lower bound of interval of integration.
        N (int): The number of random points to sample.

    Returns:
        (float): An approximation of the integral of f over [a,b].

    Example:
        >>> f = lambda x: x**2
        >>> mc_integrate1d(f, -4, 2)    # Integrate from -4 to 2.
        23.734810301138324              # The true value is 24.
    """
    # Length of the interval
    V_o = (b-a)
    # Sample points 
    points = np.random.uniform(a, b, N)
    # Estimate the integral (11.2)
    return V_o * sum(f(x) for x in points) / N

# Problem 3
def mc_integrate(f, mins, maxs, N=10000):
    """Approximate the integral of f over the box defined by mins and maxs.

    Parameters:
        f (function): The function to integrate. Accepts and returns
            1-D NumPy arrays of length n.
        mins (list): the lower bounds of integration.
        maxs (list): the upper bounds of integration.
        N (int): The number of random points to sample.

    Returns:
        (float): An approximation of the integral of f over the domain.

    Example:
        # Define f(x,y) = 3x - 4y + y^2. Inputs are grouped into an array.
        >>> f = lambda x: 3*x[0] - 4*x[1] + x[1]**2

        # Integrate over the box [1,3]x[-2,1].
        >>> mc_integrate(f, [1, -2], [3, 1])
        53.562651072181225              # The true value is 54.
    """
    delta = np.array(maxs) - np.array(mins)
    # Volume of the region (11.3)
    V_o = np.prod(delta)
    # Number of dimensions
    n = len(mins)
    # Sample points from [0,1]
    pts = np.random.uniform(0, 1, (N, n))
    # Scale the points (11.4)
    pts *= delta
    pts += mins
    # Estimate the integral (11.2)
    return V_o * sum(f(x) for x in pts) / N

# Problem 4
def prob4():
    """Let n=4 and Omega = [-3/2,3/4]x[0,1]x[0,1/2]x[0,1].
    - Define the joint distribution f of n standard normal random variables.
    - Use SciPy to integrate f over Omega.
    - Get 20 integer values of N that are roughly logarithmically spaced from
        10**1 to 10**5. For each value of N, use mc_integrate() to compute 25
        estimates of the integral of f over Omega with N samples, and average
        the estimates together. Compute the relative error of each average.
    - Plot the relative error against the sample size N on a log-log scale.
        Also plot the line 1 / sqrt(N) for comparison.
    """
    # Define the function and its mins/maxs
    f = lambda x: (1/(2*np.pi)**(len(x)/2))*np.exp(-np.dot(x,x)/2)
    mins = [-3/2, 0, 0, 0]
    maxs = [3/4, 1, 1/2, 1]
    # Calculate the exact value of f using scipy
    means, cov = np.zeros(4), np.eye(4)
    F = stats.mvn.mvnun(mins, maxs, means, cov)[0]
    # Record the relative error for varying sample sizes
    domain = np.logspace(1, 5, 20)
    error = []
    for N in domain:
        Fm = sum(mc_integrate(f, mins, maxs, N=int(N)) for i in range(25))/25
        error.append(abs((F-Fm)/F))
    # Plot the result
    plt.loglog(domain, error, label="Relative Error")
    plt.loglog(domain, 1/np.sqrt(domain), label=r'1/sqrt(N)')
    plt.legend()
    plt.show()

def main(prob):
    if prob == "prob1":
        print(ball_volume(2))
        print(ball_volume(3))
        print(ball_volume(4))

    if prob == "prob2":
        f1 = lambda x: x**2
        f2 = lambda x: np.sin(x)
        f3 = lambda x: 1/x
        f4 = lambda x: abs(np.sin(10*x)*np.cos(10*x) + np.sqrt(x)*np.sin(3*x))
        print(mc_integrate1d(f1, -4, 2))
        print(mc_integrate1d(f2, -2*np.pi, 2*np.pi))
        print(mc_integrate1d(f3, 1, 10))
        print(mc_integrate1d(f4, 1, 5))

    if prob == "prob3":
        f1 = lambda x: x[0]**2 + x[1]**2
        f2 = lambda x: 3*x[0] - 4*x[1] + x[1]**2
        f3 = lambda x: x[0] + x[1] - x[3]*(x[2]**2)
        print(mc_integrate(f1, [0,0], [1,1]))
        print(mc_integrate(f2, [1,-2], [3,1]))
        print(mc_integrate(f3, [-1,-2,-3,-4], [1,2,3,4]))

    if prob == "prob4":
        prob4()

if __name__ == "__main__":
    plt.ion()
    main(sys.argv[1])
