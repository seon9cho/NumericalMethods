# drazin.py
"""Volume 1: The Drazin Inverse.
<Name>
<Class>
<Date>
"""

import numpy as np
from scipy import linalg as la
from numpy.linalg import matrix_power


# Helper function for problems 1 and 2.
def index(A, tol=1e-5):
    """Compute the index of the matrix A.

    Parameters:
        A ((n,n) ndarray): An nxn matrix.

    Returns:
        k (int): The index of A.
    """

    # test for non-singularity
    if not np.allclose(la.det(A),0):
        return 0

    n = len(A)
    k = 1
    Ak = A.copy()
    while k <= n:
        r1 = np.linalg.matrix_rank(Ak)
        r2 = np.linalg.matrix_rank(np.dot(A,Ak))
        if r1 == r2:
            return k
        Ak = np.dot(A,Ak)
        k += 1

    return k


# Problem 1
def is_drazin(A, Ad, k):
    """Verify that a matrix Ad is the Drazin inverse of A.

    Parameters:
        A ((n,n) ndarray): An nxn matrix.
        Ad ((n,n) ndarray): A candidate for the Drazin inverse of A.
        k (int): The index of A.

    Returns:
        (bool) True of Ad is the Drazin inverse of A, False otherwise.
    """
    return np.allclose(A@Ad, Ad@A) and \
           np.allclose(matrix_power(A, k+1)@Ad, matrix_power(A, k)) and \
           np.allclose(Ad@A@Ad, Ad)

# Problem 2
def drazin_inverse(A, tol=1e-4):
    """Compute the Drazin inverse of A.

    Parameters:
        A ((n,n) ndarray): An nxn matrix.

    Returns:
       ((n,n) ndarray) The Drazin inverse of A.
    """
    # Algorithm 15.1
    n,n = A.shape
    # Sort the Schur decomposition
    f1 = lambda x: abs(x) > tol
    Q1, S, k1 = la.schur(A, sort=f1)
    f2 = lambda x: abs(x) <= tol
    Q2, T, k2 = la.schur(A, sort=f2)
    # Concatenate part of S and T column-wise
    U = np.column_stack([S[:,:k1], T[:,:n-k1]])
    U_inv = la.inv(U)
    V = U_inv@A@U
    Z = np.zeros((n,n), dtype=float)
    if k1 != 0:
        M_inv = la.inv(V[:k1,:k1])
        Z[:k1,:k1] = M_inv
    return U@Z@U_inv

# Problem 3
def effective_resistance(A):
    """Compute the effective resistance for each node in a graph.

    Parameters:
        A ((n,n) ndarray): The adjacency matrix of an undirected graph.

    Returns:
        ((n,n) ndarray) The matrix where the ijth entry is the effective
        resistance from node i to node j.
    """
    n,n = A.shape
    # Compute the degree matrix
    D = np.diag(np.sum(A, axis=1))
    # Compute the Laplacian
    L = D - A
    I = np.eye(n)
    R = np.zeros((n,n))
    for j in range(n):
        Lj = L.copy()
        # Replace the jth column of L with the jth column of an identity matrix
        Lj[:,j] = I[:,j]
        # Equation 15.4
        Ld = drazin_inverse(Lj)
        R[:,j] = np.diagonal(Ld)
        R[j,j] = 0
    return R

# Problems 4 and 5
class LinkPredictor:
    """Predict links between nodes of a network."""

    def __init__(self, filename='social_network.csv'):
        """Create the effective resistance matrix by constructing
        an adjacency matrix.

        Parameters:
            filename (str): The name of a file containing graph data.
        """
        names = set()
        connections = list()
        # Read the file
        with open(filename, 'r') as myfile:
            for line in myfile.readlines():
                con = line.strip().split(',')
                connections.append(con)
                names.add(con[0])
                names.add(con[1])
        self.names = sorted(list(names))
        n = len(self.names)
        self.n = n
        # Compute the adjacency matrix
        A = np.zeros((n,n))
        for con in connections:
            i = self.names.index(con[0])
            j = self.names.index(con[1])
            A[i,j], A[j,i] = 1,1
        self.adj = A
        # Compute the effective resistance matrix
        self.erm = effective_resistance(A)

    def predict_link(self, node=None):
        """Predict the next link, either for the whole graph or for a
        particular node.

        Parameters:
            node (str): The name of a node in the network.

        Returns:
            node1, node2 (str): The names of the next nodes to be linked.
                Returned if node is None.
            node1 (str): The name of the next node to be linked to 'node'.
                Returned if node is not None.

        Raises:
            ValueError: If node is not in the graph.
        """
        # Replace the values of the resistance matrix with 0s where
        # connections already exist
        erm = self.erm * (np.ones((self.n,self.n)) - self.adj)
        if node == None:
            minval = np.min(erm[erm>0])
            loc = np.where(erm==minval)
            return self.names[loc[0][0]], self.names[loc[1][0]]
        elif node in self.names:
            i = self.names.index(node)
            minval = np.min(erm[i][erm[i]>0])
            loc = np.where(erm==minval)
            if self.names[loc[1][0]] == node:
                return self.names[loc[0][0]]
            else:
                return self.names[loc[1][0]]
        else:
            raise ValueError("Node is not in the network")

    def add_link(self, node1, node2):
        """Add a link to the graph between node 1 and node 2 by updating the
        adjacency matrix and the effective resistance matrix.

        Parameters:
            node1 (str): The name of a node in the network.
            node2 (str): The name of a node in the network.

        Raises:
            ValueError: If either node1 or node2 is not in the graph.
        """
        if node1 not in self.names or node2 not in self.names:
            raise ValueError("Node is not in the network")
        else:
            i = self.names.index(node1)
            j = self.names.index(node2)
            self.adj[i,j] += 1
            self.adj[j,i] += 1
            self.erm = effective_resistance(self.adj)

A = np.array([[1,3,0,0],[0,1,3,0],[0,0,1,3],[0,0,0,0]])
Ad = np.array([[1,-3,9,81],[0,1,-3,-18],[0,0,1,3],[0,0,0,0]])
B = np.array([[1,1,3],[5,2,6],[-2,-1,-3]])
Bd = np.zeros((3,3))
print(is_drazin(A, Ad, 1))
print(is_drazin(B, Bd, 3))
C = np.random.random((5,5))
Cd = drazin_inverse(C)
k = index(C)
print(is_drazin(C, Cd, k))
G1 = np.array([[0,1,0,0],[1,0,1,0],[0,1,0,1],[0,0,1,0]])
print(effective_resistance(G1))
G2 = np.array([[0,1],[1,0]])
print(effective_resistance(G2))
G3 = np.array([[0,1,1],[1,0,1],[1,1,0]])
print(effective_resistance(G3))
G4 = np.array([[0,3],[3,0]])
print(effective_resistance(G4))
G5 = np.array([[0,2],[2,0]])
print(effective_resistance(G5))
G6 = np.array([[0,4],[4,0]])
print(effective_resistance(G6))
LP = LinkPredictor()
print(LP.predict_link())
print(LP.predict_link(node="Melanie"))
print(LP.predict_link(node="Alan"))
LP.add_link("Alan", "Sonia")
print(LP.predict_link(node="Alan"))
LP.add_link("Alan", "Piers")
print(LP.predict_link(node="Alan"))
