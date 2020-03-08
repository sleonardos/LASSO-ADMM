import numpy as np
import scipy.linalg
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.linalg import norm, inv


def soft_thres(x, kappa):

    # implements soft thresholding operation
    return np.maximum(x-kappa,0)-np.maximum(-x-kappa,0)


def generate_data(m = 1500, n = 5000):

    # generate synthetic data
        
    # generate matrix A, each element is normally distributed initially
    # then the columns are normalized to have unit norm
    A = np.random.randn(m,n)
    A = A/np.sqrt(np.sum(A**2, axis = 0))

    # actual vector x (sparse), only 1% nonzero entries
    xtrue = np.zeros((n,1))
    xtrue[np.random.choice(n,int(n/100))] = np.random.randn(int(n/100),1)

    # generate noise
    w = np.random.normal(0,0.001,(m,1))

    # generate b
    b = np.dot(A,xtrue) + w

    return A, b, xtrue


class ADMMLasso():

    # class that implements the ADMM algorithm for the Lasso problem
    
    def __init__(self, A, b, rho = 1, maxIter = 500, eps_abs = 0.0001, eps_rel = 0.0001):

        # paramter rho of ADMM
        self.rho = rho

        # maximum number of iterations
        self.maxIter = maxIter

        # initialize tolerance paramteres for stopping criteria
        self.eps_abs = eps_abs
        self.eps_rel = eps_rel

        # init cache for storing results
        self.cache = []

        # initialize regularization parameter
        self.l = 0.1*norm(A.T @ b, np.inf)

        # data of problem and their dimensions
        self.A, self.b = A, b
        self.m, self.n = A.shape

        # some precomputations
        self.C    = inv(np.dot(self.A.T, self.A) + self.rho*np.eye(self.n)) 
        self.Atb  = np.dot(self.A.T, self.b)
        self.CAtb = np.dot(self.C, self.Atb)


    def solve(self):

        # initialization of primal and dual variables
        x = 0.1*np.random.randn(self.n,1) 
        z = np.zeros((self.n,1))
        u = np.zeros((self.n,1))
        
        # Main loop
        for i in range(self.maxIter):

            # update primal and dual variables
            x = self.CAtb + self.rho*(np.dot(self.C,(z-u)))   
            z, zlast = soft_thres(x+u, self.l/self.rho), z
            u, ulast =  u+x-z, u

            # compute residuals
            r = norm(u-ulast)
            s = self.rho*norm((z-zlast))

            if i%5==0:
                print("Iteration number  %.0f, primal residual norm %.4f, dual residual norm %.4f" % (i, r, s))
            
            # store some information
            self.cache.append((r,s,x,z,u))
            
            # check for convergence
            eps_pri  = np.sqrt(self.n)*self.eps_abs + self.eps_rel*max(norm(x),norm(z))
            eps_dual = np.sqrt(self.n)*self.eps_abs + self.eps_rel*self.rho*norm(u)

            # check for termination
            if r < eps_pri and s < eps_dual:
                break
        
        return self.cache



    def plot_residuals(self):

        # Plot residuals over time
        
        if self.cache:
            
            plt.plot([c[0] for c in self.cache])
            plt.title("Primal Residual Norm vs Iteration Number")
            plt.yscale("log")
            plt.xlabel("Iteration Number")
            plt.ylabel("Norm of primal residual.")
            plt.show()

            plt.plot([c[1] for c in self.cache])
            plt.title("Dual residual norm")
            plt.yscale("log")
            plt.xlabel("Iteration Number")
            plt.ylabel("Norm of dual residual.")
            plt.show()
