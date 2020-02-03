import numpy as np
import scipy.linalg
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.linalg import norm, inv

def soft_thres(x: np.array, kappa: np.float):
    return np.maximum(x-kappa,0)-np.maximum(-x-kappa,0)   

# data generation
m, n = 1500, 5000
A = np.random.normal(0,1,(m,n))
A = A/np.sqrt(np.sum(A**2,0))
xtrue = np.zeros((n,))
ind = np.random.choice(n,100)
xtrue[ind] = np.random.normal(0,1,(100,))
b = A @ xtrue + np.random.normal(0,0.001,(m,))

# regularization parameter lambda
l_max = scipy.linalg.norm(A.T @ b, np.inf)
l = 0.1* l_max

# ADMM parameters
rho = 1
maxIter = 500
eps_abs, eps_rel = 0.0001, 0.01 

# Some precomputations
C    = inv(A.T @ A + rho*np.eye(n)) 
Atb  = A.T @ b
CAtb = C@ Atb

# initialization of primal and dual variables
x, z, u = np.random.normal(0,0.01,(n,)), np.zeros((n,)), np.zeros((n,))

# Preallocation for residuals
r, s = np.zeros((maxIter,)), np.zeros((maxIter,))

# Main loop
i = 0
while True:

    # update primal and dual variables
    x = CAtb + rho*(C@(z-u))   
    z, zlast = soft_thres(x+u, l/rho), z
    u, ulast =  u+x-z, u

    # compute residuals
    r[i] = norm(u-ulast)
    s[i] = rho*norm((z-zlast))

    # check for convergence
    eps_pri = np.sqrt(n)*eps_abs + eps_rel*max(norm(x),norm(z))
    eps_dual = np.sqrt(n)*eps_abs + eps_rel*rho*norm(u)    
    if r[i]<eps_pri and s[i]<eps_dual:
        break

    # check if we reached max iterations
    if i<maxIter:
        i+=1
    else:
        break


# Plot residuals over time
plt.plot(range(i+1),r[:i+1])
plt.title("Primal Residual Norm vs Iteration Number")
plt.yscale("log")
plt.xlabel("Iteration Number")
plt.ylabel("Norm of primal residual")
plt.show()

plt.plot(range(i+1),s[:i+1])
plt.title("Dual residual norm")
plt.yscale("log")
plt.xlabel("Iteration Number")
plt.ylabel("Norm of dual residual")
plt.show()


