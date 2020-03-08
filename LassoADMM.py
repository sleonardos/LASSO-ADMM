import numpy as np
from utils import soft_thres, generate_data, ADMMLasso


# data generation
A, b, x = generate_data()

# initialize an instance of the ADDMLasso class
admm = ADMMLasso(A, b)

# solve the optimization problem
cache = admm.solve()

# recover xhat from cache
xhat = cache[-1][2]

# plot primal and dual residual to verify convergence
admm.plot_residuals()
