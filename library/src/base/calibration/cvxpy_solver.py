# -*- coding: utf-8 -*-
"""
Created on Wed Mar 28 10:27:12 2018

@author: tued7001
"""

import numpy as np

from cvxpy import square, huber, Constant, Variable, Problem, Minimize, norm, multiply

def solver(A, y, huber_param, w_bounds, y_bounds, w_radi, y_sample_weights, theta, gamma = 0.01):
    '''
    Presently, we perform Huber Regression on our data matrix, with our own
    specialized way of regularizing the weights, the results of the model fitting.
    
    If we currently do not have a huber parameter to use, we will let the method
    to optimize for one
    '''
     
    w_radi_sqrt = np.sqrt( w_radi )

    #bounds on weights
    wl = Constant( w_bounds[:, 0] )
    wu = Constant( w_bounds[:, 1] )
    
    #bounds on targets
    yl = Constant( y_bounds[:, 0] )
    yu = Constant( y_bounds[:, 1] )
    
    n = A.shape[0]
    m = A.shape[1]
    
    #weight variable
    w_var = Variable(m)
    
    y_var = A.dot(w_var)
    
    deviance = y_sample_weights.T * huber(y_var - y, M = huber_param)
    
    divergence = square( norm( multiply( w_radi_sqrt, ( w_var - 1.0 ) )  ) )
    
    constraints = [wl <= w_var, w_var <= wu, y_var <= yu, yl <= y_var, divergence <= ( theta ** 2.0 )]

    cost = (1.0 - gamma)*deviance + gamma*divergence 
    
    prob = Problem(Minimize(cost), constraints)

    prob.solve(solver = 'ECOS', abstol = 1e-4, reltol = 1e-3, feastol = 1e-4, max_iters = 1000, verbose = False )
        
    w_sol = np.array( w_var.value )

    #this still needs to be done since solver only satisfies constraints 
    #within a tolerance of 1e-4
    w_sol = np.clip(w_sol, wl.value, wu.value).ravel()
    
    w_sol = np.squeeze(np.asarray(w_sol))

    return w_sol, prob.status
    
    
