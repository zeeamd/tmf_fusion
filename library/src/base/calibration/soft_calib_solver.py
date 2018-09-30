# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 09:39:23 2018

@author: tued7001
"""
import logging
from time import time

import numpy as np
import pandas as pd

from .softcalibrator import SoftCalibrator


def sparse_pivot_table(values, rows, columns, nrows, ncols):
    
    from scipy.sparse import csr_matrix
    
    A = csr_matrix((values, (rows, columns)),
                         shape=( nrows, ncols ) )

    return A

#this reads in the softcalibration data
#this probably belongs in the read files module
def readSoftCalibrationProblem( criteria_df, units_df, data_df ):
    
    data_df = data_df.rename( columns = { col : col.split('(')[0] for col in data_df.columns} )
    units_df = units_df.rename( columns = { col : col.split('(')[0] for col in units_df.columns} )
    criteria_df = criteria_df.rename( columns = { col : col.split('(')[0] for col in criteria_df.columns} )
    
    units_df['u_seq'] = range(units_df.shape[0])
    criteria_df['c_seq'] = range(criteria_df.shape[0])
    
    combined_df = pd.merge( data_df, units_df, on = 'UNIT')
        
    combined_df = pd.merge( combined_df, criteria_df, on = 'CRIT')
    
    combined_df = pd.merge( combined_df, combined_df.groupby('CRIT')['A'].sum().reset_index().rename( columns = {'A' : 'y0'} ), on = 'CRIT' )
    
    combined_df['se0'] = ( combined_df['y0'] - combined_df['YT'] ) / combined_df['YC']
    
    return combined_df.dropna(), units_df, criteria_df

def executePythonSoftCalibration( estimatedH, criteria_df, units_df, data_df ):

    soft_cal_solver_logger = logging.getLogger(__name__)

    tol = 1e-14
    
    soft_cal_solver_logger.info('Reading Calibration Problem')
    
    combined_df, units_df, criteria_df = readSoftCalibrationProblem( criteria_df, units_df, data_df )
    
    soft_cal_solver_logger.info('Creating data matrix A')
    
    idx_i = combined_df['c_seq'].values.astype(np.int32)
    idx_j = combined_df['u_seq'].values.astype(np.int32)
    
    nrows = int( criteria_df['c_seq'].max() ) + 1
    ncols = int( units_df['u_seq'].max() ) + 1
    
    vals = (combined_df['A']*combined_df['W0'] / combined_df['YC']).values
    
    A = sparse_pivot_table(vals, idx_i, idx_j, nrows, ncols)
    
    soft_cal_solver_logger.info('Setting Up Translation, Scalings, and Bounds')
    
  
    #setting up various parameters needed for our solver
    
    y = ( criteria_df['YT'].values / ( criteria_df['YC'].values + tol ) ).T
    
    w_size = ['W0', 'SIZE']
    
    sample_weights = 2.0 * ( criteria_df['YV'].values / ( criteria_df['YV'].sum() + tol ) ).T
    w_radi = 2.0 * ( units_df[w_size].prod(axis = 1).values / 
         ( units_df[w_size].prod(axis = 1).sum() + tol ) ).T
    
    w_bounds =  np.column_stack( [units_df['WL'].values / ( units_df['W0'].values + tol ),
                                  units_df['WU'].values / ( units_df['W0'].values + tol ) ] )
    
    y_bounds =  np.column_stack( [criteria_df['YL'].values / ( criteria_df['YC'].values+ tol ),
                                  criteria_df['YU'].values / ( criteria_df['YC'].values + tol )] )
    
    soft_cal_solver_logger.info('At solver')
  
    start = time()
    
    model = SoftCalibrator(epsilon = estimatedH, solver = 'cvxpy')
    
    model.fit(A, y, w_bounds, y_bounds, w_radi, sample_weights)
    
    soft_cal_solver_logger.info('Find a solution in: {}'.format( time() - start ) )
    
    soft_cal_solver_logger.info('Solver status is {}'.format( model.solve_status ) )
    soft_cal_solver_logger.info('Calibration loss is {}'.format( model.score( A, y, sample_weights ) ) )

    w_sol = model.getCoefficients()

    units_df['xf'] = units_df['W0'] * w_sol
    units_df['adj'] = w_sol

    units_df = units_df.rename( columns = {'UNIT' : 'unitLabel', 'W0' : 'x0'})
    
    return units_df[['unitLabel', 'x0', 'xf', 'adj']]

