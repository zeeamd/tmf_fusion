# -*- coding: utf-8 -*-
"""
Created on Mon Oct 16 10:32:53 2017
Author: Elise Braun
Co-Author: Edward Turner
"""

import numpy as np
import pandas as pd
from statsmodels.robust.scale import mad


#this assumes the x and weights variables are arrays with shape (n, 1)
#where n is the number of samples
def varianceWeightedMean(x, weights):
    '''
    Args:
        x: input sample values, array-like
        weights: sample weights, array-like
    Returns:
        variance of weighted mean: float
    '''
    #the sample size
    n = weights.shape[0]
    
    if n == 1:
        return x, 0.0
    else:
        #normalized weights    
        w_norm = weights / np.sum(weights)
        
        #the weighted average
        x_bar = x.T.dot( w_norm )
    
        #the weighted residual vector  
        res = w_norm*(x - x_bar)
    
        #the 2-norm of the weighted residual vector
        x_se = res.T.dot(res)
    
        #a scaling to coincide with the accepted definition
        x_se *= n / (n - 1.0)
    
        #the float value stored in the 
    
        return x_bar, x_se

#helps to quickly calculate the wegithed version of the RSE 
def weightedRSE(x, weights):
    
    x_bar, x_var = varianceWeightedMean(x, weights)
    
    return ( x_var ** 0.5 ) / max(x_bar, 1e-7)

#Calculate standard error for reach and impressions
#The relative standard error is the ratio of the standard deviation and
#the mean.  We have the option of using weights if we want to as well
def std_err(type_val, name, data_in, data_out, sqrt_effective_sample, use_weights = False): 
    
    calc = pd.DataFrame( columns = ['rse', 'targetType'] )
    
    '''        
    Suppose we are looking at a binary sequence. Since we are looking at a 
    binary squence, we can relate this to a binomial distribution of 0 and 1.  
    This means the second moment and the first moment are the same, which 
    implies the RSE is ( 1/x - 1 ) ** 0.5, where x is the mean of the sequence.
    
    '''
    
    if name is 'population':
        
        #create better formula.  We need only unique respondentid.
        if use_weights:
            w_hat = data_in.loc[:, 'weight'].sum()
            
            calc.loc[:, 'rse'] = data_in.groupby('var')['weight']\
            .apply(lambda x : ( ( w_hat / x.sum() ) - 1.0 ) ** 0.5 )
            
        else:
            n = data_in.loc[:, 'respondentid'].nunique()
        
            calc.loc[:, 'rse'] = data_in.groupby('cc')['respondentid']\
            .apply( lambda x : ( ( n / x.nunique() ) - 1.0 ) ** 0.5 )
        
        calc.loc[:, 'rse'] = calc.loc[:, 'rse'] / sqrt_effective_sample
        calc.index = ['cc::' + str(idx) for idx in calc.index]
        calc.index.name = 'var'
                 
    else:
       if use_weights:
           calc.loc[:, 'rse'] = data_in.groupby('var')[[type_val, 'weight']]\
           .apply(lambda x : weightedRSE(x.loc[:, type_val], x.loc[:, 'weight']) )
           
       else:
            calc.loc[:, 'rse'] = data_in.groupby('var')[type_val].apply(lambda x : 
                x.std() / max(x.mean(), 1e-7) )
            
       calc.loc[:, 'rse'] = calc.loc[:, 'rse'] / sqrt_effective_sample
                            
    calc.loc[:, 'targetType'] = name
    
    return pd.concat( [data_out, calc], axis = 0 )

#detects if cc or population is in criteria
def is_pop_cc( crit_val ):
    return ('cc' in crit_val) and ('population' in crit_val)

#this does the huber parameter estimation
def huberParameterEsitmation(impressions, criteria_df):
    imp_indices = impressions.loc[:, 'val'] > 0
    
    impressions = impressions.loc[imp_indices, :]
     
    #sum the weights by var value
    reach_data = impressions.groupby('var')['weight'].sum()\
    .reset_index().values
    
    #sum the weighted val by var value
    imp_data = impressions.groupby('var')[['val','weight']]\
    .apply(lambda x : x.prod(axis = 1).sum() ).reset_index().values
    
    reach_df = pd.DataFrame( data = reach_data, columns = ['criteria', 'y0'] )
    imp_df = pd.DataFrame( data = imp_data, columns = ['criteria', 'y0'] )
   
    #sets the criteria
    reach_df.loc[:, 'criteria'] = reach_df.loc[:, 'criteria'].apply(lambda x : 'reach::' + x )
    imp_df.loc[:, 'criteria'] = imp_df.loc[:, 'criteria'].apply(lambda x : 'imp::' + x)
    
    #combines results
    data = pd.concat( [reach_df, imp_df], axis = 0)
    
    #set the index of the criteria
    data.set_index('criteria', inplace = True)
    criteria_df.set_index('CRIT', inplace = True)
    
    criteria_df.dropna( inplace = True )
    
    #joins the data
    estimatedH_df = criteria_df.join(data, how = 'inner').dropna()
         
   #gets the vector we want to estimate
    vec = ( estimatedH_df.loc[:, 'YT'] - estimatedH_df.loc[:, 'y0'] ) / estimatedH_df.loc[:, 'YC']

    #returns the median absolution deviation
    return 0.5*mad( vec.astype(float) )
