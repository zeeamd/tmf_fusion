# -*- coding: utf-8 -*-
"""
Created on Wed Jan  3 11:24:27 2018

@author: tued7001
"""

import logging

import numpy as np
# data managment packages
import pandas as pd

# helper functions for creation methods
from .soft_calib_stats import std_err, is_pop_cc


#creates the Relative Square Error DataFrame
def createRSE(data, effective_sample, min_rse, max_rse):
    '''
    Argument:
        data - A dataset with population data
        effective_sample - Our scaling factor for our rse
        min_rse - Our minimum relative error accepted
        max_rse - Our maximum relative error accepted
    Return:
        rse_df - Our dataset with the RSE per categorial variable
    '''
    sqrt_effective_sample = effective_sample ** 0.5
    
    data['rch'] = (data['val'] > 0).astype(int)

    cal_data_logger = logging.getLogger(__name__)

    #Calculate standard error for population based on critical cells        
    cal_data_logger.info('Determining the Relative Standard Error for our population variables')

    #This is an improvement, and should be included in the process

    rse = std_err('member', 'population', data, pd.DataFrame(), sqrt_effective_sample)
    
    #Calculate standard error for reach and impressions     
    
    cal_data_logger.info('Determining the Relative Standard Error for our impression variables')
    rse = std_err('val', 'imp', data, rse, sqrt_effective_sample)
    
    cal_data_logger.info('Determining the Relative Standard Error for our reach variables')
    rse = std_err('rch', 'reach', data, rse, sqrt_effective_sample)
    
    rse.reset_index(inplace = True)
    
    rse.loc[:, 'rse'] = rse.loc[:, 'rse'].clip(min_rse, max_rse)
    rse.loc[:, 'rse'] = rse.loc[:, 'rse'].fillna(max_rse)
    
    data.drop('rch', axis = 1, inplace = True)
    
    return rse

#creates calibration dataframe
def createCriteria(targets, impressions, effective_sample, min_rse, max_rse):
    '''
    Arguments:
        impressions - Our impression data
        targets - Our target data
        effective_sample - Our scaling factor for our rse
        min_rse - Our minimum relative error accepted
        max_rse - Our maximum relative error accepted
    '''
    #'Creating Relative Squared Error of Impressions'
    
    rse = createRSE(impressions, effective_sample, min_rse, max_rse)

    cal_data_logger = logging.getLogger(__name__)

    #'Creating Calibration DataFrame'
    
    targets = pd.merge( targets, rse, how = 'left', on = ['var', 'targetType']).fillna(0.25)
    
    cal_data_logger.info("Determining our target reach and impressions per variable, along with sample weights")

    criteria_df = pd.DataFrame( 
            columns = ['criteria', 'YT', 'YL', 'YU', 'YC'] 
            )
    
    targets = targets.reset_index()
    targets.loc[:, 'criteria'] = targets.loc[:, 'targetType'] + '::' \
    + targets.loc[:, 'var']
    
    criteria_df.loc[:, 'criteria'] = targets.loc[:, 'criteria']
        
    is_pop_cc_vec = targets.loc[:, 'criteria'].apply(is_pop_cc).astype(int)
    
    #sets our bounds for our targets
    
    criteria_df.loc[:, 'YT'] = targets.loc[:, 'val']
    criteria_df.loc[:, 'YL'] = targets.loc[:, 'val']*(
            1.0 - 100.0 * (1 - is_pop_cc_vec) * targets.loc[:, 'rse']
            )
    criteria_df.loc[:, 'YL'] = criteria_df.loc[:, 'YL'].clip(lower = 0.0)
    criteria_df.loc[:, 'YU'] = targets.loc[:, 'val']*(
            1.0 + 100.0 *(1 - is_pop_cc_vec) * targets.loc[:, 'rse']
            )
    
    criteria_df.loc[:, 'YC'] = targets.loc[:, 'val']*targets.loc[:, 'rse']
   
    importance_groups = targets['importanceGroup'].unique()

    if len(importance_groups) == 1:

        const_ = targets.loc[:, 'importanceUnits'].tail(1).values
        const_ /= ( targets.loc[:, 'val'] ** 0.5 ).sum()

        criteria_df.loc[:, 'YV'] = ( targets.loc[:, 'val'] ** 0.5 ) * const_

    else:
        targets.loc[:, 'YV'] = 0.0

        df_targets = pd.DataFrame( columns = ['criteria', 'YV'])

        for import_group in importance_groups:

            indices = targets['importanceGroup'] == import_group
            df_group = targets.loc[indices, :].reset_index(drop = True)

            import_unit = df_group.loc[0, 'importanceUnits']
            vals_vec = df_group.loc[:, 'val'] ** 0.5
            sum_sqrt_vals = vals_vec.sum()

            df_group['YV'] = ( vals_vec.values * import_unit ) / sum_sqrt_vals

            df_targets = pd.concat([df_targets, df_group[['criteria', 'YV']]], ignore_index = True)

        #this joins all the necessary dataframes to help with the final calculation
        criteria_df = pd.merge( criteria_df, df_targets, on = 'criteria')

    criteria_df = criteria_df.rename( columns = {'criteria' : 'CRIT'} )    

    return criteria_df.loc[:, ['CRIT', 'YT', 'YL', 'YU', 'YC', 'YV']]

#this method re-weights the samples
def createUnits(impressions, loc_adj_fact, tot_adj_fact):
    '''
    Arguments:
        impressions - Our impression data
        targets - Our target data
    Return:
        units_df - Our bounds on our calibrated weights
    '''

    cal_data_logger = logging.getLogger(__name__)

    units_df = pd.DataFrame( columns = ['W0(DOUBLE)', 'SIZE(DOUBLE)'] )
    
    cc_res_group = impressions.groupby(['cc', 'respondentid'])
    
    units_df.loc[:, 'SIZE(DOUBLE)'] = cc_res_group['val'].apply( lambda x : 1 + np.sum( x > 0 ) )
    units_df.loc[:, 'W0(DOUBLE)'] = cc_res_group['weight'].max().values
    
    meanw = np.mean(impressions.loc[:, 'weight']) 
        
    units_df = units_df.reset_index()
    
    units_df = units_df.rename( columns = {'respondentid' : 'UNIT(IFACTOR)'})

    min_wt = 50
    max_wt = 50000
    
    min_w = impressions.loc[:, 'weight'].min()
    max_w = impressions.loc[:, 'weight'].max()
    

    if min_wt > min_w:
        cal_data_logger.info('Changing minimum weight from {} to {}'.format(min_wt, min_w) )

        min_wt = min_w
    
    if max_wt < max_w:
        cal_data_logger.info('Changing maximum weight from {} to {}'.format(max_wt, max_w))
        max_wt = max_w
    
    w_l_funct = lambda x: x if x < meanw / tot_adj_fact else max([x / loc_adj_fact, meanw / tot_adj_fact, min_wt])

    w_u_funct = lambda x: x if meanw * tot_adj_fact < x else min([x * loc_adj_fact, meanw * tot_adj_fact, max_wt])

    units_df['WL(DOUBLE)'] = units_df['W0(DOUBLE)'].apply(w_l_funct)
    units_df['WU(DOUBLE)'] = units_df['W0(DOUBLE)'].apply(w_u_funct)
        
    units_df = units_df.loc[:, ['UNIT(IFACTOR)', 'W0(DOUBLE)', 'WL(DOUBLE)', 'WU(DOUBLE)', 'SIZE(DOUBLE)']]\
    .sort_values(by = 'UNIT(IFACTOR)').reset_index(drop = True)
    
    return units_df

def createData(impressions):
    '''
    Arguments:
        impressions - Our impression data
        
    Return:
        data_df - our sample data
    '''
    data3 = impressions.drop_duplicates(['respondentid', 'cc']).rename( columns = {
            'respondentid' : 'UNIT(STRING)', 'cc' : 'CRIT(STRING)'} ).reset_index(drop = True)
    data3['CRIT(STRING)'] = data3['CRIT(STRING)'].apply(lambda x : 'population::cc::' + str(x))
    data3.loc[:, 'A(DOUBLE)'] = 1
    
    data1 = pd.DataFrame( columns = ['UNIT(STRING)', 'CRIT(STRING)', 'A(DOUBLE)'] )
    
    data2 = pd.DataFrame( columns = ['UNIT(STRING)', 'CRIT(STRING)', 'A(DOUBLE)'] )
        
    indices = impressions.loc[:, 'val'] > 0
    
    impressions = impressions.loc[indices, :].reset_index(drop = True)
    
    data1.loc[:, 'UNIT(STRING)'] = impressions['respondentid']
    data1.loc[:, 'CRIT(STRING)'] = 'reach::' + impressions['var']
    data1.loc[:, 'A(DOUBLE)'] = 1
    
    data2.loc[:, 'UNIT(STRING)'] = impressions['respondentid']
    data2.loc[:, 'CRIT(STRING)'] = 'imp::' + impressions['var']
    data2.loc[:, 'A(DOUBLE)'] = impressions['val']
        
    data_df = pd.concat( [data1, data2, data3], axis = 0, sort = True)\
    .loc[:, ['UNIT(STRING)', 'CRIT(STRING)', 'A(DOUBLE)']].reset_index(drop = True)
    
    return data_df
