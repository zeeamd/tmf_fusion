# -*- coding: utf-8 -*-
"""
Author: Edward Turner

The purpose of this code is to preprocess the targets and respondents, and to
perform soft calibrations on datasets before fusion.  
This code requires two datasets, with columns cc, var, val, weights,
 and respondentid.
 
Please consult the documentation.
"""
import logging
# helps with memory management and os operations
from gc import collect

# data management packages
import pandas as pd

from .calibration_data import createCriteria, createUnits, createData
# modules create to help with the preprocessing
from .process_targets_impressions import processTargets, processImpressions
from .soft_calib_solver import executePythonSoftCalibration
from .soft_calib_stats import huberParameterEsitmation


def soft_calibrate(targets, impressions, effective_sample=30000, min_rse=0.0, max_rse=0.25, loc_adj_fact=4,
                       tot_adj_fact=20, estimate_huber=True):
    """
    :param targets:
    :param impressions:
    :param effective_sample:
    :param min_rse:
    :param max_rse:
    :param loc_adj_fact:
    :param tot_adj_fact:
    :param estimate_huber:
    :return:
    """

    soft_cal_main_logger = logging.getLogger(__name__)

    soft_cal_main_logger.info("Processing our raw targets")

    #this processes the targets
    targets = processTargets(targets)

    soft_cal_main_logger.info("Processing our impressions")

    #this processes the impressions
    impressions = processImpressions(impressions)
    
    soft_cal_main_logger.info("Ensuring our target's code are in the right places")

    #we make sure the targets code is in the right places
    indices = targets['var'].apply(lambda x : 'cc' in x)
    
    targets_no_cc = targets.loc[~indices, :].reset_index(drop = True)
    
    targets_cc = targets.loc[indices, :].reset_index(drop = True)
    
    targets_cc = targets_cc.loc[ targets_cc['code'].isin(impressions['cc']), : ].reset_index(drop = True)
    
    targets = pd.concat( [targets_no_cc, targets_cc], axis = 0)
        
    targets = targets.dropna().reset_index(drop = True)
    
    del targets_no_cc, targets_cc, indices
    collect()
    
    soft_cal_main_logger.info("Determining our bounds on the calibrated targets")

    #sets up our criteria for our targets
    criteria_df = createCriteria( targets.copy(), impressions, effective_sample, min_rse, max_rse )    
    collect()
    
    soft_cal_main_logger.info("Estimating our huber parameter")

    if estimate_huber:
        estimated_huber = huberParameterEsitmation(impressions.copy(), criteria_df)

    else:
        estimated_huber = 0.5

    soft_cal_main_logger.info("Our huber parameter is {}".format( estimated_huber ) )

    soft_cal_main_logger.info("Determining our bounds on the weights")

    #sets up our bounds for our calibration weights
    units_df = createUnits(impressions.copy(), loc_adj_fact, tot_adj_fact)
    
    #free memory of targets
    del targets
    collect()
    
    soft_cal_main_logger.info("Creating our sample data")

    #creates our sample data for our soft calibration
    data_df = createData( impressions )
    
    #free memory of impressions
    del impressions
    collect()
    
    #calibrates the weights
    df = executePythonSoftCalibration(estimated_huber, criteria_df, units_df, data_df)
    
    return df
