# -*- coding: utf-8 -*-
"""
Created on Wed Dec 20 16:07:00 2017

@author: tued7001
"""

import pandas as pd

def processTargets(rawTargets):
   
    #create new category variable called delivery
    rawTargets.loc[rawTargets['var'].apply(lambda x: x.endswith("_pc")) == True, 'delivery'] = 'PC'
    rawTargets.loc[rawTargets['var'].apply(lambda x: x.endswith("_tab")) == True, 'delivery'] = 'T'
    rawTargets.loc[rawTargets['var'].apply(lambda x: x.endswith("_mob")) == True, 'delivery'] = 'M'
    rawTargets.loc[rawTargets['var'].apply(lambda x: x.endswith("_strm")) == True, 'delivery'] = 'STREAM'
    rawTargets.loc[rawTargets['var'].apply(lambda x: x.endswith("_dup")) == True, 'delivery'] = 'DUP'
    rawTargets.loc[rawTargets['var'].apply(lambda x: x.endswith("cc")) == True, 'delivery'] = 'CC'
    rawTargets.loc[rawTargets['var'].apply(lambda x: x.endswith("npm")) == True, 'delivery'] = 'NPM'
    #add a row for FBCC
    
    #transpose rawTargets to create audienceTargets
    subset = rawTargets.loc[((rawTargets['var'] != "cc") & (rawTargets['var'] != "npm") & (rawTargets['code'] > 0) & (rawTargets['imp'] > 0)) | \
        ((rawTargets['delivery'] == "DUP") & (rawTargets['code'] > 0))]
    audienceTargets = pd.melt(subset, id_vars=['var', 'delivery', 'code'], var_name = 'targetType', value_name ='val')
    audienceTargets = audienceTargets.loc[(audienceTargets['targetType'] != "imp") | (audienceTargets['val'] > 0)]

    #create importanceGroup variable
    audienceTargets.loc[audienceTargets['targetType'] == 'reach', 'importanceGroup'] = 'reach'
    audienceTargets.loc[(audienceTargets['targetType'] == 'reach') & (audienceTargets['delivery'] == 'DUP'), 'importanceGroup'] = 'dup'
    audienceTargets.loc[audienceTargets['targetType'] == 'imp', 'importanceGroup'] = 'pageviews'
    audienceTargets.loc[(audienceTargets['targetType'] == 'imp') & (audienceTargets['delivery'] == 'STREAM'), 'importanceGroup'] = 'minutes'

    #create cc targets
    ccTargets = rawTargets.loc[rawTargets['var'] == 'cc'].reset_index(drop=True)
    ccTargets['var'] = ccTargets.apply(lambda x: x['var'] + "::" + str(x['code']), axis=1)
    ccTargets['targetType'] = "population"
    ccTargets['importanceGroup'] = "cc"
    ccTargets['val'] = ccTargets['reach']
    ccTargets.drop(['reach', 'imp', 'delivery'], axis=1, inplace=True)
    
    #stack the audience and cc targets datasets
    targets = pd.concat([ audienceTargets[['var', 'code', 'targetType', 'importanceGroup', 'val']], ccTargets], ignore_index=True, sort = True)

    #we need to pass a dictionary for each of these now, and make sure we have the same number of keys as importance groups
    
    #create new variables for importance units 
    targets.loc[targets['importanceGroup'] == 'pageviews', 'importanceUnits'] = 30
    targets.loc[targets['importanceGroup'] == 'minutes', 'importanceUnits'] = 20
    targets.loc[targets['importanceGroup'] == 'reach', 'importanceUnits'] = 20
    targets.loc[targets['importanceGroup'] == 'dup', 'importanceUnits'] = 28
    targets.loc[targets['importanceGroup'] == 'fbcc', 'importanceUnits'] = 1
    targets.loc[targets['importanceGroup'] == 'cc', 'importanceUnits'] = 1

    return targets 
    
    
#This function reads in the soft calibrations impressions and preps input data
def processImpressions( rawImpressions ):
    
    #read in respondent level file	
    id_vars = ['cc', 'respondentid', 'weight']
                
    df = pd.melt(rawImpressions, id_vars=id_vars, var_name = 'var',
                        value_name ='val')
        
    return df
 
