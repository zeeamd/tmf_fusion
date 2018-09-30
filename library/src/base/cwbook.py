# -*- coding: utf-8 -*-
"""
Created on Mon May  7 15:28:23 2018

@author: tued7001
"""
import pandas as pd


# this is a configuration object to help us keep track of tasks in a fusion
class ConfigWB(object):

    def __init__(self, update_fusion, fusion_match_type, modules_run, **kwargs):
        '''
        Input:
            modules_run - a dictionary that holds all the modules that we will
            run
            fusion_type - the type of fusion we will run
            kwargs - extra parameters we will store
        '''

        assert isinstance(modules_run, dict)

        self._modules_run = modules_run

        assert isinstance(update_fusion, bool)

        self._update_fusion = update_fusion

        assert isinstance(fusion_match_type, str)

        self.fusion_match_type = fusion_match_type

    '''
    The following is our getting and setter functions
    '''

    # these parameters are passed into the object

    @property
    def modules_run(self):
        return self._modules_run

    @modules_run.setter
    def modules_run(self, val):
        assert isinstance(val, dict)

        self._modules_run = val

    @property
    def update_fusion(self):
        return self._update_fusion

    @update_fusion.setter
    def update_fusion(self, val):
        assert isinstance(val, bool)

        self._update_fusion = val

    @property
    def fusion_match_type(self):
        return self._fusion_match_type

    @fusion_match_type.setter
    def fusion_match_type(self, val):
        assert isinstance(val, str)

        self._fusion_match_type = val

    @property
    def se_eval_vars_initial(self):
        return self._se_eval_vars_initial

    @se_eval_vars_initial.setter
    def se_eval_vars_initial(self, val):
        assert isinstance(val, list)

        self._se_eval_vars_initial = val

    @property
    def se_eval_vars_final(self):
        return self._se_eval_vars_final

    @se_eval_vars_final.setter
    def se_eval_vars_final(self, val):
        assert isinstance(val, list)

        self._se_eval_vars_final = val

    @property
    def lv(self):
        return self._lv

    @lv.setter
    def lv(self, val):
        assert isinstance(val, list)

        self._lv = val

    @property
    def lv_iw(self):
        return self._lv_iw

    @lv_iw.setter
    def lv_iw(self, val):
        assert isinstance(val, list)

        self._lv_iw = val

    @property
    def lv_pc(self):
        return self._lv_pc

    @lv_pc.setter
    def lv_pc(self, val):
        assert isinstance(val, list)

        self._lv_pc = val

    @property
    def trim_grey_area(self):
        return self.__dict__.get('_trim_max_grey_area', 0.15)

    @trim_grey_area.setter
    def trim_max_grey_area(self, val):
        assert (isinstance(val, float))

        self._trim_max_grey_area = val

    @property
    def trim_iterations(self):
        return self.__dict__.get('_trim_iterations', 3)

    @trim_iterations.setter
    def trim_iterations(self, val):
        assert isinstance(val, int)

        self._trim_iterations = val

    @property
    def min_pvalue(self):
        return self.__dict__.get('_min_pvalue', 1)

    @min_pvalue.setter
    def min_pvalue(self, val):
        assert isinstance(val, float) | isinstance(val, int)

        self._min_pvalue = val

    @property
    def min_pvalue_small_samp(self):
        return self.__dict__.get('_min_pvalue_small_samp', 5)

    @min_pvalue_small_samp.setter
    def min_pvalue_small_samp(self, val):
        assert isinstance(val, float) | isinstance(val, int)

        self._min_pvalue_small_samp = val

    @property
    def max_pct_cat_diff(self):
        return self.__dict__.get('_max_pct_cat_diff', 10)

    @max_pct_cat_diff.setter
    def max_pct_cat_diff(self, val):
        assert isinstance(val, float) | isinstance(val, int)

        self._max_pct_cat_diff = val

    @property
    def iw_don_vars(self):
        return self._iw_don_vars

    @iw_don_vars.setter
    def iw_don_vars(self, val):
        assert isinstance(val, list)

        self._iw_don_vars = val

    @property
    def iw_rec_vars(self):
        return self._iw_rec_vars

    @iw_rec_vars.setter
    def iw_rec_vars(self, val):
        assert isinstance(val, list)

        self._iw_rec_vars = val

    ##TODO

    # returns a list of link variables for hot decking
    def getHotDeckLV(self):
        # config_wb.parse('Hotdeck_LV', skiprows = 4).dropna()
        return self.__dict__.get('hd_link_vars', [])

    # we set our hot decked linking variables
    def setHotDeckLV(self, hd_vars):
        self.hd_link_vars = hd_vars

    # this gets the cc labels and their meaning
    def getCCLabels(self):
        return self.__dict__.get('cc_labels', pd.DataFrame())

    # this gets the look up table
    def getLookUpTable(self):
        return self.__dict__.get('look_up_table', pd.DataFrame())

    # get the fused variables
    def getFusedVariables(self):
        return self.__dict__.get('fused_vars', [])

    # gets our output evaluation fused variables
    def getFusedVariablesOE(self):
        return self.__dict__.get('fused_vars_oe', [])

    '''
    The following is a list of behaviors for our configuration workbook
    '''

    # gets whether we perform sample evaluation by cc value
    def isSampleEvaluationBy(self):
        '''
        Default value is by cc
        '''
        return self.__dict__.get('sample_evaluation_by', 'cc')
