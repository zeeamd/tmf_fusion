# -*- coding: utf-8 -*-
"""
Created on Thu Aug  9 22:26:29 2018

@author: tued7001
"""
import logging
from abc import ABCMeta, abstractmethod

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.preprocessing import LabelEncoder

from .calibration.soft_calibration import soft_calibrate


class Preprocess(object, metaclass=ABCMeta):
    
    def __init__(self, loc_adj_fact, tot_adj_fact, estimate_huber):

        self._logger = logging.getLogger(__name__)

        assert(isinstance(loc_adj_fact, int) | isinstance(loc_adj_fact, float))
        assert(isinstance(tot_adj_fact, int) | isinstance(tot_adj_fact, float))
        assert(isinstance(estimate_huber, bool))

        self.soft_cal_params = {'loc_adj_fact': loc_adj_fact, 'tot_adj_fact': tot_adj_fact,
                                'estimate_huber': estimate_huber}
        return None
    
    @abstractmethod
    def create_donors_recips(self, **kwargs):
        pass

    @abstractmethod
    def create_soft_cal_targets(self, **kwargs):
        pass

    @abstractmethod
    def update_donor_recips(self):
        return self

    @staticmethod
    @abstractmethod
    def create_cc_entries():
        pass

    @staticmethod
    @abstractmethod
    def setup_dcr_demos():
        pass

    @property
    def linking_variables_master(self):
        return self._linking_vars_mas

    @linking_variables_master.setter
    def linking_variables_master(self, val):

        assert(isinstance(val, list))

        self._linking_vars_mas = val

    @property
    def linking_variables_pc(self):
        return self._linking_vars_pc

    @linking_variables_pc.setter
    def linking_variables_pc(self, val):
        assert (isinstance(val, list))

        self._linking_vars_pc = val

    @property
    def linking_variables_se_init(self):
        return self._linking_vars_se_init

    @linking_variables_se_init.setter
    def linking_variables_se_init(self, val):
        assert (isinstance(val, list))

        self._linking_vars_se_init = val

    @property
    def linking_variables_se_final(self):
        return self._linking_vars_se_final

    @linking_variables_se_final.setter
    def linking_variables_se_final(self, val):
        assert (isinstance(val, list))

        self._linking_vars_se_final = val

    @property
    def linking_variables_iw(self):
        return self._linking_vars_iw

    @linking_variables_iw.setter
    def linking_variables_iw(self, val):
        assert (isinstance(val, list))

        self._linking_vars_iw = val

    @property
    def iw_rec_vars(self):
        return self._iw_rec_vars

    @iw_rec_vars.setter
    def iw_rec_vars(self, val):

        assert isinstance(val, list)

        self._iw_rec_vars = val

    @property
    def iw_don_vars(self):
        return self._iw_don_vars

    @iw_don_vars.setter
    def iw_don_vars(self, val):
        assert isinstance(val, list)

        self._iw_don_vars = val

    @property
    def donors(self):
        return self._donors
    
    @donors.setter
    def donors(self, df):

        assert(isinstance(df, pd.DataFrame))

        self._donors = df

    @property
    def recips(self):
        return self._recips
    
    @recips.setter
    def recips(self, df):

        assert(isinstance(df, pd.DataFrame))

        self._recips = df

    @property
    def soft_cal_targets(self):
        return self._soft_cal_targets
    
    @soft_cal_targets.setter
    def soft_cal_targets(self, df):

        assert(isinstance(df, pd.DataFrame))

        self._soft_cal_targets = df

    @property
    def soft_cal_impress(self):
        return self._soft_cal_impress
    
    @soft_cal_impress.setter
    def soft_cal_impress(self, df):

        assert(isinstance(df, pd.DataFrame))

        self._soft_cal_impress = df

    @property
    def old_links(self):
        return self._old_links
    
    @old_links.setter
    def old_links(self, df):

        assert(isinstance(df, pd.DataFrame))

        self._old_links = df

    @staticmethod
    def pivot_table_dataframe(df, val_col, idx_col, column_col):
        """
        :param df:
        :param val_col:
        :param idx_col:
        :param column_col:
        :return:
        """

        assert isinstance(df, pd.DataFrame)
        assert isinstance(val_col, str)
        assert isinstance(idx_col, str)
        assert isinstance(column_col, str)

        lbl_encoder_idx = LabelEncoder()
        lbl_encoder_column = LabelEncoder()

        idx_u = df[idx_col].sort_values().unique()
        col_u = df[column_col].sort_values().unique()

        # labels the sorted unique values
        lbl_encoder_idx.fit(idx_u)
        lbl_encoder_column.fit(col_u)

        idx_i = lbl_encoder_idx.transform(df[idx_col])
        idx_j = lbl_encoder_column.transform(df[column_col])

        nrows = len(lbl_encoder_idx.classes_)
        ncols = len(lbl_encoder_column.classes_)

        # creates the sparse pivot table
        A_piv = csr_matrix((df[val_col].values, (idx_i, idx_j)), shape=(nrows, ncols))

        return pd.SparseDataFrame(data=A_piv, index=list(idx_u), columns=list(col_u)).to_dense().fillna(0.0)

    @staticmethod
    def create_age_gender_demos(demos):
        """
        :param demos:
        :return:
        """
        assert isinstance(demos, pd.DataFrame)

        p_lower_bounds = ['02', '02', '13', '18']
        p_upper_bounds = ['12', '17', '99', '99']

        g_lower_bounds_1 = ['13', '18', '21', '25', '30', '35', '40', '45', '50', '55', '65']
        g_upper_bounds_1 = ['17', '20', '24', '29', '34', '39', '44', '49', '54', '64', '99']

        g_lower_bounds_2 = ['13', '18', '18', '18', '18', '25', '35', '35', '45', '45', '55']
        g_upper_bounds_2 = ['99', '24', '34', '44', '99', '34', '44', '54', '54', '99', '99']

        g_lower_bounds = g_lower_bounds_1 + g_lower_bounds_2
        g_upper_bounds = g_upper_bounds_1 + g_upper_bounds_2

        n = len(g_lower_bounds)

        demo_males = ['demo_m' + g_lower_bounds[i] + g_upper_bounds[i] for i in range(n)]
        demo_females = ['demo_f' + g_lower_bounds[i] + g_upper_bounds[i] for i in range(n)]
        demo_person = ['demo_p' + p_lower_bounds[i] + p_upper_bounds[i] for i in range(4)]

        for i in range(4):
            demos[demo_person[i]] = 0

        demos['demo_male'] = 0
        demos['demo_female'] = 0
        demos['demo_total'] = 1

        # now time to deal with male values only
        indices = demos['gender'] == 1

        demos.loc[indices, 'demo_male'] = 1

        for i in range(n):
            # creates the new columns and initializes it with the value 0
            demos[demo_males[i]] = 0

            l_bound = int(g_lower_bounds[i])
            u_bound = int(g_upper_bounds[i])

            demos.loc[indices & ((demos['age'] <= u_bound) & (l_bound <= demos['age'])),
                      demo_males[i]] = 1

        # now time to deal with female values only
        indices = demos['gender'] == 2

        demos.loc[indices, 'demo_female'] = 1

        for i in range(n):
            # creates the new columns and initializes it with the value 0
            demos[demo_females[i]] = 0

            l_bound = int(g_lower_bounds[i])
            u_bound = int(g_upper_bounds[i])

            demos.loc[indices & ((demos['age'] <= u_bound) & (l_bound <= demos['age'])),
                      demo_females[i]] = 1

        # now we just look at person in this grouping
        for i in range(4):
            demos[demo_person[i]] = 0

            l_bound = int(p_lower_bounds[i])
            u_bound = int(p_upper_bounds[i])

            demos.loc[((demos['age'] <= u_bound) & (l_bound <= demos['age'])),
                      demo_person[i]] = 1

        return demos

    @staticmethod
    def create_dcr_targets(dcr_pc_df, dcr_pcv_df, dcr_pc_combine_targets, dcr_pcv_combine_targets, tot_univ, dcr_target,
                           dcr_target_sub, dcr_vid):
        """
        :param dcr_pc_df:
        :param dcr_pcv_df:
        :param dcr_pc_combine_targets:
        :param dcr_pcv_combine_targets:
        :param tot_univ:
        :param dcr_target:
        :param dcr_target_sub:
        :param dcr_vid:
        :return:
        """

        assert isinstance(dcr_pc_df, pd.DataFrame)
        assert isinstance(dcr_pcv_df, pd.DataFrame)
        assert isinstance(dcr_pc_combine_targets, pd.DataFrame)
        assert isinstance(dcr_pcv_combine_targets, pd.DataFrame)
        assert isinstance(dcr_target, pd.DataFrame)
        assert isinstance(dcr_target_sub, pd.DataFrame)
        assert isinstance(dcr_vid, pd.DataFrame)
        assert isinstance(tot_univ, float)

        # sets a function for splitting
        f = lambda x: str(x).split('_')[0]

        if isinstance(dcr_target['Target'], pd.DataFrame):
            dcr_target['name'] = dcr_target['Target'].T.squeeze().apply(f)
        else:
            dcr_target['name'] = dcr_target['Target'].apply(f)

        if isinstance(dcr_target_sub['Target'], pd.DataFrame):
            dcr_target_sub['name'] = dcr_target_sub['Target'].T.squeeze().apply(f)
        else:
            dcr_target_sub['name'] = dcr_target_sub['Target'].apply(f)

        if isinstance(dcr_vid['Target'], pd.DataFrame):
            dcr_vid['name'] = dcr_vid['Target'].T.squeeze().apply(f)
        else:
            dcr_vid['name'] = dcr_vid['Target'].apply(f)

        # concatenates the target and target_sub data
        dcr_target2 = pd.concat([dcr_target, dcr_target_sub], axis=0, sort=True).reset_index(drop=True)

        dcr_target2.columns = [col.lower() for col in dcr_target2.columns]
        dcr_vid.columns = [col.lower() for col in dcr_vid.columns]

        dcr_pc_df['use_level'] = dcr_pc_df['use_level'].astype(int)
        dcr_pcv_df['use_level'] = dcr_pcv_df['use_level'].astype(int)

        dcr_pc_df['dummy'] = 1
        dcr_pcv_df['dummy'] = 1

        # this helps restrict our attention to pairs of name ans level that are in the dcr_pc dataframe
        join1 = dcr_target2.set_index('brandid').join(
            dcr_pc_combine_targets.set_index('site_id')[['prop_reach', 'prop_dur', 'prop_pv']]
            , how='inner').reset_index(drop=True)

        dcr_targets_sub = join1.set_index(['name', 'level']).join(
            dcr_pc_df.set_index(['site', 'use_level'])['dummy'].to_frame().rename_axis(['name', 'level']), how='inner')\
            .rename(columns={'target': 'var'})

        # drop every less than 18
        indices = dcr_targets_sub['var'].apply(lambda x: 'p02' not in x)

        dcr_targets_sub = dcr_targets_sub.loc[indices, :].reset_index(drop=True)

        # sets up reach, duration, imp and code variables
        dcr_targets_sub['reach'] = dcr_targets_sub['reach'] * dcr_targets_sub['prop_reach']

        dcr_targets_sub['dur'] = dcr_targets_sub['duration'] * dcr_targets_sub['prop_dur']

        dcr_targets_sub['imp'] = dcr_targets_sub['impression'] * dcr_targets_sub['prop_pv']

        dcr_targets_sub['code'] = 1

        dcr_targets_sub = dcr_targets_sub[['var', 'code', 'imp', 'reach', 'dur']]

        dcr_targets_sub_d = dcr_targets_sub.copy()

        dcr_targets_sub.drop('dur', axis=1, inplace=True)

        dcr_targets_sub_d['var'] = dcr_targets_sub_d['var'].apply(lambda x: x.replace('_pv_', '_mm_'))

        dcr_targets_sub_d['imp'] = dcr_targets_sub_d['dur'] * 60

        dcr_targets_sub_d.drop('dur', axis=1, inplace=True)

        dcr_targets_sub2 = dcr_targets_sub.copy()
        dcr_targets_sub2['reach'] = tot_univ - dcr_targets_sub2['reach']
        dcr_targets_sub2['imp'] = 0
        dcr_targets_sub2['code'] = 0

        dcr_targets_sub2d = dcr_targets_sub_d.copy()
        dcr_targets_sub2d['reach'] = tot_univ - dcr_targets_sub2d['reach']
        dcr_targets_sub2d['imp'] = 0
        dcr_targets_sub2d['code'] = 0

        # this helps restrict our attention to pairs of name ans level that are in the dcr_pc dataframe
        join1 = dcr_vid.set_index('brandid').join(
            dcr_pcv_combine_targets.set_index('site_id')[['prop_reach', 'prop_dur']]
            , how='inner').reset_index(drop=True)

        dcr_targets_sub_vid = join1.set_index(['name', 'level']).join(
            dcr_pcv_df.set_index(['site', 'use_level'])['dummy'].to_frame() \
                .rename_axis(['name', 'level']), how='inner').rename(columns={'target': 'var'})

        indices = dcr_targets_sub_vid['var'].apply(lambda x: 'p02' not in x)
        dcr_targets_sub_vid = dcr_targets_sub_vid.loc[indices, :].reset_index(drop=True)

        dcr_targets_sub_vid['imp'] = dcr_targets_sub_vid['duration'] * dcr_targets_sub_vid['prop_dur']
        dcr_targets_sub_vid['reach'] = dcr_targets_sub_vid['reach'] * dcr_targets_sub_vid['prop_reach']
        dcr_targets_sub_vid['code'] = 1

        dcr_targets_sub_vid2 = dcr_targets_sub_vid.copy()

        dcr_targets_sub_vid2['reach'] = tot_univ - dcr_targets_sub_vid2['reach']
        dcr_targets_sub_vid2['imp'] = 0
        dcr_targets_sub_vid2['code'] = 0

        targets = pd.concat([dcr_targets_sub, dcr_targets_sub2, dcr_targets_sub_d, dcr_targets_sub2d,
                             dcr_targets_sub_vid, dcr_targets_sub_vid2], axis=0, sort=True).reset_index(drop=True)\
            .dropna(axis=1)

        return targets

    def use_level_by_site_id(self, media_usage_df, min_lvl_cnt_c):
        """
        :param media_usage_df:
        :param min_lvl_cnt_c:
        :return:
        """
        logger = self._logger

        logger.debug('Setting up DCR Demos')

        assert isinstance(media_usage_df, pd.DataFrame)
        assert isinstance(min_lvl_cnt_c, int) | isinstance(min_lvl_cnt_c, float)

        # we get the DCR Demos
        dcr_demos = self.setup_dcr_demos()

        logger.debug('Getting all breaks')

        # we get all the breaks in the dcr demos
        breaks_df = dcr_demos.loc[dcr_demos['check'] == 1, ['name', 'level']].reset_index(drop=True)

        break_names = list(breaks_df['name'].values)

        # from the breaks we are checking, we group by the site_id and sum by the names
        cols = break_names

        breaks_c = pd.melt(media_usage_df.loc[:, cols], id_vars=['site_id'],
                           value_vars=break_names,
                           var_name='break',
                           value_name='indicator')

        # now we filter
        breaks_c = breaks_c.loc[breaks_c['indicator'] == 1, :].reset_index(drop=True)

        # these are our counts, with breaks the exist only in our dataset
        breaks_c = breaks_c.groupby('break')['indicator'] \
            .sum().reset_index().rename(columns={'indicator': 'count'})

        breaks_c = pd.merge(breaks_c.rename(columns={'break': 'name'}), breaks_df[['name', 'level']], on='name')

        logger.debug('Getting all levels')

        level_c = breaks_c.groupby('level')['count'].min().reset_index()

        for i in range(9):
            ind = level_c['level'] == 8 - i

            if min_lvl_cnt_c < level_c.loc[ind, 'count'].values:
                return 8 - i

        return 99

    def create_demo_var_by_site_id(self, input_df, level, name, media_type, site_sfx):

        assert isinstance(input_df, pd.DataFrame)
        assert isinstance(level, int)
        assert isinstance(name, str)
        assert isinstance(media_type, str)
        assert isinstance(site_sfx, str)

        logger = self._logger

        if level > 8:
            logger.warning('Usage level is higher than tolerance')
            return pd.DataFrame(columns=['rn_id'])

        if level == 8:
            age_bins = ['0212', '1317', '1820', '2124', '2529', '3034', '3539', '4044', '4549', '5054', '5564', '6599']

        if level == 7:
            age_bins = ['0217', '1824', '2534', '3544', '4554', '5564', '6599']

        if level == 6:
            age_bins = ['0217', '1834', '3554', '5599']

        if level == 5:
            age_bins = ['0217', '1844', '4599']

        if level == 4:
            age_bins = ['0217', '1899']

        if level == 3:
            age_bins = ['0212', '1399']

        if level == 2:
            age_bins = ['0217', 'p1899']

        if level == 1:
            age_bins = ['0212', 'p1399']

        if level == 0:
            age_bins = ['total']

        logger.debug('Size of input: {}'.format(input_df.shape[0]))

        # drop all columns if demo is in the name
        cols_demo = [col for col in input_df.columns if 'demo' in col and not ('male' in col or 'female' in col)]

        # reverses pivot table of indicators of ages and genders
        output_df = pd.melt(input_df, id_vars=['rn_id'],
                            value_vars=cols_demo,
                            var_name='demo_var',
                            value_name='indicator'
                            )

        # this removes site_ids and rn_ids with no one in a particular bin
        output_df = output_df.loc[output_df['indicator'] == 1, :].reset_index(drop=True) \
            .drop('indicator', axis=1)

        if site_sfx in ['vd', 'sv']:
            # now we recover the columns of interest
            output_df = pd.merge(output_df, input_df[['rn_id', 'minutes']],
                                 on=['rn_id'])

            # now we create a dummy column to help us create the proper bins
            output_df['var'] = output_df['demo_var'].apply(
                lambda x: '_'.join([name, media_type, site_sfx, x.replace('demo', 'mm')]))

            indices = output_df['var'].apply(lambda x: np.any([age_bin in x for age_bin in age_bins]))

            res_df = self.pivot_table_dataframe(output_df.loc[indices, :].groupby(['rn_id', 'var'])['minutes'].sum() \
                                               .reset_index(), 'minutes', 'rn_id', 'var').reset_index() \
                .rename(columns={'index': 'rn_id'}).to_dense()
        else:
            output_df = pd.merge(output_df, input_df[['rn_id', 'pageviews', 'viewduration']], on=['rn_id'])

            # now we create a dummy column to help us create the proper bins
            output_df['var'] = output_df['demo_var'].apply(lambda x: '_'.join([name, media_type, site_sfx,
                                                                               x.replace('demo', 'pv')]))

            output_df['var2'] = output_df['demo_var'].apply(lambda x: '_'.join([name, media_type, site_sfx,
                                                                                x.replace('demo', 'mm')]))
            indices = output_df['var'].apply(lambda x: np.any([age_bin in x for age_bin in age_bins]))

            pv_df = self.pivot_table_dataframe(
                output_df.loc[indices, :].groupby(['rn_id', 'var'])['pageviews'].sum().reset_index(),
                'pageviews', 'rn_id', 'var').reset_index().rename(columns={'index': 'rn_id'}).to_dense()

            indices = output_df['var2'].apply(lambda x: np.any([age_bin in x for age_bin in age_bins]))

            mm_df = self.pivot_table_dataframe(
                output_df.loc[indices, :].groupby(['rn_id', 'var2'])['viewduration'].sum().reset_index(),
                'viewduration', 'rn_id', 'var2').reset_index().rename(
                columns={'index': 'rn_id'}).to_dense()

            res_df = pd.merge(pv_df, mm_df, on='rn_id', how='outer').fillna(0.0)

            if media_type == 'pc':
                pass
            else:
                output_df['dummy_var'] = '_'.join([name, 'pcmob', site_sfx, 'dup'])

                if 'pageviews_mob' in output_df.columns:
                    output_df['pageviews_mob'] = output_df['pageviews_mob'].fillna(0.0)
                else:
                    output_df['pageviews_mob'] = 0

                output_df['dummy_var_ind'] = ((output_df['pageviews_mob'] > 0) & (output_df['pageviews'] > 0)).astype(
                    int)

                indices = output_df['dummy_var'].apply(lambda x: np.any([age_bin in x for age_bin in age_bins]))

                temp_piv = self.pivot_table_dataframe(
                    output_df.loc[indices, :].groupby(['rn_id', 'dummy_var'])['dummy_var_ind']\
                    .sum().reset_index(), 'dummy_var_ind', 'rn_id', 'dummy_var').reset_index() \
                    .rename(columns={'index': 'rn_id'}).to_dense()

                res_df = pd.merge(temp_piv, res_df, on='rn_id')

        res_df['rn_id'] = res_df['rn_id'].astype(float)

        logging.debug('Size of output: {}'.format(res_df.shape[0]))

        return res_df

    def create_dcr_impressions_by_site_id(self, dcr_usage_df, dcr_site_name, media_type, min_lvl_cnt_c,
                                          site_sfx, use_level=None):
        """
        :param dcr_usage_df:
        :param dcr_site_name:
        :param media_type:
        :param min_lvl_cnt_c:
        :param site_sfx:
        :param use_level:
        :return:
        """
        logger = self._logger

        if use_level is None:
            use_level = self.use_level_by_site_id(dcr_usage_df, min_lvl_cnt_c)
            logger.info('Usage Level is: {}'.format(use_level))
            dcr_imps = self.create_demo_var_by_site_id(dcr_usage_df, use_level, dcr_site_name, media_type, site_sfx)
            return dcr_imps, use_level
        else:
            dcr_imps = self.create_demo_var_by_site_id(dcr_usage_df, use_level, dcr_site_name, media_type, site_sfx)
            return dcr_imps

    def calibrate_weights(self, df, soft_cal_targets, soft_cal_impressions, id_col):
        """
        :param df:
        :param soft_cal_targets:
        :param soft_cal_impressions:
        :param id_col:
        :return:
        """
        assert(isinstance(id_col, str))

        base_logger = self._logger

        base_logger.info("Performing soft calibration")

        soft_cal_res = soft_calibrate(soft_cal_targets, soft_cal_impressions,
                                      loc_adj_fact=self.soft_cal_params.get('loc_adj_fact'),
                                      tot_adj_fact=self.soft_cal_params.get('tot_adj_fact'),
                                      estimate_huber=self.soft_cal_params.get('estimate_huber'))
    
        base_logger.info('Updating weights')
        df['unitLabel'] = df[id_col].apply(lambda x: 'r' + str(x))
        
        df = pd.merge(df, soft_cal_res[['unitLabel', 'xf']] .rename(columns={'xf': 'new_weight'}), on='unitLabel',
                      how='left')

        no_new_weight = df['new_weight'].isnull()
        
        df.loc[~no_new_weight, 'weight'] = df.loc[~no_new_weight, 'new_weight']
        
        df.drop(['new_weight', 'unitLabel'], axis=1, inplace=True)
        
        return df
    
    def calibrate_donor_weights(self):
        """
        :return:
        """

        base_logger = self._logger

        base_logger.info("Calibrating our donor weights")

        donors = self.donors
        targets = self.soft_cal_targets
        impress = self.soft_cal_impress
        
        self.donors = self.calibrate_weights(donors, targets, impress, 'respondentid')
        
        return self
    
    def calibrate_recip_weights(self):
        """
        :return:
        """

        base_logger = self._logger

        base_logger.info("Calibrating our recipient weights")

        recips = self.recips
        targets = self.soft_cal_targets
        impress = self.soft_cal_impress
                
        self.recips = self.calibrate_weights(recips, targets, impress, 'rn_id')
        
        return self

    def determine_old_linkage(self, prev_donors, prev_recips, prev_linkages):
        """
        :param prev_donors:
        :param prev_recips:
        :param prev_linkages:
        :return:
        """
        
        assert(isinstance(prev_donors, pd.DataFrame))
        assert(isinstance(prev_recips, pd.DataFrame))
        assert(isinstance(prev_linkages, pd.DataFrame))

        donors = self.donors
        recips = self.recips
        
        old_donors = prev_donors.rename(columns={'respondentid' : 'donorid',
                                                   'cc': 'cc_donor_prev'})
    
        old_recips = prev_recips.rename(columns={'respondentid' : 'recipientid',
                                               'cc': 'cc_recip_prev'})
    
        cur_donors = donors.rename(columns = {'respondentid' : 'donorid', 
                                                   'cc': 'cc_donor'})
    
        cur_recips = recips.rename(columns = {'respondentid' : 'recipientid', 
                                                   'cc': 'cc_recip'})

        old_links = pd.merge(pd.merge(prev_linkages, old_donors, on='donorid'), old_recips, on='recipientid')
    
        old_links = pd.merge( pd.merge( old_links, cur_donors, on = 'donorid'),
                            cur_recips, on = 'recipientid' ) 
    
        indices = ( old_links['cc_donor'] == old_links['cc_donor_prev'] ) & \
        ( old_links['cc_recip'] == old_links['cc_recip_prev'] )
    
        self.old_links = old_links.loc[indices, :].reset_index(drop = True).\
        drop(['cc_donor', 'cc_recip', 'cc_donor_prev', 'cc_recip_prev'], axis = 1)

        return self
