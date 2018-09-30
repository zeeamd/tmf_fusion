# -*- coding: utf-8 -*-
"""
Created on Fri Aug 10 13:54:48 2018

@author: tued7001
"""
import logging

import numpy as np
import pandas as pd
from library.src.base.preprocess import Preprocess

from .data_process import buildRecipDonorIHDataset


class PreIHPreprocess(Preprocess):

    def __init__(self):

        Preprocess.__init__(self, 4, 20, True)
        self._logger = logging.getLogger(__name__)

    @property
    def internet_home(self):
        return self._internet_home

    @internet_home.setter
    def internet_home(self, df):
        """
        :type df: DataFrame
        """
        assert (isinstance(df, pd.DataFrame))

        self._internet_home = df

    @property
    def total_universe(self):
        return self._total_universe

    @total_universe.setter
    def total_universe(self, val):
        """
        :type val: float
        """
        assert (isinstance(val, float))

        self._total_universe = val

    @property
    def no_fuse_linkage(self):
        return self._no_fuse_linkage

    @no_fuse_linkage.setter
    def no_fuse_linkage(self, df):
        """
        :type df: DataFrame
        """
        assert (isinstance(df, pd.DataFrame))

        self._no_fuse_linkage = df

    @staticmethod
    def create_cc_entries(df):
        """
        :param df:
        :return:
        """

        assert (isinstance(df, pd.DataFrame))
        assert (df.shape[0] > 0)

        demo_df = df.copy()

        indices = df[['gender', 'hispanic', 'race_black', 'spanish_language1', 'age']].isnull().sum(axis=1) > 0

        demo_df = demo_df.loc[~indices, :].reset_index(drop=True)

        demo_df['cc'] = -1

        demo_df['gender'] = demo_df['gender'].astype(np.int64)
        demo_df['hispanic'] = demo_df['hispanic'].astype(np.int64)
        demo_df['race_black'] = demo_df['race_black'].astype(np.int64)
        demo_df['spanish_language1'] = demo_df['spanish_language1'].astype(np.int64)

        age_lower_bound = [2, 6, 13, 18, 25, 35, 45, 55, 65]
        age_upper_bound = [5, 12, 17, 24, 34, 44, 54, 64, 150]

        race_indices = (demo_df['hispanic'] == 2) & (demo_df['race_black'] == 2)

        gender_indices = demo_df['gender'] == 1

        for i in range(9):
            age_indices = (age_lower_bound[i] <= demo_df['age']) & (demo_df['age'] <= age_upper_bound[i])

            demo_df.loc[age_indices & gender_indices & race_indices, 'cc'] = i + 1
            demo_df.loc[age_indices & (~gender_indices) & race_indices, 'cc'] = i + 10

        race_indices = (demo_df['hispanic'] == 1) & (demo_df['spanish_language1'].isin([1, 2, 3]))

        for i in range(9):
            age_indices = (age_lower_bound[i] <= demo_df['age']) & (demo_df['age'] <= age_upper_bound[i])

            demo_df.loc[age_indices & gender_indices & race_indices, 'cc'] = i + 19
            demo_df.loc[age_indices & (~gender_indices) & race_indices, 'cc'] = i + 28

        race_indices = (demo_df['hispanic'] == 1) & (~demo_df['spanish_language1'].isin([1, 2, 3]))

        for i in range(9):
            age_indices = (age_lower_bound[i] <= demo_df['age']) & (demo_df['age'] <= age_upper_bound[i])

            demo_df.loc[age_indices & gender_indices & race_indices, 'cc'] = i + 37
            demo_df.loc[age_indices & (~gender_indices) & race_indices, 'cc'] = i + 46

        race_indices = (demo_df['race_black'] == 1) & (demo_df['hispanic'] == 2)

        for i in range(9):
            age_indices = (age_lower_bound[i] <= demo_df['age']) & (demo_df['age'] <= age_upper_bound[i])

            demo_df.loc[age_indices & gender_indices & race_indices, 'cc'] = i + 55
            demo_df.loc[age_indices & (~gender_indices) & race_indices, 'cc'] = i + 64

        demo_df = demo_df.loc[demo_df['cc'] > 0, :].reset_index(drop=True)

        return demo_df

    @staticmethod
    def setup_dcr_demos():
        pass

    def create_donors_recips(self, npm_df, hhp_nol_df, nol_df, cph_linkage_df,
                           nol_sample_df, top_200_df, top_50_strm_df,
                           nol_category_brand_df, nol_category_channel_df,
                           tv_top_100_df):
        """
        :param npm_df:
        :param hhp_nol_df:
        :param nol_df:
        :param cph_linkage_df:
        :param nol_sample_df:
        :param top_200_df:
        :param top_50_strm_df:
        :param nol_category_brand_df:
        :param nol_category_channel_df:
        :param tv_top_100_df:
        :return:
        """

        assert (isinstance(nol_df, pd.DataFrame))
        assert (isinstance(cph_linkage_df, pd.DataFrame))
        assert (isinstance(hhp_nol_df, pd.DataFrame))
        assert (isinstance(npm_df, pd.DataFrame))
        assert (isinstance(nol_sample_df, pd.DataFrame))
        assert (isinstance(nol_category_brand_df, pd.DataFrame))
        assert (isinstance(nol_category_channel_df, pd.DataFrame))
        assert (isinstance(tv_top_100_df, pd.DataFrame))
        assert (isinstance(top_200_df, pd.DataFrame))
        assert (isinstance(top_50_strm_df, pd.DataFrame))

        logger = self._logger

        logger.info('Getting Online Weights for NPM Respondents')

        npm_join_hhp = pd.merge(pd.merge(npm_df, hhp_nol_df[['respondentid', 'rn_id']],
                                         on='respondentid', how='left').fillna(-1.0),
                                nol_df[['rn_id', 'weight']].rename(columns={'weight': 'onl_weight'}),
                                on='rn_id', how ='left')

        npm_hhp_cc = self.create_cc_entries(npm_join_hhp)

        nol_cc_df = self.create_cc_entries(nol_df)

        logger.info('Creating donor and recipient datasets')

        pIh_donors_df, pIh_recips_df, internet_home, tot_uni, \
        linkage_ih_id_nofuse = buildRecipDonorIHDataset(cph_linkage_df,
                                                        npm_df, hhp_nol_df,
                                                        npm_hhp_cc, nol_cc_df,
                                                        nol_sample_df,
                                                        top_200_df, top_50_strm_df,
                                                        nol_category_brand_df,
                                                        nol_category_channel_df,
                                                        tv_top_100_df,
                                                        self.pivot_table_dataframe)

        self.total_universe = tot_uni
        self.internet_home = internet_home
        self.no_fuse_linkage = linkage_ih_id_nofuse

        self.donors = pIh_donors_df
        self.recips = pIh_recips_df

        return self

    def create_soft_cal_targets(self, npm_df, dcr_pc_df, dcr_pcv_df,
                                dcr_combine_targets_pc, dcr_combine_targets_pcv, dcr_target,
                                dcr_target_sub, dcr_vid, impressions_home, cph_donors):
        """
        :param npm_df:
        :param dcr_pc_df:
        :param dcr_pcv_df:
        :param dcr_combine_targets_pc:
        :param dcr_combine_targets_pcv:
        :param dcr_target:
        :param dcr_target_sub:
        :param dcr_vid:
        :param impressions_home:
        :param cph_donors:
        :return:
        """

        assert (isinstance(dcr_pc_df, pd.DataFrame))
        assert (isinstance(dcr_pcv_df, pd.DataFrame))
        assert (isinstance(dcr_combine_targets_pc, pd.DataFrame))
        assert (isinstance(npm_df, pd.DataFrame))
        assert (isinstance(dcr_combine_targets_pcv, pd.DataFrame))
        assert (isinstance(dcr_target, pd.DataFrame))
        assert (isinstance(dcr_target_sub, pd.DataFrame))
        assert (isinstance(dcr_vid, pd.DataFrame))
        assert (isinstance(impressions_home, pd.DataFrame))
        assert (isinstance(cph_donors, pd.DataFrame))

        logger = self._logger

        internet_home = self.internet_home

        # add cc values to full npm dataset
        npm_cc_df = self.create_cc_entries(npm_df)

        dcr_pc_combine_targets_temp = dcr_combine_targets_pc.rename(columns={'home_prop_reach': 'prop_reach',
                                                                             'home_prop_dur': 'prop_dur',
                                                                             'home_prop_pv': 'prop_pv'})

        dcr_pcv_combine_targets_temp = dcr_combine_targets_pcv.rename(columns={'home_prop_reach': 'prop_reach',
                                                                               'home_prop_dur': 'prop_dur'})

        logger.info('Creating DCR Targets')

        dcr_targets = self.create_dcr_targets(dcr_pc_df, dcr_pcv_df, dcr_pc_combine_targets_temp,
                                              dcr_pcv_combine_targets_temp, self.total_universe, dcr_target,
                                              dcr_target_sub, dcr_vid)

        indices = cph_donors['respondentid'].isin(npm_df['respondentid'])

        nol_cph_weights = pd.merge(
            internet_home.assign(rn_id=internet_home['new_rn_id'] - 99)[['cc', 'rn_id', 'weight']],
            cph_donors.loc[indices, ['rn_id', 'respondentid']], on='rn_id', how='left') \
            .rename(columns={'respondentid': 'npm_id', 'rn_id': 'rn_id_home'})

        new_internet_home = pd.concat(
            [pd.merge(internet_home.loc[internet_home['CPH'] == 1, ['rn_id', 'new_rn_id', 'weight']],
                      self.no_fuse_linkage[['recipientid', 'donorid']].rename(
                          columns={'recipientid': 'rn_id', 'donorid': 'cc'})).drop('rn_id', axis=1)\
                 .rename(columns={'new_rn_id': 'rn_id'}), internet_home.loc[internet_home['CPH'] == 0,
                                                                            ['new_rn_id', 'cc', 'weight']]\
                 .rename(columns={'new_rn_id': 'rn_id'})], ignore_index=True)

        cc_cph_rollup = nol_cph_weights.groupby('cc')['weight'].sum().reset_index()

        cc_npm_rollup = npm_cc_df.groupby('cc')['weight'].sum().reset_index().rename(columns={'weight': 'old_target'})

        cc_rollup = pd.merge(cc_npm_rollup, cc_cph_rollup, on='cc', how='left')

        cc_rollup['weight'] = cc_rollup['weight'].fillna(0)

        cc_rollup['target'] = cc_rollup['old_target'] - cc_rollup['weight']

        cc_targets = pd.concat([cc_rollup.rename(columns={'cc': 'code', 'target': 'reach'}),
                                nol_cph_weights[['npm_id', 'weight']].rename(
                                    columns={'npm_id': 'code', 'weight': 'reach'})], ignore_index=True)

        cc_targets['var'] = 'cc'
        cc_targets['imp'] = 0.0

        softcal_targets = pd.concat([dcr_targets, cc_targets], ignore_index=True)

        softcal_resp = pd.merge(new_internet_home[['rn_id', 'cc', 'weight']], pd.merge(
            internet_home.drop('rn_id', axis=1).rename(columns={'new_rn_id': 'rn_id'})['rn_id'].to_frame(),
            impressions_home, on='rn_id', how='left').fillna(0.0),
                                on='rn_id').rename(
            columns={'rn_id': 'respondentid'})

        self.soft_cal_targets = softcal_targets
        self.soft_cal_impress = softcal_resp

        return self

    def update_donor_recips(self, prev_donors, prev_recips, prev_linkages):
        """
        :param prev_donors:
        :param prev_recips:
        :param prev_linkages:
        :return:
        """

        self.determine_old_linkage(prev_donors, prev_recips, prev_linkages)

        donors = self.donors
        recips = self.recips

        old_links = self.old_links

        compare_link = pd.merge(old_links,
                                 prev_donors[['respondentid', 'weight']].rename(columns={'respondentid': 'donorid',
                                                                                    'weight': 'prev_npm_weight'}),
                                 on = 'donorid')

        compare_link = pd.merge(compare_link,
                                donors[['respondentid', 'weight']].rename(columns={'respondentid': 'donorid',
                                                                                    'weight': 'cur_npm_weight'}),
                                 on='donorid')

        compare_link = pd.merge(compare_link,
                                prev_recips[['respondentid', 'weight']].rename(columns={'respondentid': 'recipientid',
                                                                                   'weight': 'prev_nol_weight'}),
                                on='recipientid')

        compare_link = pd.merge(compare_link,
                                recips[['respondentid', 'weight']].rename(columns={'respondentid': 'recipientid',
                                                                                   'weight': 'cur_nol_weight'}),
                                on='recipientid')

        compare_link['flag_first'] = (compare_link.assign( random_num=lambda x : 1)\
                                       .groupby('donorid')['random_num'].cumsum() == 1).astype(int)

        compare_link['flag_last'] = (compare_link.assign(random_num=lambda x: 1).groupby('donorid')['random_num'] \
                                     .apply(lambda x: x.shape[0] - x.cumsum()) == 0).astype(int)

        indices = compare_link['flag_first'] == 1


        ##TODO
        # Consider the rest of the determined new or changed script

        return self