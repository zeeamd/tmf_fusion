# -*- coding: utf-8 -*-
"""
Created on Fri Aug 10 12:48:15 2018

@author: tued7001
"""
#python standard library
import logging
from functools import reduce
from gc import collect

#data management
import numpy as np
import pandas as pd
import dask.dataframe as dd

#in-house methods
from library.src.base.preprocess import Preprocess
from .donors_recips import donors_recips_pandas, donors_recips_dask

class CPHPreprocess(Preprocess):

    def __init__(self):

        Preprocess.__init__(self, 4, 20, True)
        self._logger = logging.getLogger(__name__)

    @property
    def top_100_tv_network_dp(self):
        return self._top_100_tv_network_dp

    @top_100_tv_network_dp.setter
    def top_100_tv_network_dp(self, val_dict):

        assert(isinstance(val_dict, dict))

        self._top_100_tv_network_dp=val_dict

    @property
    def prev_cc_master_donor(self):
        return self._prev_cc_master_donor

    @prev_cc_master_donor.setter
    def prev_cc_master_donor(self, df):
        assert isinstance(df, pd.DataFrame)
        self._prev_cc_master_donor=df

    @property
    def prev_cc_master_recip(self):
        return self._prev_cc_master_recip

    @prev_cc_master_donor.setter
    def prev_cc_master_recip(self, df):
        assert isinstance(df, pd.DataFrame)
        self._prev_cc_master_recip=df


    @property
    def dcr_targets_dict( self ):
        return self._dcr_targ_dict
    
    @dcr_targets_dict.setter
    def dcr_targets_dict( self, df_dict):

        assert isinstance(df_dict, dict)

        self.dcr_targ_dict=df_dict
        
    @property
    def impressions_home( self ):
        return self.impress_home

    @impressions_home.setter
    def impressions_home(self, df):
        self.impress_home=df
    
    @property
    def impressions_work( self ):
        return self.impress_work

    @impressions_work.setter
    def impressions_work(self, df):
        self.impress_work=df

    @staticmethod
    def create_cc_entries( df ):
        """
        :param df:
        :return:
        """
 
        assert(isinstance(df, pd.DataFrame))
        assert( df.shape[0] > 0 )

        demo_df=df.copy()

        indices=df[['gender', 'hispanic', 'race_black', 'spanish_language1', 'age']].isnull().sum(axis=1) > 0
        
        demo_df = demo_df.loc[~indices, :].reset_index(drop=True)

        demo_df['cc'] = -1
        
        demo_df['gender']=demo_df['gender'].astype(np.int64)
        demo_df['hispanic']=demo_df['hispanic'].astype(np.int64)
        demo_df['race_black']=demo_df['race_black'].astype(np.int64)
        demo_df['spanish_language1']=demo_df['spanish_language1'].astype(np.int64)
        
        age_lower_bound=[2, 6, 13, 18, 25, 35, 45, 55, 65]
        age_upper_bound=[5, 12, 17, 24, 34, 44, 54, 64, 150]
    
        race_indices = (demo_df['hispanic'] == 2) & (demo_df['race_black'] == 2)

        gender_indices = demo_df['gender'] == 1

        for i in range(9):
            age_indices = (age_lower_bound[i] <= demo_df['age']) & (demo_df['age'] <= age_upper_bound[i])

            demo_df.loc[age_indices & gender_indices & race_indices, 'cc']=i + 1
            demo_df.loc[age_indices & (~gender_indices) & race_indices, 'cc']=i + 10

        race_indices=(demo_df['hispanic'] == 1) & (demo_df['spanish_language1'].isin([1, 2, 3]))
        
        for i in range(9):
            age_indices=(age_lower_bound[i] <= demo_df['age']) & (demo_df['age'] <= age_upper_bound[i])

            demo_df.loc[age_indices & gender_indices & race_indices, 'cc']=i + 19
            demo_df.loc[age_indices & (~gender_indices) & race_indices, 'cc']=i + 28
        
        race_indices=(demo_df['hispanic'] == 1) & (~demo_df['spanish_language1'].isin([1, 2, 3]))
        
        for i in range(9):
            age_indices=(age_lower_bound[i] <= demo_df['age']) & (demo_df['age'] <= age_upper_bound[i])

            demo_df.loc[age_indices & gender_indices & race_indices, 'cc']=i + 37
            demo_df.loc[age_indices & (~gender_indices) & race_indices, 'cc']=i + 46
        
        race_indices=(demo_df['race_black'] == 1) & (demo_df['hispanic'] == 2)
        
        for i in range(9):
            age_indices=(age_lower_bound[i] <= demo_df['age']) & (demo_df['age'] <= age_upper_bound[i])

            demo_df.loc[age_indices & gender_indices & race_indices, 'cc']=i + 55
            demo_df.loc[age_indices & (~gender_indices) & race_indices, 'cc']=i + 64

        demo_df = demo_df.loc[demo_df['cc'] > 0, :].reset_index(drop=True)
        
        return demo_df

    @staticmethod
    def setup_dcr_demos():
        data=[['0', 'demo_total', '1'], ['1', 'demo_p0212', '0'], ['1', 'demo_p1399', '1'],
                ['2', 'demo_p0217', '0'], ['2', 'demo_p1899', '1'], ['3', 'demo_p0212', '0'],
                ['3', 'demo_m1399', '1'], ['3', 'demo_f1399', '1'], ['4', 'demo_p0217', '0'],
                ['4', 'demo_m1899', '1'], ['4', 'demo_f1899', '1'], ['5', 'demo_p0217', '0'],
                ['5', 'demo_m1844', '1'], ['5', 'demo_m4599', '1'], ['5', 'demo_f1844', '1'],
                ['5', 'demo_f4599', '1'], ['6', 'demo_p0217', '0'], ['6', 'demo_m1834', '1'],
                ['6', 'demo_m3554', '1'], ['6', 'demo_m5599', '1'], ['6', 'demo_f1834', '1'],
                ['6', 'demo_f3554', '1'], ['6', 'demo_f5599', '1'], ['7', 'demo_p0217', '0'],
                ['7', 'demo_m1824', '1'], ['7', 'demo_m2534', '1'], ['7', 'demo_m3544', '1'],
                ['7', 'demo_m4554', '1'], ['7', 'demo_m5564', '1'], ['7', 'demo_m6599', '1'],
                ['7', 'demo_f1824', '1'], ['7', 'demo_f2534', '1'], ['7', 'demo_f4554', '1'],
                ['7', 'demo_f5564', '1'], ['7', 'demo_f6599', '1'], ['8', 'demo_p0212', '1'],
                ['8', 'demo_m1317', '1'], ['8', 'demo_m1820', '1'], ['8', 'demo_m2124', '1'],
                ['8', 'demo_m2529', '1'], ['8', 'demo_m3034', '1'], ['8', 'demo_m3539', '1'],
                ['8', 'demo_m4044', '1'], ['8', 'demo_m4549', '1'], ['8', 'demo_m5054', '1'],
                ['8', 'demo_m5564', '1'], ['8', 'demo_m6599', '1'], ['8', 'demo_f1317', '1'],
                ['8', 'demo_f1820', '1'], ['8', 'demo_f2124', '1'], ['8', 'demo_f2529', '1'],
                ['8', 'demo_f3034', '1'], ['8', 'demo_f3539', '1'], ['8', 'demo_f4044', '1'],
                ['8', 'demo_f4549', '1'], ['8', 'demo_f5054', '1'], ['8', 'demo_f5564', '1'],
                ['8', 'demo_f6599', '1']]

        df=pd.DataFrame(data=data, columns=['level', 'name', 'check'])

        df['level']=df['level'].astype(int)
        df['check']=df['check'].astype(int)

        return df

    def _create_internet_data(self, nol_demo, cph_daily_weights_df):
        """
        :param nol_demo:
        :param cph_daily_weights_df:
        :return:
        """

        cph_donors_df=self.donors

        nol_demo['surf_location_id']=nol_demo['surf_location_id'].astype(int)
        nol_demo['gender_id']=nol_demo['gender_id'].astype(int)
        nol_demo['hispanic_origin_id']=nol_demo['hispanic_origin_id'].astype(int)
        nol_demo['race_id']=nol_demo['race_id'].astype(int)
        cph_daily_weights_df['respondentid']=cph_daily_weights_df['respondentid'].astype(int)
        cph_daily_weights_df['weight']=cph_daily_weights_df['weight'].astype(float)

        indices=nol_demo['surf_location_id'] == 1

        #internet home data
        internet_home=nol_demo.loc[indices, ['rn_id', 'gender_id', 'age', 'hispanic_origin_id',
                              'spanish_lang_dominance', 'race_id', 'weight']].reset_index(drop=True)

        internet_home=internet_home.rename( columns={ 'gender_id' : 'gender',
                                                     'spanish_lang_dominance' : 'spanish_language1'})

        internet_home['hispanic']=internet_home['hispanic_origin_id'].isin([1,-1]).astype(int) + 1
        internet_home['race_black']=internet_home['race_id'].apply(lambda x : x is not 2 ).astype(int) + 1

        #we need to do pass the function as input
        internet_home=self.create_cc_entries(internet_home)

        #internet work data
        internet_work=nol_demo.loc[nol_demo['surf_location_id'] == 2,
                                 ['rn_id', 'gender_id', 'age', 'hispanic_origin_id',
                              'spanish_lang_dominance', 'race_id', 'weight'] ].reset_index(drop=True)

        internet_work=internet_work.rename( columns={ 'gender_id' : 'gender',
                                                     'spanish_lang_dominance' : 'spanish_language1'})

        internet_work['hispanic']=internet_work['hispanic_origin_id'].isin([1,-1]).astype(int) + 1
        internet_work['race_black']=internet_work['race_id'].apply(lambda x : x is not 2 ).astype(int) + 1

        #this assumes we still perform cc aggregations
        internet_work=self.create_cc_entries(internet_work)

        #internet CPH data
        #full outer join creates missing values in rn_id
        internet_cph=pd.merge(cph_donors_df[['respondentid', 'rn_id', 'cc']], cph_daily_weights_df[['respondentid', 'weight']],
                                on='respondentid', how='left').fillna(0)

        #we find the adjustment to make between internet home and cph home
        ih_cc=internet_home.groupby('cc')['weight'].sum()
        cph_cc=internet_cph.groupby('cc')['weight'].sum()

        cph_adj=ih_cc.to_frame().join(cph_cc.to_frame(), lsuffix='_l', rsuffix='_r')

        cph_adj['adj']=cph_adj['weight_l'].astype(float) / cph_adj['weight_r'].astype(float)

        internet_cph=internet_cph.set_index('cc').join(cph_adj['adj'].to_frame(),
                                          how='inner' ).reset_index()

        internet_cph['weight']=internet_cph['weight'].astype(float) * internet_cph['adj']
        internet_cph.drop('adj', axis=1, inplace=True)

        return internet_home, internet_work, internet_cph

    def create_dcr_respondents_by_type(self, dcr_df, aggr_df, nol_df, internet_cph,  internet_home, internet_work,
                                       site_sfx, min_cnt_dcr_h=5, min_cnt_dcr_w=5,  min_lvl_cnt_c=10):
        """
        :param dcr_df:
        :param aggr_df:
        :param nol_df:
        :param internet_cph:
        :param internet_home:
        :param internet_work:
        :param site_sfx:
        :param min_cnt_dcr_h:
        :param min_cnt_dcr_w:
        :param min_lvl_cnt_c:
        :return:
        """
        logger = self._logger

        cph_media_space_df = pd.merge(internet_cph[['rn_id', 'weight']], nol_df, on='rn_id')
        home_media_space_df = pd.merge(internet_home[['rn_id', 'weight']], nol_df, on='rn_id')
        work_media_space_df = pd.merge(internet_work[['rn_id', 'weight']], nol_df, on='rn_id')

        if site_sfx in ['vd', 'br']:
            aggr_df = aggr_df.rename(columns={'brandid': 'site_id'})
        else:
            aggr_df = aggr_df.rename(columns={'channelid': 'site_id'})

        if site_sfx in ['vd', 'sv']:
            aggr_df['minutes'] = aggr_df['viewduration'] / 60.0
            dcr_cols = ['rn_id', 'site_id', 'minutes']
            filter_col = 'minutes'
        else:
            dcr_cols = ['rn_id', 'site_id', 'pageviews', 'viewduration']
            filter_col = 'pageviews'

        dcr_df['site_id'] = dcr_df['site_id'].astype(int)
        aggr_df['site_id'] = aggr_df['site_id'].astype(int)

        site_lists = list(set(dcr_df['site_id'].unique().tolist()) & set(aggr_df['site_id'].unique().tolist()))

        dcr_imps_cph = pd.DataFrame(columns=['rn_id'])
        dcr_imps_h = pd.DataFrame(columns=['rn_id'])
        dcr_imps_w = pd.DataFrame(columns=['rn_id'])
        level_c = pd.DataFrame()
        target_d = pd.DataFrame()

        for site in site_lists:
            dcr_site = dcr_df.loc[dcr_df['site_id'] == site, :].reset_index(drop=True)
            aggr_site = aggr_df.loc[aggr_df['site_id'] == site, :].reset_index(drop=True)
            dcr_site_name = dcr_site.loc[0, 'name']

            logger.info('Creating impressions for site: {}'.format(dcr_site_name))

            if aggr_site['surf_location_id'].nunique() < 2:
                logger.warning('Not enough respondent data')
                pass

            else:
                aggr_surf_tots = aggr_df['surf_location_id'].value_counts()

                if (aggr_surf_tots.loc[1] > min_cnt_dcr_h) & (aggr_surf_tots.loc[2] > min_cnt_dcr_w):

                    dcr_cph_usage_df = pd.merge(cph_media_space_df, aggr_site[dcr_cols], on='rn_id')
                    dcr_home_usage_df = pd.merge(home_media_space_df, aggr_site[dcr_cols], on='rn_id')
                    dcr_work_usage_df = pd.merge(work_media_space_df, aggr_site[dcr_cols], on='rn_id')

                    home_aggr_df = dcr_home_usage_df.loc[dcr_home_usage_df[filter_col] > 0, :].reset_index(drop=True)
                    work_aggr_df = dcr_work_usage_df.loc[dcr_work_usage_df[filter_col] > 0, :].reset_index(drop=True)

                    target_d_site = dcr_site.rename(columns={'computer': 'pc_target_imp',
                                                             'unique_audience': 'pc_target_reach',
                                                             'duration': 'pc_target_dur'})

                    if site_sfx in ['vd', 'sv']:
                        target_d_site['nv_h_reach'] = home_aggr_df['weight'].sum()
                        target_d_site['nv_h_dur'] = home_aggr_df[['weight', 'minutes']].prod(axis=1).sum()
                        target_d_site['nv_w_reach'] = work_aggr_df['weight'].sum()
                        target_d_site['nv_w_dur'] = work_aggr_df[['weight', 'minutes']].prod(axis=1).sum()

                        target_d_site['nv_panel_dur'] = target_d_site['nv_h_dur'] + target_d_site['nv_w_dur']
                        target_d_site['dcr_dur_adj'] = target_d_site['pc_target_dur'] / target_d_site['nv_panel_dur']
                        target_d_site['home_prop_reach'] = target_d_site['nv_h_reach'] / (
                                target_d_site['nv_h_reach'] + target_d_site['nv_w_reach'])
                        target_d_site['home_prop_dur'] = target_d_site['nv_h_dur'] / (target_d_site['nv_w_dur'] +
                                                                                      target_d_site['nv_h_dur'])

                        target_d_site = target_d_site.rename(columns={'nv_h_reach': 'nv_home_panel',
                                                            'nv_w_reach': 'nv_work_panel'})

                        target_d_site = target_d_site[['name', 'site_id', 'pc_target_imp', 'pc_target_reach',
                                             'pc_target_dur', 'nv_home_panel', 'nv_work_panel', 'nv_panel_dur',
                                             'dcr_dur_adj', 'home_prop_reach', 'home_prop_dur']]

                    else:
                        target_d_site['nv_h_reach']=home_aggr_df['weight'].sum()
                        target_d_site['nv_h_pv']=home_aggr_df[['weight', 'pageviews']].prod(axis=1).sum()
                        target_d_site['nv_h_dur']=home_aggr_df[['weight', 'viewduration']].prod(axis=1).sum()
                        target_d_site['nv_w_reach']=work_aggr_df['weight'].sum()
                        target_d_site['nv_w_pv']=work_aggr_df[['weight', 'pageviews']].prod(axis=1).sum()
                        target_d_site['nv_w_dur']=work_aggr_df[['weight', 'viewduration']].prod(axis=1).sum()

                        target_d_site['nv_pc_panel']=target_d_site['nv_h_pv'] + target_d_site['nv_w_pv']
                        target_d_site['nv_pc_panel_dur']=target_d_site['nv_h_dur'] + target_d_site['nv_w_dur']
                        target_d_site['dcr_pv_adj']=target_d_site['pc_target_imp'] / target_d_site['nv_pc_panel']
                        target_d_site['dcr_dur_adj']=target_d_site['pc_target_dur'] / target_d_site['nv_pc_panel_dur']
                        target_d_site['home_prop_pv']=target_d_site['nv_h_pv'] / (target_d_site['nv_h_pv'] +
                                                                                    target_d_site['nv_w_pv'])
                        target_d_site['home_prop_dur']=target_d_site['nv_h_dur'] / (target_d_site['nv_w_dur'] +
                                                                                      target_d_site['nv_h_dur'])
                        target_d_site['home_prop_reach']=target_d_site['nv_h_reach'] / (
                                target_d_site['nv_w_reach'] + target_d_site['nv_h_reach'])

                        target_d_site = target_d_site.rename(columns={'nv_h_pv': 'nv_pc_h_reach',
                                                                      'nv_w_pv': 'nv_pc_w_reach'})

                        target_d_site = target_d_site[['name', 'site_id', 'pc_target_imp', 'pc_target_reach',
                                             'pc_target_dur', 'nv_pc_h_reach', 'nv_pc_w_reach', 'nv_pc_panel',
                                             'nv_pc_panel_dur', 'dcr_pv_adj', 'dcr_dur_adj', 'home_prop_pv',
                                             'home_prop_dur', 'home_prop_reach']]

                    dcr_cph, use_level = self.create_dcr_impressions_by_site_id(dcr_cph_usage_df, dcr_site_name,
                                                                                'pc', min_lvl_cnt_c, site_sfx)

                    dcr_home = self.create_dcr_impressions_by_site_id(home_aggr_df, dcr_site_name, 'pc',
                                                                      min_lvl_cnt_c, site_sfx, use_level=use_level)

                    dcr_work = self.create_dcr_impressions_by_site_id(work_aggr_df, dcr_site_name, 'pc',
                                                                      min_lvl_cnt_c, site_sfx, use_level=use_level)

                    dcr_imps_cph = pd.merge(dcr_imps_cph, dcr_cph, on='rn_id', how='outer').fillna(0.0)
                    dcr_imps_h = pd.merge(dcr_imps_h, dcr_home, on='rn_id', how='outer').fillna(0.0)
                    dcr_imps_w = pd.merge(dcr_imps_w, dcr_work, on='rn_id', how='outer').fillna(0.0)
                    level_c_site = pd.DataFrame(data=[use_level], columns=['use_level'])
                    level_c_site['name'] = dcr_site['name']
                    level_c_site['site_id'] = site

                    target_d = pd.concat([target_d_site, target_d], sort=True, ignore_index=True, axis=0)
                    level_c = pd.concat([level_c_site, level_c], sort=True, ignore_index=True, axis=0)

                else:
                    pass

        return dcr_imps_cph, dcr_imps_h, dcr_imps_w, level_c, target_d

    def create_dcr_respondent(self, internet_home, internet_work, internet_cph, dcr_brands_pc_text, dcr_brands_video,
                               dcr_sub_brands_pc_text, dcr_sub_brands_video, strm_aggr_brand, strm_aggr_sub_brand,
                               surf_aggr_brand, surf_aggr_sub_brand, nol_age_gender_df):
        """
        :param internet_home:
        :param internet_work:
        :param internet_cph:
        :param dcr_brands_pc_text:
        :param dcr_brands_video:
        :param dcr_sub_brands_pc_text:
        :param dcr_sub_brands_video:
        :param strm_aggr_brand:
        :param strm_aggr_sub_brand:
        :param surf_aggr_brand:
        :param surf_aggr_sub_brand:
        :param nol_age_gender_df:
        :return:
        """

        logger=logging.getLogger(__name__)

        logger.info('Building DCR impressions')

        logger.info('Build DCR PC By Brand')

        pc_br_imps_cph, pc_br_imps_h, pc_br_imps_w, \
        pc_level_br, pc_target_d_br = self.create_dcr_respondents_by_type(dcr_brands_pc_text, surf_aggr_brand,
                                                                          nol_age_gender_df, internet_cph,
                                                                          internet_home, internet_work, 'br')
        collect()

        logger.info('Build DCR PC By SubBrand')

        pc_sb_imps_cph, pc_sb_imps_h, pc_sb_imps_w, \
        pc_level_sb, pc_target_d_sb=self.create_dcr_respondents_by_type(dcr_sub_brands_pc_text, surf_aggr_sub_brand,
                                                                          nol_age_gender_df, internet_cph,
                                                                          internet_home, internet_work, 'sb')

        collect()

        logger.info('Build DCR PCV By Brand')

        pcv_br_imps_cph, pcv_br_imps_h, pcv_br_imps_w, \
        pcv_level_br, pcv_target_d_br=self.create_dcr_respondents_by_type(dcr_brands_video, strm_aggr_brand,
                                                                            nol_age_gender_df, internet_cph,
                                                                            internet_home, internet_work, 'vd')

        collect()

        logger.info('Build DCR PCV By SubBrand')

        pcv_sb_imps_cph, pcv_sb_imps_h, pcv_sb_imps_w, \
        pcv_level_sb, pcv_target_d_sb=self.create_dcr_respondents_by_type(dcr_sub_brands_video, strm_aggr_sub_brand,
                                                                            nol_age_gender_df, internet_cph,
                                                                            internet_home, internet_work, 'sv')

        collect()

        logger.info('Merging the impressions together')

        imps_list=[pc_br_imps_cph, pc_sb_imps_cph, pcv_br_imps_cph, pcv_sb_imps_cph]

        ih_imps_list=[pc_br_imps_h, pc_sb_imps_h, pcv_br_imps_h, pcv_sb_imps_h]

        iw_imps_list=[pc_br_imps_w, pc_sb_imps_w, pcv_br_imps_w, pcv_sb_imps_w]

        impressions=reduce( lambda left, right : pd.merge(left, right, on='rn_id', how='outer'),
                              imps_list ).fillna(0.0)

        impressions_home=reduce( lambda left, right : pd.merge(left, right, on='rn_id', how='outer'),
                             ih_imps_list ).fillna(0.0)

        impressions_work=reduce( lambda left, right : pd.merge(left, right, on='rn_id', how='outer'),
                             iw_imps_list ).fillna(0.0)

        logger.info('Setting up the DCR combined targets for PC and Video')

        levels_use_pc=pd.concat([pc_level_br,pc_level_sb], axis=0, sort=True).reset_index(drop=True)
        levels_use_pcv=pd.concat([pcv_level_br, pcv_level_sb], axis=0, sort=True).reset_index(drop=True)

        dcr_combine_targets_pc=pd.concat([pc_target_d_br, pc_target_d_sb], axis=0, sort=True).reset_index(drop=True)
        dcr_combine_targets_pcv=pd.concat([pcv_target_d_br, pcv_target_d_sb], axis=0, sort=True).reset_index(drop=True)

        max_dcr_adj=4
        min_dcr_adj=0.5

        indices=dcr_combine_targets_pc['dcr_pv_adj'] > max_dcr_adj

        dcr_combine_targets_pc.loc[indices, 'dcr_pv_adj']=max_dcr_adj
        dcr_combine_targets_pc.loc[indices, 'pc_target_imp']=dcr_combine_targets_pc.loc[indices, 'nv_pc_panel']*max_dcr_adj

        indices=dcr_combine_targets_pc['dcr_dur_adj'] > max_dcr_adj

        dcr_combine_targets_pc.loc[indices, 'dcr_dur_adj']=max_dcr_adj
        dcr_combine_targets_pc.loc[indices, 'pc_target_dur']=dcr_combine_targets_pc.loc[indices, 'nv_pc_panel_dur']*max_dcr_adj

        indices=(dcr_combine_targets_pc['dcr_dur_adj'] > min_dcr_adj) & (dcr_combine_targets_pc['dcr_pv_adj'] > min_dcr_adj)

        dcr_pc_df=pd.merge( pd.concat([dcr_brands_pc_text, dcr_sub_brands_pc_text], axis=0, sort=True)\
                                          .reset_index(drop=True),
                pd.merge(dcr_combine_targets_pc.loc[indices, :]\
                         .drop('name', axis=1).reset_index(drop=True),
                         levels_use_pc, on='site_id').drop('name', axis=1),
                         on='site_id' )

        indices=dcr_combine_targets_pcv['dcr_dur_adj'] > max_dcr_adj

        dcr_combine_targets_pcv.loc[indices, 'dcr_dur_adj']=max_dcr_adj
        dcr_combine_targets_pcv.loc[indices, 'pc_target_dur']=dcr_combine_targets_pcv.loc[indices, 'nv_panel_dur'] * max_dcr_adj

        indices=dcr_combine_targets_pcv['dcr_dur_adj'] > min_dcr_adj

        dcr_pcv_df=pd.merge( pd.concat( [dcr_brands_video, dcr_sub_brands_video], axis=0, sort=True)\
                                          .reset_index(drop=True),
                pd.merge(dcr_combine_targets_pcv.loc[indices, :]\
                         .drop('name', axis=1).reset_index(drop=True),
                         levels_use_pcv, on='site_id').drop('name', axis=1),
                         on='site_id' )

        dcr_pc_df=dcr_pc_df.rename( columns={'name' : 'site'})
        dcr_pcv_df=dcr_pcv_df.rename( columns={'name' : 'site'})

        logger.info('Sample size of impressions: {}'.format( impressions.shape[0]))

        softcal_reps=pd.merge( internet_cph[['respondentid', 'rn_id', 'cc', 'weight']], impressions,
                                        on='rn_id', how='left').fillna(0.0)

        indices=softcal_reps['weight'] > 0.0

        softcal_reps=softcal_reps.loc[indices, :].reset_index(drop=True)

        softcal_reps['respondentid']=softcal_reps['respondentid'].apply(lambda x : 'r' + str(int(x)) )
        softcal_reps.drop('rn_id', axis=1, inplace=True)

        softcal_reps=softcal_reps.fillna(0.0)

        softcal_reps['cc']=softcal_reps['cc'].astype(int)

        cols_keep=[col for col in softcal_reps.columns if 'p02' not in col]

        softcal_reps=softcal_reps.loc[:, cols_keep]

        return softcal_reps, impressions_home, impressions_work, dcr_pc_df, dcr_pcv_df, dcr_combine_targets_pc, \
               dcr_combine_targets_pcv

    def create_donors_recips(self, npm_df, nol_df, hhp_nol_df, tv_view_df, top_200_df):
        """
        :param npm_df:
        :param nol_df:
        :param hhp_nol_df:
        :param tv_view_df:
        :param top_200_df:
        :return:
        """

        assert isinstance(npm_df, pd.DataFrame) | isinstance(npm_df, dd.DataFrame)
        assert isinstance(nol_df, pd.DataFrame) | isinstance(nol_df, dd.DataFrame)
        assert isinstance(tv_view_df, pd.DataFrame) | isinstance(tv_view_df, dd.DataFrame)
        assert isinstance(top_200_df, pd.DataFrame) | isinstance(top_200_df, dd.DataFrame)

        cph_logger = self._logger

        if (isinstance(npm_df, dd.DataFrame) & isinstance(nol_df, dd.DataFrame)
                & isinstance(tv_view_df, dd.DataFrame) & isinstance(top_200_df, dd.DataFrame)):

            cph_logger.info('Adding critical cell information to NPM Sample')

            npm_cc_df = npm_df.map_partitions.apply(lambda df: self.create_cc_entries(df))

            cph_donors_df, cph_recips_df, tv_100, tv_500, tv_top_100_df = donors_recips_dask(npm_cc_df, nol_df,
                                                                                             hhp_nol_df,
                                                                                             tv_view_df, top_200_df)

            self.donors = cph_donors_df.compute()
            self.recips = cph_recips_df.compute()
            self.top_100_tv_network_dp = {'top_100_networks_dp_list': tv_100,
                                          'top_100_networks_dp_df': tv_top_100_df}

        elif (isinstance(npm_df, pd.DataFrame) & isinstance(nol_df, pd.DataFrame)
              & isinstance(tv_view_df, pd.DataFrame) & isinstance(top_200_df, pd.DataFrame)):

            cph_logger.info('Adding critical cell information to NPM Sample')

            npm_cc_df = self.create_cc_entries(npm_df)

            cph_donors_df, cph_recips_df, tv_100, tv_500, tv_top_100_df = donors_recips_pandas(npm_cc_df, nol_df,
                                                                                               hhp_nol_df, tv_view_df,
                                                                                               top_200_df,
                                                                                               self.
                                                                                               pivot_table_dataframe)

            self.donors = cph_donors_df
            self.recips = cph_recips_df
            self.top_100_tv_network_dp = {'top_100_networks_dp_list': tv_100,
                                          'top_100_networks_dp_df': tv_top_100_df}


        else:
            cph_logger.critical('Not all the inputs are the same type')
            cph_logger.critical('Setting all donors and recips as empty')

            self.donors = pd.DataFrame()
            self.recips = pd.DataFrame()
            self.top_100_tv_network_dp = {}

            return self

        cph_logger.info('Setting up linking variables for configuration workbook for PyFusion Application')

        self.linking_variables_pc = tv_500

        self.linking_variables_master = list(map(str.lower, ['Age_bydob', 'Age0', 'Age1', 'cableplus', 'dvr',
                                                             'education7', 'employment1', 'gender', 'HDTV', 'hh_size1',
                                                             'hispanic', 'income9', 'kids_0to5', 'kids_12to17',
                                                             'kids_6to11', 'number_of_tvs1', 'Occupation1', 'Paycable',
                                                             'race_asian', 'race_black', 'RegionB', 'Video_Game',
                                                             'Satellite', 'Spanish_language1']))

        self.linking_variables_master += self.linking_variables_pc

        self.linking_variables_se_init = list(map(str.lower, ['Age1', 'cableplus', 'dvr', 'education7', 'employment1',
                                                              'gender', 'HDTV', 'hh_size1', 'hispanic', 'income9',
                                                              'kids_0to5', 'kids_12to17', 'kids_6to11',
                                                              'number_of_tvs1', 'Occupation1', 'Paycable', 'race_asian',
                                                              'race_black', 'RegionB', 'Video_Game', 'Satellite',
                                                              'Spanish_language1']))

        self.linking_variables_se_final = list(map(str.lower, ['Age1', 'cableplus', 'dvr', 'education7', 'employment1',
                                                               'HDTV', 'hh_size1', 'income9', 'kids_0to5',
                                                               'kids_12to17', 'kids_6to11', 'number_of_tvs1',
                                                               'Occupation1', 'Paycable', 'race_asian', 'RegionB',
                                                               'Video_Game', 'Satellite', 'Spanish_language1']))

        self.linking_variables_pc = ['age0'] + self.linking_variables_se_final

        self.iw_rec_vars = self.top_100_tv_network_dp.get('top_100_networks_dp_list', [])
        self.iw_don_vars = ['_'.join(['parent', str(i)]) for i in range(1, 101)]

        return self
    
    def create_soft_cal_targets(self, nol_df, cph_daily_weights, dcr_text, dcr_vid, dcr_sub_text, dcr_sub_vid,
                                dcr_target, dcr_target_sub, dcr_vid_target, strm_brand_df, strm_sub_brand_df,
                                surf_brand_df, surf_sub_brand_df):
        """
        :param nol_df:
        :param cph_daily_weights:
        :param dcr_text:
        :param dcr_vid:
        :param dcr_sub_text:
        :param dcr_sub_vid:
        :param strm_brand_df:
        :param strm_sub_brand_df:
        :param surf_brand_df:
        :param surf_sub_brand_df:
        :return:
        """
        cph_donors_df = self.donors
        
        assert(isinstance(nol_df, pd.DataFrame))
        assert(isinstance(cph_daily_weights, pd.DataFrame))
        assert(isinstance(dcr_text, pd.DataFrame))
        assert(isinstance(dcr_vid, pd.DataFrame))
        assert(isinstance(dcr_sub_text, pd.DataFrame))
        assert(isinstance(dcr_sub_vid, pd.DataFrame))
        assert(isinstance(dcr_target, pd.DataFrame))
        assert(isinstance(dcr_target_sub, pd.DataFrame))
        assert(isinstance(dcr_vid_target, pd.DataFrame))
        assert(isinstance(strm_brand_df, pd.DataFrame))
        assert(isinstance(strm_sub_brand_df, pd.DataFrame))
        assert(isinstance(surf_brand_df, pd.DataFrame))
        assert(isinstance(surf_sub_brand_df, pd.DataFrame))

        cph_logger = self._logger

        strm_brand_df['surf_location_id']=strm_brand_df['surf_location_id'].astype(int)
        strm_sub_brand_df['surf_location_id']=strm_sub_brand_df['surf_location_id'].astype(int)
        surf_brand_df['surf_location_id']=surf_brand_df['surf_location_id'].astype(int)
        surf_sub_brand_df['surf_location_id']=surf_sub_brand_df['surf_location_id'].astype(int)

        strm_brand_df['parentid'] = strm_brand_df['parentid'].astype(int)
        strm_sub_brand_df['parentid'] = strm_sub_brand_df['parentid'].astype(int)
        surf_brand_df['parentid'] = surf_brand_df['parentid'].astype(int)
        surf_sub_brand_df['parentid'] = surf_sub_brand_df['parentid'].astype(int)

        strm_brand_df['brandid'] = strm_brand_df['brandid'].astype(int)
        strm_sub_brand_df['brandid'] = strm_sub_brand_df['brandid'].astype(int)
        surf_brand_df['brandid'] = surf_brand_df['brandid'].astype(int)
        surf_sub_brand_df['brandid'] = surf_sub_brand_df['brandid'].astype(int)

        strm_sub_brand_df['channelid'] = strm_sub_brand_df['channelid'].astype(int)
        surf_sub_brand_df['channelid'] = surf_sub_brand_df['channelid'].astype(int)

        cph_logger.info('Determining internet usage by type and critical cell')

        internet_home, internet_work, internet_cph=self._create_internet_data(nol_df, cph_daily_weights)

        cph_logger.info('Number of donor respondents as soft calibrated candidates: {}'.format( internet_cph.shape[0]) )

        cph_logger.info("Creating age-gender demographics variables")

        nol_age_gender_df = self.create_age_gender_demos(nol_df[['rn_id', 'age', 'gender_id']]\
                                                 .rename(columns={'gender_id': 'gender'}).copy())
    
        cph_logger.info("Determining the soft calibration targets")

        donors=pd.merge(cph_donors_df[['respondentid', 'cc']], internet_cph[['respondentid', 'weight']],
                          on='respondentid', how='left').fillna(0.0)

        cph_logger.info("Creating cc targets")

        cc_targets=donors.groupby('cc')['weight'].sum().reset_index() \
            .rename(columns={'weight': 'reach', 'cc': 'code'})

        cc_targets['var'] = 'cc'
        cc_targets['imp'] = 0

        tot_univ = internet_cph['weight'].sum()

        softcal_resp, impressions_home, impressions_work, dcr_pc_df, dcr_pcv_df, dcr_pc_combine_targets, \
        dcr_pcv_combine_targets = self.create_dcr_respondent(internet_home, internet_work, internet_cph, dcr_text,
                                                              dcr_vid, dcr_sub_text, dcr_sub_vid, strm_brand_df,
                                                              strm_sub_brand_df, surf_brand_df, surf_sub_brand_df,
                                                              nol_age_gender_df)

        cph_logger.info("Determining the DCR targets")

        dcr_pc_combine_targets_temp=dcr_pc_combine_targets.rename(columns={'home_prop_reach': 'prop_reach',
                                                                        'home_prop_dur': 'prop_dur',
                                                                        'home_prop_pv': 'prop_pv'})

        dcr_pcv_combine_targets_temp = dcr_pcv_combine_targets.rename(columns={'home_prop_reach': 'prop_reach',
                                                                          'home_prop_dur': 'prop_dur'})

        dcr_targets = self.create_dcr_targets(dcr_pc_df, dcr_pcv_df, dcr_pc_combine_targets_temp,
                                              dcr_pcv_combine_targets_temp, tot_univ, dcr_target, dcr_target_sub,
                                              dcr_vid_target)

        cols = ['var', 'imp', 'reach', 'code']

        softcal_targ = pd.concat([dcr_targets[cols], cc_targets[cols]],
                                 axis = 0, sort=True).reset_index(drop=True)

        softcal_targ['code'] = softcal_targ['code'].astype(int)

        self.soft_cal_targets = softcal_targ
        self.soft_cal_impress = softcal_resp
        
        self.impressions_home = impressions_home
        self.impressions_work = impressions_work
        
        self.dcr_targets_dict = {'dcr_pc' : dcr_pc_df, 'dcr_pcv' : dcr_pcv_df, 'dcr_combine_pc' : dcr_pc_combine_targets,
                                 'dcr_combine_pcv' : dcr_pcv_combine_targets,  'dcr_target': dcr_targets,
                                 'dcr_target_sub' : dcr_target_sub
                                 }
        
        return self

    def update_donor_recips(self, prev_donors, prev_recips, prev_linkage):
        """
        :param prev_donors:
        :param prev_recips:
        :param prev_linkage:
        :return:
        """
        self.determine_old_linkage(prev_donors, prev_recips, prev_linkage)

        donors = self.donors
        recips = self.recips

        self.prev_cc_master_donor = donors.loc[donors['respondentid'].isin(self.old_links['donorid']),:].reset_index(drop=True)
        self.prev_cc_master_recip = recips.loc[recips['respondentid'].isin(self.old_links['recipientid']), :]\
            .reset_index(drop=True)

        return self