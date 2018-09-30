import dask.dataframe as dd
import pandas as pd
import numpy as np

import logging


def donors_recips_pandas(npm_df, nol_df, hhp_nol_df, tv_view_df, top_200_df, funct_pivot):
    """
    :param npm_df:
    :param nol_df:
    :param hhp_nol_df:
    :param tv_view_df:
    :param top_200_df:
    :param funct_pivot:
    :return:
    """
    logger = logging.getLogger(__name__)

    assert (isinstance(npm_df, pd.DataFrame))
    assert (isinstance(nol_df, pd.DataFrame))
    assert (isinstance(tv_view_df, pd.DataFrame))
    assert (isinstance(top_200_df, pd.DataFrame))

    logger.info("Merging NPM Demographics with mapping of respondentids and rn_id's")

    logger.info("Merging NPM Demographics with mapping of respondentids and rn_id's")

    # ensure columns are intended types
    tv_view_df['respondentid'] = tv_view_df['respondentid'].astype(np.int64)
    npm_df['internet'] = npm_df['internet'].astype(np.int64)
    hhp_nol_df['respondentid'] = hhp_nol_df['respondentid'].astype(np.int64)
    hhp_nol_df['rn_id'] = hhp_nol_df['rn_id'].astype(float)

    # we only want NPM data with internet and with person age greater than 2
    indices = (npm_df['internet'] == 1) & (2 <= npm_df['age'])

    demo_hhp_join = pd.merge(npm_df.loc[indices, :].reset_index(drop=True),
                             hhp_nol_df[['respondentid', 'rn_id']],
                             on='respondentid', how='left')

    demo_hhp_join['rn_id'] = demo_hhp_join['rn_id'].fillna(-1.0)

    logger.info("Determining the top 500 networks by dayparts")

    # we restrict our npm tv space

    indices = (tv_view_df['respondentid']).isin(demo_hhp_join['respondentid'])

    tv_500 = tv_view_df.loc[indices, :].groupby('pc')['minutes_viewed'].sum().reset_index()\
                 .sort_values(by='minutes_viewed', ascending=False).reset_index(drop=True).loc[:500, 'pc'].tolist()

    tv_100 = tv_500[:100]

    # with our restrictions in place, we parse our top 500 tv networks by dayparts and add that to our dataset

    tv_top_500_df = tv_view_df.loc[tv_view_df['pc'].isin(tv_500), :].reset_index(drop=True)

    tv_top_500_df = tv_top_500_df.groupby(['respondentid', 'pc'])['minutes_viewed'].sum().reset_index()

    tv_top_500_df = funct_pivot(tv_top_500_df, 'minutes_viewed', 'respondentid', 'pc') \
        .reset_index().rename(columns={'index': 'respondentid'})

    tv_top_100_df = tv_view_df.loc[tv_view_df['pc'].isin(tv_100), :].reset_index(drop=True)

    tv_top_100_df = tv_top_100_df.groupby(['respondentid', 'pc'])['minutes_viewed'].sum().reset_index()

    tv_top_100_df = funct_pivot(tv_top_100_df, 'minutes_viewed', 'respondentid', 'pc') \
        .reset_index().rename(columns={'index': 'respondentid'})

    logger.info("Merging NPM Demographics data with NPM TV Usage data")

    # we merge with top 500 tv data
    demo_hhp_tv = pd.merge(demo_hhp_join,
                           pd.merge(demo_hhp_join['respondentid'].to_frame(),
                                    tv_top_500_df.to_dense(),
                                    on='respondentid', how='left').fillna(0.0),
                           on='respondentid')

    logger.info("Determining the online weight of each respondent")

    donors_recips = pd.merge(demo_hhp_tv,
                             nol_df[['rn_id', 'weight']].rename(columns={'weight': 'onl_weight'}),
                             on='rn_id', how='left')

    donors_recips['onl_weight'] = donors_recips['onl_weight'].fillna(0.0)

    donor_indices = donors_recips['onl_weight'] > 0.0
    recip_indices = donors_recips['onl_weight'] <= 0.0

    logger.info("Creating the donors and recipient datasets based on having an online-weight or not")

    donors_df = donors_recips.loc[donor_indices, :].reset_index(drop=True)

    cph_recips_df = donors_recips.loc[recip_indices, :].reset_index(drop=True).drop('rn_id', axis=1)

    cph_donors_df = pd.merge(donors_df,
                             pd.merge(donors_df[['rn_id']],
                                      top_200_df, on='rn_id', how='left').fillna(0.0),
                             on='rn_id')

    # this renames a few columns to what is expected later on in the code
    cph_donors_df = cph_donors_df.rename(columns={'hh_size1_by_count': 'hh_size1',
                                                  'kids_0to5_by_count': 'kids_0to5',
                                                  'kids_6to11_by_count': 'kids_6to11',
                                                  'kids_12to17_by_count': 'kids_12to17'})

    cph_recips_df = cph_recips_df.rename(columns={'hh_size1_by_count': 'hh_size1',
                                                  'kids_0to5_by_count': 'kids_0to5',
                                                  'kids_6to11_by_count': 'kids_6to11',
                                                  'kids_12to17_by_count': 'kids_12to17'})

    return cph_donors_df, cph_recips_df, tv_100, tv_500, tv_top_100_df


def donors_recips_dask(npm_df, nol_df, hhp_nol_df, tv_view_df, top_200_df):
    """
    :param npm_df:
    :param nol_df:
    :param hhp_nol_df:
    :param tv_view_df:
    :param top_200_df:
    :return:
    """

    logger = logging.getLogger(__name__)

    assert (isinstance(npm_df, dd.DataFrame))
    assert (isinstance(nol_df, dd.DataFrame))
    assert (isinstance(tv_view_df, dd.DataFrame))
    assert (isinstance(top_200_df, dd.DataFrame))

    # ensure columns are intended types
    tv_view_df['respondentid'] = tv_view_df['respondentid'].astype(np.int64)
    npm_df['internet'] = npm_df['internet'].astype(np.int64)
    hhp_nol_df['respondentid'] = hhp_nol_df['respondentid'].astype(np.int64)
    hhp_nol_df['rn_id'] = hhp_nol_df['rn_id'].astype(float)

    # we only want NPM data with internet and with person age greater than 2
    indices = (npm_df['internet'] == 1) & (2 <= npm_df['age'])

    logger.info("Merging NPM Demographics with mapping of respondentids and rn_id's")

    demo_hhp_join = npm_df.loc[indices, :].reset_index(drop=True).merge(hhp_nol_df[['respondentid', 'rn_id']],
                                                                        on='respondentid', how='left')

    demo_hhp_join['rn_id'] = demo_hhp_join['rn_id'].fillna(-1.0)

    # we restrict our npm tv space

    logger.info('Adding the Top 500 TV by Dayparts as Features')

    indices = (tv_view_df['respondentid']).isin(demo_hhp_join['respondentid'])

    tv_500 = tv_view_df.loc[indices, :].groupby('pc')['minutes_viewed'].sum().reset_index()\
                 .sort_values(by='minutes_viewed', ascending=False).reset_index(drop=True).loc[:500, 'pc'].compute().\
        tolist()

    tv_100 = tv_500[:100]

    # with our restrictions in place, we parse our top 500 tv networks by dayparts and add that to our dataset

    tv_top_500_df = tv_view_df.loc[tv_view_df['pc'].isin(tv_500), :].reset_index(drop=True)

    tv_top_100_df = tv_view_df.loc[tv_view_df['pc'].isin(tv_100), :].reset_index(drop=True)

    tv_top_500_df = tv_top_500_df.groupby(['respondentid', 'pc'])['minutes_viewed'].sum().reset_index()
    tv_top_100_df = tv_top_100_df.groupby(['respondentid', 'pc'])['minutes_viewed'].sum().reset_index()

    tv_top_500_df = tv_top_500_df.pivot_table('minutes_viewed', 'respondentid', 'pc').reset_index()\
        .rename(columns={'index': 'respondentid'})

    tv_top_100_df = tv_top_100_df.pivot_table('minutes_viewed', 'respondentid', 'pc').reset_index()\
        .rename(columns={'index': 'respondentid'})

    demo_hhp_tv = demo_hhp_join.merge(demo_hhp_join['respondentid'].to_frame().merge(tv_top_500_df,
                                                                                     on='respondentid',
                                                                                     how='left').fillna(0.0),
                                      on='respondentid')

    donors_recips = pd.merge(demo_hhp_tv, nol_df[['rn_id', 'weight']].rename(columns={'weight': 'onl_weight'}),
                             on='rn_id', how='left')

    donors_recips['onl_weight'] = donors_recips['onl_weight'].fillna(0.0)

    donor_indices = donors_recips['onl_weight'] > 0.0
    recip_indices = donors_recips['onl_weight'] <= 0.0

    logger.info('Creating Donor and Recipient dataset by presence of an online-weight.')

    donors_df = donors_recips.loc[donor_indices, :].reset_index(drop=True)

    cph_recips_df = donors_recips.loc[recip_indices, :].reset_index(drop=True).drop('rn_id', axis=1)

    cph_donors_df = donors_df.merge(donors_df['rn_id'].to_frame().merge(top_200_df, on='rn_id', how='left').fillna(0.0),
                                    on='rn_id')

    return cph_donors_df, cph_recips_df, tv_100, tv_500, tv_top_100_df
