import logging

import numpy as np
import pandas as pd
import dask.dataframe as dd

from recoding.src.data_recoding.recoding import respondent_level_recoding


def npm_respondent_level_recoding(demo_df, lv):
    """
    :param demo_df:
    :param lv:
    :return:
    """
    gen_res_df = respondent_level_recoding(demo_df, lv, 'respondentid', 'zip_code_hh')

    logger = logging.getLogger(__name__)

    res_df = demo_df['respondentid'].to_frame()

    for demo in lv:
        logger.info('Recoding demographic: {}'.format(demo))

        try:
            if demo == 'Education7'.lower():
                demo_df['education_level_number'] = demo_df['education_level_number'].astype('int64').astype(str)
                # no response
                res_df[demo] = 0
                # High school grad or less
                res_df.loc[demo_df['education_level_number'].isin(['0', '8', '9', '10', '11', '12']), demo] = 1
                # some college
                res_df.loc[demo_df['education_level_number'].isin(['13', '14', '15']), demo] = 2
                # college grad
                res_df.loc[demo_df['education_level_number'].isin(['16']), demo] = 3
                # college grads/graduate degree
                res_df.loc[demo_df['education_level_number'].isin(['18', '19', '20']), demo] = 4

            elif demo == 'Income9'.lower():
                demo_df['income_amt'] = demo_df['income_amt'].astype('float64')
                # no response
                res_df[demo] = 0
                # 25k under
                res_df.loc[(demo_df['income_amt'] < 25.), demo] = 1
                # 25k-34k
                res_df.loc[(demo_df['income_amt'] >= 25.) & (demo_df['income_amt'] < 35.), demo] = 2
                # 35k-49k
                res_df.loc[(demo_df['income_amt'] >= 35.) & (demo_df['income_amt'] < 50.), demo] = 3
                # 50k-74k
                res_df.loc[(demo_df['income_amt'] >= 50.) & (demo_df['income_amt'] < 75.), demo] = 4
                # 75k-99k
                res_df.loc[(demo_df['income_amt'] >= 75.) & (demo_df['income_amt'] < 100.), demo] = 5
                # 100k+
                res_df.loc[(demo_df['income_amt'] >= 100.), demo] = 6

            elif demo == 'Satellite'.lower():
                res_df[demo] = '2'
                res_df.loc[demo_df['alternative_delivery_flag'] == 'Y', demo] = '1'

            elif demo == 'Spanish_Language1'.lower():
                demo_df['language_class_code'] = demo_df['language_spoken_code_psn'].astype('float64')
                demo_df['origin_code'] = demo_df['origin_code'].astype('float64')

                # no response
                res_df[demo] = 0
                # only Spanish
                res_df.loc[demo_df['language_class_code'] == 1., demo] = 1
                # mostly Spanish
                res_df.loc[demo_df['language_class_code'] == 3., demo] = 2
                # Spanish or English equally
                res_df.loc[demo_df['language_class_code'] == 5., demo] = 3
                # mostly English
                res_df.loc[demo_df['language_class_code'] == 4., demo] = 4
                # only English
                res_df.loc[demo_df['language_class_code'] == 2., demo] = 5
                # non-Hispanic
                res_df.loc[demo_df['origin_code'] == 1., demo] = 6

            elif demo == 'Number_of_TVs1'.lower():
                demo_df['number_of_tv_sets'] = demo_df['number_of_tv_sets'].astype('float64')
                res_df[demo] = 0
                res_df.loc[demo_df['number_of_tv_sets'] == 1., demo] = 1
                res_df.loc[demo_df['number_of_tv_sets'] == 2., demo] = 2
                res_df.loc[demo_df['number_of_tv_sets'] == 3., demo] = 3
                res_df.loc[demo_df['number_of_tv_sets'] == 4., demo] = 4
                res_df.loc[demo_df['number_of_tv_sets'] >= 5., demo] = 5

            elif demo == 'kids_0to5'.lower():
                demo_df['number_of_kids_less_than_6'] = demo_df['number_of_kids_less_than_6'].astype(
                    'float64')

                res_df[demo] = 0  # No response
                res_df.loc[demo_df['number_of_kids_less_than_6'] > 0., demo] = 1  # Yes
                res_df.loc[demo_df['number_of_kids_less_than_6'] == 0., demo] = 2  # No

            elif demo == 'kids_6to11'.lower():
                demo_df['number_of_kids_less_than_12'] = demo_df['number_of_kids_less_than_12'].astype('float64')
                demo_df['number_of_kids_less_than_6'] = demo_df['number_of_kids_less_than_6'].astype('float64')

                res_df[demo] = 0  # No response
                res_df.loc[demo_df['number_of_kids_less_than_12'] - demo_df['number_of_kids_less_than_6'] > 0.,
                           demo] = 1
                res_df.loc[demo_df['number_of_kids_less_than_12'] - demo_df['number_of_kids_less_than_6'] == 0.,
                           demo] = 2

            elif demo == 'kids_12to17'.lower():
                demo_df['number_of_kids_less_than_18'] = demo_df['number_of_kids_less_than_18'].astype(
                    'float64')
                demo_df['number_of_kids_less_than_12'] = demo_df['number_of_kids_less_than_12'].astype(
                    'float64')
                res_df[demo] = 0
                res_df.loc[demo_df['number_of_kids_less_than_18'] - demo_df['number_of_kids_less_than_12'] > 0.,
                           demo] = 1
                res_df.loc[demo_df['number_of_kids_less_than_18'] - demo_df['number_of_kids_less_than_12'] == 0.,
                           demo] = 2

            elif demo == 'Occupation1'.lower():
                demo_df['nielsen_occupation_code'] = demo_df['nielsen_occupation_code'].astype('int64').astype(str)

                # no response
                res_df[demo] = 0
                # professional/technical or administrator/manager
                res_df.loc[demo_df['nielsen_occupation_code'].isin(['0', '1']), demo] = 1
                # sales/clerical
                res_df.loc[demo_df['nielsen_occupation_code'].isin(['2']), demo] = 2
                # others
                res_df.loc[demo_df['nielsen_occupation_code'].isin(['3', '4', '5', '6', '7']), demo] = 3
                # retired/not seeking employment
                res_df.loc[demo_df['nielsen_occupation_code'].isin(['8']), demo] = 4

            elif demo == 'DVR'.lower():
                res_df[demo] = 2
                res_df.loc[demo_df['dvr_flag'] == 'Y', demo] = 1

            elif demo == 'CablePlus'.lower():
                res_df[demo] = 2
                res_df.loc[demo_df['cable_plus_flag'] == 'Y', demo] = 1

            elif demo == 'Video_Game'.lower():
                res_df[demo] = 2
                res_df.loc[demo_df['video_game_owner_flag'] == 'Y', demo] = 1

            elif demo == 'Internet'.lower():
                res_df[demo] = 0
                res_df.loc[demo_df['internet_access_flag'] == 'Y', demo] = 1
                res_df.loc[demo_df['internet_access_flag'] == 'N', demo] = 2

            elif demo == 'Education2'.lower():
                demo_df['education_level_number'] = demo_df['education_level_number'].astype('int64').astype(str)
                # no response
                res_df[demo] = 0
                # high school or less or college no
                res_df.loc[demo_df['education_level_number'].isin(['0', '8', '9', '10', '11', '12', '13', '14', '15']),
                           demo] = 2
                # college yes
                res_df.loc[demo_df['education_level_number'].isin(['16', '18', '19', '20']), demo] = 1

            elif demo == 'Paycable'.lower():
                res_df[demo] = 2
                res_df.loc[demo_df['pay_cable_flag'] == 'Y', demo] = 1

            elif demo == 'HDTV'.lower():
                res_df[demo] = 2
                res_df.loc[demo_df['television_high_definition_display_capability_flag'] == 'Y', demo] = 1

        except Exception as e:
            logger.fatal(e, exc_info=True)
            res_df[demo] = np.nan
            pass

    res_df = pd.merge(gen_res_df, res_df, on='respondentid')

    logger.info('Removing any recoded demographic variables with code 0')

    try:
        indices = (res_df[lv] == 0).sum(axis=1) == 0

        res_df = res_df.loc[indices, :].reset_index(drop=True)

    except Exception as e:
        logger.fatal(e, exc_info=True)
        logger.fatal('Not all demographic variables were created')

        return res_df

    return res_df


def main_recode_npm(npm_hh_df, npm_person_df, lv):
    """
    :param npm_hh_df:
    :param npm_person_df:
    :param lv:
    :return:
    """
    assert isinstance(npm_hh_df, pd.DataFrame) | isinstance(npm_hh_df, dd.DataFrame)
    assert isinstance(npm_person_df, pd.DataFrame) | isinstance(npm_person_df, dd.DataFrame)
    assert isinstance(lv, list)
    assert np.all([isinstance(x, str) for x in lv])

    logger = logging.getLogger(__name__)

    if len(lv) == 0:
        lv = ['Age0', 'Age1', 'Age7', 'CablePlus', 'DVR', 'COUNTY_SIZE', 'EDUCATION4', 'EDUCATION7', 'EMPLOYMENT1',
          'GENDER', 'HISPANIC', 'HDTV', 'INCOME1', 'INCOME9', 'kids_0to5', 'kids_12to17', 'kids_6to11',
          'number_of_tvs1', 'Occupation1', 'Paycable', 'RACE_ASIAN', 'RACE_BLACK', 'REGIONB', 'REGIONB_MIDWEST',
          'REGIONB_NORTHEAST', 'REGIONB_SOUTH', 'REGIONB_WEST', 'SPANISH_LANGUAGE1', 'Video_Game', 'Satellite']

    lv = [x.lower() for x in lv]

    # Load household demos
    npm_hh_df.drop_duplicates(subset='hhid', inplace=True)

    # Load person demos
    npm_person_df.drop_duplicates(subset=['hhid', 'personid'], inplace=True)

    # Merge household demos onto person demos
    logger.info('Merging household demos onto person demos.')
    npm_hh_df = npm_hh_df.rename(columns={'avgwt': 'hh_weight', 'cph_daily_avgwt': 'hh_cph_weight', \
                                                'countintab': 'hh_countintab', 'zip_code': 'zip_code_hh'})

    npm_person_df['respondentid'] = (npm_person_df['hhid'].astype('int64') * 1000 + \
                                    npm_person_df['personid'].astype('int64'))

    npm_person_df = npm_person_df.rename(columns={'avgwt': 'weight', 'avgwt_cph_daily': 'cph_weight', \
                                                'zip_code': 'zip_code_psn'})

    npm_recoded = npm_person_df.merge(npm_hh_df, how='inner', on='hhid')

    logger.info('Recoding for hh_size1')

    hh_size = npm_person_df[['hhid', 'personid']].groupby('hhid').count().reset_index()
    hh_size = hh_size.rename(columns={'personid': 'number_of_persons_count_all'})

    npm_person_df = npm_person_df.merge(hh_size, on='hhid', how='inner')

    npm_person_df['number_of_persons_count_all'] = npm_person_df['number_of_persons_count_all'].astype('float64')

    npm_person_df['hh_size1'] = npm_person_df['number_of_persons_count_all'].clip_upper(5.).astype(int)

    # Remove respondents that were not in tab
    npm_recoded['hh_countintab'] = npm_recoded['hh_countintab'].astype('float64')
    npm_recoded['countintab'] = npm_recoded['countintab'].astype('float64')
    npm_recoded = npm_recoded.loc[(npm_recoded['hh_countintab'] > 0.) & (npm_recoded['countintab'] > 0.)] \
        .reset_index(drop=True)

    npm_recoded = npm_recoded.drop(['hh_weight', 'hh_countintab'], axis=1)

    if isinstance(npm_recoded, pd.DataFrame):
        logger.info('NPM Sample Size: {}'.format(npm_recoded.shape[0]))
    else:
        logger.info('NPM Sample Size: {}'.format(npm_recoded.map_partitions(lambda df: df.shape[0]).compute().sum()))

    logger.info('Recoding NPM Sample')

    if isinstance(npm_recoded, pd.DataFrame):
        return pd.merge(npm_respondent_level_recoding(npm_recoded, lv), npm_person_df[['respondentid', 'hh_size1']],
                        on='respondentid')
    else:
        return npm_recoded.map_partitions(lambda df: npm_respondent_level_recoding(df, lv)).merge(
            npm_person_df[['respondentid', 'hh_size1']], on='respondentid')
