import logging

import pandas as pd
import numpy as np
import dask.dataframe as dd

from recoding.src.data_recoding.recoding import respondent_level_recoding


def emm_respondent_level_recoding(demo_df, lv):
    """
    :param demo_df:
    :param lv:
    :return:
    """
    assert isinstance(demo_df, pd.DataFrame)

    demo_df = demo_df.rename(columns={col: col.lower() for col in demo_df.columns})

    gen_res_df = respondent_level_recoding(demo_df, lv + ['age7'], 'mobileid', 'zip_code')

    logger = logging.getLogger(__name__)

    res_df = pd.DataFrame()

    logger.info('Setting the respondentid from mobile id')

    res_df['respondentid'] = demo_df['mobileid'].to_frame()

    for demo in lv:
        logger.info('Recoding demographic: {}'.format(demo))

        try:

            if demo == 'gender_age':
                gen_res_df['agegrp'] = gen_res_df['age7'].isin([1, 2]).astype(int)
                gen_res_df['agegrp'] = 2 * gen_res_df['age7'].isin([3, 4]).astype(int)
                gen_res_df['agegrp'] = 3 * (gen_res_df['age7'] == 5).astype(int)

                gen_res_df[demo] = gen_res_df['agegrp']

                indices = gen_res_df['gender'] == 2
                gen_res_df.loc[indices, 'gender_age'] += 3

                gen_res_df.drop('agegrp', axis=1, inplace=True)

        except Exception as e:
            logger.fatal(e, exc_info=True)
            logger.fatal('A column does not exist')
            res_df[demo] = np.nan

    res_df = pd.merge(gen_res_df, res_df, on='respondentid')

    logger.info('Removing any recoded demographic variables with code 0')
    indices = (res_df == 0).sum(axis=1) == 0

    res_df = res_df.loc[indices, :].reset_index(drop=True)

    return res_df


def main_emm_recode(emm_raw_df, lv):
    """
    :param emm_raw_df:
    :param lv:
    :return:
    """

    if isinstance(emm_raw_df, pd.DataFrame):
        assert isinstance(lv, list)

        return emm_respondent_level_recoding(emm_raw_df, lv)

    elif isinstance(emm_raw_df, dd.DataFrame):
        assert isinstance(lv, list)

        return emm_raw_df.map_partitions(lambda df: emm_respondent_level_recoding(df, lv))
