import pandas as pd
import numpy as np
import logging

from uszipcode import ZipcodeSearchEngine


def zip_state(df, zip_col, state_col):
    '''
    :type df: pd.DataFrame
    :param df:
    :param zip_col:
    :param state_col:
    :return:
    '''

    search = ZipcodeSearchEngine()

    f = lambda x: str(search.by_zipcode(str(x)).__dict__.get('State')).upper()

    df[state_col] = df[zip_col].map(f)

    return df


def respondent_level_recoding(demo_df, lv, id_col, zip_col):
    """
    :param demo_df:
    :param lv:
    :param id_col:
    :param zip_col:
    :return:
    """

    logger = logging.getLogger(__name__)

    res_df = demo_df[['weight', id_col]].copy()

    #defines a method locally as a helper to consolidate on space
    def age_demo_mapper(demo_age_col, demo_col, age_bounds_list):
        demo_df[age_col] = demo_df[age_col].astype('float64')

        res_df[demo] = 0

        for i in range(len(age_bounds_list) - 1):
            age_indices = (age_bounds_list[i] <= demo_df[demo_age_col]) & (demo_df[demo_age_col] < age_bounds_list[i + 1])

            res_df.loc[age_indices, demo_col] = i + 1

        return None

    for demo in lv:
        try:
            logger.info('Recoding demographic: {}'.format(demo))

            if demo == 'Age0'.lower():
                demo_df[demo] = demo_df['age'].astype('float64')
                res_df[demo] = demo_df[demo]

            elif demo == 'Age1'.lower():
                if 'age0' in demo_df.columns:
                    age_col = 'age0'
                else:
                    age_col = 'age'

                age_bounds = [2, 6, 12, 18, 25, 35, 45, 55, 65, 150]

                age_demo_mapper(age_col, demo, age_bounds)

            elif demo == 'Age7'.lower():
                if 'age0' in demo_df.columns:
                    age_col = 'age0'
                else:
                    age_col = 'age'

                age_bounds = [18, 25, 35, 45, 55, 65, 150]

                age_demo_mapper(age_col, demo, age_bounds)

            elif demo == 'Age8'.lower():
                if 'age0' in demo_df.columns:
                    age_col = 'age0'
                else:
                    age_col = 'age'

                age_bounds = [12, 18, 25, 35, 45, 55, 65, 150]

                age_demo_mapper(age_col, demo, age_bounds)

            elif demo == 'Gender'.lower():
                if 'gender_char' in demo_df.columns:
                    indices = demo_df['gender_char'] == 'M'
                    null_indices = demo_df['gender_char'].isnull()
                else:
                    if demo_df['gender_code'].dtype == int:
                        demo_df['gender_code'] = demo_df['gender_code'].astype(float)
                        indices = demo_df['gender_code'] == 0.
                    else:
                        indices = demo_df['gender_code'] == 'M'

                    null_indices = demo_df['gender_code'].isnull()

                demo_df[demo] = (2 - indices.astype(int)) * (null_indices.astype(int))

            elif demo == 'Race_Black'.lower():
                if 'raceid' in demo_df.columns:
                    demo_df['raceid'] = demo_df['raceid'].astype(float)

                    indices = demo_df['raceid'] == 86.
                    null_indices = demo_df['race_id'].isnull()
                else:
                    if demo_df['race_code'].dtype == int:
                        demo_df['race_code'] = demo_df['race_code'].astype(float)
                    else:
                        demo_df['race_code'] = demo_df['race_code'].astype(int).astype(float)

                    indices = demo_df['race_code'] == 2.
                    null_indices = demo_df['race_code'].isnull()

                res_df[demo] = (2 - indices.astype(int)) * (null_indices.astype(int))

            elif demo == 'Race_Asian'.lower():
                if 'raceid' in demo_df.columns:
                    demo_df['raceid'] = demo_df['raceid'].astype(float)

                    indices = demo_df['raceid'] == 88.

                    null_indices = demo_df['race_id'].isnull()
                else:
                    demo_df['race_code'] = demo_df['race_code'].astype(float)

                    indices = demo_df['race_code'].isin([3., 4., 5., 6., 7., 9.])
                    null_indices = demo_df['race_code'].isnull()

                res_df[demo] = (2 - indices.astype(int)) * (null_indices.astype(int))

            elif demo == 'education4':
                if 'educationid' in demo_df.columns:
                    res_df['education4'] = 2 * demo_df['educationid'].isin([101, 102, 103, 104]).astype(int)
                    res_df['education4'] = demo_df['educationid'].isin([105, 106, 107]).astype(int)
                else:
                    demo_df['education_level_number'] = demo_df['education_level_number'].astype('int64').astype(str)

                    res_df[demo] = 0

                    res_df.loc[demo_df['education_level_number'].isin(['0', '8']), demo] = 1
                    # Some high school
                    res_df.loc[demo_df['education_level_number'].isin(['9', '10', '11']), demo] = 2
                    # high school grad
                    res_df.loc[demo_df['education_level_number'].isin(['12']), demo] = 3
                    # some college
                    res_df.loc[demo_df['education_level_number'].isin(['13', '14', '15']), demo] = 4
                    # college grad
                    res_df.loc[demo_df['education_level_number'].isin(['16']), demo] = 5
                    # graduate degree
                    res_df.loc[demo_df['education_level_number'].isin(['18', '19', '20']), demo] = 6

            elif demo == 'Employment1'.lower():
                if 'employmentid' in demo_df.columns:
                    res_df['employment1'] = 1
                    res_df.loc[demo_df['employmentid'].isin([30, 32]), 'employment1'] = 3
                    res_df.loc[demo_df['employmentid'] == 31, 'employment1'] = 2
                else:

                    demo_df['working_hours_number'] = demo_df['working_hours_number'].astype('float64')
                    demo_df['nielsen_occupation_code'] = demo_df['nielsen_occupation_code'].astype('int64').astype(
                        str)
                    # no response
                    res_df[demo] = 0

                    # not working
                    res_df.loc[(demo_df['working_hours_number'] == 0.) | (demo_df['nielsen_occupation_code'] == '8'),
                               demo] = 1

                    # part-time
                    res_df.loc[(demo_df['working_hours_number'] >= 1.) & (demo_df['working_hours_number'] <= 29.),
                               demo] = 2

                    # full-time
                    res_df.loc[(demo_df['working_hours_number'] >= 30.), demo] = 3

            elif demo == 'Hispanic'.lower():
                if 'hispanicid' in demo_df.columns:
                    demo_df['hispanicid'] = demo_df['hispanicid'].astype(float)

                    indices = demo_df['hispanicid'] == 83.

                else:
                    if demo_df['origin_code'].dtype == int:
                        demo_df['origin_code'] = demo_df['origin_code'].astype('float64')

                        indices = demo_df['origin_code'] == 2.
                    else:
                        indices = demo_df['origin_code'] == '2'

                res_df[demo] = 2 - indices.astype(int)

            elif demo == 'Income1'.lower():
                if 'incomeid' in demo_df.columns:
                    res_df[demo] = 2
                    res_df[demo] = demo_df['incomeid'].isin([20, 21]).astype(int)
                    res_df[demo] = 3 * (demo_df['incomeid'] == 24).astype(int)
                    res_df[demo] = 4 * (demo_df['incomeid'] == 25).astype(int)
                    res_df[demo] = 5 * demo_df['incomeid'].isin([92, 93, 94, 95, 96]).astype(int)
                else:
                    res_df[demo] = 0
                    res_df.loc[(res_df['income_amt'] < 25.), demo] = 1
                    # 25k-49k
                    res_df.loc[(res_df['income_amt'] >= 25.) & (res_df['income_amt'] < 50.), demo] = 2
                    # 50k-74k
                    res_df.loc[(res_df['income_amt'] >= 50.) & (res_df['income_amt'] < 75.), demo] = 3
                    # 75k-99k
                    res_df.loc[(res_df['income_amt'] >= 75.) & (res_df['income_amt'] < 100.), demo] = 4
                    # 100k+
                    res_df.loc[(res_df['income_amt'] >= 100.), demo] = 5

            elif demo == 'RegionB'.lower():

                northeast_states = ['NJ', 'NY', 'PA', 'CT', 'ME', 'MA', 'NH', 'RI', 'VT']
                midwest_states = ['IL', 'IN', 'MI', 'OH', 'WI', 'IA', 'KS', 'MN', 'MO', 'ND', 'NE', 'SD']
                south_states = ['AL', 'KY', 'MS', 'TN', 'DC', 'DE', 'FL', 'GA', 'MD', 'NC', 'SC', 'VA', 'WV', 'AR',
                                'LA',
                                'OK', 'TX']
                west_states = ['AZ', 'CO', 'ID', 'MT', 'NM', 'NV', 'UT', 'WY', 'AK', 'CA', 'HI', 'OR', 'WA']

                demo_df = zip_state(demo_df, zip_col, 'st')

                res_df[demo] = 0
                res_df.loc[demo_df['st'].isin(northeast_states), demo] = 1
                res_df.loc[demo_df['st'].isin(midwest_states), demo] = 2
                res_df.loc[demo_df['st'].isin(south_states), demo] = 3
                res_df.loc[demo_df['st'].isin(west_states), demo] = 4

                res_df['regionb_northeast'] = 2 - (res_df[demo] == 1).astype(int)
                res_df['regionb_midwest'] = 2 - (res_df[demo] == 2).astype(int)
                res_df['regionb_south'] = 2 - (res_df[demo] == 3).astype(int)
                res_df['regionb_west'] = 2 - (res_df[demo] == 4).astype(int)

            elif demo == 'County_Size'.lower():
                if 'county_size_code' in demo_df.columns:
                    # no response
                    res_df[demo] = 0
                    # county_size_code='A'
                    res_df.loc[demo_df['county_size_code'] == 'A', demo] = 1
                    # county_size_code='B'
                    res_df.loc[demo_df['county_size_code'] == 'B', demo] = 2
                    # county_size_code='C'
                    res_df.loc[demo_df['county_size_code'] == 'C', demo] = 3
                    # county_size_code='D'
                    res_df.loc[demo_df['county_size_code'] == 'D', demo] = 4

        except Exception as e:
            logger.fatal(e, exc_info=True)
            res_df[demo] = np.nan

    return res_df
