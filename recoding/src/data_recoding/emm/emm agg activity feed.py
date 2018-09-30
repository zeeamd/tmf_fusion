import os

import numpy as np
import pandas as pd

#################################UPDATE THESE PARAMETERS################################

pd_name=201806
activity_feed = "emm_dcr_dsc_activity_p_M_c_US_r_NA_d_20180601_f_20180726_172757.txt"
hierachy_feed = "nfd_fdf_emm_hier_m_201806.txt"
output_path = r"//usdczgk842/Fusion/TV_Dig/RDD_Digital/EMM_Feeds/sas_input/"
input_path = r"//usdczgk842/Fusion/TV_Dig/RDD_Digital/EMM_Feeds/raw/"

emm_act_names=["period_id","period_type", "country", "panel_code", "panel_source",
               "platform", "device_type", "content_type", "data_source_type",
               "tree_id", "rule_id", "day_of_week", "daypart_id", "asset", 
               "person_rn_id", "pageviews", "duration_content",
               "duration_ads", "ad_count", "video_starts", "app_launches",
               "OS", "start_event_local", "session_id", "event id", "app_dictionary_id",
               "mobile_id", "url_name", "domain_name", "site name"]

activity_feed = pd.read_csv(input_path + str(pd_name) + "/" + str(activity_feed), 
                    delimiter="\b", header=None, names=emm_act_names)
         
#hierachy
hier_feed_names=["report_credit_id","report_credit_name", "report_credit_level_code",
                 "web_tree_association", "parentid", "brandid", "channelid"]

hier_feed = pd.read_csv(input_path + str(pd_name) + "/" + str(hierachy_feed), 
                    delimiter="\b", header=None, names=hier_feed_names)

app_feed = pd.DataFrame()
hier_app = hier_feed[hier_feed['report_credit_level_code'] == 9]

app_feed['parentid'] = hier_app['parentid']
app_feed['parentid'].fillna(0, inplace=True)
app_feed['parentid'] = app_feed['parentid'].astype(int)
app_feed['brandid'] = hier_app['brandid']
app_feed['brandid'].fillna(0, inplace=True)
app_feed['brandid'] = app_feed['brandid'].astype(int)
app_feed['channelid'] = hier_app['channelid']
app_feed['channelid'].fillna(0, inplace=True)
app_feed['channelid'] = app_feed['channelid'].astype(int)
app_feed['appid'] = hier_app['report_credit_id'].astype(int) - 200000000000
app_feed['appid'] = app_feed['appid'].astype(int)
app_feed['appid'].fillna(-5, inplace=True)
app_feed['appname'] = hier_app['report_credit_name']
app_feed['report_credit_id'] = hier_app['report_credit_id']
app_feed.sort_values('appid', inplace=True)
app_feed.reset_index(inplace=True, drop=True)

url_feed = pd.DataFrame()

hier_url = hier_feed[hier_feed['report_credit_level_code'].isin([2,3])]

url_feed['parentid'] = hier_url['parentid']
url_feed['parentid'].fillna(0, inplace=True)
url_feed['parentid'] = url_feed['parentid'].astype(int)
url_feed['brandid'] = hier_url['brandid']
url_feed['brandid'].fillna(0, inplace=True)
url_feed['brandid'] = url_feed['brandid'].astype(int)
url_feed['channelid'] = hier_url['channelid']
url_feed['channelid'].fillna(0, inplace=True)
url_feed['channelid'] = url_feed['channelid'].astype(int)
url_feed['appid'] = -5
url_feed['appid'] = url_feed['appid'].astype(int)
url_feed['report_credit_id'] = hier_url['report_credit_id']
url_feed.sort_values(['parentid', 'brandid', 'channelid', 'appid'],inplace=True)
url_feed.reset_index(inplace=True, drop=True)

activity_feed_url = activity_feed[activity_feed['data_source_type'] == 'url'] 
activity_feed_app = activity_feed[activity_feed['data_source_type'] == 'app']

activity_feed1u = pd.merge(activity_feed_url, url_feed, how = "inner", left_on = "tree_id", right_on = "report_credit_id")
activity_feed2u = activity_feed1u[["data_source_type", "pageviews", "duration_content",
               "mobile_id",  "parentid", "brandid", "channelid"]]

activity_feed2u.rename(columns = {'mobile_id':'mobileid','duration_content':'viewduration'},inplace=True)
activity_feed2u.loc[:,'appid'] = -5

activity_feed1a = pd.merge(activity_feed_app, app_feed, how = "inner", left_on = "app_dictionary_id", 
                           right_on = "report_credit_id")
activity_feed2a = activity_feed1a[["data_source_type", "pageviews", "duration_content",
               "mobile_id",  "parentid", "brandid", "channelid", "app_dictionary_id", "appname"]]
activity_feed2a.rename(columns = {'mobile_id':'mobileid','duration_content':'viewduration', 'app_dictionary_id':'appid'},inplace=True)


activity_feed2 = activity_feed2u.append(activity_feed2a, ignore_index = True)

emm_aggr = activity_feed2.groupby(["mobileid", "parentid", "brandid", "channelid", "appid"], as_index=False
        ).agg({'pageviews': np.sum, 'viewduration': np.sum})

app_feed['appid2'] = app_feed['appid'] + 200000000000
emm_aggr2 = pd.merge(emm_aggr, app_feed[["appid2","appname"]], how = "left", left_on = "appid", 
                           right_on = "appid2")
emm_aggr2['appid'] = emm_aggr2['appid2'] - 200000000000
emm_aggr2 = emm_aggr2[["mobileid", "parentid", "brandid", "channelid", "appid", "appname", "pageviews", "viewduration"]]

#export to csv
emm_aggr2.to_csv(os.path.join(os.path.join(output_path + str(pd_name)),"emm_activity_raw_" +  str(pd_name) +".csv"),index=False)
