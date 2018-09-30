import os

import numpy as np
import pandas as pd

#################################UPDATE THESE PARAMETERS################################

pd_name=201806
demo_feed = "emm_demos_raw_201806.txt"
output_path = r"//usdczgk842/Fusion/TV_Dig/RDD_Digital/EMM_Feeds/sas_input/"
input_path = r"//usdczgk842/Fusion/TV_Dig/RDD_Digital/EMM_Feeds/raw/"
#spanish_feed = "mapping_panelistid_spanishind_201805.csv"

################################END UPDATE PARAMETERS################################

################################CREATE DEMO FEED #####################################################
demo_names=["DATA_DATE","panelistid","PANEL_TYPE","PANEL_STATUS","PANEL_ELIGIBILITY","PANEL_SOURCE","AGE",
            "GENDER","DMA","REGION","COUNTRY_CODE","MOBILEID","BIRTH_YEAR","MARITAL_STATUS","ZIPCODE","DEVICE_NAME",
            "OS","educationid","EDUCATIONDESC","INCOMEID","INCOMEDESC","employmentid","EMPLOYMENTSTATUSDESC",
            "RACEID","RACEDESC","HISPANICORIGINID","HISPANICORIGINDESC","weight","AGEGROUP","spanish_language0"]

demo_feed = pd.read_table(input_path + str(pd_name) + "/" + str(demo_feed),
                    delimiter="\b", header=None, names=demo_names)
     
demo_feed2 = demo_feed[demo_feed['PANEL_ELIGIBILITY'].isin(['Y'])]
demo_feed3 = demo_feed2[demo_feed2['PANEL_STATUS'].isin(['A'])]
demo_feed4 = demo_feed3[demo_feed3['DEVICE_NAME'].isin(['PHN'])]
demo_feed4['ETHNIC_IDENTITY_CODE'] = np.where(demo_feed4['HISPANICORIGINDESC'] == "Yes, of Hispanic origin", 2, 1)
demo_feed4['hispanicid'] = np.where(demo_feed4['ETHNIC_IDENTITY_CODE'] == 1, 84, 83)
demo_feed5 = demo_feed4[["MOBILEID", "panelistid", "GENDER", "weight", "OS",
               "ZIPCODE",  "AGE", "educationid", "INCOMEID", "employmentid", "RACEID", "hispanicid", "ETHNIC_IDENTITY_CODE", "spanish_language0"]]

#spanish_names=["panelistid","spanish_language0"]
#spanish_feed = pd.read_csv(r"//usdczgk842/Fusion/TV_Dig/RDD_Digital/EMM_Feeds/raw/" + str(pd_name) + "/" + str(spanish_feed),
#                    delimiter=",", header=None, names=spanish_names)

#spanish_feed2 = pd.merge(demo_feed5, spanish_feed, how = "left", left_on = "panelistid", right_on = "panelistid")
demo_feed5['spanish_language1'] = np.where(demo_feed5['ETHNIC_IDENTITY_CODE'] == 1, 6, demo_feed5['spanish_language0'])
demo_feed5['spanish_language1'].fillna(3, inplace=True)

#demo_feed6 = pd.merge(demo_feed5, spanish_feed2, how = "outer", left_on = "panelistid", right_on = "panelistid")

demo_feed6 = demo_feed5[["MOBILEID", "panelistid", "GENDER", "weight", "OS",
               "ZIPCODE",  "AGE", "educationid", "INCOMEID", "employmentid", "RACEID", "hispanicid", "spanish_language1"]]
#demo_feed7 = demo_feed5[["MOBILEID_x", "panelistid", "GENDER_x", "weight_x", "OS_x",
#               "ZIPCODE_x",  "AGE_x", "educationid_x", "INCOMEID_x", "employmentid_x", "RACEID_x", "hispanicid_x", "spanish_language1"]]

#demo_feed7.rename(columns = {'MOBILEID_x':'MOBILEID','GENDER_x':'GENDER'
#                             ,'weight_x':'weight','OS_x':'OS'
#                             ,'ZIPCODE_x':'ZIPCODE','AGE_x':'AGE'
#                             ,'educationid_x':'educationid','INCOMEID_x':'INCOMEID'
#                             ,'employmentid_x':'employmentid','RACEID_x':'RACEID'
#                             ,'hispanicid_x':'hispanicid'},inplace=True)

#export to csv
demo_feed6.to_csv(os.path.join(output_path + str(pd_name),"emm_phn_demos_raw_" +  str(pd_name) +".csv"),index=False)


demo_feed4 = demo_feed3[demo_feed3['DEVICE_NAME'].isin(['TAB'])]
demo_feed4['ETHNIC_IDENTITY_CODE'] = np.where(demo_feed4['HISPANICORIGINDESC'] == "Yes, of Hispanic origin", 2, 1)
demo_feed4['hispanicid'] = np.where(demo_feed4['ETHNIC_IDENTITY_CODE'] == 1, 84, 83)
demo_feed5 = demo_feed4[["MOBILEID", "panelistid", "GENDER", "weight", "OS",
               "ZIPCODE",  "AGE", "educationid", "INCOMEID", "employmentid", "RACEID", "hispanicid", "ETHNIC_IDENTITY_CODE", "spanish_language0"]]

#spanish_feed2 = pd.merge(demo_feed5, spanish_feed, how = "left", left_on = "panelistid", right_on = "panelistid")
#spanish_feed2['spanish_language1'] = np.where(spanish_feed2['ETHNIC_IDENTITY_CODE'] == 1, 6, spanish_feed2['spanish_language0'])
#spanish_feed2['spanish_language1'].fillna(3, inplace=True)
demo_feed5['spanish_language1'] = np.where(demo_feed5['ETHNIC_IDENTITY_CODE'] == 1, 6, demo_feed5['spanish_language0'])
demo_feed5['spanish_language1'].fillna(3, inplace=True)

demo_feed6 = demo_feed5[["MOBILEID", "panelistid", "GENDER", "weight", "OS",
               "ZIPCODE",  "AGE", "educationid", "INCOMEID", "employmentid", "RACEID", "hispanicid", "spanish_language1"]]
#demo_feed7 = demo_feed6[["MOBILEID_x", "panelistid", "GENDER_x", "weight_x", "OS_x",
#               "ZIPCODE_x",  "AGE_x", "educationid_x", "INCOMEID_x", "employmentid_x", "RACEID_x", "hispanicid_x", "spanish_language1"]]

#demo_feed7.rename(columns = {'MOBILEID_x':'MOBILEID','GENDER_x':'GENDER'
#                             ,'weight_x':'weight','OS_x':'OS'
#                             ,'ZIPCODE_x':'ZIPCODE','AGE_x':'AGE'
#                             ,'educationid_x':'educationid','INCOMEID_x':'INCOMEID'
#                             ,'employmentid_x':'employmentid','RACEID_x':'RACEID'
#                             ,'hispanicid_x':'hispanicid'},inplace=True)

#export to csv
demo_feed6.to_csv(os.path.join(output_path + str(pd_name),"emm_tab_demos_raw_" +  str(pd_name) +".csv"),index=False)
