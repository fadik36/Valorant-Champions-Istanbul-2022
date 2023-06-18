
# region FUTURE E-SPORTS
# endregion

# region IMPORT
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

#sreamlit=1.8.1
import streamlit as st
import plotly.express as px
from PIL import Image
# endregion

# region GÖRESEL AYARLAMALAR
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 160)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)
pd.set_option("display.float_format", lambda x: "%.2f" % x)
warnings.filterwarnings("ignore")
# endregion

# region VERİ ÇEKME
path = "C:\Program Files (x86)\chromedriver.exe"
driver = webdriver.Chrome(path)
###################### LİNKİ DEĞİŞTİR ######################
driver.get("https://www.vlr.gg/84646/ninjas-in-pyjamas-vs-zeta-division-valorant-champions-tour-stage-1-masters-reykjav-k-decider-a")
# endregion

# region MAÇLARIN MAP'LERİNİ ALMA
maps1 = driver.find_element("xpath", '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[1]/div[2]/div/div[2]/div')
pick1 = maps1.text
maps2 = driver.find_element("xpath", '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[1]/div[2]/div/div[3]/div')
pick2 = maps2.text
maps3 = driver.find_element("xpath", '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[1]/div[2]/div/div[4]/div')
pick3 = maps3.text

# eğer 4.maç varsa çalıştır yoksa çalışmaz hata verir
# maps4 = driver.find_element("xpath",'//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[1]/div[2]/div/div[5]/div')
# pick4 = maps4.text


team_name1 = driver.find_element("xpath", '//*[@id="wrapper"]/div[1]/div[3]/div[1]/div[2]/a[1]/div/div[1]')
new_team1 = team_name1.text
team_name2 = driver.find_element("xpath", '//*[@id="wrapper"]/div[1]/div[3]/div[1]/div[2]/a[2]/div/div[1]')
new_team2 = team_name2.text
vs = new_team1 + "/" + new_team2

final_score1 = driver.find_element("xpath", '//*[@id="wrapper"]/div[1]/div[3]/div[1]/div[2]/div/div[2]/div[1]/span[1]')
final_score1 = final_score1.text
final_score2 = driver.find_element("xpath", '//*[@id="wrapper"]/div[1]/div[3]/div[1]/div[2]/div/div[2]/div[1]/span[3]')
final_score2 = final_score2.text
final = (final_score1 + "/" + final_score2)

team_name1 = driver.find_element("xpath", '//*[@id="wrapper"]/div[1]/div[3]/div[1]/div[2]/a[1]/div/div[1]')
new_team1 = team_name1.text
team_name2 = driver.find_element("xpath", '//*[@id="wrapper"]/div[1]/div[3]/div[1]/div[2]/a[2]/div/div[1]')
new_team2 = team_name2.text
team = new_team1 + "/" + new_team2


# endregion

# region MAÇ SAYISINA GÖRE ATTACK VE DEFENCE SCORE'LERİNİ ALMA
###### MAP_1 #####
def map1_attack():
    name = driver.find_element("xpath",
                               '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[1]/div[4]/div[1]/table/tbody/tr[1]/td[1]/div/a/div[1]')
    new_name = name.text
    acs = driver.find_element("xpath",
                              '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[1]/div[4]/div[1]/table/tbody/tr[1]/td[4]/span/span[2]')
    new_acs = acs.text
    kill = driver.find_element("xpath",
                               '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[1]/div[4]/div[1]/table/tbody/tr[1]/td[5]/span/span[2]')
    new_kill = kill.text
    death = driver.find_element("xpath",
                                '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[1]/div[4]/div[1]/table/tbody/tr[1]/td[6]/span/span[2]/span[2]')
    new_death = death.text
    assists = driver.find_element("xpath",
                                  '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[1]/div[4]/div[1]/table/tbody/tr[1]/td[7]/span/span[2]')
    new_assists = assists.text
    adr = driver.find_element("xpath",
                              '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[1]/div[4]/div[1]/table/tbody/tr[1]/td[10]/span/span[2]')
    new_adr = adr.text
    kast = driver.find_element("xpath",
                               '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[1]/div[4]/div[1]/table/tbody/tr[1]/td[8]/span/span[2]')
    new_kast = kast.text
    fk = driver.find_element("xpath",
                             '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[1]/div[4]/div[1]/table/tbody/tr[1]/td[12]/span/span[2]')
    new_fk = fk.text
    fb = driver.find_element("xpath",
                             '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[1]/div[4]/div[1]/table/tbody/tr[1]/td[13]/span/span[2]')
    new_fb = fb.text
    name2 = driver.find_element("xpath",
                                '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[1]/div[4]/div[1]/table/tbody/tr[2]/td[1]/div/a/div[1]')
    new_name2 = name2.text
    acs2 = driver.find_element("xpath",
                               '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[1]/div[4]/div[1]/table/tbody/tr[2]/td[4]/span/span[2]')
    new_acs2 = acs2.text
    kill2 = driver.find_element("xpath",
                                '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[1]/div[4]/div[1]/table/tbody/tr[2]/td[5]/span/span[2]')
    new_kill2 = kill2.text
    death2 = driver.find_element("xpath",
                                 '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[1]/div[4]/div[1]/table/tbody/tr[2]/td[6]/span/span[2]/span[2]')
    new_death2 = death2.text
    assists2 = driver.find_element("xpath",
                                   '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[1]/div[4]/div[1]/table/tbody/tr[2]/td[7]/span/span[2]')
    new_assists2 = assists2.text
    adr2 = driver.find_element("xpath",
                               '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[1]/div[4]/div[1]/table/tbody/tr[2]/td[10]/span/span[2]')
    new_adr2 = adr2.text
    fk2 = driver.find_element("xpath",
                              '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[1]/div[4]/div[1]/table/tbody/tr[2]/td[12]/span/span[2]')
    new_fk2 = fk2.text
    fb2 = driver.find_element("xpath",
                              '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[1]/div[4]/div[1]/table/tbody/tr[2]/td[13]/span/span[2]')
    new_fb2 = fb2.text
    name3 = driver.find_element("xpath",
                                '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[1]/div[4]/div[1]/table/tbody/tr[3]/td[1]/div/a/div[1]')
    new_name3 = name3.text
    acs3 = driver.find_element("xpath",
                               '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[1]/div[4]/div[1]/table/tbody/tr[3]/td[4]/span/span[2]')
    new_acs3 = acs3.text
    kill3 = driver.find_element("xpath",
                                '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[1]/div[4]/div[1]/table/tbody/tr[3]/td[5]/span/span[2]')
    new_kill3 = kill3.text
    death3 = driver.find_element("xpath",
                                 '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[1]/div[4]/div[1]/table/tbody/tr[3]/td[6]/span/span[2]/span[2]')
    new_death3 = death3.text
    assists3 = driver.find_element("xpath",
                                   '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[1]/div[4]/div[1]/table/tbody/tr[3]/td[7]/span/span[2]')
    new_assists3 = assists3.text
    adr3 = driver.find_element("xpath",
                               '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[1]/div[4]/div[1]/table/tbody/tr[3]/td[10]/span/span[2]')
    new_adr3 = adr3.text
    fk3 = driver.find_element("xpath",
                              '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[1]/div[4]/div[1]/table/tbody/tr[3]/td[12]/span/span[2]')
    new_fk3 = fk3.text
    fb3 = driver.find_element("xpath",
                              '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[1]/div[4]/div[1]/table/tbody/tr[3]/td[13]/span/span[2]')
    new_fb3 = fb3.text
    name4 = driver.find_element("xpath",
                                '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[1]/div[4]/div[1]/table/tbody/tr[4]/td[1]/div/a/div[1]')
    new_name4 = name4.text
    acs4 = driver.find_element("xpath",
                               '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[1]/div[4]/div[1]/table/tbody/tr[4]/td[4]/span/span[2]')
    new_acs4 = acs4.text
    kill4 = driver.find_element("xpath",
                                '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[1]/div[4]/div[1]/table/tbody/tr[4]/td[5]/span/span[2]')
    new_kill4 = kill4.text
    death4 = driver.find_element("xpath",
                                 '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[1]/div[4]/div[1]/table/tbody/tr[4]/td[6]/span/span[2]')
    new_death4 = death4.text
    assists4 = driver.find_element("xpath",
                                   '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[1]/div[4]/div[1]/table/tbody/tr[4]/td[7]/span/span[2]')
    new_assists4 = assists4.text
    adr4 = driver.find_element("xpath",
                               '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[1]/div[4]/div[1]/table/tbody/tr[4]/td[10]/span/span[2]')
    new_adr4 = adr4.text
    fk4 = driver.find_element("xpath",
                              '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[1]/div[4]/div[1]/table/tbody/tr[4]/td[12]/span/span[2]')
    new_fk4 = fk4.text
    fb4 = driver.find_element("xpath",
                              '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[1]/div[4]/div[1]/table/tbody/tr[4]/td[13]/span/span[2]')
    new_fb4 = fb4.text
    name5 = driver.find_element("xpath",
                                '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[1]/div[4]/div[1]/table/tbody/tr[5]/td[1]/div/a/div[1]')
    new_name5 = name5.text
    acs5 = driver.find_element("xpath",
                               '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[1]/div[4]/div[1]/table/tbody/tr[5]/td[4]/span/span[2]')
    new_acs5 = acs5.text
    kill5 = driver.find_element("xpath",
                                '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[1]/div[4]/div[1]/table/tbody/tr[5]/td[5]/span/span[2]')
    new_kill5 = kill5.text
    death5 = driver.find_element("xpath",
                                 '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[1]/div[4]/div[1]/table/tbody/tr[5]/td[6]/span/span[2]/span[2]')
    new_death5 = death5.text
    assists5 = driver.find_element("xpath",
                                   '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[1]/div[4]/div[1]/table/tbody/tr[5]/td[7]/span/span[2]')
    new_assists5 = assists5.text
    adr5 = driver.find_element("xpath",
                               '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[1]/div[4]/div[1]/table/tbody/tr[5]/td[10]/span/span[2]')
    new_adr5 = adr5.text
    fk5 = driver.find_element("xpath",
                              '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[1]/div[4]/div[1]/table/tbody/tr[5]/td[12]/span/span[2]')
    new_fk5 = fk5.text
    fb5 = driver.find_element("xpath",
                              '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[1]/div[4]/div[1]/table/tbody/tr[5]/td[13]/span/span[2]')
    new_fb5 = fb5.text
    name6 = driver.find_element("xpath",
                                '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[1]/div[4]/div[2]/table/tbody/tr[1]/td[1]/div/a/div[1]')
    new_name6 = name6.text
    acs6 = driver.find_element("xpath",
                               '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[1]/div[4]/div[2]/table/tbody/tr[1]/td[4]/span/span[2]')
    new_acs6 = acs6.text
    kill6 = driver.find_element("xpath",
                                '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[1]/div[4]/div[2]/table/tbody/tr[1]/td[5]/span/span[2]')
    new_kill6 = kill6.text
    death6 = driver.find_element("xpath",
                                 '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[1]/div[4]/div[2]/table/tbody/tr[1]/td[6]/span/span[2]/span[2]')
    new_death6 = death6.text
    assists6 = driver.find_element("xpath",
                                   '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[1]/div[4]/div[2]/table/tbody/tr[1]/td[7]/span/span[2]')
    new_assists6 = assists6.text
    adr6 = driver.find_element("xpath",
                               '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[1]/div[4]/div[2]/table/tbody/tr[1]/td[10]/span/span[2]')
    new_adr6 = adr6.text
    fk6 = driver.find_element("xpath",
                              '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[1]/div[4]/div[2]/table/tbody/tr[1]/td[12]/span/span[2]')
    new_fk6 = fk6.text
    fb6 = driver.find_element("xpath",
                              '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[1]/div[4]/div[2]/table/tbody/tr[1]/td[13]/span/span[2]')
    new_fb6 = fb6.text
    name7 = driver.find_element("xpath",
                                '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[1]/div[4]/div[2]/table/tbody/tr[2]/td[1]/div/a/div[1]')
    new_name7 = name7.text
    acs7 = driver.find_element("xpath",
                               '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[1]/div[4]/div[2]/table/tbody/tr[2]/td[4]/span/span[2]')
    new_acs7 = acs7.text
    kill7 = driver.find_element("xpath",
                                '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[1]/div[4]/div[2]/table/tbody/tr[2]/td[5]/span/span[2]')
    new_kill7 = kill7.text
    death7 = driver.find_element("xpath",
                                 '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[1]/div[4]/div[2]/table/tbody/tr[2]/td[6]/span/span[2]/span[2]')
    new_death7 = death7.text
    assists7 = driver.find_element("xpath",
                                   '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[1]/div[4]/div[2]/table/tbody/tr[2]/td[7]/span/span[2]')
    new_assists7 = assists7.text
    adr7 = driver.find_element("xpath",
                               '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[1]/div[4]/div[2]/table/tbody/tr[2]/td[10]/span/span[2]')
    new_adr7 = adr7.text
    fk7 = driver.find_element("xpath",
                              '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[1]/div[4]/div[2]/table/tbody/tr[2]/td[12]/span/span[2]')
    new_fk7 = fk7.text
    fb7 = driver.find_element("xpath",
                              '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[1]/div[4]/div[2]/table/tbody/tr[2]/td[13]/span/span[2]')
    new_fb7 = fb7.text
    name8 = driver.find_element("xpath",
                                '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[1]/div[4]/div[2]/table/tbody/tr[3]/td[1]/div/a/div[1]')
    new_name8 = name8.text
    acs8 = driver.find_element("xpath",
                               '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[1]/div[4]/div[2]/table/tbody/tr[3]/td[4]/span/span[2]')
    new_acs8 = acs8.text
    kill8 = driver.find_element("xpath",
                                '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[1]/div[4]/div[2]/table/tbody/tr[3]/td[5]/span/span[2]')
    new_kill8 = kill8.text
    death8 = driver.find_element("xpath",
                                 '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[1]/div[4]/div[2]/table/tbody/tr[3]/td[6]/span/span[2]/span[2]')
    new_death8 = death8.text
    assists8 = driver.find_element("xpath",
                                   '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[1]/div[4]/div[2]/table/tbody/tr[3]/td[7]/span/span[2]')
    new_assists8 = assists8.text
    adr8 = driver.find_element("xpath",
                               '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[1]/div[4]/div[2]/table/tbody/tr[3]/td[10]/span/span[2]')
    new_adr8 = adr8.text
    fk8 = driver.find_element("xpath",
                              '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[1]/div[4]/div[2]/table/tbody/tr[3]/td[12]/span/span[2]')
    new_fk8 = fk8.text
    fb8 = driver.find_element("xpath",
                              '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[1]/div[4]/div[2]/table/tbody/tr[3]/td[13]/span/span[2]')
    new_fb8 = fb8.text
    name9 = driver.find_element("xpath",
                                '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[1]/div[4]/div[2]/table/tbody/tr[4]/td[1]/div/a/div[1]')
    new_name9 = name9.text
    acs9 = driver.find_element("xpath",
                               '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[1]/div[4]/div[2]/table/tbody/tr[4]/td[4]/span/span[2]')
    new_acs9 = acs9.text
    kill9 = driver.find_element("xpath",
                                '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[1]/div[4]/div[2]/table/tbody/tr[4]/td[5]/span/span[2]')
    new_kill9 = kill9.text
    death9 = driver.find_element("xpath",
                                 '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[1]/div[4]/div[2]/table/tbody/tr[4]/td[6]/span/span[2]/span[2]')
    new_death9 = death9.text
    assists9 = driver.find_element("xpath",
                                   '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[1]/div[4]/div[2]/table/tbody/tr[4]/td[7]/span/span[2]')
    new_assists9 = assists9.text
    adr9 = driver.find_element("xpath",
                               '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[1]/div[4]/div[2]/table/tbody/tr[4]/td[10]/span/span[2]')
    new_adr9 = adr9.text
    fk9 = driver.find_element("xpath",
                              '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[1]/div[4]/div[2]/table/tbody/tr[4]/td[12]/span/span[2]')
    new_fk9 = fk9.text
    fb9 = driver.find_element("xpath",
                              '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[1]/div[4]/div[2]/table/tbody/tr[4]/td[13]/span/span[2]')
    new_fb9 = fb9.text
    name10 = driver.find_element("xpath",
                                 '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[1]/div[4]/div[2]/table/tbody/tr[5]/td[1]/div/a/div[1]')
    new_name10 = name10.text
    acs10 = driver.find_element("xpath",
                                '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[1]/div[4]/div[2]/table/tbody/tr[5]/td[4]/span/span[2]')
    new_acs10 = acs10.text
    kill10 = driver.find_element("xpath",
                                 '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[1]/div[4]/div[2]/table/tbody/tr[5]/td[5]/span/span[2]')
    new_kill10 = kill10.text
    death10 = driver.find_element("xpath",
                                  '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[1]/div[4]/div[2]/table/tbody/tr[5]/td[6]/span/span[2]/span[2]')
    new_death10 = death10.text
    assists10 = driver.find_element("xpath",
                                    '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[1]/div[4]/div[2]/table/tbody/tr[5]/td[7]/span/span[2]')
    new_assists10 = assists10.text
    adr10 = driver.find_element("xpath",
                                '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[1]/div[4]/div[2]/table/tbody/tr[5]/td[10]/span/span[2]')
    new_adr10 = adr10.text
    fk10 = driver.find_element("xpath",
                               '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[1]/div[4]/div[2]/table/tbody/tr[5]/td[12]/span/span[2]')
    new_fk10 = fk10.text
    fb10 = driver.find_element("xpath",
                               '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[1]/div[4]/div[2]/table/tbody/tr[5]/td[13]/span/span[2]')
    new_fb10 = fb10.text
    new_kast = driver.find_element("xpath",
                                   '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[1]/div[4]/div[1]/table/tbody/tr[1]/td[9]/span/span[2]')
    new_kast = new_kast.text
    new_kast2 = driver.find_element("xpath",
                                    '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[1]/div[4]/div[1]/table/tbody/tr[2]/td[9]/span/span[2]')
    new_kast2 = new_kast2.text
    new_kast3 = driver.find_element("xpath",
                                    '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[1]/div[4]/div[1]/table/tbody/tr[3]/td[9]/span/span[2]')
    new_kast3 = new_kast3.text
    new_kast4 = driver.find_element("xpath",
                                    '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[1]/div[4]/div[1]/table/tbody/tr[4]/td[9]/span/span[2]')
    new_kast4 = new_kast4.text
    new_kast5 = driver.find_element("xpath",
                                    '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[1]/div[4]/div[1]/table/tbody/tr[5]/td[9]/span/span[2]')
    new_kast5 = new_kast5.text
    new_kast6 = driver.find_element("xpath",
                                    '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[1]/div[4]/div[2]/table/tbody/tr[1]/td[9]/span/span[2]')
    new_kast6 = new_kast6.text
    new_kast7 = driver.find_element("xpath",
                                    '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[1]/div[4]/div[2]/table/tbody/tr[2]/td[9]/span/span[2]')
    new_kast7 = new_kast7.text
    new_kast8 = driver.find_element("xpath",
                                    '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[1]/div[4]/div[2]/table/tbody/tr[3]/td[9]/span/span[2]')
    new_kast8 = new_kast8.text
    new_kast9 = driver.find_element("xpath",
                                    '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[1]/div[4]/div[2]/table/tbody/tr[4]/td[9]/span/span[2]')
    new_kast9 = new_kast9.text
    new_kast10 = driver.find_element("xpath",
                                     '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[1]/div[4]/div[2]/table/tbody/tr[5]/td[9]/span/span[2]')
    new_kast10 = new_kast10.text

    isim = [new_name, new_name2, new_name3, new_name4, new_name5, new_name6, new_name7, new_name8, new_name9,
            new_name10]
    acs = [new_acs, new_acs2, new_acs3, new_acs4, new_acs5, new_acs6, new_acs7, new_acs8, new_acs9, new_acs10]
    kill = [new_kill, new_kill2, new_kill3, new_kill4, new_kill5, new_kill6, new_kill7, new_kill8, new_kill9,
            new_kill10]
    death = [new_death, new_death2, new_death3, new_death4, new_death5, new_death6, new_death7, new_death8, new_death9,
             new_death10]
    assits = [new_assists, new_assists2, new_assists3, new_assists4, new_assists5, new_assists6, new_assists7,
              new_assists8, new_assists9, new_assists10]
    adr = [new_adr, new_adr2, new_adr3, new_adr4, new_adr5, new_adr6, new_adr7, new_adr8, new_adr9, new_adr10]
    kast = [new_kast, new_kast2, new_kast3, new_kast4, new_kast5, new_kast6, new_kast7, new_kast8, new_kast9,
            new_kast10]
    fk = [new_fk, new_fk2, new_fk3, new_fk4, new_fk5, new_fk6, new_fk7, new_fk8, new_fk9, new_fk10]
    fb = [new_fb, new_fb2, new_fb3, new_fb4, new_fb5, new_fb6, new_fb7, new_fb8, new_fb9, new_fb10]
    map = [pick1, pick1, pick1, pick1, pick1, pick1, pick1, pick1, pick1, pick1]
    team = [new_team1, new_team1, new_team1, new_team1, new_team1, new_team2, new_team2, new_team2, new_team2,
            new_team2]
    karsılasma = [vs, vs, vs, vs, vs, vs, vs, vs, vs, vs]
    Agent = ["-", "-", "-", "-", "-", "-", "-", "-", "-", "-"]
    final_score = [final, final, final, final, final, final, final, final, final, final]
    rating = ["-", "-", "-", "-", "-", "-", "-", "-", "-", "-"]
    ad = ["attack", "attack", "attack", "attack", "attack", "attack", "attack", "attack", "attack", "attack"]
    winner = ["-", "-", "-", "-", "-", "-", "-", "-", "-", "-"]
    attack_round_win = ["-", "-", "-", "-", "-", "-", "-", "-", "-", "-"]
    defense_round_win = ["-", "-", "-", "-", "-", "-", "-", "-", "-", "-"]
    pistol_round = ["-", "-", "-", "-", "-", "-", "-", "-", "-", "-"]
    df = pd.DataFrame({'isim': isim,
                       'acs': acs,
                       'kill': kill,
                       'death': death,
                       'assits': assits,
                       'adr': adr,
                       'kast': kast,
                       'fk': fk,
                       'fb': fb,
                       'map': map,
                       'Team': team,
                       'VS': karsılasma,
                       'Agent': Agent,
                       'Score': final_score,
                       'Rating': rating,
                       'Side': ad,
                       'Winner': winner,
                       'Attack Round': attack_round_win,
                       'Defense Round': defense_round_win,
                       'Pistol_Round': pistol_round})
    return df
map1_attack = map1_attack()
def map1_defense():
    name_DEF = driver.find_element("xpath",
                                   '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[1]/div[4]/div[1]/table/tbody/tr[1]/td[1]/div/a/div[1]')
    new_name_DEF = name_DEF.text
    acs_DEF = driver.find_element("xpath",
                                  '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[1]/div[4]/div[1]/table/tbody/tr[1]/td[4]/span/span[3]')
    new_acs_DEF = acs_DEF.text
    kill_DEF = driver.find_element("xpath",
                                   '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[1]/div[4]/div[1]/table/tbody/tr[1]/td[5]/span/span[3]')
    new_kill_DEF = kill_DEF.text
    death_DEF = driver.find_element("xpath",
                                    '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[1]/div[4]/div[1]/table/tbody/tr[1]/td[6]/span/span[2]/span[3]')
    new_death_DEF = death_DEF.text
    assists_DEF = driver.find_element("xpath",
                                      '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[1]/div[4]/div[1]/table/tbody/tr[1]/td[7]/span/span[3]')
    new_assists_DEF = assists_DEF.text
    adr_DEF = driver.find_element("xpath",
                                  '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[1]/div[4]/div[1]/table/tbody/tr[1]/td[10]/span/span[3]')
    new_adr_DEF = adr_DEF.text
    fk_DEF = driver.find_element("xpath",
                                 '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[1]/div[4]/div[1]/table/tbody/tr[1]/td[12]/span/span[3]')
    new_fk_DEF = fk_DEF.text
    fb_DEF = driver.find_element("xpath",
                                 '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[1]/div[4]/div[1]/table/tbody/tr[1]/td[13]/span/span[3]')
    new_fb_DEF = fb_DEF.text
    name2_DEF = driver.find_element("xpath",
                                    '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[1]/div[4]/div[1]/table/tbody/tr[2]/td[1]/div/a/div[1]')
    new_name2_DEF = name2_DEF.text
    acs2_DEF = driver.find_element("xpath",
                                   '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[1]/div[4]/div[1]/table/tbody/tr[2]/td[4]/span/span[3]')
    new_acs2_DEF = acs2_DEF.text
    kill2_DEF = driver.find_element("xpath",
                                    '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[1]/div[4]/div[1]/table/tbody/tr[2]/td[5]/span/span[3]')
    new_kill2_DEF = kill2_DEF.text
    death2_DEF = driver.find_element("xpath",
                                     '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[1]/div[4]/div[1]/table/tbody/tr[2]/td[6]/span/span[2]/span[3]')
    new_death2_DEF = death2_DEF.text
    assists2_DEF = driver.find_element("xpath",
                                       '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[1]/div[4]/div[1]/table/tbody/tr[2]/td[7]/span/span[3]')
    new_assists2_DEF = assists2_DEF.text
    adr2_DEF = driver.find_element("xpath",
                                   '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[1]/div[4]/div[1]/table/tbody/tr[2]/td[10]/span/span[3]')
    new_adr2_DEF = adr2_DEF.text
    fk2_DEF = driver.find_element("xpath",
                                  '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[1]/div[4]/div[1]/table/tbody/tr[2]/td[12]/span/span[3]')
    new_fk2_DEF = fk2_DEF.text
    fb2_DEF = driver.find_element("xpath",
                                  '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[1]/div[4]/div[1]/table/tbody/tr[2]/td[13]/span/span[3]')
    new_fb2_DEF = fb2_DEF.text
    name3_DEF = driver.find_element("xpath",
                                    '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[1]/div[4]/div[1]/table/tbody/tr[3]/td[1]/div/a/div[1]')
    new_name3_DEF = name3_DEF.text
    acs3_DEF = driver.find_element("xpath",
                                   '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[1]/div[4]/div[1]/table/tbody/tr[3]/td[4]/span/span[3]')
    new_acs3_DEF = acs3_DEF.text
    kill3_DEF = driver.find_element("xpath",
                                    '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[1]/div[4]/div[1]/table/tbody/tr[3]/td[5]/span/span[3]')
    new_kill3_DEF = kill3_DEF.text
    death3_DEF = driver.find_element("xpath",
                                     '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[1]/div[4]/div[1]/table/tbody/tr[3]/td[6]/span/span[2]/span[3]')
    new_death3_DEF = death3_DEF.text
    assists3_DEF = driver.find_element("xpath",
                                       '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[1]/div[4]/div[1]/table/tbody/tr[3]/td[7]/span/span[3]')
    new_assists3_DEF = assists3_DEF.text
    adr3_DEF = driver.find_element("xpath",
                                   '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[1]/div[4]/div[1]/table/tbody/tr[3]/td[10]/span/span[3]')
    new_adr3_DEF = adr3_DEF.text
    fk3_DEF = driver.find_element("xpath",
                                  '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[1]/div[4]/div[1]/table/tbody/tr[3]/td[12]/span/span[3]')
    new_fk3_DEF = fk3_DEF.text
    fb3_DEF = driver.find_element("xpath",
                                  '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[1]/div[4]/div[1]/table/tbody/tr[3]/td[13]/span/span[3]')
    new_fb3_DEF = fb3_DEF.text
    name4_DEF = driver.find_element("xpath",
                                    '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[1]/div[4]/div[1]/table/tbody/tr[4]/td[1]/div/a/div[1]')
    new_name4_DEF = name4_DEF.text
    acs4_DEF = driver.find_element("xpath",
                                   '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[1]/div[4]/div[1]/table/tbody/tr[4]/td[4]/span/span[3]')
    new_acs4_DEF = acs4_DEF.text
    kill4_DEF = driver.find_element("xpath",
                                    '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[1]/div[4]/div[1]/table/tbody/tr[4]/td[5]/span/span[3]')
    new_kill4_DEF = kill4_DEF.text
    death4_DEF = driver.find_element("xpath",
                                     '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[1]/div[4]/div[1]/table/tbody/tr[4]/td[6]/span/span[2]/span[3]')
    new_death4_DEF = death4_DEF.text
    assists4_DEF = driver.find_element("xpath",
                                       '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[1]/div[4]/div[1]/table/tbody/tr[4]/td[7]/span/span[3]')
    new_assists4_DEF = assists4_DEF.text
    adr4_DEF = driver.find_element("xpath",
                                   '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[1]/div[4]/div[1]/table/tbody/tr[4]/td[10]/span/span[3]')
    new_adr4_DEF = adr4_DEF.text
    fk4_DEF = driver.find_element("xpath",
                                  '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[1]/div[4]/div[1]/table/tbody/tr[4]/td[12]/span/span[3]')
    new_fk4_DEF = fk4_DEF.text
    fb4_DEF = driver.find_element("xpath",
                                  '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[1]/div[4]/div[1]/table/tbody/tr[4]/td[13]/span/span[3]')
    new_fb4_DEF = fb4_DEF.text
    name5_DEF = driver.find_element("xpath",
                                    '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[1]/div[4]/div[1]/table/tbody/tr[5]/td[1]/div/a/div[1]')
    new_name5_DEF = name5_DEF.text
    acs5_DEF = driver.find_element("xpath",
                                   '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[1]/div[4]/div[1]/table/tbody/tr[5]/td[4]/span/span[3]')
    new_acs5_DEF = acs5_DEF.text
    kill5_DEF = driver.find_element("xpath",
                                    '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[1]/div[4]/div[1]/table/tbody/tr[5]/td[5]/span/span[3]')
    new_kill5_DEF = kill5_DEF.text
    death5_DEF = driver.find_element("xpath",
                                     '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[1]/div[4]/div[1]/table/tbody/tr[5]/td[6]/span/span[2]/span[3]')
    new_death5_DEF = death5_DEF.text
    assists5_DEF = driver.find_element("xpath",
                                       '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[1]/div[4]/div[1]/table/tbody/tr[5]/td[7]/span/span[3]')
    new_assists5_DEF = assists5_DEF.text
    adr5_DEF = driver.find_element("xpath",
                                   '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[1]/div[4]/div[1]/table/tbody/tr[5]/td[10]/span/span[3]')
    new_adr5_DEF = adr5_DEF.text
    fk5_DEF = driver.find_element("xpath",
                                  '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[1]/div[4]/div[1]/table/tbody/tr[5]/td[12]/span/span[3]')
    new_fk5_DEF = fk5_DEF.text
    fb5_DEF = driver.find_element("xpath",
                                  '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[1]/div[4]/div[1]/table/tbody/tr[5]/td[13]/span/span[3]')
    new_fb5_DEF = fb5_DEF.text
    name6_DEF = driver.find_element("xpath",
                                    '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[1]/div[4]/div[2]/table/tbody/tr[1]/td[1]/div/a/div[1]')
    new_name6_DEF = name6_DEF.text
    acs6_DEF = driver.find_element("xpath",
                                   '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[1]/div[4]/div[2]/table/tbody/tr[1]/td[4]/span/span[3]')
    new_acs6_DEF = acs6_DEF.text
    kill6_DEF = driver.find_element("xpath",
                                    '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[1]/div[4]/div[2]/table/tbody/tr[1]/td[5]/span/span[3]')
    new_kill6_DEF = kill6_DEF.text
    death6_DEF = driver.find_element("xpath",
                                     '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[1]/div[4]/div[2]/table/tbody/tr[1]/td[6]/span/span[2]/span[3]')
    new_death6_DEF = death6_DEF.text
    assists6_DEF = driver.find_element("xpath",
                                       '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[1]/div[4]/div[2]/table/tbody/tr[1]/td[7]/span/span[3]')
    new_assists6_DEF = assists6_DEF.text
    adr6_DEF = driver.find_element("xpath",
                                   '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[1]/div[4]/div[2]/table/tbody/tr[1]/td[10]/span/span[3]')
    new_adr6_DEF = adr6_DEF.text
    fk6_DEF = driver.find_element("xpath",
                                  '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[1]/div[4]/div[2]/table/tbody/tr[1]/td[12]/span/span[3]')
    new_fk6_DEF = fk6_DEF.text
    fb6_DEF = driver.find_element("xpath",
                                  '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[1]/div[4]/div[2]/table/tbody/tr[1]/td[13]/span/span[3]')
    new_fb6_DEF = fb6_DEF.text
    name7_DEF = driver.find_element("xpath",
                                    '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[1]/div[4]/div[2]/table/tbody/tr[2]/td[1]/div/a/div[1]')
    new_name7_DEF = name7_DEF.text
    acs7_DEF = driver.find_element("xpath",
                                   '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[1]/div[4]/div[2]/table/tbody/tr[2]/td[4]/span/span[3]')
    new_acs7_DEF = acs7_DEF.text
    kill7_DEF = driver.find_element("xpath",
                                    '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[1]/div[4]/div[2]/table/tbody/tr[2]/td[5]/span/span[3]')
    new_kill7_DEF = kill7_DEF.text
    death7_DEF = driver.find_element("xpath",
                                     '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[1]/div[4]/div[2]/table/tbody/tr[2]/td[6]/span/span[2]/span[3]')
    new_death7_DEF = death7_DEF.text
    assists7_DEF = driver.find_element("xpath",
                                       '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[1]/div[4]/div[2]/table/tbody/tr[2]/td[7]/span/span[3]')
    new_assists7_DEF = assists7_DEF.text
    adr7_DEF = driver.find_element("xpath",
                                   '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[1]/div[4]/div[2]/table/tbody/tr[2]/td[10]/span/span[3]')
    new_adr7_DEF = adr7_DEF.text
    fk7_DEF = driver.find_element("xpath",
                                  '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[1]/div[4]/div[2]/table/tbody/tr[2]/td[12]/span/span[3]')
    new_fk7_DEF = fk7_DEF.text
    fb7_DEF = driver.find_element("xpath",
                                  '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[1]/div[4]/div[2]/table/tbody/tr[2]/td[13]/span/span[3]')
    new_fb7_DEF = fb7_DEF.text
    name8_DEF = driver.find_element("xpath",
                                    '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[1]/div[4]/div[2]/table/tbody/tr[3]/td[1]/div/a/div[1]')
    new_name8_DEF = name8_DEF.text
    acs8_DEF = driver.find_element("xpath",
                                   '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[1]/div[4]/div[2]/table/tbody/tr[3]/td[4]/span/span[3]')
    new_acs8_DEF = acs8_DEF.text
    kill8_DEF = driver.find_element("xpath",
                                    '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[1]/div[4]/div[2]/table/tbody/tr[3]/td[5]/span/span[3]')
    new_kill8_DEF = kill8_DEF.text
    death8_DEF = driver.find_element("xpath",
                                     '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[1]/div[4]/div[2]/table/tbody/tr[3]/td[6]/span/span[2]/span[3]')
    new_death8_DEF = death8_DEF.text
    assists8_DEF = driver.find_element("xpath",
                                       '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[1]/div[4]/div[2]/table/tbody/tr[3]/td[7]/span/span[3]')
    new_assists8_DEF = assists8_DEF.text
    adr8_DEF = driver.find_element("xpath",
                                   '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[1]/div[4]/div[2]/table/tbody/tr[3]/td[10]/span/span[3]')
    new_adr8_DEF = adr8_DEF.text
    fk8_DEF = driver.find_element("xpath",
                                  '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[1]/div[4]/div[2]/table/tbody/tr[3]/td[12]/span/span[3]')
    new_fk8_DEF = fk8_DEF.text
    fb8_DEF = driver.find_element("xpath",
                                  '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[1]/div[4]/div[2]/table/tbody/tr[3]/td[13]/span/span[3]')
    new_fb8_DEF = fb8_DEF.text
    name9_DEF = driver.find_element("xpath",
                                    '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[1]/div[4]/div[2]/table/tbody/tr[4]/td[1]/div/a/div[1]')
    new_name9_DEF = name9_DEF.text
    acs9_DEF = driver.find_element("xpath",
                                   '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[1]/div[4]/div[2]/table/tbody/tr[4]/td[4]/span/span[3]')
    new_acs9_DEF = acs9_DEF.text
    kill9_DEF = driver.find_element("xpath",
                                    '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[1]/div[4]/div[2]/table/tbody/tr[4]/td[5]/span/span[3]')
    new_kill9_DEF = kill9_DEF.text
    death9_DEF = driver.find_element("xpath",
                                     '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[1]/div[4]/div[2]/table/tbody/tr[4]/td[6]/span/span[2]/span[3]')
    new_death9_DEF = death9_DEF.text
    assists9_DEF = driver.find_element("xpath",
                                       '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[1]/div[4]/div[2]/table/tbody/tr[4]/td[7]/span/span[3]')
    new_assists9_DEF = assists9_DEF.text
    adr9_DEF = driver.find_element("xpath",
                                   '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[1]/div[4]/div[2]/table/tbody/tr[4]/td[10]/span/span[3]')
    new_adr9_DEF = adr9_DEF.text
    fk9_DEF = driver.find_element("xpath",
                                  '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[1]/div[4]/div[2]/table/tbody/tr[4]/td[12]/span/span[3]')
    new_fk9_DEF = fk9_DEF.text
    fb9_DEF = driver.find_element("xpath",
                                  '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[1]/div[4]/div[2]/table/tbody/tr[4]/td[13]/span/span[3]')
    new_fb9_DEF = fb9_DEF.text
    name10_DEF = driver.find_element("xpath",
                                     '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[1]/div[4]/div[2]/table/tbody/tr[5]/td[1]/div/a/div[1]')
    new_name10_DEF = name10_DEF.text
    acs10_DEF = driver.find_element("xpath",
                                    '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[1]/div[4]/div[2]/table/tbody/tr[5]/td[4]/span/span[3]')
    new_acs10_DEF = acs10_DEF.text
    kill10_DEF = driver.find_element("xpath",
                                     '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[1]/div[4]/div[2]/table/tbody/tr[5]/td[5]/span/span[3]')
    new_kill10_DEF = kill10_DEF.text
    death10_DEF = driver.find_element("xpath",
                                      '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[1]/div[4]/div[2]/table/tbody/tr[5]/td[6]/span/span[2]/span[3]')
    new_death10_DEF = death10_DEF.text
    assists10_DEF = driver.find_element("xpath",
                                        '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[1]/div[4]/div[2]/table/tbody/tr[5]/td[7]/span/span[3]')
    new_assists10_DEF = assists10_DEF.text
    adr10_DEF = driver.find_element("xpath",
                                    '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[1]/div[4]/div[2]/table/tbody/tr[5]/td[10]/span/span[3]')
    new_adr10_DEF = adr10_DEF.text
    fk10_DEF = driver.find_element("xpath",
                                   '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[1]/div[4]/div[2]/table/tbody/tr[5]/td[12]/span/span[3]')
    new_fk10_DEF = fk10_DEF.text
    fb10_DEF = driver.find_element("xpath",
                                   '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[1]/div[4]/div[2]/table/tbody/tr[5]/td[13]/span/span[3]')
    new_fb10_DEF = fb10_DEF.text
    new_kast = driver.find_element("xpath",
                                   '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[1]/div[4]/div[1]/table/tbody/tr[1]/td[9]/span/span[3]')
    new_kast = new_kast.text
    new_kast2 = driver.find_element("xpath",
                                    '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[1]/div[4]/div[1]/table/tbody/tr[2]/td[9]/span/span[3]')
    new_kast2 = new_kast2.text
    new_kast3 = driver.find_element("xpath",
                                    '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[1]/div[4]/div[1]/table/tbody/tr[3]/td[9]/span/span[3]')
    new_kast3 = new_kast3.text
    new_kast4 = driver.find_element("xpath",
                                    '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[1]/div[4]/div[1]/table/tbody/tr[4]/td[9]/span/span[3]')
    new_kast4 = new_kast4.text
    new_kast5 = driver.find_element("xpath",
                                    '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[1]/div[4]/div[1]/table/tbody/tr[5]/td[9]/span/span[3]')
    new_kast5 = new_kast5.text
    new_kast6 = driver.find_element("xpath",
                                    '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[1]/div[4]/div[2]/table/tbody/tr[1]/td[9]/span/span[3]')
    new_kast6 = new_kast6.text
    new_kast7 = driver.find_element("xpath",
                                    '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[1]/div[4]/div[2]/table/tbody/tr[2]/td[9]/span/span[3]')
    new_kast7 = new_kast7.text
    new_kast8 = driver.find_element("xpath",
                                    '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[1]/div[4]/div[2]/table/tbody/tr[3]/td[9]/span/span[3]')
    new_kast8 = new_kast8.text
    new_kast9 = driver.find_element("xpath",
                                    '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[1]/div[4]/div[2]/table/tbody/tr[4]/td[9]/span/span[3]')
    new_kast9 = new_kast9.text
    new_kast10 = driver.find_element("xpath",
                                     '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[1]/div[4]/div[2]/table/tbody/tr[5]/td[9]/span/span[3]')
    new_kast10 = new_kast10.text
    #######################

    isim_DEF = [new_name_DEF, new_name2_DEF, new_name3_DEF, new_name4_DEF, new_name5_DEF, new_name6_DEF, new_name7_DEF,
                new_name8_DEF, new_name9_DEF, new_name10_DEF]
    acs_DEF = [new_acs_DEF, new_acs2_DEF, new_acs3_DEF, new_acs4_DEF, new_acs5_DEF, new_acs6_DEF, new_acs7_DEF,
               new_acs8_DEF, new_acs9_DEF, new_acs10_DEF]
    kill_DEF = [new_kill_DEF, new_kill2_DEF, new_kill3_DEF, new_kill4_DEF, new_kill5_DEF, new_kill6_DEF, new_kill7_DEF,
                new_kill8_DEF, new_kill9_DEF,
                new_kill10_DEF]
    death_DEF = [new_death_DEF, new_death2_DEF, new_death3_DEF, new_death4_DEF, new_death5_DEF, new_death6_DEF,
                 new_death7_DEF, new_death8_DEF, new_death9_DEF,
                 new_death10_DEF]
    assits_DEF = [new_assists_DEF, new_assists2_DEF, new_assists3_DEF, new_assists4_DEF, new_assists5_DEF,
                  new_assists6_DEF, new_assists7_DEF,
                  new_assists8_DEF, new_assists9_DEF, new_assists10_DEF]
    adr_DEF = [new_adr_DEF, new_adr2_DEF, new_adr3_DEF, new_adr4_DEF, new_adr5_DEF, new_adr6_DEF, new_adr7_DEF,
               new_adr8_DEF, new_adr9_DEF, new_adr10_DEF]
    kast_DEF = [new_kast, new_kast2, new_kast3, new_kast4, new_kast5, new_kast6, new_kast7,
                new_kast8, new_kast9, new_kast10]
    fk_DEF = [new_fk_DEF, new_fk2_DEF, new_fk3_DEF, new_fk4_DEF, new_fk5_DEF, new_fk6_DEF, new_fk7_DEF, new_fk8_DEF,
              new_fk9_DEF, new_fk10_DEF]
    fb_DEF = [new_fb_DEF, new_fb2_DEF, new_fb3_DEF, new_fb4_DEF, new_fb5_DEF, new_fb6_DEF, new_fb7_DEF, new_fb8_DEF,
              new_fb9_DEF, new_fb10_DEF]
    map = [pick1, pick1, pick1, pick1, pick1, pick1, pick1, pick1, pick1, pick1]
    team = [new_team1, new_team1, new_team1, new_team1, new_team1, new_team2, new_team2, new_team2, new_team2,
            new_team2]
    karsılasma = [vs, vs, vs, vs, vs, vs, vs, vs, vs, vs]
    Agent = ["-", "-", "-", "-", "-", "-", "-", "-", "-", "-"]
    final_score = [final, final, final, final, final, final, final, final, final, final]
    rating = ["-", "-", "-", "-", "-", "-", "-", "-", "-", "-"]
    ad = ["defense", "defense", "defense", "defense", "defense", "defense", "defense", "defense", "defense", "defense"]
    winner = ["-", "-", "-", "-", "-", "-", "-", "-", "-", "-"]
    dff = pd.DataFrame({'isim': isim_DEF,
                        'acs': acs_DEF,
                        'kill': kill_DEF,
                        'death': death_DEF,
                        'assits': assits_DEF,
                        'adr': adr_DEF,
                        'kast': kast_DEF,
                        'fk': fk_DEF,
                        'fb': fb_DEF,
                        'map': map,
                        'Team': team,
                        'VS': karsılasma,
                        'Agent': Agent,
                        'Score': final_score,
                        'Rating': rating,
                        'Side': ad,
                        'Winner': winner})
    return dff
map1_defense = map1_defense()
###### MAP_2 #####
def map2_attack():
    name = driver.find_element("xpath",
                               '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[3]/div[4]/div[1]/table/tbody/tr[1]/td[1]/div/a/div[1]')
    new_name = name.text  # //*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[3]/div[4]/div[1]/table/tbody/tr[3]/td[1]/div/a/div[1]
    acs = driver.find_element("xpath",
                              '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[3]/div[4]/div[1]/table/tbody/tr[1]/td[4]/span/span[2]')
    new_acs = acs.text  # //*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[3]/div[4]/div[1]/table/tbody/tr[2]/td[4]/span/span[2]--tr
    kill = driver.find_element("xpath",
                               '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[3]/div[4]/div[1]/table/tbody/tr[1]/td[5]/span/span[2]')
    new_kill = kill.text  # //*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[3]/div[4]/div[1]/table/tbody/tr[2]/td[5]/span/span[2]--tr
    death = driver.find_element("xpath",
                                '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[3]/div[4]/div[1]/table/tbody/tr[1]/td[6]/span/span[2]/span[2]')
    new_death = death.text  # //*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[3]/div[4]/div[1]/table/tbody/tr[2]/td[6]/span/span[2]/span[2]
    assists = driver.find_element("xpath",
                                  '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[3]/div[4]/div[1]/table/tbody/tr[1]/td[7]/span/span[2]')
    new_assists = assists.text
    adr = driver.find_element("xpath",
                              '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[3]/div[4]/div[1]/table/tbody/tr[1]/td[10]/span/span[2]')
    new_adr = adr.text
    kast = driver.find_element("xpath",
                               '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[3]/div[4]/div[1]/table/tbody/tr[1]/td[9]/span/span[2]')
    new_kast = kast.text
    fk = driver.find_element("xpath",
                             '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[3]/div[4]/div[1]/table/tbody/tr[1]/td[12]/span/span[2]')
    new_fk = fk.text
    fb = driver.find_element("xpath",
                             '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[3]/div[4]/div[1]/table/tbody/tr[1]/td[13]/span/span[2]')
    new_fb = fb.text
    name2 = driver.find_element("xpath",
                                '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[3]/div[4]/div[1]/table/tbody/tr[2]/td[1]/div/a/div[1]')
    new_name2 = name2.text
    acs2 = driver.find_element("xpath",
                               '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[3]/div[4]/div[1]/table/tbody/tr[2]/td[4]/span/span[2]')
    new_acs2 = acs2.text
    kill2 = driver.find_element("xpath",
                                '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[3]/div[4]/div[1]/table/tbody/tr[2]/td[5]/span/span[2]')
    new_kill2 = kill2.text
    death2 = driver.find_element("xpath",
                                 '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[3]/div[4]/div[1]/table/tbody/tr[2]/td[6]/span/span[2]/span[2]')
    new_death2 = death2.text
    assists2 = driver.find_element("xpath",
                                   '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[3]/div[4]/div[1]/table/tbody/tr[2]/td[7]/span/span[2]')
    new_assists2 = assists2.text
    adr2 = driver.find_element("xpath",
                               '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[3]/div[4]/div[1]/table/tbody/tr[2]/td[10]/span/span[2]')
    new_adr2 = adr2.text
    kast2 = driver.find_element("xpath",
                                '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[3]/div[4]/div[1]/table/tbody/tr[2]/td[9]/span/span[2]')
    new_kast2 = kast2.text
    fk2 = driver.find_element("xpath",
                              '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[3]/div[4]/div[1]/table/tbody/tr[2]/td[12]/span/span[2]')
    new_fk2 = fk2.text
    fb2 = driver.find_element("xpath",
                              '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[3]/div[4]/div[1]/table/tbody/tr[2]/td[13]/span/span[2]')
    new_fb2 = fb2.text
    name3 = driver.find_element("xpath",
                                '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[3]/div[4]/div[1]/table/tbody/tr[3]/td[1]/div/a/div[1]')
    new_name3 = name3.text
    acs3 = driver.find_element("xpath",
                               '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[3]/div[4]/div[1]/table/tbody/tr[3]/td[4]/span/span[2]')
    new_acs3 = acs3.text
    kill3 = driver.find_element("xpath",
                                '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[3]/div[4]/div[1]/table/tbody/tr[3]/td[5]/span/span[2]')
    new_kill3 = kill3.text
    death3 = driver.find_element("xpath",
                                 '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[3]/div[4]/div[1]/table/tbody/tr[3]/td[6]/span/span[2]/span[2]')
    new_death3 = death3.text
    assists3 = driver.find_element("xpath",
                                   '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[3]/div[4]/div[1]/table/tbody/tr[3]/td[7]/span/span[2]')
    new_assists3 = assists3.text
    adr3 = driver.find_element("xpath",
                               '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[3]/div[4]/div[1]/table/tbody/tr[3]/td[10]/span/span[2]')
    new_adr3 = adr3.text
    kast3 = driver.find_element("xpath",
                                '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[3]/div[4]/div[1]/table/tbody/tr[3]/td[9]/span/span[2]')
    new_kast3 = kast3.text
    fk3 = driver.find_element("xpath",
                              '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[3]/div[4]/div[1]/table/tbody/tr[3]/td[12]/span/span[2]')
    new_fk3 = fk3.text
    fb3 = driver.find_element("xpath",
                              '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[3]/div[4]/div[1]/table/tbody/tr[3]/td[13]/span/span[2]')
    new_fb3 = fb3.text
    name4 = driver.find_element("xpath",
                                '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[3]/div[4]/div[1]/table/tbody/tr[4]/td[1]/div/a/div[1]')
    new_name4 = name4.text
    acs4 = driver.find_element("xpath",
                               '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[3]/div[4]/div[1]/table/tbody/tr[4]/td[4]/span/span[2]')
    new_acs4 = acs4.text
    kill4 = driver.find_element("xpath",
                                '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[3]/div[4]/div[1]/table/tbody/tr[4]/td[5]/span/span[2]')
    new_kill4 = kill4.text
    death4 = driver.find_element("xpath",
                                 '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[3]/div[4]/div[1]/table/tbody/tr[4]/td[6]/span/span[2]/span[2]')
    new_death4 = death4.text
    assists4 = driver.find_element("xpath",
                                   '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[3]/div[4]/div[1]/table/tbody/tr[4]/td[7]/span/span[2]')
    new_assists4 = assists4.text
    adr4 = driver.find_element("xpath",
                               '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[3]/div[4]/div[1]/table/tbody/tr[4]/td[10]/span/span[2]')
    new_adr4 = adr4.text
    kast4 = driver.find_element("xpath",
                                '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[3]/div[4]/div[1]/table/tbody/tr[4]/td[9]/span/span[2]')
    new_kast4 = kast4.text
    fk4 = driver.find_element("xpath",
                              '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[3]/div[4]/div[1]/table/tbody/tr[4]/td[12]/span/span[2]')
    new_fk4 = fk4.text
    fb4 = driver.find_element("xpath",
                              '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[3]/div[4]/div[1]/table/tbody/tr[4]/td[13]/span/span[2]')
    new_fb4 = fb4.text
    name5 = driver.find_element("xpath",
                                '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[3]/div[4]/div[1]/table/tbody/tr[5]/td[1]/div/a/div[1]')
    new_name5 = name5.text
    acs5 = driver.find_element("xpath",
                               '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[3]/div[4]/div[1]/table/tbody/tr[5]/td[4]/span/span[2]')
    new_acs5 = acs5.text
    kill5 = driver.find_element("xpath",
                                '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[3]/div[4]/div[1]/table/tbody/tr[5]/td[5]/span/span[2]')
    new_kill5 = kill5.text
    death5 = driver.find_element("xpath",
                                 '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[3]/div[4]/div[1]/table/tbody/tr[5]/td[6]/span/span[2]/span[2]')
    new_death5 = death5.text
    assists5 = driver.find_element("xpath",
                                   '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[3]/div[4]/div[1]/table/tbody/tr[5]/td[7]/span/span[2]')
    new_assists5 = assists5.text
    adr5 = driver.find_element("xpath",
                               '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[3]/div[4]/div[1]/table/tbody/tr[5]/td[10]/span/span[2]')
    new_adr5 = adr5.text
    kast5 = driver.find_element("xpath",
                                '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[3]/div[4]/div[1]/table/tbody/tr[5]/td[9]/span/span[2]')
    new_kast5 = kast5.text
    fk5 = driver.find_element("xpath",
                              '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[3]/div[4]/div[1]/table/tbody/tr[5]/td[12]/span/span[2]')
    new_fk5 = fk5.text
    fb5 = driver.find_element("xpath",
                              '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[3]/div[4]/div[1]/table/tbody/tr[5]/td[13]/span/span[2]')
    new_fb5 = fb5.text
    name6 = driver.find_element("xpath",
                                '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[3]/div[4]/div[2]/table/tbody/tr[1]/td[1]/div/a/div[1]')
    new_name6 = name6.text  # //*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[3]/div[4]/div[2]/table/tbody/tr[1]/td[1]/div/a/div[1]
    acs6 = driver.find_element("xpath",
                               '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[3]/div[4]/div[2]/table/tbody/tr[1]/td[4]/span/span[2]')
    new_acs6 = acs6.text
    kill6 = driver.find_element("xpath",
                                '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[3]/div[4]/div[2]/table/tbody/tr[1]/td[5]/span/span[2]')
    new_kill6 = kill6.text
    death6 = driver.find_element("xpath",
                                 '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[3]/div[4]/div[2]/table/tbody/tr[1]/td[6]/span/span[2]/span[2]')
    new_death6 = death6.text
    assists6 = driver.find_element("xpath",
                                   '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[3]/div[4]/div[2]/table/tbody/tr[1]/td[7]/span/span[2]')
    new_assists6 = assists6.text
    adr6 = driver.find_element("xpath",
                               '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[3]/div[4]/div[2]/table/tbody/tr[1]/td[10]/span/span[2]')
    new_adr6 = adr6.text
    kast6 = driver.find_element("xpath",
                                '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[3]/div[4]/div[2]/table/tbody/tr[1]/td[9]/span/span[2]')
    new_kast6 = kast6.text
    fk6 = driver.find_element("xpath",
                              '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[3]/div[4]/div[2]/table/tbody/tr[1]/td[12]/span/span[2]')
    new_fk6 = fk6.text
    fb6 = driver.find_element("xpath",
                              '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[3]/div[4]/div[2]/table/tbody/tr[1]/td[13]/span/span[2]')
    new_fb6 = fb6.text
    name7 = driver.find_element("xpath",
                                '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[3]/div[4]/div[2]/table/tbody/tr[2]/td[1]/div/a/div[1]')
    new_name7 = name7.text  # //*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[3]/div[4]/div[2]/table/tbody/tr[1]/td[1]/div/a/div[1]
    acs7 = driver.find_element("xpath",
                               '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[3]/div[4]/div[2]/table/tbody/tr[2]/td[4]/span/span[2]')
    new_acs7 = acs7.text  # //*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[3]/div[4]/div[2]/table/tbody/tr[2]/td[4]/span/span[2]
    kill7 = driver.find_element("xpath",
                                '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[3]/div[4]/div[2]/table/tbody/tr[2]/td[5]/span/span[2]')
    new_kill7 = kill7.text  # //*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[3]/div[4]/div[2]/table/tbody/tr[2]/td[5]/span/span[2]
    death7 = driver.find_element("xpath",
                                 '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[3]/div[4]/div[2]/table/tbody/tr[2]/td[6]/span/span[2]/span[2]')
    new_death7 = death7.text
    assists7 = driver.find_element("xpath",
                                   '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[3]/div[4]/div[2]/table/tbody/tr[2]/td[7]/span/span[2]')
    new_assists7 = assists7.text
    adr7 = driver.find_element("xpath",
                               '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[3]/div[4]/div[2]/table/tbody/tr[2]/td[10]/span/span[2]')
    new_adr7 = adr7.text
    kast7 = driver.find_element("xpath",
                                '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[3]/div[4]/div[2]/table/tbody/tr[2]/td[9]/span/span[2]')
    new_kast7 = kast7.text
    fk7 = driver.find_element("xpath",
                              '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[3]/div[4]/div[2]/table/tbody/tr[2]/td[12]/span/span[2]')
    new_fk7 = fk7.text
    fb7 = driver.find_element("xpath",
                              '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[3]/div[4]/div[2]/table/tbody/tr[2]/td[13]/span/span[2]')
    new_fb7 = fb7.text
    name8 = driver.find_element("xpath",
                                '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[3]/div[4]/div[2]/table/tbody/tr[3]/td[1]/div/a/div[1]')
    new_name8 = name8.text
    acs8 = driver.find_element("xpath",
                               '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[3]/div[4]/div[2]/table/tbody/tr[3]/td[4]/span/span[2]')
    new_acs8 = acs8.text
    kill8 = driver.find_element("xpath",
                                '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[3]/div[4]/div[2]/table/tbody/tr[3]/td[5]/span/span[2]')
    new_kill8 = kill8.text  # //*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[3]/div[4]/div[2]/table/tbody/tr[3]/td[5]/span/span[2]
    death8 = driver.find_element("xpath",
                                 '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[3]/div[4]/div[2]/table/tbody/tr[3]/td[6]/span/span[2]/span[2]')
    new_death8 = death8.text
    assists8 = driver.find_element("xpath",
                                   '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[3]/div[4]/div[2]/table/tbody/tr[3]/td[7]/span/span[2]')
    new_assists8 = assists8.text
    adr8 = driver.find_element("xpath",
                               '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[3]/div[4]/div[2]/table/tbody/tr[3]/td[10]/span/span[2]')
    new_adr8 = adr8.text
    kast8 = driver.find_element("xpath",
                                '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[3]/div[4]/div[2]/table/tbody/tr[3]/td[9]/span/span[2]')
    new_kast8 = kast8.text
    fk8 = driver.find_element("xpath",
                              '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[3]/div[4]/div[2]/table/tbody/tr[3]/td[12]/span/span[2]')
    new_fk8 = fk8.text
    fb8 = driver.find_element("xpath",
                              '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[3]/div[4]/div[2]/table/tbody/tr[3]/td[13]/span/span[2]')
    new_fb8 = fb8.text
    name9 = driver.find_element("xpath",
                                '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[3]/div[4]/div[2]/table/tbody/tr[4]/td[1]/div/a/div[1]')
    new_name9 = name9.text
    acs9 = driver.find_element("xpath",
                               '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[3]/div[4]/div[2]/table/tbody/tr[4]/td[4]/span/span[2]')
    new_acs9 = acs9.text
    kill9 = driver.find_element("xpath",
                                '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[3]/div[4]/div[2]/table/tbody/tr[4]/td[5]/span/span[2]')
    new_kill9 = kill9.text
    death9 = driver.find_element("xpath",
                                 '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[3]/div[4]/div[2]/table/tbody/tr[4]/td[6]/span/span[2]/span[2]')
    new_death9 = death9.text
    assists9 = driver.find_element("xpath",
                                   '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[3]/div[4]/div[2]/table/tbody/tr[4]/td[7]/span/span[2]')
    new_assists9 = assists9.text
    adr9 = driver.find_element("xpath",
                               '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[3]/div[4]/div[2]/table/tbody/tr[4]/td[10]/span/span[2]')
    new_adr9 = adr9.text
    kast9 = driver.find_element("xpath",
                                '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[3]/div[4]/div[2]/table/tbody/tr[4]/td[9]/span/span[2]')
    new_kast9 = kast9.text
    fk9 = driver.find_element("xpath",
                              '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[3]/div[4]/div[2]/table/tbody/tr[4]/td[12]/span/span[2]')
    new_fk9 = fk9.text
    fb9 = driver.find_element("xpath",
                              '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[3]/div[4]/div[2]/table/tbody/tr[4]/td[13]/span/span[2]')
    new_fb9 = fb9.text
    name10 = driver.find_element("xpath",
                                 '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[3]/div[4]/div[2]/table/tbody/tr[5]/td[1]/div/a/div[1]')
    new_name10 = name10.text
    acs10 = driver.find_element("xpath",
                                '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[3]/div[4]/div[2]/table/tbody/tr[5]/td[4]/span/span[2]')
    new_acs10 = acs10.text
    kill10 = driver.find_element("xpath",
                                 '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[3]/div[4]/div[2]/table/tbody/tr[5]/td[5]/span/span[2]')
    new_kill10 = kill10.text
    death10 = driver.find_element("xpath",
                                  '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[3]/div[4]/div[2]/table/tbody/tr[5]/td[6]/span/span[2]/span[2]')
    new_death10 = death10.text
    assists10 = driver.find_element("xpath",
                                    '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[3]/div[4]/div[2]/table/tbody/tr[5]/td[7]/span/span[2]')
    new_assists10 = assists10.text
    adr10 = driver.find_element("xpath",
                                '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[3]/div[4]/div[2]/table/tbody/tr[5]/td[10]/span/span[2]')
    new_adr10 = adr10.text
    kast10 = driver.find_element("xpath",
                                 '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[3]/div[4]/div[2]/table/tbody/tr[5]/td[9]/span/span[2]')
    new_kast10 = kast10.text
    fk10 = driver.find_element("xpath",
                               '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[3]/div[4]/div[2]/table/tbody/tr[5]/td[12]/span/span[2]')
    new_fk10 = fk10.text
    fb10 = driver.find_element("xpath",
                               '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[3]/div[4]/div[2]/table/tbody/tr[5]/td[13]/span/span[2]')
    new_fb10 = fb10.text

    isim = [new_name, new_name2, new_name3, new_name4, new_name5, new_name6, new_name7, new_name8, new_name9,
            new_name10]
    acs = [new_acs, new_acs2, new_acs3, new_acs4, new_acs5, new_acs6, new_acs7, new_acs8, new_acs9, new_acs10]
    kill = [new_kill, new_kill2, new_kill3, new_kill4, new_kill5, new_kill6, new_kill7, new_kill8, new_kill9,
            new_kill10]
    death = [new_death, new_death2, new_death3, new_death4, new_death5, new_death6, new_death7, new_death8, new_death9,
             new_death10]
    assits = [new_assists, new_assists2, new_assists3, new_assists4, new_assists5, new_assists6, new_assists7,
              new_assists8, new_assists9, new_assists10]
    adr = [new_adr, new_adr2, new_adr3, new_adr4, new_adr5, new_adr6, new_adr7, new_adr8, new_adr9, new_adr10]
    kast = [new_kast, new_kast2, new_kast3, new_kast4, new_kast5, new_kast6, new_kast7, new_kast8, new_kast9,
            new_kast10]
    fk = [new_fk, new_fk2, new_fk3, new_fk4, new_fk5, new_fk6, new_fk7, new_fk8, new_fk9, new_fk10]
    fb = [new_fb, new_fb2, new_fb3, new_fb4, new_fb5, new_fb6, new_fb7, new_fb8, new_fb9, new_fb10]
    map = [pick2, pick2, pick2, pick2, pick2, pick2, pick2, pick2, pick2, pick2]
    team = [new_team1, new_team1, new_team1, new_team1, new_team1, new_team2, new_team2, new_team2, new_team2,
            new_team2]
    karsılasma = [vs, vs, vs, vs, vs, vs, vs, vs, vs, vs]
    Agent = ["-", "-", "-", "-", "-", "-", "-", "-", "-", "-"]
    final_score = [final, final, final, final, final, final, final, final, final, final]
    rating = ["-", "-", "-", "-", "-", "-", "-", "-", "-", "-"]
    ad = ["attack", "attack", "attack", "attack", "attack", "attack", "attack", "attack", "attack", "attack"]
    winner = ["-", "-", "-", "-", "-", "-", "-", "-", "-", "-"]
    attack_round_win = ["-", "-", "-", "-", "-", "-", "-", "-", "-", "-"]
    defense_round_win = ["-", "-", "-", "-", "-", "-", "-", "-", "-", "-"]
    pistol_round = ["-", "-", "-", "-", "-", "-", "-", "-", "-", "-"]
    df = pd.DataFrame({'isim': isim,
                       'acs': acs,
                       'kill': kill,
                       'death': death,
                       'assits': assits,
                       'adr': adr,
                       'kast': kast,
                       'fk': fk,
                       'fb': fb,
                       'map': map,
                       'Team': team,
                       'VS': karsılasma,
                       'Agent': Agent,
                       'Score': final_score,
                       'Rating': rating,
                       'Side': ad,
                       'Winner': winner,
                       'Attack Round': attack_round_win,
                       'Defense Round': defense_round_win,
                       'Pistol_Round': pistol_round})
    return df
map2_attack = map2_attack()
def map2_defense():
    name_DEF = driver.find_element("xpath",
                                   '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[3]/div[4]/div[1]/table/tbody/tr[1]/td[1]/div/a/div[1]')
    new_name_DEF = name_DEF.text
    acs_DEF = driver.find_element("xpath",
                                  '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[3]/div[4]/div[1]/table/tbody/tr[1]/td[4]/span/span[3]')
    new_acs_DEF = acs_DEF.text
    kill_DEF = driver.find_element("xpath",
                                   '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[3]/div[4]/div[1]/table/tbody/tr[1]/td[5]/span/span[3]')
    new_kill_DEF = kill_DEF.text
    death_DEF = driver.find_element("xpath",
                                    '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[3]/div[4]/div[1]/table/tbody/tr[1]/td[6]/span/span[2]/span[3]')
    new_death_DEF = death_DEF.text
    assists_DEF = driver.find_element("xpath",
                                      '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[3]/div[4]/div[1]/table/tbody/tr[1]/td[7]/span/span[3]')
    new_assists_DEF = assists_DEF.text
    adr_DEF = driver.find_element("xpath",
                                  '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[3]/div[4]/div[1]/table/tbody/tr[1]/td[10]/span/span[3]')
    new_adr_DEF = adr_DEF.text
    kast_DEF = driver.find_element("xpath",
                                   '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[3]/div[4]/div[1]/table/tbody/tr[1]/td[9]/span/span[3]')
    new_kast_DEF = kast_DEF.text
    fk_DEF = driver.find_element("xpath",
                                 '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[3]/div[4]/div[1]/table/tbody/tr[1]/td[12]/span/span[3]')
    new_fk_DEF = fk_DEF.text
    fb_DEF = driver.find_element("xpath",
                                 '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[3]/div[4]/div[1]/table/tbody/tr[1]/td[13]/span/span[3]')
    new_fb_DEF = fb_DEF.text
    name2_DEF = driver.find_element("xpath",
                                    '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[3]/div[4]/div[1]/table/tbody/tr[2]/td[1]/div/a/div[1]')
    new_name2_DEF = name2_DEF.text
    acs2_DEF = driver.find_element("xpath",
                                   '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[3]/div[4]/div[1]/table/tbody/tr[2]/td[4]/span/span[3]')
    new_acs2_DEF = acs2_DEF.text
    kill2_DEF = driver.find_element("xpath",
                                    '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[3]/div[4]/div[1]/table/tbody/tr[2]/td[5]/span/span[3]')
    new_kill2_DEF = kill2_DEF.text
    death2_DEF = driver.find_element("xpath",
                                     '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[3]/div[4]/div[1]/table/tbody/tr[2]/td[6]/span/span[2]/span[3]')
    new_death2_DEF = death2_DEF.text
    assists2_DEF = driver.find_element("xpath",
                                       '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[3]/div[4]/div[1]/table/tbody/tr[2]/td[7]/span/span[3]')
    new_assists2_DEF = assists2_DEF.text
    adr2_DEF = driver.find_element("xpath",
                                   '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[3]/div[4]/div[1]/table/tbody/tr[2]/td[10]/span/span[3]')
    new_adr2_DEF = adr2_DEF.text
    kast_DEF2 = driver.find_element("xpath",
                                    '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[3]/div[4]/div[1]/table/tbody/tr[2]/td[9]/span/span[3]')
    new_kast_DEF2 = kast_DEF2.text
    fk2_DEF = driver.find_element("xpath",
                                  '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[3]/div[4]/div[1]/table/tbody/tr[2]/td[12]/span/span[3]')
    new_fk2_DEF = fk2_DEF.text
    fb2_DEF = driver.find_element("xpath",
                                  '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[3]/div[4]/div[1]/table/tbody/tr[2]/td[13]/span/span[3]')
    new_fb2_DEF = fb2_DEF.text
    name3_DEF = driver.find_element("xpath",
                                    '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[3]/div[4]/div[1]/table/tbody/tr[3]/td[1]/div/a/div[1]')
    new_name3_DEF = name3_DEF.text
    acs3_DEF = driver.find_element("xpath",
                                   '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[3]/div[4]/div[1]/table/tbody/tr[3]/td[4]/span/span[3]')
    new_acs3_DEF = acs3_DEF.text
    kill3_DEF = driver.find_element("xpath",
                                    '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[3]/div[4]/div[1]/table/tbody/tr[3]/td[5]/span/span[3]')
    new_kill3_DEF = kill3_DEF.text
    death3_DEF = driver.find_element("xpath",
                                     '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[3]/div[4]/div[1]/table/tbody/tr[3]/td[6]/span/span[2]/span[3]')
    new_death3_DEF = death3_DEF.text
    assists3_DEF = driver.find_element("xpath",
                                       '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[3]/div[4]/div[1]/table/tbody/tr[3]/td[7]/span/span[3]')
    new_assists3_DEF = assists3_DEF.text
    adr3_DEF = driver.find_element("xpath",
                                   '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[3]/div[4]/div[1]/table/tbody/tr[3]/td[10]/span/span[3]')
    new_adr3_DEF = adr3_DEF.text
    kast_DEF3 = driver.find_element("xpath",
                                    '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[3]/div[4]/div[1]/table/tbody/tr[3]/td[9]/span/span[3]')
    new_kast_DEF3 = kast_DEF3.text
    fk3_DEF = driver.find_element("xpath",
                                  '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[3]/div[4]/div[1]/table/tbody/tr[3]/td[12]/span/span[3]')
    new_fk3_DEF = fk3_DEF.text
    fb3_DEF = driver.find_element("xpath",
                                  '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[3]/div[4]/div[1]/table/tbody/tr[3]/td[13]/span/span[3]')
    new_fb3_DEF = fb3_DEF.text
    name4_DEF = driver.find_element("xpath",
                                    '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[3]/div[4]/div[1]/table/tbody/tr[4]/td[1]/div/a/div[1]')
    new_name4_DEF = name4_DEF.text
    acs4_DEF = driver.find_element("xpath",
                                   '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[3]/div[4]/div[1]/table/tbody/tr[4]/td[4]/span/span[3]')
    new_acs4_DEF = acs4_DEF.text
    kill4_DEF = driver.find_element("xpath",
                                    '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[3]/div[4]/div[1]/table/tbody/tr[4]/td[5]/span/span[3]')
    new_kill4_DEF = kill4_DEF.text
    death4_DEF = driver.find_element("xpath",
                                     '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[3]/div[4]/div[1]/table/tbody/tr[4]/td[6]/span/span[2]/span[3]')
    new_death4_DEF = death4_DEF.text
    assists4_DEF = driver.find_element("xpath",
                                       '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[3]/div[4]/div[1]/table/tbody/tr[4]/td[7]/span/span[3]')
    new_assists4_DEF = assists4_DEF.text
    adr4_DEF = driver.find_element("xpath",
                                   '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[3]/div[4]/div[1]/table/tbody/tr[4]/td[10]/span/span[3]')
    new_adr4_DEF = adr4_DEF.text
    kast_DEF4 = driver.find_element("xpath",
                                    '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[3]/div[4]/div[1]/table/tbody/tr[4]/td[9]/span/span[3]')
    new_kast_DEF4 = kast_DEF4.text
    fk4_DEF = driver.find_element("xpath",
                                  '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[3]/div[4]/div[1]/table/tbody/tr[4]/td[12]/span/span[3]')
    new_fk4_DEF = fk4_DEF.text
    fb4_DEF = driver.find_element("xpath",
                                  '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[3]/div[4]/div[1]/table/tbody/tr[4]/td[13]/span/span[3]')
    new_fb4_DEF = fb4_DEF.text
    name5_DEF = driver.find_element("xpath",
                                    '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[3]/div[4]/div[1]/table/tbody/tr[5]/td[1]/div/a/div[1]')
    new_name5_DEF = name5_DEF.text
    acs5_DEF = driver.find_element("xpath",
                                   '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[3]/div[4]/div[1]/table/tbody/tr[5]/td[4]/span/span[3]')
    new_acs5_DEF = acs5_DEF.text
    kill5_DEF = driver.find_element("xpath",
                                    '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[3]/div[4]/div[1]/table/tbody/tr[5]/td[5]/span/span[3]')
    new_kill5_DEF = kill5_DEF.text
    death5_DEF = driver.find_element("xpath",
                                     '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[3]/div[4]/div[1]/table/tbody/tr[5]/td[6]/span/span[2]/span[3]')
    new_death5_DEF = death5_DEF.text
    assists5_DEF = driver.find_element("xpath",
                                       '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[3]/div[4]/div[1]/table/tbody/tr[5]/td[7]/span/span[3]')
    new_assists5_DEF = assists5_DEF.text
    adr5_DEF = driver.find_element("xpath",
                                   '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[3]/div[4]/div[1]/table/tbody/tr[5]/td[10]/span/span[3]')
    new_adr5_DEF = adr5_DEF.text
    kast_DEF5 = driver.find_element("xpath",
                                    '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[3]/div[4]/div[1]/table/tbody/tr[5]/td[9]/span/span[3]')
    new_kast_DEF5 = kast_DEF5.text
    fk5_DEF = driver.find_element("xpath",
                                  '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[3]/div[4]/div[1]/table/tbody/tr[5]/td[12]/span/span[3]')
    new_fk5_DEF = fk5_DEF.text
    fb5_DEF = driver.find_element("xpath",
                                  '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[3]/div[4]/div[1]/table/tbody/tr[5]/td[13]/span/span[3]')
    new_fb5_DEF = fb5_DEF.text
    name6_DEF = driver.find_element("xpath",
                                    '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[3]/div[4]/div[2]/table/tbody/tr[1]/td[1]/div/a/div[1]')
    new_name6_DEF = name6_DEF.text
    acs6_DEF = driver.find_element("xpath",
                                   '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[3]/div[4]/div[2]/table/tbody/tr[1]/td[4]/span/span[3]')
    new_acs6_DEF = acs6_DEF.text
    kill6_DEF = driver.find_element("xpath",
                                    '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[3]/div[4]/div[2]/table/tbody/tr[1]/td[5]/span/span[3]')
    new_kill6_DEF = kill6_DEF.text
    death6_DEF = driver.find_element("xpath",
                                     '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[3]/div[4]/div[2]/table/tbody/tr[1]/td[6]/span/span[2]/span[3]')
    new_death6_DEF = death6_DEF.text
    assists6_DEF = driver.find_element("xpath",
                                       '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[3]/div[4]/div[2]/table/tbody/tr[1]/td[7]/span/span[3]')
    new_assists6_DEF = assists6_DEF.text
    adr6_DEF = driver.find_element("xpath",
                                   '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[3]/div[4]/div[2]/table/tbody/tr[1]/td[10]/span/span[3]')
    new_adr6_DEF = adr6_DEF.text
    kast_DEF6 = driver.find_element("xpath",
                                    '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[3]/div[4]/div[2]/table/tbody/tr[1]/td[9]/span/span[3]')
    new_kast_DEF6 = kast_DEF6.text
    fk6_DEF = driver.find_element("xpath",
                                  '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[3]/div[4]/div[2]/table/tbody/tr[1]/td[12]/span/span[3]')
    new_fk6_DEF = fk6_DEF.text
    fb6_DEF = driver.find_element("xpath",
                                  '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[3]/div[4]/div[2]/table/tbody/tr[1]/td[13]/span/span[3]')
    new_fb6_DEF = fb6_DEF.text
    name7_DEF = driver.find_element("xpath",
                                    '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[3]/div[4]/div[2]/table/tbody/tr[2]/td[1]/div/a/div[1]')
    new_name7_DEF = name7_DEF.text
    acs7_DEF = driver.find_element("xpath",
                                   '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[3]/div[4]/div[2]/table/tbody/tr[2]/td[4]/span/span[3]')
    new_acs7_DEF = acs7_DEF.text
    kill7_DEF = driver.find_element("xpath",
                                    '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[3]/div[4]/div[2]/table/tbody/tr[2]/td[5]/span/span[3]')
    new_kill7_DEF = kill7_DEF.text
    death7_DEF = driver.find_element("xpath",
                                     '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[3]/div[4]/div[2]/table/tbody/tr[2]/td[6]/span/span[2]/span[3]')
    new_death7_DEF = death7_DEF.text
    assists7_DEF = driver.find_element("xpath",
                                       '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[3]/div[4]/div[2]/table/tbody/tr[2]/td[7]/span/span[3]')
    new_assists7_DEF = assists7_DEF.text
    adr7_DEF = driver.find_element("xpath",
                                   '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[3]/div[4]/div[2]/table/tbody/tr[2]/td[10]/span/span[3]')
    new_adr7_DEF = adr7_DEF.text
    kast_DEF7 = driver.find_element("xpath",
                                    '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[3]/div[4]/div[2]/table/tbody/tr[2]/td[9]/span/span[3]')
    new_kast_DEF7 = kast_DEF7.text
    fk7_DEF = driver.find_element("xpath",
                                  '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[3]/div[4]/div[2]/table/tbody/tr[2]/td[12]/span/span[3]')
    new_fk7_DEF = fk7_DEF.text
    fb7_DEF = driver.find_element("xpath",
                                  '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[3]/div[4]/div[2]/table/tbody/tr[2]/td[13]/span/span[3]')
    new_fb7_DEF = fb7_DEF.text
    name8_DEF = driver.find_element("xpath",
                                    '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[3]/div[4]/div[2]/table/tbody/tr[3]/td[1]/div/a/div[1]')
    new_name8_DEF = name8_DEF.text
    acs8_DEF = driver.find_element("xpath",
                                   '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[3]/div[4]/div[2]/table/tbody/tr[3]/td[4]/span/span[3]')
    new_acs8_DEF = acs8_DEF.text
    kill8_DEF = driver.find_element("xpath",
                                    '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[3]/div[4]/div[2]/table/tbody/tr[3]/td[5]/span/span[3]')
    new_kill8_DEF = kill8_DEF.text
    death8_DEF = driver.find_element("xpath",
                                     '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[3]/div[4]/div[2]/table/tbody/tr[3]/td[6]/span/span[2]/span[3]')
    new_death8_DEF = death8_DEF.text
    assists8_DEF = driver.find_element("xpath",
                                       '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[3]/div[4]/div[2]/table/tbody/tr[3]/td[7]/span/span[3]')
    new_assists8_DEF = assists8_DEF.text
    adr8_DEF = driver.find_element("xpath",
                                   '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[3]/div[4]/div[2]/table/tbody/tr[3]/td[10]/span/span[3]')
    new_adr8_DEF = adr8_DEF.text
    kast_DEF8 = driver.find_element("xpath",
                                    '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[3]/div[4]/div[2]/table/tbody/tr[3]/td[9]/span/span[3]')
    new_kast_DEF8 = kast_DEF8.text
    fk8_DEF = driver.find_element("xpath",
                                  '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[3]/div[4]/div[2]/table/tbody/tr[3]/td[12]/span/span[3]')
    new_fk8_DEF = fk8_DEF.text
    fb8_DEF = driver.find_element("xpath",
                                  '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[3]/div[4]/div[2]/table/tbody/tr[3]/td[13]/span/span[3]')
    new_fb8_DEF = fb8_DEF.text
    name9_DEF = driver.find_element("xpath",
                                    '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[3]/div[4]/div[2]/table/tbody/tr[4]/td[1]/div/a/div[1]')
    new_name9_DEF = name9_DEF.text
    acs9_DEF = driver.find_element("xpath",
                                   '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[3]/div[4]/div[2]/table/tbody/tr[4]/td[4]/span/span[3]')
    new_acs9_DEF = acs9_DEF.text
    kill9_DEF = driver.find_element("xpath",
                                    '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[3]/div[4]/div[2]/table/tbody/tr[4]/td[5]/span/span[3]')
    new_kill9_DEF = kill9_DEF.text
    death9_DEF = driver.find_element("xpath",
                                     '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[3]/div[4]/div[2]/table/tbody/tr[4]/td[6]/span/span[2]/span[3]')
    new_death9_DEF = death9_DEF.text
    assists9_DEF = driver.find_element("xpath",
                                       '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[3]/div[4]/div[2]/table/tbody/tr[4]/td[7]/span/span[3]')
    new_assists9_DEF = assists9_DEF.text
    adr9_DEF = driver.find_element("xpath",
                                   '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[3]/div[4]/div[2]/table/tbody/tr[4]/td[10]/span/span[3]')
    new_adr9_DEF = adr9_DEF.text
    kast_DEF9 = driver.find_element("xpath",
                                    '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[3]/div[4]/div[2]/table/tbody/tr[4]/td[9]/span/span[3]')
    new_kast_DEF9 = kast_DEF9.text
    fk9_DEF = driver.find_element("xpath",
                                  '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[3]/div[4]/div[2]/table/tbody/tr[4]/td[12]/span/span[3]')
    new_fk9_DEF = fk9_DEF.text
    fb9_DEF = driver.find_element("xpath",
                                  '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[3]/div[4]/div[2]/table/tbody/tr[4]/td[13]/span/span[3]')
    new_fb9_DEF = fb9_DEF.text
    name10_DEF = driver.find_element("xpath",
                                     '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[3]/div[4]/div[2]/table/tbody/tr[5]/td[1]/div/a/div[1]')
    new_name10_DEF = name10_DEF.text
    acs10_DEF = driver.find_element("xpath",
                                    '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[3]/div[4]/div[2]/table/tbody/tr[5]/td[4]/span/span[3]')
    new_acs10_DEF = acs10_DEF.text
    kill10_DEF = driver.find_element("xpath",
                                     '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[3]/div[4]/div[2]/table/tbody/tr[5]/td[5]/span/span[3]')
    new_kill10_DEF = kill10_DEF.text
    death10_DEF = driver.find_element("xpath",
                                      '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[3]/div[4]/div[2]/table/tbody/tr[5]/td[6]/span/span[2]/span[3]')
    new_death10_DEF = death10_DEF.text
    assists10_DEF = driver.find_element("xpath",
                                        '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[3]/div[4]/div[2]/table/tbody/tr[5]/td[7]/span/span[3]')
    new_assists10_DEF = assists10_DEF.text
    adr10_DEF = driver.find_element("xpath",
                                    '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[3]/div[4]/div[2]/table/tbody/tr[5]/td[10]/span/span[3]')
    new_adr10_DEF = adr10_DEF.text
    kast_DEF10 = driver.find_element("xpath",
                                     '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[3]/div[4]/div[2]/table/tbody/tr[5]/td[9]/span/span[3]')
    new_kast_DEF10 = kast_DEF10.text
    fk10_DEF = driver.find_element("xpath",
                                   '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[3]/div[4]/div[2]/table/tbody/tr[5]/td[12]/span/span[3]')
    new_fk10_DEF = fk10_DEF.text
    fb10_DEF = driver.find_element("xpath",
                                   '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[3]/div[4]/div[2]/table/tbody/tr[5]/td[13]/span/span[3]')
    new_fb10_DEF = fb10_DEF.text
    #######################

    isim_DEF = [new_name_DEF, new_name2_DEF, new_name3_DEF, new_name4_DEF, new_name5_DEF, new_name6_DEF, new_name7_DEF,
                new_name8_DEF, new_name9_DEF, new_name10_DEF]
    acs_DEF = [new_acs_DEF, new_acs2_DEF, new_acs3_DEF, new_acs4_DEF, new_acs5_DEF, new_acs6_DEF, new_acs7_DEF,
               new_acs8_DEF, new_acs9_DEF, new_acs10_DEF]
    kill_DEF = [new_kill_DEF, new_kill2_DEF, new_kill3_DEF, new_kill4_DEF, new_kill5_DEF, new_kill6_DEF, new_kill7_DEF,
                new_kill8_DEF, new_kill9_DEF,
                new_kill10_DEF]
    death_DEF = [new_death_DEF, new_death2_DEF, new_death3_DEF, new_death4_DEF, new_death5_DEF, new_death6_DEF,
                 new_death7_DEF, new_death8_DEF, new_death9_DEF,
                 new_death10_DEF]
    assits_DEF = [new_assists_DEF, new_assists2_DEF, new_assists3_DEF, new_assists4_DEF, new_assists5_DEF,
                  new_assists6_DEF, new_assists7_DEF,
                  new_assists8_DEF, new_assists9_DEF, new_assists10_DEF]
    adr_DEF = [new_adr_DEF, new_adr2_DEF, new_adr3_DEF, new_adr4_DEF, new_adr5_DEF, new_adr6_DEF, new_adr7_DEF,
               new_adr8_DEF, new_adr9_DEF, new_adr10_DEF]
    kast_DEF = [new_kast_DEF, new_kast_DEF2, new_kast_DEF3, new_kast_DEF4, new_kast_DEF5, new_kast_DEF6, new_kast_DEF7,
                new_kast_DEF8, new_kast_DEF9, new_kast_DEF10]
    fk_DEF = [new_fk_DEF, new_fk2_DEF, new_fk3_DEF, new_fk4_DEF, new_fk5_DEF, new_fk6_DEF, new_fk7_DEF, new_fk8_DEF,
              new_fk9_DEF, new_fk10_DEF]
    fb_DEF = [new_fb_DEF, new_fb2_DEF, new_fb3_DEF, new_fb4_DEF, new_fb5_DEF, new_fb6_DEF, new_fb7_DEF, new_fb8_DEF,
              new_fb9_DEF, new_fb10_DEF]
    map = [pick2, pick2, pick2, pick2, pick2, pick2, pick2, pick2, pick2, pick2]
    team = [new_team1, new_team1, new_team1, new_team1, new_team1, new_team2, new_team2, new_team2, new_team2,
            new_team2]
    karsılasma = [vs, vs, vs, vs, vs, vs, vs, vs, vs, vs]
    Agent = ["-", "-", "-", "-", "-", "-", "-", "-", "-", "-"]
    final_score = [final, final, final, final, final, final, final, final, final, final]
    rating = ["-", "-", "-", "-", "-", "-", "-", "-", "-", "-"]
    ad = ["defense", "defense", "defense", "defense", "defense", "defense", "defense", "defense", "defense", "defense"]
    winner = ["-", "-", "-", "-", "-", "-", "-", "-", "-", "-"]
    dff = pd.DataFrame({'isim': isim_DEF,
                        'acs': acs_DEF,
                        'kill': kill_DEF,
                        'death': death_DEF,
                        'assits': assits_DEF,
                        'adr': adr_DEF,
                        'kast': kast_DEF,
                        'fk': fk_DEF,
                        'fb': fb_DEF,
                        'map': map,
                        'Team': team,
                        'VS': karsılasma,
                        'Agent': Agent,
                        'Score': final_score,
                        'Rating': rating,
                        'Side': ad,
                        'Winner': winner})
    return dff
map2_defense = map2_defense()
###### MAP_3 #####
def map3_attack():
    name = driver.find_element("xpath",
                               '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[4]/div[4]/div[1]/table/tbody/tr[1]/td[1]/div/a/div[1]')
    new_name = name.text
    acs = driver.find_element("xpath",
                              '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[4]/div[4]/div[1]/table/tbody/tr[1]/td[4]/span/span[2]')
    new_acs = acs.text
    kill = driver.find_element("xpath",
                               '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[4]/div[4]/div[1]/table/tbody/tr[1]/td[5]/span/span[2]')
    new_kill = kill.text
    death = driver.find_element("xpath",
                                '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[4]/div[4]/div[1]/table/tbody/tr[1]/td[6]/span/span[2]/span[2]')
    new_death = death.text
    assists = driver.find_element("xpath",
                                  '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[4]/div[4]/div[1]/table/tbody/tr[1]/td[7]/span/span[2]')
    new_assists = assists.text
    adr = driver.find_element("xpath",
                              '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[4]/div[4]/div[1]/table/tbody/tr[1]/td[10]/span/span[2]')
    new_adr = adr.text
    kast = driver.find_element("xpath",
                               '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[4]/div[4]/div[1]/table/tbody/tr[1]/td[9]/span/span[2]')
    new_kast = kast.text
    fk = driver.find_element("xpath",
                             '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[4]/div[4]/div[1]/table/tbody/tr[1]/td[12]/span/span[2]')
    new_fk = fk.text
    fb = driver.find_element("xpath",
                             '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[4]/div[4]/div[1]/table/tbody/tr[1]/td[13]/span/span[2]')
    new_fb = fb.text
    name2 = driver.find_element("xpath",
                                '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[4]/div[4]/div[1]/table/tbody/tr[2]/td[1]/div/a/div[1]')
    new_name2 = name2.text
    acs2 = driver.find_element("xpath",
                               '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[4]/div[4]/div[1]/table/tbody/tr[2]/td[4]/span/span[2]')
    new_acs2 = acs2.text
    kill2 = driver.find_element("xpath",
                                '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[4]/div[4]/div[1]/table/tbody/tr[2]/td[5]/span/span[2]')
    new_kill2 = kill2.text
    death2 = driver.find_element("xpath",
                                 '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[4]/div[4]/div[1]/table/tbody/tr[2]/td[6]/span/span[2]/span[2]')
    new_death2 = death2.text
    assists2 = driver.find_element("xpath",
                                   '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[4]/div[4]/div[1]/table/tbody/tr[2]/td[7]/span/span[2]')
    new_assists2 = assists2.text
    adr2 = driver.find_element("xpath",
                               '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[4]/div[4]/div[1]/table/tbody/tr[2]/td[10]/span/span[2]')
    new_adr2 = adr2.text
    kast2 = driver.find_element("xpath",
                                '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[4]/div[4]/div[1]/table/tbody/tr[2]/td[9]/span/span[2]')
    new_kast2 = kast2.text
    fk2 = driver.find_element("xpath",
                              '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[4]/div[4]/div[1]/table/tbody/tr[2]/td[12]/span/span[2]')
    new_fk2 = fk2.text
    fb2 = driver.find_element("xpath",
                              '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[4]/div[4]/div[1]/table/tbody/tr[2]/td[13]/span/span[2]')
    new_fb2 = fb2.text
    name3 = driver.find_element("xpath",
                                '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[4]/div[4]/div[1]/table/tbody/tr[3]/td[1]/div/a/div[1]')
    new_name3 = name3.text  # +
    acs3 = driver.find_element("xpath",
                               '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[4]/div[4]/div[1]/table/tbody/tr[3]/td[4]/span/span[2]')
    new_acs3 = acs3.text
    kill3 = driver.find_element("xpath",
                                '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[4]/div[4]/div[1]/table/tbody/tr[3]/td[5]/span/span[2]')
    new_kill3 = kill3.text
    death3 = driver.find_element("xpath",
                                 '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[4]/div[4]/div[1]/table/tbody/tr[3]/td[6]/span/span[2]/span[2]')
    new_death3 = death3.text
    assists3 = driver.find_element("xpath",
                                   '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[4]/div[4]/div[1]/table/tbody/tr[3]/td[7]/span/span[2]')
    new_assists3 = assists3.text
    adr3 = driver.find_element("xpath",
                               '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[4]/div[4]/div[1]/table/tbody/tr[3]/td[10]/span/span[2]')
    new_adr3 = adr3.text
    kast3 = driver.find_element("xpath",
                                '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[4]/div[4]/div[1]/table/tbody/tr[3]/td[9]/span/span[2]')
    new_kast3 = kast3.text
    fk3 = driver.find_element("xpath",
                              '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[4]/div[4]/div[1]/table/tbody/tr[3]/td[12]/span/span[2]')
    new_fk3 = fk3.text
    fb3 = driver.find_element("xpath",
                              '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[4]/div[4]/div[1]/table/tbody/tr[3]/td[13]/span/span[2]')
    new_fb3 = fb3.text
    name4 = driver.find_element("xpath",
                                '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[4]/div[4]/div[1]/table/tbody/tr[4]/td[1]/div/a/div[1]')
    new_name4 = name4.text  # +
    acs4 = driver.find_element("xpath",
                               '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[4]/div[4]/div[1]/table/tbody/tr[4]/td[4]/span/span[2]')
    new_acs4 = acs4.text
    kill4 = driver.find_element("xpath",
                                '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[4]/div[4]/div[1]/table/tbody/tr[4]/td[5]/span/span[2]')
    new_kill4 = kill4.text
    death4 = driver.find_element("xpath",
                                 '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[4]/div[4]/div[1]/table/tbody/tr[4]/td[6]/span/span[2]/span[2]')
    new_death4 = death4.text
    assists4 = driver.find_element("xpath",
                                   '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[4]/div[4]/div[1]/table/tbody/tr[4]/td[7]/span/span[2]')
    new_assists4 = assists4.text
    adr4 = driver.find_element("xpath",
                               '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[4]/div[4]/div[1]/table/tbody/tr[4]/td[10]/span/span[2]')
    new_adr4 = adr4.text
    kast4 = driver.find_element("xpath",
                                '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[4]/div[4]/div[1]/table/tbody/tr[4]/td[9]/span/span[2]')
    new_kast4 = kast4.text
    fk4 = driver.find_element("xpath",
                              '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[4]/div[4]/div[1]/table/tbody/tr[4]/td[12]/span/span[2]')
    new_fk4 = fk4.text
    fb4 = driver.find_element("xpath",
                              '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[4]/div[4]/div[1]/table/tbody/tr[4]/td[13]/span/span[2]')
    new_fb4 = fb4.text
    name5 = driver.find_element("xpath",
                                '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[4]/div[4]/div[1]/table/tbody/tr[5]/td[1]/div/a/div[1]')
    new_name5 = name5.text  # +
    acs5 = driver.find_element("xpath",
                               '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[4]/div[4]/div[1]/table/tbody/tr[5]/td[4]/span/span[2]')
    new_acs5 = acs5.text
    kill5 = driver.find_element("xpath",
                                '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[4]/div[4]/div[1]/table/tbody/tr[5]/td[5]/span/span[2]')
    new_kill5 = kill5.text
    death5 = driver.find_element("xpath",
                                 '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[4]/div[4]/div[1]/table/tbody/tr[5]/td[6]/span/span[2]/span[2]')
    new_death5 = death5.text
    assists5 = driver.find_element("xpath",
                                   '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[4]/div[4]/div[1]/table/tbody/tr[5]/td[7]/span/span[2]')
    new_assists5 = assists5.text
    adr5 = driver.find_element("xpath",
                               '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[4]/div[4]/div[1]/table/tbody/tr[5]/td[10]/span/span[2]')
    new_adr5 = adr5.text
    kast5 = driver.find_element("xpath",
                                '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[4]/div[4]/div[1]/table/tbody/tr[5]/td[9]/span/span[2]')
    new_kast5 = kast5.text
    fk5 = driver.find_element("xpath",
                              '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[4]/div[4]/div[1]/table/tbody/tr[5]/td[12]/span/span[2]')
    new_fk5 = fk5.text
    fb5 = driver.find_element("xpath",
                              '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[4]/div[4]/div[1]/table/tbody/tr[5]/td[13]/span/span[2]')
    new_fb5 = fb5.text
    ###############################################################################################################
    name6 = driver.find_element("xpath",
                                '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[4]/div[4]/div[2]/table/tbody/tr[1]/td[1]/div/a/div[1]')
    new_name6 = name6.text
    acs6 = driver.find_element("xpath",
                               '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[4]/div[4]/div[2]/table/tbody/tr[1]/td[4]/span/span[2]')
    new_acs6 = acs6.text
    kill6 = driver.find_element("xpath",
                                '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[4]/div[4]/div[2]/table/tbody/tr[1]/td[5]/span/span[2]')
    new_kill6 = kill6.text
    death6 = driver.find_element("xpath",
                                 '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[4]/div[4]/div[2]/table/tbody/tr[1]/td[6]/span/span[2]/span[2]')
    new_death6 = death6.text
    assists6 = driver.find_element("xpath",
                                   '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[4]/div[4]/div[2]/table/tbody/tr[1]/td[7]/span/span[2]')
    new_assists6 = assists6.text
    adr6 = driver.find_element("xpath",
                               '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[4]/div[4]/div[2]/table/tbody/tr[1]/td[10]/span/span[2]')
    new_adr6 = adr6.text
    kast6 = driver.find_element("xpath",
                                '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[4]/div[4]/div[2]/table/tbody/tr[1]/td[9]/span/span[2]')
    new_kast6 = kast6.text
    fk6 = driver.find_element("xpath",
                              '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[4]/div[4]/div[2]/table/tbody/tr[1]/td[12]/span/span[2]')
    new_fk6 = fk6.text
    fb6 = driver.find_element("xpath",
                              '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[4]/div[4]/div[2]/table/tbody/tr[1]/td[13]/span/span[2]')
    new_fb6 = fb6.text
    name7 = driver.find_element("xpath",
                                '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[4]/div[4]/div[2]/table/tbody/tr[2]/td[1]/div/a/div[1]')
    new_name7 = name7.text
    acs7 = driver.find_element("xpath",
                               '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[4]/div[4]/div[2]/table/tbody/tr[2]/td[4]/span/span[2]')
    new_acs7 = acs7.text
    kill7 = driver.find_element("xpath",
                                '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[4]/div[4]/div[2]/table/tbody/tr[2]/td[5]/span/span[2]')
    new_kill7 = kill7.text
    death7 = driver.find_element("xpath",
                                 '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[4]/div[4]/div[2]/table/tbody/tr[2]/td[6]/span/span[2]/span[2]')
    new_death7 = death7.text
    assists7 = driver.find_element("xpath",
                                   '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[4]/div[4]/div[2]/table/tbody/tr[2]/td[7]/span/span[2]')
    new_assists7 = assists7.text
    adr7 = driver.find_element("xpath",
                               '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[4]/div[4]/div[2]/table/tbody/tr[2]/td[10]/span/span[2]')
    new_adr7 = adr7.text
    kast7 = driver.find_element("xpath",
                                '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[4]/div[4]/div[2]/table/tbody/tr[2]/td[9]/span/span[2]')
    new_kast7 = kast7.text
    fk7 = driver.find_element("xpath",
                              '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[4]/div[4]/div[2]/table/tbody/tr[2]/td[12]/span/span[2]')
    new_fk7 = fk7.text
    fb7 = driver.find_element("xpath",
                              '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[4]/div[4]/div[2]/table/tbody/tr[2]/td[13]/span/span[2]')
    new_fb7 = fb7.text
    name8 = driver.find_element("xpath",
                                '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[4]/div[4]/div[2]/table/tbody/tr[3]/td[1]/div/a/div[1]')
    new_name8 = name8.text
    acs8 = driver.find_element("xpath",
                               '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[4]/div[4]/div[2]/table/tbody/tr[3]/td[4]/span/span[2]')
    new_acs8 = acs8.text
    kill8 = driver.find_element("xpath",
                                '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[4]/div[4]/div[2]/table/tbody/tr[3]/td[5]/span/span[2]')
    new_kill8 = kill8.text
    death8 = driver.find_element("xpath",
                                 '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[4]/div[4]/div[2]/table/tbody/tr[3]/td[6]/span/span[2]/span[2]')
    new_death8 = death8.text
    assists8 = driver.find_element("xpath",
                                   '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[4]/div[4]/div[2]/table/tbody/tr[3]/td[7]/span/span[2]')
    new_assists8 = assists8.text
    adr8 = driver.find_element("xpath",
                               '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[4]/div[4]/div[2]/table/tbody/tr[3]/td[10]/span/span[2]')
    new_adr8 = adr8.text
    kast8 = driver.find_element("xpath",
                                '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[4]/div[4]/div[2]/table/tbody/tr[4]/td[9]/span/span[2]')
    new_kast8 = kast8.text
    fk8 = driver.find_element("xpath",
                              '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[4]/div[4]/div[2]/table/tbody/tr[3]/td[12]/span/span[2]')
    new_fk8 = fk8.text
    fb8 = driver.find_element("xpath",
                              '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[4]/div[4]/div[2]/table/tbody/tr[3]/td[13]/span/span[2]')
    new_fb8 = fb8.text
    name9 = driver.find_element("xpath",
                                '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[4]/div[4]/div[2]/table/tbody/tr[4]/td[1]/div/a/div[1]')
    new_name9 = name9.text
    acs9 = driver.find_element("xpath",
                               '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[4]/div[4]/div[2]/table/tbody/tr[4]/td[4]/span/span[2]')
    new_acs9 = acs9.text
    kill9 = driver.find_element("xpath",
                                '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[4]/div[4]/div[2]/table/tbody/tr[4]/td[5]/span/span[2]')
    new_kill9 = kill9.text
    death9 = driver.find_element("xpath",
                                 '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[4]/div[4]/div[2]/table/tbody/tr[4]/td[6]/span/span[2]/span[2]')
    new_death9 = death9.text
    assists9 = driver.find_element("xpath",
                                   '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[4]/div[4]/div[2]/table/tbody/tr[4]/td[7]/span/span[2]')
    new_assists9 = assists9.text
    adr9 = driver.find_element("xpath",
                               '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[4]/div[4]/div[2]/table/tbody/tr[4]/td[10]/span/span[2]')
    new_adr9 = adr9.text
    kast9 = driver.find_element("xpath",
                                '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[4]/div[4]/div[2]/table/tbody/tr[4]/td[9]/span/span[2]')
    new_kast9 = kast9.text
    fk9 = driver.find_element("xpath",
                              '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[4]/div[4]/div[2]/table/tbody/tr[4]/td[12]/span/span[2]')
    new_fk9 = fk9.text
    fb9 = driver.find_element("xpath",
                              '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[4]/div[4]/div[2]/table/tbody/tr[4]/td[13]/span/span[2]')
    new_fb9 = fb9.text
    name10 = driver.find_element("xpath",
                                 '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[4]/div[4]/div[2]/table/tbody/tr[5]/td[1]/div/a/div[1]')
    new_name10 = name10.text
    acs10 = driver.find_element("xpath",
                                '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[4]/div[4]/div[2]/table/tbody/tr[5]/td[4]/span/span[2]')
    new_acs10 = acs10.text
    kill10 = driver.find_element("xpath",
                                 '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[4]/div[4]/div[2]/table/tbody/tr[5]/td[5]/span/span[2]')
    new_kill10 = kill10.text
    death10 = driver.find_element("xpath",
                                  '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[4]/div[4]/div[2]/table/tbody/tr[5]/td[6]/span/span[2]/span[2]')
    new_death10 = death10.text
    assists10 = driver.find_element("xpath",
                                    '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[4]/div[4]/div[2]/table/tbody/tr[5]/td[7]/span/span[2]')
    new_assists10 = assists10.text
    adr10 = driver.find_element("xpath",
                                '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[4]/div[4]/div[2]/table/tbody/tr[5]/td[10]/span/span[2]')
    new_adr10 = adr10.text
    kast10 = driver.find_element("xpath",
                                 '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[4]/div[4]/div[2]/table/tbody/tr[5]/td[9]/span/span[2]')
    new_kast10 = kast10.text
    fk10 = driver.find_element("xpath",
                               '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[4]/div[4]/div[2]/table/tbody/tr[5]/td[12]/span/span[2]')
    new_fk10 = fk10.text
    fb10 = driver.find_element("xpath",
                               '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[4]/div[4]/div[2]/table/tbody/tr[5]/td[13]/span/span[2]')
    new_fb10 = fb10.text

    isim = [new_name, new_name2, new_name3, new_name4, new_name5, new_name6, new_name7, new_name8, new_name9,
            new_name10]
    acs = [new_acs, new_acs2, new_acs3, new_acs4, new_acs5, new_acs6, new_acs7, new_acs8, new_acs9, new_acs10]
    kill = [new_kill, new_kill2, new_kill3, new_kill4, new_kill5, new_kill6, new_kill7, new_kill8, new_kill9,
            new_kill10]
    death = [new_death, new_death2, new_death3, new_death4, new_death5, new_death6, new_death7, new_death8, new_death9,
             new_death10]
    assits = [new_assists, new_assists2, new_assists3, new_assists4, new_assists5, new_assists6, new_assists7,
              new_assists8, new_assists9, new_assists10]
    adr = [new_adr, new_adr2, new_adr3, new_adr4, new_adr5, new_adr6, new_adr7, new_adr8, new_adr9, new_adr10]
    kast = [new_kast, new_kast2, new_kast3, new_kast4, new_kast5, new_kast6, new_kast7, new_kast8, new_kast9,
            new_kast10]
    fk = [new_fk, new_fk2, new_fk3, new_fk4, new_fk5, new_fk6, new_fk7, new_fk8, new_fk9, new_fk10]
    fb = [new_fb, new_fb2, new_fb3, new_fb4, new_fb5, new_fb6, new_fb7, new_fb8, new_fb9, new_fb10]
    map = [pick3, pick3, pick3, pick3, pick3, pick3, pick3, pick3, pick3, pick3]
    team = [new_team1, new_team1, new_team1, new_team1, new_team1, new_team2, new_team2, new_team2, new_team2,
            new_team2]
    karsılasma = [vs, vs, vs, vs, vs, vs, vs, vs, vs, vs]
    Agent = ["-", "-", "-", "-", "-", "-", "-", "-", "-", "-"]
    final_score = [final, final, final, final, final, final, final, final, final, final]
    rating = ["-", "-", "-", "-", "-", "-", "-", "-", "-", "-"]
    ad = ["attack", "attack", "attack", "attack", "attack", "attack", "attack", "attack", "attack", "attack"]
    winner = ["-", "-", "-", "-", "-", "-", "-", "-", "-", "-"]
    attack_round_win = ["-", "-", "-", "-", "-", "-", "-", "-", "-", "-"]
    defense_round_win = ["-", "-", "-", "-", "-", "-", "-", "-", "-", "-"]
    pistol_round = ["-", "-", "-", "-", "-", "-", "-", "-", "-", "-"]
    df = pd.DataFrame({'isim': isim,
                       'acs': acs,
                       'kill': kill,
                       'death': death,
                       'assits': assits,
                       'adr': adr,
                       'kast': kast,
                       'fk': fk,
                       'fb': fb,
                       'map': map,
                       'Team': team,
                       'VS': karsılasma,
                       'Agent': Agent,
                       'Score': final_score,
                       'Rating': rating,
                       'Side': ad,
                       'Winner': winner,
                       'Attack Round': attack_round_win,
                       'Defense Round': defense_round_win,
                       'Pistol_Round': pistol_round})
    return df
map3_attack = map3_attack()
def map3_defense():
    name_DEF = driver.find_element("xpath",
                                   '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[4]/div[4]/div[1]/table/tbody/tr[1]/td[1]/div/a/div[1]')
    new_name_DEF = name_DEF.text
    acs_DEF = driver.find_element("xpath",
                                  '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[4]/div[4]/div[1]/table/tbody/tr[1]/td[4]/span/span[3]')
    new_acs_DEF = acs_DEF.text
    kill_DEF = driver.find_element("xpath",
                                   '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[4]/div[4]/div[1]/table/tbody/tr[1]/td[5]/span/span[3]')
    new_kill_DEF = kill_DEF.text
    death_DEF = driver.find_element("xpath",
                                    '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[4]/div[4]/div[1]/table/tbody/tr[1]/td[6]/span/span[2]/span[3]')
    new_death_DEF = death_DEF.text
    assists_DEF = driver.find_element("xpath",
                                      '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[4]/div[4]/div[1]/table/tbody/tr[1]/td[7]/span/span[3]')
    new_assists_DEF = assists_DEF.text
    adr_DEF = driver.find_element("xpath",
                                  '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[4]/div[4]/div[1]/table/tbody/tr[1]/td[10]/span/span[3]')
    new_adr_DEF = adr_DEF.text
    kast = driver.find_element("xpath",
                               '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[4]/div[4]/div[1]/table/tbody/tr[1]/td[9]/span/span[3]')
    new_kast = kast.text
    fk_DEF = driver.find_element("xpath",
                                 '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[4]/div[4]/div[1]/table/tbody/tr[1]/td[12]/span/span[3]')
    new_fk_DEF = fk_DEF.text
    fb_DEF = driver.find_element("xpath",
                                 '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[4]/div[4]/div[1]/table/tbody/tr[1]/td[13]/span/span[3]')
    new_fb_DEF = fb_DEF.text
    name2_DEF = driver.find_element("xpath",
                                    '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[4]/div[4]/div[1]/table/tbody/tr[2]/td[1]/div/a/div[1]')
    new_name2_DEF = name2_DEF.text
    acs2_DEF = driver.find_element("xpath",
                                   '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[4]/div[4]/div[1]/table/tbody/tr[2]/td[4]/span/span[3]')
    new_acs2_DEF = acs2_DEF.text
    kill2_DEF = driver.find_element("xpath",
                                    '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[4]/div[4]/div[1]/table/tbody/tr[2]/td[5]/span/span[3]')
    new_kill2_DEF = kill2_DEF.text
    death2_DEF = driver.find_element("xpath",
                                     '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[4]/div[4]/div[1]/table/tbody/tr[2]/td[6]/span/span[2]/span[3]')
    new_death2_DEF = death2_DEF.text
    assists2_DEF = driver.find_element("xpath",
                                       '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[4]/div[4]/div[1]/table/tbody/tr[2]/td[7]/span/span[3]')
    new_assists2_DEF = assists2_DEF.text
    adr2_DEF = driver.find_element("xpath",
                                   '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[4]/div[4]/div[1]/table/tbody/tr[2]/td[10]/span/span[3]')
    new_adr2_DEF = adr2_DEF.text
    kast2 = driver.find_element("xpath",
                                '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[4]/div[4]/div[1]/table/tbody/tr[2]/td[9]/span/span[3]')
    new_kast2 = kast2.text
    fk2_DEF = driver.find_element("xpath",
                                  '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[4]/div[4]/div[1]/table/tbody/tr[2]/td[12]/span/span[3]')
    new_fk2_DEF = fk2_DEF.text
    fb2_DEF = driver.find_element("xpath",
                                  '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[4]/div[4]/div[1]/table/tbody/tr[2]/td[13]/span/span[3]')
    new_fb2_DEF = fb2_DEF.text
    name3_DEF = driver.find_element("xpath",
                                    '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[4]/div[4]/div[1]/table/tbody/tr[3]/td[1]/div/a/div[1]')
    new_name3_DEF = name3_DEF.text
    acs3_DEF = driver.find_element("xpath",
                                   '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[4]/div[4]/div[1]/table/tbody/tr[3]/td[4]/span/span[3]')
    new_acs3_DEF = acs3_DEF.text
    kill3_DEF = driver.find_element("xpath",
                                    '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[4]/div[4]/div[1]/table/tbody/tr[3]/td[5]/span/span[3]')
    new_kill3_DEF = kill3_DEF.text
    death3_DEF = driver.find_element("xpath",
                                     '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[4]/div[4]/div[1]/table/tbody/tr[3]/td[6]/span/span[2]/span[3]')
    new_death3_DEF = death3_DEF.text
    assists3_DEF = driver.find_element("xpath",
                                       '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[4]/div[4]/div[1]/table/tbody/tr[3]/td[7]/span/span[3]')
    new_assists3_DEF = assists3_DEF.text
    adr3_DEF = driver.find_element("xpath",
                                   '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[4]/div[4]/div[1]/table/tbody/tr[3]/td[10]/span/span[3]')
    new_adr3_DEF = adr3_DEF.text
    kast3 = driver.find_element("xpath",
                                '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[4]/div[4]/div[1]/table/tbody/tr[3]/td[9]/span/span[3]')
    new_kast3 = kast3.text
    fk3_DEF = driver.find_element("xpath",
                                  '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[4]/div[4]/div[1]/table/tbody/tr[3]/td[12]/span/span[3]')
    new_fk3_DEF = fk3_DEF.text
    fb3_DEF = driver.find_element("xpath",
                                  '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[4]/div[4]/div[1]/table/tbody/tr[3]/td[13]/span/span[3]')
    new_fb3_DEF = fb3_DEF.text
    name4_DEF = driver.find_element("xpath",
                                    '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[4]/div[4]/div[1]/table/tbody/tr[4]/td[1]/div/a/div[1]')
    new_name4_DEF = name4_DEF.text
    acs4_DEF = driver.find_element("xpath",
                                   '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[4]/div[4]/div[1]/table/tbody/tr[4]/td[4]/span/span[3]')
    new_acs4_DEF = acs4_DEF.text
    kill4_DEF = driver.find_element("xpath",
                                    '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[4]/div[4]/div[1]/table/tbody/tr[4]/td[5]/span/span[3]')
    new_kill4_DEF = kill4_DEF.text
    death4_DEF = driver.find_element("xpath",
                                     '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[4]/div[4]/div[1]/table/tbody/tr[4]/td[6]/span/span[2]/span[3]')
    new_death4_DEF = death4_DEF.text
    assists4_DEF = driver.find_element("xpath",
                                       '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[4]/div[4]/div[1]/table/tbody/tr[4]/td[7]/span/span[3]')
    new_assists4_DEF = assists4_DEF.text
    adr4_DEF = driver.find_element("xpath",
                                   '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[4]/div[4]/div[1]/table/tbody/tr[4]/td[10]/span/span[3]')
    new_adr4_DEF = adr4_DEF.text
    kast4 = driver.find_element("xpath",
                                '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[4]/div[4]/div[1]/table/tbody/tr[4]/td[9]/span/span[3]')
    new_kast4 = kast4.text
    fk4_DEF = driver.find_element("xpath",
                                  '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[4]/div[4]/div[1]/table/tbody/tr[4]/td[12]/span/span[3]')
    new_fk4_DEF = fk4_DEF.text
    fb4_DEF = driver.find_element("xpath",
                                  '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[4]/div[4]/div[1]/table/tbody/tr[4]/td[13]/span/span[3]')
    new_fb4_DEF = fb4_DEF.text
    name5_DEF = driver.find_element("xpath",
                                    '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[4]/div[4]/div[1]/table/tbody/tr[5]/td[1]/div/a/div[1]')
    new_name5_DEF = name5_DEF.text
    acs5_DEF = driver.find_element("xpath",
                                   '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[4]/div[4]/div[1]/table/tbody/tr[5]/td[4]/span/span[3]')
    new_acs5_DEF = acs5_DEF.text
    kill5_DEF = driver.find_element("xpath",
                                    '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[4]/div[4]/div[1]/table/tbody/tr[5]/td[5]/span/span[3]')
    new_kill5_DEF = kill5_DEF.text
    death5_DEF = driver.find_element("xpath",
                                     '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[4]/div[4]/div[1]/table/tbody/tr[5]/td[6]/span/span[2]/span[3]')
    new_death5_DEF = death5_DEF.text
    assists5_DEF = driver.find_element("xpath",
                                       '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[4]/div[4]/div[1]/table/tbody/tr[5]/td[7]/span/span[3]')
    new_assists5_DEF = assists5_DEF.text
    adr5_DEF = driver.find_element("xpath",
                                   '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[4]/div[4]/div[1]/table/tbody/tr[5]/td[10]/span/span[3]')
    new_adr5_DEF = adr5_DEF.text
    kast5 = driver.find_element("xpath",
                                '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[4]/div[4]/div[1]/table/tbody/tr[5]/td[9]/span/span[3]')
    new_kast5 = kast5.text
    fk5_DEF = driver.find_element("xpath",
                                  '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[4]/div[4]/div[1]/table/tbody/tr[5]/td[12]/span/span[3]')
    new_fk5_DEF = fk5_DEF.text
    fb5_DEF = driver.find_element("xpath",
                                  '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[4]/div[4]/div[1]/table/tbody/tr[5]/td[13]/span/span[3]')
    new_fb5_DEF = fb5_DEF.text
    name6_DEF = driver.find_element("xpath",
                                    '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[4]/div[4]/div[2]/table/tbody/tr[1]/td[1]/div/a/div[1]')
    new_name6_DEF = name6_DEF.text
    acs6_DEF = driver.find_element("xpath",
                                   '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[4]/div[4]/div[2]/table/tbody/tr[1]/td[4]/span/span[3]')
    new_acs6_DEF = acs6_DEF.text
    kill6_DEF = driver.find_element("xpath",
                                    '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[4]/div[4]/div[2]/table/tbody/tr[1]/td[5]/span/span[3]')
    new_kill6_DEF = kill6_DEF.text
    death6_DEF = driver.find_element("xpath",
                                     '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[4]/div[4]/div[2]/table/tbody/tr[1]/td[6]/span/span[2]/span[3]')
    new_death6_DEF = death6_DEF.text
    assists6_DEF = driver.find_element("xpath",
                                       '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[4]/div[4]/div[2]/table/tbody/tr[1]/td[7]/span/span[3]')
    new_assists6_DEF = assists6_DEF.text
    adr6_DEF = driver.find_element("xpath",
                                   '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[4]/div[4]/div[2]/table/tbody/tr[1]/td[10]/span/span[3]')
    new_adr6_DEF = adr6_DEF.text
    kast6 = driver.find_element("xpath",
                                '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[4]/div[4]/div[2]/table/tbody/tr[1]/td[9]/span/span[3]')
    new_kast6 = kast6.text
    fk6_DEF = driver.find_element("xpath",
                                  '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[4]/div[4]/div[2]/table/tbody/tr[1]/td[12]/span/span[3]')
    new_fk6_DEF = fk6_DEF.text
    fb6_DEF = driver.find_element("xpath",
                                  '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[4]/div[4]/div[2]/table/tbody/tr[1]/td[13]/span/span[3]')
    new_fb6_DEF = fb6_DEF.text
    name7_DEF = driver.find_element("xpath",
                                    '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[4]/div[4]/div[2]/table/tbody/tr[2]/td[1]/div/a/div[1]')
    new_name7_DEF = name7_DEF.text
    acs7_DEF = driver.find_element("xpath",
                                   '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[4]/div[4]/div[2]/table/tbody/tr[2]/td[4]/span/span[3]')
    new_acs7_DEF = acs7_DEF.text
    kill7_DEF = driver.find_element("xpath",
                                    '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[4]/div[4]/div[2]/table/tbody/tr[2]/td[5]/span/span[3]')
    new_kill7_DEF = kill7_DEF.text
    death7_DEF = driver.find_element("xpath",
                                     '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[4]/div[4]/div[2]/table/tbody/tr[2]/td[6]/span/span[2]/span[3]')
    new_death7_DEF = death7_DEF.text
    assists7_DEF = driver.find_element("xpath",
                                       '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[4]/div[4]/div[2]/table/tbody/tr[2]/td[7]/span/span[3]')
    new_assists7_DEF = assists7_DEF.text
    adr7_DEF = driver.find_element("xpath",
                                   '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[4]/div[4]/div[2]/table/tbody/tr[2]/td[10]/span/span[3]')
    new_adr7_DEF = adr7_DEF.text
    kast7 = driver.find_element("xpath",
                                '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[4]/div[4]/div[2]/table/tbody/tr[2]/td[9]/span/span[3]')
    new_kast7 = kast7.text
    fk7_DEF = driver.find_element("xpath",
                                  '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[4]/div[4]/div[2]/table/tbody/tr[2]/td[12]/span/span[3]')
    new_fk7_DEF = fk7_DEF.text
    fb7_DEF = driver.find_element("xpath",
                                  '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[4]/div[4]/div[2]/table/tbody/tr[2]/td[13]/span/span[3]')
    new_fb7_DEF = fb7_DEF.text
    name8_DEF = driver.find_element("xpath",
                                    '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[4]/div[4]/div[2]/table/tbody/tr[3]/td[1]/div/a/div[1]')
    new_name8_DEF = name8_DEF.text
    acs8_DEF = driver.find_element("xpath",
                                   '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[4]/div[4]/div[2]/table/tbody/tr[3]/td[4]/span/span[3]')
    new_acs8_DEF = acs8_DEF.text
    kill8_DEF = driver.find_element("xpath",
                                    '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[4]/div[4]/div[2]/table/tbody/tr[3]/td[5]/span/span[3]')
    new_kill8_DEF = kill8_DEF.text
    death8_DEF = driver.find_element("xpath",
                                     '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[4]/div[4]/div[2]/table/tbody/tr[3]/td[6]/span/span[2]/span[3]')
    new_death8_DEF = death8_DEF.text
    assists8_DEF = driver.find_element("xpath",
                                       '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[4]/div[4]/div[2]/table/tbody/tr[3]/td[7]/span/span[3]')
    new_assists8_DEF = assists8_DEF.text
    adr8_DEF = driver.find_element("xpath",
                                   '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[4]/div[4]/div[2]/table/tbody/tr[3]/td[10]/span/span[3]')
    new_adr8_DEF = adr8_DEF.text
    kast8 = driver.find_element("xpath",
                                '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[4]/div[4]/div[2]/table/tbody/tr[3]/td[9]/span/span[3]')
    new_kast8 = kast8.text
    fk8_DEF = driver.find_element("xpath",
                                  '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[4]/div[4]/div[2]/table/tbody/tr[3]/td[12]/span/span[3]')
    new_fk8_DEF = fk8_DEF.text
    fb8_DEF = driver.find_element("xpath",
                                  '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[4]/div[4]/div[2]/table/tbody/tr[3]/td[13]/span/span[3]')
    new_fb8_DEF = fb8_DEF.text
    name9_DEF = driver.find_element("xpath",
                                    '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[4]/div[4]/div[2]/table/tbody/tr[4]/td[1]/div/a/div[1]')
    new_name9_DEF = name9_DEF.text
    acs9_DEF = driver.find_element("xpath",
                                   '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[4]/div[4]/div[2]/table/tbody/tr[4]/td[4]/span/span[3]')
    new_acs9_DEF = acs9_DEF.text
    kill9_DEF = driver.find_element("xpath",
                                    '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[4]/div[4]/div[2]/table/tbody/tr[4]/td[5]/span/span[3]')
    new_kill9_DEF = kill9_DEF.text
    death9_DEF = driver.find_element("xpath",
                                     '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[4]/div[4]/div[2]/table/tbody/tr[4]/td[6]/span/span[2]/span[3]')
    new_death9_DEF = death9_DEF.text
    assists9_DEF = driver.find_element("xpath",
                                       '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[4]/div[4]/div[2]/table/tbody/tr[4]/td[7]/span/span[3]')
    new_assists9_DEF = assists9_DEF.text
    adr9_DEF = driver.find_element("xpath",
                                   '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[4]/div[4]/div[2]/table/tbody/tr[4]/td[10]/span/span[3]')
    new_adr9_DEF = adr9_DEF.text
    kast9 = driver.find_element("xpath",
                                '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[4]/div[4]/div[2]/table/tbody/tr[4]/td[9]/span/span[3]')
    new_kast9 = kast9.text
    fk9_DEF = driver.find_element("xpath",
                                  '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[4]/div[4]/div[2]/table/tbody/tr[4]/td[12]/span/span[3]')
    new_fk9_DEF = fk9_DEF.text
    fb9_DEF = driver.find_element("xpath",
                                  '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[4]/div[4]/div[2]/table/tbody/tr[4]/td[13]/span/span[3]')
    new_fb9_DEF = fb9_DEF.text
    name10_DEF = driver.find_element("xpath",
                                     '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[4]/div[4]/div[2]/table/tbody/tr[5]/td[1]/div/a/div[1]')
    new_name10_DEF = name10_DEF.text
    acs10_DEF = driver.find_element("xpath",
                                    '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[4]/div[4]/div[2]/table/tbody/tr[5]/td[4]/span/span[3]')
    new_acs10_DEF = acs10_DEF.text
    kill10_DEF = driver.find_element("xpath",
                                     '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[4]/div[4]/div[2]/table/tbody/tr[5]/td[5]/span/span[3]')
    new_kill10_DEF = kill10_DEF.text
    death10_DEF = driver.find_element("xpath",
                                      '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[4]/div[4]/div[2]/table/tbody/tr[5]/td[6]/span/span[2]/span[3]')
    new_death10_DEF = death10_DEF.text
    assists10_DEF = driver.find_element("xpath",
                                        '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[4]/div[4]/div[2]/table/tbody/tr[5]/td[7]/span/span[3]')
    new_assists10_DEF = assists10_DEF.text
    adr10_DEF = driver.find_element("xpath",
                                    '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[4]/div[4]/div[2]/table/tbody/tr[5]/td[10]/span/span[3]')
    new_adr10_DEF = adr10_DEF.text
    kast10 = driver.find_element("xpath",
                                 '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[4]/div[4]/div[2]/table/tbody/tr[5]/td[9]/span/span[3]')
    new_kast10 = kast10.text
    fk10_DEF = driver.find_element("xpath",
                                   '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[4]/div[4]/div[2]/table/tbody/tr[5]/td[12]/span/span[3]')
    new_fk10_DEF = fk10_DEF.text
    fb10_DEF = driver.find_element("xpath",
                                   '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[4]/div[4]/div[2]/table/tbody/tr[5]/td[13]/span/span[3]')
    new_fb10_DEF = fb10_DEF.text
    #######################

    isim_DEF = [new_name_DEF, new_name2_DEF, new_name3_DEF, new_name4_DEF, new_name5_DEF, new_name6_DEF, new_name7_DEF,
                new_name8_DEF, new_name9_DEF, new_name10_DEF]
    acs_DEF = [new_acs_DEF, new_acs2_DEF, new_acs3_DEF, new_acs4_DEF, new_acs5_DEF, new_acs6_DEF, new_acs7_DEF,
               new_acs8_DEF, new_acs9_DEF, new_acs10_DEF]
    kill_DEF = [new_kill_DEF, new_kill2_DEF, new_kill3_DEF, new_kill4_DEF, new_kill5_DEF, new_kill6_DEF, new_kill7_DEF,
                new_kill8_DEF, new_kill9_DEF,
                new_kill10_DEF]
    death_DEF = [new_death_DEF, new_death2_DEF, new_death3_DEF, new_death4_DEF, new_death5_DEF, new_death6_DEF,
                 new_death7_DEF, new_death8_DEF, new_death9_DEF,
                 new_death10_DEF]
    assits_DEF = [new_assists_DEF, new_assists2_DEF, new_assists3_DEF, new_assists4_DEF, new_assists5_DEF,
                  new_assists6_DEF, new_assists7_DEF,
                  new_assists8_DEF, new_assists9_DEF, new_assists10_DEF]
    adr_DEF = [new_adr_DEF, new_adr2_DEF, new_adr3_DEF, new_adr4_DEF, new_adr5_DEF, new_adr6_DEF, new_adr7_DEF,
               new_adr8_DEF, new_adr9_DEF, new_adr10_DEF]
    kast_DEF = [new_kast, new_kast2, new_kast3, new_kast4, new_kast5, new_kast6, new_kast7,
                new_kast8, new_kast9, new_kast10]
    fk_DEF = [new_fk_DEF, new_fk2_DEF, new_fk3_DEF, new_fk4_DEF, new_fk5_DEF, new_fk6_DEF, new_fk7_DEF, new_fk8_DEF,
              new_fk9_DEF, new_fk10_DEF]
    fb_DEF = [new_fb_DEF, new_fb2_DEF, new_fb3_DEF, new_fb4_DEF, new_fb5_DEF, new_fb6_DEF, new_fb7_DEF, new_fb8_DEF,
              new_fb9_DEF, new_fb10_DEF]
    map = [pick3, pick3, pick3, pick3, pick3, pick3, pick3, pick3, pick3, pick3]
    team = [new_team1, new_team1, new_team1, new_team1, new_team1, new_team2, new_team2, new_team2, new_team2,
            new_team2]
    karsılasma = [vs, vs, vs, vs, vs, vs, vs, vs, vs, vs]
    Agent = ["-", "-", "-", "-", "-", "-", "-", "-", "-", "-"]
    final_score = [final, final, final, final, final, final, final, final, final, final]
    rating = ["-", "-", "-", "-", "-", "-", "-", "-", "-", "-"]
    ad = ["defense", "defense", "defense", "defense", "defense", "defense", "defense", "defense", "defense", "defense"]
    winner = ["-", "-", "-", "-", "-", "-", "-", "-", "-", "-"]
    dff = pd.DataFrame({'isim': isim_DEF,
                        'acs': acs_DEF,
                        'kill': kill_DEF,
                        'death': death_DEF,
                        'assits': assits_DEF,
                        'adr': adr_DEF,
                        'kast': kast_DEF,
                        'fk': fk_DEF,
                        'fb': fb_DEF,
                        'map': map,
                        'Team': team,
                        'VS': karsılasma,
                        'Agent': Agent,
                        'Score': final_score,
                        'Rating': rating,
                        'Side': ad,
                        'Winner': winner})
    return dff
map3_defense = map3_defense()
###### MAP_4 #####
def map4_attack():
    name = driver.find_element("xpath",
                               '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[5]/div[4]/div[1]/table/tbody/tr[1]/td[1]/div/a/div[1]')
    new_name = name.text
    acs = driver.find_element("xpath",
                              '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[5]/div[4]/div[1]/table/tbody/tr[1]/td[4]/span/span[2]')
    new_acs = acs.text
    kill = driver.find_element("xpath",
                               '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[5]/div[4]/div[1]/table/tbody/tr[1]/td[5]/span/span[2]')
    new_kill = kill.text
    death = driver.find_element("xpath",
                                '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[5]/div[4]/div[1]/table/tbody/tr[1]/td[6]/span/span[2]/span[2]')
    new_death = death.text
    assists = driver.find_element("xpath",
                                  '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[5]/div[4]/div[1]/table/tbody/tr[1]/td[7]/span/span[2]')
    new_assists = assists.text
    adr = driver.find_element("xpath",
                              '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[5]/div[4]/div[1]/table/tbody/tr[1]/td[10]/span/span[2]')
    new_adr = adr.text
    kast = driver.find_element("xpath",
                               '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[5]/div[4]/div[1]/table/tbody/tr[1]/td[9]/span/span[2]')
    new_kast = kast.text
    fk = driver.find_element("xpath",
                             '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[5]/div[4]/div[1]/table/tbody/tr[1]/td[12]/span/span[2]')
    new_fk = fk.text
    fb = driver.find_element("xpath",
                             '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[5]/div[4]/div[1]/table/tbody/tr[1]/td[13]/span/span[2]')
    new_fb = fb.text
    name2 = driver.find_element("xpath",
                                '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[5]/div[4]/div[1]/table/tbody/tr[2]/td[1]/div/a/div[1]')
    new_name2 = name2.text
    acs2 = driver.find_element("xpath",
                               '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[5]/div[4]/div[1]/table/tbody/tr[2]/td[4]/span/span[2]')
    new_acs2 = acs2.text
    kill2 = driver.find_element("xpath",
                                '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[5]/div[4]/div[1]/table/tbody/tr[2]/td[5]/span/span[2]')
    new_kill2 = kill2.text
    death2 = driver.find_element("xpath",
                                 '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[5]/div[4]/div[1]/table/tbody/tr[2]/td[6]/span/span[2]/span[2]')
    new_death2 = death2.text
    assists2 = driver.find_element("xpath",
                                   '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[5]/div[4]/div[1]/table/tbody/tr[2]/td[7]/span/span[2]')
    new_assists2 = assists2.text
    adr2 = driver.find_element("xpath",
                               '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[5]/div[4]/div[1]/table/tbody/tr[2]/td[10]/span/span[2]')
    new_adr2 = adr2.text
    kast2 = driver.find_element("xpath",
                                '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[5]/div[4]/div[1]/table/tbody/tr[2]/td[9]/span/span[2]')
    new_kast2 = kast2.text
    fk2 = driver.find_element("xpath",
                              '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[5]/div[4]/div[1]/table/tbody/tr[2]/td[12]/span/span[2]')
    new_fk2 = fk2.text
    fb2 = driver.find_element("xpath",
                              '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[5]/div[4]/div[1]/table/tbody/tr[2]/td[13]/span/span[2]')
    new_fb2 = fb2.text
    name3 = driver.find_element("xpath",
                                '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[5]/div[4]/div[1]/table/tbody/tr[3]/td[1]/div/a/div[1]')
    new_name3 = name3.text
    acs3 = driver.find_element("xpath",
                               '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[5]/div[4]/div[1]/table/tbody/tr[3]/td[4]/span/span[2]')
    new_acs3 = acs3.text
    kill3 = driver.find_element("xpath",
                                '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[5]/div[4]/div[1]/table/tbody/tr[3]/td[5]/span/span[2]')
    new_kill3 = kill3.text
    death3 = driver.find_element("xpath",
                                 '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[5]/div[4]/div[1]/table/tbody/tr[3]/td[6]/span/span[2]/span[2]')
    new_death3 = death3.text
    assists3 = driver.find_element("xpath",
                                   '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[5]/div[4]/div[1]/table/tbody/tr[3]/td[7]/span/span[2]')
    new_assists3 = assists3.text
    adr3 = driver.find_element("xpath",
                               '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[5]/div[4]/div[1]/table/tbody/tr[3]/td[10]/span/span[2]')
    new_adr3 = adr3.text
    kast3 = driver.find_element("xpath",
                                '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[5]/div[4]/div[1]/table/tbody/tr[3]/td[9]/span/span[2]')
    new_kast3 = kast3.text
    fk3 = driver.find_element("xpath",
                              '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[5]/div[4]/div[1]/table/tbody/tr[3]/td[12]/span/span[2]')
    new_fk3 = fk3.text
    fb3 = driver.find_element("xpath",
                              '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[5]/div[4]/div[1]/table/tbody/tr[3]/td[13]/span/span[2]')
    new_fb3 = fb3.text
    name4 = driver.find_element("xpath",
                                '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[5]/div[4]/div[1]/table/tbody/tr[4]/td[1]/div/a/div[1]')
    new_name4 = name4.text
    acs4 = driver.find_element("xpath",
                               '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[5]/div[4]/div[1]/table/tbody/tr[4]/td[4]/span/span[2]')
    new_acs4 = acs4.text
    kill4 = driver.find_element("xpath",
                                '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[5]/div[4]/div[1]/table/tbody/tr[4]/td[5]/span/span[2]')
    new_kill4 = kill4.text
    death4 = driver.find_element("xpath",
                                 '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[5]/div[4]/div[1]/table/tbody/tr[4]/td[6]/span/span[2]/span[2]')
    new_death4 = death4.text
    assists4 = driver.find_element("xpath",
                                   '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[5]/div[4]/div[1]/table/tbody/tr[4]/td[7]/span/span[2]')
    new_assists4 = assists4.text
    adr4 = driver.find_element("xpath",
                               '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[5]/div[4]/div[1]/table/tbody/tr[4]/td[10]/span/span[2]')
    new_adr4 = adr4.text
    kast4 = driver.find_element("xpath",
                                '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[5]/div[4]/div[1]/table/tbody/tr[4]/td[9]/span/span[2]')
    new_kast4 = kast4.text
    fk4 = driver.find_element("xpath",
                              '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[5]/div[4]/div[1]/table/tbody/tr[4]/td[12]/span/span[2]')
    new_fk4 = fk4.text
    fb4 = driver.find_element("xpath",
                              '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[5]/div[4]/div[1]/table/tbody/tr[4]/td[13]/span/span[2]')
    new_fb4 = fb4.text
    name5 = driver.find_element("xpath",
                                '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[5]/div[4]/div[1]/table/tbody/tr[5]/td[1]/div/a/div[1]')
    new_name5 = name5.text
    acs5 = driver.find_element("xpath",
                               '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[5]/div[4]/div[1]/table/tbody/tr[5]/td[4]/span/span[2]')
    new_acs5 = acs5.text
    kill5 = driver.find_element("xpath",
                                '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[5]/div[4]/div[1]/table/tbody/tr[5]/td[5]/span/span[2]')
    new_kill5 = kill5.text
    death5 = driver.find_element("xpath",
                                 '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[5]/div[4]/div[1]/table/tbody/tr[5]/td[6]/span/span[2]/span[2]')
    new_death5 = death5.text
    assists5 = driver.find_element("xpath",
                                   '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[5]/div[4]/div[1]/table/tbody/tr[5]/td[7]/span/span[2]')
    new_assists5 = assists5.text
    adr5 = driver.find_element("xpath",
                               '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[5]/div[4]/div[1]/table/tbody/tr[5]/td[10]/span/span[2]')
    new_adr5 = adr5.text
    kast5 = driver.find_element("xpath",
                                '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[5]/div[4]/div[1]/table/tbody/tr[5]/td[9]/span/span[2]')
    new_kast5 = kast5.text
    fk5 = driver.find_element("xpath",
                              '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[5]/div[4]/div[1]/table/tbody/tr[5]/td[12]/span/span[2]')
    new_fk5 = fk5.text
    fb5 = driver.find_element("xpath",
                              '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[5]/div[4]/div[1]/table/tbody/tr[5]/td[13]/span/span[2]')
    new_fb5 = fb5.text
    name6 = driver.find_element("xpath",
                                '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[5]/div[4]/div[2]/table/tbody/tr[1]/td[1]/div/a/div[1]')
    new_name6 = name6.text
    acs6 = driver.find_element("xpath",
                               '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[5]/div[4]/div[2]/table/tbody/tr[1]/td[4]/span/span[2]')
    new_acs6 = acs6.text
    kill6 = driver.find_element("xpath",
                                '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[5]/div[4]/div[2]/table/tbody/tr[1]/td[5]/span/span[2]')
    new_kill6 = kill6.text
    death6 = driver.find_element("xpath",
                                 '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[5]/div[4]/div[2]/table/tbody/tr[1]/td[6]/span/span[2]/span[2]')
    new_death6 = death6.text
    assists6 = driver.find_element("xpath",
                                   '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[5]/div[4]/div[2]/table/tbody/tr[1]/td[7]/span/span[2]')
    new_assists6 = assists6.text
    adr6 = driver.find_element("xpath",
                               '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[5]/div[4]/div[2]/table/tbody/tr[1]/td[10]/span/span[2]')
    new_adr6 = adr6.text
    kast6 = driver.find_element("xpath",
                                '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[5]/div[4]/div[2]/table/tbody/tr[1]/td[9]/span/span[2]')
    new_kast6 = kast6.text
    fk6 = driver.find_element("xpath",
                              '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[5]/div[4]/div[2]/table/tbody/tr[1]/td[12]/span/span[2]')
    new_fk6 = fk6.text
    fb6 = driver.find_element("xpath",
                              '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[5]/div[4]/div[2]/table/tbody/tr[1]/td[13]/span/span[2]')
    new_fb6 = fb6.text
    name7 = driver.find_element("xpath",
                                '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[5]/div[4]/div[2]/table/tbody/tr[2]/td[1]/div/a/div[1]')
    new_name7 = name7.text
    acs7 = driver.find_element("xpath",
                               '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[5]/div[4]/div[2]/table/tbody/tr[2]/td[4]/span/span[2]')
    new_acs7 = acs7.text
    kill7 = driver.find_element("xpath",
                                '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[5]/div[4]/div[2]/table/tbody/tr[2]/td[5]/span/span[2]')
    new_kill7 = kill7.text
    death7 = driver.find_element("xpath",
                                 '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[5]/div[4]/div[2]/table/tbody/tr[1]/td[6]/span/span[2]/span[2]')
    new_death7 = death7.text
    assists7 = driver.find_element("xpath",
                                   '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[5]/div[4]/div[2]/table/tbody/tr[2]/td[7]/span/span[2]')
    new_assists7 = assists7.text
    adr7 = driver.find_element("xpath",
                               '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[5]/div[4]/div[2]/table/tbody/tr[2]/td[10]/span/span[2]')
    new_adr7 = adr7.text
    kast7 = driver.find_element("xpath",
                                '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[5]/div[4]/div[2]/table/tbody/tr[2]/td[9]/span/span[2]')
    new_kast7 = kast7.text
    fk7 = driver.find_element("xpath",
                              '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[5]/div[4]/div[2]/table/tbody/tr[2]/td[12]/span/span[2]')
    new_fk7 = fk7.text
    fb7 = driver.find_element("xpath",
                              '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[5]/div[4]/div[2]/table/tbody/tr[2]/td[13]/span/span[2]')
    new_fb7 = fb7.text
    name8 = driver.find_element("xpath",
                                '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[5]/div[4]/div[2]/table/tbody/tr[3]/td[1]/div/a/div[1]')
    new_name8 = name8.text
    acs8 = driver.find_element("xpath",
                               '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[5]/div[4]/div[2]/table/tbody/tr[3]/td[4]/span/span[2]')
    new_acs8 = acs8.text
    kill8 = driver.find_element("xpath",
                                '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[5]/div[4]/div[2]/table/tbody/tr[3]/td[5]/span/span[2]')
    new_kill8 = kill8.text
    death8 = driver.find_element("xpath",
                                 '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[5]/div[4]/div[2]/table/tbody/tr[3]/td[5]/span/span[2]')
    new_death8 = death8.text
    assists8 = driver.find_element("xpath",
                                   '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[5]/div[4]/div[2]/table/tbody/tr[3]/td[7]/span/span[2]')
    new_assists8 = assists8.text
    adr8 = driver.find_element("xpath",
                               '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[5]/div[4]/div[2]/table/tbody/tr[3]/td[10]/span/span[2]')
    new_adr8 = adr8.text
    kast8 = driver.find_element("xpath",
                                '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[5]/div[4]/div[2]/table/tbody/tr[3]/td[9]/span/span[2]')
    new_kast8 = kast8.text
    fk8 = driver.find_element("xpath",
                              '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[5]/div[4]/div[2]/table/tbody/tr[3]/td[12]/span/span[2]')
    new_fk8 = fk8.text
    fb8 = driver.find_element("xpath",
                              '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[5]/div[4]/div[2]/table/tbody/tr[3]/td[13]/span/span[2]')
    new_fb8 = fb8.text
    name9 = driver.find_element("xpath",
                                '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[5]/div[4]/div[2]/table/tbody/tr[4]/td[1]/div/a/div[1]')
    new_name9 = name9.text
    acs9 = driver.find_element("xpath",
                               '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[5]/div[4]/div[2]/table/tbody/tr[4]/td[4]/span/span[2]')
    new_acs9 = acs9.text
    kill9 = driver.find_element("xpath",
                                '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[5]/div[4]/div[2]/table/tbody/tr[4]/td[5]/span/span[2]')
    new_kill9 = kill9.text
    death9 = driver.find_element("xpath",
                                 '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[5]/div[4]/div[2]/table/tbody/tr[4]/td[6]/span/span[2]/span[2]')
    new_death9 = death9.text
    assists9 = driver.find_element("xpath",
                                   '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[5]/div[4]/div[2]/table/tbody/tr[4]/td[7]/span/span[2]')
    new_assists9 = assists9.text
    adr9 = driver.find_element("xpath",
                               '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[5]/div[4]/div[2]/table/tbody/tr[4]/td[10]/span/span[2]')
    new_adr9 = adr9.text
    kast9 = driver.find_element("xpath",
                                '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[5]/div[4]/div[2]/table/tbody/tr[4]/td[9]/span/span[2]')
    new_kast9 = kast9.text
    fk9 = driver.find_element("xpath",
                              '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[5]/div[4]/div[2]/table/tbody/tr[4]/td[12]/span/span[2]')
    new_fk9 = fk9.text
    fb9 = driver.find_element("xpath",
                              '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[5]/div[4]/div[2]/table/tbody/tr[4]/td[13]/span/span[2]')
    new_fb9 = fb9.text
    name10 = driver.find_element("xpath",
                                 '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[5]/div[4]/div[2]/table/tbody/tr[5]/td[1]/div/a/div[1]')
    new_name10 = name10.text
    acs10 = driver.find_element("xpath",
                                '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[5]/div[4]/div[2]/table/tbody/tr[5]/td[4]/span/span[2]')
    new_acs10 = acs10.text
    kill10 = driver.find_element("xpath",
                                 '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[5]/div[4]/div[2]/table/tbody/tr[5]/td[5]/span/span[2]')
    new_kill10 = kill10.text
    death10 = driver.find_element("xpath",
                                  '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[5]/div[4]/div[2]/table/tbody/tr[5]/td[6]/span/span[2]/span[2]')
    new_death10 = death10.text
    assists10 = driver.find_element("xpath",
                                    '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[5]/div[4]/div[2]/table/tbody/tr[5]/td[7]/span/span[2]')
    new_assists10 = assists10.text
    adr10 = driver.find_element("xpath",
                                '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[5]/div[4]/div[2]/table/tbody/tr[5]/td[10]/span/span[2]')
    new_adr10 = adr10.text
    kast10 = driver.find_element("xpath",
                                 '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[5]/div[4]/div[2]/table/tbody/tr[5]/td[9]/span/span[2]')
    new_kast10 = kast10.text
    fk10 = driver.find_element("xpath",
                               '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[5]/div[4]/div[2]/table/tbody/tr[5]/td[12]/span/span[2]')
    new_fk10 = fk10.text
    fb10 = driver.find_element("xpath",
                               '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[5]/div[4]/div[2]/table/tbody/tr[5]/td[13]/span/span[2]')
    new_fb10 = fb10.text

    isim = [new_name, new_name2, new_name3, new_name4, new_name5, new_name6, new_name7, new_name8, new_name9,
            new_name10]
    acs = [new_acs, new_acs2, new_acs3, new_acs4, new_acs5, new_acs6, new_acs7, new_acs8, new_acs9, new_acs10]
    kill = [new_kill, new_kill2, new_kill3, new_kill4, new_kill5, new_kill6, new_kill7, new_kill8, new_kill9,
            new_kill10]
    death = [new_death, new_death2, new_death3, new_death4, new_death5, new_death6, new_death7, new_death8, new_death9,
             new_death10]
    assits = [new_assists, new_assists2, new_assists3, new_assists4, new_assists5, new_assists6, new_assists7,
              new_assists8, new_assists9, new_assists10]
    adr = [new_adr, new_adr2, new_adr3, new_adr4, new_adr5, new_adr6, new_adr7, new_adr8, new_adr9, new_adr10]
    kast = [new_kast, new_kast2, new_kast3, new_kast4, new_kast5, new_kast6, new_kast7, new_kast8, new_kast9,
            new_kast10]
    fk = [new_fk, new_fk2, new_fk3, new_fk4, new_fk5, new_fk6, new_fk7, new_fk8, new_fk9, new_fk10]
    fb = [new_fb, new_fb2, new_fb3, new_fb4, new_fb5, new_fb6, new_fb7, new_fb8, new_fb9, new_fb10]
    map = [pick4, pick4, pick4, pick4, pick4, pick4, pick4, pick4, pick4, pick4]
    team = [new_team1, new_team1, new_team1, new_team1, new_team1, new_team2, new_team2, new_team2, new_team2,
            new_team2]
    karsılasma = [vs, vs, vs, vs, vs, vs, vs, vs, vs, vs]
    Agent = ["-", "-", "-", "-", "-", "-", "-", "-", "-", "-"]
    final_score = [final, final, final, final, final, final, final, final, final, final]
    rating = ["-", "-", "-", "-", "-", "-", "-", "-", "-", "-"]
    ad = ["attack", "attack", "attack", "attack", "attack", "attack", "attack", "attack", "attack", "attack"]
    winner = ["-", "-", "-", "-", "-", "-", "-", "-", "-", "-"]
    attack_round_win = ["-", "-", "-", "-", "-", "-", "-", "-", "-", "-"]
    defense_round_win = ["-", "-", "-", "-", "-", "-", "-", "-", "-", "-"]
    pistol_round = ["-", "-", "-", "-", "-", "-", "-", "-", "-", "-"]
    df = pd.DataFrame({'isim': isim,
                       'acs': acs,
                       'kill': kill,
                       'death': death,
                       'assits': assits,
                       'adr': adr,
                       'kast': kast,
                       'fk': fk,
                       'fb': fb,
                       'map': map,
                       'Team': team,
                       'VS': karsılasma,
                       'Agent': Agent,
                       'Score': final_score,
                       'Rating': rating,
                       'Side': ad,
                       'Winner': winner,
                       'Attack Round': attack_round_win,
                       'Defense Round': defense_round_win,
                       'Pistol_Round': pistol_round})
    return df
map4_attack = map4_attack()
def map4_defense():
    name_DEF = driver.find_element("xpath",
                                   '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[5]/div[4]/div[1]/table/tbody/tr[1]/td[1]/div/a/div[1]')
    new_name_DEF = name_DEF.text
    acs_DEF = driver.find_element("xpath",
                                  '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[5]/div[4]/div[1]/table/tbody/tr[1]/td[4]/span/span[3]')
    new_acs_DEF = acs_DEF.text
    kill_DEF = driver.find_element("xpath",
                                   '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[5]/div[4]/div[1]/table/tbody/tr[1]/td[5]/span/span[3]')
    new_kill_DEF = kill_DEF.text
    death_DEF = driver.find_element("xpath",
                                    '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[5]/div[4]/div[1]/table/tbody/tr[1]/td[6]/span/span[2]/span[3]')
    new_death_DEF = death_DEF.text
    assists_DEF = driver.find_element("xpath",
                                      '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[5]/div[4]/div[1]/table/tbody/tr[1]/td[7]/span/span[3]')
    new_assists_DEF = assists_DEF.text
    adr_DEF = driver.find_element("xpath",
                                  '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[5]/div[4]/div[1]/table/tbody/tr[1]/td[10]/span/span[3]')
    new_adr_DEF = adr_DEF.text
    kast = driver.find_element("xpath",
                               '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[5]/div[4]/div[1]/table/tbody/tr[1]/td[9]/span/span[3]')
    new_kast = kast.text
    fk_DEF = driver.find_element("xpath",
                                 '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[5]/div[4]/div[1]/table/tbody/tr[1]/td[12]/span/span[3]')
    new_fk_DEF = fk_DEF.text
    fb_DEF = driver.find_element("xpath",
                                 '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[5]/div[4]/div[1]/table/tbody/tr[1]/td[13]/span/span[3]')
    new_fb_DEF = fb_DEF.text
    name2_DEF = driver.find_element("xpath",
                                    '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[5]/div[4]/div[1]/table/tbody/tr[2]/td[1]/div/a/div[1]')
    new_name2_DEF = name2_DEF.text
    acs2_DEF = driver.find_element("xpath",
                                   '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[5]/div[4]/div[1]/table/tbody/tr[2]/td[4]/span/span[3]')
    new_acs2_DEF = acs2_DEF.text
    kill2_DEF = driver.find_element("xpath",
                                    '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[5]/div[4]/div[1]/table/tbody/tr[2]/td[5]/span/span[3]')
    new_kill2_DEF = kill2_DEF.text
    death2_DEF = driver.find_element("xpath",
                                     '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[5]/div[4]/div[1]/table/tbody/tr[2]/td[6]/span/span[2]/span[3]')
    new_death2_DEF = death2_DEF.text
    assists2_DEF = driver.find_element("xpath",
                                       '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[5]/div[4]/div[1]/table/tbody/tr[2]/td[7]/span/span[3]')
    new_assists2_DEF = assists2_DEF.text
    adr2_DEF = driver.find_element("xpath",
                                   '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[5]/div[4]/div[1]/table/tbody/tr[2]/td[10]/span/span[3]')
    new_adr2_DEF = adr2_DEF.text
    kast2 = driver.find_element("xpath",
                                '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[5]/div[4]/div[1]/table/tbody/tr[2]/td[9]/span/span[3]')
    new_kast2 = kast2.text
    fk2_DEF = driver.find_element("xpath",
                                  '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[5]/div[4]/div[1]/table/tbody/tr[2]/td[12]/span/span[3]')
    new_fk2_DEF = fk2_DEF.text
    fb2_DEF = driver.find_element("xpath",
                                  '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[5]/div[4]/div[1]/table/tbody/tr[2]/td[13]/span/span[3]')
    new_fb2_DEF = fb2_DEF.text
    name3_DEF = driver.find_element("xpath",
                                    '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[5]/div[4]/div[1]/table/tbody/tr[3]/td[1]/div/a/div[1]')
    new_name3_DEF = name3_DEF.text
    acs3_DEF = driver.find_element("xpath",
                                   '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[5]/div[4]/div[1]/table/tbody/tr[3]/td[4]/span/span[3]')
    new_acs3_DEF = acs3_DEF.text
    kill3_DEF = driver.find_element("xpath",
                                    '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[5]/div[4]/div[1]/table/tbody/tr[3]/td[5]/span/span[3]')
    new_kill3_DEF = kill3_DEF.text
    death3_DEF = driver.find_element("xpath",
                                     '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[5]/div[4]/div[1]/table/tbody/tr[3]/td[6]/span/span[2]/span[3]')
    new_death3_DEF = death3_DEF.text
    assists3_DEF = driver.find_element("xpath",
                                       '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[5]/div[4]/div[1]/table/tbody/tr[3]/td[7]/span/span[3]')
    new_assists3_DEF = assists3_DEF.text
    adr3_DEF = driver.find_element("xpath",
                                   '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[5]/div[4]/div[1]/table/tbody/tr[3]/td[10]/span/span[3]')
    new_adr3_DEF = adr3_DEF.text
    kast3 = driver.find_element("xpath",
                                '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[5]/div[4]/div[1]/table/tbody/tr[3]/td[9]/span/span[3]')
    new_kast3 = kast3.text
    fk3_DEF = driver.find_element("xpath",
                                  '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[5]/div[4]/div[1]/table/tbody/tr[3]/td[12]/span/span[3]')
    new_fk3_DEF = fk3_DEF.text
    fb3_DEF = driver.find_element("xpath",
                                  '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[5]/div[4]/div[1]/table/tbody/tr[3]/td[13]/span/span[3]')
    new_fb3_DEF = fb3_DEF.text
    name4_DEF = driver.find_element("xpath",
                                    '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[5]/div[4]/div[1]/table/tbody/tr[4]/td[1]/div/a/div[1]')
    new_name4_DEF = name4_DEF.text
    acs4_DEF = driver.find_element("xpath",
                                   '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[5]/div[4]/div[1]/table/tbody/tr[4]/td[4]/span/span[3]')
    new_acs4_DEF = acs4_DEF.text
    kill4_DEF = driver.find_element("xpath",
                                    '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[5]/div[4]/div[1]/table/tbody/tr[4]/td[5]/span/span[3]')
    new_kill4_DEF = kill4_DEF.text
    death4_DEF = driver.find_element("xpath",
                                     '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[5]/div[4]/div[1]/table/tbody/tr[4]/td[6]/span/span[2]/span[3]')
    new_death4_DEF = death4_DEF.text
    assists4_DEF = driver.find_element("xpath",
                                       '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[5]/div[4]/div[1]/table/tbody/tr[4]/td[7]/span/span[3]')
    new_assists4_DEF = assists4_DEF.text
    adr4_DEF = driver.find_element("xpath",
                                   '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[5]/div[4]/div[1]/table/tbody/tr[4]/td[10]/span/span[3]')
    new_adr4_DEF = adr4_DEF.text
    kast4 = driver.find_element("xpath",
                                '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[5]/div[4]/div[1]/table/tbody/tr[4]/td[9]/span/span[3]')
    new_kast4 = kast4.text
    fk4_DEF = driver.find_element("xpath",
                                  '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[5]/div[4]/div[1]/table/tbody/tr[4]/td[12]/span/span[3]')
    new_fk4_DEF = fk4_DEF.text
    fb4_DEF = driver.find_element("xpath",
                                  '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[5]/div[4]/div[1]/table/tbody/tr[4]/td[13]/span/span[3]')
    new_fb4_DEF = fb4_DEF.text
    name5_DEF = driver.find_element("xpath",
                                    '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[5]/div[4]/div[1]/table/tbody/tr[5]/td[1]/div/a/div[1]')
    new_name5_DEF = name5_DEF.text
    acs5_DEF = driver.find_element("xpath",
                                   '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[5]/div[4]/div[1]/table/tbody/tr[5]/td[4]/span/span[3]')
    new_acs5_DEF = acs5_DEF.text
    kill5_DEF = driver.find_element("xpath",
                                    '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[5]/div[4]/div[1]/table/tbody/tr[5]/td[5]/span/span[3]')
    new_kill5_DEF = kill5_DEF.text
    death5_DEF = driver.find_element("xpath",
                                     '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[5]/div[4]/div[1]/table/tbody/tr[5]/td[6]/span/span[2]/span[3]')
    new_death5_DEF = death5_DEF.text
    assists5_DEF = driver.find_element("xpath",
                                       '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[5]/div[4]/div[1]/table/tbody/tr[5]/td[7]/span/span[3]')
    new_assists5_DEF = assists5_DEF.text
    adr5_DEF = driver.find_element("xpath",
                                   '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[5]/div[4]/div[1]/table/tbody/tr[5]/td[10]/span/span[3]')
    new_adr5_DEF = adr5_DEF.text
    kast5 = driver.find_element("xpath",
                                '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[5]/div[4]/div[1]/table/tbody/tr[5]/td[9]/span/span[3]')
    new_kast5 = kast5.text
    fk5_DEF = driver.find_element("xpath",
                                  '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[5]/div[4]/div[1]/table/tbody/tr[5]/td[12]/span/span[3]')
    new_fk5_DEF = fk5_DEF.text
    fb5_DEF = driver.find_element("xpath",
                                  '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[5]/div[4]/div[1]/table/tbody/tr[5]/td[13]/span/span[3]')
    new_fb5_DEF = fb5_DEF.text
    name6_DEF = driver.find_element("xpath",
                                    '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[5]/div[4]/div[2]/table/tbody/tr[1]/td[1]/div/a/div[1]')
    new_name6_DEF = name6_DEF.text
    acs6_DEF = driver.find_element("xpath",
                                   '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[5]/div[4]/div[2]/table/tbody/tr[1]/td[4]/span/span[3]')
    new_acs6_DEF = acs6_DEF.text
    kill6_DEF = driver.find_element("xpath",
                                    '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[5]/div[4]/div[2]/table/tbody/tr[1]/td[5]/span/span[3]')
    new_kill6_DEF = kill6_DEF.text
    death6_DEF = driver.find_element("xpath",
                                     '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[5]/div[4]/div[2]/table/tbody/tr[1]/td[6]/span/span[2]/span[3]')
    new_death6_DEF = death6_DEF.text
    assists6_DEF = driver.find_element("xpath",
                                       '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[5]/div[4]/div[2]/table/tbody/tr[1]/td[7]/span/span[3]')
    new_assists6_DEF = assists6_DEF.text
    adr6_DEF = driver.find_element("xpath",
                                   '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[5]/div[4]/div[2]/table/tbody/tr[1]/td[10]/span/span[3]')
    new_adr6_DEF = adr6_DEF.text
    kast6 = driver.find_element("xpath",
                                '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[5]/div[4]/div[2]/table/tbody/tr[1]/td[9]/span/span[3]')
    new_kast6 = kast6.text
    fk6_DEF = driver.find_element("xpath",
                                  '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[5]/div[4]/div[2]/table/tbody/tr[1]/td[12]/span/span[3]')
    new_fk6_DEF = fk6_DEF.text
    fb6_DEF = driver.find_element("xpath",
                                  '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[5]/div[4]/div[2]/table/tbody/tr[1]/td[13]/span/span[3]')
    new_fb6_DEF = fb6_DEF.text
    name7_DEF = driver.find_element("xpath",
                                    '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[5]/div[4]/div[2]/table/tbody/tr[3]/td[1]/div/a/div[1]')
    new_name7_DEF = name7_DEF.text
    acs7_DEF = driver.find_element("xpath",
                                   '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[5]/div[4]/div[2]/table/tbody/tr[2]/td[4]/span/span[3]')
    new_acs7_DEF = acs7_DEF.text
    kill7_DEF = driver.find_element("xpath",
                                    '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[5]/div[4]/div[2]/table/tbody/tr[2]/td[5]/span/span[3]')
    new_kill7_DEF = kill7_DEF.text
    death7_DEF = driver.find_element("xpath",
                                     '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[5]/div[4]/div[2]/table/tbody/tr[2]/td[6]/span/span[2]/span[3]')
    new_death7_DEF = death7_DEF.text
    assists7_DEF = driver.find_element("xpath",
                                       '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[5]/div[4]/div[2]/table/tbody/tr[2]/td[7]/span/span[3]')
    new_assists7_DEF = assists7_DEF.text
    adr7_DEF = driver.find_element("xpath",
                                   '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[5]/div[4]/div[2]/table/tbody/tr[2]/td[10]/span/span[3]')
    new_adr7_DEF = adr7_DEF.text
    kast7 = driver.find_element("xpath",
                                '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[5]/div[4]/div[2]/table/tbody/tr[2]/td[9]/span/span[3]')
    new_kast7 = kast7.text
    fk7_DEF = driver.find_element("xpath",
                                  '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[5]/div[4]/div[2]/table/tbody/tr[2]/td[12]/span/span[3]')
    new_fk7_DEF = fk7_DEF.text
    fb7_DEF = driver.find_element("xpath",
                                  '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[5]/div[4]/div[2]/table/tbody/tr[2]/td[13]/span/span[3]')
    new_fb7_DEF = fb7_DEF.text
    name8_DEF = driver.find_element("xpath",
                                    '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[5]/div[4]/div[2]/table/tbody/tr[3]/td[1]/div/a/div[1]')
    new_name8_DEF = name8_DEF.text
    acs8_DEF = driver.find_element("xpath",
                                   '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[5]/div[4]/div[2]/table/tbody/tr[3]/td[4]/span/span[3]')
    new_acs8_DEF = acs8_DEF.text
    kill8_DEF = driver.find_element("xpath",
                                    '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[5]/div[4]/div[2]/table/tbody/tr[3]/td[5]/span/span[3]')
    new_kill8_DEF = kill8_DEF.text
    death8_DEF = driver.find_element("xpath",
                                     '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[5]/div[4]/div[2]/table/tbody/tr[3]/td[6]/span/span[2]/span[3]')
    new_death8_DEF = death8_DEF.text
    assists8_DEF = driver.find_element("xpath",
                                       '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[5]/div[4]/div[2]/table/tbody/tr[3]/td[7]/span/span[3]')
    new_assists8_DEF = assists8_DEF.text
    adr8_DEF = driver.find_element("xpath",
                                   '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[5]/div[4]/div[2]/table/tbody/tr[3]/td[10]/span/span[3]')
    new_adr8_DEF = adr8_DEF.text
    kast8 = driver.find_element("xpath",
                                '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[5]/div[4]/div[2]/table/tbody/tr[3]/td[9]/span/span[3]')
    new_kast8 = kast8.text
    fk8_DEF = driver.find_element("xpath",
                                  '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[5]/div[4]/div[2]/table/tbody/tr[3]/td[12]/span/span[3]')
    new_fk8_DEF = fk8_DEF.text
    fb8_DEF = driver.find_element("xpath",
                                  '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[5]/div[4]/div[2]/table/tbody/tr[3]/td[13]/span/span[3]')
    new_fb8_DEF = fb8_DEF.text
    name9_DEF = driver.find_element("xpath",
                                    '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[5]/div[4]/div[2]/table/tbody/tr[4]/td[1]/div/a/div[1]')
    new_name9_DEF = name9_DEF.text
    acs9_DEF = driver.find_element("xpath",
                                   '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[5]/div[4]/div[2]/table/tbody/tr[4]/td[4]/span/span[3]')
    new_acs9_DEF = acs9_DEF.text
    kill9_DEF = driver.find_element("xpath",
                                    '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[5]/div[4]/div[2]/table/tbody/tr[4]/td[5]/span/span[3]')
    new_kill9_DEF = kill9_DEF.text
    death9_DEF = driver.find_element("xpath",
                                     '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[5]/div[4]/div[2]/table/tbody/tr[4]/td[6]/span/span[2]/span[3]')
    new_death9_DEF = death9_DEF.text
    assists9_DEF = driver.find_element("xpath",
                                       '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[5]/div[4]/div[2]/table/tbody/tr[4]/td[7]/span/span[3]')
    new_assists9_DEF = assists9_DEF.text
    adr9_DEF = driver.find_element("xpath",
                                   '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[5]/div[4]/div[2]/table/tbody/tr[4]/td[10]/span/span[3]')
    new_adr9_DEF = adr9_DEF.text
    kast9 = driver.find_element("xpath",
                                '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[5]/div[4]/div[2]/table/tbody/tr[4]/td[9]/span/span[3]')
    new_kast9 = kast9.text
    fk9_DEF = driver.find_element("xpath",
                                  '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[5]/div[4]/div[2]/table/tbody/tr[4]/td[12]/span/span[3]')
    new_fk9_DEF = fk9_DEF.text
    fb9_DEF = driver.find_element("xpath",
                                  '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[5]/div[4]/div[2]/table/tbody/tr[4]/td[13]/span/span[3]')
    new_fb9_DEF = fb9_DEF.text
    name10_DEF = driver.find_element("xpath",
                                     '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[5]/div[4]/div[2]/table/tbody/tr[5]/td[1]/div/a/div[1]')
    new_name10_DEF = name10_DEF.text
    acs10_DEF = driver.find_element("xpath",
                                    '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[5]/div[4]/div[2]/table/tbody/tr[5]/td[4]/span/span[3]')
    new_acs10_DEF = acs10_DEF.text
    kill10_DEF = driver.find_element("xpath",
                                     '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[5]/div[4]/div[2]/table/tbody/tr[5]/td[5]/span/span[3]')
    new_kill10_DEF = kill10_DEF.text
    death10_DEF = driver.find_element("xpath",
                                      '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[5]/div[4]/div[2]/table/tbody/tr[5]/td[6]/span/span[2]/span[3]')
    new_death10_DEF = death10_DEF.text
    assists10_DEF = driver.find_element("xpath",
                                        '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[5]/div[4]/div[2]/table/tbody/tr[5]/td[7]/span/span[3]')
    new_assists10_DEF = assists10_DEF.text
    adr10_DEF = driver.find_element("xpath",
                                    '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[5]/div[4]/div[2]/table/tbody/tr[5]/td[10]/span/span[3]')
    new_adr10_DEF = adr10_DEF.text
    kast10 = driver.find_element("xpath",
                                 '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[5]/div[4]/div[2]/table/tbody/tr[5]/td[9]/span/span[3]')
    new_kast10 = kast10.text
    fk10_DEF = driver.find_element("xpath",
                                   '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[5]/div[4]/div[2]/table/tbody/tr[5]/td[12]/span/span[3]')
    new_fk10_DEF = fk10_DEF.text
    fb10_DEF = driver.find_element("xpath",
                                   '//*[@id="wrapper"]/div[1]/div[3]/div[5]/div/div[3]/div[5]/div[4]/div[2]/table/tbody/tr[5]/td[13]/span/span[3]')
    new_fb10_DEF = fb10_DEF.text
    #######################

    isim_DEF = [new_name_DEF, new_name2_DEF, new_name3_DEF, new_name4_DEF, new_name5_DEF, new_name6_DEF, new_name7_DEF,
                new_name8_DEF, new_name9_DEF, new_name10_DEF]
    acs_DEF = [new_acs_DEF, new_acs2_DEF, new_acs3_DEF, new_acs4_DEF, new_acs5_DEF, new_acs6_DEF, new_acs7_DEF,
               new_acs8_DEF, new_acs9_DEF, new_acs10_DEF]
    kill_DEF = [new_kill_DEF, new_kill2_DEF, new_kill3_DEF, new_kill4_DEF, new_kill5_DEF, new_kill6_DEF, new_kill7_DEF,
                new_kill8_DEF, new_kill9_DEF,
                new_kill10_DEF]
    death_DEF = [new_death_DEF, new_death2_DEF, new_death3_DEF, new_death4_DEF, new_death5_DEF, new_death6_DEF,
                 new_death7_DEF, new_death8_DEF, new_death9_DEF,
                 new_death10_DEF]
    assits_DEF = [new_assists_DEF, new_assists2_DEF, new_assists3_DEF, new_assists4_DEF, new_assists5_DEF,
                  new_assists6_DEF, new_assists7_DEF,
                  new_assists8_DEF, new_assists9_DEF, new_assists10_DEF]
    adr_DEF = [new_adr_DEF, new_adr2_DEF, new_adr3_DEF, new_adr4_DEF, new_adr5_DEF, new_adr6_DEF, new_adr7_DEF,
               new_adr8_DEF, new_adr9_DEF, new_adr10_DEF]
    kast_DEF = [new_kast, new_kast2, new_kast3, new_kast4, new_kast5, new_kast6, new_kast7,
                new_kast8, new_kast9, new_kast10]
    fk_DEF = [new_fk_DEF, new_fk2_DEF, new_fk3_DEF, new_fk4_DEF, new_fk5_DEF, new_fk6_DEF, new_fk7_DEF, new_fk8_DEF,
              new_fk9_DEF, new_fk10_DEF]
    fb_DEF = [new_fb_DEF, new_fb2_DEF, new_fb3_DEF, new_fb4_DEF, new_fb5_DEF, new_fb6_DEF, new_fb7_DEF, new_fb8_DEF,
              new_fb9_DEF, new_fb10_DEF]
    map = [pick4, pick4, pick4, pick4, pick4, pick4, pick4, pick4, pick4, pick4]
    team = [new_team1, new_team1, new_team1, new_team1, new_team1, new_team2, new_team2, new_team2, new_team2,
            new_team2]
    karsılasma = [vs, vs, vs, vs, vs, vs, vs, vs, vs, vs]
    Agent = ["-", "-", "-", "-", "-", "-", "-", "-", "-", "-"]
    final_score = [final, final, final, final, final, final, final, final, final, final]
    rating = ["-", "-", "-", "-", "-", "-", "-", "-", "-", "-"]
    ad = ["defense", "defense", "defense", "defense", "defense", "defense", "defense", "defense", "defense", "defense"]
    winner = ["-", "-", "-", "-", "-", "-", "-", "-", "-", "-"]
    dff = pd.DataFrame({'isim': isim_DEF,
                        'acs': acs_DEF,
                        'kill': kill_DEF,
                        'death': death_DEF,
                        'assits': assits_DEF,
                        'adr': adr_DEF,
                        'kast': kast_DEF,
                        'fk': fk_DEF,
                        'fb': fb_DEF,
                        'map': map,
                        'Team': team,
                        'VS': karsılasma,
                        'Agent': Agent,
                        'Score': final_score,
                        'Rating': rating,
                        'Side': ad,
                        'Winner': winner})
    return dff
map4_defense = map4_defense()

# Toplam maç sayısı 2 ise
endof = pd.concat([map1_attack, map1_defense, map2_attack, map2_defense])
# Toplam maç sayısı 3 ise
endof = pd.concat([map1_attack, map1_defense, map2_attack, map2_defense, map3_attack, map3_defense])
# Toplam maç sayısı 4 ise
endof = pd.concat(
    [map1_attack, map1_defense, map2_attack, map2_defense, map3_attack, map3_defense, map4_attack, map4_defense])
# endregion

# region ALINAN MAP'LERDE DÜZELTMELER YAPIP EXCEL'E DÖNÜŞTÜRME
date = driver.find_element("xpath", '//*[@id="wrapper"]/div[1]/div[3]/div[1]/div[1]/div[2]/div/div[1]')
new_date = date.text
endof["Date"] = new_date

tournament = driver.find_element("xpath", '//*[@id="wrapper"]/div[1]/div[3]/div[1]/div[1]/div[1]/a/div/div[1]')
new_tournament = tournament.text
endof["Tournament"] = new_tournament
endof.shape
mach_id = team + '_' + new_date
endof["Mach_Id"] = mach_id
endof = endof.reset_index()
endof.to_excel('a-opening2.xlsx')
# endregion

# region EXCEL'E DÖNÜŞTÜRÜLEN VERİLERİ DF'LERE DÖNÜŞTÜRME VE TEK BİR DF OLARAK BİRLEŞTİRME
veriler = pd.read_excel("excel_dosyalari/1-b8-esports-vs-smaracis-esports-r1.xlsx")
veriler2 = pd.read_excel("excel_dosyalari/1-bigetron-arctic-vs-hike-digital-esports-r1.xlsx")
veriler3 = pd.read_excel("excel_dosyalari/1-boom-esports-vs-hike-digital-esports-r1.xlsx")
veriler4 = pd.read_excel("excel_dosyalari/1-edward-gaming-vs-on-sla2ers-q1.xlsx")
veriler5 = pd.read_excel("excel_dosyalari/1-eta-division-vs-team-liquid-r1.xlsx")
veriler6 = pd.read_excel("excel_dosyalari/1-fnatic-vs-ninjas-in-pyjamas-opening.xlsx")
veriler7 = pd.read_excel("excel_dosyalari/1-g2-esports-vs-zeta-division-q1.xlsx")
veriler8 = pd.read_excel("excel_dosyalari/1-movistar-optix-vs-9z-team-eliminator.xlsx")
veriler9 = pd.read_excel("excel_dosyalari/1-on-sla2ers-vs-maru-gaming-r1.xlsx")
veriler10 = pd.read_excel("excel_dosyalari/1-optic-gaming-vs-xerxia-esports-opening.xlsx")
veriler11 = pd.read_excel("excel_dosyalari/1_One_Breath_gaming_vs_B8_esports.xlsx")
veriler12 = pd.read_excel("excel_dosyalari/2-aim-attack-vs-funplus-phoenix-r1.xlsx")
veriler13 = pd.read_excel("excel_dosyalari/2-alter-ego-vs-persija-esports-r1.xlsx")
veriler14 = pd.read_excel("excel_dosyalari/2-bigetron-arctic-vs-persija-esports-r1.xlsx")
veriler15 = pd.read_excel("excel_dosyalari/2-damwon-gaming-vs-kone-esc-r1.xlsx")
veriler16 = pd.read_excel("excel_dosyalari/2-drx-vs-zeta-division-opening.xlsx")
veriler17 = pd.read_excel("excel_dosyalari/2-kr-esports-vs-leviat-n-upperfinal.xlsx")
veriler18 = pd.read_excel("excel_dosyalari/2-kr-esports-vs-team-liquid-valorant-opening.xlsx")
veriler19 = pd.read_excel("excel_dosyalari/2-loud-vs-team-liquid-q1.xlsx")
veriler20 = pd.read_excel("excel_dosyalari/2-maru-gaming-vs-reject-q1.xlsx")
veriler21 = pd.read_excel("excel_dosyalari/2-paper-rex-vs-the-guard-r1.xlsx")
veriler22 = pd.read_excel("excel_dosyalari/2_Natus_Vincere_vs_SMARACIS_esports.xlsx")
veriler23 = pd.read_excel("excel_dosyalari/3-5mokes-vs-smaracis-esports-r2.xlsx")
veriler24 = pd.read_excel("excel_dosyalari/3-boom-esports-vs-bigetron-arctic-r2.xlsx")
veriler25 = pd.read_excel("excel_dosyalari/3-crazy-raccoon-vs-damwon-gaming-q1.xlsx")
veriler26 = pd.read_excel("excel_dosyalari/3-crazy-raccoon-vs-on-sla2ers-r2.xlsx")
veriler27 = pd.read_excel("excel_dosyalari/3-drx-vs-zeta-division-r2.xlsx")
veriler27 = pd.read_excel("excel_dosyalari/3-ninjas-in-pyjamas-vs-drx-winners.xlsx")
veriler28 = pd.read_excel("excel_dosyalari/3-onic-g-vs-bigetron-arctic-semifinal.xlsx")
veriler29 = pd.read_excel("excel_dosyalari/3-paper-rex-vs-drx-q1.xlsx")
veriler30 = pd.read_excel("excel_dosyalari/3-xerxia-esports-vs-team-liquid-winners.xlsx")
veriler31 = pd.read_excel("excel_dosyalari/3_5MOKES_vs_Aim.Attack.xlsx")
veriler32 = pd.read_excel("excel_dosyalari/4-alter-ego-vs-boom-esports-final.xlsx")
veriler33 = pd.read_excel("excel_dosyalari/4-boom-esports-vs-alter-ego-semifinal.xlsx")
veriler34 = pd.read_excel("excel_dosyalari/4-fnatic-vs-zeta-division-elimination.xlsx")
veriler35 = pd.read_excel("excel_dosyalari/4-g2-esports-vs-paper-rex-r2.xlsx")
veriler36 = pd.read_excel("excel_dosyalari/4-northeption-vs-kone-esc-q1.xlsx")
veriler37 = pd.read_excel("excel_dosyalari/4-one-breath-gaming-vs-funplus-phoenix-r2.xlsx")
veriler38 = pd.read_excel("excel_dosyalari/4-optic-gaming-vs-kr-esports-elimination.xlsx")
veriler39 = pd.read_excel("excel_dosyalari/4-reject-vs-kone-esc-r2.xlsx")
veriler40 = pd.read_excel("excel_dosyalari/4-the-guard-vs-optic-gaming-q1.xlsx")
veriler41 = pd.read_excel("excel_dosyalari/4_KPI_Gaming_vsFun_Plus_Phoenix.xlsx")
veriler42 = pd.read_excel("excel_dosyalari/5-5mokes-vs-funplus-phoenix-r3.xlsx")
veriler43 = pd.read_excel("excel_dosyalari/5-edward-gaming-vs-reject-semifinal.xlsx")
veriler44 = pd.read_excel("excel_dosyalari/5-g2-esports-vs-loud-semifinal.xlsx")
veriler45 = pd.read_excel("excel_dosyalari/5-on-sla2ers-vs-kone-esc-r3.xlsx")
veriler46 = pd.read_excel("excel_dosyalari/5-one-breath-gaming-vs-natus-vincere_semifinal.xlsx")
veriler47 = pd.read_excel("excel_dosyalari/5-onic-g-vs-alter-ego-final.xlsx")
veriler48 = pd.read_excel("excel_dosyalari/5-xerxia-esports-vs-optic-gaming-decider.xlsx")
veriler49 = pd.read_excel("excel_dosyalari/5-zeta-division-vs-paper-rex-r3.xlsx")
veriler50 = pd.read_excel("excel_dosyalari/6-5mokes-vs-kpi-gaming-semifinal.xlsx")
veriler51 = pd.read_excel("excel_dosyalari/6-crazy-raccoon-vs-northeption-semifinal.xlsx")
veriler52 = pd.read_excel("excel_dosyalari/6-drx-vs-optic-gaming-semifinal.xlsx")
veriler53 = pd.read_excel("excel_dosyalari/6-kpi-gaming-vs-funplus-phoenix-r4.xlsx")
veriler54 = pd.read_excel("excel_dosyalari/6-northeption-vs-on-sla2ers-final.xlsx")
veriler55 = pd.read_excel("excel_dosyalari/6-onic-g-vs-alter-ego-grandfinal.xlsx")
veriler56 = pd.read_excel("excel_dosyalari/6-optic-gaming-vs-zeta-division-final.xlsx")
veriler57 = pd.read_excel("excel_dosyalari/7-edward-gaming-vs-northeption-final.xlsx")
veriler58 = pd.read_excel("excel_dosyalari/7-loud-vs-optic-gaming-final.xlsx")
veriler59 = pd.read_excel("excel_dosyalari/7-natus-vincere-vs-funplus-phoenix-upper_final.xlsx")
veriler60 = pd.read_excel("excel_dosyalari/8-edward-gaming-vs-on-sla2ers-grand_final.xlsx")
veriler61 = pd.read_excel("excel_dosyalari/8-loud-vs-optic-gaming-grandfinal.xlsx")
veriler62 = pd.read_excel("excel_dosyalari/smaracis-esports-vs-one-breath-gaming-final.xlsx")

df = pd.concat([veriler, veriler2, veriler3, veriler4, veriler5, veriler5, veriler6,
                veriler7, veriler8, veriler9, veriler10, veriler11, veriler12, veriler13,
                veriler14, veriler15, veriler16, veriler17, veriler18, veriler19, veriler20,
                veriler21, veriler22, veriler23, veriler24, veriler25, veriler26, veriler27, veriler27,
                veriler28, veriler28, veriler29, veriler30, veriler31, veriler32, veriler33, veriler34,
                veriler35, veriler36, veriler37, veriler38, veriler39, veriler40, veriler41, veriler42,
                veriler43, veriler44, veriler45, veriler46, veriler47, veriler48, veriler49, veriler50,
                veriler51, veriler52, veriler53, veriler54, veriler55, veriler56, veriler57, veriler58,
                veriler59, veriler60, veriler61, veriler62])
# endregion

# region DF'E HIZLI BİR GÖZ ATMA VE VERİ ÖN İŞELEMEDE BULUNMA
def veriyi_hazirlama(dataframe):
    a = ["Rating", "acs", "adr", "kast", "Score", "Winner", "Attack Round", "Defense Round", "Pistol_Round"]
    for i in a:
        return dataframe[i] == pd.to_numeric(dataframe[i], errors="coerce")


veriyi_hazirlama(df)
df.info()

# ajanların frekansı
df.groupby("Agent").agg({"Agent": ["count"]})

# Player'ların frekansı
df.groupby("isim").agg({"isim": "count"})
c = df.groupby("isim").agg({"isim": "count"})
c.shape  # 186 player var.

# c df'i düzenleme
c["frekans"] = c["isim"]
c.reset_index(inplace=True)
c.drop("isim", axis=1, inplace=True)
c.sort_values("frekans", ascending=False)


# Oyuncuları sırlamak için ağırlıklandırma
def weighted_player_score(dataframe, dataframe2, frekans=40, rating=60):
    return (dataframe["Rating"] * frekans / 100 +
            dataframe2["frekans"] * rating / 100)


l = weighted_player_score(df, c)
l.sort_values(ascending=False)
l.info()
l
df["frekans"] = df["isim"]
df.groupby("isim").agg({"Rating": "mean",
                        "frekans": "count"}).sort_values("frekans", ascending=False)
############################################
# Tip değişimi
############################################
df["Rating"] = pd.to_numeric(df["Rating"], errors='coerce')
df["acs"] = pd.to_numeric(df["acs"], errors='coerce')
df["adr"] = pd.to_numeric(df["adr"], errors='coerce')
# df["kast"] = pd.to_numeric(df["kast"], errors='coerce')
# df["Score"] = pd.to_numeric(df["Score"], errors='coerce')
df["Winner"] = pd.to_numeric(df["Winner"], errors='coerce')
# df["Attack Round"] = pd.to_numeric(df["Attack Round"], errors='coerce')
df["Defense Round"] = pd.to_numeric(df["Defense Round"], errors='coerce')
df["Pistol_Round"] = pd.to_numeric(df["Pistol_Round"], errors='coerce')
df["Date"] = df["Date"].astype("datetime64[ns]")

##########################################
# Ajanların uniq'liği
##########################################

df1 = pd.read_csv("df_final1")
df1.info()
df1["Rating"] = pd.to_numeric(df1["Rating"], errors='coerce')
df1.groupby("Agent").agg({"Agent": ["count"]})
df.groupby("Agent").agg({"Agent": ["count"]})
c = df.groupby("isim").agg({"isim": "count"})
c["frekans"] = c["isim"]
c.reset_index(inplace=True)
c.sort_values("frekans", ascending=False)
c.drop("isim", axis=1, inplace=True)
c.sort_values(by="isim", ascending=False)
c["frekans"].sum()
df.isnull().sum()

##########################################
# Takımların Winner'lık durumuna göre 0 ve 1'leri atama
##########################################

df2 = pd.read_excel("valorant_tum_veriler_son_hali.xlsx")
df2.info()
df2.drop("Unnamed: 0", axis=1, inplace=True)

a = df2[["Score1", "Score2", "Team1", "Team2", "Winner"]]
a["Winner"] = a["Winner"].replace("-", "")
a["Winner"] = pd.to_numeric(a["Winner"], errors='coerce')
a["Winner"] = a["Team1"]
a
a["fark"] = a["Score2"] - a["Score1"]

for index, i in enumerate(a["fark"]):
    if i > 0:
        a["Winner"][index] = a["Team2"][index]


df2["Winner"] = a["Winner"]
df2.head(100)
df3 = pd.DataFrame()
df3["Team"] = df2["Team"]
df3["Winner"] = df2["Winner"]
df3
df3["Winner2"] = df2.apply(lambda x: 1 if x in df2["Winner"] else 0 for x in df2["Team"])
df3 = pd.get_dummies(df2, columns=["Winner"], drop_first=True)
df3.columns
# endregion

# ADIM1
# region
df = pd.read_excel(r"C:\Users\USER\PycharmProjects\pythonProject\VALOProject1\Valorant_veri_son _inş.xlsx")
df["Rating"] = pd.to_numeric(df["Rating"], errors='coerce')
df["map_win"] = pd.to_numeric(df["map_win"], errors='coerce')
df["map_win"] = pd.to_numeric(df["map_win"], errors='coerce')
df["Attack Round"] = pd.to_numeric(df["Attack Round"], errors='coerce')

bb =df.groupby("isim").agg({"Rating": "mean",
                        "Attack Round": "sum",
                        "map_win": "sum"})

def weighted_score(dataframe, w1=99, w2=1 ):
    return (dataframe["Rating"] * w1 / 100 +
            dataframe["Attack Round"] * w2 / 100)

bb["weighted_score"] = weighted_score(bb)
bb.reset_index(inplace=True)
bb["weighted_score"] = bb["weighted_score"] * 90000
bb = bb.sort_values("weighted_score", ascending=False)
bb.reset_index(inplace=True)

df12_son = df.merge(bb, on='isim')
df12_son.to_csv('son_galiba1299.csv')


dff1_son = pd.read_csv(r'C:\Users\USER\PycharmProjects\pythonProject\VALOProject1\olduxxbb.csv')
dff1_son.groupby(["isim",])["Agent"].agg("count")
# endregion

# ADIM2 OYUNCULARI SIRALAMA

# region  OYUNCU VE AJAN SCORE'LERİNİ ÇIKARMA VE AĞIRLIKLANDIRMA
# veriyi hazırlama
def veriyi_hazirlama(dataframe):
    a = ["Rating", "acs", "adr", "kast", "Score", "Winner", "Attack Round", "Defense Round", "Pistol_Round"]
    for i in a:
        return dataframe[i] == pd.to_numeric(dataframe[i], errors="coerce")


veriyi_hazirlama(df)
df.info()

# ajanların frekansı
df.groupby("Agent").agg({"Agent": ["count"]})

# Player'ların frekansı
df.groupby("isim").agg({"isim": "count"})
c = df.groupby("isim").agg({"isim": "count"})
c.shape  # 186 player var.

# c df'i düzenleme
c["frekans"] = c["isim"]
c.reset_index(inplace=True)
c.drop("isim", axis=1, inplace=True)
c.sort_values("frekans", ascending=False)


# Oyuncuları sıralamak için ağrlıklandırma
def weighted_player_score(dataframe, dataframe2, frekans=40, rating=60):
    return (dataframe["Rating"] * frekans / 100 +
            dataframe2["frekans"] * rating / 100)


l = weighted_player_score(df, c)
l.sort_values(ascending=False)
l.info()
l
df["frekans"] = df["isim"]
df.groupby("isim").agg({"Rating": "mean",
                        "frekans": "count"}).sort_values("frekans", ascending=False)
# endregion

# ADIM3 VALORANT 2022 ISTANBUL CAHMPIONS MAÇ TAHMİNLERİNDE BULUNMA

# region BASKILAMA
def outlier_threshold(dataframe, col_name, q1=0.05, q3=0.95):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_lmt = quartile3 + 1.5 * interquantile_range
    low_lmt = quartile1 - 1.5 * interquantile_range
    return low_lmt, up_lmt


def check_outlier(dataframe, col_name):
    low_lmt, up_lmt = outlier_threshold(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_lmt) | (dataframe[col_name] < low_lmt)].any(axis=None):
        return True
    else:
        return False


def replace_with_thresholds(dataframe, variable):
    low_lmt, up_lmt = outlier_threshold(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_lmt), variable] = low_lmt
    dataframe.loc[(dataframe[variable] > up_lmt), variable] = up_lmt


# df için:
num_cols = [col for col in df.columns if df[col].dtypes != "O"]
for col in num_cols:
    print(col, check_outlier(df, col))

for col in num_cols:
    print(col, replace_with_thresholds(df, col))

for col in num_cols:
    print(col, check_outlier(df, col))

# df1 için:
num_cols = [col for col in df1.columns if df1[col].dtypes != "O"]
for col in num_cols:
    print(col, check_outlier(df1, col))

for col in num_cols:
    print(col, replace_with_thresholds(df1, col))

for col in num_cols:
    print(col, check_outlier(df1, col))

# df0 için:
num_cols = [col for col in df0.columns if df0[col].dtypes != "O"]
for col in num_cols:
    print(col, check_outlier(df0, col))

for col in num_cols:
    print(col, replace_with_thresholds(df0, col))

for col in num_cols:
    print(col, check_outlier(df0, col))
# endregion

# region TEST VE TRAIN AYIRMA
df0.loc[(df0["map_win_home"] < 0.5), "win_result"] = "Away"
df0.loc[(df0["map_win_home"] >= 0.5), "win_result"] = "Home"
df0.loc[(df0["win_result"] == "Home"), "home_Win"] = "1"
df0.loc[(df0["win_result"] == "Away"), "home_Win"] = "0"

df1.loc[(df1["map_win_home"] < 0.5), "win_result"] = "Away"
df1.loc[(df1["map_win_home"] >= 0.5), "win_result"] = "Home"
df1.loc[(df1["win_result"] == "Home"), "home_Win"] = "1"
df1.loc[(df1["win_result"] == "Away"), "home_Win"] = "0"

x_train = df1.drop(["Home", "Away", "map_win_away", "Unnamed: 0", "map_win_home", "win_result", "home_Win"], axis=1)
y_train = df1["home_Win"]
df1.shape
x_test = df0.drop(["Home", "Away", "map_win_away", "Unnamed: 0", "map_win_home", "win_result", "home_Win"], axis=1)
y_test = df0["home_Win"]
x_test.shape
# endregion

# region STANDARTLAŞTIRMA VE MODELİ KURMA
ss = StandardScaler()
x_train = ss.fit_transform(x_train)
x_test = ss.transform(x_test)
# endregion

# region TÜM MODELLERDE TAHMİNLEME YAPMA VE EN İYİ SONUCU BULMA
model = KNeighborsClassifier().fit(x_train, y_train)
y_pred = model.predict(x_test)
print(classification_report(y_test, y_pred))
model.score(x_train, y_train)
model.score(x_test, y_test)

model = DecisionTreeClassifier().fit(x_train, y_train)
y_pred = model.predict(x_test)
print(classification_report(y_test, y_pred))

model = LogisticRegression().fit(x_train, y_train)
y_pred = model.predict(x_test)
print(classification_report(y_test, y_pred))

model = RandomForestClassifier().fit(x_train, y_train)
y_pred = model.predict(x_test)
print(classification_report(y_test, y_pred))

model = CatBoostClassifier().fit(x_train, y_train)
y_pred = model.predict(x_test)
print(classification_report(y_test, y_pred))

model = LGBMClassifier().fit(x_train, y_train)
y_pred = model.predict(x_test)
print(classification_report(y_test, y_pred))

model = XGBClassifier().fit(x_train, y_train)
y_pred = model.predict(x_test)
print(classification_report(y_test, y_pred))
# endregion

# region MODEL BAŞARISINI
model.intercept_[0]  # b = 0.3810612854387374
model.coef_[0][0]  # w = 1.2902895804637795
model.score(x_train, y_train)  # 0.9095238095238095
model.score(x_test, y_test)  # 0.7857142857142857

mean_squared_error(y_test, y_pred)  # 0.21428571428571427

y_test.mean()  # 0.5714285714285714
y_test.std()  # 0.5039526306789697

y_test = pd.to_numeric(y_test)
y_test = y_test.astype("string")
x_test.mean()  # 0.11810646000105288
x_test.std()  # 0.829419017184578

y_pred = model.predict(x_test)
y_prob = model.predict_proba(x_test)[:, 1]
roc_auc_score(y_test, y_prob)  # 0.9114583333333334
# endregion

# region EN İYİ MODELE GÖRE TAHMİNLEMEDE BULUNMA
train = df1.drop(["Home", "Away", "map_win_away", "Unnamed: 0", "map_win_home", "win_result", "home_Win", "acs_away"], axis=1)
train = pd.DataFrame(train)
train.to_excel("train.xlsx")
train =pd.read_excel("train.xlsx")
y_train = df1["home_Win"]
y_train.to_excel("y_train.xlsx")
y_train = pd.read_excel("y_train.xlsx")

x_test2 = pd.read_excel("makine_ogr_artık_sonn.xlsx")
x_test2 = pd.DataFrame(x_test2)
y_test = df0["home_Win"]

ss = StandardScaler()
train = ss.fit_transform(train)
x_test2 = ss.transform(x_test2)

df1.columns
XX = pd.DataFrame(x_test2)
random_user = XX
k = model.predict(random_user)
k.shape
k=list(k)
mo=pd.DataFrame(k)
mo.shape

mo.to_excel("tahminler.xlsx")
preds = RandomForestClassifier.predict(x_test2)
from sklearn.metrics import accuracy_score
acc = accuracy_score(preds, mo)
acc
# endregion

# region TAKIMLARI AĞIRLIKLANDIRMA
df1.groupby("Home")['kill_home', 'death_home', 'assits_home', 'adr_home', 'kast_home', 'fb_home',
                    'fk_home', 'Rating_home', 'Pistol_Round_home', 'acs_away', 'kill_away',
                    'death_away', 'assits_away', 'adr_away', 'kast_away', 'fb_away', 'fk_away', 'Rating_away',
                    'Pistol_Round_away'].mean()

k = pd.DataFrame()
k = pd.read_excel("random_text.xlsx")

df1.columns

df3 = pd.concat([df0, df1], ignore_index=True)

df3.groupby("Home")['kill_home', 'death_home', 'assits_home', 'adr_home', 'kast_home', 'fb_home',
                    'fk_home', 'Rating_home', 'Pistol_Round_home', 'acs_away', 'kill_away',
                    'death_away', 'assits_away', 'adr_away', 'kast_away', 'fb_away', 'fk_away', 'Rating_away',
                    'Pistol_Round_away', "home_Win"].mean()

df3.groupby("Home").agg({"Home": "count",
                         "home_Win": "sum"})

df3["home_Win"] = df3["home_Win"].astype(int)
a = df3.groupby("Home").agg({"Home": "count",
                             "home_Win": "sum"})


a["Count"] = a["Home"]
a["agırlıklı_win"] = a["home_Win"] / a["Home"]
df3["Home"].unique()  # 73 Takım
a.sort_values(by="agırlıklı_win", ascending=False)

c.describe().T

a.loc[(a["Count"] <= 4), "mac_sayisi_agırlık"] = 0.25
a.loc[(a["Count"] <= 8) & (a["Count"] > 4), "mac_sayisi_agırlık"] = 0.30
a.loc[(a["Count"] <= 12) & (a["Count"] > 8), "mac_sayisi_agırlık"] = 0.35
a.loc[(a["Count"] <= 14) & (a["Count"] > 12), "mac_sayisi_agırlık"] = 0.40

a.to_excel("Ağırlıklandırma.xlsx")

c = pd.read_excel("ratingssee.xlsx")

c.loc[(c["rating"] <= 1500), "rating_siralama"] = 0.10
c.loc[(c["rating"] <= 1800) & (c["rating"] > 1500), "rating_siralama"] = 0.20
c.loc[(c["rating"] <= 2200) & (c["rating"] > 1800), "rating_siralama"] = 0.30
c.loc[(c["rating"] <= 2400) & (c["rating"] > 2200), "rating_siralama"] = 0.40

c["net_agirlik"] = c["agırlıklı_win"] * 0.15 + c["mac_sayisi_agırlık"] * 0.35 + c["rating_siralama"] * 0.50

c.sort_values(by="net_agirlik", ascending=False)


d = pd.DataFrame()
d["home"] = c["Home"]
d["agirliklandırma"] = c["net_agirlik"]
d.sort_values(by="agirliklandırma", ascending=False)

d.groupby("home").agg({"home": "count"})
d["Home"] = d["home"]

d.drop("home", axis=1, inplace=True)
fatma = df3.groupby("Home")['kill_home', 'death_home', 'assits_home', 'adr_home', 'kast_home', 'fb_home',
                            'fk_home', 'Rating_home', 'Pistol_Round_home', 'acs_away', 'kill_away',
                            'death_away', 'assits_away', 'adr_away', 'kast_away', 'fb_away', 'fk_away', 'Rating_away',
                            'Pistol_Round_away'].mean()

df4 = fatma.merge(d, on="Home", how="inner")

df4.columns
df4["kill_home"] = df4["kill_home"] * df4["agirliklandırma"]
df4["death_home"] = df4["death_home"] * df4["agirliklandırma"]
df4["assits_home"] = df4["assits_home"] * df4["agirliklandırma"]
df4["adr_home"] = df4["adr_home"] * df4["agirliklandırma"]
df4["kast_home"] = df4["kast_home"] * df4["agirliklandırma"]
df4["fb_home"] = df4["fb_home"] * df4["agirliklandırma"]
df4["fk_home"] = df4["fk_home"] * df4["agirliklandırma"]
df4["Rating_home"] = df4["Rating_home"] * df4["agirliklandırma"]
df4["Pistol_Round_home"] = df4["Pistol_Round_home"] * df4["agirliklandırma"]
df4["acs_away"] = df4["acs_away"] * df4["agirliklandırma"]
df4["kill_away"] = df4["kill_away"] * df4["agirliklandırma"]
df4["death_away"] = df4["death_away"] * df4["agirliklandırma"]
df4["assits_away"] = df4["assits_away"] * df4["agirliklandırma"]
df4["adr_away"] = df4["adr_away"] * df4["agirliklandırma"]
df4["kast_away"] = df4["kast_away"] * df4["agirliklandırma"]
df4["fb_away"] = df4["fb_away"] * df4["agirliklandırma"]
df4["fk_away"] = df4["fk_away"] * df4["agirliklandırma"]
df4["Rating_away"] = df4["Rating_away"] * df4["agirliklandırma"]
df4["Pistol_Round_away"] = df4["Pistol_Round_away"] * df4["agirliklandırma"]
df4

df4.drop(["acs_away", "kill_away", "death_away", "assits_away", "adr_away",
          "kast_away", "fb_away", "fk_away", "Rating_away", "Pistol_Round_away", "agirliklandırma"], axis=1,
         inplace=True)
df4.drop("agirliklandırma", axis=1, inplace=True)

df4.sort_values(by="kill_home", ascending=False)


def weighted_score(df, w1=30, w2=35, w3=35):
    return (df["kast_home"] * w1 / 100 + df["Pistol_Round_home"] * w2 / 100 + df["kill_home"] * w3 / 100)


weighted_score(df4)
df4["weighted_score"] = weighted_score(df4)
df4.sort_values(by="weighted_score", ascending=False)
# endregion

# region MAÇ TAHMİNLEMESİ İÇİN TEST SETİ OLUŞTURMA
df0.groupby("Home")
df0.loc[df1["Home"] == "100 Thieves"][['kill_home', 'death_home', 'assits_home', 'adr_home', 'kast_home', 'fb_home',
                                       'fk_home', 'Rating_home', 'Pistol_Round_home']].mean()

home = df0[['Home', 'kill_home', 'death_home', 'assits_home', 'adr_home', 'kast_home', 'fb_home',
            'fk_home', 'Rating_home', 'Pistol_Round_home']]
away = df0[
    ['Away', 'kill_away', 'death_away', 'assits_away', 'adr_away', 'kast_away', 'fb_away', 'fk_away', 'Rating_away',
     'Pistol_Round_away']]
a = for_all("100 Thieves")

home.rename(columns={'Home': 'Name'}, inplace=True)
home.to_excel("home.xlsx")
away.to_excel("away.xlsx")
away.rename(columns={'Away': 'Name'}, inplace=True)


home_away = pd.read_excel("home_away.xlsx")
home_away["Name"].unique()

def for_all(name):
    return (home_away.loc[(home_away["Name"] == name)][
                ['kill', 'death', 'assits', 'adr', 'kast', 'fb',
                 'fk', 'Rating', 'Pistol_Round']].mean())


for_all("OpTic Gaming")

team_name = ['LOUD', 'DRX', 'FunPlus Phoenix', 'OpTic Gaming', '100 Thieves',
       'Paper Rex', 'FNATIC', 'KRÜ Esports', 'EDward Gaming',
       'BOOM Esports', 'XERXIA Esports', 'Leviatán', 'XSET',
       'ZETA DIVISION', 'Team Liquid', 'FURIA']
team_name = pd.DataFrame(team_name)
tema_list = list(team_name)

for index, c in enumerate(team_name, 1):
    print("a=for_all(" + "'" + c + "'" + ")")
    print("a = a.reset_index()")
    print("q" + str(index) + " = a.transpose()")
team_name
a = a.reset_index()
transpose = a.transpose()
print(transpose)

# region
a = for_all('100 Thieves')
a = a.reset_index()
q1 = a.transpose()
a = for_all('47 Gaming')
a = a.reset_index()
q2 = a.transpose()
a = for_all('5MOKES')
a = a.reset_index()
q3 = a.transpose()
a = for_all('9z Team')
a = a.reset_index()
q4 = a.transpose()
a = for_all('ABC')
a = a.reset_index()
q5 = a.transpose()
a = for_all('Acend')
a = a.reset_index()
q6 = a.transpose()
a = for_all('Aim.Attack')
a = a.reset_index()
q7 = a.transpose()
a = for_all('Alter Ego')
a = a.reset_index()
q8 = a.transpose()
a = for_all('Aricat Esport')
a = a.reset_index()
q9 = a.transpose()
a = for_all('Attack All Around')
a = a.reset_index()
q10 = a.transpose()
a = for_all('B8 Esports')
a = a.reset_index()
q11 = a.transpose()
a = for_all('BBL Esports')
a = a.reset_index()
q12 = a.transpose()
a = for_all('BLEED')
a = a.reset_index()
q13 = a.transpose()
a = for_all('BOOM Esports')
a = a.reset_index()
q14 = a.transpose()
a = for_all('BearClaw Gaming')
a = a.reset_index()
q15 = a.transpose()
a = for_all('Bigetron Arctic')
a = a.reset_index()
q16 = a.transpose()
a = for_all('CERBERUS Esports')
a = a.reset_index()
q17 = a.transpose()
a = for_all('Cloud9')
a = a.reset_index()
q18 = a.transpose()
a = for_all('Crazy Raccoon')
a = a.reset_index()
q19 = a.transpose()
a = for_all('Crest Gaming Zst')
a = a.reset_index()
q20 = a.transpose()
a = for_all('DAMWON Gaming')
a = a.reset_index()
q21 = a.transpose()
a = for_all('DRX')
a = a.reset_index()
q22 = a.transpose()
a = for_all('Daytrade Nursery')
a = a.reset_index()
q23 = a.transpose()
a = for_all('E-Xolos LAZER')
a = a.reset_index()
q24 = a.transpose()
a = for_all('EDward Gaming')
a = a.reset_index()
q25 = a.transpose()
a = for_all('Enigma Gaming')
a = a.reset_index()
q26 = a.transpose()
a = for_all('FAV gaming')
a = a.reset_index()
q27 = a.transpose()
a = for_all('FNATIC')
a = a.reset_index()
q28 = a.transpose()
a = for_all('FURIA')
a = a.reset_index()
q29 = a.transpose()
a = for_all('FUSION')
a = a.reset_index()
q30 = a.transpose()
a = for_all('FW Esports')
a = a.reset_index()
q31 = a.transpose()
a = for_all('Fancy United Esports')
a = a.reset_index()
q32 = a.transpose()
a = for_all('Free Banana and Icecream')
a = a.reset_index()
q33 = a.transpose()
a = for_all('FunPlus Phoenix')
a = a.reset_index()
q34 = a.transpose()
a = for_all('G2 Esports')
a = a.reset_index()
q35 = a.transpose()
a = for_all('Gaimin Gladiators')
a = a.reset_index()
q36 = a.transpose()
a = for_all('Ghetto Artist')
a = a.reset_index()
q37 = a.transpose()
a = for_all('Global Esports')
a = a.reset_index()
q38 = a.transpose()
a = for_all('Griffin E-Sports')
a = a.reset_index()
q39 = a.transpose()
a = for_all('Guild Esports')
a = a.reset_index()
q40 = a.transpose()
a = for_all('JaiSlai')
a = a.reset_index()
q41 = a.transpose()
a = for_all('KONE eSC')
a = a.reset_index()
q42 = a.transpose()
a = for_all('KRÜ Esports')
a = a.reset_index()
q43 = a.transpose()
a = for_all('LOUD')
a = a.reset_index()
q44 = a.transpose()
a = for_all('LaZe')
a = a.reset_index()
q45 = a.transpose()
a = for_all('Leviatán')
a = a.reset_index()
q46 = a.transpose()
a = for_all('Los Grandes')
a = a.reset_index()
q47 = a.transpose()
a = for_all('Lumsum')
a = a.reset_index()
q48 = a.transpose()
a = for_all('M3 Champions')
a = a.reset_index()
q49 = a.transpose()
a = for_all('MIBR')
a = a.reset_index()
q50 = a.transpose()
a = for_all('Made in Thailand')
a = a.reset_index()
q51 = a.transpose()
a = for_all('Maru Gaming')
a = a.reset_index()
q52 = a.transpose()
a = for_all('NORTHEPTION')
a = a.reset_index()
q53 = a.transpose()
a = for_all('Natus Vincere')
a = a.reset_index()
q54 = a.transpose()
a = for_all('Ninjas In Pyjamas')
a = a.reset_index()
q55 = a.transpose()
a = for_all('No Namers')
a = a.reset_index()
q56 = a.transpose()
a = for_all('ONIC G')
a = a.reset_index()
q57 = a.transpose()
a = for_all('ORDER')
a = a.reset_index()
q58 = a.transpose()
a = for_all('One Breath Gaming')
a = a.reset_index()
q59 = a.transpose()
a = for_all('OpTic Gaming')
a = a.reset_index()
q60 = a.transpose()
a = for_all('PAMPAS')
a = a.reset_index()
q61 = a.transpose()
a = for_all('Paper Rex')
a = a.reset_index()
q62 = a.transpose()
a = for_all('Pass Gaming')
a = a.reset_index()
q63 = a.transpose()
a = for_all('Purple Mood E-Sport')
a = a.reset_index()
q64 = a.transpose()
a = for_all('REIGNITE')
a = a.reset_index()
q65 = a.transpose()
a = for_all('REJECT')
a = a.reset_index()
q66 = a.transpose()
a = for_all('Rex Regum Qeon')
a = a.reset_index()
q67 = a.transpose()
a = for_all('SPEAR GAMING')
a = a.reset_index()
q68 = a.transpose()
a = for_all('Sainuahualouis')
a = a.reset_index()
q69 = a.transpose()
a = for_all('SunXet Club')
a = a.reset_index()
q70 = a.transpose()
a = for_all('Team Liquid')
a = a.reset_index()
q71 = a.transpose()
a = for_all('Team Secret')
a = a.reset_index()
q72 = a.transpose()
a = for_all('XERXIA Esports')
a = a.reset_index()
q73 = a.transpose()
# endregion q2Q
q74 = pd.concat(
    [q1, q2, q3, q4, q5, q6, q7, q8, q9, q10, q11, q12, q13, q14, q15, q16, q17, q18, q19, q20, q21, q22, q23, q24, q25,
     q26, q27, q28, q29, q30, q31, q32, q33, q34, q35, q36, q37, q38, q39,
     q40, q41, q42, q43, q44, q45, q46, q47, q48, q49, q50, q51, q52, q53, q54, q55, q56, q57, q58, q59, q60, q61, q62,
     q63, q64, q65, q66, q67, q68, q69, q70, q71, q72, q73])
# region  Ist Champions
a=for_all('LOUD')
a = a.reset_index()
q1 = a.transpose()
a=for_all('DRX')
a = a.reset_index()
q2 = a.transpose()
a=for_all('FunPlus Phoenix')
a = a.reset_index()
q3 = a.transpose()
a=for_all('OpTic Gaming')
a = a.reset_index()
q4 = a.transpose()
a=for_all('100 Thieves')
a = a.reset_index()
q5 = a.transpose()
a=for_all('Paper Rex')
a = a.reset_index()
q6 = a.transpose()
a=for_all('FNATIC')
a = a.reset_index()
q7 = a.transpose()
a=for_all('KRÜ Esports')
a = a.reset_index()
q8 = a.transpose()
a=for_all('EDward Gaming')
a = a.reset_index()
q9 = a.transpose()
a=for_all('BOOM Esports')
a = a.reset_index()
q10 = a.transpose()
a=for_all('XERXIA Esports')
a = a.reset_index()
q11 = a.transpose()
a=for_all('Leviatán')
a = a.reset_index()
q12 = a.transpose()
a=for_all('XSET')
a = a.reset_index()
q13 = a.transpose()
a=for_all('ZETA DIVISION')
a = a.reset_index()
q14 = a.transpose()
a=for_all('Team Liquid')
a = a.reset_index()
q15 = a.transpose()
a=for_all('FURIA')
a = a.reset_index()
q16 = a.transpose()
# endregion

q_cham = pd.concat([q1, q2, q3, q4, q5, q6, q7, q8, q9, q10, q11, q12, q13, q14, q15, q16])
q_cham.to_excel("q_cham.xlsx")
# endregion

# region DEPLOY

############################################BİRİNCİ ADIM####################################################


st.title('FUTURE OF E-SPORTS ')


col1, col2= st.columns(2)

with col1:
   #başlık
   #st.title('Future of e-sports')
   # dataframe
   df = pd.read_csv(r'C:\Users\USER\PycharmProjects\pythonProject\VALOProject1\son_galiba1299.csv')

   df["Pistol_Round"] = pd.to_numeric(df["Pistol_Round"], errors='coerce')
   df.rename(columns={'weighted_score': 'Transfer Bedeli'}, inplace=True)
   df.rename(columns={'Attack Round_x': 'Round İstatistikleri'}, inplace=True)
   df.rename(columns={'map': 'Harita'}, inplace=True)
   df.rename(columns={'Side': 'Bölge'}, inplace=True)
   df["Transfer Bedeli"] = pd.to_numeric(df["Transfer Bedeli"], errors='coerce')

   # bütçe
   st.header('Takımın için en iyisini seç')
   bedel = st.slider(label='Karşılanabilcek tutar', min_value=0, max_value=1000000, value=500, step=100)

   # ajan
   sorted_unique_agent = sorted(df.Agent.unique())
   selected_agent = st.multiselect('Ajanlar', sorted_unique_agent)

   #dataframenin filtre edilmesi
   df_result = df[df["Transfer Bedeli"] < bedel]
   df_result2 = df_result[(df_result.Agent.isin(selected_agent))]
   df_result3 = df_result2.groupby(["isim", "Team"]).agg({"Transfer Bedeli": "mean"}).sort_values("Transfer Bedeli", ascending=False)

   st.error('İlgili girilen bilgiler doğrultusunda: ' + str(df_result3.shape[0]) + ' tane oyuncu bulundu. ')
   st.write(df_result3)

with col2:
   #video
   like=open("Top 10 Champions Plays So Far _ VALORANT Champions İstanbul 2022.mp4", "rb")
   vi = like.read()
   st.video(vi, start_time=0)
   #st.video("https://www.youtube.com/watch?v=5cftfXdRCKU", )
st.text("")
st.text("")
st.text("")
st.text("")
st.text("")
st.text("")
st.text("")
st.text("")
st.text("")
st.text("")
st.text("")

###################################################İKİNCİ ADIM###########################################################
st.header("Geçmişten bu güne")

col11, col22, col33 = st.columns(3)

with col11:
   st.subheader("Takımını Seç")
   sorted_unique_Team = sorted(df.Team.unique())
   selected_Team = st.multiselect('Takımınız', sorted_unique_Team)
   df_Team = df[(df.Team.isin(selected_Team))]
   seçilen_takım = (df_Team.groupby(by=["Harita"]).mean()[["Round İstatistikleri"]])
   fig_chart = px.bar(seçilen_takım, x=seçilen_takım.index, y="Round İstatistikleri", orientation="v", title="Haritalara göre kazanma oranları",
                       color_discrete_sequence=["#AFAFAF"] * len(seçilen_takım), template="plotly_white")
   st.plotly_chart(fig_chart,use_container_width=True)
   seçilen_takım2 = (df_Team.groupby(by=["Harita"]).mean()[["Pistol_Round"]])
   fig_chart2 = px.bar(seçilen_takım2, x=seçilen_takım2.index, y="Pistol_Round", orientation="v", title="Haritalara göre pistol round kazanma oranları",
                        color_discrete_sequence=["#AFAFAF"] * len(seçilen_takım2), template="plotly_white")
   st.plotly_chart(fig_chart2,use_container_width=True)
   seçilen_takım3 = (df_Team.groupby(by=["Bölge"]).mean()[["Round İstatistikleri"]])
   fig_charttt = px.bar(seçilen_takım3, x=seçilen_takım3.index, y="Round İstatistikleri", orientation="v",
                        title="Bölgelere göre kazanma oranları",
                        color_discrete_sequence=["#AFAFAF"] * len(seçilen_takım3), template="plotly_white")
   st.plotly_chart(fig_charttt, use_container_width=True)
   #df_Team = df_Team.groupby(["Team", "ap"]).agg({"Pistol_Round":"mean" ,"fb":"mean"})
   #st.write(df_Team)

with col22:
   st.text("")
   st.text("")
   st.subheader("Neler yapıyoruz")
   st.warning("Takımınızın bu zamana kadar oynadıkları karşılaşmaları değerlendiriyoruz."
              " ArdındaN haritaları gözeterek elde edilen başarı istatistiklerini grafiğe döküyoruz. "
              "Aynı zaman rakip takım içinde bunu yaptığımızdan ötürü, güçlü yanlarınızı ve zayıf yönlerinizi fark etmenizi sağlıyoruz ")




   image = Image.open(r'C:\Users\USER\PycharmProjects\pythonProject\VALOProject1\valorantttttt.jpg')
   st.image(image, use_column_width=True)
   st.text("")
   st.text("")
   st.text("")
   st.text("")
   st.text("")
   st.text("")
   st.text("")
   st.text("")
   st.text("")
   st.success('Yaptığımız analizler sonucu şunu biliyoruz ki birinci ve onüçüncü roundaları (pistol round) kazanan takımlar başladıkları bölgeyi ortalama kazanma oranları %73. '
              'Bunun en büyük nedeni bahsedilen roundları kazanan takım çok büyük ekonomik gelir elde ediyor.'
              ' Bunun sonucunda daha iyi silahlar ve özel ajan yetenekleri satın alımları yapılarak gelecek roundalar için büyük avantaj sağlanmış oluyor.')
   st.text("")
   st.text("")
   st.text("")
   st.text("")
   st.text("")
   st.text("")
   st.text("")
   st.text("")
   st.text("")
   st.text("")
   st.text("")
   st.text("")
   st.text("")
   st.text("")
   st.text("")
   st.text("")
   st.text("")
   st.info('Tekrardan bahsi geçen takımların geçmişte oynadıkları maçları analiz ederek, takımın atakta mı yoksa defansta mı zayıf olduğunu ortaya çıkartıyoruz. '
           'Bu sayede antrenmanlarda ağırlık olarak hangi kısma odak vermelisiniz bu sonucu çıkartıyoruz')


with col33:
   st.subheader("Rakip Takımı Seç")
   sorted_unique_Team_e = sorted(df.Team.unique())
   selected_Team_e = st.multiselect('Rakip takım', sorted_unique_Team_e)
   df_Team_e = df[(df.Team.isin(selected_Team_e))]
   seçilen_takımm = (df_Team_e.groupby(by=["Harita"]).mean()[["Round İstatistikleri"]])
   fig_chartt = px.bar(seçilen_takımm,x=seçilen_takımm.index,y="Round İstatistikleri",orientation="v",title="Haritalara göre kazanma oranları",
                       color_discrete_sequence=["#AFAFAF"]*len(seçilen_takımm),template="plotly_white")
   st.plotly_chart(fig_chartt,use_container_width=True)
   seçilen_takımm2 = (df_Team_e.groupby(by=["Harita"]).mean()[["Pistol_Round"]])
   fig_charttt = px.bar(seçilen_takımm2, x=seçilen_takımm2.index, y="Pistol_Round", orientation="v", title="Haritalara göre pistol round kazanma oranları",
                       color_discrete_sequence=["#AFAFAF"] * len(seçilen_takımm2), template="plotly_white")
   st.plotly_chart(fig_charttt,use_container_width=True)
   seçilen_takımm3 = (df_Team_e.groupby(by=["Bölge"]).mean()[["Round İstatistikleri"]])
   fig_charttt = px.bar(seçilen_takımm3, x=seçilen_takımm3.index, y="Round İstatistikleri", orientation="v",
                        title="Bölgelere göre kazanma oranları",
                        color_discrete_sequence=["#AFAFAF"] * len(seçilen_takımm3), template="plotly_white")
   st.plotly_chart(fig_charttt, use_container_width=True)
   #df_Team_e_1 = df_Team_e.groupby(["Team","ap"]).agg({"Pistol_Round":"mean" ,"fb":"mean"})
   #st.write(df_Team_e_1)

st.text("")
st.text("")
st.text("")
st.text("")
st.text("")
st.text("")
##################################################ÜÇÜNCÜ ADIM###########################################################

st.header("Günümüzden Geleceğe")
col111, col222= st.columns(2)

with col111:
   dff = pd.read_excel(r'C:\Users\USER\PycharmProjects\pythonProject\VALOProject1\Sonuclar.xlsx')
   dff.dropna(inplace = True)
   dff["Tahminler"] = dff["Tahminler"].astype("int")
   Seçilen_takım = sorted(dff.Takımınız.unique())
   Seçilen_takımın_adları = st.multiselect('seçilen takım', Seçilen_takım)
   Seçilen_takımın_adları_e = dff[(dff.Takımınız.isin(Seçilen_takımın_adları))]

   #df_result33 = Seçilen_takımın_adları.groupby(["Takiminiz", "Rakip Takim"]).agg({"Tahminler": "count"})
   st.write(Seçilen_takımın_adları_e)


with col222:
    image = Image.open(
        r'C:\Users\USER\PycharmProjects\pythonProject\VALOProject1\valorant-champions-istanbul-bilet-fiyatlari-ve-duzenlenecegi-mekan-belli-oldu.jpg')
    st.image(image, use_column_width=True)


# endregion



