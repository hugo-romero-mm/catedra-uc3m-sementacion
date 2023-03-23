# -*- coding: utf-8 -*-
"""
Created on Wed Oct 19 10:22:44 2022

@author: romer
"""

from apyori_rome import apriori
import pandas as pd        

# master = pd.read_csv("./DataSet/master_beta.csv")[["provincia", "edad", "genero", "renta_bruta_media", "alarma", "energy"]]
# master["alarma"] = master["alarma"].apply(lambda x: "yes_alarma" if x=="INSTALADA" else "no_alarma")
# cuantil75 = master["renta_bruta_media"].quantile(.75)
# cuantil25 = master["renta_bruta_media"].quantile(.25)
# master["renta_bruta_media"] = master["renta_bruta_media"].apply(lambda x: "renta_alta" if x>cuantil75 else x)
# master["renta_bruta_media"] = master["renta_bruta_media"].apply(lambda x: "renta_media" if not isinstance(x, str) and x>cuantil25 else x)
# master["renta_bruta_media"] = master["renta_bruta_media"].apply(lambda x: "renta_baja" if not isinstance(x, str) else x)                                               

# master["energy"] = master["energy"].apply(lambda x: "yes_energia" if x > 0 else "no_energia")

#Discretization
# master = master[[]]
# '''  ENERGIA  '''
# transactions = master[master.energy == "yes_energia"].drop(columns=["energy"]).values
# denergy = apriori(transactions, min_support=.2,min_confidence=.8,min_lift = .6)

# '''  MASTER  '''
# transactions = master.sample(n=50_000).values
# dmaster = apriori(transactions, min_support=.1,min_confidence=.8,min_lift = 1)

# '''  MASTER  consequent energy'''
# transactions = master.values
# dmastergr = apriori(transactions, min_support=.001,min_confidence=.002,min_lift = 1.05, verbose = True, createDF = True)

# df = dmastergr.good_rules_df

# print(results)
# interesting = []
# for res in r_master:
#     for stat in res.ordered_statistics:
#         if "yes_energia" in stat.items_add:
#             interesting.append(res)
# print("Interesantes")            
# print(interesting)