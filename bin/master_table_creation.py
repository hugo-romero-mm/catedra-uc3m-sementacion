# -*- coding: utf-8 -*-
"""
Created on Thu Nov  3 10:30:04 2022

@author: romer
"""

import pandas as pd
from tools import getStatsFromCPs, getStatsFromCPsBoosted

df_customers = pd.read_csv("./DataSet/CustomerRentaOnlyAlarmas.csv", dtype={"CP":"object"})
df_customers.drop_duplicates(keep = 'first', subset=["customer_id"], inplace=True)
customer_relation = pd.read_csv("./DataSet/energygo_customer_relation.csv")
energygo_services = pd.read_csv("./DataSet/energygo_services.csv")


customers = df_customers.customer_id.values
customer_dict = {c:0 for c in customers}

## Create dict
relation_dict = {}
for _,customer_id, start,end,energy_id in customer_relation.values:
    if customer_id not in relation_dict:
        relation_dict[customer_id] = [[start,end,energy_id]]
    else:
        relation_dict[customer_id].append([start,end, energy_id])

for i,customer in enumerate(customer_dict):
    if i%10000 == 0:
        print(f"Evaluando customer {i} de {len(customer_dict)}")
    if customer in relation_dict:
        relations = relation_dict[customer]
        for st,end,energy_id in relations:
            activations_in_energy = list(map(lambda x: int(''.join(x.split("-"))),energygo_services[energygo_services.id == energy_id]["activation_date"].values))
            for activ in activations_in_energy:
                if st<activ<end:
                    customer_dict[customer] += 1

df_customers["energy"] = customer_dict.values()
            
            
    
    
    
    
    
    



# yoigo_services_alarmas = pd.read_csv("./DataSet/yoigo_services_alarmas.csv")
# yoigo_services_energia = pd.read_csv("./DataSet/yoigo_services_energia.csv")