# -*- coding: utf-8 -*-
"""
Created on Mon Oct 24 10:40:33 2022

@author: romer
"""

path = "./DataSet/renta_bruta_media_sin_limpiar.txt"
import pandas as pd
import numpy as np

with open(path, encoding='utf-8') as file:
    lines = file.readlines()

a = {"CP":[], "Region":[], "Zonas": [], "Renta bruta media": [], "Renta disponible media": []}
i = 0
while(i) < len(lines):
    if "Contrae" in lines[i]:
        
        region = lines[i].split("-")[0].replace("Contrae la tabla ", "").strip()
        # print(f"Contrae detectado {region}", lines[i])
        i+=1
    elif "Resto" in lines[i]:
        i+= 2
    else:
        # Lugar
        cp = lines[i].split("-")[0]
        zonas = "-".join(list(map(lambda x: x.strip(), lines[i].split("-")[1:])))
        a["CP"].append(cp)
        a["Region"].append(region)
        a["Zonas"].append(zonas)
        i+=1
        # Datos
        datos = lines[i].strip().split("\t")[1:3]
        datos = list(map(lambda x: int(x.replace(".", "")), datos))
        a["Renta bruta media"].append(datos[0])
        a["Renta disponible media"].append(datos[1])
        i+= 1


df = pd.DataFrame(a)
df.to_csv("./DataSet/renta_por_cp2020(1).csv", index = False)

        
    
    