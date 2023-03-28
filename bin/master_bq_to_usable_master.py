# -*- coding: utf-8 -*-
"""
Created on Sun Dec 25 20:57:47 2022

@author: romer
"""

import pandas as pd
import numpy as np

df = pd.read_csv("./master_bq.csv")

##################### Transformar renta

renta = df.renta_bruta_media.values
q25,q75 = np.quantile(renta[~np.isnan(renta)], [.25, .75])

def auxiliar_renta(x, q25, q75, campo = 'renta'):
    if not pd.isnull(x):
        if x <= q25:
            return f'{campo}_baja'
        if x >= q75:
            return f'{campo}_alta'
        return f'{campo}_media'
    return x

df['renta_bruta_media'] = df['renta_bruta_media'].apply(lambda x: auxiliar_renta(x, q25, q75))

######################## Transformar tipo servicio
df['tipo_servicio'] = df['tipo_servicio'].apply(lambda x: x if '--' not in x else 'FIXandMOBILE')

######################## Quedarme con tarifas o combinaciones relevantes
relevantes = {group:len(lista) for group,lista in df.groupby('tarifa_id').groups.items() if len(lista) > 2000}
df['tarifa_id'] = df['tarifa_id'].apply(lambda x: x if x in relevantes else np.nan)

######################## Transformar coste de tarifa
tarifa = df.coste_tarifa.values
q25,q75 = np.quantile(tarifa[~np.isnan(tarifa)], [.25, .75])

df['coste_tarifa'] = df['coste_tarifa'].apply(lambda x: auxiliar_renta(x, q25, q75, campo = 'coste_tarifa'))

######################## Quedarme con toperadores de portabilidad o combinaciones relevantes
relevantes_port = {group:len(lista) for group,lista in df.groupby('operador_portabilidad').groups.items() if len(lista) > 2000}
df['operador_portabilidad'] = df['operador_portabilidad'].apply(lambda x: x if x in relevantes_port else np.nan)

###################### Transformar tipo de facturacion
df['tipo_facturacion'] = df['tipo_facturacion'].apply(lambda x: x if '--' not in x else 'PREandPOSTPAID')

######################## Transformar numero de ventas
num_ventas = df.num_ventas.values
q25,q75 = np.quantile(num_ventas[~np.isnan(num_ventas)], [.25, .75])

df['num_ventas'] = df['num_ventas'].apply(lambda x: auxiliar_renta(x, q25, q75, campo = 'numero_ventas'))

####################### Transformar alarma
df['alarma'] = df['alarma'].apply(lambda x: x if pd.isnull(x) else 'alarma')

###################### Tranformar energia
df['energia'] = df['energia'].apply(lambda x: x if pd.isnull(x) else 'energia')

#####################  Quedarme con servicios adicionales relevantes

df['servicios_adicionales'] = df['servicios_adicionales'].apply(lambda x: 'HEBECUSTSVA--DOCTORGOCUSTSVA' if (not pd.isnull(x) and 'HEBECUSTSVA' in x and 'DOCTORGOCUSTSVA' in x) else x)
relevantes1 = {group:len(lista) for group,lista in df.groupby('servicios_adicionales').groups.items() if len(lista) > 1000}
df['servicios_adicionales'] = df['servicios_adicionales'].apply(lambda x: x if x in relevantes1 or pd.isnull(x) else 'OTHER_SERVICE')


#################### Transformar financiacion

relevantes2 = {group:len(lista) for group,lista in df.groupby('meses_financiacion').groups.items() if len(lista) > 1000}
# df['financia'] = df['']
df['financia'] = df['cuota'].apply(lambda x: x if pd.isnull(x) else 'financia')
#TODO: no me sirve lo que tengo actualmente porque quiero ver si la financiacion es de un movil o de que tipo de aparato es.

df.to_csv('master_beta_faltafinanciacion.csv',index = False)
