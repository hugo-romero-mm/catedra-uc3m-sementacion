# -*- coding: utf-8 -*-
"""
Created on Tue Oct 18 11:11:39 2022

@author: romer
"""
import pandas as pd

def getStatFromCP(cp: str):
    """
    Given a postal code returns the renta bruta media, or False if it doesnt
    find the postal code corrrespondence
    """
    df = pd.read_csv("./DataSet/renta_por_cp2020.csv", dtype={'CP':'object'})
    fila = df[df.CP == cp]
    if not fila.empty:
        return fila
    print(f"Dato no encontrado: {cp}")
    return False

def getStatsFromCPs(zips_: list, mode: str = "listas") -> list:
    """

    Parameters
    ----------
    zips_ : list
        DESCRIPTION.
    mode : str, optional
        DESCRIPTION. The default is "listas".

    Returns
    -------
    list
        DESCRIPTION.
        Given a list of postal codes, returns a list of renta bruta media of the same
        length and in the same order. If the mode is "listas" returns a list
        of lists, otherwise it returns a list of numpy arrays.

    """
    df = pd.read_csv("./DataSet/renta_por_cp2020.csv", dtype={'CP':'object'})
    res = []
    found = 0
    notfound = 0
    for i,zip_ in enumerate(zips_):
        if i%10000 == 0:
            print(i, " / ", len(zips_))
        fila = df[df.CP == zip_]
        if not fila.empty:
            found += 1
            if mode == "listas":
                res.append(list(fila.values[0]))
            else:
                res.append(fila)
        else:
            if mode == "listas":
                res.append([])
            else:
                res.append(fila)
            notfound += 1
        
    print(f'Found: {found}/{len(zips_)}')
    print(f'Notfound: {notfound}/{len(zips_)}')
    return res

def getStatsFromCPsBoosted(zips_: list) -> list:
    df = pd.read_csv("./DataSet/renta_por_cp2020.csv", dtype={'CP':'object'})
    my_dict = dict(zip(df.values[:,0],df.values[:,3]))
    res = []
    found = 0
    notfound = 0
    for i,zip_ in enumerate(zips_):
        if i%100000 == 0:
            print(i, " / ", len(zips_))
        if zip_ in my_dict:
            found += 1
            res.append(my_dict[zip_])
        else:
            res.append(0)
            notfound += 1 
    print(f'Found: {found}/{len(zips_)} ({round(100*found/len(zips_),1)} %)')
    print(f'Notfound: {notfound}/{len(zips_)} ({round(100*notfound/len(zips_),1)} %)')
    return res

def uniquevals(a: pd.DataFrame()) -> dict:
    """

    Parameters
    ----------
    a : pd.DataFrame()
        DESCRIPTION.

    Returns
    -------
    dict
        DESCRIPTION.
        Given a dataframe, print a brief description of the columns in it
        The description consists of how many different vals does the column
        have when there are more than 20. In the case where there are less
        than 20 it just prints the values.
    """
    print(f'Info dataset de {len(a)} filas, {len(a.columns)} columnas:')
    uniques = {}
    for col in a:
        u = a[col].unique()
        if len(u) <= 20:
            print(f'\t{col} valores {u}')
        else:
            print(f'\t{col} {len(u)} valores distintos')
        uniques[col] = u
    return uniques

def joincsv(prefix: str, num_files: int) -> pd.DataFrame():
    """
    Given the name of the file without numbers and the number of files,
    returns a dataframe of the files concated. The format it follows is 
    the one google cloud storage (bucket) gives to files when exporting
    multiple files from BigQuery ('example_name000000000001.csv')
    """
    
    lista = []
    for i in range(num_files):
        number = f'{i:012d}'
        print(number)
        name = f'{prefix}{number}.csv'
        lista.append(pd.read_csv(name, dtype={"zip_code":"object"}))
    concated = pd.concat(lista, ignore_index=True)
    return concated
    