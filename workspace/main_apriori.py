import pandas as pd
from genetic_dealer import GeneticDealer
import wandb
from apyori_rome import apriori
import os

file = {
    1: "../inputs/cooked_groceries.csv",
    2: "../inputs/master_beta_faltafinanciacion.csv",
}

def prepping_master_beta_faltafinanciacion(file):
    columns_to_use = ['provincia', 'edad', 'genero', 'renta_bruta_media', 'tipo_servicio', 'tipo_facturacion',
                      'tarifa_id', 'coste_tarifa', 'tipo_venta', 'operador_portabilidad', 'num_ventas', 'alarma',
                      'servicios_adicionales', 'financia', 'energia']
    transaction_df = pd.read_csv(file)
    transaction_df.drop_duplicates(keep='first', subset=["customer_id"], inplace=True)
    transaction_df.reset_index(inplace=True)
    transaction_df = transaction_df[columns_to_use]
    return transaction_df


if __name__ == "__main__":
    ### ------------------ Tuneable
    file_key = 1
    ### ------------------ HIPERPARÁMETROS
    consequent = "beef"
    consequent_value = "beef"
    seed = 43

    ### ------------------ Lógica del programa y logging
    if file_key == 2:
        transaction_df = prepping_master_beta_faltafinanciacion(file[file_key])
    else:
        transaction_df = pd.read_csv(file[file_key])

    config = {
        "seed": 43,
        "min_support": .01,
        "min_confidence": .02,
        "min_lift": 1.25,
    }
    # wandb.init(
    #     # set the wandb project where this run will be logged
    #     project="apriori" + file[file_key].split("/")[-1],
    #     # track hyper-parameters and run metadata
    #     config=config
    # )
    transactions = transaction_df.sample(frac = .02)
    dealer = apriori(transactions, min_support=config["min_support"], min_confidence=config["min_confidence"], min_lift=config["min_lift"], verbose=True, createDF=True)
    df = dealer.good_rules_df

    #wandb.log({"Best fenotype table": d.wandb_table})
    hyperparameter_table = wandb.Table(data=[list(config.values())], columns=list(config.keys()))
    # wandb.log({"Hyperparameters": hyperparameter_table})
    # wandb.finish()

    # filename = f"{consequent_value}_{fitness_function}_gs{generation_size}_g{num_generations}_seed{seed}.csv"
    # df.to_csv("./SAVED RESULTS/"+filename,index = False)


    # print(results)
    # interesting = []
    # for res in r_master:
    #     for stat in res.ordered_statistics:
    #         if "yes_energia" in stat.items_add:
    #             interesting.append(res)
    # print("Interesantes")
    # print(interesting)