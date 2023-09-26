import pandas as pd
from genetic_dealer import GeneticDealer
from collections import Counter
import wandb
import os
import re

file = {
    1: "../inputs/cooked_groceries.csv",
    2: "../inputs/master_beta_faltafinanciacion.csv",
}

fitness_function = {
    1: "lift_fitness",
    2: "lift_and_fisher_fitness",
    3: "composed_fitness",
    4: "composed_fitness_length_fitness",
    5: "interestingness_fitness",
    6: "conviction_fitness",
}


def prepping_master_beta_faltafinanciacion(file):
    directory = "../../inputs"
    pattern = r'master_beta_faltafinanciacion\d+\.csv'
    
    
    columns_to_use = ['provincia', 'edad', 'genero', 'renta_bruta_media', 'tipo_servicio', 'tipo_facturacion',
                      'tarifa_id', 'coste_tarifa', 'tipo_venta', 'operador_portabilidad', 'num_ventas', 'alarma',
                      'servicios_adicionales', 'financia', 'energia']
    
    filenames = [filename for filename in os.listdir(directory) if re.match(pattern, filename)]
    print(filenames)
    dfs = []  # List to store the individual dataframes
    
    for filename in filenames:
        file_path = os.path.join(directory, filename)
        df = pd.read_csv(file_path)
        dfs.append(df)
    
    transaction_df = pd.concat(dfs, ignore_index=True)
    transaction_df.drop_duplicates(keep='first', subset=["customer_id"], inplace=True)
    transaction_df.reset_index(inplace=True)
    transaction_df = transaction_df[columns_to_use]
    return transaction_df


if __name__ == "__main__":
    ### ------------------ Tuneable
    file_key = 2
    fitness_function_key = 4
    ### ------------------ HIPERPARÁMETROS
    consequent = "financia"
    consequent_value = "financia"
    fitness_function = fitness_function[fitness_function_key]
    generation_size = 1000
    num_generations = 500
    seed = 43
    probability_of_zero = .5  # filekey = 1: .95 , filekey = 2: .6
    mutation_probability = .004
    wandb_project = file[file_key].split("/")[-1]
    wandb_project = "financia_final"

    ### ------------------ Lógica del programa y logging
    if file_key == 2:
        transaction_df = prepping_master_beta_faltafinanciacion(file[file_key])
    else:
        transaction_df = pd.read_csv(file[file_key])

    config = {
        "consequent_value": consequent_value,
        "fitness_function": fitness_function,
        "generation_size": generation_size,
        "num_generations": num_generations,
        "seed": seed,
        "probability_of_zero": probability_of_zero,
        "mutation_probability": mutation_probability,
    }
    
    wandb.init(
        # set the wandb project where this run will be logged
        project=wandb_project,
        # track hyper-parameters and run metadata
        config=config
    )

    d = GeneticDealer(transaction_df, consequent=consequent, consequent_value=consequent_value, seed=seed,
                      probability_of_zero=probability_of_zero)
    d.run(generation_size=generation_size, num_generations=num_generations, fitness_function=fitness_function,
          mutation_probability=mutation_probability)
    wandb.log({"Best fenotype table": d.wandb_table})
    hyperparameter_table = wandb.Table(data=[list(config.values())], columns=list(config.keys()))
    wandb.log({"Hyperparameters": hyperparameter_table})
    #log table with the unique fenotypes of d.generations and the number of times they appear
    tuple_list = [tuple(lst) for lst in d.generation]
    counter = Counter(tuple_list)
    freq_dict = {}
    for k,v in counter.items():
        freq_dict[k] = v
    convergence_table = [[k,d.decode_fenotype(k), v] for k,v in freq_dict.items()]
    convergence_table_wandb = wandb.Table(data=convergence_table, columns=["Fenotype","Decoded fenotype", "Count"])
    wandb.log({"Convergence_table": convergence_table_wandb})
    wandb.finish()

    # filename = f"{consequent_value}_{fitness_function}_gs{generation_size}_g{num_generations}_seed{seed}.csv"
    # df.to_csv("./SAVED RESULTS/"+filename,index = False)
