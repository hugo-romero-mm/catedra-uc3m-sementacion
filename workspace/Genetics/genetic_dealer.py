# -*- coding: utf-8 -*-
"""
@author: romer
"""
import pandas as pd
import random
import time
import numpy as np
from scipy import stats
from fitness_functions.fitness_functions import composed_fitness,lift_fitness, lift_and_fisher_fitness, composed_fitness_length_fitness
import wandb
import os

def reorder_cols(df: pd.DataFrame, consequent: str):
    columns = list(df.columns)
    columns.remove(consequent)
    columns.append(consequent)
    df = df[columns]
    return df


class GeneticDealer:
    def __init__(self, df, consequent, consequent_value, seed = 42, verbose = 1, probability_of_zero = .4):
        self.verbose = verbose
        if self.verbose: print("Creating dealer...")
        self.consequent = consequent
        self.consequent_value = consequent_value
        self.probability_of_zero = probability_of_zero
        self.df = reorder_cols(df, consequent)
        self.T = self.df.index.size
        
        
        random.seed(seed)
        
        self.columns = list(self.df.columns)
        self.consequent_support = df[df[consequent] == consequent_value].index.size/df.index.size
        self.bounds = []
        
        self.index_column_dict = {}
        for i,col in enumerate(self.columns):
            self.index_column_dict[i] = col
            self.index_column_dict[col] = i
            uniques = list(filter(lambda v: v==v, self.df[col].unique()))
            if col != consequent: self.bounds.append(len(uniques)+1)
            for j,value in enumerate(sorted(uniques)):
                self.index_column_dict[(i, j+1)] = value
                
        
        self.binaries = self.generate_binaries()
        self.generation = None
        self.stats_cache = {}
        self.previous_generations = []
        self.history_stats_df = pd.DataFrame(columns=["Size", "Mean_fitness", "Best_fenotype", "Decoded","Fitness", "Lift", "Fisher"])
        
    def generate_binaries(self):
        '''Funcion auxiliar para el constructor'''
        grouped_indexes = {}
        for col in self.columns:
            if self.verbose: print('Generating binaries for', col)
            grouped_indexes[col] = {}
            for group,lista in self.df.groupby(col).groups.items():
                aux = [0]*self.T
                for presente in lista:
                    aux[presente] = 1
                string = ''.join(map(str, aux))
                binary = int(string, 2)
                grouped_indexes[col][group] = binary
        return grouped_indexes
    
    def run(self, num_generations = 5, generation_size = 200, fitness_function = "lift", mutation_probability = .01):
        if not self.generation: self.generation = [self.generate_fenotype() for _ in range(generation_size)]
        for i in range(num_generations):
            if self.verbose: print(f'Generation {i}/{num_generations-1}')

            stats = self.stats_of_generation(self.generation, fitness_function)
    
            self.previous_generations.append(self.generation.copy())
            self.generation = self.reproduction(self.generation, stats, fitness_function = fitness_function)
            self.mutations(self.generation, mutation_probability)
      
    def reproduction(self, generation, stats, T = 2, fitness_function = "composed_fitness"):
        next_generation = []
        start = time.time()
        while(len(next_generation) < len(generation)):
            if self.verbose and len(next_generation)%100 == 0:
                print(f'\t{len(next_generation)}/{len(generation)}  {int(time.time() - start)} segundos')
            parent1_index = self.tournament_parent(generation, stats, T = T)
            parent2_index = self.tournament_parent(generation, stats, T = T)
            parent1 = generation[parent1_index]
            parent2 = generation[parent2_index]
            rand_pos = random.randint(1, len(parent1)-1)
            son1 = parent1[:rand_pos] + parent2[rand_pos:]
            son2 = parent2[:rand_pos] + parent1[rand_pos:]
            if not sum(son1):
                son1 = parent1
            if not sum(son2):
                son2 = parent2
            
            pool = [parent1, parent2, son1, son2]
            pool = [p for p in pool if sum(p) != 0]
            
            pool_val = {tuple(son1): self.stats(son1, fitness_function)['fitness'], tuple(son2): self.stats(son2, fitness_function)['fitness']}
            pool_val[tuple(parent1)] = stats[parent1_index]["fitness"]
            pool_val[tuple(parent2)] = stats[parent2_index]["fitness"]
            pool = [p for p in pool if sum(p) != 0]
            pool.sort(key = lambda x: pool_val[tuple(x)], reverse = True)           
            next_generation += pool[:2] 
        return next_generation
    
    def mutations(self, generation, mutation_probability = .001):
        for i in range(len(generation)):
            if random.random() < mutation_probability:
                for mutated_position in range(len(self.bounds)):
                # mutated_position = random.randint(0, len(self.bounds) - 1)
                    pre = generation[i][mutated_position]
                    generation[i][mutated_position] = random.randint(0, self.bounds[mutated_position]-1)
                    # if pre != generation[i][mutated_position]:
                        # print("Mutated")
                        # time.sleep(5)
                    if sum(generation[i]) == 0:
                        generation[i][mutated_position] = random.randint(1, self.bounds[mutated_position]-1)
    
    def tournament_parent(self, generation, stats, T):
        selected = [random.randint(0, len(generation)-1) for _ in range(T)]
        tuples = [(stats[s]["fitness"],s) for s in selected]
        return max(tuples)[1]
      
    
    def generate_fenotype(self):
        fen_length = len(self.bounds)
        fenotype = [0]*fen_length
        while not sum(fenotype):
            for i in range(fen_length):
                if random.random() > self.probability_of_zero:
                    fenotype[i] = random.randint(1,self.bounds[i]-1)
        return fenotype
    
    def decode_fenotype(self, fenotype):
        result = []
        for i,gen in enumerate(fenotype):
            if gen:
                result.append(self.index_column_dict[(i,gen)])
        return "{" + ",".join(result) + "}"
            
        
        
    def stats_of_generation(self, generation, fitness_function):
        start_time = time.time()
        stats = [self.stats(fenotype, fitness_function) for fenotype in generation]
        if self.verbose: print(f'Calculating stats took: {time.time()-start_time} seconds')
        fitnesses = np.array([stat['fitness'] for stat in stats])
        lifts = np.array([stat['lift'] for stat in stats])
        fishers = np.array([stat['fisher'] for stat in stats if 'fisher' in stat])
        convictions = np.array([stat['conviction'] for stat in stats])
        interestingness = np.array([stat['interestingness'] for stat in stats])
        bfi = fitnesses.argmax()
        best_fenotype = generation[bfi]
        gen_stats = [len(generation), fitnesses.mean(), str(best_fenotype) ,self.decode_fenotype(best_fenotype), fitnesses[bfi], lifts[bfi], fishers[bfi]]
        
        wandb.log({"fitness mean": fitnesses.mean(), 
                   "lift mean": lifts.mean(),
                   "fisher mean": fishers.mean(),
                   "fitness max": fitnesses.max(), 
                   "lift min": lifts.min(),
                   "lift max": lifts.max(),
                    "fisher min": fishers.min(),
                    "convinction min": convictions.min(),
                    "conviction max": convictions.max(),
                    "interestingness min": interestingness.min(),
                    "interestingness max": interestingness.max(),
                    "conviction mean": convictions.mean(),
                    "interestingness mean": interestingness.mean(),
                    
                   "unique fenotypes": len(set([tuple(lst) for lst in generation])),
        })
        decoded_best_fenotype = self.decode_fenotype(best_fenotype)
        if not hasattr(self, "wandb_table"):
            self.wandb_table = wandb.Table(columns = ["Generation", "Best fenotype", "Fitness", "Fisher", "Lift"])

        if not self.wandb_table.get_column("Best fenotype") \
           or self.wandb_table.get_column("Best fenotype")[-1] != decoded_best_fenotype:
               self.wandb_table.add_data(len(self.previous_generations), decoded_best_fenotype, fitnesses[bfi], fishers[bfi], lifts[bfi])
        self.history_stats_df.loc[len(self.history_stats_df.index)] = gen_stats
        return stats
    
    def stats(self, fenotype, fitness_function):
        key = tuple(fenotype)
        if key in self.stats_cache: return self.stats_cache[key]
        precedent_bin = None
        for i,gen in enumerate(fenotype):
            if gen:
                col_name = self.index_column_dict[i]
                value = self.index_column_dict[(i, gen)]
                new_bin = self.binaries[col_name][value]
                if precedent_bin == None:
                    precedent_bin = new_bin
                elif precedent_bin:
                    precedent_bin &= new_bin
        all_bin = int("1"*self.T, 2)
        consequent_bin = self.binaries[self.consequent][self.consequent_value]
        non_precedent_bin = all_bin ^ precedent_bin

        _ab = bin(non_precedent_bin & consequent_bin).count('1')
        a = bin(precedent_bin).count('1')
        b = bin(consequent_bin).count('1')
        ab = bin(precedent_bin & consequent_bin).count('1')
        a_b = a - ab
        _a_b = bin(non_precedent_bin).count('1') - _ab

        fisher = stats.fisher_exact(
            table=[[ab, _ab], [a_b, _a_b]], alternative="less")[1]
        try:
            confidence = ab/a
        except:
            confidence = 0
        lift = confidence/self.consequent_support
        precedent_support = a/self.T
        conviction = (1- self.consequent_support)/(1-confidence) if confidence != 1 else 10_000
        interestingness = (ab**2*self.T-ab**3)/(a*b*self.T) if a*b != 0 else 0

        if fitness_function == "composed_fitness":
            fitness = composed_fitness(lift, fisher, a)
        elif fitness_function == "lift_fitness":
            fitness = lift_fitness(lift, precedent_support, a, fenotype)
        elif fitness_function == "lift_and_fisher_fitness":
            fitness = lift_and_fisher_fitness(lift, fisher,precedent_support, a)
            fitness = lift_fitness(lift, precedent_support, a, fenotype)
        elif fitness_function == "composed_fitness_length_fitness":
            fitness = composed_fitness_length_fitness(lift, fisher,precedent_support, a, fenotype)
        elif fitness_function == "conviction_fitness":
            fitness = conviction
        elif fitness_function == "interestingness_fitness":
            fitness = interestingness
            
            
        value_dict = {"fitness": fitness,
                      "lift": lift,
                      "precedent_support": precedent_support,
                      "fisher": fisher,
                      "conviction": conviction,
                      "interestingness": interestingness
                      }
        self.stats_cache[key] = value_dict
        return value_dict

    def describe_generation(self, fitness_function):
        tuples = [tuple(f) for f in self.generation]
        count = {fenotype_tuple:tuples.count(fenotype_tuple) for fenotype_tuple in tuples}
        
        generation = [list(fenotype_tuple) for fenotype_tuple in count]
        stats = [self.stats(fenotype, fitness_function) for fenotype in generation]
        fitnesses = np.array([stat['fitness'] for stat in stats])
        lifts = np.array([stat['lift'] for stat in stats])
        fishers = np.array([stat['fisher'] for stat in stats if 'fisher' in stat])
        convictions = np.array([stat['conviction'] for stat in stats])
        interestingness = np.array([stat['interestingness'] for stat in stats])
        lifts = np.array([stat['lift'] for stat in stats])
        precedent_supports = np.array([stat['precedent_support'] for stat in stats if 'fisher' in stat])
        bfi = fitnesses.argmax()
        best_fenotype = generation[bfi]
        df = pd.DataFrame(columns=["Size", "Mean_fitness", "Best_fenotype","Decoded", "Fitness", "Lift", "Fisher"])
        gen_stats = [len(generation), fitnesses.mean(), str(best_fenotype) ,self.decode_fenotype(best_fenotype), fitnesses[bfi], lifts[bfi], fishers[bfi]]
        df.loc[len(df.index)] = gen_stats
        print(df)
        
        df = pd.DataFrame()
        df["Fenotype"] = generation
        df["Appearances"] = [count[tuple(fenotype)] for fenotype in generation]
        df["Decoded"] = [self.decode_fenotype(fenotype) for fenotype in generation]
        df["Fitness"] = fitnesses
        df["Lift"] = lifts
        df["Fisher"] = fishers
        df["Precedent_support"] = precedent_supports
        df["Precedent_count"] = [round(ps*self.T) for ps in precedent_supports]
        df["Conviction"] = convictions
        df["Interestingness"] = interestingness
        df.sort_values(by=["Fitness"], inplace = True, ascending = False)
        print(df)
        return df
        
        
        
        
