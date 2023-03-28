# -*- coding: utf-8 -*-
"""
Genetic search

@author: romer
"""

import pandas as pd
import numpy as np
import random
from scipy import stats
import time
import sys

class Dealer:
    def __init__(self, df, min_lift, min_precedent_support, consequent, consequent_value, seed):
        self.verbose = 1
        random.seed(seed)
        if self.verbose: print("Creating dealer...")
        reorder_cols = list(df.columns)
        reorder_cols.append(reorder_cols.pop(reorder_cols.index(consequent)))
        self.df = df[reorder_cols]
        self.consequent = consequent
        self.consequent_value = consequent_value
        
        self.consequent_support = len(df[df[consequent] == consequent_value].index)/len(df.index)
        self.boundaries, self.dict_num_col, self.dicts_num_val, self.dicts_val_num = self.generate_dicts()
        self.grouped_indexes = self.generate_groups()
        self.genotype_order = [col for col in self.df.columns if col != self.consequent]
        if self.verbose: print(f'Tamaño del espacio de busqueda: {np.array([b+1 for b in self.boundaries]).prod()} precedentes')
        self.generations_history = []
        self.seen_fenotypes = {}
        #self.run(gen_size = 200, num_generations = num_generations)
        
        
    def run(self, gen_size = 200, num_generations = 5, fitness_function = "composed"):
        print('run')
        generation = [self.generate_random_genotype() for _ in range(gen_size)]
        for i in range(num_generations):
            start = time.time()
            stats = [self.fitness(genotype, fitness_function) for genotype in generation]
            print(f'Calculating stats took: {time.time()-start} seconds')
            fitnesses = np.array([stat['fitness'] for stat in stats])
            lifts = np.array([stat['lift'] for stat in stats])
            supports = np.array([stat['precedent_support'] for stat in stats])
            fishers = np.array([stat['fisher'] for stat in stats if 'fisher' in stat])
            self.generations_history.append(generation.copy())
            generation_resume = f'Generation {i} (size = {len(generation)}):\nFitness: max={fitnesses.max()}\
                mean={fitnesses.mean()} min={fitnesses.min()}.\nBest fenotype: {generation[fitnesses.argmax()]}\
                {self.decode_fenotype(generation[fitnesses.argmax()])} lift: {lifts[fitnesses.argmax()]}\
                fisher: {fishers[fitnesses.argmax()]}\n'
            print(generation_resume)
            if i == num_generations: break
            print(f'\tentering reproduction phase of generation {i}')
            generation = self.reproduction(generation, fitnesses,lifts, supports, fitness_function = fitness_function)
            print(f'\tentering mutation phase of generation {i}')
            generation = self.mutations(generation)
            
        self.generation = generation
    
    def run_1(self):
        generation = self.generation
        start = time.time()
        stats = [self.fitness(genotype) for genotype in generation]
        print(f'Calculating stats took: {time.time()-start} seconds')
        fitnesses = np.array([stat['fitness'] for stat in stats])
        lifts = np.array([stat['lift'] for stat in stats])
        supports = np.array([stat['precedent_support'] for stat in stats])
        fishers = np.array([stat['fisher'] for stat in stats if 'fisher' in stat])
        i = len(self.generations_history)
        generation_resume = f'Generation {i} (size = {len(generation)}):\nFitness: max={fitnesses.max()}\
            mean={fitnesses.mean()} min={fitnesses.min()}.\nBest fenotype: {generation[fitnesses.argmax()]}\
            {self.decode_fenotype(generation[fitnesses.argmax()])} lift: {lifts[fitnesses.argmax()]}\
            fisher: {fishers[fitnesses.argmax()]}\n'
        print(generation_resume)
        print(f'\tentering reproduction phase of generation {i}')
        generation = self.reproduction(generation, fitnesses,lifts, supports)
        print(f'\tentering mutation phase of generation {i}')
        generation = self.mutations(generation)
        self.generations_history.append(generation.copy())
            
        self.generation = generation
        
            
        
    def tournament_parent(self, generation, fitnesses ,lifts, supports, T = 2):
        selected = [random.randint(0, len(generation)-1) for _ in range(T)]
        winner = generation[selected[0]]
        best_fitness = fitnesses[selected[0]]
        for sel in selected[1:]:
            if fitnesses[sel]>best_fitness:
                winner = generation[sel]
                best_fitness = lifts[sel]
        return winner
    
    def reproduction(self, generation, fitnesses, lifts, supports, T = 2, fitness_function = "composed"):
        next_generation = []
        start = time.time()
        while(len(next_generation) < len(generation)):
            if len(next_generation)%50 == 0:
                print(f'\t{len(next_generation)}/{len(generation)}  {int(time.time() - start)} segundos')
            parent1 = self.tournament_parent(generation, fitnesses, lifts, supports, T = T)
            parent2 = self.tournament_parent(generation, fitnesses, lifts, supports, T = T)
            gen_pos = random.randint(1, len(parent1)-1)
            son1 = parent1[:gen_pos] + parent2[gen_pos:]
            son2 = parent2[:gen_pos] + parent1[gen_pos:]
            pool = [parent1, parent2, son1, son2]
            pool_val = {tuple(son1): self.fitness(son1, fitness_function)['fitness'], tuple(son2): self.fitness(son2, fitness_function)['fitness']}
            pool_val[tuple(parent1)] = fitnesses[generation.index(parent1)]
            pool_val[tuple(parent2)] = fitnesses[generation.index(parent2)]
            pool = [p for p in pool if sum(p) != 0]
            pool.sort(key = lambda x: pool_val[tuple(x)], reverse = True)           
            next_generation += pool[:2] 
        return next_generation
    
    def mutations(self, generation, mutation_probability = .2):
        next_generation = generation.copy()
        for i in range(len(generation)):
            if random.random() < mutation_probability:
                mutated_position = random.randint(0, len(self.boundaries) - 1)
                next_generation[i][mutated_position] = random.randint(0, self.boundaries[mutated_position])
                if sum(next_generation[i]) == 0:
                    next_generation[i][mutated_position] = random.randint(1, self.boundaries[mutated_position])
        return next_generation
        
    
    def generate_groups(self):
        '''Funcion auxiliar para el constructor'''
        grouped_indexes = {}
        for col in self.df.columns:
            if self.verbose: print('Generating binaries for', col)
            
            # grouped_indexes[col] = {group:set(lista) for group,lista in self.df.groupby(col).groups.items()}
            grouped_indexes[col] = {}
            for group,lista in self.df.groupby(col).groups.items():
                aux = [0]*len(self.df.index)
                for presente in lista:
                    aux[presente] = 1
                string = ''.join(map(str, aux))
                binary = int(string, 2)
                grouped_indexes[col][group] = binary
        return grouped_indexes
        
        
    def generate_dicts(self):
        '''Funcion auxiliar para el constructor'''
        dict_num_col = {}
        dicts_num_val = {}
        dicts_val_num = {}
        boundaries = []
        for index,col in enumerate(self.df.columns):
            dict_num_col[index] = col
            
            if col != self.consequent:
                
                boundaries.append(len(self.df[self.df[col].notna()][col].unique()))
                dict_num_val = {i+1:val for i,val in enumerate(self.df[self.df[col].notna()][col].unique())}
                dict_val_num = {val:i+1 for i,val in enumerate(self.df[self.df[col].notna()][col].unique())}
                dicts_num_val[col] = dict_num_val
                dicts_val_num[col] = dict_val_num
        return boundaries, dict_num_col, dicts_num_val, dicts_val_num
    
    def decode_fenotype(self, fenotype):
        result = []
        for i,f in enumerate(fenotype):
            if f != 0:
                result.append(self.dicts_num_val[self.genotype_order[i]][f])
        return result
      
        
    def generate_random_genotype(self, probability_of_zero = .4):
        genotype = [0]*len(self.boundaries)
        while(sum(genotype) == 0):
            genotype = [0]*len(self.boundaries)
            for i in range(len(self.boundaries)):
                if random.random() > probability_of_zero:
                    genotype[i] = random.randint(1,self.boundaries[i])
        return genotype
    
    def fitness(self, fenotype, fitness_function = "composed1") -> dict:
        if fitness_function == "composed":
            return self.fitness_lift_if_fisher(fenotype)
        if fitness_function == "lift":
            return self.fitness_lift(fenotype)
    
    def fitness_lift_times_support(self, fenotype):
        '''funcion de fitness que multiplica el lift de la regla por el soporte del precedente'''
        _, lift, precedent_support = self.fitness_lift(fenotype)
        return lift*precedent_support, lift, precedent_support
    
    def fitness_lift_if_fisher(self, fenotype) -> dict:
        key = tuple(fenotype)
        if key in self.seen_fenotypes:
            return self.seen_fenotypes[key]
        
        precedent_set = None
        for i in range(len(fenotype)):
            if fenotype[i] != 0:
                col_name = self.dict_num_col[i]
                val = self.dicts_num_val[col_name][fenotype[i]]
                if not precedent_set:
                    precedent_set = self.grouped_indexes[col_name][val]
                else:
                    precedent_set = precedent_set & self.grouped_indexes[col_name][val]
                    
        # print("202", precedent_set)
                   
        # _a_set = set(self.df.index) - precedent_set
        tot = int("1"*len(self.df.index), 2)
        b_set = self.grouped_indexes[self.consequent][self.consequent_value]
        _a_set = tot ^ precedent_set
        # _b_set = tot ^ b_set
        # _ab_set = _a_set.intersection(self.grouped_indexes[self.consequent][self.consequent_value])
        _ab_set = _a_set & b_set
        ab_set = precedent_set & b_set
        a = bin(precedent_set).count('1')
        # rule_set = precedent_set.intersection(self.grouped_indexes[self.consequent][self.consequent_value])
        # Defining contingency table values
        # ab = len(rule_set)
        ab = bin(ab_set).count('1')
        a_b = a - ab
        _ab = bin(_ab_set).count('1')
        _a_b = bin(_a_set).count('1') - _ab
        fisher = stats.fisher_exact(table=[[ab,_ab],[a_b,_a_b]], alternative="less")[1]
        try:
            confidence = ab/a
        except:
            confidence = 0
        lift = confidence/self.consequent_support
        precedent_support = a/len(self.df.index)
        personas = a
        if fisher < .05 and personas >= 25:
            if lift > 1:
                fitness = lift + (lift-1)
            else:
                if lift != 0:
                    fitness = 1 / lift
                else:
                    fitness = 10 #Not going to happen
        else:
            fitness = - fisher
        self.seen_fenotypes[key] = {"fitness": fitness, "lift": lift, "precedent_support": precedent_support, "fisher": fisher}
        return self.seen_fenotypes[key]
        
    
    def fitness_fisher(self, fenotype):
        precedent_set = set(self.df.index)
        for i in range(len(fenotype)):
            if fenotype[i] != 0:
                col_name = self.dict_num_col[i]
                val = self.dicts_num_val[col_name][fenotype[i]]
                precedent_set = precedent_set.intersection(self.grouped_indexes[col_name][val])
        _a_set = set(self.df.index) - precedent_set
        _ab_set = _a_set.intersection(self.grouped_indexes[self.consequent][self.consequent_value])
        rule_set = precedent_set.intersection(self.grouped_indexes[self.consequent][self.consequent_value])
        # Defining contingency table values
        ab = len(rule_set)
        a_b = len(precedent_set) - ab
        _ab = len(_ab_set)
        _a_b = len(_a_set) - _ab
        fisher = stats.fisher_exact(table=[[ab,_ab],[a_b,_a_b]], alternative="less")[1]
        
        try:
            confidence = len(rule_set)/len(precedent_set)
        except:
            confidence = 0
        lift = confidence/self.consequent_support
        precedent_support = len(precedent_set)/len(self.df.index)
        fitness = -fisher
        return fitness, lift, precedent_support
            
    def fitness_lift(self, fenotype): # Genotype example [0,2,1, 0, 0] = {20-30, masc}
        '''
        
        Devuelve el lift de la regla y el soporte del precedente
        '''
        key = tuple(fenotype)
        if key in self.seen_fenotypes:
            return self.seen_fenotypes[key]
        
        precedent_set = None
        for i in range(len(fenotype)):
            if fenotype[i] != 0:
                col_name = self.dict_num_col[i]
                val = self.dicts_num_val[col_name][fenotype[i]]
                if not precedent_set:
                    precedent_set = self.grouped_indexes[col_name][val]
                else:
                    precedent_set = precedent_set & self.grouped_indexes[col_name][val]
                    
        # print("202", precedent_set)
                   
        # _a_set = set(self.df.index) - precedent_set
        tot = int("1"*len(self.df.index), 2)
        b_set = self.grouped_indexes[self.consequent][self.consequent_value]
        _a_set = tot ^ precedent_set
        # _b_set = tot ^ b_set
        # _ab_set = _a_set.intersection(self.grouped_indexes[self.consequent][self.consequent_value])
        _ab_set = _a_set & b_set
        ab_set = precedent_set & b_set
        a = bin(precedent_set).count('1')
        # rule_set = precedent_set.intersection(self.grouped_indexes[self.consequent][self.consequent_value])
        # Defining contingency table values
        # ab = len(rule_set)
        ab = bin(ab_set).count('1')
        a_b = a - ab
        _ab = bin(_ab_set).count('1')
        _a_b = bin(_a_set).count('1') - _ab
        fisher = stats.fisher_exact(table=[[ab,_ab],[a_b,_a_b]], alternative="less")[1]
        try:
            confidence = ab/a
        except:
            confidence = 0
        lift = confidence/self.consequent_support
        precedent_support = a/len(self.df.index)
        personas = a
        fitness = lift
        self.seen_fenotypes[key] = {"fitness": fitness, "lift": lift, "precedent_support": precedent_support, "fisher": fisher}
        return self.seen_fenotypes[key]
        
        
        
        
        '''
        '''
        precedent_set = set(self.df.index)
        for i in range(len(fenotype)):
            if fenotype[i] != 0:
                col_name = self.dict_num_col[i]
                val = self.dicts_num_val[col_name][fenotype[i]]
                precedent_set = precedent_set.intersection(self.grouped_indexes[col_name][val])
                
        rule_set = precedent_set.intersection(self.grouped_indexes[self.consequent][self.consequent_value])
        try:
            confidence = len(rule_set)/len(precedent_set)
        except:
            confidence = 0
        lift = confidence/self.consequent_support
        precedent_support = len(precedent_set)/len(self.df.index)
        fitness = lift
        return fitness, lift, precedent_support
    
    def describe_generation(self, generation = None):
        if not generation:
            generation = self.generation
        stats = {tuple(fenotype):self.fitness(fenotype) for fenotype in generation}
        sorted_generation = sorted(generation, key = lambda x: stats[tuple(x)]['fitness'], reverse = True)
        
        key_checker = list(map(tuple, generation))
        # fitnesses = stats[:,0]
        # lifts = stats[:,1]
        # supports = stats[:,2] 
        
        fitnesses = np.array([stats[tuple(fen)]['fitness'] for fen in sorted_generation])
        lifts = np.array([stats[tuple(fen)]['lift'] for fen in sorted_generation])
        supports = np.array([stats[tuple(fen)]['precedent_support'] for fen in sorted_generation])
        fishers = np.array([stats[tuple(fen)]['fisher'] for fen in sorted_generation])
        
        print(f'Current generation ({len(self.generation)} pop)')
        print(f"fitness -> mean:{fitnesses.mean()}, max:{fitnesses.max()}, min:{fitnesses.min()}")
        print(f"lift -> mean:{lifts.mean()}, max:{lifts.max()}, min:{lifts.min()}")
        print(f"support -> mean:{supports.mean()}, max:{supports.max()}, min:{supports.min()}")
        print(f"fisher -> mean:{fishers.mean()}, max:{fishers.max()}, min:{fishers.min()}")
        printed_gen_set = set()
        for gen in sorted_generation:
            key = tuple(gen)
            if key not in printed_gen_set:
                fitness = stats[key]['fitness']
                lift = stats[key]['lift']
                precedent_support = stats[key]['precedent_support']
                fisher = stats[key]['fisher']

                print(f'\t{self.decode_fenotype(gen)}\n\t\tfitness:{fitness}, lift:{lift}, fisher:{fisher},precedent_support:{round(precedent_support*len(self.df.index))}personas ({round(precedent_support, 8)}), appearances: {key_checker.count(key)}')
                printed_gen_set.add(key)

################################ END OF DEALER CLASS #########################################

def genetic_mining(transactions, **kwargs):
    min_lift = kwargs.get('min_lift', 1.01)
    min_precedent_support = kwargs.get('min_precedent_support', .01)
    consequent = kwargs.get('consequent')
    consequent_value = kwargs.get('consequent_value', 1)
    seed = kwargs.get('consequent_value', 42)
    return Dealer(transactions, min_lift, min_precedent_support, consequent, consequent_value, seed)



if __name__== "__main__":
    seed = 42
    file = "./cooked_groceries.csv"
    # columns_to_use = ['provincia', 'edad', 'genero', 'renta_bruta_media', 'tipo_servicio', 'tipo_facturacion','tarifa_id','coste_tarifa','tipo_venta', 'operador_portabilidad', 'num_ventas','alarma','servicios_adicionales', 'financia', 'energia']
    
    transaction_df = pd.read_csv(file).sample(frac = 1, random_state = seed)
    # transaction_df.drop_duplicates(keep = 'first', subset=["customer_id"], inplace=True)
    transaction_df.reset_index(inplace = True)
    transaction_df = transaction_df.drop(columns = list(transaction_df.columns)[:1])
    d = genetic_mining(transaction_df, min_precedent_support = .01,min_lift = 1.02, consequent = "beef", consequent_value = "beef", seed = seed)
    d.run(gen_size = 500, num_generations = 200, fitness_function = "lift")
    