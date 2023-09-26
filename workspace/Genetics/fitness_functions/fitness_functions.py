# -*- coding: utf-8 -*-
"""
Created on Thu Feb  2 19:48:48 2023

@author: romer
"""

def composed_fitness(lift, fisher, precedent_count):
    fitness = 0
    if fisher < .05 and precedent_count >= 25:
        if lift > 1:
            fitness = 10*lift
        else:
            if lift != 0:
                fitness = 1 / lift
            else:
                fitness = 10 #Not going to happen
    else:
        fitness = - fisher
    return fitness

def composed_fitness_length_fitness(lift, fisher, precedent_support,precedent_count, fenotype):
    fitness = 0
    non_zeros = len([i for i in fenotype if i])
    if fisher < .05 and precedent_count >= 25:
        if lift > 1:
            fitness = 10*lift
        else:
            if lift != 0:
                fitness = (1 / lift)/non_zeros
            else:
                fitness = 5 #Not going to happen
    else:
        fitness = - fisher
    return fitness

    
def lift_fitness(lift, precedent_support, precedent_count, fenotype):
    non_zeros = len([i for i in fenotype if i])
    if precedent_count > 0:
        return lift/non_zeros
    return 0

def lift_and_fisher_fitness(lift,fisher ,precedent_support, precedent_count):
    if lift > 1:
        return 1/fisher
    return lift -1