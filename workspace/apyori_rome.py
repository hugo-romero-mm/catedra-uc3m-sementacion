# -*- coding: utf-8 -*-
"""
Created on Fri Nov 18 16:25:33 2022

@author: romer
"""

from apyori import apriori as apriori_old


class Dealer:
    def __init__(self, apriori_result: list, verbose=False, create_df=False):
        print("Creating apriori dealer...")
        self.result = apriori_result
        self.good_rules = self.get_good_rules(verbose=verbose, createDF=create_df)

        self.itemsets_supports = dict()
        self.itemsets = self.get_itemsets()

        self.large_itemsets = self.get_largest_itemsets()
        self.maximals = self.get_maximal_itemsets()
        self.maximals_supports = {str(i): self.itemsets_supports[str(i)] for i in self.maximals}

    def get_itemsets(self):
        itemsets = []
        supports = []
        for res in self.result:
            itemsets.append(list(res.items))
            supports.append(res.support)
        print("\tItemsets found: ", len(itemsets))
        self.itemsets_supports = dict(zip(map(str, itemsets), supports))
        return itemsets

    def get_largest_itemsets(self):
        largest = len(max(self.itemsets, key=lambda x: len(x)))
        itemsets = list(filter(lambda x: len(x) == largest, self.itemsets))
        print("\tLarge itemsets: ", len(itemsets), "--- of size: ", largest)
        return itemsets

    def get_maximal_itemsets(self):
        contained = set()
        for it1 in self.itemsets:
            for it2 in self.itemsets:
                if it1 != it2:
                    if set(it1) <= set(it2):
                        contained.add(str(it1))
        maximals = [item for item in self.itemsets if str(item) not in contained]
        return maximals

    def get_good_rules(self, good='financia', verbose=False, createDF=False):
        import pandas as pd
        precedents = []
        consequents = []
        confidences = []
        lifts = []

        print(f"Searching for rules where {good} is the consequent...")
        good_rules = []
        self.good_rules_resume = ""
        for res in self.result:
            for ordered_statistic in res.ordered_statistics:
                if ordered_statistic.items_base and len(
                        ordered_statistic.items_add) == 1 and good in ordered_statistic.items_add:
                    good_rules.append(res)
                    self.good_rules_resume += f"Good Rule found---\n\
                        \tPrecedent {ordered_statistic.items_base}\n\
                        \tConsequent {ordered_statistic.items_add}\n\
                        \tConfidence {ordered_statistic.confidence}\n\
                        \tLift {ordered_statistic.lift}\n"
                    precedents.append(list(ordered_statistic.items_base))
                    consequents.append(list(ordered_statistic.items_add)[0])
                    confidences.append(ordered_statistic.confidence)
                    lifts.append(ordered_statistic.lift)
        if createDF:
            self.good_rules_df = pd.DataFrame()
            self.good_rules_df["precedents"] = precedents
            self.good_rules_df["consequents"] = consequents
            self.good_rules_df["confidences"] = confidences
            self.good_rules_df["lifts"] = lifts

        print(f"Found {len(good_rules)} with {good} as consequent. Property of object .good_rules")
        if verbose:
            print(self.good_rules_resume)

        return good_rules


################################

def apriori(transactions, min_support=.1, min_confidence=.9, min_lift=1, verbose=False, createDF=False):
    print("Start of running apriori...")
    result = list(apriori_old(transactions, min_support=min_support, min_confidence=min_confidence, min_lift=min_lift))
    return Dealer(result, verbose=verbose, create_df=createDF)
