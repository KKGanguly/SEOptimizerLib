import pandas as pd
import numpy as np
import random


class ReplayMemory(object):

    def __init__(self):
        self.actions = []
        self.rewards = []

    def push(self, action, reward):
        self.actions.append(action.tolist())
        self.rewards.append(reward.tolist())

    def get_all(self):
        return self.actions, self.rewards

class solution_holder:
    def __init__(self, id, decisions, objective, rank):
        self.id = id
        self.decision = decisions
        self.objective = objective
        self.rank = rank


class file_data:
    def __init__(self, name, training_set, testing_set, all_set, independent_set, features, dict_search):
        self.name = name
        self.training_set = training_set
        self.testing_set = testing_set
        self.all_set = all_set
        self.independent_set = independent_set 
        self.features = features
        self.dict_search = dict_search


def get_data(filename, initial_size=5,seed=1):
    random.seed(seed)
    pdcontent = pd.read_csv(filename)
    indepcolumns = [col for col in pdcontent.columns if "$<" not in col]  
    depcolumns = [col for col in pdcontent.columns if "$<" in col]        

    tmp_sortindepcolumns = []
    for i in range(len(indepcolumns)):
        tmp_sortindepcolumns.append(sorted(list(set(pdcontent[indepcolumns[i]]))))

    sortpdcontent = pdcontent.sort_values(by=depcolumns[-1])  
    ranks = {}
    for i, item in enumerate(sorted(set(sortpdcontent[depcolumns[-1]].tolist()))):
        ranks[item] = i

    content = list()
    for c in range(len(sortpdcontent)):
        content.append(solution_holder(
            c,
            sortpdcontent.iloc[c][indepcolumns].tolist(),
            sortpdcontent.iloc[c][depcolumns].tolist(),
            ranks[sortpdcontent.iloc[c][depcolumns].tolist()[-1]]
        )
        )
    dict_search = dict(zip([tuple(i.decision) for i in content], [i.objective[-1] for i in content]))
    random.shuffle(content)
    indexes = range(len(content))
    train_indexes, test_indexes = indexes[:
                                          initial_size],  indexes[initial_size:]
    assert (len(train_indexes) + len(test_indexes)
            == len(indexes)), "Something is wrong"
    train_set = [content[i] for i in train_indexes]
    test_set = [content[i] for i in test_indexes]

    file = file_data(filename, train_set, test_set,
                     content, tmp_sortindepcolumns, indepcolumns, dict_search)
    print("Finish reading data")
    return file

def load_features (features_fileName):
    header = features_fileName.features
    features = [t.decision for t in features_fileName.all_set]
    target = [t.objective[-1] for t in features_fileName.all_set]
    return header , features , target
    

def get_data_rank(filename, initial_size=5):
    random.seed(1)
    pdcontent = pd.read_csv(filename)
    indepcolumns = [col for col in pdcontent.columns if "$<" not in col]  
    depcolumns = [col for col in pdcontent.columns if "$<" in col]        


    tmp_sortindepcolumns = []
    for i in range(len(indepcolumns)):
        tmp_sortindepcolumns.append(sorted(list(set(pdcontent[indepcolumns[i]]))))


    sortpdcontent = pdcontent.sort_values(by=depcolumns[-1])  
    ranks = {}
    for i, item in enumerate(sorted(set(sortpdcontent[depcolumns[-1]].tolist()))):
        ranks[item] = i

    content = list()
    for c in range(len(sortpdcontent)):
        content.append(solution_holder(
            c,
            sortpdcontent.iloc[c][indepcolumns].tolist(),
            sortpdcontent.iloc[c][depcolumns].tolist(),
            ranks[sortpdcontent.iloc[c][depcolumns].tolist()[-1]]
        )
        )
    dict_search = dict(zip([tuple(i.decision) for i in content], [i.objective[-1] for i in content]))
    random.shuffle(content)
    indexes = range(len(content))
    train_indexes, test_indexes = indexes[:
                                          initial_size],  indexes[initial_size:]
    assert (len(train_indexes) + len(test_indexes)
            == len(indexes)), "Something is wrong"
    train_set = [content[i] for i in train_indexes]
    test_set = [content[i] for i in test_indexes]

    file = file_data(filename, train_set, test_set,
                     content, tmp_sortindepcolumns, indepcolumns, dict_search)
    print("Finish reading data with rank")
    return file,ranks