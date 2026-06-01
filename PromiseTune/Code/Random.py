import numpy as np
import random
import warnings
from util import get_objective
from util import read_file

warnings.filterwarnings('ignore')

def random_sample(filename, max_lives, budget, seed):

    random.seed(seed)
    np.random.seed(seed)
    
    initial_size = 0
    lives = max_lives
    steps = 0
    
    file = read_file.get_data(filename, initial_size, seed)
    training_dep = [t.objective[-1] for t in file.training_set]
    training_indep = [t.decision for t in file.training_set]
    
    best_result = min(training_dep)
    results, x_axis, configurations = [], [], []
    existing_configurations = set(tuple(config) for config in training_indep)
    
    while steps < budget:
        configuration = [random.choice(file.independent_set[i]) 
                        for i in range(len(file.independent_set))]
        
        reward, config_str = get_objective.get_objective(file.dict_search, configuration)
        
        training_indep.append(configuration)
        training_dep.append(reward)
        x_axis.append(steps)
        results.append(reward)
        configurations.append(config_str)
        
        config_tuple = tuple(configuration)
        if config_tuple in existing_configurations:
            budget += 1
        else:
            existing_configurations.add(config_tuple)
        
        if reward < best_result:
            best_result = reward
            lives = max_lives
        else:
            lives -= 1
        
        if lives == 0:
            break
            
        steps += 1
        print(f'Step: {steps}, Reward: {reward:.2f}, Config: {config_str}')
    
    return np.array(configurations), np.array(results), best_result

def run_experiment(seed, system_name):

    configurations, results, best_result = random_sample(
        filename=f"./Data/{system_name}.csv",
        max_lives=200,
        budget=100,
        seed=seed
    )
    print(f"System: {system_name}, Seed: {seed}")
    print(f"Best result: {best_result:.2f}")
    return configurations, results, best_result

def main():

    seeds = [1]
    systems = [
        'exastencils', 'dconvert', '7z', 'BDBC', 'deeparch',
        'PostgreSQL', 'javagc', 'storm', 'x264',
        'redis', 'HSQLDB', 'LLVM'
    ]
    
    for system_name in systems:
        for seed in seeds:
            run_experiment(seed, system_name)

if __name__ == '__main__':
    main()