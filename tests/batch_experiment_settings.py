from numpy import random

import json
import itertools
import sys

def __main__(*, filename='dump.json'):
    """
    Generate all permutations of given settings of experiment variables.
    Writes settings as a list of dictionaries.
    
    Args:
        filename (str): name of JSON file that is written.
    """
    # Generate random seed for batch experiments
    seed = random.randint(low=0, high=10000)
    
    # All settings for experiment variables
    # Change according to desired experimental setup.
    var_dict = {
                'seed': [seed], 
                'n_runs': [30], 
                'mixing_rate': [[1.0, 0.0]],
                'n_particles': [15],
                'n_vars': [20],
                'n_observations': [200],
                'graph_type': ['sf'],
                'n_queries': [0],
                'init_queries': [0],
                'expert_reliability': [1.0],
                'struct_eq_type': ['linear'],
                'steps': [9000],
                'burn_in_steps': [6000], # set = 1 to skip
                'updates': [100] # can't be zero, set =1 to skip
               }
    
    # Generate list of variable permutations
    keys, values = zip(*var_dict.items())
    permutations_dicts = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    # Write list of variable permutations to JSON file 
    with open(filename, 'w') as write_file:
        json.dump(permutations_dicts, write_file)
        
    print(f'Wrote experiment settings to: {filename}. Number of experiments: {len(permutations_dicts)}')
    
if __name__ == '__main__':
    __main__(filename=sys.argv[1])
