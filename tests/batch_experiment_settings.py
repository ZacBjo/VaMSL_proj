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
    # All settings for experiment variables
    var_dict = {
                'seed': [924], 
                'n_runs': [30], 
                'mixing_rate': [[0.5,0.5]],
                'n_particles': [15],
                'n_vars': [20],
                'n_observations': [2000],
                'graph_type': ['sf'],
                'n_queries': [0, 2, 4, 6, 8],
                'expert_reliability': [1],
                'struct_eq_type': ['linear'],
                'steps': [3000],
                'burn_in_steps': [1000],
                'updates': [10]
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
