import jax
import jax.random as random
from jax import vmap
import jax.numpy as jnp
import numpy as np

import sys
import json
import pickle

def get_experiment_output(*, filename):
    """
    Loads and unpacks the results of one experiments with n_runs replications.

    Args:
        filename (str): path/filename to experiment output file.

    Returns:
        ground_truth_gs (list): list of shape [n_runs, n_components, n_vars, n_vars] 
                                ground truth graphs for each replication.
        ground_truth_thetas (list): list of shape [n_runs, n_components, n_vars, n_vars] 
                                    ground truth thetas for each replication.
        x_inds (list): list of shape [n_runs, n_observations , n_vars+1] with observations
                       and extra column of indicators for ground truth source of observation.
        E (list): list of shape [n_runs, n_component,  n_vars, n_vars] with elicited edge information.
        dts (list): list of shape [n_runs] with running times for each replication.
    """
    # Load in experiment output object
    res = pickle.load(open(filename, "rb"))
    
    # Unpack results
    ground_truth_gs = [np.array(res[i]['gs']) for i in range(len(res))] 
    ground_truth_thetas = [np.array(res[i]['thetas']) for i in range(len(res))]
    x_inds = [np.array(res[i]['indicated_data']) for i in range(len(res))]
    posts = [[np.array(post) for post in res[i]['posteriors']] for i in range(len(res))]
    E = [np.array(res[i]['E']) for i in range(len(res))]
    dts = [res[i]['delta_time'] for i in range(len(res))]
    
    return [ground_truth_gs, ground_truth_thetas, x_inds, posts, E, dts]
    

def __main__(experiment_settings_filename):
    with open(experiment_settings_filename, "r") as read_file:
        experiment_settings = json.load(read_file)

    cond_file = experiment_settings_filename[:-5]+'_cond.p'
    conditional = pickle.load(open(cond_file, "rb"))
    
    # Create conditional which is true if index is with respect to specified dictionary
    cond = lambda i: experiment_settings[i]['mixing_rate'] in conditional['mixing_rate'] and \
                     experiment_settings[i]['n_queries'] in conditional['n_queries'] and \
                     experiment_settings[i]['n_runs'] in conditional['n_runs'] and \
                     experiment_settings[i]['n_particles'] in conditional['n_particles'] and \
                     experiment_settings[i]['n_observations'] in conditional['n_observations'] and \
                     experiment_settings[i]['graph_type'] in conditional['graph_type'] and \
                     experiment_settings[i]['expert_reliability'] in conditional['expert_reliability'] and \
                     experiment_settings[i]['steps'] in conditional['steps'] and \
                     experiment_settings[i]['burn_in_steps'] in conditional['burn_in_steps'] and \
                     experiment_settings[i]['updates'] in conditional['updates'] and \
                     experiment_settings[i]['init_queries'] in conditional['init_queries']

    exp_indices = [i for i in range(len(experiment_settings)) if cond(i)]
    
    def get_results(experiment_settings_filename, i):
        experiment_output_filename = 'experiment_results_dibs/' + experiment_settings_filename[20:-5] + '_run_' + str(i) + '.p'
        return get_experiment_output(filename=experiment_output_filename)
    
    # Metrics of shape [experiments, 4, n_runs, component] (four metrics)
    outputs = [get_results(experiment_settings_filename, i) for i in exp_indices]

    outputs_filename = cond_file[:-6]+str('outputs_dibs.p')
    pickle.dump([outputs, exp_indices],  open(outputs_filename, "wb"))
    print(f'pickled outputs to {outputs_filename}...')
    

if __name__ == '__main__':
    __main__(experiment_settings_filename=sys.argv[1])