from calculate_experiment_metrics import *

import sys
import json
import pickle

def __main__(*, experiment_settings_filename, cond_file):
    with open(experiment_settings_filename, "r") as read_file:
        experiment_settings = json.load(read_file)

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
                     experiment_settings[i]['updates'] in conditional['updates']

    exp_indices = [i for i in range(len(experiment_settings)) if cond(i)]
    
    def get_results(experiment_settings_filename, i):
        print(f"Experiment: {i+1}, mixing rate: {experiment_settings[i]['mixing_rate']}")
        return calculate_results(experiment_settings_filename, i)
    
    # Metrics of shape [experiments, 4, n_runs, component] (four metrics)
    metrics = [get_results(experiment_settings_filename, i) for i in exp_indices]

    metrics_filename = cond_file[:-6]+str('metrics.p')
    pickle.dump([metrics, exp_indices],  open(metrics_filename, "wb"))
    print(f'pickled metrics to {metrics_filename}...')


if __name__ == '__main__':
    __main__(experiment_settings_filename=sys.argv[1], cond_file=sys.argv[2])