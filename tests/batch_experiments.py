import numpy as np
import jax.numpy as jnp
from jax import vmap
from jax import random

from vamsl.inference import VaMSL
from vamsl.models import MixtureLinearGaussian, MixtureDenseNonlinearGaussian
from vamsl.target import make_linear_gaussian_model, make_nonlinear_gaussian_model
from vamsl.elicitation.simulators import bernoulli_simulator
from vamsl.elicitation import edgeElicitation, graphOracle

import sys, ast
import time 
import pickle
import json


def generate_experiment_data(key, mixing_rate, n_vars, n_observations, graph_type, struct_eq_type):
    """
    Generate data for one experiment run. Data is generated from n_components number of randomly generated graphs. 
    The data is stacked and then shuffled.
    
    Args:
        subk (KeyArrayLike): PRNG key.
        n_components (int): number of components in mixture model.
        n_vars (int): number of variables in graph (d).
        n_observations (array): number of observations per component.
        graph_type (str): prior for generated graph type ('er' or 'sf').
        struct_eq_type (str): class of equations (linear or non-linear).
        
    Output:
        pass
    """
    # Generate ubkeys for each component
    key, *subks = random.split(key, len(mixing_rate)+1)
    
    # Generate either linear or non-linear data
    if struct_eq_type == "linear":
        make_data_model = make_linear_gaussian_model
        likelihood_model =  MixtureLinearGaussian(n_vars=n_vars)
    elif struct_eq_type == 'nonlinear':
        make_data_model = make_nonlinear_gaussian_model
        likelihood_model =  MixtureDenseNonlinearGaussian(n_vars=n_vars)
    else:
        raise ValueError(f'Supplied unknown data_model: {data_model}. Should be either linear or non-linear.')
    
    # Determine number of observations per component based on mixing rates
    comp_observations = jnp.floor(n_observations * mixing_rate)
    
    xs = []
    graphs = []
    thetas = []
    # loop over components
    for subk, n_comp_obs in zip(jnp.array(subks), comp_observations):
        data, graph_model, component_lik_model = make_data_model(key=subk, n_vars=n_vars, 
                                                                 graph_prior_str=graph_type,
                                                                 n_observations=int(n_comp_obs.item()))    
        xs.append(data.x)
        graphs.append(data.g)
        thetas.append(data.theta)
    
    # Combine observation sets
    x = np.concatenate(xs, axis=0)
    graphs = jnp.array(graphs)
    if struct_eq_type == "linear":
        thetas = jnp.array(thetas)
    
    # Add ground truth indicator for components
    indicator = np.concatenate([k * np.ones(int(comp_observations[k].item())) for k in range(comp_observations.shape[0])],axis=0).reshape((-1,1))
    x = np.hstack([x, indicator])
    
    # shuffle data 
    np.random.shuffle(x)
    
    # Return data sets
    return jnp.array(x), graphs, thetas, likelihood_model, component_lik_model, graph_model


def experiment_replication(key, n_particles, n_vars, n_observations,
                           graph_type, n_queries, expert_reliability, struct_eq_type, 
                           steps, burn_in_steps, updates, mixing_rate):
    """
    Args:
        subk (KeyArrayLike): PRNG key.
        n_particles (int): number of SVGD particles per component.
        n_vars (int): number of variables in graph (d).
        graph_type (str): prior for generated graph type ('er' or 'sf').
        struct_eq_type (str): class of equations (linear or non-linear). 
        queries (int): number of queries during variational updates.
        expert_reliability (float): expeted number of correct answers to queries.
        
    Returns:
        pass
    """
    n_components = mixing_rate.shape[0]
    # Generate data for experiment
    key, subk = random.split(key)
    x_ind, ground_truth_graphs, ground_truth_thetas, lik, component_lik, graph_model = generate_experiment_data(subk,
                                                                                                                mixing_rate,
                                                                                                                n_vars,
                                                                                                                n_observations,
                                                                                                                graph_type,
                                                                                                                struct_eq_type)

    # remove indicator vector from data
    # [n_observations, n_vars]
    x = x_ind[:,0:n_vars]
    
    # initalize timer
    dt = 0
    
    # Randomly initial assignments 
    c = np.zeros((N:=x.shape[0], K:=n_components)) # matrix of assignment to cluster (boolean mask)
    # fill c with random responsibilities
    for n in range(N):
        k = np.random.randint(0, K) # randomly assign each datapoint to a cluster a priori
        c[n,k] = 1
    q_c = jnp.array(c)
    
    # Create VaMSL and initialize posteriors (remove indicator vecor from dataset)
    vamsl = VaMSL(x=x, graph_model=graph_model, mixture_likelihood_model=lik, component_likelihood_model=component_lik)
    key, subk = random.split(key)
    vamsl.initialize_posteriors(key=subk, init_q_c=q_c, n_particles=n_particles)
   
    # If queries, create necessary objects
    if not n_queries == 0:
        # Create query selector and oracle for simulating expert elicitation
        elicitor = edgeElicitation(simulator=bernoulli_simulator(), expected_utility='Rao-Blackwellized EIG')
        oracle = graphOracle(ground_truth_graphs)
        # List all possible experiments [n_components, n_vars**2, 2]
        experiment_lists = jnp.array([[(i, j) for i in range(n_vars) for j in range(n_vars)] for k in range(n_components)])
        # Empty initial eliciation matrix
        E = jnp.zeros((n_components, n_vars,n_vars))
    
    # SAMPLE BURN IN POSTERIORS
    for step in range(0, burn_in_steps, int(burn_in_steps/updates)):
        # Elicitation if queries 
        if not n_queries == 0:
            indices_list = []
            for component in range(n_components):
                # Get component parameters which correspond to particle-wise edge probabilities
                q_z_k, E_k = vamsl.get_posteriors()[0][component], E[component]
                parameter_samples = vmap(vamsl.edge_probs, (0, None, None))(q_z_k, step, E_k)
                # Get optimal queries
                exps, EIGs, indices = elicitor.optimal_queries(parameter_samples=parameter_samples, 
                                                               experiment_list=experiment_lists[component],
                                                               n_queries=n_queries)
                # Update elicitation matrix
                key, subk = random.split(key)
                E = oracle.update_elicitation_matrix(E=E, component=component, queries=exps,
                                                     stochastic=True, key=subk, reliability=expert_reliability)
                indices_list.append(indices)
            
            # Remove experiments that were queried from experiment list
            experiment_lists = jnp.array([jnp.delete(exp_list, exp_is, axis=0) for exp_list, exp_is in zip(experiment_lists, indices_list)])
            # Update the VaMSL elicitation matrix
            vamsl.set_E(E)
            
        # Time of posterior inference
        t1 = time.time()
        # Update to optimal q(c) and q(\pi)
        vamsl.update_responsibilities_and_weights()
        # Optimize q(Z, \Theta)
        key, subk = random.split(key)
        vamsl.update_particle_posteriors(key=subk, steps=step)
        dt += time.time() - t1
        
        
    # SAMPLE FINAL POSTERIORS 
    # Time of posterior inference
    t1 = time.time()
    # Update to final posteriors q(c) and q(\pi)
    vamsl.update_responsibilities_and_weights()
    # Sample final particle posteriors q(Z, \Theta)
    key, subk = random.split(key)
    vamsl.update_particle_posteriors(key=subk, steps=steps-burn_in_steps)
    dt += time.time() - t1
    
    # Return posteriors and experiment data
    return {'gs': ground_truth_graphs, 'thetas': ground_truth_thetas, 'posteriors': vamsl.get_posteriors(), 
            'E': vamsl.get_E(), 'Indicated_data': x_ind, 'delta_time': dt}


def run_experiments(*, seed, n_runs, mixing_rate, n_particles, n_vars, n_observations,
                    graph_type, n_queries, expert_reliability, struct_eq_type, steps,
                    burn_in_steps, updates):
    """
    Run n_runs replications of experiments.
    
    Args:
        seed (int): seed for PRNG key.
        n_runs (int): number of replicate eperiments to run 
        n_components (int): number of components in mixture model.
        n_particles (int): number of SVGD particles per component.
        n_vars (int): number of variables in graph (d).
        n_observations (int): number of observations.
        graph_type (str): prior for generated graph type.
        struct_eq_type (str): class of equations (linear or non-linear). 
        queries (int): number of queries during variational updates.
        expert_reliability (float): expeted number of correct answers to queries.
        
    Returns:
        results (ndarray): 
    """
    key = random.PRNGKey(seed)
    # Generate PRNG subkey for each experiment replication
    key, *subks = random.split(key, n_runs+1)
    
    """
    # Run experiments
    return vmap(experiment_replication, (0, None, None, None, None, None, None, None, None, None, None, None))(jnp.array(subks), 
                                                                                                                n_particles, 
                                                                                                                n_vars, 
                                                                                                                n_observations,
                                                                                                                graph_type, 
                                                                                                                n_queries, 
                                                                                                                expert_reliability, 
                                                                                                                struct_eq_type, 
                                                                                                                steps,
                                                                                                                burn_in_steps,
                                                                                                                updates,
                                                                                                                mixing_rate)
                                                                                                                """
    return [experiment_replication(subk, n_particles, n_vars, n_observations,graph_type, n_queries, expert_reliability, struct_eq_type, steps,burn_in_steps,updates, mixing_rate) for subk in jnp.array(subks)]


def __main__(*, exp_dict_file, exp_num):
    # Get experiment variable dictionary from expriment index
    with open(exp_dict_file, "r") as read_file:
        exp_dict = json.load(read_file)[exp_num]
    
    # Load variables
    seed = exp_dict['seed']
    n_runs  = exp_dict['n_runs']
    mixing_rate = jnp.array(exp_dict['mixing_rate'])
    n_components = mixing_rate.shape[0]
    n_particles = exp_dict['n_particles']
    n_vars = exp_dict['n_vars']
    n_observations = exp_dict['n_observations']
    graph_type = exp_dict['graph_type']
    n_queries = exp_dict['n_queries']
    expert_reliability = exp_dict['expert_reliability']
    struct_eq_type = exp_dict['struct_eq_type']
    steps = exp_dict['steps']
    burn_in_steps = exp_dict['burn_in_steps']
    updates = exp_dict['updates']
    
    # Checks 
    assert graph_type in ['er', 'sf'], f"graph_type must be either 'er' or 'sf'. Got {graph_type}."
    assert 0 <= expert_reliability <= 1, f"expert_reliability must be [0,1] probability. Got {expert_reliability}."
    assert struct_eq_type in ['linear', 'nonlinear'], f"struct_eq_type must either 'linear' or 'nonlinear'. Got {struct_eq_type}."
    
    # Run experiments
    res = run_experiments(seed=seed, n_runs=n_runs, mixing_rate=mixing_rate, n_particles=n_particles,
                          n_vars=n_vars, n_observations=n_observations, graph_type=graph_type,
                          n_queries=n_queries, expert_reliability=expert_reliability, struct_eq_type=struct_eq_type,
                          steps=steps,burn_in_steps=burn_in_steps, updates=updates)
    
    # Create filename based on experiment dictionary soruce and experiment index
    result_filename = 'results/' + exp_dict_file[:-5] + '_run_' + str(exp_num) + '.p'
    
    # Pickle results
    pickle.dump(res,  open(result_filename, "wb"))
    print('pickled')


if __name__ == '__main__':
    res = __main__(exp_dict_file=sys.argv[1], exp_num=int(sys.argv[2]))
    pickle.dump(res,  open("init_batch_test.p", "wb"))
    print('pickled')