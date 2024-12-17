import jax
import jax.random as random
from jax import vmap
import jax.numpy as jnp
import numpy as np

from vamsl.inference import VaMSL
from vamsl.models import LinearGaussian, DenseNonlinearGaussian
from vamsl.metrics import expected_shd, threshold_metrics, neg_ave_log_likelihood
from vamsl.metrics import ParticleDistribution
from vamsl.utils.func import zero_diagonal
from vamsl.graph_utils import mat_to_graph

from sklearn.metrics import accuracy_score
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import confusion_matrix

import sys
import json
import pickle

def particle_to_g_lim(z, E_k):
        """
        Returns :math:`G` corresponding to :math:`\\alpha = \\infty` for particles `z`

        Args:
            z (ndarray): latent variables ``[..., d, k, 2]``

        Returns:
            graph adjacency matrices of shape ``[..., d, d]``
        """
        u, v = z[..., 0], z[..., 1]
        scores = jnp.einsum('...ik,...jk->...ij', u, v)
        g_samples = (E_k**2)*((1+E_k)/2) + (1-E_k**2)*(scores > 0).astype(jnp.int32)

        # mask diagonal since it is explicitly not modeled
        return zero_diagonal(g_samples)
    
    
def eltwise_particle_to_g_lim(q_z_k, E_k):
    return vmap(particle_to_g_lim, (0, None))(q_z_k, E_k)


def compwise_particle_to_g_lim(q_z, E):
    return vmap(eltwise_particle_to_g_lim, (0, 0))(q_z, E)


def identify_ordering(component_dists, mixture_data, eltwise_log_likelihood_observ):
        """
        Identify component indices based on n_componets number of ground truth graphs.
        Returns allocation that minimizes negative log likelihood between components and ground truths. 
        
        Args:
            mixture_data (ndarray): ndarray of shape [n_components, n_observations, n_vars] with data for each component.
            
        Outputs:
            order (array): array of with order of components
        """
        # expected log lik
        e_log_lik = lambda dist, x: -neg_ave_log_likelihood(dist=dist,
                                                            eltwise_log_likelihood=eltwise_log_likelihood_observ,
                                                            x=x)
        
        # Calculate log likelihood of all components for data
        cost_matrix = [[e_log_lik(dist, data) for data in mixture_data] for dist in component_dists]
        
        # use Hungaraian algorithm to find optimal allocation
        assignments = linear_sum_assignment(cost_matrix)
        
        # Return list of optimal allocations as order of components
        return assignments[0]


def identify_ESHD_ordering(component_dists, ground_truth_graphs, expected_shd):
        """
        Identify component indices based on n_componets number of ground truth graphs
        Returns allocation that minimizes ESHD between components and ground truths. 
        
        Args:
            mixture_data (ndarray): ndarray of shape [n_components, n_observations, n_vars] with data for each component.
            
        Outputs:
            order (array): array of with order of components
        """
        # expected log lik
        ESHD = lambda dist, graph: expected_shd(dist=dist, g=graph)
        
        # Calculate log likelihood of all components for data
        cost_matrix = [[ESHD(dist, graph) for graph in ground_truth_graphs] for dist in component_dists]
        
        # use Hungaraian algorithm to find optimal allocation
        assignments = linear_sum_assignment(cost_matrix)
        
        # Return list of optimal allocations as order of components
        return assignments[0]


def identify_classification_ordering(q_c, ground_truth_indicator):
        """
        Identify component indices based on n_componets number of ground truth graphs.
        Returns allocation that minimizes negative log likelihood betweent components and ground truths. 
        
        Args:
            mixture_data (ndarray): ndarray of shape [n_components, n_observations, n_vars] with data for each component.
            
        Outputs:
            order (array): array of with order of components
        """
        labels = jnp.arange(q_c.shape[1])
        # Get target and predicited assignments
        y_target = ground_truth_indicator
        y_pred = [jnp.argmax(c_i) for c_i in q_c] 

        # Solve linear assignment problem maximizing correct assignments
        cm = confusion_matrix(y_pred, y_target, labels=labels)        
        indexes = linear_sum_assignment(cm, maximize=True)
        
        # Return list of optimal allocations as order of components
        return indexes[1] 
    

def get_dist(*, g, theta, mixture=False, component_log_joint_prob=None, x=None, interv_mask=None):
    """
    Converts batch of binary (adjacency) matrices and parameters into particle distribution
    where mixture weights correspond to counts/occurrences or unnormalized target (i.e. posterior) probabilities.

    Args:
        g (ndarray): batch of graph samples ``[n_particles, d, d]`` with binary values
        theta (Any): PyTree with leading dim ``n_particles``

    Returns:
        :class:`~dibs.metrics.ParticleDistribution`:
        particle distribution of graph and parameter samples and associated log probabilities
    """
    N, _, _ = g.shape
    
    if mixture:
        # mixture weighted by respective joint probabilities
        eltwise_log_joint_target = vmap(lambda single_g, single_theta:
                                        component_log_joint_prob(single_g, single_theta, x, interv_mask, None),
                                        (0, 0), 0)
        logp = eltwise_log_joint_target(g, theta)
        logp -= logsumexp(logp)
    else:
        # since theta continuous, each particle (G, theta) is unique always
        logp = - jnp.log(N) * jnp.ones(N)

    return ParticleDistribution(logp=logp, g=g, theta=theta)


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
    ground_truth_gs = [res[i]['gs'] for i in range(len(res))] 
    ground_truth_thetas = [res[i]['thetas'] for i in range(len(res))]
    x_inds = [res[i]['indicated_data'] for i in range(len(res))]
    posts = [res[i]['posteriors'] for i in range(len(res))]
    E = [res[i]['E'] for i in range(len(res))]
    dts = [res[i]['delta_time'] for i in range(len(res))]
    
    return ground_truth_gs, ground_truth_thetas, x_inds, posts, E, dts
    

def calculate_results(exp_settings_file, exp_num, seed=897, mixture=False):
    """
    Calculates performance metrics of every replication in an experiment.
    
    Args:
        exp_settings_file (str): filename of experiment settings 
                                 Also used to form experiment output filename.
        exp_num (int): index of experiment in exp_settings_file.
        seed (int): random seed for generating held-out observations.
        mixture (bool): flag for get metrics with mixture particel dist. 
    
    Returns:
        eshds (ndarray): ndarray of shape [n_runs, n_components] with
                         ESHD for each component of each replication.
        aurocs (ndarray): ndarray of shape [n_runs, n_components] with
                          AUROC for each component of each replication.
        neglls (ndarray): ndarray of shape [n_runs, n_components] with
                          negative log likelihood for each component of each replication.
        ho_neglls (ndarray): ndarray of shape [n_runs, n_components] with
                             held-out negative log likelihood for each component of each replication.
        classification_accuracies (ndarray):  ndarray of shape [n_runs, 1] with
                                              classification accuracy of argmax of the responsibilities. 
                             
    """
    key = random.PRNGKey(seed)
    # Get experiment settings
    with open(exp_settings_file, "r") as read_file:
        exp_settings = json.load(read_file)[exp_num]

    print(f'Calculating metrics for experiments with settings:\n{exp_settings}')
        
    # Set likelihood model for calulcating negative log likelihood
    if exp_settings['struct_eq_type'] == 'linear':
        likelihood_model = LinearGaussian(n_vars=exp_settings['n_vars'])
    eltwise_log_likelihood_observ = vmap(lambda g, theta, x_ho:
            likelihood_model.interventional_log_joint_prob(g, theta, x_ho, jnp.zeros_like(x_ho), None), (0, 0, None), 0)
        
    # Get experiment output
    experiment_output_filename = 'experiment_results/' + exp_settings_file[20:-5] + '_run_' + str(exp_num) + '.p'
    ground_truth_gs, ground_truth_thetas, x_inds, posts, Es, dts  = get_experiment_output(filename=experiment_output_filename)
    
    # Create data structures to save data in
    n_components = len(exp_settings['mixing_rate'])
    n_runs = exp_settings['n_runs']
    eshds = np.ones((n_runs, n_components))
    aurocs = np.ones((n_runs, n_components))
    neglls = np.ones((n_runs, n_components))
    ho_neglls = np.ones((n_runs, n_components))
    classification_accuracies = np.ones((n_runs, 1))
    
    # loop over experiments
    for run in range(exp_settings['n_runs']):
        print(f'run: {run}')
        # Due to label switching, we first need to assign each ground truth graph to a component
        component_dists = []
        x_ho = []
        n_particles, n_vars = exp_settings['n_particles'], exp_settings['n_vars']
        for component in range(n_components):
            q_z_k, q_theta_k, E_k = posts[run][0][component], posts[run][1][component], Es[run][component]
            assert q_z_k.shape == (n_particles, n_vars, n_vars, 2), 'q_z_k is wrong shape: {q_z_k.shape}.'
            
            # Get graphs for component particles (cyclic graphs are discarded)
            q_g_k = eltwise_particle_to_g_lim(q_z_k, E_k)
            
            # Get particle dist for experiment replicate
            component_dists.append(get_dist(g=q_g_k, theta=q_theta_k, mixture=mixture))
        
        # Generate data from ground truth graphs
        for graph, theta in zip(ground_truth_gs[run], ground_truth_thetas[run]):
            key, subk = random.split(key)
            x_ho.append(likelihood_model.sample_obs(key=subk, n_samples=100, g=mat_to_graph(graph), theta=theta))

        """
        # Identify optimal mapping from ground truths to components
        order = identify_ordering(component_dists, jnp.stack(x_ho), eltwise_log_likelihood_observ)
        print(f'ho_negll order: {order}')
        order = identify_ESHD_ordering(component_dists, ground_truth_gs[run], expected_shd)
        print(f'ESHD order: {order}')
        """
        order = identify_classification_ordering(posts[run][2], x_inds[run][:, exp_settings['n_vars']])
        print(f'Classification order: {order}')
        
        
        # Calculate component-wise metrics with respect to assigned ground truth graph
        for component in range(n_components):
            # Get assigned ground truth graph and theta
            gt_i = order[component]
            #graph_k, theta_k = ground_truth_gs[run][0], ground_truth_thetas[run][0]
            graph_k, theta_k = ground_truth_gs[run][gt_i], ground_truth_thetas[run][gt_i]
            # get data and sample held-out observations from ground truth model
            i_indicator = exp_settings['n_vars']
            data_indices = x_inds[run][:, i_indicator] == gt_i # get observations from ground truth
            x_k = x_inds[run][data_indices, 0:exp_settings['n_vars']] # remove ground truth indicator
            key, subk = random.split(key)
            x_ho_k = likelihood_model.sample_obs(key=subk, n_samples=100, g=mat_to_graph(graph_k), theta=theta_k)
                      
            # Get component particle distribution
            if mixture:
                dist_k = get_dist(graph_k, theta_k, # Fix not to use ground truths
                                  mixture=True, 
                                  component_log_joint_prob=likelihood_model.interventional_log_joint_prob(), 
                                  x=x_k, 
                                  interv_mask=jnp.zeros_like(x_k))
            else:
                dist_k = component_dists[component]
            
            # Calculate component metrics
            eshds[run, gt_i] = expected_shd(dist=dist_k, g=graph_k)        
            aurocs[run, gt_i] = threshold_metrics(dist=dist_k, g=graph_k)['roc_auc']
            neglls[run, gt_i] = neg_ave_log_likelihood(dist=dist_k, eltwise_log_likelihood=eltwise_log_likelihood_observ, x=x_k)
            ho_neglls[run, gt_i] = neg_ave_log_likelihood(dist=dist_k, eltwise_log_likelihood=eltwise_log_likelihood_observ, x=x_ho_k)
            
            # ADD CALCULATION FOR INTERVENTIONAL data
            descr = f'run: {run}, component {component}' 
            print(f'{descr} |  E-SHD: {eshds[run, gt_i]:4.1f}    AUROC: {aurocs[run, gt_i]:5.2f}    neg. LL {neglls[run, gt_i]:5.2f}')
            print('\n') if component == n_components-1 else print('')

        # Calculate classification accuracy of argmax(q_c)
        y_target = x_inds[run][:, exp_settings['n_vars']]
        print(posts[run][3])
        q_c = posts[run][2][:, order] # reorder responsibilities using identified ordering 
        print(jnp.sum(q_c, axis=0))
        print(f'Sum of responsibilites: {jnp.sum(q_c, axis=0)}')
        y_pred = [jnp.argmax(c_i) for c_i in q_c] 
        classification_accuracies[run] = accuracy_score(y_target, y_pred)

    return eshds, aurocs, neglls, ho_neglls, classification_accuracies


def __main__(exp_settings_file, exp_num, seed=897, mixture=False):
    return calculate_results(exp_settings_file, exp_num, seed=seed, mixture=mixture)


if __name__ == '__main__':
    __main__(exp_dict_file=sys.argv[1], exp_num=int(sys.argv[2]))