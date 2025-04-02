import jax.numpy as jnp
from jax import vmap
import jax.random as random
from jax.lax import top_k

from jax.scipy.special import logsumexp

class edgeElicitation:
    def __init__(self, *, simulator, expected_utility):
        self.simulator = simulator
        self.expected_utility = expected_utility
        
        
    def Rao_Blackwell_EIG(self, experiment_liks, err=10**-20):
        # Get number of parameter samples
        N = experiment_liks.size
        
        # Get complements p( y = 0 | \theta, \xi_ij ) of p( y = 1 | \theta, \xi_ij )
        complement_experiment_liks = 1 - experiment_liks

        # p( y = 1 | \xi_ij )
        marginal_success = jnp.sum(experiment_liks) / N
        # p( y = 0 | \xi_ij )
        marginal_failure = jnp.sum(complement_experiment_liks) / N
        
        # \sum_y p( y | \xi_ij ) * log p( y | \xi_ij )
        expected_log_marginal = marginal_success * jnp.log(marginal_success + err) + \
                                marginal_failure * jnp.log(marginal_failure + err)
        
        # p( y = 1 | \theta, \xi_ij ) * log p( y = 1 | \theta, \xi_ij )
        expected_log_lik = experiment_liks * jnp.log(experiment_liks + err)
        # p( y = 0 | \theta, \xi_ij ) * log p( y = 0 | \theta, \xi_ij )
        expected_log_complement_lik = complement_experiment_liks * jnp.log(complement_experiment_liks + err)
        
        # 1/N * [ \sum_N sum_y p( y | \theta, \xi_ij ) * log p( y | \theta, \xi_ij ) ]
        # - \sum_y p( y | \xi_ij ) * log p( y | \xi_ij )
        RB_estimate = (1/N) * jnp.sum(expected_log_lik + expected_log_complement_lik) - expected_log_marginal
        
        return RB_estimate
    
    
    def nested_monte_carlo_EIG(self, key, parameter_samples, experiment):
        # Get number of parameter samples
        N = parameter_samples.shape[0]
        
        # draw y^(s) from each Z_k^(s) | exp_ij
        key, *batch_subk = random.split(key, N + 1)
        simulate_response = lambda key, p, exp: self.simulator.run_experiment(key=key, edge_probs=p, experiment=exp)
        ys = vmap(simulate_response, (0, 0, None))(jnp.array(batch_subk), parameter_samples, experiment)
        
        # Compute likelihood of each draw p(y^(s)|Z_k^(s), exp_ij )
        log_y_liks = vmap(self.simulator.get_experiment_log_likelihood, (0, 0, None))(ys, parameter_samples, experiment)
        
        # marginalize likelihood for each draw  
        # nu^(s) = 1/s' \sum_s' p(y^(s)|Z_k^(s'), exp_ij), s=1,...,S
        log_nu_s = lambda y, parameters, experiment: logsumexp(a=jnp.array([self.simulator.get_experiment_log_likelihood(y, parameter, experiment) for parameter in parameters]), b=(1/N))
        log_nus = vmap(log_nu_s, (0, None, None))(ys, parameter_samples, experiment)
        
        # normalize each log likelihood log( p(y^(s)|Z_k^(s'), exp_ij) / nu^(s) )
        normalize = lambda log_p_y_s, log_nu_s: log_p_y_s - log_nu_s
        norm_log_y_liks = vmap(normalize, (0, 0))(log_y_liks, log_nus) 
        
        # MC average normalized log liks
        NMC_estimate = (1/N)*norm_log_y_liks.sum()
        
        return NMC_estimate
    
    
    def experiment_EIG(self, parameter_samples, experiment, key):
        # Calculate Rao-Blackwellized estimator for EIG
        if self.expected_utility == 'Rao-Blackwellized EIG':
            # get experiment likelihoods p(y=1 | \theta, \xi_ij)
            experiment_liks = vmap(self.simulator.get_experiment_likelihood, (0, None))(parameter_samples, experiment)
            EIG = self.Rao_Blackwell_EIG(experiment_liks)
        elif self.expected_utility == 'NMC EIG':
            EIG = self.nested_monte_carlo_EIG(key, parameter_samples, experiment)
            
        return EIG
        
        
    def optimal_queries(self, *, parameter_samples, experiment_list, n_queries=1, key=None):
        """
        Description...
        
        Args:
            parameter_samples: array of size ```[n_particles, n_var, n_var]``` with edge prob matrices.
            
        Returns:
            max_experiments: array of size ```[n_queries, 2]``` with optimal queries ordered by best first.
            max_EIGs: array of size ```[n_queries, 1]``` with EIG values for corresponding optimal queries. 
        """
        assert (len(parameter_samples.shape) == 3), "Parameter samples is of shape: {0}.".format(parameter_samples.shape)
        
        # EIG for all experiments
        EIGs = vmap(self.experiment_EIG, (None, 0, None))(parameter_samples, experiment_list, key)
        
        # Get experiments with highest EIGs
        max_EIGs, max_indices = top_k(EIGs, n_queries)
        max_experiments = experiment_list[max_indices]
        
        return max_experiments, max_EIGs, max_indices