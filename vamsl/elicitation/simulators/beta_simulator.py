import jax.numpy as jnp
import jax.random as random

from jax.scipy.stats import beta

class beta_simulator:
    r"""
    Simple continuous experiment simulator used to simulate outcomes for queries about edge existence.
    
    An experiment either returns a probability for there being an edge as defined by the experiment.
    The experiment is simulated as a Beta distributed sample parametrized by an latent edge probability. 
    The edge probabilities are calculated using the VaMSL (DiBS) method.
     
    In the terminology of Rainforth et al. (2023) - Modern Bayesian Experimental Design:
    - The parameter \theta: A [n_var, n_var] matrix of edge probabilities.
    - The experiment \xi_{ij}: The query whether the edge i -> j exist in the graph.
    - The outcome y_{ij}: A continuous variable y_{ij} \in \[0,1\] indidcating edge existence.
    - The simulator p(y|\theta, \xi_[ij}): A beta distributed sample parametrized by the edge probability \theta_{ij}
      and its complement. 
    """
    def __init__(self, alpha_s=1):
        self.alpha_s = alpha_s
        
        
    def get_exp_log_prob(self, y, edge_probs, experiment, alpha_s=None, beta_s=None):
        # An experiment corresponds to asking about a single edge
        i, j = experiment[0], experiment[1]
        
        # Return edge probability 
        if alpha_s is None:
            a, b = 1+(self.alpha_s*edge_probs[i,j]),  1+(self.alpha_s*(1-edge_probs[i,j]))
        elif beta_s is None:
            a, b = 1+(alpha_s*edge_probs[i,j]), 1+(alpha_s*(1-edge_probs[i,j]))
        else:
            a, b = 1+(alpha_s*edge_probs[i,j]), 1+(beta_s*(1-edge_probs[i,j]))
        
        return beta.logpdf(y, a, b=b)
    
    
    def get_experiment_log_likelihood(self, y, edge_probs, experiment, alpha=None, beta=None):
        return self.get_exp_log_prob(y, edge_probs, experiment, alpha, beta)
    
    
    def get_edge_prob(self, edge_probs, experiment):
        # An experiment corresponds to asking about a single edge
        i, j = experiment[0], experiment[1]
        
        # Return edge probability 
        return edge_probs[i,j]
    
        
    def run_experiment(self, key, edge_probs, experiment):
        # Get edge probability for edge defined by experiment
        edge_prob = self.get_edge_prob(edge_probs, experiment)
        
        return random.beta(key, 1+(self.alpha_s*edge_prob), 1+(self.alpha_s*(1-edge_prob)))