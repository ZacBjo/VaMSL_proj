import jax.numpy as jnp
import jax.random as random

from jax.scipy.stats import beta

class beta_simulator:
    r"""
    Simple continuous experiment simulator used to simulate outcomes for queries about edge existence.
    
    An experiment either returns either 1 (indicating existing edge) or 0 (indicating missing edge).
    The experiment is simulated as a Beta distributed sample parametrized by an edge probability. 
    The edge probabilities are calculated using the DiBS method.
     
    In the terminology of Rainforth et al. (2023) - Modern Bayesian Experimental Design:
    - The parameter \theta: A [n_var, n_var] matrix of edge probabilities.
    - The experiment \xi_{ij}: The query whether the edge i -> j exist in the graph.
    - The outcome y_{ij}: A binary variable y_{ij} \in \{0,1\} idnidcating edge (non-)existence.
    - The simulator p(y|\theta, \xi_[ij}): A beta distributed sample parametrized by the edge probability \theta_{ij}
      and its complement. 
    """
    def __init__(self, c=1):
        self.c = c
        
        
    def get_exp_log_prob(self, y, edge_probs, experiment):
        # An experiment corresponds to asking about a single edge
        i, j = experiment[0], experiment[1]
        
        # Return edge probability 
        a, b = 1+edge_probs[i,j], 1+(1-edge_probs[i,j])
        
        return beta.logpdf(y, a, b=b)
    
    
    def get_experiment_log_likelihood(self, y, edge_probs, experiment):
        return self.get_exp_log_prob(y, edge_probs, experiment)
    
    
    def get_edge_prob(self, edge_probs, experiment):
        # An experiment corresponds to asking about a single edge
        i, j = experiment[0], experiment[1]
        
        # Return edge probability 
        return edge_probs[i,j]
    
        
    def run_experiment(self, key, edge_probs, experiment):
        # Get edge probability for edge defined by experiment
        edge_prob = self.get_edge_prob(edge_probs, experiment)
        
        return random.beta(key, 1+(self.c*edge_prob), 1+(self.c*(1-edge_prob)))