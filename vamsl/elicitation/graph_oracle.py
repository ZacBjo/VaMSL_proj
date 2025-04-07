import jax.numpy as jnp
from jax import vmap
from jax import random
from jax.lax import cond
import numpy as np

from jax.scipy.stats import beta

class graphOracle:
    def __init__(self, graphs):
        """
        Graph oracle used to answer queries about ground truth graphs. 
        Includes stochastic repsonse function.  
        
        Args:
            graphs: array  of size ```[n_components, n_vars, n_vars]``` 
                    with ground truth adjacency matrices for each component.
        """
        self.graphs = graphs
        
        
    def answer_query(self, component, query, soft):
        """
        Returns the edge existence (0/1) for the edge in the given ground truth graph. 
        
        Args:
            component (int): index of component.
            query (tuple): index of edge in question.
                    
        Output:
            reponse (int): a binary indicator for edge existence/absence. 
        """
        i, j = query[0], query[1]
        if soft:
            return jnp.abs(0.9 - 1 + self.graphs[component][i, j])
        else:
            return self.graphs[component][i, j]
    
    
    def answer_queries(self, *, component, queries, soft):
        """
        Batch version of answer_query.
        """
        return vmap(self.answer_query, (None, 0, None))(component, queries, soft)
    
    
    def stochastically_answer_query(self, key, reliability, component, query, soft):
        """
        Stochastically returns the edge existence (0/1) for the edge in the given ground truth graph.
        The accuracy of the repsonses is determined by the [0,1] reliability value.
        
        Args:
            key (KeyArrayLike): PRNG key.
            reliability (float): Probability of returning incorrect response to query.
            component (int): index of component.
            query (tuple): index of edge in question.
                    
        Output:
            reponse (int): a binary indicator for edge existence/absence. 
        """
        i, j = query[0], query[1]
        if soft:
            m, s = reliability # unpack mean and std from sof reliability
            m = jnp.abs(m - 1 + self.graphs[component][i, j]) # invert mean depending on edge existence
            get_alpha = lambda m, s: m * (((m * (1-m)) / s**2))
            get_beta = lambda m, s: (1-m) * (((m * (1-m)) / s**2))
            e_ij = random.beta(key=key, a=get_alpha(m,s), b=get_beta(m,s))
        else:
            # Get a uniform sample to determine correctness of response 
            u = random.uniform(key=key, minval=0, maxval=1)
            # if sample is above reliability coef, return wrong answer
            e_ij = cond(u > reliability, 
                        lambda i, j: 1 - self.graphs[component][i, j],
                        lambda i, j: self.graphs[component][i, j],
                        i,j)
            
        return e_ij
    
    
    def stochastically_answer_queries(self, *, key, reliability, component, queries, soft):
        """
        Batch version of stochastically_answer_query.
        """
        key, *batch_subk = random.split(key, num=queries.shape[0]+1)
        
        return vmap(self.stochastically_answer_query, (0, None, None, 0, None))(jnp.array(batch_subk),
                                                                                reliability,
                                                                                component,
                                                                                queries,
                                                                                soft)
    
    
    def update_elicitation_matrix(self, *, E, component, queries, stochastic=False, key=None, reliability=None, soft=False):
        if stochastic and not reliability == 1:
            responses = self.stochastically_answer_queries(key=key, reliability=reliability,
                                                           component=component, queries=queries,soft=soft)
        else:
            responses = self.answer_queries(component=component, queries=queries,soft=soft)
        
        # Update entries in elicitation matrix with responses
        if soft:
            for q, r in zip(queries, responses):
                E = E.at[component, q[0], q[1]].set(r)
        else:
            for q, r in zip(queries, responses):
                if r == 1:
                    E = E.at[component, q[0], q[1]].set(1)
                    E = E.at[component, q[1], q[0]].set(0)
                else:
                    E = E.at[component, q[0], q[1]].set(0)
        
        return E