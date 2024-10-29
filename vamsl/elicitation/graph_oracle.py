import jax.numpy as jnp
from jax import vmap
from jax import random
from jax.lax import cond

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
        
        
    def answer_query(self, component, query):
        """
        Returns the edge existence (0/1) for the edge in the given ground truth graph. 
        
        Args:
            component (int): index of component.
            query (tuple): index of edge in question.
                    
        Output:
            reponse (int): a binary indicator for edge existence/absence. 
        """
        i, j = query[0], query[1]
        
        return self.graphs[component][i, j]
    
    
    def answer_queries(self, *, component, queries):
        """
        Batch version of answer_query.
        """
        return vmap(self.answer_query, (None, 0))(component, queries)
    
    
    def stochastically_answer_query(self, key, reliability, component, query):
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
        
        # Get a uniform sample to determine correctness of response 
        u = random.uniform(key=key, minval=0, maxval=1)
        # if sample is above reliability coef, return wrong answer
        e_ij = cond(u > reliability, 
                    lambda i, j: 1 - self.graphs[component][i, j],
                    lambda i, j: self.graphs[component][i, j],
                    i,j)
        
        return e_ij
    
    
    def stochastically_answer_queries(self, *, key, reliability, component, queries):
        """
        Batch version of stochastically_answer_query.
        """
        key, *batch_subk = random.split(key, num=queries.shape[0]+1)
        
        return vmap(self.stochastically_answer_query, (0, None, None, 0))(jnp.array(batch_subk),
                                                                          reliability, 
                                                                          component,
                                                                          queries)
    
    
    def update_elicitation_matrix(self, *, E, component, queries, stochastic=False, key=None, reliability=None):
        if stochastic and reliability < 1:
            responses = self.stochastically_answer_queries(key=key, reliability=reliability,
                                                           component=component, queries=queries)
        else:
            responses = self.answer_queries(component=component, queries=queries)
        
        # Update entries in elicitation matrix with responses 
        for q, r in zip(queries, responses):
            if r == 1:
                E = E.at[component, q[0], q[1]].set(1)
                E = E.at[component, q[1], q[0]].set(-1)
            else:
                E = E.at[component, q[0], q[1]].set(-1)
        
        return E