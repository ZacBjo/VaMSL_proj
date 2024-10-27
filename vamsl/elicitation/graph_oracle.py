from jax import vmap
from jax import random

class graphOracle:
    def __init__(self, graphs):
        """
            Args:
                graphs: array  of size ```[n_components, n_vars, n_vars]``` 
                        with ground truth adjacency matrices for each component.
        """
        self.graphs = graphs
        
        
    def answer_query(self, component, query):
        i, j = query[0], query[1]
        
        return self.graphs[component][i, j]
    
    
    def answer_queries(self, *, component, queries):
        return vmap(self.answer_query, (None, 0))(component, queries)
    
    
    def stochastically_answer_query(self, key, reliability, component, query):
        i, j = query[0], query[1]
        
        # Get a uniform sample to determine correctness of response 
        u = random.uniform(key=key, minval=0, maxval=1)
        # if sample is lesser than fail rate, return wrong answer
        e_ij = jax.lax.cond(u >= reliability, 
                            lambda i, j: 1 - self.graphs[component][i, j],
                            lambda i, j: self.graphs[component][i, j],
                            i,j)
        
        return e_ij
    
    
    def stochastically_answer_queries(self, *, key, reliability, component, queries):
        keys, *batch_subk = random.split(key, num=queries.shape[0]+1)
        
        return key, vmap(self.stochastically_answer_query, (0, None, None, 0))(jnp.array(batch_subk), 
                                                                               fail_rate, 
                                                                               component, queries)
    
    
    def update_elicitation_matrix(self, *, E, component, queries, stochastic=False, key=None, reliability=None):
        if stochastic and reliability < 100:
            responses = self.stochastically_answer_queries(key=key, reliability=reliability,
                                                           component=component, queries=queries)
        else:
            responses = self.answer_queries(component=component, queries=queries)
        
        # Update entries in elicitation matrix with responses 
        for q, r in zip(queries, responses):
            if r == 1:
                E = E.at[q[0], q[1]].set(1)
                E = E.at[q[1], q[0]].set(-1)
            else:
                E = E.at[q[0], q[1]].set(-1)
        
        return E