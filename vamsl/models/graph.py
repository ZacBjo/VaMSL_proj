import igraph as ig
import random as pyrandom
import jax.numpy as jnp
from jax import random
from jax import vmap
from jax.scipy.stats import dirichlet, beta
from jax.scipy.special import betaln
from jax.scipy.stats import binom

from vamsl.graph_utils import mat_to_graph, graph_to_mat, mat_is_dag
from vamsl.utils.func import zero_diagonal

from jax import debug


class ErdosReniDAGDistribution:
    """
    Randomly oriented Erdos-Reni random graph model with i.i.d. edge probability.
    The pmf is defined as

    :math:`p(G) \\propto p^e (1-p)^{\\binom{d}{2} - e}`

    where :math:`e` denotes the total number of edges in :math:`G`
    and :math:`p` is chosen to satisfy the requirement of sampling ``n_edges_per_node``
    edges per node in expectation.

    Args:
        n_vars (int): number of variables in DAG
        n_edges_per_node (int): number of edges sampled per variable in expectation

    """

    def __init__(self, n_vars, n_edges_per_node=2):
        self.n_vars = n_vars
        self.n_edges = n_edges_per_node * n_vars
        self.p = self.n_edges / ((self.n_vars * (self.n_vars - 1)) / 2)

    def sample_G(self, key, return_mat=False):
        """Samples DAG

        Args:
            key (ndarray): rng
            return_mat (bool): if ``True``, returns adjacency matrix of shape ``[n_vars, n_vars]``

        Returns:
            ``iGraph.graph`` / ``jnp.array``:
            DAG
        """

        key, subk = random.split(key)
        mat = random.bernoulli(subk, p=self.p, shape=(self.n_vars, self.n_vars)).astype(jnp.int32)

        # make DAG by zeroing above diagonal; k=-1 indicates that diagonal is zero too
        dag = jnp.tril(mat, k=-1)

        # randomly permute
        key, subk = random.split(key)
        P = random.permutation(subk, jnp.eye(self.n_vars, dtype=jnp.int32))
        dag_perm = P.T @ dag @ P

        if return_mat:
            return dag_perm
        else:
            g = mat_to_graph(dag_perm)
            return g

    def unnormalized_log_prob_single(self, *, g, j):
        """
        Computes :math:`\\log p(G_j)` up the normalization constant

        Args:
            g (iGraph.graph): graph
            j (int): node index:

        Returns:
            unnormalized log probability of node family of :math:`j`

        """
        parent_edges = g.incident(j, mode='in')
        n_parents = len(parent_edges)
        return n_parents * jnp.log(self.p) + (self.n_vars - n_parents - 1) * jnp.log(1 - self.p)

    def unnormalized_log_prob(self, *, g):
        """
        Computes :math:`\\log p(G)` up the normalization constant

        Args:
            g (iGraph.graph): graph

        Returns:
            unnormalized log probability of :math:`G`

        """
        N = self.n_vars * (self.n_vars - 1) / 2.0
        E = len(g.es)

        return E * jnp.log(self.p) + (N - E) * jnp.log(1 - self.p)

    def unnormalized_log_prob_soft(self, *, soft_g):
        """
        Computes :math:`\\log p(G)` up the normalization constant
        where :math:`G` is the matrix of edge probabilities

        Args:
            soft_g (ndarray): graph adjacency matrix, where entries
                may be probabilities and not necessarily 0 or 1

        Returns:
            unnormalized log probability corresponding to edge probabilities in :math:`G`

        """
        N = self.n_vars * (self.n_vars - 1) / 2.0
        E = soft_g.sum()
        return E * jnp.log(self.p) + (N - E) * jnp.log(1 - self.p)


class ScaleFreeDAGDistribution:
    """
    Randomly-oriented scale-free random graph with power-law degree distribution.
    The pmf is defined as

    :math:`p(G) \\propto \\prod_j (1 + \\text{deg}(j))^{-3}`

    where :math:`\\text{deg}(j)` denotes the in-degree of node :math:`j`

    Args:
        n_vars (int): number of variables in DAG
        n_edges_per_node (int): number of edges sampled per variable

    """

    def __init__(self, n_vars, verbose=False, n_edges_per_node=2):
        self.n_vars = n_vars
        self.n_edges_per_node = n_edges_per_node
        self.verbose = verbose


    def sample_G(self, key, return_mat=False):
        """Samples DAG

        Args:
            key (ndarray): rng
            return_mat (bool): if ``True``, returns adjacency matrix of shape ``[n_vars, n_vars]``

        Returns:
            ``iGraph.graph`` / ``jnp.array``:
            DAG
        """

        pyrandom.seed(int(key.sum()))
        perm = random.permutation(key, self.n_vars).tolist()
        g = ig.Graph.Barabasi(n=self.n_vars, m=self.n_edges_per_node, directed=True).permute_vertices(perm)

        if return_mat:
            return graph_to_mat(g)
        else:
            return g

    def unnormalized_log_prob_single(self, *, g, j):
        """
        Computes :math:`\\log p(G_j)` up the normalization constant

        Args:
            g (iGraph.graph): graph
            j (int): node index:

        Returns:
            unnormalized log probability of node family of :math:`j`

        """
        parent_edges = g.incident(j, mode='in')
        n_parents = len(parent_edges)
        return -3 * jnp.log(1 + n_parents)

    def unnormalized_log_prob(self, *, g):
        """
        Computes :math:`\\log p(G)` up the normalization constant

        Args:
            g (iGraph.graph): graph

        Returns:
            unnormalized log probability of :math:`G`

        """
        return jnp.array([self.unnormalized_log_prob_single(g=g, j=j) for j in range(self.n_vars)]).sum()

    def unnormalized_log_prob_soft(self, *, soft_g):
        """
        Computes :math:`\\log p(G)` up the normalization constant
        where :math:`G` is the matrix of edge probabilities

        Args:
            soft_g (ndarray): graph adjacency matrix, where entries
                may be probabilities and not necessarily 0 or 1

        Returns:
            unnormalized log probability corresponding to edge probabilities in :math:`G`

        """
        soft_indegree = soft_g.sum(0)
        return jnp.sum(-3 * jnp.log(1 + soft_indegree))


class UniformDAGDistributionRejection:
    """
    Naive implementation of a uniform distribution over DAGs via rejection
    sampling. This is efficient up to roughly :math:`d = 5`.
    Properly sampling a uniformly-random DAG is possible but nontrivial
    and not implemented here.

    Args:
        n_vars (int): number of variables in DAG

    """

    def __init__(self, n_vars):
        self.n_vars = n_vars 

    def sample_G(self, key, return_mat=False):
        """Samples DAG

        Args:
            key (ndarray): rng
            return_mat (bool): if ``True``, returns adjacency matrix of shape ``[n_vars, n_vars]``

        Returns:
            ``iGraph.graph`` / ``jnp.array``:
            DAG
        """
        while True:
            key, subk = random.split(key)
            mat = random.bernoulli(subk, p=0.5, shape=(self.n_vars, self.n_vars)).astype(jnp.int32)
            mat = zero_diagonal(mat)

            if mat_is_dag(mat):
                if return_mat:
                    return mat
                else:
                    return mat_to_graph(mat)

    def unnormalized_log_prob_single(self, *, g, j):
        """
        Computes :math:`\\log p(G_j)` up the normalization constant

        Args:
            g (iGraph.graph): graph
            j (int): node index:

        Returns:
            unnormalized log probability of node family of :math:`j`

        """
        return jnp.array(0.0)

    def unnormalized_log_prob(self, *, g):
        """
        Computes :math:`\\log p(G)` up the normalization constant

        Args:
            g (iGraph.graph): graph

        Returns:
            unnormalized log probability of :math:`G`

        """
        return jnp.array(0.0)

    def unnormalized_log_prob_soft(self, *, soft_g):
        """
        Computes :math:`\\log p(G)` up the normalization constant
        where :math:`G` is the matrix of edge probabilities

        Args:
            soft_g (ndarray): graph adjacency matrix, where entries
                may be probabilities and not necessarily 0 or 1

        Returns:
            unnormalized log probability corresponding to edge probabilities in :math:`G`

        """
        return jnp.array(0.0)
    

class DirichletSimilarity:
    """
    Randomly-oriented scale-free random graph with power-law degree distribution.
    The pmf is defined as

    :math:`p(G) \\propto \\prod_j (1 + \\text{deg}(j))^{-3}`

    where :math:`\\text{deg}(j)` denotes the in-degree of node :math:`j`

    Args:
        n_vars (int): number of variables in DAG
        n_edges_per_node (int): number of edges sampled per variable

    """

    def __init__(self):
        pass

    def joint_log_prob_soft(self, *, soft_g, E, N):
        """
        Computes :math:`\\log p(G|E_soft)` where :math:`G` is the matrix of edge probabilities

        Args:
            soft_g (ndarray): graph adjacency matrix, where entries
                may be probabilities and not necessarily 0 or 1
            E (ndarray): elcitied edge probabilities of shape
            N (int): number of observations, scales the impact of expert beliefs on prior probs

        Returns:
            log joint probability corresponding to edge probabilities in :math:`G | E_soft`

        """
        # Mask hard constraints as uniform to remove their influence (hard constraints are applied via p(G|Z,E))
        E = jnp.where(E == 1.0,
                      0.5,
                      jnp.where(E == 0.0, 
                                0.5,
                                # [n_observations, n_vars]
                                E
                               )
                     )
        
        # Add noise to probabilities representing certanties for numeric stability 
        soft_g = jnp.where(soft_g == 1.0,
                           1.0-10**-10,
                           jnp.where(soft_g == 0.0,
                                     10**-10,
                                     # [n_observations, n_vars]
                                     soft_g
                                    )
                          )
        
        
        # Quadratic function for concentration parameter
        # Sets concentration = N when e_ij = 0 or 1
        S = 1 + 4*(30-1)*(E-0.5)**2
        S = 10*jnp.ones_like(E)
        
        # Calculate logpdf of predicting elicited belief given soft graph
        #beta_logpdf = lambda g_ij, e_ij, s_ij: beta.logpdf(x=g_ij, a=s_ij*(0.5+e_ij), b=s_ij*(0.5+(1-e_ij)))
        
        # joint log prob of all edges in soft graph 
        #logjoint = vmap(lambda g_i, e_i, s_i: vmap(beta_logpdf, (0,0, 0))(g_i, e_i, s_i), (0, 0, 0))(soft_g, E, S)
        
        # Calculate logpdf of predicting elicited belief given soft graph
        # joint log prob of all edges in soft graph 
        logjoint = jnp.where(E-0.5 > 0,
                             beta.logpdf(x=soft_g, a=(1+S*(E-0.5)), b=1),
                             beta.logpdf(x=soft_g, a=1, b=(1+S*(0.5-E)))
                            )
        
        
        # Remove influence of uniform beliefs and return log joint 
        return jnp.sum(
                    jnp.where(
                        # [n_observations, n_vars]
                        E == 0.5,  # mask uniform beliefs
                        0.0,
                        # mask hard constraints
                        logjoint
                    )
                )
        
        logjoint = beta.logpdf(x=soft_g, a=E, b=1-E)
        
        return jnp.sum(logjoint)
    
    
class ElicitationBernoulli:
    """
    Elicted prior term based on elictied edge probabilities.  

    :math:`p(Y \\mid G) \\propto p(G \\mid Y)p(Y)`

    where :math:`p(G \\mid Y)` denotes elicitation likelihood
    and :math:`p(Y)` denotes prior over expert edge beliefs.

    Args:
        n_vars (int): number of variables in DAG
        n_edges_per_node (int): number of edges sampled per variable

    """

    def __init__(self):
        pass

    def joint_log_prob(self, *, G, E, alpha_e=1, beta_e=1):
        """
        Computes :math:`\\log p(G|E_soft)` where :math:`G` is the matrix of edge probabilities

        Args:
            G (ndarray): graph adjacency matrix
            E (ndarray): elcitied edge probabilities of shape
            N (int): number of observations, scales the impact of expert beliefs on prior probs

        Returns:
            log joint probability corresponding to edge probabilities in :math:`G | E_soft`

        """
        alphas, betas = alpha_e * jnp.ones_like(E), beta_e * jnp.ones_like(E)
        # Mask hard constraints as uniform to remove their influence (hard constraints are applied via p(G|Z,E))
        E = jnp.where(E == 1.0,
                      0.5,
                      jnp.where(E == 0.0, 
                                0.5,
                                # [n_observations, n_vars]
                                E
                               )
                     )
        
        # get bernoulli probs for edges and complement for absent edges 
        elicited_logliks = jnp.where(G == 1,
                                     jnp.log(E),
                                     jnp.where(G == 0,
                                               jnp.log(1-E),
                                               # [n_observations, n_vars]
                                               jnp.zeros_like(E)
                                              )
                                    )
        
        elicited_logpriors = beta.logpdf(E, a=alphas, b=betas)
        elicited_lognumerator = elicited_logliks+elicited_logpriors
        logdenominator = betaln(alphas + G, betas + (1-G)) - betaln(alphas, betas)
        
        return jnp.sum(jnp.where(E == 0.5, 0, elicited_lognumerator))
        
        # Remove probs from (un)elicited values, default at 0.5
        elicited_logpost = jnp.sum(jnp.where(E == 0.5, 0, elicited_lognumerator)) - jnp.sum(jnp.where(E == 0.5, 0, logdenominator))
        
        return elicited_logpost
    
    
class SoftGraphElicitationBeta:
    """
    Elicted prior term based on elictied edge probabilities.  

    :math:`log p(Y, G(Z)) = log p(G(Z) \\mid Y) + log p(Y)`

    where :math:`p(G \\mid Y)` denotes elicitation likelihood
    and :math:`p(Y)` denotes prior over expert edge beliefs.

    Args:
        n_vars (int): number of variables in DAG
        n_edges_per_node (int): number of edges sampled per variable

    """

    def __init__(self):
        pass

    def joint_unnorm_log_prob(self, *, soft_G, E, alpha_e=1, beta_e=1, std_e_lik=0.3):
        """
        Computes :math:`\\log p(G|E_soft)` where :math:`G` is the matrix of edge probabilities

        Args:
            G (ndarray): graph adjacency matrix
            E (ndarray): elcitied edge probabilities of shape
            N (int): number of observations, scales the impact of expert beliefs on prior probs

        Returns:
            log joint probability corresponding to edge probabilities in :math:`G | E_soft`

        """
        prior_alphas, prior_betas = alpha_e * jnp.ones_like(E), beta_e * jnp.ones_like(E)
        # Mask hard constraints as uniform to remove their influence (hard constraints are applied via p(G|Z,E))
        E = jnp.where(E == 1.0,
                      0.5,
                      jnp.where(E == 0.0, 
                                0.5,
                                # [n_observations, n_vars]
                                E
                               )
                     )
        
        soft_G = jnp.where(soft_G == 1.0,
                      soft_G-10**-7,
                      jnp.where(soft_G == 0.0, 
                                soft_G+10**-7,
                                # [n_observations, n_vars]
                                soft_G
                               )
                     )
        
        #get_alpha = lambda m, s: - m * ((s**2 + m**2 - m) / s**2)
        get_alpha = lambda m, s: (m**2 *(1-m)) / s**2
        #get_alpha = lambda m, s: m * (((m * (1-m)) / s**2)-1)
        #get_beta = lambda m, s: ((s**2+m**2-m)*(m-1))/s**2
        get_beta = lambda m, s: (m * (1-m)**2) / s**2
        #get_beta = lambda m, s: (1-m) * (((m * (1-m)) / s**2)-1)
        
        #debug.print('a: {x}',x=jnp.absolute(get_alpha(E, std_e_lik).mean()))
        #debug.print('b: {x}',x=jnp.absolute(get_beta(E, std_e_lik).mean()))
        
        # get bernoulli probs for edges and complement for absent edges 
        elicited_logliks = beta.logpdf(soft_G, a=get_alpha(E, std_e_lik), b=get_beta(E, std_e_lik))
        #debug.print('soft_g: \n{x}',x=soft_G)
        #debug.print('lik: {x}',x=jnp.absolute(elicited_logliks.mean()))
        
        elicited_logpriors = beta.logpdf(E, a=prior_alphas, b=prior_betas)
        #debug.print('Prio: {x}',x=jnp.absolute(elicited_logpriors.mean()))
        elicited_lognumerator = elicited_logliks+elicited_logpriors
        #debug.print('num: {x}',x=jnp.absolute(elicited_lognumerator.mean()))
        
        # Remove probs from (un)elicited values, default at 0.5
        elicited_unnorm_logpost = jnp.sum(jnp.where(E == 0.5, 0, elicited_lognumerator))
        #debug.print('logpost: {x}',x=elicited_unnorm_logpost)
        
        return elicited_unnorm_logpost
    
    
class Test_elicitation:
    """
    Elicted prior term based on elictied edge probabilities.  

    :math:`log p(Y, G(Z)) = log p(G(Z) \\mid Y) + log p(Y)`

    where :math:`p(G \\mid Y)` denotes elicitation likelihood
    and :math:`p(Y)` denotes prior over expert edge beliefs.

    Args:
        n_vars (int): number of variables in DAG
        n_edges_per_node (int): number of edges sampled per variable

    """

    def __init__(self):
        pass

    def joint_unnorm_log_prob(self, *, soft_G, E, lamda=1, t=1, alpha_e=1, beta_e=1, std_e_lik=0.1):
        """
        Computes :math:`\\log p(G|E_soft)` where :math:`G` is the matrix of edge probabilities

        Args:
            G (ndarray): graph adjacency matrix
            E (ndarray): elcitied edge probabilities of shape
            N (int): number of observations, scales the impact of expert beliefs on prior probs

        Returns:
            log joint probability corresponding to edge probabilities in :math:`G | E_soft`

        """
        prior_alphas, prior_betas = alpha_e * jnp.ones_like(E), beta_e * jnp.ones_like(E)
        # Mask hard constraints as uniform to remove their influence (hard constraints are applied via p(G|Z,E))
        E = jnp.where(E == 1.0,
                      0.5,
                      jnp.where(E == 0.0, 
                                0.5,
                                # [n_observations, n_vars]
                                E
                               )
                     )
        
        err = 10**-7
        soft_G = jnp.where(soft_G == 1.0,
                      soft_G - err,
                      jnp.where(soft_G == 0.0, 
                                soft_G + err,
                                # [n_observations, n_vars]
                                soft_G
                               )
                     )
        
        
        if isinstance(lamda, int):
            N = lamda
            linear_concentration, inv_t = True, 1
        elif len(lamda) == 2:
            N, concentration_function = lamda
            linear_concentration = True if concentration_function == 'linear' else False
            inv_t = 1
        else:
            N, concentration_function, schedule = lamda
            linear_concentration = True if concentration_function == 'linear' else False
            inv_t = jnp.max(jnp.array([schedule+1 / (t+1), 1]))
        
        linear = lambda E, N: 1 + 2*(jnp.abs(E-0.5)*(N-1)) # Linear function for concentration parameter
        quadratic = lambda E, N: 1 + 4*(N-1)*(E-0.5)**2 # Quadratic function for concentration parameter
        concentrations = linear(E, N) if linear_concentration else quadratic(E, N)
        
        elicited_logliks = binom.logpmf(k = inv_t * concentrations * soft_G,
                                        n = inv_t * concentrations * (1-soft_G),
                                        p = E)
        
        #debug.print('soft_g: \n{x}',x=soft_G)
        #debug.print('concentrations: \n{x}',x=inv_t * concentrations * (soft_G))
        #debug.print('not concentrations: \n{x}',x=inv_t * concentrations * (1-soft_G))
        #debug.print('inv_t: \n{x}',x=inv_t)
        #debug.print('lik: {x}',x=jnp.absolute(elicited_logliks.mean()))
        elicited_logpriors = beta.logpdf(E, a=prior_alphas, b=prior_betas)
        
        #debug.print('Prio: {x}',x=jnp.absolute(elicited_logpriors.mean()))
        elicited_lognumerator = elicited_logliks+elicited_logpriors
        #debug.print('num: {x}',x=jnp.absolute(elicited_lognumerator.mean()))
        
        # Remove probs from (un)elicited values, default at 0.5
        elicited_unnorm_logpost = jnp.sum(jnp.where(E == 0.5, 0, elicited_lognumerator))
        #debug.print('logpost: {x}',x=elicited_unnorm_logpost)
        
        return elicited_unnorm_logpost