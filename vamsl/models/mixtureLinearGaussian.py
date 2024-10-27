import jax.numpy as jnp
from jax import random, vmap
from jax.scipy.stats import norm as jax_normal
from jax.scipy.special import gammaln
from vamsl.utils.func import _slogdet_jax
from jax import debug
from jax.scipy.special import logsumexp


class MixtureLinearGaussian:
    """
    Linear Gaussian BN model corresponding to linear structural equation model (SEM) with additive Gaussian noise.

    Each variable distributed as Gaussian with mean being the linear combination of its parents
    weighted by a Gaussian parameter vector (i.e., with Gaussian-valued edges).
    The noise variance at each node is equal by default, which implies the causal structure is identifiable.

    Args:
        n_vars (int): number of variables (nodes in the graph)
        obs_noise (float, optional): variance of additive observation noise at nodes
        mean_edge (float, optional): mean of Gaussian edge weight
        sig_edge (float, optional): std dev of Gaussian edge weight
        min_edge (float, optional): minimum linear effect of parent on child

    """

    def __init__(self, *, n_vars, obs_noise=0.1, mean_edge=0.0, sig_edge=1.0, min_edge=0.5):
        self.n_vars = n_vars
        self.obs_noise = obs_noise
        self.mean_edge = mean_edge
        self.sig_edge = sig_edge
        self.min_edge = min_edge

        self.no_interv_targets = jnp.zeros(self.n_vars).astype(bool)


    def get_theta_shape(self, *, n_vars):
        """Returns tree shape of the parameters of the linear model

        Args:
            n_vars (int): number of variables in model

        Returns:
            PyTree of parameter shape
        """
        return jnp.array((n_vars, n_vars))


    def sample_parameters(self, *, key, n_vars, n_particles=0, batch_size=0):
        """Samples batch of random parameters given dimensions of graph from :math:`p(\\Theta | G)`

        Args:
            key (ndarray): rng
            n_vars (int): number of variables in BN
            n_particles (int): number of parameter particles sampled
            batch_size (int): number of batches of particles being sampled

        Returns:
            Parameters ``theta`` of shape ``[batch_size, n_particles, n_vars, n_vars]``, dropping dimensions equal to 0
        """
        shape = (batch_size, n_particles, *self.get_theta_shape(n_vars=n_vars))
        theta = self.mean_edge + self.sig_edge * random.normal(key, shape=tuple(d for d in shape if d != 0))
        theta += jnp.sign(theta) * self.min_edge
        return theta


    def sample_obs(self, *, key, n_samples, g, theta, toporder=None, interv=None):
        """Samples ``n_samples`` observations given graph ``g`` and parameters ``theta``

        Args:
            key (ndarray): rng
            n_samples (int): number of samples
            g (igraph.Graph): graph
            theta (Any): parameters
            interv (dict): intervention specification of the form ``{intervened node : clamp value}``

        Returns:
            observation matrix of shape ``[n_samples, n_vars]``
        """
        if interv is None:
            interv = {}
        if toporder is None:
            toporder = g.topological_sorting()

        x = jnp.zeros((n_samples, len(g.vs)))

        key, subk = random.split(key)
        z = jnp.sqrt(self.obs_noise) * random.normal(subk, shape=(n_samples, len(g.vs)))

        # ancestral sampling
        for j in toporder:

            # intervention
            if j in interv.keys():
                x = x.at[:, j].set(interv[j])
                continue

            # regular ancestral sampling
            parent_edges = g.incident(j, mode='in')
            parents = list(g.es[e].source for e in parent_edges)

            if parents:
                mean = x[:, jnp.array(parents)] @ theta[jnp.array(parents), j]
                x = x.at[:, j].set(mean + z[:, j])

            else:
                x = x.at[:, j].set(z[:, j])

        return x

    """
    The following functions need to be functionally pure and @jit-able
    """

    def log_prob_parameters(self, *, theta, g):
        """Computes parameter prior :math:`\\log p(\\Theta | G)``
        In this model, the parameter prior is Gaussian.

        Arguments:
            theta (ndarray): parameter matrix of shape ``[n_vars, n_vars]``
            g (ndarray): graph adjacency matrix of shape ``[n_vars, n_vars]``

        Returns:
            log prob
        """
        return jnp.sum(g * jax_normal.logpdf(x=theta, loc=self.mean_edge, scale=self.sig_edge))


    def log_likelihood(self, *, x, c, theta, g, interv_targets):
        """Computes likelihood :math:`p(D | G, \\Theta)`.
        In this model, the noise per observation and node is additive and Gaussian.

        Arguments:
            x (ndarray): observations of shape ``[n_observations, n_vars]``
            c (ndarray): responsibilities of shape ``[n_observations, 1]``
            theta (ndarray): parameters of shape ``[n_vars, n_vars]``
            g (ndarray): graph adjacency matrix of shape ``[n_vars, n_vars]``
            interv_targets (ndarray): binary intervention indicator vector of shape ``[n_observations, n_vars]``

        Returns:
            log prob
        """
        assert x.shape == interv_targets.shape
        
        c_mask = c.reshape(-1,1).repeat(x.shape[1], 1)
        assert x.shape == c_mask.shape
        
        # sum scores for all nodes and data
        log_sum =  jnp.sum(
            jnp.where(
                # [n_observations, n_vars]
                interv_targets,
                0.0,
                # [n_observations, n_vars]
                c_mask * jax_normal.logpdf(x=x, loc=x @ (g * theta), scale=jnp.sqrt(self.obs_noise))
            )
        )
        
        return log_sum

    """
    Distributions used by DiBS for inference:  prior and joint likelihood 
    """

    def interventional_log_joint_prob(self, g, theta, x, c, interv_targets, rng):
        """Computes interventional joint likelihood :math:`\\log p(\\Theta, D | G)``

        Arguments:
            g (ndarray): graph adjacency matrix of shape ``[n_vars, n_vars]``
            theta (ndarray): parameter matrix of shape ``[n_vars, n_vars]``
            x (ndarray): observational data of shape ``[n_observations, n_vars]``
            c (ndarray): responsibilities of shape ``[n_observations, 1]``
            interv_targets (ndarray): indicator mask of interventions of shape ``[n_observations, n_vars]``
            rng (ndarray): rng; skeleton for minibatching (TBD)

        Returns:
            log prob of shape ``[1,]``
        """
        log_prob_theta = self.log_prob_parameters(g=g, theta=theta)
        log_likelihood = self.log_likelihood(g=g, theta=theta, x=x, c=c, interv_targets=interv_targets)
        return log_prob_theta + log_likelihood

