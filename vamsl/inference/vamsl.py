import numpy as np
import jax
import jax.numpy as jnp
from jax import vmap, random, grad
from jax.nn import sigmoid, log_sigmoid
from jax.scipy.special import logsumexp
from scipy.optimize import linear_sum_assignment
from jax.scipy.special import digamma
from jax.random import categorical

from vamsl.inference import MixtureJointDiBS
from vamsl.metrics import ParticleDistribution, neg_ave_log_likelihood 
from vamsl.kernel import AdditiveFrobeniusSEKernel, JointAdditiveFrobeniusSEKernel
from vamsl.utils.func import stable_softmax
from vamsl.graph_utils import elwise_acyclic_constr_nograd
from vamsl.utils.tree import tree_mul, tree_select
from vamsl.utils.func import expand_by, zero_diagonal


class VaMSL(MixtureJointDiBS):
    def __init__(self, *,
                 x,
                 q_z=None,
                 q_theta=None,
                 q_c=None,
                 q_pi = None,
                 sf_baselines = None,
                 E = None,
                 n_particles = None,
                 graph_model,
                 mixture_likelihood_model,
                 component_likelihood_model,
                 interv_mask=None,
                 kernel=JointAdditiveFrobeniusSEKernel,
                 kernel_param=None,
                 optimizer="rmsprop",
                 optimizer_param=None,
                 alpha_linear=0.05,
                 beta_linear=1.0,
                 tau=1.0,
                 n_grad_mc_samples=128,
                 n_acyclicity_mc_samples=32,
                 grad_estimator_z="reparam",
                 score_function_baseline=0.0,
                 latent_prior_std=None,
                 verbose=False):
        self.q_z = q_z
        self.q_theta = q_theta
        self.q_c = q_c
        self.q_pi = q_pi
        self.E = E
        self.n_particles = n_particles
        self.sf_baselines = sf_baselines

        self.component_likelihood_model = component_likelihood_model
        # functions for post-hoc likelihood evaluations
        self.eltwise_component_log_likelihood_observ = vmap(lambda g, theta, x_ho: 
            component_likelihood_model.interventional_log_joint_prob(g, theta, x_ho, jnp.zeros_like(x_ho), None), (0, 0, None), 0)
        self.eltwise_component_log_likelihood_interv = vmap(lambda g, theta, x_ho, interv_msk_ho:
            component_likelihood_model.interventional_log_joint_prob(g, theta, x_ho, interv_msk_ho, None), (0, 0, None, None), 0)

        # init MDiBS SVGD superclass methods
        super().__init__(
            x=x,
            graph_model=graph_model,
            likelihood_model=mixture_likelihood_model,
            interv_mask=interv_mask,
            kernel=kernel,
            kernel_param=kernel_param,
            optimizer=optimizer,
            optimizer_param=optimizer_param,
            alpha_linear=alpha_linear,
            beta_linear=beta_linear,
            tau=tau,
            n_grad_mc_samples=n_grad_mc_samples,
            n_acyclicity_mc_samples=n_acyclicity_mc_samples,
            grad_estimator_z=grad_estimator_z,
            score_function_baseline=score_function_baseline,
            latent_prior_std=latent_prior_std,
            verbose=verbose
        )
        
        
    def initialize_posteriors(self, *, key, init_q_c, n_particles, E=None, linear=True):
        self.n_particles = n_particles
        self.q_c = jnp.log(init_q_c)
        self.update_mixing_weigths()
        n_components = self.q_c.shape[1]
        self.q_z, self.q_theta = self._sample_intial_component_particles(key=key,
                                                                         n_components=n_components, 
                                                                         n_particles=n_particles,
                                                                         n_dim=self.n_vars,
                                                                         linear=linear)
        
        self.sf_baselines = jnp.zeros((n_components, self.n_particles))
        
        if E or self.E:
            self.E = E
        else:
            self.E = jnp.zeros((n_components, self.n_vars, self.n_vars))


    def sample_assignments(self, key):
        assignments = categorical(key=key, logits=self.q_c)
        return jax.nn.one_hot(assignments, num_classes=self.q_c.shape[1])   

    
    def update_particle_posteriors(self, *, key, steps, callback=None, callback_every=None, linear=True):
        key, subk = random.split(key)
        # Sample observation assignments
        cs = self.sample_assignments(subk)
        
        # Sample new variational posteriors for graphs and parameters
        self.q_z, self.q_theta, self.sf_baselines = self.sample(key=key, n_particles=self.n_particles, 
                                                                steps=steps, callback=callback, 
                                                                callback_every=callback_every, 
                                                                q_c=cs,
                                                                init_q_z=self.q_z, 
                                                                init_q_theta=self.q_theta, 
                                                                init_sf_baselines=self.sf_baselines,
                                                                E=self.E,
                                                                linear=linear)

    
    def update_mixing_weigths(self):
        self.q_pi = jnp.exp(logsumexp(self.q_c, axis=0)) + 10**-6 # constant for numeric stability
        

    def compute_responsibility(self, x_n, dist_k, pi_k):
        expected_log_lik = - neg_ave_log_likelihood(dist=dist_k,
                                                    eltwise_log_likelihood=self.eltwise_component_log_likelihood_observ,
                                                    x=jnp.array(x_n.reshape((1,-1))))
                                                      
        return expected_log_lik + digamma(pi_k) - digamma(jnp.sum(self.q_pi))
    
    
    def update_responsibilities_and_weights(self):
        N = self.x.shape[0]
        K = self.q_c.shape[1]
        
        # Get graphs for MC-estimating expected data log likelihoods
        # [n_components, n_particles, n_vars, n_vars]
        component_gs = self.compwise_particle_to_g_lim(self.q_z, self.E)
        
        # Get particle distributions for each component (list comprehension since get_empirical is impure)
        # [n_components, ParticleDistribution]
        component_dists = [self.get_empirical(component_gs[k], self.q_theta[k]) for k in range(K)]
        
        # Get unnormalized repsonsiibilities for components (list comprehension since component_dists can differ in size)
        # [n_observations, n_components]
        unnorm_responsibilities = jnp.transpose(jnp.array([vmap(self.compute_responsibility, (0, None, None))(self.x, 
                                                                                                              component_dists[k], 
                                                                                                              self.q_pi[k]) for k in range(K)]))
        
        unnorm_log_sum = logsumexp(unnorm_responsibilities, axis=1)
        def log_normalize(unorm_log_c_n, unnorm_log_sum):
            return unorm_log_c_n - unnorm_log_sum
            
        # Get softmax-normalized component responsibilities
        # [n_observations, n_components]
        log_responsibilities = vmap(log_normalize, (0, 0))(unnorm_responsibilities, unnorm_log_sum)
        
        self.q_c = log_responsibilities
        self.update_mixing_weigths()
        
        
    def visualize_posteriors(self, callback, steps=0):
        for k in range(self.q_c.shape[1]):
            callback(dibs=self,
                     t=steps,
                     k=k+1,
                     zs=self.q_z[k],
                     thetas=self.q_theta[k],
                     E_k=self.E[k],
                     ipython=True if k < n_components-1 else False)
    
    
    def component_classification_accuracy(self, *, component, c_k_targets):
        # TODO
        pass

    
    def identify_ordering(self, *, mixture_data):
        """
        Identify component indices based on n_componets number of ground truth graphs
        
        Args:
            mixture_data (ndarray): ndarray of shape [n_components, n_observations, n_vars] with data for each component.
            
        Outputs:
            order (array): array of with order of components
        """
        # [n_components, n_particles, n_vars, n_vars]
        component_gs = self.compwise_particle_to_g_lim(self.q_z, self.E)
        
        # Get particle distributions for each component (list comprhension since get_empirical is impure)
        # [n_components, ParticleDistribution]
        component_dists = [self.get_empirical(component_gs[k], self.q_theta[k]) for k in range(self.q_c.shape[1])]
        
        # expected log lik
        e_log_lik = lambda dist, x: -neg_ave_log_likelihood(dist=dist,
                                                            eltwise_log_likelihood=self.eltwise_component_log_likelihood_observ,
                                                            x=x)
        
        # Calculate log likelihood of all components for data
        cost_matrix = [[e_log_lik(dist, data) for data in mixture_data] for dist in component_dists]
        
        # use Hungaraian algorithm to find optimal allocation
        assignments = linear_sum_assignment(cost_matrix)
        
        # Return list of optimal allocations as order of components
        return assignments[1]
    
    
    #
    # Getters and setters
    #
    
    def get_posteriors(self):
        return self.q_z, self.q_theta, self.q_c, self.q_pi

    
    def set_E(self, E):
        self.E = E

    
    def get_E(self):
        return self.E
    
    #
    # Overriding functions for conditional graph information (prior and elicited)  
    #
    
    def particle_to_g_lim(self, z, E_k):
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
    
    
    def eltwise_particle_to_g_lim(self, q_z_k, E_k):
        return vmap(self.particle_to_g_lim, (0, None))(q_z_k, E_k)
    
    
    def compwise_particle_to_g_lim(self, q_z, E):
        return vmap(self.eltwise_particle_to_g_lim, (0, 0))(q_z, E)
    
    
    #@override
    def particle_to_soft_graph(self, z, eps, t, E_k):
        """
        Gumbel-softmax / concrete distribution using Logistic(0,1) samples ``eps``

        Args:
            z (ndarray): a single latent tensor :math:`Z` of shape ``[d, k, 2]```
            eps (ndarray): random i.i.d. Logistic(0,1) noise  of shape ``[d, d]``
            t (int): step

        Returns:
            Gumbel-softmax sample of adjacency matrix [d, d]
        """
        scores = jnp.einsum('...ik,...jk->...ij', z[..., 0], z[..., 1])

        # soft reparameterization using gumbel-softmax/concrete distribution
        # eps ~ Logistic(0,1)
        #soft_graph = sigmoid(self.tau * (eps + self.alpha(t) * scores)) #original
        soft_graph = (E_k**2)*((1+E_k)/2) + (1-E_k**2)*sigmoid(self.tau * (eps + self.alpha(t) * scores)) # TEST

        # mask diagonal since it is explicitly not modeled
        return zero_diagonal(soft_graph)

    
    #@override
    def edge_probs(self, z, t, E_k):
        """
        Edge probabilities encoded by latent representation

        Args:
            z (ndarray): latent tensors :math:`Z`  ``[..., d, k, 2]``
            t (int): step

        Returns:
            edge probabilities of shape ``[..., d, d]``
        """
        u, v = z[..., 0], z[..., 1]
        scores = jnp.einsum('...ik,...jk->...ij', u, v)
        #probs = sigmoid(self.alpha(t) * scores) # original
        probs = (E_k**2)*((1+E_k)/2) + (1-E_k**2)*sigmoid(self.alpha(t) * scores) # TEST

        # mask diagonal since it is explicitly not modeled
        return zero_diagonal(probs)

    
    #@override
    def edge_log_probs(self, z, t, E_k):
        """
        Edge log probabilities encoded by latent representation

        Args:
            z (ndarray): latent tensors :math:`Z` ``[..., d, k, 2]``
            t (int): step

        Returns:
            tuple of tensors ``[..., d, d], [..., d, d]`` corresponding to ``log(p)`` and ``log(1-p)``
        """ 
        u, v = z[..., 0], z[..., 1]
        scores = jnp.einsum('...ik,...jk->...ij', u, v)
        
        #log_probs, log_probs_neg = log_sigmoid(self.alpha(t) * scores), log_sigmoid(self.alpha(t) * -scores) #original
        log_probs = jnp.log((E_k**2)*((1+E_k)/2) + (1-E_k**2)*sigmoid(self.alpha(t) * scores))
        log_probs_neg = jnp.log((E_k**2)*((1+E_k)/2) + (1-E_k**2)*sigmoid(self.alpha(t) * -scores))
        
        # mask diagonal since it is explicitly not modeled
        # NOTE: this is not technically log(p), but the way `edge_log_probs_` is used, this is correct
        return zero_diagonal(log_probs), zero_diagonal(log_probs_neg)
