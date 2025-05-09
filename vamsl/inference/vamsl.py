import numpy as np
import jax
import jax.numpy as jnp
from jax import vmap, random, grad
from jax.nn import sigmoid, log_sigmoid
from jax.scipy.special import logsumexp
from scipy.optimize import linear_sum_assignment
from jax.scipy.special import digamma
from jax.random import categorical
from sklearn.metrics import confusion_matrix

from vamsl.inference import MixtureJointDiBS
from vamsl.metrics import ParticleDistribution, neg_ave_log_likelihood 
from vamsl.kernel import AdditiveFrobeniusSEKernel, JointAdditiveFrobeniusSEKernel
from vamsl.utils.func import stable_softmax
from vamsl.graph_utils import elwise_acyclic_constr_nograd
from vamsl.utils.tree import tree_mul, tree_select
from vamsl.utils.func import expand_by, zero_diagonal
from vamsl.models.graph import DirichletSimilarity

from jax import debug

from vamsl.metrics import expected_log_likelihood


class VaMSL(MixtureJointDiBS):
    def __init__(self, *,
                 x,
                 q_z=None,
                 q_theta=None,
                 log_q_c=None,
                 q_pi = None,
                 alphas = None,
                 sf_baselines = None,
                 E = None,
                 E_particles=None,
                 n_particles = None,
                 graph_model,
                 elicitation_graph_model=DirichletSimilarity(),
                 mixture_likelihood_model,
                 component_likelihood_model,
                 interv_mask=None,
                 kernel=JointAdditiveFrobeniusSEKernel,
                 kernel_param=None,
                 optimizer="rmsprop",
                 optimizer_param=None,
                 alpha_linear=0.05, # referred to as \omega in VaMSL paper
                 beta_linear=1.0,
                 tau=1.0,
                 lamda=0.0,
                 elicitation_prior=None, #'soft', 'hard' or None
                 n_grad_mc_samples=128,
                 n_acyclicity_mc_samples=32,
                 n_mixture_grad_mc_samples=32,
                 n_elicitation_grad_mc_samples=1,
                 stochastic_elicitation=False,
                 grad_estimator_z="reparam",
                 score_function_baseline=0.0,
                 latent_prior_std=None,
                 parallell_computation=True,
                 verbose=False):
        self.q_z = q_z
        self.q_theta = q_theta
        self.log_q_c = log_q_c
        self.q_pi = q_pi
        self.alphas = alphas
        self.E = E
        self.E_particles=E_particles
        self.stochastic_elicitation = stochastic_elicitation
        self.n_particles = n_particles
        self.sf_baselines = sf_baselines

        self.component_likelihood_model = component_likelihood_model
        # functions for post-hoc likelihood evaluations
        # Likelihood p(x | G, \Theta)
        self.eltwise_component_log_likelihood_observ = vmap(lambda g, theta, x_ho: 
            component_likelihood_model.log_likelihood(g=g, theta=theta, x=x_ho, interv_targets=jnp.zeros_like(x_ho)), (0, 0, None), 0)
        self.eltwise_component_log_likelihood_interv = vmap(lambda g, theta, x_ho, interv_msk_ho:
            component_likelihood_model.log_likelihoodb(g=g, theta=theta, x=x_ho, interv_targets=interv_msk_ho), (0, 0, None, None), 0)
        # Joint likelihood p(x, \Theta | G)
        self.eltwise_component_log_joint_likelihood_observ = vmap(lambda g, theta, x_ho: 
            component_likelihood_model.interventional_log_joint_prob(g, theta, x_ho, jnp.zeros_like(x_ho), None), (0, 0, None), 0)
        self.eltwise_component_log_joint_likelihood_interv = vmap(lambda g, theta, x_ho, interv_msk_ho:
            component_likelihood_model.interventional_log_joint_prob(g, theta, x_ho, interv_msk_ho, None), (0, 0, None, None), 0)

        # init VaMSL SVGD superclass methods
        super().__init__(
            x=x,
            graph_model=graph_model,
            elicitation_graph_model=elicitation_graph_model,
            likelihood_model=mixture_likelihood_model,
            interv_mask=interv_mask,
            kernel=kernel,
            kernel_param=kernel_param,
            optimizer=optimizer,
            optimizer_param=optimizer_param,
            alpha_linear=alpha_linear,
            beta_linear=beta_linear,
            tau=tau,
            lamda=lamda,
            elicitation_prior=elicitation_prior,
            n_grad_mc_samples=n_grad_mc_samples,
            n_acyclicity_mc_samples=n_acyclicity_mc_samples,
            n_mixture_grad_mc_samples=n_mixture_grad_mc_samples,
            n_elicitation_grad_mc_samples=n_elicitation_grad_mc_samples,
            grad_estimator_z=grad_estimator_z,
            score_function_baseline=score_function_baseline,
            latent_prior_std=latent_prior_std,
            parallell_computation=parallell_computation,
            verbose=verbose
        )
        
    
    def sample_E_particles(self, *, key=None):
        E_particles = jnp.stack([jnp.repeat(self.E[k, np.newaxis, :, :], self.n_particles, axis=0) for k in range(self.q_z.shape[0])])
        for k in range(self.q_z.shape[0]):
            for p in range(self.n_particles):
                E_k = E_particles[k,p,:,:].astype('float32') # handle hard constraints
                if self.stochastic_elicitation:
                    probs = jnp.where(E_k > 0.5, E_k, 1-E_k)
                    key, subk = random.split(key)
                    # mask edges that haven't been queried
                    trials = jnp.where(E_k == 0.5, 1, random.bernoulli(subk, p=probs, shape=E_k.shape))
                else:
                    trials = jnp.ones_like(E_k)
                E_particles = E_particles.at[k,p,:,:].set(jnp.where(trials, E_k, 0.5))
                    
        return E_particles
        
        
    def initialize_posteriors(self, *, key, n_components, n_particles, init_q_c=None, alphas=None, E=None, E_particles=None, linear=True):
        """
        Initializes variational posteriors for mixing weights q(\pi) and embbedded graph and parameter 
        particles q(Z, \Theta).

        Args:
            key (ndarray): prng key
            n_components (int): Number of components in mixture model
            n_particles (int): Number of SVGD particles per component. 
            init_q_c (ndarray): Initial assignment probabilities of shape ``[n_observations, n_components]``
            alphas (int): Prior hyperparameters for distribution over mixing weights
            E (ndarray): Matrix of elicited hard edge constraints of shape ``[n_components, n_vars, n_vars]``
            linear (boolean): Boolean value for using (non-)linear SCMs as nonlinear parameters require 
                              different datatype for storing. 

        Returns:
            None

        """
        self.n_particles = n_particles
        # Set initial responsibilities
        if init_q_c is None:
            # Default to uniform responsibilities
            uniform_q_c = 1/n_components * jnp.ones((self.x.shape[0], n_components))
            self.log_q_c = jnp.log(uniform_q_c) 
        else:
            self.log_q_c = jnp.log(init_q_c)
            n_components = self.log_q_c.shape[1]
         
        # Set initial mixing weights
        if alphas is None:
            # Default to uniform dirichlet
            self.q_pi = jnp.ones((n_components))
            self.alphas = jnp.ones((n_components))
        else:
            self.q_pi = alphas
            self.alphas = alphas
            
        # Sample initial emmbedded graph and paramter particles 
        key, subk = random.split(key)
        self.q_z, self.q_theta = self._sample_intial_component_particles(key=subk,
                                                                         n_components=n_components, 
                                                                         n_particles=self.n_particles,
                                                                         n_dim=self.n_vars,
                                                                         linear=linear)
        
        self.sf_baselines = jnp.zeros((n_components, self.n_particles)) # not used, but expected as input by DiBS
        # If given, set elicitation matrix
        if E is None:
            self.E = 0.5*jnp.ones((n_components, self.n_vars, self.n_vars))
        else:
            self.E = E
        
        if E_particles is None:
            self.E_particles = self.sample_E_particles(key=key)

            
    def get_component_dists(self, empirical=True, key=None):
        """
        Get particle distributions for components. Either get empirical dist. or mixture dist. as 
        detailed in original DiBS article. 

        Args:
            None

        Returns:
            List of length n_components with ParticleDistribution objects.

        """
        K = self.log_q_c.shape[1]
        # Get graphs for MC-estimating expected data log likelihoods
        # [n_components, n_particles, n_vars, n_vars]
        component_gs = self.compwise_particle_to_g_lim(self.q_z, self.E)
        
        # Get particle distributions for each component (list comprehension since get_empirical is impure)
        # [n_components] of ParticleDistribution objects
        if empirical:
            component_dists = [self.get_empirical(component_gs[k], self.q_theta[k]) for k in range(K)]
        else:
            cs = self.sample_assignments(key)
            component_dists = [self.get_mixture(component_gs[k], self.q_theta[k], cs[k]) for k in range(K)]
        
        return component_dists
    
    #
    # q(z, \Theta) CAVI update
    #

    def sample_assignments(self, key):
        """
        Samples component assignments based on variational assignment distribution q(c).

        Args:
            key (ndarray): prng key

        Returns:
            one_hot_assignments (ndarray): One hot assignment vectors of shape ``[n_observations, n_components]``

        """
        n_mixture_grad_mc_samples = self.n_mixture_grad_mc_samples
        # Sample assignments from categorical distribution parametrized by responsibilities
        # [n_mc_grad_samples, n_observations]
        assignments = random.categorical(key=key, logits=self.log_q_c, shape=(n_mixture_grad_mc_samples, self.log_q_c.shape[0]))

        # Construct matrix of one-hot vectors from sampled assignmnets
        # [n_mixture_grad_mc_samples, n_observations, n_components]
        one_hot_assignments = jax.nn.one_hot(assignments, num_classes=self.log_q_c.shape[1])
        
        # Switch leading dimensions
        # [n_components, n_mixture_grad_mc_samples, n_observations]
        one_hot_assignments = jnp.array([one_hot_assignments[...,col] for col in range(self.log_q_c.shape[1])])
        
        return one_hot_assignments

    
    def update_particle_posteriors(self, *, key, steps, callback=None, callback_every=None, linear=True):
        """
        Initializes variational posteriors for mixing weights q(\pi) and embbedded graph and paramter 
        particles q(Z, \Theta). If no  

        Args:
            key (ndarray): prng key
            steps (int): number of SVGD steps performed 
            callback: function to be called every ``callback_every`` steps of SVGD.
            callback_every: if ``None``, ``callback`` is only called after particle updates have finished
            linear (boolean): Boolean value for using (non-)linear SCMs as nonlinear parameters require 
                              different datatype for storing. 

        Returns:
            None

        """
        key, subk = random.split(key)
        # Sample observation assignments
        cs = self.sample_assignments(subk)
        # Sample new variational posteriors for graphs and parameters
        self.q_z, self.q_theta, self.sf_baselines = self.sample(key=key, 
                                                                n_particles=self.n_particles, 
                                                                steps=steps, 
                                                                callback=callback, 
                                                                callback_every=callback_every, 
                                                                cs=cs,
                                                                init_q_z=self.q_z, 
                                                                init_q_theta=self.q_theta, 
                                                                init_sf_baselines=self.sf_baselines,
                                                                E=self.E,
                                                                E_stack=self.E_particles,
                                                                linear=linear)

    
    #
    # q(\pi) CAVI update
    #
    
    def update_mixing_weigths(self):
        """
        CAVI update for variational mixing weight distribution q(\pi).

        Args:
            None 

        Returns:
            None

        """
        self.q_pi = self.alphas + jnp.exp(logsumexp(self.log_q_c, axis=0)) + 10**-6 # constant for numeric stability
        

    #
    # q(c) CAVI update
    #
    
    def compute_log_responsibility(self, x_n, dist_k, pi_k):
        """
        Compute component responsibility for single observation.

        Args:
            x_n (ndarray): Single observation of shape ``[]``
            dist_k (:class:`dibs.metrics.ParticleDistribution`): particle distribution for component k
            pi_k (float): Mixing weight for component k

        Returns:
            log_responsibility (float): log responsibility for component k with respect to observation x_n

        """
        expected_log_lik = - neg_ave_log_likelihood(dist=dist_k,
                                                    eltwise_log_likelihood=self.eltwise_component_log_likelihood_observ,
                                                    x=x_n)
        
        ## TEST
        #expected_log_lik = - neg_log_posterior_predictive_density(key=random.PRNGKey(1), q_pi=jnp.ones((1,)), dists=[dist_k], eltwise_log_likelihood=self.eltwise_component_log_likelihood_observ,x=x_n.reshape(1,-1),n_mixing_mc_samples=1)
        #expected_log_lik = expected_log_likelihood(x=x_n, dist=dist_k, eltwise_log_likelihood=self.eltwise_component_log_likelihood_observ)
        ## END TEST
        
        log_responsibility = expected_log_lik + digamma(pi_k) - digamma(jnp.sum(self.q_pi))
                                                      
        return log_responsibility
    
        
    def compute_log_responsibilities(self, *, x):
        """
        CAVI updates for variational assigment and mixing weight distributions.

        Args:
            x: N new observations of shape ```[N, n_vars]```

        Returns:
            log_responsibilities: responsibilities of shape ```[N, n_components]```

        """
        K = self.log_q_c.shape[1] # number of components
        # Get particle distributions for each component
        # [n_components] of ParticleDistribution objects
        component_dists = self.get_component_dists()
        
        # Get unnormalized repsonsibilities for components (list comprehension since component_dists can differ in size)
        # [n_observations, n_components]
        unnorm_responsibilities = jnp.transpose(jnp.array([vmap(self.compute_log_responsibility, (0, None, None))(x, 
                                                                                                                  component_dists[k], 
                                                                                                                  self.q_pi[k]) for k in range(K)]))
        unnorm_log_sum = logsumexp(unnorm_responsibilities, axis=1)
        log_normalize = lambda unorm_log_c_n, unnorm_log_sum: unorm_log_c_n - unnorm_log_sum
        # Get softmax-normalized component responsibilities
        # [n_observations, n_components]
        log_responsibilities = vmap(log_normalize, (0, 0))(unnorm_responsibilities, unnorm_log_sum)
        
        return log_responsibilities
    
    
    def update_responsibilities_and_weights(self):
        """
        CAVI updates for variational assigment and mixing weight distributions.
        Responsibilities and weights are updated together to avoid going out of sync. 

        Args:
            None

        Returns:
            None

        """                    
        # Update variational distributions for responsibilities and mixing weights 
        self.log_q_c =  self.compute_log_responsibilities(x=self.x)
        self.update_mixing_weigths()
    
    #
    # Getters and setters
    #
    
    def set_posteriors(self, *, q_z, q_theta, log_q_c, q_pi):
        """
        Sets all variational posteriors to inputs.

        Args:
            q_z (ndarray): Embedded graph particles of shape ``[n_components, n_particles, n_vars, l_dim, 2]``
            q_theta (Any): PyTree with leading dim ``n_particles``
            log_q_c (ndarray): Component log responsibilities of shape ``[n_observations, n_components]`` 
            q_pi (ndarray): Mixing weights of shape ``[n_components,]``

        Returns:
            None 

        """
        self.q_z, self.q_theta, self.log_q_c, self.q_pi = q_z, q_theta, log_q_c, q_pi
    
    
    def get_posteriors(self):
        """
        Returns all variational posteriors.

        Args:
            None

        Returns:
            q_z (ndarray): Embedded graph particles of shape ``[n_components, n_particles, n_vars, l_dim, 2]``
            q_theta (Any): PyTree with leading dim ``n_particles``
            log_q_c (ndarray): Component log responsibilities of shape ``[n_observations, n_components]`` 
            q_pi (ndarray): Mixing weights of shape ``[n_components,]`` 

        """
        return self.q_z, self.q_theta, self.log_q_c, self.q_pi

    
    def set_E(self, E, key=None):
        """
        Sets elicitation matrix.

        Args:
            E (ndarray): Elicited component-wise hard edge constraints ``[n_components, n_vars, n_vars]`` 

        Returns:
            None

        """ 
        self.E = E
        if not key is None:
            self.E_particles = self.sample_E_particles(key=key)
        elif not self.stochastic_elicitation:
            self.E_particles = self.sample_E_particles(key=key)

    
    def get_E(self):
        """
        Returns elicitation matrix.

        Args:
            None

        Returns:
            E (ndarray): Elicited component-wise hard edge constraints ``[n_components, n_vars, n_vars]`` 

        """
        return self.E
    
    
    def get_E_particles(self):
        return self.E_particles
    
    
    def set_lamda(self, lamda):
        self.lamda = lamda
        
        
    def set_n_mixture_grad_mc_samples(self, n_mixture_grad_mc_samples):
        self.n_mixture_grad_mc_samples = n_mixture_grad_mc_samples
    
    
    #
    # Overriding functions for conditional graph information (prior and elicited)  
    #
    
    def hard_constraint_mask(self, probs, E_k):
        return jnp.where(E_k==1, 1, jnp.where(E_k==0, 0, probs))
    
    
    def hard_constraint_mask_zeros(self, probs, E_k):
        return jnp.where(E_k==1, 0, jnp.where(E_k==0, 0, probs))
    
    
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
        g_samples = self.hard_constraint_mask((scores > 0).astype(jnp.int32), E_k)

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
        soft_graph = self.hard_constraint_mask(sigmoid(self.tau * (eps + self.alpha(t) * scores)), E_k)

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
        probs = self.hard_constraint_mask(sigmoid(self.alpha(t) * scores), E_k)
        
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
        
        log_probs, log_probs_neg = log_sigmoid(self.alpha(t) * scores), log_sigmoid(self.alpha(t) * -scores)
        
        # mask diagonal since it is explicitly not modeled
        # NOTE: this is not technically log(p), but the way `edge_log_probs_` is used, this is correct
        return zero_diagonal(log_probs), zero_diagonal(log_probs_neg)
    
    
    def latent_log_prob(self, single_g, single_z, E_k, t):
        """
        Log likelihood of generative graph model

        Args:
            single_g (ndarray): single graph adjacency matrix ``[d, d]``
            single_z (ndarray): single latent tensor ``[d, k, 2]``
            t (int): step

        Returns:
            log likelihood :math:`log p(G | Z)` of shape ``[1,]``
        """
        # [d, d], [d, d]
        log_p, log_1_p = self.edge_log_probs(single_z, t, E_k)

        # [d, d]
        log_prob_g_ij = single_g * log_p + (1 - single_g) * log_1_p
        
        # [d, d] # mask hard constraints to zero, so they don't affect gradients
        masked_log_prob_g_ij = self.hard_constraint_mask_zeros(log_prob_g_ij, E_k)
        
        # [1,] # diagonal is masked inside `edge_log_probs`
        log_prob_g = jnp.sum(masked_log_prob_g_ij)

        return log_prob_g
    

    def eltwise_grad_latent_log_prob(self, gs, single_z, E_k, t):
        """
        Gradient of log likelihood of generative graph model w.r.t. :math:`Z`
        i.e. :math:`\\nabla_Z \\log p(G | Z)`
        Batched over samples of :math:`G` given a single :math:`Z`.

        Args:
            gs (ndarray): batch of graph matrices ``[n_graphs, d, d]``
            single_z (ndarray): latent variable ``[d, k, 2]``
            t (int): step

        Returns:
            batch of gradients of shape ``[n_graphs, d, k, 2]``
        """
        dz_latent_log_prob = grad(self.latent_log_prob, 1)
        return vmap(dz_latent_log_prob, (0, None, None, None), 0)(gs, single_z, E_k, t)

    
    #
    # Functionalities
    #
        
    def visualize_posteriors(self, callback, steps=0):
        for k in range(self.log_q_c.shape[1]):
            callback(dibs=self,
                     t=steps,
                     k=k+1,
                     zs=self.q_z[k],
                     thetas=self.q_theta[k],
                     E_k=self.E[k],
                     ipython=True if k < self.n_components-1 else False)
    
    
    def component_classification_accuracy(self, *, component, c_k_targets):
        # TODO
        pass

    
    def identify_ordering(self, *, mixture_data):
        """
        Identify component indices based on n_components number of ground truth graphs
        
        Args:
            mixture_data (ndarray): ndarray of shape [n_components, n_observations, n_vars] with data for each component.
            
        Returns:
            order (array): array of with order of components
        """
        # [n_components, n_particles, n_vars, n_vars]
        component_gs = self.compwise_particle_to_g_lim(self.q_z, self.E)
        
        # Get particle distributions for each component (list comprhension since get_empirical is impure)
        # [n_components, ParticleDistribution]
        component_dists = [self.get_empirical(component_gs[k], self.q_theta[k]) for k in range(self.log_q_c.shape[1])]
        
        e_log_lik = lambda dist, x: neg_ave_log_likelihood(dist=dist,
                                                           eltwise_log_likelihood=self.eltwise_component_log_likelihood_observ,
                                                           x=x)
        # Calculate log likelihood of all components for data
        cost_matrix = [[e_log_lik(dist, data) for data in mixture_data] for dist in component_dists]
        
        # use Hungaraian algorithm to find optimal allocation
        assignments = linear_sum_assignment(cost_matrix)
        
        # Return list of optimal allocations as order of components
        return assignments
    
    
    def get_MAP_classification_predicitions(self, *, responsibilities=None):
        if responsibilities is None:
            return jnp.argmax(self.log_q_c, axis=1)
        return jnp.argmax(responsibilities, axis=1)
    
    
    def identify_MAP_classification_ordering(self, *, ground_truth_indicator, responsibilities=None):
        """
        Identify component indices based on ordering that maximizes MAP classification accuracy. 

        Args:
            ground_truth_indicators (ndarray): ndarray of shape [n_observations,] with ground truth component indicators.

        Outputs:
            order (array): array of with order of components
        """
        labels = jnp.arange(self.log_q_c.shape[1])
        # Get target and predicited assignments
        y_target = ground_truth_indicator
        y_pred = self.get_MAP_classification_predicitions(responsibilities=responsibilities)

        # Solve linear assignment problem maximizing correct assignments
        cm = confusion_matrix(y_pred, y_target, labels=labels)        
        indexes = linear_sum_assignment(cm, maximize=True)

        # Return list of optimal allocations as order of components
        return indexes[1]
    
    
    
