import numpy as np
import jax
import jax.numpy as jnp
from jax import vmap
from jax.nn import sigmoid, log_sigmoid
jax.config.update("jax_debug_nans", True)

from vamsl.inference import MixtureJointDiBS
from vamsl.metrics import ParticleDistribution, ave_log_likelihood, neg_ave_log_likelihood 
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
        
        
    def initialize_posteriors(self, *, key, init_q_c, n_particles, E=None):
        self.n_particles = n_particles
        self.q_c = init_q_c
        self.update_mixing_weigths()
        n_components = self.q_c.shape[1]
        self.q_z, self.q_theta = self._sample_intial_component_particles(key=key,
                                                                         n_components=n_components, 
                                                                         n_particles=n_particles,
                                                                         n_dim=self.n_vars)
        
        self.sf_baselines = jnp.zeros((n_components, self.n_particles))
        
        if E or self.E:
            self.E = E
        else:
            self.E = jnp.zeros((n_components, self.n_vars, self.n_vars))

        
    def update_particle_posteriors(self, *, key, steps, callback=None, callback_every=None):
        self.q_z, self.q_theta, self.sf_baselines = self.sample(key=key, n_particles=self.n_particles, 
                                                                steps=steps, callback=callback, 
                                                                callback_every=callback_every, 
                                                                q_c=self.q_c, init_model=False,
                                                                init_q_z=self.q_z, 
                                                                init_q_theta=self.q_theta, 
                                                                init_sf_baselines=self.sf_baselines,
                                                                E=self.E)

    
    def update_mixing_weigths(self):
        self.q_pi = (1 / self.x.shape[0]) * jnp.sum(self.q_c, axis=0)
        

    def compute_responsibility(self, x_n, dist_k, pi_k):
        expected_log_lik = - neg_ave_log_likelihood(dist=dist_k,
                                                    eltwise_log_likelihood=self.eltwise_component_log_likelihood_observ,
                                                    x=jnp.array(x_n.reshape((1,-1))))
                                                      
        return np.log(pi_k) + expected_log_lik
    
    
    def update_responsibilities(self):
        N = self.x.shape[0]
        K = self.q_c.shape[1]
        
        # Get graphs for MC-estimating expected data log likelihoods
        # [n_components, n_particles, n_vars, n_vars]
        component_gs = self.compwise_particle_to_g_lim(self.q_z, self.E)
        
        # Get particle distributions for each component (list comprhension since get_empirical is impure)
        # [n_components, ParticleDistribution]
        component_dists = [self.get_empirical(component_gs[k], self.q_theta[k]) for k in range(K)]
        
        # Get unnormalized repsonsiibilities for components (list comprehension since component_dists can differ in size)
        # [n_observations, n_components]
        unnorm_responsibilities = jnp.transpose(jnp.array([vmap(self.compute_responsibility, (0, None, None))(self.x, component_dists[k], self.q_pi[k]) for k in range(K)]))
        
        # Get softmax-normalized component responsibilities
        # [n_observations, n_components]
        responsibilities = vmap(lambda unorm_c_n: stable_softmax(unorm_c_n), (0))(unnorm_responsibilities)
        
        self.q_c = responsibilities
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
        pass

    
    def order_components(self):
        pass
    
    
    ## Getters and setters
    
    def get_posteriors(self):
        return self.q_z, self.q_theta, self.q_c, self.q_pi
    
    
    def set_E(self, E):
        self.E = E