import jax.numpy as jnp
import jax.random as random
from jax import vmap
from jax.scipy.special import logsumexp
from igraph import compare_communities

from vamsl.utils.tree import tree_mul, tree_select
from vamsl.graph_utils import elwise_acyclic_constr_nograd

from sklearn import metrics as sklearn_metrics

from typing import Any, NamedTuple


class ParticleDistribution(NamedTuple):
    """ NamedTuple for structuring sampled particles :math:`(G, \\Theta)` (or :math:`G`)
    and their assigned log probabilities

    Args:
        logp (ndarray): vector of log probabilities or weights of shape ``[M, ]``
        g (ndarray): batch of graph adjacency matrix of shape ``[M, d, d]``
        theta (ndarray): batch of parameter PyTrees with leading dimension ``M``

    """
    logp: Any
    g: Any
    theta: Any = None


def pairwise_structural_hamming_distance(*, x, y):
    """
    Computes pairwise Structural Hamming distance, i.e.
    the number of edge insertions, deletions or flips in order to transform one graph to another
    This means, edge reversals do not double count, and that getting an undirected edge wrong only counts 1

    Args:
        x (ndarray): batch of adjacency matrices  [N, d, d]
        y (ndarray): batch of adjacency matrices  [M, d, d]

    Returns:
        matrix of shape ``[N, M]``  where elt ``i,j`` is  SHD(``x[i]``, ``y[j]``)
    """

    # all but first axis is usually used for the norm, assuming that first dim is batch dim
    assert(x.ndim == 3 and y.ndim == 3)

    # via computing pairwise differences
    pw_diff = jnp.abs(jnp.expand_dims(x, axis=1) - jnp.expand_dims(y, axis=0))
    pw_diff = pw_diff + pw_diff.transpose((0, 1, 3, 2))

    # ignore double edges
    pw_diff = jnp.where(pw_diff > 1, 1, pw_diff)
    shd = jnp.sum(pw_diff, axis=(2, 3)) / 2

    return shd


def expected_shd(*, dist, g):
    """
    Computes expected structural hamming distance metric, defined as

    :math:`\\text{expected SHD}(p, G^*) := \\sum_G p(G | D)  \\text{SHD}(G, G^*)`

    Args:
        dist (:class:`dibs.metrics.ParticleDistribution`): particle distribution
        g (ndarray): ground truth adjacency matrix of shape ``[d, d]``

    Returns: 
        expected SHD ``[1, ]``
    """
    n_vars = g.shape[0]

    # select acyclic graphs
    is_dag = elwise_acyclic_constr_nograd(dist.g, n_vars) == 0
    if is_dag.sum() == 0:
        # score as "wrong on every edge"
        return n_vars * (n_vars - 1) / 2
    
    particles = dist.g[is_dag, :, :]
    log_weights = dist.logp[is_dag] - logsumexp(dist.logp[is_dag])
    
    # compute shd for each graph
    shds = pairwise_structural_hamming_distance(x=particles, y=g[None]).squeeze(1)

    # expected SHD = sum_G p(G) SHD(G)
    log_expected_shd, log_expected_shd_sgn = logsumexp(
        log_weights, b=shds.astype(log_weights.dtype), axis=0, return_sign=True)

    eshd = log_expected_shd_sgn * jnp.exp(log_expected_shd)
    return eshd


def expected_edges(*, dist):
    """
    Computes expected number of edges, defined as

    :math:`\\text{expected edges}(p) := \\sum_G p(G | D)  |\\text{edges}(G)|`

    Args:
        dist (:class:`dibs.metrics.ParticleDistribution`): particle distribution

    Returns:
        expected number of edges ``[1, ]``
    """

    n_vars = dist.g.shape[-1]

    # select acyclic graphs
    is_dag = elwise_acyclic_constr_nograd(dist.g, n_vars) == 0
    if is_dag.sum() == 0:
        # if no acyclic graphs, count the edges of the cyclic graphs; more consistent 
        n_edges_cyc = dist.g.sum(axis=(-1, -2))
        log_expected_edges_cyc, log_expected_edges_cyc_sgn = logsumexp(
            dist.logp, b=n_edges_cyc.astype(dist.logp.dtype), axis=0, return_sign=True)

        expected_edges_cyc = log_expected_edges_cyc_sgn * jnp.exp(log_expected_edges_cyc)
        return expected_edges_cyc
    
    particles = dist.g[is_dag, :, :]
    log_weights = dist.logp[is_dag] - logsumexp(dist.logp[is_dag])
    
    # count edges for each graph
    n_edges = particles.sum(axis=(-1, -2))

    # expected edges = sum_G p(G) edges(G)
    log_expected_edges, log_expected_edges_sgn = logsumexp(
        log_weights, b=n_edges.astype(log_weights.dtype), axis=0, return_sign=True)

    edges = log_expected_edges_sgn * jnp.exp(log_expected_edges)
    return edges


def threshold_metrics(*, dist, g):
    """
    Computes various threshold metrics (e.g. ROC, precision-recall, ...)

    Args:
        dist (:class:`dibs.metrics.ParticleDistribution`): sampled particle distribution
        g (ndarray): ground truth adjacency matrix of shape ``[d, d]``

    Returns:
        dict of metrics
    """
    n_vars = g.shape[0]
    g_flat = g.reshape(-1)

    # select acyclic graphs
    is_dag = elwise_acyclic_constr_nograd(dist.g, n_vars) == 0
    if is_dag.sum() == 0:
        # score as random/junk classifier
        # for AUROC: 0.5
        # for precision-recall: no. true edges/ no. possible edges
        return {
            'roc_auc': 0.5,
            'prc_auc': (g.sum() / (n_vars * (n_vars - 1))).item(),
            'ave_prec': (g.sum() / (n_vars * (n_vars - 1))).item(),
        }

    particles = dist.g[is_dag, :, :]
    log_weights = dist.logp[is_dag] - logsumexp(dist.logp[is_dag])

    # P(G_ij = 1) = sum_G w_G 1[G = G] in log space
    log_edge_belief, log_edge_belief_sgn = logsumexp(
        log_weights[..., jnp.newaxis, jnp.newaxis], 
        b=particles.astype(log_weights.dtype), 
        axis=0, return_sign=True)

    # L1 edge error
    p_edge = log_edge_belief_sgn * jnp.exp(log_edge_belief)
    p_edge_flat = p_edge.reshape(-1)

    # threshold metrics 
    fpr_, tpr_, _ = sklearn_metrics.roc_curve(g_flat, p_edge_flat)
    roc_auc_ = sklearn_metrics.auc(fpr_, tpr_)
    precision_, recall_, _ = sklearn_metrics.precision_recall_curve(g_flat, p_edge_flat)
    prc_auc_ = sklearn_metrics.auc(recall_, precision_)
    ave_prec_ = sklearn_metrics.average_precision_score(g_flat, p_edge_flat)
    
    return {
        'fpr': fpr_.tolist(),
        'tpr': tpr_.tolist(),
        'roc_auc': roc_auc_,
        'precision': precision_.tolist(),
        'recall': recall_.tolist(),
        'prc_auc': prc_auc_,
        'ave_prec': ave_prec_,
    }


def neg_ave_log_marginal_likelihood(*, dist, eltwise_log_marginal_likelihood, x):
    """
    Computes neg. ave log marginal likelihood for a marginal posterior over :math:`G`, defined as

    :math:`\\text{neg. MLL}(p, G^*) := - \\sum_G p(G | D)  p(D^{\\text{test}} | G)`

    Args:
        dist (:class:`dibs.metrics.ParticleDistribution`): particle distribution
        eltwise_log_marginal_likelihood (callable):
            function evaluting the marginal log likelihood :math:`p(D | G)` for a batch of graph samples given
            a data set of held-out observations;
            must satisfy the signature
            ``[:, d, d], [N, d] -> [:,]``
        x (ndarray): held-out observations of shape ``[N, d]``

    Returns:
        neg. ave log marginal likelihood metric of shape ``[1,]``
    """
    n_ho_observations, n_vars = x.shape

    # select acyclic graphs
    is_dag = elwise_acyclic_constr_nograd(dist.g, n_vars) == 0
    if is_dag.sum() == 0:
        # score as empty graph only
        g = jnp.zeros((1, n_vars, n_vars), dtype=dist.g.dtype)
        log_weights = jnp.array([0.0], dtype=dist.logp.dtype)

    else:
        g = dist.g[is_dag, :, :]
        log_weights = dist.logp[is_dag] - logsumexp(dist.logp[is_dag])
        
    log_likelihood = eltwise_log_marginal_likelihood(g, x)

     # - sum_G p(G | D) log(p(x | G))
    log_score, log_score_sgn = logsumexp(
        log_weights, b=log_likelihood, axis=0, return_sign=True)
    score = - log_score_sgn * jnp.exp(log_score)
    return score


def neg_ave_log_likelihood(*, dist, eltwise_log_likelihood, x):
    """
    Computes neg. ave log likelihood for a joint posterior over :math:`(G, \\Theta)`, defined as

    :math:`\\text{neg. LL}(p, G^*) := - \\sum_G \\int_{\\Theta} p(G, \\Theta | D)  log p(D^{\\text{test}} | G, \\Theta)`

    Args:
        dist (:class:`dibs.metrics.ParticleDistribution`): particle distribution
        eltwise_log_likelihood (callable):
            function evaluting the log likelihood :math:`p(D | G, \\Theta)` for a batch of graph samples given
            a data set of held-out observations;
            must satisfy the signature
            ``[:, d, d], PyTree(leading dim :), [N, d] -> [:,]``
        x (ndarray): held-out observations of shape ``[N, d]``

    Returns:
        neg. ave log likelihood metric of shape ``[1,]``
    """
    assert dist.theta is not None
    if x.ndim == 1:
        x = x.reshape((1,-1))
    n_ho_observations, n_vars = x.shape

    # select acyclic graphs
    is_dag = elwise_acyclic_constr_nograd(dist.g, n_vars) == 0
    if is_dag.sum() == 0:
        # score as empty graph only
        g = tree_mul(dist.g, 0.0)
        theta = tree_mul(dist.theta, 0.0)
        log_weights = tree_mul(dist.logp, 0.0)

    else:
        g = dist.g[is_dag, :, :]
        theta = tree_select(dist.theta, is_dag)
        log_weights = dist.logp[is_dag] - logsumexp(dist.logp[is_dag])
        
    log_likelihood = eltwise_log_likelihood(g, theta, x)

    # - sum_G p(G, theta | D) log(p(x | G, theta))
    log_score, log_score_sgn = logsumexp(
        log_weights, b=log_likelihood, axis=0, return_sign=True)
    score = - log_score_sgn * jnp.exp(log_score)
    
    return score


def empirical_sample_BNs(*, dist, n_vars):
    assert dist.theta is not None
    # select acyclic graphs
    is_dag = elwise_acyclic_constr_nograd(dist.g, n_vars) == 0
    if is_dag.sum() == 0:
        # score as empty graph only
        g = tree_mul(dist.g, 0.0)
        theta = tree_mul(dist.theta, 0.0)
        log_weights = tree_mul(dist.logp, 0.0)
    else:
        g = dist.g[is_dag, :, :]
        theta = tree_select(dist.theta, is_dag)
        log_weights = dist.logp[is_dag] - logsumexp(dist.logp[is_dag])
    
    return (g, theta, log_weights)


def expected_log_likelihood(*, x, dist, eltwise_log_likelihood, return_sign=False):
    """
    Compute MC approximation of expected log likelihood for observations x 
    given a particle distribution.
    """
    assert dist.theta is not None
    if x.ndim == 1:
        x = x.reshape((1,-1))
        
    n_observations, n_vars = x.shape
    # select acyclic graphs
    g, theta, log_weights = empirical_sample_BNs(dist=dist, n_vars=n_vars)

    # [len(g), 1] 
    # log p(x* | G, theta)
    log_likelihoods = eltwise_log_likelihood(g, theta, x)

    # [len(g), 1] 
    # log p(x*, G, theta | D) = log p(x* | G, theta) + log p(G, theta | D)
    log_joint = log_likelihoods + log_weights

    # [1,], [1,]
    # log p(x* | D) = Log Sum_G Exp(log p(x*, G_k, theta_k | D))
    if return_sign:
        return logsumexp(log_joint, axis=0, return_sign=return_sign)
    log_score = logsumexp(log_joint, axis=0)
    
    return log_score


def single_neg_log_posterior_predictive_density(*, key, q_pi, dists, eltwise_log_likelihood, x_n, n_mixing_mc_samples=100):
    """
    Returns approximated posterior predictive density: -log p(x* | D).
    """
    # sample mixing weights pi^(s) ~ q_pi
    # [n_mixing_mc_samples, q_pi.shape[0]]
    pis = random.dirichlet(key, q_pi, shape=(n_mixing_mc_samples,))
    
    log_liks, log_liks_sgn = jnp.zeros_like(q_pi), jnp.zeros_like(q_pi)
    
    # calculate mc approximatrion of log p(x* | D, c=k) for each k
    for k, dist in zip(range(len(dists)), dists):
        # [1,], [1,]
        # log p(x*_n | D, c_n=k) = Sum_G,theta_k 1/|G| * Exp(log p(x*, G_k, theta_k | D, c_n=k))
        log_score, log_score_sgn = expected_log_likelihood(x=x_n, 
                                                           dist=dists[k], 
                                                           eltwise_log_likelihood=eltwise_log_likelihood, 
                                                           return_sign=True)
        
        # store component log likelihood
        log_liks = log_liks.at[k].set(log_score)
        log_liks_sgn = log_liks_sgn.at[k].set(log_score_sgn)
    
    
    
    # sum over component variable for each sample pi^(s) = {pi^(s)_k}
    # log p(x*_n, c_n=k | D, pi) = Log Sum_k pi_k * Exp(log p(x*_n | D, c_n=k))
    # [n_mixing_mc_samples,]
    func = lambda log_liks, log_liks_sgn, pi: logsumexp(log_liks , b=log_liks_sgn * pi, axis=0, return_sign=True)
    log_pi_scores, log_pi_scores_sgns = vmap(func, (None, None, 0))(log_liks, log_liks_sgn, pis)
    
    # sum over samples pi ~ q_pi
    # - log p(x*_n | D) = - Log Sum_pi 1/|pi| * Exp(log p(x*_n| D, c_n=k))
    neg_log_posterior_predictive_density = - logsumexp(log_pi_scores, 
                                                       b=(1/pis.shape[0]) * log_pi_scores_sgns,
                                                       axis=0)
    
    return neg_log_posterior_predictive_density


def mixture_lppd(*, key, q_pi, dists, eltwise_log_likelihood, x, n_mixing_mc_samples=100):
    """
    Computes the log point-wise predictive density for observations given a mixture model.
    """
    mixture_lppds = 0
    # calculate mc approximation of \sum_n log p(x*_n | D)
    key, *batch_subk = random.split(key, x.shape[0] + 1)
    for n, subk in zip(range(x.shape[0]), jnp.array(batch_subk)):
        # [1,], [1,]
        # log p(x*_n | D) = Log Sum_pi 1/|pi| * Exp(log p(x*| D, c=k))
        mixture_lppds += - single_neg_log_posterior_predictive_density(key=subk,
                                                                q_pi=q_pi,
                                                                dists=dists, 
                                                                eltwise_log_likelihood=eltwise_log_likelihood, 
                                                                x_n=x[n], 
                                                                n_mixing_mc_samples=n_mixing_mc_samples)
        
    # Sum_n log p(x*_n| D)
    return mixture_lppds


def single_expected_log_mixture_lik(*, key, q_pi, dists, eltwise_log_likelihood, x_n, n_mixing_mc_samples=100):
    """
    Returns approximated expectation: E_G(log p(x*_n | G)).
    """
    if x_n.ndim == 1:
        x_n = x_n.reshape((1,-1))
        
    # K number of tuples (g, theta, log_weights) 
    samples = [empirical_sample_BNs(dist=dist, n_vars=x_n.shape[1]) for dist in dists]
    num_samples, K = min([sample[0].shape[0] for sample in samples]), len(dists)
    
    sample_scores = jnp.zeros((num_samples,))
    for s in range(num_samples):
        log_liks = jnp.zeros((K,))
        # sample mixing weights pi^(s) ~ q_pi
        # [n_mixing_mc_samples, q_pi.shape[0]]
        pis = random.dirichlet(key, q_pi, shape=(n_mixing_mc_samples,))
        for k in range(K):
            g_k, theta_k = samples[k][0][s], samples[k][1][s]
            log_liks = log_liks.at[k].set(eltwise_log_likelihood(jnp.array([g_k]), 
                                                                 jnp.array([theta_k]), 
                                                                 x_n).item())
            
        func = lambda log_liks, pi: logsumexp(log_liks , b=pi, axis=0)
        log_pi_scores = vmap(func, (None, 0))(log_liks, pis)
        sample_scores  = sample_scores.at[s].set(log_pi_scores.mean())
    
    return sample_scores.mean()


def expected_log_mixture_lik(*, key, q_pi, dists, eltwise_log_likelihood, x, n_mixing_mc_samples=100):
    """
    Computes the expected score (marginalizing the mixture components) for observations given a mixture model.
    """
    if x.ndim == 1:
        x = x.reshape((1,-1))
        
    expected_mixture_scores = 0
    key, *batch_subk = random.split(key, x.shape[0] + 1)
    for n, subk in zip(range(x.shape[0]), jnp.array(batch_subk)):
        # [1,], [1,]
        # log p(x*_n | D) = Log Sum_pi 1/|pi| * Exp(log p(x*| D, c=k))
        expected_mixture_scores += single_expected_log_mixture_lik(key=subk,
                                                                   q_pi=q_pi,
                                                                   dists=dists,
                                                                   eltwise_log_likelihood=eltwise_log_likelihood,
                                                                   x_n=x[n],
                                                                   n_mixing_mc_samples=n_mixing_mc_samples)
        
    # Sum_n sum_pi sum_G,Theta log(sum_k pi_k * p(x_n|G_k, Theta_k))
    return expected_mixture_scores


def posterior_predictive_SHD(*, key, q_pi, dists, g, n_mixing_mc_samples=100):
    """
    Returns predictive SHD to graph g for mixture model.
    """
    comp_eshd = jnp.zeros_like(q_pi)
    # calculate ESHD for each k
    for k, dist in zip(range(len(dists)), dists):
        eshd_k = expected_shd(dist=dist, g=g)
        # store component ESHD
        comp_eshd = comp_eshd.at[k].set(eshd_k)
    
    # sample mixing weights pi^(s) ~ q_pi
    # [n_mixing_mc_samples, q_pi.shape[0]]
    pis = random.dirichlet(key, q_pi, shape=(n_mixing_mc_samples,))
    
    # sum over component variable for each sample pi^(s) = {pi^(s)_k}
    # Sum_k pi_k * ESHD(G^*, q(G_k))
    # [n_mixing_mc_samples,]
    mc_sample_eshds = vmap(lambda comp_eshd, pi: (comp_eshd * pi).sum(axis=0), (None, 0))(comp_eshd, pis)
    
    # mean over MC samples pi ~ q_pi
    posterior_predictive_SHD = mc_sample_eshds.mean(axis=0)
    
    return posterior_predictive_SHD


def expected_VI(*, key, indicator, log_q_c, n_cluster_mc_samples=100):
    """
    MC approximates expected VI for posterior responsibilities given ground truth indicator. 
    """
    # [n_cluster_mc_samples, n_observations]
    c_samples = random.categorical(key, log_q_c, axis=1, shape=(n_cluster_mc_samples, log_q_c.shape[0]))
    
    VI = lambda c, c_pred: compare_communities(c, c_pred, method='vi')
    # Calculate VI for each MC cluster sample 
    # [n_cluster_mc_samples,]
    VI_samples = jnp.array([VI(indicator, c_sample) for c_sample in c_samples])
    
    # Return MC average
    return VI_samples.mean()


def ordered_MAP_classification_accuracy(*, indicator, order, MAP_preds):
    """
    Computes accuracy of classification based on log responsibilities and supplied ordering 
    of indicators with respect to component indeces. 
    """
    # Get MAP predicitions from responsibilities and order according to ground truths
    ordered_MAP_preds = [order[k] for k in MAP_preds]
        
    return sklearn_metrics.accuracy_score(indicator, ordered_MAP_preds)


def MAP_assigned_lppd(*, x, MAP_assignments, dists, eltwise_log_likelihood):
    """
    Computes the point-wise negative log predictive density for all observations given 
    component-wise assignments. 
    """
    lppds, lppds_sgn = jnp.zeros((x.shape[0],)), jnp.zeros((x.shape[0],))
    # calculate mc approximation of \sum_n log p(x*_n | D, c_n=k)
    for n in range(x.shape[0]):
        # [1,], [1,]
        # log p(x*_n | D, c_n=k) = Sum_G,theta_k 1/|G| * Exp(log p(x*, G_k, theta_k | D, c_n=k))
        log_score, log_score_sgn = expected_log_likelihood(x=x[n], 
                                                           dist=dists[MAP_assignments[n]], 
                                                           eltwise_log_likelihood=eltwise_log_likelihood, 
                                                           return_sign=True)
        
        lppds = lppds.at[n].set(log_score)
        lppds_sgn = lppds_sgn.at[n].set(log_score_sgn)
    
    # - Sum_n log p(x*| D, c=k)
    lppd = jnp.sum(lppds*lppds_sgn)
    
    return lppd


def MAP_assigned_neg_ave_log_lik(*, x, MAP_assignments, dists, eltwise_log_likelihood):
    """
    Computes neg. ave log likelihood for observations assigned to different components
    of a mixture model.
    """
    negLL = 0
    for k in range(len(dists)):
        negLL += neg_ave_log_likelihood(dist=dists[k],
                                        eltwise_log_likelihood=eltwise_log_likelihood,
                                        x=x[MAP_assignments==k])
        
    return negLL
        
    
    

        
        
    
