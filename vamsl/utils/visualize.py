import os
import matplotlib.pyplot as plt
import imageio
import matplotlib as mpl


def visualize_ground_truth(mat, size=4.0, graph_label='Ground truth $G^*$'):
    """    
    `mat`: (d, d) 
    """
    plt.rcParams['figure.figsize'] = [size, size]
    fig, ax = plt.subplots(1, 1)
    im = ax.matshow(mat, vmin=0, vmax=1)
    plt.setp(ax.get_xticklabels(), visible=False)
    plt.setp(ax.get_yticklabels(), visible=False)
    ax.tick_params(axis='both', which='both', length=0)
    ax.set_title(f'{graph_label}', pad=10)
    fig.colorbar(im, ax=ax, shrink=0.7)
    plt.show()
    return


def visualize_ground_truths(mats, size=4.0, columns=7,graph_label='Ground truth graph'):
    """    
    `mats`: ndarray of shape (n_ground_truths, d, d) 
    """
    if len(mats.shape) == 2:
        return visualize_ground_truth(mats, graph_label=graph_label)
    elif len(mats.shape) == 3 and mats.shape[0] == 1:
        return visualize_ground_truth(mats[0], graph_label=graph_label)
    cols = min(len(mats), columns)
    rows = 1 + ((len(mats)-1)//cols)
    plt.rcParams['figure.figsize'] = [cols*size, rows*size]
    fig, axs = plt.subplots(rows, cols)
    axs = axs.reshape((-1, cols))
    [axs[i,j].set_axis_off() for i in range(rows) for j in range(cols)]
    for n, mat in enumerate(mats):
        i,j = n//cols, n%cols
        axs[i,j].set_axis_on()
        im = axs[i,j].matshow(mat, vmin=0, vmax=1)
        plt.setp(axs[i,j].get_xticklabels(), visible=False)
        plt.setp(axs[i,j].get_yticklabels(), visible=False)
        axs[i,j].tick_params(axis='both', which='both', length=0)
        axs[i,j].set_title(f'{graph_label} {n+1}', pad=10)
        #if j == cols-1:

    fig.colorbar(im, ax=axs.ravel().tolist(), shrink=0.7)
    plt.show()
    return


def visualize_elicitation(mats, size=4.0, columns=7, component=None, order=None):
    """    
    `mats`: ndarray of shape (n_ground_truths, d, d) 
    """
    assert len(mats) >= 3, 'Need to specify: ground truth graph, particle(s) and eliciation matrix.'
    cols = min(len(mats), columns)
    rows = 1 + ((len(mats)-1)//cols)
    plt.rcParams['figure.figsize'] = [cols*size, rows*size]
    fig, axs = plt.subplots(rows, cols)
    axs = axs.reshape((-1, cols))
    [axs[i,j].set_axis_off() for i in range(rows) for j in range(cols)]
    for n, mat in enumerate(mats):
        i,j = n//cols, n%cols
        axs[i,j].set_axis_on()
        im = axs[i,j].matshow(mat, vmin=0, vmax=1)
        plt.setp(axs[i,j].get_xticklabels(), visible=False)
        plt.setp(axs[i,j].get_yticklabels(), visible=False)
        axs[i,j].tick_params(axis='both', which='both', length=0)
        if n == 0:
            if order is None:
                axs[i,j].set_title(f'Ground truth graph', pad=10)
            else:
                component = 0 if component is None else component
                axs[i,j].set_title(f'Ground truth graph {order[component]+1}', pad=10)
        elif n == len(mats)-1:
            axs[i,j].set_title(f'Elicited responses', pad=10)
        else:
            axs[i,j].set_title(f'Particle {n}', pad=10)
    
    fig.suptitle(f'Elicitation for component {component+1}', size='xx-large')
    fig.colorbar(im, ax=axs.ravel().tolist(), shrink=0.7)
    plt.show()
    return


def visualize(mats, t, save_path=None, n_cols=7, size=2.5, show=False):
    """
    Based on visualization by https://github.com/JannerM/gamma-models/blob/main/gamma/visualization/pendulum.py
    
    `mats` should have shape (N, d, d) and take values in [0,1]
    """

    N = mats.shape[0]
    n_rows = N // n_cols
    if N % n_cols:
        n_rows += 1
    
    plt.rcParams['figure.figsize'] = [size * n_cols, size * n_rows]
    fig, axs = plt.subplots(n_rows, n_cols)
    axes = axs.flatten()

    # for j, (ax, mat) in enumerate(zip(axes[:len(mats)], mats)):
    for j, ax in enumerate(axes): 
        if j < len(mats):
            # plot matrix of edge probabilities
            im = ax.matshow(mats[j], vmin=0, vmax=1)
            ax.tick_params(axis='both', which='both', length=0)
            ax.set_title(r'$Z^{('f'{j}'r')}$', pad=3)
        plt.setp(ax.get_xticklabels(), visible=False)
        plt.setp(ax.get_yticklabels(), visible=False)
        ax.axis('off')

    fig.colorbar(im, ax=axs.ravel().tolist(), shrink=0.7)
    # save
    if save_path is not None:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        plt.savefig(save_path + f'/img{t}.png')
        img = imageio.imread(save_path + f'/img{t}.png')
    else:
        img = None
    if show:
        plt.show()
    plt.close()
    return img
        
