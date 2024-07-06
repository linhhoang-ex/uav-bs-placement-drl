import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import ticker
import imageio
# from mpl_toolkits.mplot3d import Axes3D

from params_sys import *
from utils import *

mpl.rcParams.update(mpl.rcParamsDefault)


def plot_moving_average(raw_data, rolling_intv, ylabel, filepath=None, title=None):
    data_array = np.asarray(raw_data)
    df = pd.DataFrame(raw_data)

    mpl.style.use('seaborn')
    fig, ax = plt.subplots(figsize=(15, 8))

    plt.plot(np.arange(len(data_array)) + 1,
             np.hstack(df.rolling(window=1, min_periods=1).mean().values),
             'b', linewidth=0.5, label='Raw Data')

    plt.plot(np.arange(len(data_array)) + 1,
             np.hstack(df.rolling(window=rolling_intv, min_periods=1).mean().values),
             'r', label='Moving Average (w={x})'.format(x=rolling_intv))

    plt.fill_between(np.arange(len(data_array)) + 1,
                     np.hstack(df.rolling(rolling_intv, min_periods=1).min()[0].values),
                     np.hstack(df.rolling(rolling_intv, min_periods=1).max()[0].values),
                     color='b', alpha=0.2)

    plt.legend()

    plt.ylabel(ylabel)
    plt.xlabel('Time (sec)')
    plt.title(title)

    if filepath is None:
        plt.show()
    else:
        plt.savefig(filepath + '/' + ylabel + '.png')


def plt_uav_trajectory(uavs, name):
    fig, ax = plt.subplots(figsize=(4, 4))
    for i, uav in enumerate(uavs):
        plt.plot(uav.x, uav.y, label=f'UAV {i}')
    ax.set(xlim=(-boundary, boundary), ylim=(-boundary, boundary), aspect='equal')
    plt.grid(True, linestyle=':')
    plt.legend()
    plt.savefig(os.path.join(sim_folder_path, f'uav-trajectory-{name}.png'))
    plt.show()


'''
-------------------------------------
    Plot images with clusters
-------------------------------------
'''


def gen_image_with_clusters(xaxis_all, yaxis_all, uavswarm, t, folder_name, marker_sz, color_codes, markers, dpi=72, figsize=(6, 6)):
    fig, ax = plt.subplots(figsize=figsize)    # figsize in inch
    for id in range(n_users):
        plt.plot(xaxis_all[id][t], yaxis_all[id][t], linestyle='None', marker=markers[id],
                 markeredgecolor=color_codes[id], markerfacecolor='None', markersize=marker_sz)     # markeredgewidth
    for id, uav in enumerate(uavswarm):
        plt.plot(uav.x[t], uav.y[t], linestyle='None', marker='x', label=uav.name,
                 markeredgecolor=colors[id], markerfacecolor='None', markersize=7, markeredgewidth=2)
    ax.set(xlim=(-boundary, boundary), ylim=(-boundary, boundary))
    ax.set(title=f'time = {t} (second)', xlabel='x (m)', ylabel='y (m)')
    plt.legend(bbox_to_anchor=(1.02, 0.3), loc='lower left')       # options: 'best', 'upper left', 'upper right', 'lower left', 'lower right'
    plt.grid(True, linestyle=':')
    plt.savefig(os.path.join(os.getcwd(), folder_name, f"t{t}.png"), bbox_inches='tight', dpi=dpi)
    plt.close()


def gen_image_with_clusters_3D(xaxis_all, yaxis_all, uavswarm, t, folder_name, marker_sz, color_codes, markers, alphas, dpi=72, figsize=(6, 6)):
    fig = plt.figure(figsize=figsize)    # figsize in inch
    ax = fig.add_subplot(111, projection='3d')

    for id in range(n_users):
        s_ = 1.5 * marker_sz if markers[id] == '^' else marker_sz
        ax.scatter(xaxis_all[id][t], yaxis_all[id][t], linestyle='None', marker=markers[id],
                   edgecolors=color_codes[id], color='none', s=s_, alpha=alphas[id])   # linewidths

    for id, uav in enumerate(uavswarm):
        s_ = marker_sz  # 4*marker_sz if markers_cluster[id] == '^' else 3*marker_sz
        zvec = uav.z[t] * np.arange(0, 0.99, 0.98)
        xvec = uav.x[t] * np.ones(2)
        yvec = uav.y[t] * np.ones(2)
        ax.scatter(uav.x[t], uav.y[t], uav.z[t], linestyle='None', marker=markers_cluster[id], label=uav.name,
                   edgecolors=colors[id], color='none', s=s_, linewidths=1)
        ax.scatter(uav.x[t], uav.y[t], uav.z[t], marker='1', s=40, linewidths=1)    # edgecolors=colors[id]
        ax.text(uav.x[t] + 5, uav.y[t] + 5, uav.z[t] + 5, uav.name[:5] + f' ({uav.z[t]:.0f}m)')
        ax.plot(xvec, yvec, zvec, linestyle='--', linewidth=1, dashes=(5, 5), alpha=0.7)  # dashes=(length,interval_length)

    ax.view_init(elev=15)  # azim=120
    ax.set_box_aspect([1, 1, 0.6])        # ax.set_aspect('auto')
    ax.xaxis.set_major_locator(ticker.LinearLocator(5))
    ax.yaxis.set_major_locator(ticker.LinearLocator(5))
    ax.zaxis.set_major_locator(ticker.LinearLocator(4))
    ax.set(xlim=(-boundary, boundary), ylim=(-boundary, boundary), zlim=(0, z_max))
    ax.set(xlabel='x (m)', ylabel='y (m)', zlabel='z (m)')
    ax.set_title(label=f'time = {t} / {n_slots-1} (s)', fontdict={'color': 'red'}, y=0.95)
    # ax.legend(loc='best', ncol=2)       # options: 'best', 'upper left', 'upper right', 'lower left', 'lower right', bbox_to_anchor=(1.02, 0.2)
    # ax.grid(visible=True, linestyle='--') # not working on 3d plots
    fig.tight_layout()
    plt.savefig(os.path.join(os.getcwd(), folder_name, f"t{t}.png"), dpi=dpi, bbox_inches='tight', pad_inches=0.5)
    plt.close()


def gen_animation_with_clusters(users, uavswarm, xaxis_all, yaxis_all, dim='2d', t_min=0, t_max=100, t_step=10,
                                folder_name='dev/png2gif', gifname='animation', fps=3, dpi=72, figsize=(6, 6)):
    # plot user locations in each time slot as png images
    for t in range(t_min, t_max, t_step):
        color_codes = [colors[int(user.cluster[t])] for user in users]
        marker_codes = [markers_cluster[int(user.cluster[t])] for user in users]
        alphas = np.ones(len(users))    # [1 if user.active_state[t] == True else 0.6 for user in users]
        if dim == '2d':
            gen_image_with_clusters(xaxis_all, yaxis_all, uavswarm, t, folder_name, marker_sz=5,
                                    color_codes=color_codes, markers=marker_codes, dpi=dpi, figsize=figsize)
        elif dim == '3d':
            gen_image_with_clusters_3D(xaxis_all, yaxis_all, uavswarm, t, folder_name, marker_sz=20, color_codes=color_codes,
                                       markers=marker_codes, alphas=alphas, dpi=dpi, figsize=figsize)

    # save all image into a list
    images = []
    for t in range(t_min, t_max, t_step):
        image = imageio.v2.imread(os.path.join(os.getcwd(), folder_name, f"t{t}.png"))
        images.append(image)

    # combine all images into a GIF
    path_to_gif = os.path.join(os.getcwd(), folder_name, gifname)
    imageio.mimsave(path_to_gif, ims=images, fps=fps)  # ims = list of input images, [fps = frames per second]

    return path_to_gif


'''
-------------------------------------
    Plot images W/O clusters
-------------------------------------
'''

# Reference: https://towardsdatascience.com/how-to-create-a-gif-from-matplotlib-plots-in-python-6bec6c0c952c


def create_image_of_user_locations(
    xaxis_all, yaxis_all, t=0, folder_name='dev/png2gif', marker_sz=2
):
    fig, ax = plt.subplots(figsize=(5, 5))
    for id in range(n_users):
        plt.plot(
            xaxis_all[id][t], yaxis_all[id][t],
            linestyle='None', marker='.', color='k', markersize=marker_sz
        )
    plt.plot(
        xaxis_all[id][t], yaxis_all[id][t],
        linestyle='None', marker='+', color='k', markersize=marker_sz, label='mobile users'
    )
    ax.set(xlim=(-boundary, boundary), ylim=(-boundary, boundary))
    ax.set(title=f'time = {t} (second)', xlabel='x (m)', ylabel='y (m)')
    plt.legend(loc='upper right')
    plt.grid(True, linestyle=':')
    plt.savefig(os.path.join(os.getcwd(), folder_name, f"t{t}.png"), bbox_inches='tight', dpi=150)
    plt.close()


def gen_animation_user_locations(
    xaxis_all, yaxis_all, t_max=100, t_step=10, folder_name='dev/png2gif'
):
    # plot user locations in each time slot as png images
    for t in range(0, t_max, t_step):
        create_image_of_user_locations(xaxis_all, yaxis_all, t, folder_name)

    # save all image into a list
    images = []
    for t in range(0, t_max, t_step):
        image = imageio.v2.imread(os.path.join(os.getcwd(), folder_name, f"t{t}.png"))
        images.append(image)

    # combine all images into a GIF
    path_to_gif = os.path.join(os.getcwd(), folder_name, 'user_location.gif')
    imageio.mimsave(path_to_gif, ims=images, fps=1)

    return path_to_gif
