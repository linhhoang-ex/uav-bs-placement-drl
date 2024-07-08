import os
import imageio
import seaborn as sns
from IPython.display import Image
from copy import deepcopy
from sklearn.cluster import KMeans
from scipy.optimize import linprog

from channel_model import *
from mobility_model import *
from params_sys import *
from utils import *


def cal_channel_fading_gain(x_user, y_user, x_uav, y_uav, uav_altitude):
    '''Calculate the channel fading gain from locations of the user and the UAV'''
    radius_distance = np.sqrt((x_user - x_uav)**2 + (y_user - y_uav)**2)
    propagation_distance = np.sqrt((x_user - x_uav)**2 + (y_user - y_uav)**2 + uav_altitude**2)
    elev_angle_deg = rad2deg(np.arctan(uav_altitude / (1e-3 + radius_distance)))   # avoid dividing by 0
    channel_gain = channel_fading_gain_mean(elev_angle_deg) \
        * ref_receive_pw / propagation_distance**path_loss_exponent(elev_angle_deg)
    return channel_gain


def cal_channel_capacity_Mb(channel_gain, alpha):
    '''
    Calculate channel capacity (Mb) in one time slot and spectral efficiency (bps/Hz).

    ### Params
        - channel_gain: channel gain in real values (e.g., 1e-6 mW)
        - alpha: bandwidth allocation, 0 (not allocated) or 1 (allocated)

    ### Returns
        - capacity_Mb: the throughput/channel capacity (Mb) for one time slot
        - capacity_bpsHz: the spectral efficiency (bps/Hz)
    '''
    SNR = pTx_downlink * channel_gain / noise_pw_total
    capacity_Mb = to_Mbit(alpha * bw_per_user * np.log2(1 + SNR) * slot_len)
    capacity_bpsHz = np.log2(1 + SNR)
    return capacity_Mb, capacity_bpsHz


def find_active_users(queue_len_Mb=0, incoming_traffic_Mb=0):
    '''
    Find active users (queue length > 0 or incoming traffic > 0).

    Parameters:
        - queue_len_Mb (shape=(n_users,)): current queue length in Mb
        - incoming_traffic_Mb (shape=(n_users,)): incoming traffic in Mb

    Returns:
        - active_state (shape=(n_users,)): binary (False=inactive, True=active)
    '''
    traffic_total_Mb = queue_len_Mb + incoming_traffic_Mb      # shape = (n_users,)
    active_state = np.array(traffic_total_Mb > 0)              # shape = (n_users,)
    return active_state


def cal_bandwidth_equal(active_state):
    '''
    Equal bandwidth allocation for each active user.
    Example: 5 out of 10 users are active -> allocate 1/5 of the total bandwidth to each
    Parameters:
        - active_state: shape = (n_users,), =1 if queue + traffic_arrival > 0
    Returns:
        - alpha: bandwidth allocation, in range (0,1), shape=(n_users,)
    NOTE 2023/04/19: alpha = 1 for all users, since users are allocated fixed bandwidth during the process
    '''
    # alpha = np.ones(shape=(n_users)) * 1/n_users              # shape = (n_users,), equally shared between all users
    # alpha = active_state*1/np.sum(active_state) + 1e-6        # shape = (n_users,), equally shared between active users
    alpha = np.ones_like(active_state)
    return alpha


def update_queue_Mb(qlen_prev_Mb, arrival_Mb, departure_Mb):
    '''Return the updated qlen based on the previous qlen, arrival traffic, and departure rate'''
    qlen_next_Mb = np.max([qlen_prev_Mb + arrival_Mb - departure_Mb, 0])
    return qlen_next_Mb


def cal_downlink_throughput_Mb(qlen_prev_Mb, arrival_Mb, channel_capacity_Mb):
    '''Calculate the user's download speed (Mbps). If qlen+traffic=0, then throughput=0'''
    throughput_Mb = np.min([channel_capacity_Mb, qlen_prev_Mb + arrival_Mb])
    return throughput_Mb


def load_user_properties(users_list, xaxis_all, yaxis_all, arrival_traffic_Mb, traffic_state, traffic_type, ema_traffic_Mb, lambd_Mb_alluser):
    '''Load the movements and arrival traffic to each user object'''
    for uid, user in enumerate(users_list):
        user.x = deepcopy(xaxis_all[uid, :])
        user.y = deepcopy(yaxis_all[uid, :])
        user.arrival_traffic_Mb = deepcopy(arrival_traffic_Mb[uid, :])
        user.traffic_state = deepcopy(traffic_state[uid, :])
        user.traffic_type = deepcopy(traffic_type[uid, :])
        user.ema_drate_Mbps = deepcopy(ema_traffic_Mb[uid, :])
        user.lambd_Mb = lambd_Mb_alluser[uid]


'''
----------------------------
    Bandwidth Allocation
----------------------------
'''


def allocate_bandwidth_linprog(load_Mb, chcapa_Mb_ref, mask, alpha_min):
    '''
    Optimize the bandwidth allocation based on the downlink load using linear programming

    ### Params
        load_Mb: Q_i(t) + A_i(t), shape=(n_users,)
        chcapa_Mb_ref: shape=(n_users,), maximum drate if the user occupies the whole channel
        mask: shape=(n_users,), denote active users of the considered cluster

    ### Returns
        alpha: the optimal bs allocation, shape=(n_users,), in range [0,1]

    Reference: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linprog.html
    '''
    n_users = len(mask)
    c = (-1) * load_Mb * chcapa_Mb_ref * mask    # 1-D array
    A_ub = np.ones(shape=(1, n_users))           # must be a 2-D array
    b_ub = [1]                                   # 1-D array
    bounds = [(alpha_min, 1) for i in range(0, n_users)]
    options = {'maxiter': 100}
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, options=options)
    return res.x, res.status, res.nit, res.message


def allocate_bandwidth_based_on_demand(load_Mb, mask):
    '''
    Greedily allocation the channel based on the downlink demand (Q_i(t) + A_i(t))

    ### Params
        load_Mb: the downlink demand, Q_i(t) + A_i(t), shape=(n_users,)
        mask: shape=(n_users,), denote active users in the considered cluster

    ### Returns
        alpha: the optimal bs allocation, shape=(n_users,), in range [0,1]
    '''
    load_Mb_masked = load_Mb * mask
    alpha = (load_Mb_masked) / np.sum(load_Mb_masked)

    return alpha


def allocate_bandwidth_equal_to_active_users(is_active_allusers):
    alpha = is_active_allusers * 1 / np.sum(is_active_allusers)
    return alpha


def allocate_bandwidth_equal_fixed(n_users):
    return np.ones(n_users) / n_users


def allocate_bandwidth_limited(n_users):
    return np.ones(n_users)


'''
---------------------
    User Heatmap
---------------------
'''


def gen_heatmap(
    x_locations, y_locations, val, norm_val,
    n_grids=n_grids, grid_size=grid_size, boundary=boundary
):
    '''
    Generate a heatmap for the system statistic of interest.

    The heatmap should have (0,0)=top left corner\\
    (0,0) (0,1) (0,2)\\
    (1,0) (1,1) (1,2)\\
    (2,0) (2,1) (2.2)

    ### Parameters
        (x, y): the current location of each user, x.shape = y.shape = (n_users,)
        val: the network statistic of interest (e.g., queue size), shape=(n_users,)
        norm_val: a coefficient for normalization

    ### Returns
        heatmap: shape = (n_grids, n_grids)
    '''
    heatmap = 1e-6 * rng.uniform(size=(n_grids, n_grids))        # random noise
    x_locs = np.clip(x_locations, -boundary + 1, boundary - 1)
    y_locs = np.clip(y_locations, -boundary + 1, boundary - 1)

    for uid in range(x_locations.size):
        # col = int(np.minimum(2 * boundary - 1, x_locs[uid] + boundary) / grid_size)  # in range 0 -> (n_grids-1)
        # row = int(np.minimum(2 * boundary - 1, boundary - y_locs[uid]) / grid_size)  # in range 0 -> (n_grids-1)
        col = int(np.floor((x_locs[uid] + boundary) / grid_size))
        row = int(np.floor((boundary - y_locs[uid]) / grid_size))
        heatmap[row, col] += val[uid]

    heatmap /= norm_val     # normalization

    return heatmap


def create_image_heatmap(xaxis_all, yaxis_all, t, val_all, norm_val):
    heatmap = gen_heatmap(
        x_locations=xaxis_all[:, t],
        y_locations=yaxis_all[:, t],
        val=val_all,
        norm_val=norm_val,
        n_grids=n_grids,
        grid_size=grid_size,
        boundary=boundary
    )
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(heatmap, interpolation='hamming')
    ax.set(title=f't={t}s, Image: {n_grids}x{n_grids}, Grid: {grid_size}x{grid_size}m')
    plt.tight_layout()
    plt.savefig(os.path.join(os.getcwd(), 'dev', 'heatmap', f"t{t}.png"))
    plt.close()


def gen_animation_heatmap(xaxis_all, yaxis_all, t_max=100, t_step=10):
    '''Create an animation (gif image) of user heatmaps'''
    for t in range(0, t_max, t_step):
        create_image_heatmap(xaxis_all, yaxis_all, t, val_all=np.ones(xaxis_all.shape[0]), norm_val=1)
    images = []
    for t in range(0, t_max, t_step):
        image = imageio.v2.imread(os.path.join(os.getcwd(), 'dev', 'heatmap', f"t{t}.png"))
        images.append(image)
    path_to_gif = os.path.join(os.getcwd(), 'dev', 'heatmap', 'heatmap.gif')
    imageio.mimsave(path_to_gif, ims=images, fps=3)  # ims = list of input images, [fps = frames per second]

    Image(filename=path_to_gif, width=350)  # display

    return path_to_gif


'''
------------------------------------
    User's Quality of Experience
------------------------------------
'''


def mos_func(qlen, lambda_):
    ''' Estimate the Mean Opinion Score (MOS) of users
        Parameters:
            - qlen: the current queue length, scalar or array-like
            - lambda_: the expected arrival rate, scalar or array-like
        Return:
            - mos_val: scalar or array-like
        Reference: ITU-T G.1030 (02/2014), eq. (II-2) and Fig. II.4, page 17
    '''
    # a, b = 5.72, 0.936        # 2.16s to 155s
    # a, b = 13.284, 2.436      # 30s to 155s
    # a, b = 22.256, 4.215      # 60s to 155s
    # a, b = 4.27, 1.82         # 0.67s to 6s
    a = 4 * (-np.log(ss_min)) / np.log(ss_min / ss_max) + 5
    b = (-1) * 4 / np.log(ss_min / ss_max)
    mos_val = a - b * np.log((qlen + 1e-4) / (lambda_ + 1e-4))  # avoid dividing by 0 and log(0)
    mos_val_min, mos_val_max = 1.0, 5.0
    mos_val = np.minimum(mos_val, mos_val_max)
    mos_val = np.maximum(mos_val, mos_val_min)
    return mos_val


def mos_func_ss(ss_time):
    ''' MOS vs the session time (second) '''
    a = 4 * (-np.log(ss_min)) / np.log(ss_min / ss_max) + 5
    b = (-1) * 4 / np.log(ss_min / ss_max)
    mos_val = a - b * np.log(ss_time)  # avoid dividing by 0 and log(0)
    return mos_val


def qoe_fairness(mos, axis=None):
    '''Calculate QoE fairness following the F index proposed by (Hopfeld, 2017).
    Note: Users without QoE score (mos = np.NaN) are not counted.

    ### Params
        - mos: mean opinion score (MOS) of all users

    ### Returns
        - f_index: the QoE fairness index F

    ### Reference
    T. Hopfeld et al., "Definition of QoE Fairness in Shared Systems,"
    in IEEE Communications Letters, vol. 21, no. 1, pp. 184-187, Jan. 2017.
    '''
    # H, L = 5, 1     # 5-point MOS scale
    # fairness_index = 1 - (2 * np.std(mos)) / (H - L)

    fairness_index = 1 - np.nanstd(mos, axis=axis) / 2    # 5-point MOS scale: H=5, L=1

    return fairness_index


'''
-----------------------
    User Clustering
-----------------------
'''


def kmeans_clustering(loc_users, loc_uavs=None, n_clusters=n_uavs):
    '''Cluster users in to groups using KMeans Clustering, each corresponding to a UAV
    Params:
        - loc_users: positions of all users, shape=(n_users, 3)
        - loc_uavs: positions of UAVs, shape=(n_uavs, 3)
    Return:
        - cluster_mat : shape=(n_users, n_uavs), cluster_mat[i,j] = 1 -> user i is associated to uav j

    Reference: https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
    '''
    if loc_uavs is None:
        kmeans = KMeans(n_clusters=n_clusters, verbose=1).fit(loc_users)     # standard X.shape = (n_samples, n_features)
        cluster_mat = np.eye(n_clusters)[kmeans.labels_]
    else:
        kmeans = KMeans(n_clusters=n_clusters, init=loc_uavs,   # initial locations for clustering
                        n_init=1, verbose=0).fit(loc_users)     # standard X.shape = (n_samples, n_features)
        cluster_mat = np.eye(n_clusters)[kmeans.labels_]

    return cluster_mat, kmeans.labels_, kmeans.cluster_centers_


'''
---------------------
    UAV's energy
---------------------
'''


def cal_uav_propulsion_energy(V=0):
    ''' Propulsion energy model for rotary-wing UAVs.
    Reference: eq (13) in (Zeng Yong et al., 2019) Energy Minimization for
    Wireless Communication with Rotary-Wing UAV

    Args:
        V: velocity in m/s

    Return:
        p_sum: propulsion power consumption (W)
        p_blade_profile: blade profile power (W)
        p_induced: induced power (W)
        p_parasite: parasite power (W)
    '''
    Utip = 120      # tip speed of the rotor blade (m/s)
    v0 = 4.03       # mean rotor induced velocity in hover
    d0 = 0.6        # fuselage drag ratio
    rho = 1.225     # air density in kg/m3
    s = 0.05        # rotor solidity
    A = 0.503       # rotor disc area in m2 for rotor radius = 0.4 m

    # delta = 0.012   # profile drag coefficient
    # omega = 300     # blade angular velocity (radian/s)
    # R = 0.4         # rotor radius (m)
    # k = 0.1         # incremental correction factor to induced power
    # W = 20          # air craft weight in Newton

    # P0 = delta / 8 * rho * s * A * omega**3 * R**3      # eq. (64)
    # Pi = (1 + k) * W**1.5 / np.sqrt(2 * rho * A)        # eq. (64)

    P0 = 79.85628
    Pi = 88.62793

    p_blade_profile = P0 * (1 + 3 * V**2 / Utip**2)
    p_induced = Pi * np.sqrt(np.sqrt(1 + V**4 / (4 * v0**4)) - V**2 / (2 * v0**2))
    p_parasite = 0.5 * d0 * rho * s * A * V**3
    p_sum = p_blade_profile + p_induced + p_parasite

    return p_sum, p_blade_profile, p_induced, p_parasite


'''
--------------------------------
    Initial location of UAVs
--------------------------------
'''

# Case 1: 4 UAVs start from edge points

if n_uavs == 2:
    init_locations_uav = np.array([
        (-1, 1),        # UAV 1
        (1, -1),        # UAV 2
    ]) * boundary

if n_uavs == 3:
    init_locations_uav = np.array([
        (0, 1),         # UAV 1
        (-1, -1),       # UAV 2
        (1, -1),        # UAV 3
    ]) * boundary

if n_uavs == 4:
    init_locations_uav = np.array([
        (-1, 1),        # UAV 1
        (-1, -1),       # UAV 2
        (1, -1),        # UAV 3
        (1, 1)          # UAV 4
    ]) * boundary

if n_uavs == 5:
    init_locations_uav = np.array([
        (0, 1),         # UAV 1
        (-1, 0),        # UAV 2
        (1, 0),         # UAV 3
        (-1, -1),       # UAV 4
        (1, -1)         # UAV 5
    ]) * boundary

if n_uavs == 6:
    init_locations_uav = np.array([
        (-1, 1),        # UAV 1
        (0, 1),         # UAV 2
        (1, 1),         # UAV 3
        (-1, -1),       # UAV 4
        (0, -1),        # UAV 5
        (1, -1)         # UAV 6
    ]) * boundary

if n_uavs == 7:
    init_locations_uav = np.array([
        (-1, 1),        # UAV 1
        (0, 1),         # UAV 2
        (1, 1),         # UAV 3
        (-1, 0),        # UAV 4
        (-1, -1),       # UAV 5
        (0, -1),        # UAV 6
        (1, -1)         # UAV 7
    ]) * boundary

if n_uavs == 8:
    init_locations_uav = np.array([
        (-1, 1),        # UAV 1
        (-1, -1),       # UAV 2
        (1, -1),        # UAV 3
        (1, 1),         # UAV 4
        (0, 1),         # UAV 5
        (0, -1),        # UAV 6
        (1, 0),         # UAV 7
        (-1, 0)         # UAV 8
    ]) * boundary


# # Case 2: UAVs start at the first centroid (K-means clustering)
# t=0
# loc_users = np.hstack((np.expand_dims(xaxis_all[:,t], axis=1), np.expand_dims(yaxis_all[:,t], axis=1)))  # shape = (n_users, 3)
# cmat, cids, c_centroids = kmeans_clustering(loc_users, n_clusters=n_uavs)
# init_locations_uav = np.array(c_centroids)

# Print to the terminal
print("Initial location of UAVs:\n", init_locations_uav)


'''
----------------------
    For testing
----------------------
'''
if __name__ == '__main__':

    '''
    Visualize the MOS function
    '''
    qlen = np.arange(0, 15, step=1)     # Mb

    plt.figure()
    for ld in ON_data_arrival_mean_Mb:
        lambda_ = ld * np.ones(shape=(len(qlen)))     # Mbps
        mos = mos_func(qlen=qlen, lambda_=lambda_)
        df_mos = pd.DataFrame({'qlen_Mb': qlen, 'arrival_rate_Mbps': lambda_, 'mos': mos})
        sns.lineplot(data=df_mos, x='qlen_Mb', y='mos')

    df_mos.to_excel(os.path.join(sim_folder_path, 'mos-function.xlsx'))
    print(df_mos.head(10))

    plt.grid(True)
    plt.xlim(left=0)
    plt.xlabel('Queue length (Mb)')
    plt.ylabel('Mean Opinion Score (MOS)')
    plt.tight_layout()
    plt.title(f"MOS vs Queue Length, Given Arrival Rate = {lambda_[0]} Mbps")
    plt.tight_layout()
    plt.savefig(os.path.join(sim_folder_path, 'MOS-function.png'), bbox_inches='tight')
    plt.show()

    '''
    Plot the UAV's energy model
    '''
    V = np.linspace(0, 30, 31)
    p_sum, p_blade_profile, p_induced, p_parasite = cal_uav_propulsion_energy(V)
    fig, ax = plt.subplots()
    plt.plot(V, p_sum, label="total")
    plt.plot(V, p_blade_profile, label="blade profile")
    plt.plot(V, p_induced, label="induced")
    plt.plot(V, p_parasite, label="parasite")
    plt.legend()
    plt.xlabel("velocity (m/s)")
    plt.ylabel("power (W)")
    plt.title("UAV's propulsion energy model")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(sim_folder_path, 'uav-energy-model.png'))
    plt.show()

    '''
    Test bandwidth allocation using linear programming
    '''
    # n_users = 5
    # load_Mb = np.array([50, 0, 30, 4, 1])   # np.arange(0,5)+10
    # chcapa_Mb_ref = [1, 2, 1, 20, 1]
    # mask = np.ones(shape=(n_users,))
    # alpha_min = 0   # 1/(4*n_users)
    # x, status, nit, message = allocate_bandwidth_linprog(load_Mb, chcapa_Mb_ref, mask, alpha_min)
    # print(f'downlink traffic in Mb: {load_Mb}')
    # print(f"bw allocation: {x}")
    # print(f'status: {status}, {message}')
    # print(f'# of iterations: {nit}')

    '''
    Test bandwidth allocation based on the downlink demand
    '''
    # n_users_test = 5
    # load_Mb = np.array([50, 0, 30, 4, 1])
    # mask = np.array([0, 1, 0, 0, 1])
    # alpha_min = 0       # 1/(4*n_users)
    # x = allocate_bandwidth_based_on_demand(load_Mb, mask)
    # print(x)
