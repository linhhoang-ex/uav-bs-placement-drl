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


def update_queue_Mb(qlen_prev_Mb, arrival_Mb, departure_Mb):
    '''Return the updated qlen based on the previous qlen, arrival traffic, and departure rate'''
    qlen_next_Mb = np.max([qlen_prev_Mb + arrival_Mb - departure_Mb, 0])
    return qlen_next_Mb


def update_vqlen_prop(
    vqlen_curr: np.ndarray, pw_prop: np.ndarray, pw_thres: np.ndarray = PW_THRES
) -> np.ndarray:
    ''' Update the virtual queues for controlling the UAV's propulsion power.

    Params
    ------
    - vqlen_curr: the current queue length of all virtual queues
    - pw_prop: the propulsion power consumption for all UAVs
    - pw_thres: the power consumption threshold for all UAVs

    Returns
    -------
    - vqlen_next: the next state of all virtual queues
    '''
    vqlen_next = vqlen_curr + pw_prop - pw_thres
    vqlen_next = np.where(vqlen_next > 0, vqlen_next, 0)

    return vqlen_next


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


def allocate_bandwidth_limited(
    mos: np.ndarray, cids: np.ndarray, n_channels: int, n_uavs: int
):
    '''
    Allocate channels to users based on the downlink traffic.\\
    If allocated a channel, each user obtains a fixed frequency bandwidth.\\
    The # of channels that can be allocated by the UAV is limited.\\
    The UAV priority users with low MOS to allocate channels.

    ### Params
        - mos: the current QoE of all users in the previous time slot
        - cids: the cluster id of all users in the current time slot
        - n_channels: # of channels for one UAV
        - n_uavs: # of UAVs

    ### Returns
        - alpha: the channel allocation vector for all users (1 -> Yes, 0 -> No)
    '''
    alpha = np.zeros_like(mos)
    for cid in range(n_uavs):
        u_ids = np.where(cids == cid)[0]
        if len(u_ids) <= n_channels:
            alpha[u_ids] = 1
        else:
            mos_values = mos[u_ids]
            allocated = np.argsort(mos_values)[:n_channels]
            alpha[u_ids[allocated]] = 1

    return alpha


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


def mos_func(qlen: np.ndarray, lambda_: float, d0: float = d0):
    ''' Estimate the Mean Opinion Score (MOS) of users.

        Parameters
        ----------
        - qlen: the current queue length, scalar or array-like
        - lambda_: the expected arrival rate, scalar or array-like
        - d0: minimum delay for communications overhead (in second)

        Returns
        -------
        - mos_val: scalar or array-like

        References
        ----------
        ITU-T G.1030 (02/2014), eq. (II-2) and Fig. II.4, page 17
    '''
    # a, b = 5.72, 0.936        # 2.16s to 155s
    # a, b = 13.284, 2.436      # 30s to 155s
    # a, b = 22.256, 4.215      # 60s to 155s
    # a, b = 4.27, 1.82         # 0.67s to 6s
    a = 4 * (-np.log(ss_min)) / np.log(ss_min / ss_max) + 5
    b = (-1) * 4 / np.log(ss_min / ss_max)
    mos_val = a - b * np.log(d0 + qlen / np.max([lambda_, 1e-4]))  # avoid dividing by 0
    mos_val_min, mos_val_max = 1.0, 5.0
    mos_val = np.clip(mos_val, mos_val_min, mos_val_max)

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


def qoe_jain_fairness(mos, axis=None):
    '''Calculate QoE fairness following Jain's fairness index.
    Note: Users without QoE score (mos = np.NaN) are not counted.

    ### Params
        - mos: mean opinion score (MOS) of all users

    ### Returns
        - fairness_index: Jain's fairness index
    '''
    fairness_index = np.nanmean(mos, axis=axis)**2 / np.nanmean(mos**2, axis=axis)

    return fairness_index


'''
-----------------------
    User Clustering
-----------------------
'''


def kmeans_clustering(loc_users, loc_uavs=None, n_clusters=n_uavs):
    '''Cluster users in to groups using KMeans Clustering, each corresponding to a UAV

    Params
    ------
        - loc_users: positions of all users, shape=(n_users, 3)
        - loc_uavs: positions of UAVs, shape=(n_uavs, 3)

    Returns
    -------
        - cluster_mat : shape=(n_users, n_uavs), cluster_mat[i,j] = 1 -> user i is associated to uav j

    Reference: https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
    '''
    if loc_uavs is None:
        kmeans = KMeans(
            n_clusters=n_clusters,
            verbose=1
        ).fit(loc_users)     # standard X.shape = (n_samples, n_features)
        cluster_mat = np.eye(n_clusters)[kmeans.labels_]
    else:
        kmeans = KMeans(
            n_clusters=n_clusters,
            init=loc_uavs,          # initial locations for clustering
            n_init=1,
            verbose=0
        ).fit(loc_users)            # standard X.shape = (n_samples, n_features)
        cluster_mat = np.eye(n_clusters)[kmeans.labels_]

    return cluster_mat, kmeans.labels_, kmeans.cluster_centers_


def voronoi_clustering(
    loc_users: np.ndarray, loc_uavs=None, n_clusters=n_uavs
):
    pass


def zone_clustering(
    loc_users: np.ndarray, loc_uavs: np.ndarray
) -> int:
    ''' Cluster users based on predefined zones, currently with 4 UAVs only.

    Params
    ------
        - loc_users: 3D positions of all users, shape=(n_users, 3)
        - loc_uavs: positions of UAVs, shape=(n_uavs, 3)

    Returns
    -------
        - cids: the zone id of users, shape=(n_users,), id = 0 to 3
    '''
    assert loc_uavs[0][0] < 0 and loc_uavs[0][1] > 0, "Expect UAV 0 in Zone 0"
    assert loc_uavs[1][0] < 0 and loc_uavs[1][1] < 0, "Expect UAV 1 in Zone 1"
    assert loc_uavs[2][0] > 0 and loc_uavs[2][1] < 0, "Expect UAV 2 in Zone 2"
    assert loc_uavs[3][0] > 0 and loc_uavs[3][1] > 0, "Expect UAV 3 in Zone 3"

    n_users, _ = loc_users.shape
    cids = np.full(shape=n_users, fill_value=np.NaN)
    for i in range(n_users):
        xloc, yloc = loc_users[i][0], loc_users[i][1]
        if xloc < 0 and yloc >= 0:
            cids[i] = 0
        elif xloc < 0 and yloc < 0:
            cids[i] = 1
        elif xloc >= 0 and yloc < 0:
            cids[i] = 2
        elif xloc >= 0 and yloc >= 0:
            cids[i] = 3

    return cids.astype(int)


'''
---------------------
    UAV's energy
---------------------
'''


def cal_uav_propulsion_energy(V=0):
    ''' Calculate the UAV's propulsion energy consumption (in Watts).

    Params
    ------
        V: velocity in m/s

    Returns
    -------
        p_sum: propulsion power consumption (W)
        p_blade_profile: blade profile power (W)
        p_induced: induced power (W)
        p_parasite: parasite power (W)

    References
    ----------
    Eq (13) in (Zeng Yong et al., 2019), "Energy Minimization for Wireless
    Communication with Rotary-Wing UAV".
    '''
    omega = 200     # blade angular velocity (rad/s), ref = 300
    R = 0.4         # rotor radius (m)
    v0 = 4.03       # mean rotor induced velocity in hover
    d0 = 0.6        # fuselage drag ratio
    s = 0.12        # rotor solidity for hovering and low-speed UAVs, ref = 0.05
    rho = 1.225     # air density in kg/m3
    A = 3.14 * R**2     # rotor disc area in sq. meter
    Utip = omega * R    # tip speed of the rotor blade (m/s)

    W = 10          # air craft weight in Newton, ref = 20
    delta = 0.012   # profile drag coefficient
    k = 0.1         # incremental correction factor to induced power

    P0 = delta / 8 * rho * s * A * omega**3 * R**3      # eq. (64)
    Pi = (1 + k) * W**1.5 / np.sqrt(2 * rho * A)        # eq. (64)

    # P0 = 79.85628     # (Zeng Yong et al., 2019)
    # Pi = 88.62793     # (Zeng Yong et al., 2019)

    p_blade_profile = P0 * (1 + 3 * V**2 / Utip**2)
    p_induced = Pi * np.sqrt(np.sqrt(1 + V**4 / (4 * v0**4)) - V**2 / (2 * v0**2))
    p_parasite = 0.5 * d0 * rho * s * A * V**3
    p_sum = p_blade_profile + p_induced + p_parasite

    return p_sum, p_blade_profile, p_induced, p_parasite


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
        # lambda_ = ld * np.ones(shape=(len(qlen)))     # Mbps
        lambda_ = ld
        mos = mos_func(qlen=qlen, lambda_=lambda_)
        df_mos = pd.DataFrame({'qlen_Mb': qlen, 'arrival_rate_Mbps': lambda_, 'mos': mos})
        sns.lineplot(data=df_mos, x='qlen_Mb', y='mos')

    df_mos.to_excel(os.path.join(sim_out_path, 'mos-function.xlsx'))
    print(df_mos.head(10))

    plt.grid(True)
    plt.xlim(left=0)
    plt.xlabel('Queue length (Mb)')
    plt.ylabel('Mean Opinion Score (MOS)')
    plt.tight_layout()
    plt.title(f"MOS vs Queue Length, Given Arrival Rate = {lambda_} Mbps")
    plt.tight_layout()
    plt.savefig(os.path.join(sim_out_path, 'MOS-function.png'), bbox_inches='tight')
    plt.show()

    '''
    Plot the UAV's energy model
    '''
    V = np.linspace(0, 30, 121)
    p_sum, p_blade_profile, p_induced, p_parasite = cal_uav_propulsion_energy(V)
    fig, ax = plt.subplots()
    plt.plot(V, p_sum, label="Total Power")
    plt.plot(V, p_blade_profile, label="Blade Profile")
    plt.plot(V, p_induced, label="Induced")
    plt.plot(V, p_parasite, label="Parasite")
    plt.axhline(y=PW_THRES, label='Power Threhold', color='k', linestyle="--")
    plt.legend(fontsize="medium")
    plt.xlabel("UAV Velocity (m/s)", fontsize="large")
    plt.ylabel("Propulsin Power (W)", fontsize="large")
    plt.xlim(left=0, right=np.max(V))
    plt.ylim(bottom=0, top=700)
    plt.xticks(fontsize="large")
    plt.yticks(fontsize="large")
    # plt.title("UAV's propulsion energy model")
    plt.grid(True, linestyle="--")
    plt.tight_layout()
    plt.savefig(os.path.join(sim_out_path, 'uav-energy-model.pdf'))
    plt.show()

    print("\n")
    print(f'propulsion power (min) = {np.min(p_sum)}')
    print(f'propulsion power (max) = {np.max(p_sum)}')
    print(f"propulsion power (hovering) = {p_sum[0]}")
    print(f"argmin(prop pw): v (m/s) = {V[np.argmin(p_sum)]}")
    print("\n")

    '''
    Test bandwidth allocation using linear programming
    '''
    n_users = 5
    load_Mb = np.array([50, 0, 30, 4, 1])   # np.arange(0,5)+10
    chcapa_Mb_ref = [1, 2, 1, 20, 1]
    mask = np.ones(shape=(n_users,))
    alpha_min = 0   # 1/(4*n_users)
    x, status, nit, message = allocate_bandwidth_linprog(load_Mb, chcapa_Mb_ref, mask, alpha_min)
    print(f'downlink traffic in Mb: {load_Mb}')
    print(f"bw allocation: {x}")
    print(f'status: {status}, {message}')
    print(f'# of iterations: {nit}')

    '''
    Test bandwidth allocation based on the downlink demand
    '''
    n_users_test = 5
    load_Mb = np.array([50, 0, 30, 4, 1])
    mask = np.array([0, 1, 0, 0, 1])
    alpha_min = 0       # 1/(4*n_users)
    x = allocate_bandwidth_based_on_demand(load_Mb, mask)
    print(x)

    '''
    Test bandwidth allocation based on the downlink demand
    '''
    alpha = allocate_bandwidth_limited(
        mos=np.array([np.nan, 1, 3, 5, 4, 3]),
        cids=np.array([0, 1, 0, 1, 1, 0]),
        n_channels=2,
        n_uavs=2,
    )
    print(alpha)
