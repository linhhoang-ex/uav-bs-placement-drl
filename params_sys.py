import numpy as np
import os
from utils import *

rng = np.random.default_rng()
np.random.seed(0)


class User:
    def __init__(self, name) -> None:
        self.name = name
        self.x = None               # x-axis
        self.y = None               # y-axis

        self.cluster = np.zeros(shape=(n_slots))                    # assigned cluster in each slot
        self.chgains_downlink = np.zeros(shape=(n_slots))           # real number, non negative
        self.alpha = np.zeros(shape=(n_slots))                      # bandwidth allocation, in range(0,1)
        self.channel_capacity_Mb = np.zeros(shape=(n_slots))        # channel capcit (Mb/time slot)
        self.queue_length_Mb = np.zeros(shape=(n_slots))            # queue length (Mb)
        self.downlink_throughput_Mb = np.zeros(shape=(n_slots))     # download speed (Mb/time slot)

        self.arrival_traffic_Mb = np.zeros(shape=(n_slots))
        self.traffic_state = np.zeros(shape=(n_slots))
        self.active_state = np.zeros(shape=(n_slots))
        self.traffic_type = np.zeros(shape=(n_slots))

        self.delay = np.zeros(shape=(n_slots,))          # queueing delay
        self.mos = np.zeros(shape=(n_slots))             # Mean Opinion Score (MOS)
        self.ema_qlen_Mb = np.zeros(shape=(n_slots))     # Exponential Moving Average (EMA) of the queue length
        self.ema_drate_Mbps = np.zeros(shape=(n_slots))  # EMA of the data rate (arrival traffic)
        self.lambd_Mb = None                             # All-time average arrival rate (Mb)


class UAV:
    def __init__(self, name, x_init, y_init, z_init, xlim=np.array([-1, 1]),
                 ylim=np.array([-1, 1]), speedxy_max=5, speedz_max=1):
        self.name = name
        self.x = np.zeros(shape=(n_slots))      # x-axis
        self.y = np.zeros(shape=(n_slots))      # y-axis
        self.z = np.zeros(shape=(n_slots))      # z-axis (altitude)

        self.x[0] = x_init
        self.y[0] = y_init
        self.z[0] = z_init      # initial location
        self.xlim = xlim
        self.ylim = ylim        # operation area: UAV not moving out of the target zone
        self.speedxy_max = speedxy_max
        self.speedz_max = speedz_max    # maximum moving speed on Oxy (horizontal) and Oz (vertical) plane, m/s

        # For the energy consumption model
        self.speedxy = np.zeros(shape=(n_slots))
        self.speedz = np.zeros(shape=(n_slots))
        self.p_propulsion = np.zeros(shape=(n_slots))


'''
-----------------
System parameters
-----------------
'''
n_users = 100                           # number of users
n_uavs = 4                              # number of UAVs
n_slots = 1 + np.int32(1.0e3)           # no. of slots in total
slot_len = 1                            # one second, fixed
ss_min, ss_max = 0.67, 6                # in second, for the QoE model


'''
-----------------------------
Parameters for communications
-----------------------------
'''
ref_receive_pw = dB(-40)                # reference received signal strength at 1 meter
# bw_1user = MHz(1)                     # bandwidth for one user
channel_bandwidth = MHz(30)             # total bandwith for all users, reference: 12 MHz for 5 users
pTx_downlink = mW(200)                  # transmit power (fixed) in mW of the UAV
noise_pw_total = dBm(-90)               # total noise power


'''
-------------------------------------------
Parameters for the mobility model for users
-------------------------------------------
'''
boundary = 250              # reference: 250 m with 5 users
speed_user_avg = 1          # m/s

initial_distance_range_x = range(-(boundary - 10), boundary - 10, 10)
initial_distance_range_y = range(-(boundary - 10), boundary - 10, 10)

hotspot_range = 0
upper_left = (-boundary, boundary)
lower_right = (-(boundary - hotspot_range), boundary - hotspot_range)


'''
-------------------------------------
Parameters for the data traffic model
-------------------------------------
'''

# Case 1: UAV-BSs start at the initial centroids (K-means clustering), users are ON and OFF
# lambd_Mb = 8.5            # Mbps
# ON_data_arrival_mean_Mb = np.array([1/3, 2/3, 1]) * lambd_Mb # np.array([1/3, 2/3, 1]), np.array([1/2, 3/4, 1])
# ON_duration_mean_tslot = 0.3e3              # in # of time slots
# OFF_duration_mean_tslot =  0.7e3            # in # of time slots

# Case 1(b): UAV-BSs start at the initial centroids (K-means clustering), users are always ON
# lambd_Mb = 8.5            # Mbps
# ON_data_arrival_mean_Mb = np.array([1/4, 2/4, 3/4, 1]) * lambd_Mb # np.array([1/3, 2/3, 1]), np.array([1/2, 3/4, 1])
# ON_duration_mean_tslot = n_slots            # in # of time slots
# OFF_duration_mean_tslot =  1                # in # of time slots

# Case 2: UAV-BSs start at the four corners, users are always active
lambd_Mb = 2                                # Mbps
ON_data_arrival_mean_Mb = [lambd_Mb]
ON_duration_mean_tslot = n_slots            # in # of time slots
OFF_duration_mean_tslot = 1                 # in # of time slots

# Expoential Moving Average (EMA): s(t) = alpha*x(t) + (1-alpha)*s(t-1)
alpha_ema = 0.999                        # the smoothing factor of EMA

requesting_rate = ON_duration_mean_tslot / (ON_duration_mean_tslot + OFF_duration_mean_tslot)   # Rate (on average) at which users sending a download request
traffic_mean_Mb = np.mean(ON_data_arrival_mean_Mb) * requesting_rate                          # Average traffic load if a user sending a request

print(f'Requesting rate: {requesting_rate:.1f} \t\t# Rate (on average) at which users sending a download request')
print(f'Arrival traffic: {traffic_mean_Mb:.1f} Mbps \t# Average traffic load (of all services) if a user sending a request')
print('\n')


'''
----------------------
Parameters for the UAV
----------------------
'''

# UAV's altitude
z_init = 150
z_init_proposed = 150
z_min = 100
z_max = 200

# UAV's moving speed
uav_speedxy_max = 10                        # horizontal speed, in meter/second (m/s)
uav_speedz_max = 3                          # vertical speed, m/s
uav_vxy_fixed = speed_user_avg              # ~ user movement speed, 1 m/s
uav_vz_fixed = 0.5

# Coverage area of each UAV, overlaping with others
xlim_uav = np.array([(-1, 1) for i in range(0, n_uavs)]) * boundary
ylim_uav = np.array([(-0.5, 1) for i in range(0, n_uavs)]) * boundary

# Colors used for plotting, each color is used for one UAV
colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']
markers_cluster = ['o', '^', 's', 'D', 'v', 'o', '^', 's', 'D', 'v']
markers_uav = ['v', '*', '^', 'D', 'v', 'o', '^', 's', 'D', 'v']


'''
-------------------------------------
Parameters for reinforcement learning
-------------------------------------
'''
n_grids = 25
grid_size = int(2 * boundary / n_grids)     # IMPORTANT: expected shape of the DNN input: 25x25
n_decisions = 10
lyapunov_param = n_users * 500                      # parameter V in the Lyapunov framework
# t_training_start = np.max([500, ON_duration_mean_tslot*2])
t_training_start = 0
training_interval = 5       # in time slots

print(f'Coverage area (m): \t{2*boundary} x {2*boundary}')
print(f'Grid size (m): \t\t{grid_size} x {grid_size}')
print(f'Heat map image: \t{n_grids} x {n_grids}')
print('\n')


# Normalization coefficients
user_counter_norm = np.min([5.0, 20 * n_users / (n_grids * n_grids)])           # 30x the average no. of users in a grid
queue_norm_Mb = 20 * np.max(ON_data_arrival_mean_Mb)          # in Mb
ch_capacity_norm_Mb = 5 * np.max(ON_data_arrival_mean_Mb)     # 10x the arrival traffic [Mb] in a time slot

print(f'Normalization coefficients: \nuser counter norm (1 grid): \t{user_counter_norm:.1f} \nqueue_len_norm: \t{queue_norm_Mb} Mb \nch_capacity_norm: \t{ch_capacity_norm_Mb} Mb')
print('\n')


'''
-----------------------------
For exporting simulation data
-----------------------------
'''
folder_name = f"test, n_slots={n_slots}, n_users={n_users}, n_uavs={n_uavs}, lambda={lambd_Mb:.1f} Mbps, W={channel_bandwidth/1e6:.0f} MHz, v_user_avg={speed_user_avg}, vxy_uav_max={uav_speedxy_max}, uOFF={OFF_duration_mean_tslot:.0f}, k={n_decisions}"

sim_folder_path = os.path.join(os.getcwd(), "dev", folder_name)        # simulation output to be saved here

if os.path.exists(sim_folder_path) is False:
    os.mkdir(sim_folder_path)

print(f'Folder for simulation outputs: {sim_folder_path}\n')


'''
----------
For teting
----------
'''
if __name__ == '__main__':
    print(sim_folder_path)
