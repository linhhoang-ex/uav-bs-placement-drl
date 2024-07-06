import numpy as np
from params_sys import *


def generate_traffic_ON_OFF(
    n_slots=1000,
    ON_duration_mean_tslot=30,
    OFF_duration_mean_tslot=60,
    data_arrival_mean_Mb=[1.5, 5.0]
):
    '''
    Generate the downlink traffic using the ON-OFF model.
    The ON (OFF) duration follows exponential distribution.
        - during ON state: Pareto distributed file size in each time slot
        - during OFF state: no arrival data

    Parameters:
        - ON_duration_mean_tslot (in # of time slots)
        - OFF_duration_mean_tslot (in # of time slots)
        - data_arrival_mean_Mb (in Mbits), as list[float]

    Returns:
        - arrival_traffic: shape=(n_slots,)
    '''
    # Generate the ON/OFF state and service type of the incoming traffic
    ON, OFF = True, False         # state = 1 -> ON and vice versa
    traffic_state = np.full(shape=(n_slots), fill_value=False, dtype=bool)
    traffic_state[0] = ON if rng.uniform() <= ON_duration_mean_tslot / (ON_duration_mean_tslot + OFF_duration_mean_tslot) else OFF

    n_types = len(data_arrival_mean_Mb)             # no. of service types
    traffic_type_fixed = np.random.choice(n_types)

    traffic_type = np.full(shape=(n_slots), fill_value=-1, dtype=int)       # -1 indicates the OFF state
    traffic_type[0] = traffic_type_fixed if traffic_state[0] == ON else -1

    traffic_mu = np.zeros(shape=(n_slots))          # mean of arrival traffic over time, corresponding to each service type
    traffic_mu[0] = data_arrival_mean_Mb[traffic_type[0]] if traffic_state[0] == ON else 0

    t_start = 0
    t_stop = 0
    n_samples_exponential = 1
    n_samples_pareto = 1

    while t_stop < n_slots:

        if traffic_state[t_start] == ON:
            samples = rng.exponential(scale=ON_duration_mean_tslot, size=n_samples_exponential)
        else:
            samples = rng.exponential(scale=OFF_duration_mean_tslot, size=n_samples_exponential)

        duration = np.int32(np.mean(samples))
        t_stop = t_start + duration

        traffic_state[t_start:np.min([t_stop, n_slots])] = traffic_state[t_start]
        traffic_type[t_start:np.min([t_stop, n_slots])] = traffic_type[t_start]
        traffic_mu[t_start:np.min([t_stop, n_slots])] = traffic_mu[t_start]

        # Update t_start
        if t_stop < n_slots:
            t_start = t_stop
            traffic_state[t_start] = ~np.array(traffic_state[t_stop - 1], dtype=bool)     # revert the traffic state
            traffic_type[t_start] = traffic_type_fixed if traffic_state[t_start] == ON else -1
            traffic_mu[t_start] = data_arrival_mean_Mb[traffic_type[t_start]] if traffic_state[t_start] == ON else 0

    # Generate Pareto distributed variables
    # Ref: https://towardsdatascience.com/generating-pareto-distribution-in-python-2c2f77f70dbf
    alpha = 2.0                                                 # Pareto's shape coefficient: alpha > 0 (real)
    np.expand_dims(traffic_mu, axis=0)                          # shape=(1, n_slots)
    x_m = traffic_mu * (alpha - 1) / alpha                      # Pareto's scale coefficient: x_m > 0 (real), shape=(1, n_slots)
    traffic_arrival = (rng.pareto(alpha, size=(n_samples_pareto, n_slots)) + 1) * x_m   # shape=(n_samples_pareto, n_slots)
    traffic_arrival = np.mean(traffic_arrival, axis=0)          # shape=(n_slots,)
    traffic_arrival = traffic_arrival * traffic_state           # shape=(n_slots,)

    # bounded Pareto
    bound = 3 * traffic_mu
    traffic_arrival = np.where(traffic_arrival <= bound, traffic_arrival, bound)

    return traffic_state, traffic_arrival, traffic_type


if __name__ == "__main__":
    n_slots = 20
    ON_duration_mean_tslot = 3
    OFF_duration_mean_tslot = 3
    data_arrival_mean_Mb = 5    
    traffic_arrival = generate_traffic_ON_OFF(
        n_slots=n_slots,
        ON_duration_mean_tslot=ON_duration_mean_tslot,
        OFF_duration_mean_tslot=OFF_duration_mean_tslot,
        data_arrival_mean_Mb=data_arrival_mean_Mb
    )
    print(traffic_arrival)
