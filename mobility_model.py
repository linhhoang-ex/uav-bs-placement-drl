import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng()


def deg2rad(deg):
    ''' Convert degree to radian '''
    return deg * np.pi / 180


def rad2deg(rad):
    '''Convert radian to degree'''
    return rad * 180 / np.pi


def check_out_of_boundary(x, y, boundary=150):
    '''Return True if the location (x,y) is out of the area'''
    return True if np.abs(x) >= boundary or np.abs(y) >= boundary else False


def adjust_location(loc, bound=150):
    ''' Force the location to be in in boundary in case the user move out of the boundary'''
    if loc >= bound:
        return bound
    if loc <= -bound:
        return -bound
    return loc


def adjust_altitude(z, zmin=50, zmax=150):
    ''' Force the UAV altitude within zmin and zmax'''
    z = np.minimum(z, zmax)
    z = np.maximum(z, zmin)
    return z


def stop_wandering(x, y, upper_left, lower_right):
    '''Return True if the location (x,y) of the user is in the hot spot area -> very likely that no more movement will be made'''
    return True if upper_left[0] <= x <= lower_right[0] and lower_right[1] <= y <= upper_left[1] else False


def generate_mobility_GM(x0=0, y0=0, n_slots=1000, slot_len=1, boundary=150, upper_left=(-150, 150), lower_right=(-100, 100), speed_avg=3):
    '''
    Gauss-Markov Mobility model
    Arguments:
        x0, y0: initial locations in meters
        n_slots: # of slots
        slot_len: slot length in second
        boundary: the boundary of the area in meters
        upper_left and lower_right: the corners that define the hot spot
    Return:
        x, y: real-time location, shape=(n_slots,)
    Reference:
        2014. An enhanced Gauss-Markov mobility model for simulations of unmanned aerial ad hoc networks
    '''
    randomness_deg = 0.5                        # degree of randomness, 0 -> fully random, 1 -> no randomness
    # speed_avg = 3                             # meters per second
    std_speed = speed_avg / 3                   # standard deviation of the random variable for the speed

    direction_dev_avg = 0                       # in degree, the mean direction deviation
    std_direction_dev = 10                      # in degree, the standard deviation of the random variable for the direction deviation

    direction_dev_avg_oob = 30                  # similar to direction_dev_avg, but in case out of boundary
    std_direction_dev_oob = 5                   # similar to std_direction_dev, but in case out of boundary

    movement_prob_wandering = 1.0                # prob. that the user make a movement when not in the hot spot
    # movement_prob_hotspot = 0.10               # movement probability in the hot spot

    x = np.empty(shape=(n_slots), dtype='float')
    y = np.empty(shape=(n_slots), dtype='float')
    x.fill(np.nan)
    y.fill(np.nan)

    speed = np.empty(shape=(n_slots))               # speed
    direction_dev_deg = np.empty(shape=(n_slots))   # direction deviation
    direction_deg = np.empty(shape=(n_slots))       # direction
    is_out_of_boundary = np.empty(shape=(n_slots))  # check whether the current location is out of the boundary
    speed.fill(np.nan)
    direction_dev_deg.fill(np.nan)
    direction_deg.fill(np.nan)
    is_out_of_boundary.fill(np.nan)

    x[0], y[0] = x0, y0
    speed[0] = speed_avg
    direction_deg[0] = rng.choice(10 * (np.arange(360) // 10))
    direction_dev_deg[0] = 0    # in degree

    for slot in range(1, n_slots):
        if check_out_of_boundary(x[slot - 1], y[slot - 1], boundary) is False:
            dire_dev_avg = direction_dev_avg
            std_dir_dev = std_direction_dev
        else:
            dire_dev_avg = direction_dev_avg_oob
            std_dir_dev = std_direction_dev_oob 

        # # movement_prob = movement_prob_hotspot if stop_wandering(x[slot-1], y[slot-1], upper_left, lower_right) == True else movement_prob_wandering
        # if stop_wandering(x[slot-1], y[slot-1], upper_left, lower_right):
        #     movement_prob = movement_prob_hotspot
        # else:
        #     movement_prob = movement_prob_wandering
        movement_prob = movement_prob_wandering

        if rng.random() <= movement_prob:        # the user make a movement 
            speed[slot] = randomness_deg * speed[slot - 1] + (1 - randomness_deg) * speed_avg + \
                np.sqrt(1 - randomness_deg**2) * rng.normal(loc=0, scale=std_speed)
            direction_dev_deg[slot] = randomness_deg * direction_dev_deg[slot - 1] + (1 - randomness_deg) * dire_dev_avg + \
                np.sqrt(1 - randomness_deg**2) * rng.normal(loc=0, scale=std_dir_dev)
            direction_deg[slot] = (direction_deg[slot - 1] + direction_dev_deg[slot]) % 360
            x[slot] = x[slot - 1] + speed[slot] * np.sin(deg2rad(direction_deg[slot])) * slot_len
            y[slot] = y[slot - 1] + speed[slot] * np.cos(deg2rad(direction_deg[slot])) * slot_len
            x[slot] = adjust_location(x[slot], bound=boundary)
            y[slot] = adjust_location(y[slot], bound=boundary)
        else:                                   # the user does not make a movement
            speed[slot] = speed[slot - 1]
            direction_dev_deg[slot] = direction_dev_deg[slot - 1]
            direction_deg[slot] = direction_deg[slot - 1]
            x[slot] = x[slot - 1]
            y[slot] = y[slot - 1]

    return (x, y)


if __name__ == '__main__':
    n_slots = np.int32(1.0e3)   # no. of slots
    boundary = 300
    x0, y0 = boundary - 5, boundary - 5            # initial location
    # x0, y0 = 0, 0

    x, y = generate_mobility_GM(x0=x0, y0=y0, n_slots=n_slots, boundary=boundary)

    """Plot real-time locations"""
    fig, ax = plt.subplots(figsize=(5, 5))
    plt.plot(x, y)
    plt.plot(x[0], y[0], 'sr', label="initial location")
    plt.legend()
    plt.grid(True)
    ax.set(xlabel='x (m)', ylabel='y (m)', title='Location of one user over time')
    plt.xlim(-(boundary + 5), boundary + 5)
    plt.ylim(-(boundary + 5), boundary + 5)
    plt.show()

    """Plot radius distance"""
    radius = np.sqrt(x**2 + y**2)       # shape = (n_slots,)
    plt.plot(radius)
    plt.title('Radius distance (m)')
    plt.grid(True)
    plt.show()
