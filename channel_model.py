import numpy as np


def prob_LoS(elev_angle_deg):
    """
    Calculate the LoS probability based on the elevation angle, theta (in degree)
    """
    a, b = 12.08, 0.11
    # a, b = 9.61, 0.16

    return 1 / (1 + a * np.exp(-b * (elev_angle_deg - a)))


def path_loss_exponent(elev_angle_deg, prob_LoS=prob_LoS):
    """
    Calculate the pass loss exponent based on the elevation angle, theta (in degree)
    """
    a, b = -1.5, 3.5

    return a * prob_LoS(elev_angle_deg) + b


def rician_factor(elev_angle_deg):
    """
    Calculate the rician factor from the elevation angle in degree
    """
    a, b = 1, 1.72
    elev_angle_rad = np.pi / 180 * elev_angle_deg

    return a * np.exp(b * elev_angle_rad)


def channel_fading_gain_mean(elev_angle_deg, rician_factor=rician_factor, prob_LoS=prob_LoS):
    """
    Calculate the channel fading gain (a summ of both LoS and NLoS links) from the elevation angle in degree
    """
    # chgain_LoS = 2 + 2*rician_factor(elev_angle_deg)
    # chgain_NLoS = 1
    # LoS_prob = prob_LoS(elev_angle_deg)
    # chgain_avg = LoS_prob * chgain_LoS + (1-LoS_prob) * chgain_NLoS

    LoS_prob = prob_LoS(elev_angle_deg)
    zeta = 0.2
    chgain_avg = LoS_prob + (1 - LoS_prob) * zeta

    return chgain_avg


if __name__ == '__main__':
    elev_angle_deg = np.array([0, 30, 45, 90])
    print("elevation angle: ", elev_angle_deg)
    print("LoS prob: ", prob_LoS(elev_angle_deg))
    print("path loss exponent: ", path_loss_exponent(elev_angle_deg))
    print("Rician factor: ", rician_factor(elev_angle_deg))
    print("Channel fading gain (LoS+NLoS): ", channel_fading_gain_mean(elev_angle_deg))
