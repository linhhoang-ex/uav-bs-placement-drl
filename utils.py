import numpy as np
import pickle
import pandas as pd


'''
-------------------
For unit conversion
-------------------
'''


def dBm(dBm):
    '''Convert dBm -> Watts'''
    return 10**((dBm - 30) / 10)


def dB(dB):
    '''Convert dB -> real value'''
    return 10**(dB / 10)


def to_dB(x):
    '''Convert real value -> dB'''
    return 10 * np.log10(x)


def to_dBm(Watt):
    '''Convert W -> dBm'''
    return 10 * np.log10(Watt * 1e3)


def MHz(Mhz):
    '''Convert MHz -> Hz'''
    return Mhz * 10**6


def GHz(GHz):
    '''Convert GHz -> Hz'''
    return GHz * 10**9


def msec(msec):
    '''Convet msec -> seconds'''
    return msec * 10**(-3)


def Mbits(Mbits):
    '''Convert kbits -> bits'''
    return Mbits * 10**6


def to_Mbit(bit):
    '''Convert bit -> Mbit'''
    return bit / 1e6


def mW(mW):
    '''Convert mW -> W'''
    return mW * 10**(-3)


def normalize(x0):
    '''Firt-order normalization function for a vector'''
    return x0 / np.linalg.norm(x0, ord=1)


'''
-----------------------------------
For export and load simulation data
-----------------------------------
'''


def to_excel(data, fname):
    df_x = pd.DataFrame(data).transpose()
    df_x.to_excel(fname)


def save_data(obj, filepath):
    '''
    Save data to a pickle file.
    Ref: https://www.askpython.com/python/examples/save-data-in-python
    '''
    try:
        with open(filepath, "wb") as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception as ex:
        print("Error during pickling object (Possibly unsupported):", ex)


def load_data(filepath):
    '''
    Load data from a pickle file.
    Ref: https://www.askpython.com/python/examples/save-data-in-python
    '''
    try:
        with open(filepath, "rb") as f:
            return pickle.load(f)
    except Exception as ex:
        print("Error during unpickling object (Possibly unsupported):", ex)


def export_moving_average(raw_data, rolling_intv=1, min_periods=1):
    df = pd.DataFrame(np.asarray(raw_data))
    y_axis = np.hstack(
        df.rolling(
            window=rolling_intv,
            min_periods=min_periods
        ).mean().values
    )

    return y_axis


def export_moving_min(raw_data, rolling_intv=1, min_periods=1):
    df = pd.DataFrame(np.asarray(raw_data))
    y_axis = np.hstack(
        df.rolling(
            window=rolling_intv, min_periods=min_periods
        ).min().values
    )

    return y_axis


def export_moving_max(raw_data, rolling_intv=1, min_periods=1):
    df = pd.DataFrame(np.asarray(raw_data))
    y_axis = np.hstack(
        df.rolling(
            window=rolling_intv,
            min_periods=min_periods
        ).max().values
    )

    return y_axis
