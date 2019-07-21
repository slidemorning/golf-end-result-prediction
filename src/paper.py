import os
import glob
import pandas as pd
import numpy as np
import preprocessing
import matplotlib.pyplot as plt

DATA_PATH = "/Users/slidemorning/PycharmProjects/Golf/DataSet/GolfDataSet/c/"
NAMES = ['SEC','G_X','G_Y','G_Z','A_X','A_Y','A_Z','M_X','M_Y','M_Z','NAN']

def printGraph_10(data, seed):
    plt.subplot(2, 5, 1)
    plt.plot(data[seed + 0], 'r')
    #plt.plot(LGY[seed + 0], 'b')
    plt.subplot(2, 5, 2)
    plt.plot(data[seed + 1], 'b')
    #plt.plot(LGY[seed + 1], 'b')
    plt.subplot(2, 5, 3)
    plt.plot(data[seed + 2], 'r')
    #plt.plot(LGY[seed + 2], 'b')

    plt.subplot(2, 5, 4)
    plt.plot(data[seed + 3], 'r')
    #plt.plot(LGY[seed + 3], 'b')
    plt.subplot(2, 5, 5)
    plt.plot(data[seed + 4], 'r')
    #plt.plot(LGY[seed + 4], 'b')
    plt.subplot(2, 5, 6)
    plt.plot(data[seed + 5], 'r')
    #plt.plot(LGY[seed + 5], 'b')
    plt.subplot(2, 5, 7)
    plt.plot(data[seed + 6], 'r')
    #plt.plot(LGY[seed + 6], 'b')
    plt.subplot(2, 5, 8)
    plt.plot(data[seed + 7], 'r')
    #plt.plot(LGY[seed + 7], 'b')
    plt.subplot(2, 5, 9)
    plt.plot(data[seed + 8], 'r')
    #plt.plot(LGY[seed + 8], 'b')
    plt.subplot(2, 5, 10)
    plt.plot(data[seed + 9], 'r')
    #plt.plot(LGY[seed + 9], 'b')

    plt.show()


def standardization(data):
    cpy = np.copy(data)
    shape = data.shape
    _std = data.std()
    _mean = data.mean()
    for i in range(shape[0]):
        data[i] = (cpy[i]-_mean)/_std
    return data


GX, AX, MX = [], [], []
GY, AY, MY = [], [], []
GZ, AZ, MZ = [], [], []

gx, gy, gz, ax, ay, az, mx, my, mz = range(9)

if __name__ == '__main__':

    os.chdir(DATA_PATH)
    pc = preprocessing.Preprocessing()
    for file in sorted(glob.glob("*")):
        data = pd.read_csv(file, delimiter=" ", names=NAMES).values
        data = data[:, 1:10]
        GX.append(pc.lowPassFilter(data[:, gx]))
        AX.append(data[:, ax])
        MX.append(data[:, mx])
        GY.append(pc.lowPassFilter(data[:, gy]))
        AY.append(data[:, ay])
        MY.append(data[:, my])
        GZ.append(pc.lowPassFilter(data[:, gz]))
        AZ.append(data[:, az])
        MZ.append(data[:, mz])

    SEED = 20
    printGraph_10(MY, SEED)