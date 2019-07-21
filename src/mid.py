import glob
import os
import pandas as pd
import matplotlib.pyplot as plt
import preprocessing

RAW_PATH = "/Users/slidemorning/PycharmProjects/Golf/DataSet/Experiment/2019_1_14_2/"
EXTRACT_PATH = "/Users/slidemorning/PycharmProjects/Golf/DataSet/Extract/2019_1_14_2/"
NAMES = ['sec','g_x','g_y','g_z','a_x','a_y','a_z','m_x','m_y','m_z','nan']
arr = []
if __name__ == '__main__':
    os.chdir(EXTRACT_PATH)
    pc = preprocessing.Preprocessing()
    #for file in sorted(glob.glob("*")):
    #    filename = os.path.splitext(file)[0]
    #    print(filename)
    data = pd.read_csv(EXTRACT_PATH+"extract_190114_135158 강정훈1.csv")
    #data = data.values[:, 1:11]
    data.to_csv("/Users/slidemorning/1.csv")



