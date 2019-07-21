from scipy.signal import butter, lfilter
from pandas import DataFrame
import os
import glob
import pandas as pd
import numpy as np


NAMES = ['SEC','G_X','G_Y','G_Z','A_X','A_Y','A_Z','M_X','M_Y','M_Z','NAN']
FEATURE_NAMES = ['SEC','G_X','G_Y','G_Z','A_X','A_Y','A_Z','M_X','M_Y','M_Z']
FEATURE = 10

class Preprocessing:
    def __init__(self):
        print("Object of Preprocessing Class is Created")

    def selectMax(self, x, y):
        if x >= y:
            return x
        else:
            return y

    def lowPassFilter(self, data):
        nyq = 0.5*130.0
        normal_cutoff = 2.667/nyq
        b, a = butter(5, normal_cutoff, btype='low', analog=False)
        y = lfilter(b, a, data)
        return y

    def readRawData(self, path):
        os.chdir(path)
        data = pd.read_csv(path, delimiter=" ", names=NAMES).values
        data = data[:, 0:FEATURE]
        return data

    def extractDataAndSave_Raw_Debug(self, raw_data_path, save_path, front, rear):

        os.chdir(raw_data_path)
        for folder in sorted(glob.glob("*")):
            folder_name = os.path.splitext(folder)[0]
            print("Current Folder : "+folder_name)
            os.chdir(raw_data_path+folder_name)
            EXCEPTION = []

            for file in sorted(glob.glob("*")):

                file_name = os.path.splitext(file)[0]

                data = pd.read_csv(file, delimiter=" ", names=NAMES).values
                data = data[:, 0:FEATURE] # include second axis

                acc_x, acc_y, acc_z = list(data[:, 4]), list(data[:, 5]), list(data[:, 6])
                mag_x, mag_y, mag_z = list(data[:, 7]), list(data[:, 8]), list(data[:, 9])

                # if detect start signal on acc x, y, z sensor
                cand_acc_x = self.selectMax(max(acc_x), abs(min(acc_x)))
                cand_acc_y = self.selectMax(max(acc_y), abs(min(acc_y)))
                cand_acc_z = self.selectMax(max(acc_z), abs(min(acc_z)))

                #cand_mag_x =

                if cand_acc_x > 20 or cand_acc_y > 20 or cand_acc_z > 20:
                    if cand_acc_x > 20:
                        pass
                    elif cand_acc_y > 20:
                        pass
                    elif cand_acc_z > 20:
                        pass



    def extractDataAndSave_Raw(self, raw_data_path, save_path, front, rear):
        os.chdir(raw_data_path)
        identity = 97
        for folder in sorted(glob.glob("*")):
            folder_name = os.path.splitext(folder)[0]
            print(folder_name)
            os.chdir(raw_data_path+os.path.splitext(folder)[0])
            EXCEPTION = []
            EXCEPTION_ORDER = []
            IDX = 0
            for file in sorted(glob.glob("*")):
                IDX += 1
                filename = os.path.splitext(file)[0]
                data = pd.read_csv(file, delimiter=" ", names=NAMES).values
                data = data[:, 0:FEATURE] # SEC 포함(0)
                if abs(max(data[:, 6])) > 20 and max(data[:, 2]) > 20:  # detect impulse of a_z data
                    temp = list(data[:, 6])
                    idx = temp.index(max(temp))
                    gy = list(data[:, 2])
                    gy = self.lowPassFilter(gy)
                    threshold = max(gy) - 10
                    for i in range(idx, len(gy)):
                        if gy[i] > threshold:
                            idx = i
                            break
                    if idx < 100 or idx > 1200:
                        EXCEPTION.append(filename)
                        EXCEPTION_ORDER.append(IDX)
                    else:
                        data = data[idx+front:idx+rear, :] # determine extract data length
                        if max(gy[idx+front:idx+rear]) < 20:
                            EXCEPTION.append(filename)
                            EXCEPTION_ORDER.append(IDX)
                        elif data.shape[0] != rear-front:
                            EXCEPTION.append(filename)
                            EXCEPTION_ORDER.append(IDX)
                        else:
                            #########################
                            df = DataFrame(data)
                            df.columns = FEATURE_NAMES
                            df.to_csv(save_path+filename+".csv")
                else:
                    if max(data[:, 9]) > -10 and max(data[:, 2]) > 20:  # detect impulse of m_z data
                        temp = list(data[:, 9])
                        idx = temp.index(max(temp))
                        gy = list(data[:, 2])
                        gy = self.lowPassFilter(gy)
                        threshold = max(gy) - 10
                        for i in range(idx, len(gy)):
                            if gy[i] > threshold:
                                idx = i
                                break
                        if idx < 100 or idx > 1200:
                            EXCEPTION.append(filename)
                            EXCEPTION_ORDER.append(IDX)
                        else:
                            data = data[idx+front:idx+rear, :] # determine extract data length
                            if max(gy[idx+front:idx+rear]) < 20:
                                EXCEPTION.append(filename)
                                EXCEPTION_ORDER.append(IDX)
                            elif data.shape[0] != rear-front:
                                EXCEPTION.append(filename)
                                EXCEPTION_ORDER.append(IDX)
                            else:
                                #########################
                                df = DataFrame(data)
                                df.columns = FEATURE_NAMES
                                df.to_csv(save_path+filename+".csv")
                    else:
                        if max(data[:, 2]) < 20:
                            EXCEPTION.append(filename)
                            EXCEPTION_ORDER.append(IDX)
                        else:  # max-threshold algorithm
                            idx = 0
                            temp = list(data[:, 2])
                            temp = self.lowPassFilter(temp)
                            threshold = max(temp) - 10
                            for i in range(len(temp)):
                                if temp[i] > threshold:
                                    idx = i
                                    break
                            if idx < 100 or idx > 1200:
                                EXCEPTION.append(filename)
                                EXCEPTION_ORDER.append(IDX)
                            else:
                                data = data[idx+front:idx+rear, :] # determine extract data length
                                if max(temp[idx+front:idx+rear]) < 20:
                                    EXCEPTION.append(filename)
                                    EXCEPTION_ORDER.append(IDX)
                                elif data.shape[0] != rear-front:
                                    EXCEPTION.append(filename)
                                    EXCEPTION_ORDER.append(IDX)
                                else:
                                    #########################
                                    df = DataFrame(data)
                                    df.columns = FEATURE_NAMES
                                    df.to_csv(save_path+filename+".csv")

            if len(EXCEPTION) == 0:
                print("Complete Extract Data And Saved At "+ save_path)
                print("All data is extracted at " + save_path)
            else:
                print("Complete Extract Data And Save to " + save_path)
                print("Exception Data List : ", EXCEPTION)
                print("Exception File Number : ", EXCEPTION_ORDER)
                print(int((((IDX - len(EXCEPTION)) / IDX)) * 100), "percent extract")
            identity+=1

    def extractDataAndSave_Filter(self, raw_data_path, save_path, front, rear):
        os.chdir(raw_data_path)
        identity = 97
        for folder in sorted(glob.glob("*")):
            folder_name = os.path.splitext(folder)[0]
            print(folder_name)
            os.chdir(raw_data_path+os.path.splitext(folder)[0])
            EXCEPTION = []
            EXCEPTION_ORDER = []
            IDX = 0
            for file in sorted(glob.glob("*")):
                IDX += 1
                filename = os.path.splitext(file)[0]
                data = pd.read_csv(file, delimiter=" ", names=NAMES).values
                data = data[:, 0:FEATURE] # SEC 포함(0)
                if abs(max(data[:, 6])) > 20 and max(data[:, 2]) > 20:  # detect impulse of a_z data
                    temp = list(data[:, 6])
                    idx = temp.index(max(temp))
                    gy = list(data[:, 2])
                    gy = self.lowPassFilter(gy)
                    threshold = max(gy) - 10
                    for i in range(idx, len(gy)):
                        if gy[i] > threshold:
                            idx = i
                            break
                    if idx < 100 or idx > 1200:
                        EXCEPTION.append(filename)
                        EXCEPTION_ORDER.append(IDX)
                    else:
                        for i in range(1, 10):
                            data[:, i] = self.lowPassFilter(data[:, i])
                        data_len = data.shape[0]
                        data = data[idx+front:data_len, :] # determine extract data length
                        if max(gy[idx+front:data_len]) < 20:
                            EXCEPTION.append(filename)
                            EXCEPTION_ORDER.append(IDX)
                        elif data.shape[0] != data_len-(idx+front):
                            EXCEPTION.append(filename)
                            EXCEPTION_ORDER.append(IDX)
                        else:
                            #########################
                            df = DataFrame(data)
                            df.columns = FEATURE_NAMES
                            df.to_csv(save_path+filename+".csv")
                else:
                    if max(data[:, 9]) > -10 and max(data[:, 2]) > 20:  # detect impulse of m_z data
                        temp = list(data[:, 9])
                        idx = temp.index(max(temp))
                        gy = list(data[:, 2])
                        gy = self.lowPassFilter(gy)
                        threshold = max(gy) - 10
                        for i in range(idx, len(gy)):
                            if gy[i] > threshold:
                                idx = i
                                break
                        if idx < 100 or idx > 1200:
                            EXCEPTION.append(filename)
                            EXCEPTION_ORDER.append(IDX)
                        else:
                            for i in range(1, 10):
                                data[:, i] = self.lowPassFilter(data[:, i])
                            data_len = data.shape[0]
                            data = data[idx+front:data_len, :] # determine extract data length
                            if max(gy[idx+front:data_len]) < 20:
                                EXCEPTION.append(filename)
                                EXCEPTION_ORDER.append(IDX)
                            elif data.shape[0] != data_len-(idx+front):
                                EXCEPTION.append(filename)
                                EXCEPTION_ORDER.append(IDX)
                            else:
                                #########################
                                df = DataFrame(data)
                                df.columns = FEATURE_NAMES
                                df.to_csv(save_path+filename+".csv")
                    else:
                        if max(data[:, 2]) < 20:
                            EXCEPTION.append(filename)
                            EXCEPTION_ORDER.append(IDX)
                        else:  # max-threshold algorithm
                            idx = 0
                            temp = list(data[:, 2])
                            temp = self.lowPassFilter(temp)
                            threshold = max(temp) - 10
                            for i in range(len(temp)):
                                if temp[i] > threshold:
                                    idx = i
                                    break
                            if idx < 100 or idx > 1200:
                                EXCEPTION.append(filename)
                                EXCEPTION_ORDER.append(IDX)
                            else:
                                for i in range(1,10):
                                    data[:, i] = self.lowPassFilter(data[:, i])
                                data_len = data.shape[0]
                                data = data[idx+front:data_len, :] # determine extract data length
                                if max(temp[idx+front:data_len]) < 20:
                                    EXCEPTION.append(filename)
                                    EXCEPTION_ORDER.append(IDX)
                                elif data.shape[0] != data_len-(idx+front):
                                    EXCEPTION.append(filename)
                                    EXCEPTION_ORDER.append(IDX)
                                else:
                                    #########################
                                    df = DataFrame(data)
                                    df.columns = FEATURE_NAMES
                                    df.to_csv(save_path+filename+".csv")

            if len(EXCEPTION) == 0:
                print("Complete Extract Data And Saved At "+ save_path)
                print("All data is extracted at " + save_path)
            else:
                print("Complete Extract Data And Save to " + save_path)
                print("Exception Data List : ", EXCEPTION)
                print("Exception File Number : ", EXCEPTION_ORDER)
                print(int((((IDX - len(EXCEPTION)) / IDX)) * 100), "percent extract")
            identity+=1

    def extractDataAndSave_Raw_2(self, raw_data_path, save_path, front, rear):
        os.chdir(raw_data_path)
        identity = 97
        for folder in sorted(glob.glob("*")):
            folder_name = os.path.splitext(folder)[0]  #
            print("In " + folder_name + " folder")
            os.chdir(raw_data_path + os.path.splitext(folder)[0])
            EXCEPTION = []
            EXCEPTION_ORDER = []
            IDX = 0
            for file in sorted(glob.glob("*")):
                IDX += 1

                filename = os.path.splitext(file)[0]  # a111
                data = pd.read_csv(file, delimiter=" ", names=NAMES).values
                data = data[:, 0:FEATURE]  # SEC 포함(0)

                low_gyro_y = self.lowPassFilter(data[:, 2])
                low_gyro_y = list(low_gyro_y)
                idx = low_gyro_y.index(max(low_gyro_y))
                data = data[idx+front:idx+rear, :]
                df = DataFrame(data)
                #print(df.shape)
                if df.shape[0] == (idx+rear)-(idx+front):
                    df.columns = FEATURE_NAMES
                    df.to_csv(save_path + filename + ".csv")
                else:
                    print(filename)


            if len(EXCEPTION) == 0:
                print("Complete Extract Data And Saved At " + save_path)
                print("All data is extracted at " + save_path)
            else:
                print("Complete Extract Data And Save to " + save_path)
                print("Exception Data List : ", EXCEPTION)
                print("Exception File Number : ", EXCEPTION_ORDER)
                print(int((((IDX - len(EXCEPTION)) / IDX)) * 100), "percent extract")
            identity += 1



    # z-score(normalization) : x-min/max-min # # standardization : x-mean/std
    def standardization(self, data):
        cpy = np.copy(data)
        shape = data.shape
        for i in range(shape[1]):
            col = cpy[:, i]
            _std = col.std()
            _mean = col.mean()
            for j in range(shape[0]):
                data[j, i] = (cpy[j, i]-_mean)/_std
        return data

    def normalization(self, data):
        cpy = np.copy(data)
        shape = data.shape # shape is tuple(#rows, #cols)
        #print(shape)
        for i in range(shape[1]):
            lst = list(cpy[:, i])
            _max = max(lst)
            _min = min(lst)
            for j in range(shape[0]):
                data[j, i] = (cpy[j, i]-_min)/(_max-_min)
        return data

    def shuffleLableInfo(self, LabelInfo):
        np.random.shuffle(LabelInfo)

    def getLabelInfo(self, label_path, putting_path):
        ret = []
        label_list = []
        IDX = 0

        os.chdir(label_path)
        for label_file in sorted(glob.glob("*.txt")):
            label = pd.read_csv(label_file, delimiter=" ", names=['order', 'result']).values
            label_list.append(label)

        #print(label_list)
        identity = -1
        os.chdir(putting_path)
        for file in sorted(glob.glob("*.csv")):
            file_name = os.path.splitext(file)[0]
            #print(file_name, end=" ")
            length = len(file_name)
            char = file_name[length-1]
            if identity != ord(char)-97:
                IDX = 0
            identity = ord(char)-97
            #print(IDX)
            #print("identity : ", identity)
            #print(file_name)
            #file_order = int(file_name[1:length])
            #print(file_order)
            file_label = label_list[identity][IDX][1]
            if file_label == 2:
                file_label = 1
                tmp = [file_name, file_label]
                ret.append(tmp)
            else:
                tmp = [file_name, file_label]
                ret.append(tmp)
            IDX+=1
        print ("return np array which shape is ", np.array(ret).shape)
        #(960, 2)
        # miss : 482
        # hit : 427
        # holein : 51
        return np.array(ret)

    def getHashTable(self, label_path):
        ret = {}
        os.chdir(label_path)
        for file in sorted(glob.glob("*.txt")):
            temp = pd.read_csv(file, delimiter=" ", names=['order', 'result']).values
            idx = 111
            file_name = os.path.splitext(file)[0]
            file_name = file_name[0]
            for i in range(temp.shape[0]):
                identity = file_name + str(idx)
                result = temp[i, 1]
                ret.update({identity : result})
                idx+=1
        #print (ret)
        return ret




if __name__ == '__main__':

    READ_PATH = "/Users/slidemorning/PycharmProjects/Golf/DataSet/GolfDataSet/"
    SAVE_PATH = "/Users/slidemorning/PycharmProjects/Golf/DataSet/NewDataSet/"
    LABEL_PATH = "/Users/slidemorning/PycharmProjects/Golf/DataSet/Result2/"
    EXTRACT_PATH = "/Users/slidemorning/PycharmProjects/Golf/DataSet/Raw_Putting_150/"

    pc = Preprocessing()
    pc.extractDataAndSave_Filter(READ_PATH, SAVE_PATH, -120, 10)
    #a = pc.getLabelInfo(LABEL_PATH, EXTRACT_PATH)
    #dic = pc.getHashTable(LABEL_PATH)



    #print(len(dic))
    #print(a)
