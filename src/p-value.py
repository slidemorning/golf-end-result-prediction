'''
    Test p-value using t test
'''

import preprocessing
from scipy import stats
import pandas
import numpy as np
import glob
import os


LABEL_PATH = "/Users/slidemorning/PycharmProjects/Golf/DataSet/Result6/"
PUTTING_PATH = "/Users/slidemorning/PycharmProjects/Golf/DataSet/FP150/"

pc = preprocessing.Preprocessing()
hash_table = pc.getHashTable(LABEL_PATH)
#label_info = pc.getLabelInfo(LABEL_PATH, PUTTING_PATH)
#num_total_data = label_info.shape[0]

order = 0
#for i in range(num_total_data):
    #print(order, " "+label_info[i,0])
    #order+=1

# Get X, Y data Using Label & Data Information
num_total_data = 951
LENGTH = 150
FEATURE = 9
label_info = []
x_total = np.zeros((num_total_data, 150, FEATURE), dtype='float32')
y_total = np.zeros((num_total_data, 1), dtype='int32')
os.chdir(PUTTING_PATH)
c0, c1, c2, c3, c4, c5 = 0, 0, 0, 0, 0, 0
for putting_file in sorted(glob.glob("*.csv")):
    file_name = os.path.splitext(putting_file)[0] # a111
    tmp = [file_name, hash_table[file_name]]
    label_info.append(tmp)
    if hash_table[file_name] == 0:
        c0 += 1
    elif hash_table[file_name] == 1:
        c1 += 1
    elif hash_table[file_name] == 2:
        c2 += 1
    elif hash_table[file_name] == 3:
        c3 += 1
    elif hash_table[file_name] == 4:
        c4 += 1
    elif hash_table[file_name] == 5:
        c5 +=1

print(c0)
print(c1)
print(c2)
print(c3)
print(c4)
print(c5)
label_info = np.array(label_info)
#print(label_info)
print(label_info.shape)


'''
#
gx, gy, gz, ax, ay, az, mx, my, mz = range(9)
index_list = []
identity = 'z'
for i in range(num_total_data):
    file_name = label_info[i, 0]
    if identity != file_name[0]:
        identity = file_name[0]
        index_list.append(i)
        print("person : "+identity+" satrt index : ", i)

print(len(index_list))
index_list.append(951)
mother_table = np.zeros((951, 9), dtype='float32')

for i in range(0, 10):
    for j in range(index_list[i], index_list[i+1]):
        x = pandas.read_csv(PUTTING_PATH+label_info[j, 0]+".csv", delimiter=",").values
        x = x[:, 2:11]
        for k in range(9):
            mother_table[j, k] = x[:, k].mean()
print(mother_table.shape)
print(mother_table)

# sensor : gx
sensor_pval_list = []
gx_pval, gy_pval, gz_pval = 0.0, 0.0, 0.0
ax_pval, ay_pval, az_pval = 0.0, 0.0, 0.0
mx_pval, my_pval, mz_pval = 0.0, 0.0, 0.0
for i in range(951):
    gx_pval += stats.ttest_1samp(mother_table[:, 0], mother_table[i, 0])[1]
    gy_pval += stats.ttest_1samp(mother_table[:, 1], mother_table[i, 1])[1]
    gz_pval += stats.ttest_1samp(mother_table[:, 2], mother_table[i, 2])[1]
    ax_pval += stats.ttest_1samp(mother_table[:, 3], mother_table[i, 3])[1]
    ay_pval += stats.ttest_1samp(mother_table[:, 4], mother_table[i, 4])[1]
    az_pval += stats.ttest_1samp(mother_table[:, 5], mother_table[i, 5])[1]
    mx_pval += stats.ttest_1samp(mother_table[:, 6], mother_table[i, 6])[1]
    my_pval += stats.ttest_1samp(mother_table[:, 7], mother_table[i, 7])[1]
    mz_pval += stats.ttest_1samp(mother_table[:, 8], mother_table[i, 8])[1]
sensor_pval_list.append(gx_pval/951)
sensor_pval_list.append(gy_pval/951)
sensor_pval_list.append(gz_pval/951)
sensor_pval_list.append(ax_pval/951)
sensor_pval_list.append(ay_pval/951)
sensor_pval_list.append(az_pval/951)
sensor_pval_list.append(mx_pval/951)
sensor_pval_list.append(my_pval/951)
sensor_pval_list.append(mz_pval/951)
print(sensor_pval_list)

person_a = np.zeros((97-0,150,9), dtype='float32')
person_b = np.zeros((188-97,150,9), dtype='float32')
person_c = np.zeros((277-188,150,9), dtype='float32')
person_d = np.zeros((372-277,150,9), dtype='float32')
person_e = np.zeros((467-372,150,9), dtype='float32')
person_f = np.zeros((563-467,150,9), dtype='float32')
person_g = np.zeros((656-563,150,9), dtype='float32')
person_h = np.zeros((755-656,150,9), dtype='float32')
person_i = np.zeros((852-755,150,9), dtype='float32')
person_j = np.zeros((951-852,150,9), dtype='float32')

person_a = x_total[0:97] # 0-96
person_b = x_total[97:188] # 97-187
person_c = x_total[188:277] # 187-276
person_d = x_total[277:372] # 277-371
person_e = x_total[372:467] # 372-466
person_f = x_total[467:563] # 467-562
person_g = x_total[563:656] # 563-655
person_h = x_total[656:755] # 656-754
person_i = x_total[755:852] # 755-851
person_j = x_total[852:950] # 852-949

pval_gx, pval_gy, pval_gz = 0.0, 0.0, 0.0
pval_ax, pval_ay, pval_az = 0.0, 0.0, 0.0
pval_mx, pval_my, pval_mz = 0.0, 0.0, 0.0

print("person : a")
print()
printPValue(person_a.shape[0])
'''