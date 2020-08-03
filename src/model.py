import os
import glob
import keras
from keras.utils import np_utils
import pandas
import matplotlib.pyplot as plt
import numpy as np
from keras import Sequential
from keras.layers import Dense, LSTM, Flatten, Dropout
from keras.layers.convolutional import Conv2D, MaxPooling2D, Conv1D, MaxPooling1D
import preprocessing

# Define Path
LABEL_2_PATH = "/Users/slidemorning/PycharmProjects/Golf/DataSet/Result2/"
LABEL_6_PATH = "/Users/slidemorning/PycharmProjects/Golf/DataSet/Result6/"
LABEL_3_3_PATH = "/Users/slidemorning/PycharmProjects/Golf/DataSet/Label_3_3/"
RP_PATH = "/Users/slidemorning/PycharmProjects/Golf/DataSet/NewDataSet/"
FP_PATH = "/Users/slidemorning/PycharmProjects/Golf/DataSet/FP150/"

##################################
SAMPLE_FLAG = False
CLASS = 9
FEATURE = 9
DATA_PATH = RP_PATH
LABEL_PATH = LABEL_3_3_PATH

num_total_data = 939
LENGTH = 100
##################################

# Callback
class CustomHistory(keras.callbacks.Callback):
    def init(self):
        self.train_loss = []
        self.val_loss = []
        self.train_acc = []
        self.val_acc = []

    def on_epoch_end(self, batch, logs={}):
        self.train_loss.append(logs.get('loss'))
        self.val_loss.append(logs.get('val_loss'))
        self.train_acc.append(logs.get('acc'))
        self.val_acc.append(logs.get('val_acc'))

# Create Preprocessing Object
pc = preprocessing.Preprocessing()

# Get Label & Data Information
hash_table = pc.getHashTable(LABEL_PATH)

# Get Label Information

label_info = []
x_total = np.zeros((num_total_data, LENGTH, FEATURE), dtype='float32')
y_total = np.zeros((num_total_data, 1), dtype='int32')

os.chdir(DATA_PATH)
for putting_file in sorted(glob.glob("*.csv")):
    file_name = os.path.splitext(putting_file)[0] # a111
    tmp = [file_name, hash_table[file_name]]
    label_info.append(tmp)
label_info = np.array(label_info)
print("Get label infor : ",label_info.shape)

# Shuffle Label & Data Information
print("Shuffle Data Set")
pc.shuffleLableInfo(label_info)
#print(label_info)

# Get X, Y data Using Label & Data Information
gx, gy, gz, ax, ay, az, mx, my, mz = 0, 1, 2, 3, 4, 5, 6, 7, 8
x_total = np.zeros((num_total_data, LENGTH, FEATURE), dtype='float32')
y_total = np.zeros((num_total_data, 1), dtype='int32')
for i in range(num_total_data):
    x = pandas.read_csv(DATA_PATH+str(label_info[i, 0])+".csv", delimiter=",").values
    x = x[:, 2:11]
    #print(x.shape)
    # RAW or Normalization or Standardization
    #print(label_info[i, 0])
    x_total[i] = x
    y_total[i] = label_info[i, 1]




#print(y_total)
print("Get X data had shape ", x_total.shape)
print("Get X data had shape ", y_total.shape)

# One Hot Encoding
print("One Hot Encoding")
y_total = np_utils.to_categorical(y_total)
print("After One Hot Encoding y shape is", y_total.shape)

# Divide Data Set : 1)Train Set, 2)Validation Set, 3)Test Set
x_train = x_total[:700]
y_train = y_total[:700]
x_validation = x_total[700:num_total_data]
y_validation = y_total[700:num_total_data]
#x_test = x_total[860:num_total_data]
#y_test = y_total[860:num_total_data]

### Sampling ###
if SAMPLE_FLAG:
    LENGTH = LENGTH/10
    # Train Data
    print("Train Data Sampling")
    xy_sample_train = []
    for i in range(len(x_train)):
        for j in range(10):
            tmp = j
            x_sample = []
            for k in range(15):
                x_sample.append(x_train[i, tmp])
                tmp += 10
            xy_sample_train.append([x_sample, y_train[i]])
    xy_sample_train = np.array(xy_sample_train)
    print("Get xy_sample_train :", xy_sample_train.shape)
    print("Shuffle Train Data")
    np.random.shuffle((xy_sample_train))
    x_train = xy_sample_train[:, 0]
    y_train = xy_sample_train[:, 1]
    x_train = np.array(list(x_train)).reshape(7000, 15, 9)
    y_train = np.array(list(y_train)).reshape(7000, 6)
    print("x_train :", x_train.shape)
    print("y_train :", y_train.shape)


    #Validation Data
    xy_sample_validation = []
    for i in range(len(x_validation)):
        for j in range(10):
            tmp = j
            x_sample = []
            for k in range(15):
                x_sample.append(x_validation[i, tmp])
                tmp += 10
            xy_sample_validation.append([x_sample, y_validation[i]])
    xy_sample_validation = np.array(xy_sample_validation)
    print("Get xy_sample_test :", xy_sample_validation.shape)
    print("Shuffle Test Data")
    np.random.shuffle((xy_sample_validation))
    x_validation = xy_sample_validation[:, 0]
    y_validation = xy_sample_validation[:, 1]
    x_validation = np.array(list(x_validation)).reshape(1590, 15, 9)
    y_validation = np.array(list(y_validation)).reshape(1590, 6)



# Model Configure

model = Sequential()

model.add(LSTM(32, input_shape=(LENGTH, FEATURE)))
model.add(Dense(16, activation='relu'))
model.add(Dense(CLASS, activation='softmax'))
model.compile(loss='categorical_crossentropy',optimizer='adam', metrics=['accuracy'])





custom_hist = CustomHistory()
custom_hist.init()

for epoch_idx in range(100):
    print ('epochs : ' + str(epoch_idx) )
    model.fit(x_train, y_train, epochs=1, batch_size=50, callbacks=[custom_hist], validation_data=(x_validation, y_validation))


fig, loss_ax = plt.subplots()

acc_ax = loss_ax.twinx()

loss_ax.plot(custom_hist.train_loss, 'y', label='train loss')
loss_ax.plot(custom_hist.val_loss, 'r', label='val loss')

acc_ax.plot(custom_hist.train_acc, 'b', label='train acc')
acc_ax.plot(custom_hist.val_acc, 'g', label='val acc')

loss_ax.set_xlabel('epoch')
loss_ax.set_ylabel('loss')
acc_ax.set_ylabel('accuray')

loss_ax.legend(loc='upper left')
acc_ax.legend(loc='lower left')

plt.show()

#scores = model.evaluate(x_test, y_test)
#print("%s: %.2f%%" %(model.metrics_names[1], scores[1]*100))
