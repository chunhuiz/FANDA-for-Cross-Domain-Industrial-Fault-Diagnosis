import numpy as np
import scipy.io as sio
from sklearn import preprocessing


class CWRU_data():
    def __init__(self, source, target, length_signal = 1200):
        self.source = source
        self.target = target
        self.length_signal = length_signal

        # Normal
        normal0 = sio.loadmat('97.mat')['X097_DE_time'] # 1797
        normal_1797_X = normal0[:-(len(normal0)%self.length_signal)].reshape(-1, self.length_signal)

        normal1 = sio.loadmat('98.mat')['X098_DE_time'] # 1772
        normal_1772_X = normal1[:-(len(normal1)%self.length_signal)].reshape(-1, self.length_signal)

        normal2 = sio.loadmat('99.mat')['X099_DE_time'] # 1750
        normal_1750_X = normal2[:-(len(normal2)%self.length_signal)].reshape(-1, self.length_signal)

        normal3 = sio.loadmat('100.mat')['X100_DE_time'] # 1730
        normal_1730_X = normal3[:-(len(normal3)%self.length_signal)].reshape(-1, self.length_signal)

        # 0.007
        # Fault1
        IF_7_1797 = sio.loadmat('105.mat')['X105_DE_time']
        IF_7_1797_X = IF_7_1797[:-(len(IF_7_1797)%self.length_signal)].reshape(-1, self.length_signal)

        IF_7_1772 = sio.loadmat('106.mat')['X106_DE_time']
        IF_7_1772_X = IF_7_1772[:-(len(IF_7_1772)%self.length_signal)].reshape(-1, self.length_signal)

        IF_7_1750 = sio.loadmat('107.mat')['X107_DE_time']
        IF_7_1750_X = IF_7_1750[:-(len(IF_7_1750)%self.length_signal)].reshape(-1, self.length_signal)

        IF_7_1730 = sio.loadmat('108.mat')['X108_DE_time']
        IF_7_1730_X = IF_7_1730[:-(len(IF_7_1730)%self.length_signal)].reshape(-1, self.length_signal)

        # Fault2
        BF_7_1797 = sio.loadmat('118.mat')['X118_DE_time']
        BF_7_1797_X = BF_7_1797[:-(len(BF_7_1797)%self.length_signal)].reshape(-1, self.length_signal)

        BF_7_1772 = sio.loadmat('119.mat')['X119_DE_time']
        BF_7_1772_X = BF_7_1772[:-(len(BF_7_1772)%self.length_signal)].reshape(-1, self.length_signal)

        BF_7_1750 = sio.loadmat('120.mat')['X120_DE_time']
        BF_7_1750_X = BF_7_1750[:-(len(BF_7_1750)%self.length_signal)].reshape(-1, self.length_signal)

        BF_7_1730 = sio.loadmat('121.mat')['X121_DE_time']
        BF_7_1730_X = BF_7_1730[:-(len(BF_7_1730)%self.length_signal)].reshape(-1, self.length_signal)

        # Fault 3
        OF_7_1797 = sio.loadmat('130.mat')['X130_DE_time']
        OF_7_1797_X = OF_7_1797[:-(len(OF_7_1797)%self.length_signal)].reshape(-1, self.length_signal)

        OF_7_1772 = sio.loadmat('131.mat')['X131_DE_time']
        OF_7_1772_X = OF_7_1772[:-(len(OF_7_1772)%self.length_signal)].reshape(-1, self.length_signal)

        OF_7_1750 = sio.loadmat('132.mat')['X132_DE_time']
        OF_7_1750_X = OF_7_1750[:-(len(OF_7_1750)%self.length_signal)].reshape(-1, self.length_signal)

        OF_7_1730 = sio.loadmat('133.mat')['X133_DE_time']
        OF_7_1730_X = OF_7_1730[:-(len(OF_7_1730)%self.length_signal)].reshape(-1, self.length_signal)

        # 0.014
        # Fault 1
        IF_14_1797 = sio.loadmat('169.mat')['X169_DE_time']
        IF_14_1797_X = IF_14_1797[:-(len(IF_14_1797)%self.length_signal)].reshape(-1, self.length_signal)

        IF_14_1772 = sio.loadmat('170.mat')['X170_DE_time']
        IF_14_1772_X = IF_14_1772[:-(len(IF_14_1772)%self.length_signal)].reshape(-1, self.length_signal)

        IF_14_1750 = sio.loadmat('171.mat')['X171_DE_time']
        IF_14_1750_X = IF_14_1750[:-(len(IF_14_1750)%self.length_signal)].reshape(-1, self.length_signal)

        IF_14_1730 = sio.loadmat('172.mat')['X172_DE_time']
        IF_14_1730_X = IF_14_1730[:-(len(IF_14_1730)%self.length_signal)].reshape(-1, self.length_signal)

        # Fault 2
        BF_14_1797 = sio.loadmat('185.mat')['X185_DE_time']
        BF_14_1797_X = BF_14_1797[:-(len(BF_14_1797)%self.length_signal)].reshape(-1, self.length_signal)

        BF_14_1772 = sio.loadmat('186.mat')['X186_DE_time']
        BF_14_1772_X = BF_14_1772[:-(len(BF_14_1772)%self.length_signal)].reshape(-1, self.length_signal)

        BF_14_1750 = sio.loadmat('187.mat')['X187_DE_time']
        BF_14_1750_X = BF_14_1750[:-(len(BF_14_1750)%self.length_signal)].reshape(-1, self.length_signal)

        BF_14_1730 = sio.loadmat('188.mat')['X188_DE_time']
        BF_14_1730_X = BF_14_1730[:-(len(BF_14_1730)%self.length_signal)].reshape(-1, self.length_signal)

        # Fault 3
        OF_14_1797 = sio.loadmat('197.mat')['X197_DE_time']
        OF_14_1797_X = OF_14_1797[:-(len(OF_14_1797)%self.length_signal)].reshape(-1, self.length_signal)

        OF_14_1772 = sio.loadmat('198.mat')['X198_DE_time']
        OF_14_1772_X = OF_14_1772[:-(len(OF_14_1772)%self.length_signal)].reshape(-1, self.length_signal)

        OF_14_1750 = sio.loadmat('199.mat')['X199_DE_time']
        OF_14_1750_X = OF_14_1750[:-(len(OF_14_1750)%self.length_signal)].reshape(-1, self.length_signal)

        OF_14_1730 = sio.loadmat('200.mat')['X200_DE_time']
        OF_14_1730_X = OF_14_1730[:-(len(OF_14_1730)%self.length_signal)].reshape(-1, self.length_signal)

        # 0.021
        # Fault 1
        IF_21_1797 = sio.loadmat('209.mat')['X209_DE_time']
        IF_21_1797_X = IF_21_1797[:-(len(IF_21_1797)%self.length_signal)].reshape(-1, self.length_signal)

        IF_21_1772 = sio.loadmat('210.mat')['X210_DE_time']
        IF_21_1772_X = IF_21_1772[:-(len(IF_21_1772)%self.length_signal)].reshape(-1, self.length_signal)

        IF_21_1750 = sio.loadmat('211.mat')['X211_DE_time']
        IF_21_1750_X = IF_21_1750[:-(len(IF_21_1750)%self.length_signal)].reshape(-1, self.length_signal)

        IF_21_1730 = sio.loadmat('212.mat')['X212_DE_time']
        IF_21_1730_X = IF_21_1730[:-(len(IF_21_1730)%self.length_signal)].reshape(-1, self.length_signal)

        # Fault 2
        BF_21_1797 = sio.loadmat('222.mat')['X222_DE_time']
        BF_21_1797_X = BF_21_1797[:-(len(BF_21_1797)%self.length_signal)].reshape(-1, self.length_signal)

        BF_21_1772 = sio.loadmat('223.mat')['X223_DE_time']
        BF_21_1772_X = BF_21_1772[:-(len(BF_21_1772)%self.length_signal)].reshape(-1, self.length_signal)

        BF_21_1750 = sio.loadmat('224.mat')['X224_DE_time']
        BF_21_1750_X = BF_21_1750[:-(len(BF_21_1750)%self.length_signal)].reshape(-1, self.length_signal)

        BF_21_1730 = sio.loadmat('225.mat')['X225_DE_time']
        BF_21_1730_X = BF_21_1730[:-(len(BF_21_1730)%self.length_signal)].reshape(-1, self.length_signal)

        # Fault 3
        OF_21_1797 = sio.loadmat('234.mat')['X234_DE_time']
        OF_21_1797_X = OF_21_1797[:-(len(OF_21_1797)%self.length_signal)].reshape(-1, self.length_signal)

        OF_21_1772 = sio.loadmat('235.mat')['X235_DE_time']
        OF_21_1772_X = OF_21_1772[:-(len(OF_21_1772)%self.length_signal)].reshape(-1, self.length_signal)

        OF_21_1750 = sio.loadmat('236.mat')['X236_DE_time']
        OF_21_1750_X = OF_21_1750[:-(len(OF_21_1750)%self.length_signal)].reshape(-1, self.length_signal)

        OF_21_1730 = sio.loadmat('237.mat')['X237_DE_time']
        OF_21_1730_X = OF_21_1730[:-(len(OF_21_1730)%self.length_signal)].reshape(-1, self.length_signal)

        # Way 2
        # Source: 7
        # Target: 21

        S_NC_X = np.concatenate([normal_1730_X])
        S_NC_y = [0]*S_NC_X.shape[0]

        S_IF_X = np.concatenate([IF_7_1797_X, IF_7_1772_X, IF_7_1750_X, IF_7_1730_X])
        S_IF_y = [1]*S_IF_X.shape[0]

        S_BF_X = np.concatenate([BF_7_1797_X, BF_7_1772_X, BF_7_1750_X, BF_7_1730_X])
        S_BF_y = [2]*S_BF_X.shape[0]

        S_OF_X = np.concatenate([OF_7_1797_X, OF_7_1772_X, OF_7_1750_X, OF_7_1730_X])
        S_OF_y = [3]*S_OF_X.shape[0]

        self.Source_X = np.concatenate((S_NC_X, S_IF_X, S_BF_X, S_OF_X))
        self.Source_y = S_NC_y+S_IF_y+S_BF_y+S_OF_y

        T_NC_X = np.concatenate([normal_1750_X])
        T_NC_y = [0]*T_NC_X.shape[0]

        T_IF_X = np.concatenate([IF_21_1797_X, IF_21_1772_X, IF_21_1750_X, IF_21_1730_X])
        T_IF_y = [1]*T_IF_X.shape[0]

        T_BF_X = np.concatenate([BF_21_1797_X, BF_21_1772_X, BF_21_1750_X, BF_21_1730_X])
        T_BF_y = [2]*T_BF_X.shape[0]

        T_OF_X = np.concatenate([OF_21_1797_X, OF_21_1772_X, OF_21_1750_X, OF_21_1730_X])
        T_OF_y = [3]*T_OF_X.shape[0]

        self.Target_X = np.concatenate((T_NC_X, T_IF_X, T_BF_X, T_OF_X))
        self.Target_y = T_NC_y+T_IF_y+T_BF_y+T_OF_y
        #
        self.Source_X = self.Source_X[:4864, :]
        self.Source_y = self.Source_y[:4864]

        self.Target_X = self.Target_X[:4864, :]
        self.Target_y = self.Target_y[:4864]

        scaler = preprocessing.StandardScaler().fit(self.Source_X)
        self.Source_X = scaler.transform(self.Source_X)

        scaler2 = preprocessing.StandardScaler().fit(self.Target_X)
        self.Target_X = scaler2.transform(self.Target_X)


    def __getitem__(self, index):
        if self.source:
            data = self.Source_X[index, :]
            label = self.Source_y[index]
        else:
            data = self.Target_X[index, :]
            label = self.Target_y[index]
        return data.reshape(1, len(data)), int(label)


    def __len__(self):
        if self.source:
            return len(self.Source_y)
        return len(self.Target_y)

