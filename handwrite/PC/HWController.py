import numpy as np
import pickle
import scipy.signal as signal

class HWController:

    def __init__(self):
        self.dataArray = []
        self.lastStr = ""
        self.h_thre = 0
        self.l_thre = 0
        self.thre = 0
        self.threholdFilePath = "threholds.txt"
        self.normal_win_size = 1000 #  1second
        self.delay_size = 1000  # 1second
        self.image_len = 400

    @classmethod
    def write_file(cls, data, filename="iss.txt"):
        file = open(filename, 'w')
        for r in data:
            for item in r:
                file.write(str(item) + ',')
            file.write("\n")
        file.close()

    # decode the data from the smartwatch
    def decode_str(self, data_str):
        data_str = self.lastStr + data_str
        data_str_a = data_str.split("\n")
        is_end = False
        if len(data_str_a) == 1:  # maybe the str is not complete
            self.lastStr = data_str
            if 'E' == data_str[0]:
                is_end = True
        else:
            for i in range(len(data_str_a)-1):
                str_a = data_str_a[i].split(",")
                item_a = list(map(eval, str_a[1:]))  # ignore the first cell "S" or "E"
                self.dataArray.append(item_a)
                if 'E' == str_a[0]:
                    is_end = True
            # keep the last str item
            self.lastStr = data_str_a[-1]
        return is_end

    # 获取数据预处理阈值
    def initial(self):
        data = np.array(self.dataArray)
        magnetic = data[data[:, 0] == 2, :]
        magnetic_start = magnetic[0, :]  # 获取背景值
        magnetic[:, 1:5] = magnetic[:, 1:5] - magnetic_start[1:5]
        m_total = np.sqrt(np.sum(np.square(magnetic[:, 2:5]), axis=1))
        m_diff = np.diff(magnetic[:, 2:5], axis=0)
        m_a = np.sqrt(np.sum(np.square(m_diff[:, [0, 1, 2]]), axis=1))
        self.h_thre = np.max(m_a) * 0.3
        self.l_thre = np.max(m_a) * 0.05
        self.thre = np.max(m_total) * 0.1
        threholds = [self.h_thre, self.l_thre, self.thre]
        # save the threhold to the file
        self.save_threholds(threholds, self.threholdFilePath)
        print("Threholds:")
        print(threholds)
        HWController.write_file(data, "initial.txt")

    def save_threholds(self, v, filename):
        f = open(filename, 'wb')
        pickle.dump(v, f)
        f.close()

    def load_threholds(self, filename):
        f = open(filename, 'rb')
        r = pickle.load(f)
        self.h_thre = r[0]
        self.l_thre = r[1]
        self.thre = r[2]
        f.close()

    # 滑动窗口
    @classmethod
    def slide_win(cls, data, rate, win_time):
        while data[-1][1] - data[0][1] > win_time * rate:
            data.pop(0)

    # 判断是否有手写事件发生
    def is_start_handwriting(self, data):
        data = np.array(data)
        # 中值滤波*3
        data = signal.medfilt(data[:, [2, 3, 4]], (5, 1))
        data = signal.medfilt(data, (5, 1))
        data = signal.medfilt(data, (5, 1))
        diff_value = np.diff(data, axis=0)
        diff_value = np.sqrt(np.sum(np.square(diff_value[:, [0, 1, 2]]), axis=1))
        # 高阈值
        for item in diff_value:
            if item > self.h_thre:
                print("发生手写事件")
                return True
        return False

    # 去背景，判断结束
    def is_end(self, data, start_index):
        data = np.array(data)
        # 中值滤波*3
        data = signal.medfilt(data[:, [2, 3, 4]], (5, 1))
        data = signal.medfilt(data, (5, 1))
        data = signal.medfilt(data, (5, 1))
        # 向前寻找背景值
        b_vs = data[0, :]
        # remove background
        data = data - b_vs
        sqs = np.sqrt(np.sum(np.square(data), axis=1))
        i_e = start_index + 50
        for item in sqs[i_e:]:
            if item < self.thre:
                return True
        return False
