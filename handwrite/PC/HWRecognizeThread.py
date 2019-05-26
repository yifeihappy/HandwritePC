import threading
import scipy.signal as signal
from scipy.interpolate import interp1d
import numpy as np


# 识别手写字符
class HWRecognizeThread(threading.Thread):
    def __init__(self, magnetic, recognizerHelper, hwController):
        threading.Thread.__init__(self)
        self.magnetic = magnetic
        self.recognizerHelper = recognizerHelper
        self.hwController = hwController

    def run(self):
        data = self.normalize_data()
        self.character_recog(data)

    # 获取有效数据段：识别起点位置
    # 线性插值
    # 归一化
    # 保存数据
    def normalize_data(self):
        data = np.array(self.magnetic)
        # 中值滤波
        data[:, 2:5] = signal.medfilt(data[:, 2:5], (5, 1))
        data[:, 2:5] = signal.medfilt(data[:, 2:5], (5, 1))
        data[:, 2:5] = signal.medfilt(data[:, 2:5], (5, 1))
        mag = data[:, [2, 3, 4]]
        # 双阈值法, 判断开始
        i_l, i_h = self.double_threld_start(mag, self.hwController.h_thre, self.hwController.l_thre)
        # 获取目标数据段
        target = data[i_l:, 1:5]
        #线性插值
        t_interp = np.linspace(target[0, 0], target[-1, 0], num=self.hwController.image_len)
        interp_fun = interp1d(target[:, 0], target[:, 1:4], axis=0, kind='linear')
        inter_m = interp_fun(t_interp)
        images = np.c_[inter_m, inter_m[:, 0] - inter_m[:, 1], inter_m[:, 0] - inter_m[:, 2],
                       inter_m[:, 1] - inter_m[:, 2]]
        max_v = np.max(images, axis=0)
        min_v = np.min(images, axis=0)
        mag = (images - min_v)/(max_v - min_v)
        return mag

    # 调用训练好的模型，识别字符
    def character_recog(self, data):
        self.recognizerHelper.recognize_handwriting_character(data)

    def double_threld_start(self, mag, high_t, low_t):
        diff_value = np.diff(mag, axis=0)
        diff_value = np.sqrt(np.sum(np.square(diff_value[:, [0, 1, 2]]), axis=1))
        i_h = 0
        # 高阈值
        for item in diff_value:
            if item > high_t:
                break
            i_h += 1
        # 低阈值
        i_l = i_h
        while i_l > 0:
            if diff_value[i_l] < low_t:
                break
            i_l -= 1
        return i_l, i_h

