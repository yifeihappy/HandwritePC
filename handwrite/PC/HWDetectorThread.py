import threading
import time
import numpy as np
from HWController import *
from HWRecognizeThread import *


class HWDetectorThread(threading.Thread):
    def __init__(self, dq, hw_controller, recognize_helper):
        threading.Thread.__init__(self)
        self.dataQueue = dq
        self.exitFlag = False
        self.mag_win = []
        self.hwFlag = False
        self.endFlag = False
        self.recorderData = []
        self.hwController = hw_controller
        self.startIndex = 0
        self.recognize_helper = recognize_helper

    def run(self):
        print("Start hw detector!")
        while not self.exitFlag:
            data_list = []
            while not self.dataQueue.empty():
                list_item = self.dataQueue.get()
                data_list.append(list_item)
            self.handwriting_detection(data_list)
            time.sleep(0.1)
        print("HWDetectorThread exit...")

    def handwriting_detection(self, data_list):
        data = np.array(data_list)
        # 获取磁力计数据
        index = 0
        data_len = len(data)
        input_flag = True  # 标记手是否处于水平位置
        while index < data_len:
            self.recorderData.append(data[index, :])
            if 9 == data[index, 0]:
                if np.abs(data[index, 4]) > 7:
                    input_flag = True
                else:
                    input_flag = False
                    self.mag_win = []
            elif input_flag:
                if 2 == data[index, 0]:
                    self.mag_win.append(data[index, :])
                    if not self.hwFlag:  # 没有手写事件
                        if self.mag_win[-1][1] - self.mag_win[0][1] > self.hwController.normal_win_size:  # 判断窗口大小
                            self.hwFlag = self.hwController.is_start_handwriting(self.mag_win)  # 检测有没有手写事件
                            if not self.hwFlag:  # 没有手写事件，窗口减半
                                HWController.slide_win(self.mag_win, 0.5, self.hwController.normal_win_size)
                                print('none')
                            else:
                                self.startIndex = len(self.mag_win)
                    else:   # 发生手写事件
                        print('handwrite')
                        if self.mag_win[-1][1] - self.mag_win[self.startIndex][1] > self.hwController.delay_size:
                            if self.hwController.is_end(self.mag_win, self.startIndex):
                                # 识别手写字符
                                hw_recognize_thread = HWRecognizeThread(self.mag_win, self.recognize_helper, self.hwController)
                                hw_recognize_thread.start()
                                # 重新对手写事件进行监听
                                self.hwFlag = False
                                self.mag_win = []
                                self.startIndex = 0
            index += 1









