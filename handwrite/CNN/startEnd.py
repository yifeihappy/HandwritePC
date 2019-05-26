# 获取开始和结束索引
# 通过磁力计变化，截取有效数据部分
import pandas as pd
import numpy as np
import scipy.signal as signal


def start_end_indexs(mag):
    mag = signal.medfilt(mag, (5, 1))
    mag = signal.medfilt(mag, (5, 1))
    sum_square = np.sqrt(np.sum(np.square(mag), axis=1))
    # sum_square = np.sqrt(np.sum(np.square(np.diff(mag, axis=0)), axis=1))
    s = 0
    thre_m = 10  # 10
    for item in sum_square:
        if item > thre_m:
            break
        s += 1
    e = sum_square.shape[0] - 1
    while e > s:
        if sum_square[e] > thre_m:
            break
        e -= 1
    return s, e


# 双阈值法, 判断开始
def double_threld_start(mag, high_t, low_t):
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


# 去背景，判断结束
def rb_threld_start_end(mag, i_l, i_h, thre):
    # 向前寻找背景值
    i_s = i_l
    if i_s > 20:
        i_s = i_s - 20
    b_vs = mag[i_s - 10, :]
    # remove background
    mag = mag - b_vs
    sqs = np.sqrt(np.sum(np.square(mag), axis=1))
    i_e = i_h + 50
    for item in sqs[i_e:]:
        if item < thre:
            break
        i_e += 1
    return i_s, i_e + 20


# 获取数据预处理阈值
def get_threld(watch, date, ring, position):
    # file_path = "C:\\Users\\DYF\\Desktop\\master paper\\MInput\\data\\ticwatch" + str(watch) + "\\" + date + "\\ring" \
    #             + ring + "\\" + gesture + "\\sensor" + str(1) + "_1.txt"
    file_path = "D:/STUDY/paper3/master paper/MInput/data/" + watch + "/" + date + "/" + ring + "/" + position + \
                "/sensor" + str(1) + "_" + str(1) + ".txt"
    dataFrame = pd.read_table(file_path, sep=',')
    data = dataFrame.values.copy()[:, [0, 1, 3, 4, 5]]
    start_time = data[0, 1]  # 开始时间
    magnetic = data[data[:, 0] == 2, :]
    magnetic_start = magnetic[0, :]  # 获取背景值
    magnetic[:, 1:5] = magnetic[:, 1:5] - magnetic_start[1:5]
    m_total = np.sqrt(np.sum(np.square(magnetic[:, 2:5]), axis=1))
    m_diff = np.diff(magnetic[:, 2:5], axis=0)
    m_a = np.sqrt(np.sum(np.square(m_diff[:, [0, 1, 2]]), axis=1))
    h_thre = np.max(m_a) * 0.3
    l_thre = np.max(m_a) * 0.05
    thre = np.max(m_total) * 0.1
    return h_thre, l_thre, thre


# 归一化
def normalization_np(mag):
    max_v = np.max(mag, axis=0)
    min_v = np.min(mag, axis=0)
    mag = (mag - min_v)/(max_v - min_v)
    return mag


if __name__ == '__main__':
    a = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
    ma = normalization_np(a)
    print(ma)
