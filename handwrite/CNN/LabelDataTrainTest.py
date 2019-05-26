# 测试手写字母的可行性
# 绘制图像
# 获取特征并用SVM分类


import pickle
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import random
import scipy.signal as signal
from startEnd import *


# 图像点击事件函数
def on_press(event):
    print('position x= %f, y= %f' % (event.xdata, event.ydata))


# 获取训练集下标
def get_train_index(file_num, train_num):
    indexs = []
    for i in range(file_num):
        indexs.append(i)
    return random.sample(indexs, train_num)


data_images_train = []
data_images_test = []
data_labels_train = []
data_labels_test = []
samples = {}
image_len = 400
# **********参数设置************
label_start = 0
position = "p12"
nclass = 10
watch = "ticwatch1"
ring = 'ring1'
padding = False
draw = True
file_num = 20
train_num = 5
date = '20190315'
save_file = True
h_thre, l_thre, thre = get_threld(watch, date, ring, position)
# ********** END ************
train_indexs = get_train_index(file_num, train_num)
time_last = []
# 读取文件
for X in range(nclass):
    if draw:
        fig, ax = plt.subplots(1, 1)
        if nclass == 26:
            fig.suptitle(chr(ord('A') + X) + str(X))
        else:
            fig.suptitle(str(X))
        lengen_i = 0
    for Y in range(file_num):
        # if draw:
        #     fig, ax = plt.subplots(1, 1)
        #     fig.suptitle(str(Y))
        file_path = "D:/STUDY/paper3/master paper/MInput/data/" + watch + "/" + date + "/" + ring + "/" + position\
                    + "/sensor" + str(X) + "_" + str(Y) + ".txt"
        dataFrame = pd.read_table(file_path, sep=',')
        data = dataFrame.values.copy()[:, [0, 1, 3, 4, 5]]
        #   1:accelerometer;2:magnetic;4:gyroscope;9:gravity;10:linear accelerometer
        # 数据采集的顺序是以[1，2，4，9，10]循环采样，第一个值不知道是1，还是2，或者。。。
        start_time = data[0, 1]  # 开始时间
        magnetic = data[data[:, 0] == 2, :]
        # 获取有效数据段
        magnetic_start = magnetic[0, :]     #获取背景值
        magnetic = magnetic - magnetic_start

        # 中值滤波
        magnetic[:, 2:5] = signal.medfilt(magnetic[:, 2:5], (5, 1))
        magnetic[:, 2:5] = signal.medfilt(magnetic[:, 2:5], (5, 1))
        magnetic[:, 2:5] = signal.medfilt(magnetic[:, 2:5], (5, 1))

        mag = magnetic[:, [2, 3, 4]]
        i_l, i_h = double_threld_start(mag, h_thre, l_thre)
        s, e = rb_threld_start_end(mag, i_l, i_h, thre)
        magnetic = magnetic[s:e, 1:5]
        # 获取开始时间
        magnetic_start_time = magnetic[0, 0]
        magnetic[:, 0] = magnetic[:, 0] - magnetic_start_time
        data_all = magnetic[:, 1:4]     # 没有时间列
        if padding:
            p_padding = np.zeros(shape=[image_len, data_all.shape[1]], dtype=np.float32)
            p_i = 0
            for p in data_all:
                p_padding[p_i, :] = p
                p_i += 1
            images = p_padding
        else:
            t_interp = np.linspace(magnetic[0, 0], magnetic[-1, 0], num=image_len)
            time_last.append(magnetic[-1, 0] - magnetic[0, 0])
            interp_fun = interp1d(magnetic[:, 0], magnetic[:, 1:4], axis=0, kind='linear')
            inter_m = interp_fun(t_interp)
            images = np.c_[inter_m, inter_m[:, 0] - inter_m[:, 1], inter_m[:, 0] - inter_m[:, 2],
                           inter_m[:, 1] - inter_m[:, 2]]
            images = normalization_np(images)
        if draw:
            # 绘制图像
            t = np.arange(0, image_len)
            lengen_i += 1
            if lengen_i == 20:
                ax.plot(t, images[:, 0], 'r', label='x')
                ax.plot(t, images[:, 1], 'g', label='y')
                ax.plot(t, images[:, 2], 'b', label='z')
                handles, labels = ax.get_legend_handles_labels()
                font1 = {'family': 'Times New Roman',
                         'weight': 'normal',
                         'size': 20,
                         }
                ax.legend(handles, labels, prop=font1)
                plt.tick_params(labelsize=20)
                labels = ax.get_xticklabels() + ax.get_yticklabels()
                [label.set_fontname('Times New Roman') for label in labels]
            else:
                ax.plot(t, images[:, 0], 'r')
                ax.plot(t, images[:, 1], 'g')
                ax.plot(t, images[:, 2], 'b')
            font2 = {'family': 'Times New Roman', 'weight': 'normal', 'size': 20}
            ax.set_xlabel('Time(s)', font2)
            ax.set_ylabel('Magnetic Field', font2)
            fig.show()

        # 添加标签
        label = np.zeros(nclass)
        label[X] = 1.0
        if Y in train_indexs:
            data_images_train.append(images)
            data_labels_train.append(label)
        else:
            data_images_test.append(images)
            data_labels_test.append(label)

if save_file:
    samples['data_images_train'] = data_images_train
    samples['data_labels_train'] = data_labels_train
    samples['data_images_test'] = data_images_test
    samples['data_labels_test'] = data_labels_test
    save_file = open('./data/' + watch + '/' + ring + '/' + position + '_' + str(train_num) + '.bin', 'wb')
    pickle.dump(samples, save_file)
    save_file.close()

np_time_last = np.array(time_last)
print("Time min: %f, max: %f, mean: %f" % (np_time_last.min(), np_time_last.max(), np_time_last.mean()))




