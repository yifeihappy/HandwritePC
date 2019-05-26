import pickle
import random
import numpy as np
from sklearn.model_selection import train_test_split


# 载入文件
class DataController:
    # def __init__(self, file_name, random_state=0, train_size=0):
    #     """
    #     生成训练数据和测试数据
    #     :param file_name: 存放dataLabel生成的数据的路径
    #     :param random_state: 按比例生成训练数据和测试数据的随机种子
    #     :param train_size: 训练数据所占的比重
    #     """
    #     file_load = open(file_name, 'rb')
    #     self.samples = pickle.load(file_load)
    #     if train_size == 0:
    #         self.images_train = self.samples['data_images_train']
    #         self.images_test = self.samples['data_images_test']
    #         self.labels_train = self.samples['data_labels_train']
    #         self.labels_test = self.samples['data_labels_test']
    #     else:
    #         self.images_train, self.images_test, self.labels_train, self.labels_test = train_test_split(
    #             self.samples['data_images'], self.samples['data_labels'], random_state=random_state,
    #             train_size=train_size, test_size=1-train_size)
    #     self.batch_id = 0
    # def __init__(self, file_name, random_state=0, train_size=0):
    #     """
    #     生成训练数据和测试数据
    #     :param file_name: 存放dataLabel生成的数据的路径
    #     :param random_state: 按比例生成训练数据和测试数据的随机种子
    #     :param train_size: 训练数据所占的比重
    #     """
    #     file_load = open(file_name, 'rb')
    #     self.samples = pickle.load(file_load)
    #     if train_size == 0:
    #         self.images_train = self.samples['data_images_train']
    #         self.images_test = self.samples['data_images_test']
    #         self.labels_train = self.samples['data_labels_train']
    #         self.labels_test = self.samples['data_labels_test']
    #     else:
    #         self.images_train, self.images_test, self.labels_train, self.labels_test = train_test_split(
    #             self.samples['data_images'], self.samples['data_labels'], random_state=random_state,
    #             train_size=train_size, test_size=1-train_size)
    #     self.batch_id = 0

    def __init__(self, file_name, *files):
        """
        生成训练数据和测试数据
        :param file_name: 存放dataLabel生成的数据的路径
        :param random_state: 按比例生成训练数据和测试数据的随机种子
        :param train_size: 训练数据所占的比重
        """
        file_load = open(file_name, 'rb')
        self.samples = pickle.load(file_load)
        self.images_train = self.samples['data_images_train']
        self.images_test = self.samples['data_images_test']
        self.labels_train = self.samples['data_labels_train']
        self.labels_test = self.samples['data_labels_test']
        self.nclass = len(self.labels_test[0])
        file_load.close()
        for file in files:
            file_load = open(file, 'rb')
            f_s = pickle.load(file_load)
            self.images_train += f_s['data_images_train']
            self.images_test += f_s['data_images_test']
            self.labels_train += f_s['data_labels_train']
            self.labels_test += f_s['data_labels_test']
            file_load.close()
        self.batch_id = 0

    def get_sample_size(self):
        return self.images_train[0].shape

    def next(self, batch_size=-1, is_shuffled=True):

        batch_size = int(batch_size)
        if not is_shuffled:
            if batch_size < 0:
                return self.images_train, self.labels_train
            if self.batch_id >= len(self.images_train):
                self.batch_id = 0
            batch_x = self.images_train[self.batch_id:min(self.batch_id + batch_size, len(self.images_train))]
            batch_y = self.labels_train[self.batch_id:min(self.batch_id + batch_size, len(self.images_train))]
            self.batch_id = min(self.batch_id + batch_size, len(self.images_train))
        else:
            indexes = list(range(len(self.images_train)))
            random.shuffle(indexes)
            if len(indexes) > batch_size > 0:
                indexes = indexes[:batch_size]
            # indexes = np.array(indexes)
            batch_x = np.array(self.images_train)[indexes]
            batch_y = np.array(self.labels_train)[indexes]
        return batch_x, batch_y

    def get_test_data(self):
        return self.images_test, self.labels_test


if __name__ == "__main__":
    b_p = './data/TT/ticwatch1/ringR/5P0_T0_D_R_hand_back_'
    # dataControllerU = DataControllerU(b_p + 'df' + '_TT.bin', b_p + 'yjh' + '_TT.bin', b_p + 'zy' + '_TT.bin')
    dataController = DataController(b_p + 'df' + '_TT.bin')
    batch_x, batch_y = dataController.next(3)
    test_x, test_y = dataController.get_test_data()

