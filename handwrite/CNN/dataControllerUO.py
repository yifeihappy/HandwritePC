import pickle
import random
import numpy as np


# 载入文件
class DataControllerUO:
    def __init__(self, file_name, *files):
        """
        生成训练数据和测试数据,被测试的第三用户也 不 被包含在动作识别的训练测试数据里
        :param file_name: 存放dataLabel生成的数据的路径
        :param random_state: 按比例生成训练数据和测试数据的随机种子
        :param train_size: 训练数据所占的比重
        """
        self.uclass = len(files) + 1
        file_load = open(file_name, 'rb')
        self.samples = pickle.load(file_load)
        self.images_train = self.samples['data_images_train']
        self.images_test = self.samples['data_images_test']
        self.labels_train = self.samples['data_labels_train']
        self.labels_test = self.samples['data_labels_test']
        self.nclass = len(self.labels_test[0])
        u_id = 0
        self.user_images_train = list.copy(self.samples['data_images_train'])
        self.user_images_test = list.copy(self.samples['data_images_test'])
        self.user_l_train = self.get_user_label(u_id=u_id, data_l=len(self.images_train))
        self.user_l_test = self.get_user_label(u_id=u_id, data_l=len(self.images_test))
        file_load.close()
        for file in files:
            u_id += 1
            file_load = open(file, 'rb')
            f_s = pickle.load(file_load)
            if len(files) != u_id:
                self.images_train += f_s['data_images_train']
                self.images_test += f_s['data_images_test']
                self.labels_train += f_s['data_labels_train']
                self.labels_test += f_s['data_labels_test']
            self.user_images_train += list.copy(f_s['data_images_train'])
            self.user_images_test += list.copy(f_s['data_images_test'])
            self.user_l_train += self.get_user_label(u_id=u_id, data_l=len(f_s['data_images_train']))
            self.user_l_test += self.get_user_label(u_id=u_id, data_l=len(f_s['data_images_test']))
            file_load.close()
        self.batch_id = 0

    def get_user_label(self, u_id, data_l):
        user_l = np.zeros(shape=[data_l, self.uclass])
        user_l[:, u_id] = 1
        return list(user_l)

    def get_sample_size(self):
        return self.images_train[0].shape

    def nextA(self, batch_size=-1, is_shuffled=True):
        batch_size = int(batch_size)
        if not is_shuffled:
            if batch_size < 0:
                return self.images_train, self.labels_train
            if self.batch_id >= len(self.images_train):
                self.batch_id = 0
            batch_x = self.images_train[self.batch_id:min(self.batch_id + batch_size, len(self.images_train))]
            batch_y = self.labels_train[self.batch_id:min(self.batch_id + batch_size, len(self.images_train))]
            batch_ux = self.user_images_train[self.batch_id:min(self.batch_id + batch_size, len(self.images_train))]
            batch_uy = self.user_l_train[self.batch_id:min(self.batch_id + batch_size, len(self.images_train))]
            self.batch_id = min(self.batch_id + batch_size, len(self.images_train))
        else:
            indexes = list(range(len(self.images_train)))
            random.shuffle(indexes)
            if len(indexes) > batch_size > 0:
                indexes = indexes[:batch_size]
            # indexes = np.array(indexes)
            batch_x = np.array(self.images_train)[indexes]
            batch_y = np.array(self.labels_train)[indexes]
            batch_ux = np.array(self.user_images_train)[indexes]
            batch_uy = np.array(self.user_l_train)[indexes]
        return batch_x, batch_y, batch_ux, batch_uy

    def nextU(self, batch_size=-1, is_shuffled=True):
        batch_size = int(batch_size)
        if not is_shuffled:
            if batch_size < 0:
                return self.images_train, self.labels_train
            if self.batch_id >= len(self.images_train):
                self.batch_id = 0
            batch_ux = self.user_images_train[self.batch_id:min(self.batch_id + batch_size, len(self.images_train))]
            batch_uy = self.user_l_train[self.batch_id:min(self.batch_id + batch_size, len(self.images_train))]
            self.batch_id = min(self.batch_id + batch_size, len(self.images_train))
        else:
            indexes = list(range(len(self.user_images_train)))
            random.shuffle(indexes)
            if len(indexes) > batch_size > 0:
                indexes = indexes[:batch_size]
            batch_ux = np.array(self.user_images_train)[indexes]
            batch_uy = np.array(self.user_l_train)[indexes]
        return batch_ux, batch_uy

    def get_test_data(self):
        return self.images_test, self.labels_test, self.user_images_test, self.user_l_test


if __name__ == "__main__":
    b_p = './data/TT/ticwatch1/ringR/5P0_T0_D_R_hand_back_'
    dataControllerUO = DataControllerUO(b_p + 'df' + '_TT.bin', b_p + 'yjh' + '_TT.bin', b_p + 'zy' + '_TT.bin')
    batch_x, batch_y, batch_ux, batch_uy = dataControllerUO.nextA(3)
    test_x, test_y, test_ux, test_uy = dataControllerUO.get_test_data()

