import tensorflow as tf
from dataController import DataController
from HWRecognizeThread import HWRecognizeThread


class RecognizeHelper:

    def __init__(self, frame):
        print("RecognizeHelper start...")
        self.frame = frame
        # ****** 参数设置 ******
        user = 'p0top12_p12_5'
        test_user = '../CNN/data/ticwatch1/ring1/p3_5.bin'
        self.testDataController = DataController(test_user)
        nclass = self.testDataController.nclass
        # ****** END ******
        # 读取数据
        sample_size = self.testDataController.get_sample_size()
        self.x = tf.placeholder(dtype=tf.float32, shape=[None, sample_size[0], sample_size[1]])
        self.y = tf.placeholder(dtype=tf.float32, shape=[None, nclass])
        self.keep_prob = tf.placeholder(dtype=tf.float32)
        self.is_training = tf.placeholder(dtype=tf.bool)
        # 数据变成四维
        x_images = tf.reshape(tensor=self.x, shape=[-1, sample_size[0], sample_size[1], 1])  #
        # CNN3 构建网络
        y_ = self.cnnBN(inputs=x_images, is_training=self.is_training, keep_prob=self.keep_prob, nclass=nclass)
        #  定义交叉商
        self.cross_entropy = -tf.reduce_mean(input_tensor=self.y * tf.log(y_))
        # 学习率随着步数变化
        global_step = tf.Variable(initial_value=0, trainable=False)
        lr = tf.train.exponential_decay(learning_rate=0.001, global_step=global_step, decay_steps=100, decay_rate=0.99,
                                        staircase=True)
        # 优化损失函数， 梯度下降方法
        # train_step = tf.train.AdamOptimizer(lr).minimize(loss=cross_entropy, global_step=global_step)
        train_step = tf.train.AdamOptimizer(lr).minimize(loss=self.cross_entropy, global_step=global_step)
        # 评测
        correct_prediction = tf.equal(tf.argmax(input=self.y, axis=1), tf.argmax(input=y_, axis=1))
        # 混淆矩阵
        confusion_matrix = tf.contrib.metrics.confusion_matrix(tf.argmax(input=self.y, axis=1), tf.argmax(input=y_, axis=1))
        # 计算精度
        self.accuracy = tf.reduce_mean(tf.cast(x=correct_prediction, dtype=tf.float32))
        # 预处值
        self.predict_y = tf.argmax(input=y_, axis=1)
        # 训练
        with tf.Session().as_default() as sess:
            tf.global_variables_initializer().run()
            saver = tf.train.Saver(max_to_keep=3)
            model_file = tf.train.latest_checkpoint('../CNN/model/' + user + '/')
            saver.restore(sess, model_file)

            # validate_x, validate_y = self.testDataController.get_test_data()
            # validate_loss, validate_accuracy, pre_y = sess.run([self.cross_entropy, self.accuracy, self.predict_y],
            #                                                    feed_dict={self.x: validate_x,
            #                                                               self.y: validate_y,
            #
            #                                                               self.keep_prob: 1,
            #                                                               self.is_training: False})
            # print('VALIDATE ACCURACY: %f, LOSS: %f ' % (validate_accuracy, validate_loss))
            # print(pre_y)
        self.recog_session = sess
        print("Load CNN model successfully!")
        self.frame.m_staticText_model.SetLabel('Model OK')

    def recognize_handwriting_character(self, data):
        data_list = []
        data_list.append(data)
        pre_y = self.recog_session.run([self.predict_y], feed_dict={self.x: data_list, self.keep_prob: 1,
                                                                    self.is_training: False})
        print("predict:")
        print(pre_y[0])
        self.frame.m_textCtrl.SetValue(self.frame.m_textCtrl.GetValue() + str(pre_y[0]))

    # 识别动作
    # 权重变量
    def get_weight_variable(self, shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    # 偏移变量
    def get_bias_variable(self, shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    # 卷积
    def conv2d(self, x, w):
        return tf.nn.conv2d(input=x, filter=w, strides=[1, 1, 1, 1], padding='SAME')

    # 池化, 数据长度宽度减半
    def max_pool_2x2(self, x):
        return tf.nn.max_pool(value=x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    # Batch normalization layer
    def bn_layer(self, inputs, is_train, scope=None):
        return tf.cond(is_train, lambda: tf.contrib.layers.batch_norm(inputs=inputs, is_training=True, scale=True,
                                                                      updates_collections=None, scope=scope),
                       lambda: tf.contrib.layers.batch_norm(inputs=inputs, is_training=False, scale=True,
                                                            updates_collections=None, scope=scope, reuse=True))

    #  构建网络
    def cnnBN(self, inputs, is_training, keep_prob, nclass, reuse=False, name='cnnBN'):
        with tf.variable_scope(name_or_scope=name, reuse=reuse) as scope:
            # 第一层卷积网络
            w_conv1 = self.get_weight_variable(shape=[5, 3, 1, 8])
            b_conv1 = self.get_bias_variable(shape=[8])
            # 卷积
            conv1 = self.conv2d(x=inputs, w=w_conv1) + b_conv1
            #  batch normalization
            bn_conv1 = self.bn_layer(inputs=conv1, is_train=is_training, scope='BN1')
            # ReLu 层
            rl_conv1 = tf.nn.relu(bn_conv1)
            # max_pool层
            mp_conv1 = self.max_pool_2x2(rl_conv1)
            # 第二层卷积网络
            w_conv2 = self.get_weight_variable(shape=[5, 3, 8, 16])
            b_conv2 = self.get_bias_variable(shape=[16])
            conv2 = self.conv2d(x=mp_conv1, w=w_conv2) + b_conv2
            bn_conv2 = self.bn_layer(inputs=conv2, is_train=is_training, scope='BN2')
            rl_conv2 = tf.nn.relu(bn_conv2)
            mp_conv2 = self.max_pool_2x2(rl_conv2)
            # 第三层卷积网络
            w_conv3 = self.get_weight_variable(shape=[5, 3, 16, 32])
            b_conv3 = self.get_bias_variable(shape=[32])
            conv3 = self.conv2d(x=mp_conv2, w=w_conv3) + b_conv3
            bn_conv3 = self.bn_layer(inputs=conv3, is_train=is_training, scope='BN3')
            rl_conv3 = tf.nn.relu(bn_conv3)
            # mp_conv3 = self.max_pool_2x2(rl_conv3)
            mp_conv3_shape = rl_conv3.get_shape().as_list()
            # 第四层全连接
            w_fc1 = self.get_weight_variable(shape=[mp_conv3_shape[1] * mp_conv3_shape[2] * mp_conv3_shape[3], 160])
            b_fc1 = self.get_bias_variable(shape=[160])
            mp_conv3_rshp = tf.reshape(tensor=rl_conv3,
                                       shape=[-1, mp_conv3_shape[1] * mp_conv3_shape[2] * mp_conv3_shape[3]])
            rl_fc1 = tf.nn.relu(features=tf.matmul(mp_conv3_rshp, w_fc1) + b_fc1)
            # dropout 操作
            dp_fc1 = tf.nn.dropout(x=rl_fc1, keep_prob=keep_prob)
            # 输出分类结果
            w_fc2 = self.get_weight_variable(shape=[160, nclass])
            b_fc2 = self.get_bias_variable(shape=[nclass])
            sf_fc2 = tf.nn.softmax(logits=tf.matmul(dp_fc1, w_fc2) + b_fc2)
            sf_fc2_clip = tf.clip_by_value(t=sf_fc2, clip_value_min=1e-10, clip_value_max=1.0)
            return sf_fc2_clip


if __name__ == '__main__':
    recognizeHelper = RecognizerHelper()
    validate_x, validate_y = recognizeHelper.testDataController.get_test_data()
    hwRecognizeThread = HWRecognizeThread(0, validate_x, validate_y, recognizeHelper)
    hwRecognizeThread.start()

