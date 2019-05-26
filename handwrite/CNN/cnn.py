import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import time
from dataController import DataController


# 识别动作
# 权重变量
def get_weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


# 偏移变量
def get_bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


# 卷积
def conv2d(x, w):
    return tf.nn.conv2d(input=x, filter=w, strides=[1, 1, 1, 1], padding='SAME')


# 池化, 数据长度宽度减半
def max_pool_2x2(x):
    return tf.nn.max_pool(value=x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


# Batch normalization layer
def bn_layer(inputs, is_train, scope=None):
    return tf.cond(is_train, lambda: tf.contrib.layers.batch_norm(inputs=inputs, is_training=True, scale=True,
                                                                  updates_collections=None, scope=scope),
                   lambda: tf.contrib.layers.batch_norm(inputs=inputs, is_training=False, scale=True,
                                                        updates_collections=None, scope=scope, reuse=True))


#  构建网络
def cnnBN(inputs, is_training, keep_prob, nclass, reuse=False, name='cnnBN'):
    with tf.variable_scope(name_or_scope=name, reuse=reuse) as scope:
        # 第一层卷积网络
        w_conv1 = get_weight_variable(shape=[5, 3, 1, 8])
        b_conv1 = get_bias_variable(shape=[8])
        # 卷积
        conv1 = conv2d(x=inputs, w=w_conv1) + b_conv1
        #  batch normalization
        bn_conv1 = bn_layer(inputs=conv1, is_train=is_training, scope='BN1')
        # ReLu 层
        rl_conv1 = tf.nn.relu(bn_conv1)
        # max_pool层
        mp_conv1 = max_pool_2x2(rl_conv1)
        # 第二层卷积网络
        w_conv2 = get_weight_variable(shape=[5, 3, 8, 16])
        b_conv2 = get_bias_variable(shape=[16])
        conv2 = conv2d(x=mp_conv1, w=w_conv2) + b_conv2
        bn_conv2 = bn_layer(inputs=conv2, is_train=is_training, scope='BN2')
        rl_conv2 = tf.nn.relu(bn_conv2)
        mp_conv2 = max_pool_2x2(rl_conv2)
        # 第三层卷积网络
        w_conv3 = get_weight_variable(shape=[5, 3, 16, 32])
        b_conv3 = get_bias_variable(shape=[32])
        conv3 = conv2d(x=mp_conv2, w=w_conv3) + b_conv3
        bn_conv3 = bn_layer(inputs=conv3, is_train=is_training, scope='BN3')
        rl_conv3 = tf.nn.relu(bn_conv3)
        # mp_conv3 = max_pool_2x2(rl_conv3)
        mp_conv3_shape = rl_conv3.get_shape().as_list()
        # 第四层全连接
        w_fc1 = get_weight_variable(shape=[mp_conv3_shape[1]*mp_conv3_shape[2]*mp_conv3_shape[3], 160])
        b_fc1 = get_bias_variable(shape=[160])
        mp_conv3_rshp = tf.reshape(tensor=rl_conv3, shape=[-1, mp_conv3_shape[1]*mp_conv3_shape[2]*mp_conv3_shape[3]])
        rl_fc1 = tf.nn.relu(features=tf.matmul(mp_conv3_rshp, w_fc1) + b_fc1)
        # dropout 操作
        dp_fc1 = tf.nn.dropout(x=rl_fc1, keep_prob=keep_prob)
        # 输出分类结果
        w_fc2 = get_weight_variable(shape=[160, nclass])
        b_fc2 = get_bias_variable(shape=[nclass])
        sf_fc2 = tf.nn.softmax(logits=tf.matmul(dp_fc1, w_fc2) + b_fc2)
        sf_fc2_clip = tf.clip_by_value(t=sf_fc2, clip_value_min=1e-10, clip_value_max=1.0)
        return sf_fc2_clip


# ****** 参数设置 ******
keep_p = 0.6
batch_size = 40
base_p = './data/ticwatch1/ring1/'
uun = 4
# user = 'zy_xht_dyy_yjh_dyf' + str(uun)
# user = 'p0top12_p3_5'
user = 'p0p1p2_5'
print(user)
is_validate = True
dataController = DataController(base_p + 'p1_5.bin', base_p + 'p2_5.bin', base_p + 'p0_5.bin', base_p + 'p4_5.bin',
                                base_p + 'p5_5.bin', base_p + 'p6_5.bin', base_p + 'p7_5.bin', base_p + 'p8_5.bin',
                                base_p + 'p9_5.bin', base_p + 'p10_5.bin', base_p + 'p11_5.bin', base_p + 'p12_5.bin')
# dataController = DataController(base_p + '0_T0_D_R_hand_back_dyf_TT.bin')
test_user = './data/ticwatch1/ring1/p3_5.bin'
model_save_path = './model/' + user + '/cnn.ckpt'
iteration_time = 5000
nclass = dataController.nclass
print(user)
# nclass = 2
# ****** END ******
# 读取数据
sample_size = dataController.get_sample_size()
x = tf.placeholder(dtype=tf.float32, shape=[None, sample_size[0], sample_size[1]])
y = tf.placeholder(dtype=tf.float32, shape=[None, nclass])
keep_prob = tf.placeholder(dtype=tf.float32)
is_training = tf.placeholder(dtype=tf.bool)
# 数据变成四维
x_images = tf.reshape(tensor=x, shape=[-1, sample_size[0], sample_size[1], 1])  #
# CNN3 构建网络
y_ = cnnBN(inputs=x_images, is_training=is_training, keep_prob=keep_prob, nclass=nclass)
#  定义交叉商
cross_entropy = -tf.reduce_mean(input_tensor=y*tf.log(y_))
# 学习率随着步数变化
global_step = tf.Variable(initial_value=0, trainable=False)
lr = tf.train.exponential_decay(learning_rate=0.001, global_step=global_step, decay_steps=100, decay_rate=0.99,
                                staircase=True)
# 优化损失函数， 梯度下降方法
# train_step = tf.train.AdamOptimizer(lr).minimize(loss=cross_entropy, global_step=global_step)
train_step = tf.train.AdamOptimizer(lr).minimize(loss=cross_entropy, global_step=global_step)
# 评测
correct_prediction = tf.equal(tf.argmax(input=y, axis=1), tf.argmax(input=y_, axis=1))
# 混淆矩阵
confusion_matrix = tf.contrib.metrics.confusion_matrix(tf.argmax(input=y, axis=1), tf.argmax(input=y_, axis=1))
# 计算精度
accuracy = tf.reduce_mean(tf.cast(x=correct_prediction, dtype=tf.float32))
# 记录最高精度
max_acc = 0
max_acc_loss = 0
# 训练
with tf.Session() as sess:
    tf.global_variables_initializer().run()
    start_time = time.time()
    last_time = time.time()
    # test_x = mnist.test.images
    # test_y = mnist.test.labels
    test_x, test_y = dataController.get_test_data()
    saver = tf.train.Saver(max_to_keep=3)
    if is_validate:
        model_file = tf.train.latest_checkpoint('./model/' + user + '/')
        saver.restore(sess, model_file)
        testDataController = DataController(test_user)
        validate_x, validate_y = testDataController.get_test_data()
        validate_loss, validate_accuracy = sess.run([cross_entropy, accuracy], feed_dict={x: validate_x,
                                                                                          y: validate_y,
                                                                                          keep_prob: 1,
                                                                                          is_training: False})
        print('VALIDATE ACCURACY: %f, LOSS: %f ' % (validate_accuracy, validate_loss))
        # c_f = sess.run(confusion_matrix, feed_dict={x: validate_x, y: validate_y, keep_prob: 1, is_training: False})
        # print(c_f)
        # c_f_p = np.around(c_f / 15, decimals=2)
        # draw_heatmap(c_f_p)
    else:
        # log_file = open(file='./log/' + user + '.txt', mode='w')
        for i in range(iteration_time):
            train_x, train_y = dataController.next(batch_size=batch_size, is_shuffled=True)
            train_step.run(feed_dict={x: train_x, y: train_y, keep_prob: keep_p, is_training: True})   # 训练
            if i % 20 == 0:     # 每20次查看精度
                train_loss, train_accuracy = sess.run([cross_entropy, accuracy], feed_dict={x: train_x, y: train_y,
                                                                                            keep_prob: keep_p,
                                                                                            is_training: True})
                test_loss, test_accuracy = sess.run([cross_entropy, accuracy], feed_dict={x: test_x, y: test_y,
                                                                                          keep_prob: 1,
                                                                                          is_training: False})
                current_time = time.time()
                start_current_time = current_time - start_time
                last_current_time = current_time - last_time
                last_time = current_time
                print('STEP: %d, %0fs, %0fs, TRAIN ACCURACY: %g, LOSS: %g; TEST ACCURACY: %g LOSS: %g' % (i,
                        start_current_time, last_current_time, train_accuracy, train_loss, test_accuracy, test_loss))

                if test_accuracy > max_acc:
                    max_acc = test_accuracy
                    max_acc_loss = test_loss
                    saver.save(sess=sess, save_path=model_save_path, global_step=global_step)
        # log_file.write('STEP: %f ACCURACY: %f LOSS: %f' % (i, max_acc, max_acc_loss))
        # log_file.close()
