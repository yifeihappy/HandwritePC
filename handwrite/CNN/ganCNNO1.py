import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import time
from dataControllerUO import DataControllerUO


# 实现迁移学习
# 权重变量
# 识别器和判别器共享CNN前面的部分
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
def cnnBN(inputs, is_training, reuse=False, name='cnnBN'):
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

        cnn_var_list = [w_conv1, b_conv1, w_conv2, b_conv2, w_conv3, b_conv3]
        return rl_conv3, mp_conv3_shape, cnn_var_list


# 字符识别
def text_recognizer(rl_conv3, mp_conv3_shape, keep_prob, nclass, reuse=False, name='TEXT_R'):
    with tf.variable_scope(name_or_scope=name, reuse=reuse) as scope:
        # Text Recognizer
        # 第四层全连接
        w_fc1 = get_weight_variable(shape=[mp_conv3_shape[1] * mp_conv3_shape[2] * mp_conv3_shape[3], 160])
        b_fc1 = get_bias_variable(shape=[160])
        mp_conv3_rshp = tf.reshape(tensor=rl_conv3,
                                   shape=[-1, mp_conv3_shape[1] * mp_conv3_shape[2] * mp_conv3_shape[3]])
        rl_fc1 = tf.nn.relu(features=tf.matmul(mp_conv3_rshp, w_fc1) + b_fc1)
        # dropout 操作
        dp_fc1 = tf.nn.dropout(x=rl_fc1, keep_prob=keep_prob)
        # 输出分类结果
        w_fc2 = get_weight_variable(shape=[160, nclass])
        b_fc2 = get_bias_variable(shape=[nclass])
        sf_fc2 = tf.nn.softmax(logits=tf.matmul(dp_fc1, w_fc2) + b_fc2)
        sf_fc2_clip = tf.clip_by_value(t=sf_fc2, clip_value_min=1e-10, clip_value_max=1.0)
        text_var_list = [w_fc1, b_fc1, w_fc2, b_fc2]
        return sf_fc2_clip, text_var_list


# 用户识别
def user_discriminator(rl_conv3, mp_conv3_shape, keep_prob, uclass,  reuse=False, name='USER_D'):
    # User Discriminator
    # 第四层全连接
    uw_fc1 = get_weight_variable(shape=[mp_conv3_shape[1] * mp_conv3_shape[2] * mp_conv3_shape[3], 160])
    ub_fc1 = get_bias_variable(shape=[160])
    ump_conv3_rshp = tf.reshape(tensor=rl_conv3, shape=[-1, mp_conv3_shape[1] * mp_conv3_shape[2] * mp_conv3_shape[3]])
    url_fc1 = tf.nn.relu(features=tf.matmul(ump_conv3_rshp, uw_fc1) + ub_fc1)
    # dropout 操作
    udp_fc1 = tf.nn.dropout(x=url_fc1, keep_prob=keep_prob)
    # 输出分类结果
    uw_fc2 = get_weight_variable(shape=[160, uclass])
    ub_fc2 = get_bias_variable(shape=[uclass])
    usf_fc2 = tf.nn.softmax(logits=tf.matmul(udp_fc1, uw_fc2) + ub_fc2)
    usf_fc2_clip = tf.clip_by_value(t=usf_fc2, clip_value_min=1e-10, clip_value_max=1.0)
    user_var_list = [uw_fc1, ub_fc1, uw_fc2, ub_fc2]
    return usf_fc2_clip, user_var_list


# ****** 参数设置 ******
keep_p = 0.6
batch_size = 40
b_p = './data/TT/ticwatch1/ringR/10P0_T0_D_R_hand_back_'
user = 'dyf_df_zy_xht_yjh_dyy'
print(user)
is_validate = True
dataControllerUO = DataControllerUO(b_p + 'dyf' + '_TT.bin', b_p + 'df' + '_TT.bin', b_p + 'zy' + '_TT.bin',
                                    b_p + 'xht' + '_TT.bin', b_p + 'yjh' + '_TT.bin', b_p + 'dyy' + '_TT.bin')
# dataControllerUO = DataControllerUO(b_p + 'dyy' + '_TT.bin', b_p + 'dyf' + '_TT.bin')
# dataControllerUO = DataControllerUO(b_p + 'df' + '_TT.bin', b_p + 'dyf' + '_TT.bin')
test_user = './data/TT/ticwatch1/ringR/5P0_T0_D_R_hand_back_dyy_TT.bin'
iteration_time = 5000
r = 0    # 两个loss的权重
model_save_path = './model/ganCNNO1/' + user + '/' + str(r) + '/cnn.ckpt'
nclass = dataControllerUO.nclass
uclass = dataControllerUO.uclass
# ****** END ******
# 活动识别
# 读取数据
sample_size = dataControllerUO.get_sample_size()
# x = tf.placeholder(dtype=tf.float32, shape=[None, mnist.train.images.shape[1]])
x = tf.placeholder(dtype=tf.float32, shape=[None, sample_size[0], sample_size[1]])
y = tf.placeholder(dtype=tf.float32, shape=[None, nclass])
keep_prob = tf.placeholder(dtype=tf.float32)
is_training = tf.placeholder(dtype=tf.bool)
# 数据变成四维
x_images = tf.reshape(tensor=x, shape=[-1, sample_size[0], sample_size[1], 1])  #
# CNN3 构建网络
rl_conv3, mp_conv3_shape, cnn_var_list = cnnBN(inputs=x_images, is_training=is_training)
y_, text_var_list = text_recognizer(rl_conv3, mp_conv3_shape, keep_p, nclass)
text_var_list += cnn_var_list
uy_, user_var_list = user_discriminator(rl_conv3, mp_conv3_shape, keep_p, uclass)
#  定义交叉商
loss_a = -tf.reduce_mean(input_tensor=y*tf.log(y_))

# 用户识别
uy = tf.placeholder(dtype=tf.float32, shape=[None, uclass])
R = tf.Variable(initial_value=r, dtype=tf.float32)
# 定义交叉熵
loss_u = -tf.reduce_mean(input_tensor=uy*tf.log(uy_))
# goal and train
loss = loss_a - R * loss_u

# 学习率随着步数变化
global_step = tf.Variable(initial_value=0, trainable=False)
lr = tf.train.exponential_decay(learning_rate=0.001, global_step=global_step, decay_steps=100, decay_rate=0.99,
                                staircase=True)
global_step_u = tf.Variable(initial_value=0, trainable=False)
lr_u = tf.train.exponential_decay(learning_rate=0.001, global_step=global_step_u, decay_steps=100, decay_rate=0.99,
                                  staircase=True)
# 优化损失函数， 梯度下降方法
train_step = tf.train.AdamOptimizer(lr).minimize(loss=loss, global_step=global_step, var_list=text_var_list)
# 优化user discriminator 损失函数，梯度下降的方法
train_step_u = tf.train.AdamOptimizer(lr_u).minimize(loss=loss_u, global_step=global_step_u, var_list=user_var_list)
# 评测
correct_pre_a = tf.equal(tf.argmax(input=y, axis=1), tf.argmax(input=y_, axis=1))
correct_pre_u = tf.equal(tf.argmax(input=uy, axis=1), tf.argmax(input=uy_, axis=1))
# 计算精度
accuracy_a = tf.reduce_mean(tf.cast(x=correct_pre_a, dtype=tf.float32))
accuracy_u = tf.reduce_mean(tf.cast(x=correct_pre_u, dtype=tf.float32))
# 记录最高精度
max_acc_a = 0
R_ = tf.print(R, [R])
# 训练
with tf.Session() as sess:
    tf.global_variables_initializer().run()
    start_time = time.time()
    last_time = time.time()
    test_x, test_y, test_ux, test_uy = dataControllerUO.get_test_data()
    saver = tf.train.Saver(max_to_keep=3)
    if is_validate:
        model_file = tf.train.latest_checkpoint('model/ganCNNO1/%s/%s/' % (user, str(r)))
        saver.restore(sess, model_file)
        dataControl_UO = DataControllerUO(test_user)
        validate_x, validate_y, validate_ux, validate_uy = dataControl_UO.get_test_data()
        validate_loss, validate_accuracy = sess.run([loss_a, accuracy_a], feed_dict={x: validate_x, y: validate_y,
                                                                                     keep_prob: 1,
                                                                                     is_training: False})
        print('VALIDATE ACCURACY: %f, LOSS_A: %f ' % (validate_accuracy, validate_loss))
        sess.run(R_)
    else:
        log_file = open(file='./log/' + user + '.txt', mode='w')
        for i in range(iteration_time):
            train_x, train_y, train_ux, train_uy = dataControllerUO.nextA(batch_size=batch_size, is_shuffled=True)
            train_step.run(feed_dict={x: train_x, y: train_y, uy: train_uy, keep_prob: keep_p,
                                      is_training: True})
            bath_ux, bath_uy = dataControllerUO.nextU(batch_size=batch_size, is_shuffled=True)
            train_step_u.run(feed_dict={x: bath_ux, uy: bath_uy, keep_prob: keep_p, is_training: True})
            if i % 20 == 0:     # 每20次查看精度
                tr_ls_a, tr_acc_a, tr_ls_u, tr_acc_u, tr_ls = sess.run([loss_a, accuracy_a, loss_u, accuracy_u, loss],
                                                                       feed_dict={x: train_x, y: train_y,
                                                                                  uy: train_uy, keep_prob: keep_p,
                                                                                  is_training: True})
                te_ls_a, te_acc_a = sess.run([loss_a, accuracy_a], feed_dict={x: test_x, y: test_y,
                                                                              keep_prob: 1, is_training: False})
                te_ls_u, te_acc_u = sess.run([loss_u, accuracy_u], feed_dict={x: test_ux, uy: test_uy,
                                                                              keep_prob: 1, is_training: False})
                current_time = time.time()
                start_current_time = current_time - start_time
                last_current_time = current_time - last_time
                last_time = current_time
                print('STEP %d; %0fs,%0fs; TRAIN:ACC_a=%g,LS_a=%g,acc_u=%g,ls_u=%g,LOSS=%g; TEST:ACC_a=%g,LS_a=%g,'
                      'acc_u=%g,ls_u=%g;' % (i, start_current_time, last_current_time,
                                             tr_acc_a, tr_ls_a, tr_acc_u, tr_ls_u, tr_ls,
                                             te_acc_a, te_ls_a, te_acc_u, te_ls_u))

                if te_acc_a > max_acc_a:
                    max_acc_a = te_acc_a
                    saver.save(sess=sess, save_path=model_save_path, global_step=global_step)
                    log_file.write('STEP: %f ACCURACY: %f LOSS_A: %f' % (i, te_acc_a, te_ls_a))
        log_file.close()