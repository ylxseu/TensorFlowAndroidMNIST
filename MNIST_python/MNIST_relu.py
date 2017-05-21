# -*- coding:utf-8-*-

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# MNIST数据集相关常数
INPUT_NODE = 784    # 输入层的节点数
OUTPUT_NODE = 10    # 输出层的节点数

# 配置神经网络的参数
LAYER1_NODE = 500   # 隐藏层的节点数

BATCH_SIZE = 100    # 一个batch中的训练数据个数

LEARNING_RATE_BASE = 0.8    # 基础学习率
LEARNING_RATE_DECAY = 0.99  # 学习率的衰减率

REGULARIZATION_RATE = 0.0001    # 描述模型复杂度的正则化项在损失函数中的系数
TRAINING_STEPS = 30000           # 训练的轮数
MOVING_AVERAGE_DECAY = 0.99      # 滑动平均衰减率



# 辅助函数，用给定的神经网络输入和所有参数，给出前向传播结果
# 激活函数：relu
# 隐藏层：1层
def inference(input_tensor, avg_class, weights1, biases1, weights2, biases2):
    # 如果没有提供滑动平均类,直接使用计算值
    if avg_class == None:
        layer1 = tf.nn.relu(tf.matmul(input_tensor,weights1) + biases1)
        return tf.matmul(layer1,weights2) + biases2

    else:
        layer1=tf.nn.relu(tf.matmul(input_tensor,avg_class.average(weights1)) + avg_class.average(biases1))
        return tf.matmul(layer1,avg_class.average(weights2))+avg_class.average(biases2)

# 训练过程函数
def train(mnist):
    x = tf.placeholder(tf.float32,[None,INPUT_NODE],name="x-input")
    y_ = tf.placeholder(tf.float32,[None,OUTPUT_NODE],name="y-input")

    # 生成隐藏层的参数
    weights1 = tf.Variable(tf.truncated_normal([INPUT_NODE,LAYER1_NODE],stddev=0.1))
    biases1 = tf.Variable(tf.constant(0.1, shape=[LAYER1_NODE]))

    # 生成输出层的参数
    weights2 = tf.Variable(tf.truncated_normal([LAYER1_NODE,OUTPUT_NODE],stddev=0.1))
    biases2 = tf.Variable(tf.constant(0.1,shape=[OUTPUT_NODE]))

    # 计算在当前参数下神经网络前向传播的结果
    y = inference(x,None,weights1,biases1, weights2, biases2)

    # 定义存储训练轮数的变量
    global_step = tf.Variable(0, trainable = False)

    # 给定滑动平均衰减率和训练轮数的变量，初始化滑动平均类
    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)

    # 在所有代表神经网络参数的变量上使用滑动平均
    variable_averages_op = variable_averages.apply(tf.trainable_variables())

    # 计算使用了滑动平均值后的前向传播结果
    average_y = inference(x, variable_averages, weights1, biases1, weights2, biases2)

    # 计算交叉熵作为刻画预测值和真实值之间差距的损失函数
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(y, tf.argmax(y_,1))
    cross_entropy_mean = tf.reduce_mean(cross_entropy)

    # 计算L2正则化损失函数
    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
    # 计算模型的正则化损失
    regularization = regularizer(weights1) + regularizer(weights2)
    # 总损失 = 交叉熵损失了+ 正则化损失
    loss = cross_entropy_mean + regularization
    # 设置指数衰减的学习率
    learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE, 
                                               global_step,
                                               mnist.train.num_examples / BATCH_SIZE,
                                               LEARNING_RATE_DECAY)
    
    # 使用梯度下降算法来优化损失函数
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss,global_step=global_step)
    
    with tf.control_dependencies([train_step,variable_averages_op]):
        train_op = tf.no_op(name='train')
    
    # 检验使用了滑动平均的神经网络前向传播结果是否正确
    correct_prediction = tf.equal(tf.argmax(average_y,1),tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
    
    
    #初始化会话并开始训练过程
    with tf.Session() as sess:
        tf.initialize_all_variables().run()
        # 准备验证数据
        validate_feed = {x:mnist.validation.images,
                         y_:mnist.validation.labels}
    
        # 准备测试数据
        test_feed = {x:mnist.test.images,y_:mnist.test.labels}
    
        # 迭代地训练神经网络
        for i in range(TRAINING_STEPS):
            # 每1000轮输出一次再验证数据集上的测试结果
            if i%1000 == 0:
                # 计算滑动平均模型在验证数据上的结果
                validate_acc = sess.run(accuracy, feed_dict=validate_feed)
                print("After %d training steps, validation accuracy "
                     "using average model is %g " % (i,validate_acc))
    
    
            # 产生这一轮使用的一个batch的训练数据，并运行训练过程
            xs,ys = mnist.train.next_batch(BATCH_SIZE)
            sess.run(train_op,feed_dict={x:xs,y_:ys})
    
    
        # 训练结束，再测试数据上检测神经网络模型的最终正确率
        test_acc = sess.run(accuracy,feed_dict=test_feed)
        print("After %d training steps, test accuracy using average model is %g"
             % (TRAINING_STEPS,test_acc))
    
    
# 主程序入口
def main(argv=None):
    # MNIST数据处理类
    mnist = input_data.read_data_sets("./data",one_hot=True)
    train(mnist)

if __name__ == '__main__':
    tf.app.run()
