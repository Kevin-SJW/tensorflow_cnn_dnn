# coding:utf-8

import tensorflow as tf 

#定义对所用变量的数据汇总
def variable_summaries(var):
	with tf.name_scope('summaries'):
		mean = tf.reduce_mean(var)
		tf.summary.scalar('mean', mean)
		with tf.name_scope('stddev'):
			stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
		tf.summary.scalar('stddev', stddev)
		tf.summary.scalar('max', tf.reduce_max(var))
		tf.summary.scalar('min', tf.reduce_min(var))
		#记录变量的直方图数据
		tf.summary.histogram('histogram', var)

#全链接层 Relu(Wx + b)
def nn_layer(input_tensor, input_dim, output_dim, layer_name, act=tf.nn.relu):
	with tf.name_scope(layer_name):
		
		with tf.name_scope('weights'):
			weights = tf.Variable(tf.truncated_normal([input_dim, output_dim], stddev=0.1))
			variable_summaries(weights)
		
		with tf.name_scope('biases'):
			biases = tf.Variable(tf.constant(0.1, shape=[output_dim]))
			variable_summaries(biases)
		
		with tf.name_scope('Wx_plus_b'):
			preactivate = tf.matmul(input_tensor, weights) + biases
			tf.summary.histogram('pre_activate', preactivate)
		
		activations = act(preactivate, name='activation')
		tf.summary.histogram('activations', activations)
		
		return activations

#卷积操作
def conv(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

#卷积层 	Relu(Wx + b)
def conv2d(x_image, k1, k2, layer_name):
	with tf.name_scope(layer_name):
		with tf.name_scope('weights'):
			W_conv = tf.Variable(tf.truncated_normal([5, 5, k1, k2], stddev=0.1))
			variable_summaries(W_conv)
		with tf.name_scope('biases'):
			b_conv = tf.Variable(tf.constant(0.1, shape=[k2]))
		with tf.name_scope('Wx_plus_b'):
			variable_summaries(b_conv)
			# preactivate = tf.matmul(input_tensor, weights) + biases
			h_conv = tf.nn.relu(conv(x_image, W_conv) + b_conv)
			tf.summary.histogram('h_conv1', h_conv)
	return  h_conv


#池化操作
def max_pool_2x2(x, layer_name):
	return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')  


#DNN&CNN 前向传播过程
def inference_op(x, hidden_sizes, keep_prob, kernel):

	if kernel == True: 
		#CNN
		x_image = tf.reshape(x, [-1,28,28,1])
        #第一层卷积                
		h_conv1 = conv2d(x_image, 1, 32, 'conv1')
		#第一层池化
		h_pool1 = max_pool_2x2(h_conv1, 'pool1')
		#第二层卷积
		h_conv2 = conv2d(h_pool1, 32, 64, 'conv2')
		#第二层池化
		h_pool2 = max_pool_2x2(h_conv2, 'pool2')

		#将第二层池化输出拉伸成一维向量
		h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
		#第一层全连接层
		h_fc1 = nn_layer(h_pool2_flat, 7 * 7 * 64, 1024, 'layer1')
		#dropout
		with tf.name_scope('dropout_conv'):
			tf.summary.scalar('dropout_keep_probability_conv', keep_prob)
			h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
		#第二层全连接层
		y = nn_layer(h_fc1_drop, 1024, 10 ,'layer2', act=tf.identity)

		return y
	else:
		#DNN
		#第一层全连接层
		hidden1 = nn_layer(x, 784, hidden_sizes,'layer1')
		#dropout
		with tf.name_scope('dropout'):
			tf.summary.scalar('dropout_keep_probability', keep_prob)
			dropped = tf.nn.dropout(hidden1, keep_prob)
		#第二层全连接层
		y = nn_layer(dropped, hidden_sizes, 10 ,'layer2', act=tf.identity)

		return y

